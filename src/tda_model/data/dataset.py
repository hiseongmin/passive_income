"""
PyTorch Dataset for TDA model.

Handles:
- OHLCV sequence extraction
- TDA feature computation/caching
- Complexity score (from data or placeholder)
- Label extraction
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import Config

logger = logging.getLogger(__name__)


class TDADataset(Dataset):
    """
    PyTorch Dataset for multi-task TDA model.

    For each valid sample at time t:
    - OHLCV sequence: [t - seq_len + 1, t] normalized
    - TDA features: computed from [t - window_size + 1, t] close prices
    - Complexity score: from 'complexity' column if available, else placeholder
    - Labels: Depends on mode:
        - Classification mode: Trigger (bool), Max_Pct (float)
        - Regression mode: forward returns at 1h, 4h, 24h horizons

    The complexity column should be pre-computed using src/complexity/compute_complexity.py
    which computes complexity on 1-min data and resamples to 15-min.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        cache_dir: Optional[Path] = None,
        split: str = "train",
        precompute_tda: bool = True,
        mode: str = "classification",
    ):
        """
        Initialize TDA Dataset.

        Args:
            df: DataFrame with OHLCV + labels
            config: Configuration object
            cache_dir: Directory for TDA feature cache
            split: Data split name ('train', 'val', 'test')
            precompute_tda: Whether to precompute TDA features
            mode: 'classification' for binary trigger prediction,
                  'regression' for multi-horizon return prediction
        """
        self.config = config
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mode = mode

        # Store raw data
        self.df = df.sort_values("open_time").reset_index(drop=True)

        # Parameters
        self.window_size = config.tda.window_size
        self.seq_len = config.model.ohlcv_sequence_length
        self.complexity_placeholder = config.data.complexity_placeholder

        # Minimum index to have full window
        self.min_idx = max(self.window_size, self.seq_len)

        # Valid indices (have enough history)
        self.valid_indices = list(range(self.min_idx, len(self.df)))

        logger.info(f"Dataset '{split}': {len(self.valid_indices)} valid samples")
        logger.info(f"  Window size: {self.window_size}, Seq length: {self.seq_len}")

        # Precompute normalized OHLCV
        self._prepare_ohlcv()

        # TDA features (lazy loading or precomputed)
        self.tda_features = None
        if precompute_tda and self.cache_dir is not None:
            self._load_or_compute_tda()

    def _prepare_ohlcv(self):
        """Prepare normalized OHLCV data, volume features, technical indicators, and complexity."""
        from .preprocessing import add_technical_indicators

        # Add technical indicators to DataFrame
        self.df = add_technical_indicators(self.df)

        # Use log returns for OHLC normalization
        close = self.df["close"].values
        open_price = self.df["open"].values
        high = self.df["high"].values
        low = self.df["low"].values

        # Log returns for OHLC
        self.close_returns = np.zeros_like(close)
        self.close_returns[1:] = np.log(close[1:] / close[:-1])

        self.open_returns = np.zeros_like(open_price)
        self.open_returns[1:] = np.log(open_price[1:] / close[:-1])

        self.high_returns = np.zeros_like(high)
        self.high_returns[1:] = np.log(high[1:] / close[:-1])

        self.low_returns = np.zeros_like(low)
        self.low_returns[1:] = np.log(low[1:] / close[:-1])

        # Volume features (normalized by log)
        # volume, buy_volume, sell_volume, volume_delta, cvd
        volume_cols = ["volume", "buy_volume", "sell_volume", "volume_delta", "cvd"]
        self.has_volume = all(col in self.df.columns for col in volume_cols)

        if self.has_volume:
            # Normalize volumes by log (add 1 to handle zeros)
            self.volume_norm = np.log1p(self.df["volume"].values.astype(np.float32))
            self.buy_volume_norm = np.log1p(self.df["buy_volume"].values.astype(np.float32))
            self.sell_volume_norm = np.log1p(self.df["sell_volume"].values.astype(np.float32))

            # Volume delta and CVD: normalize by rolling std
            vd = self.df["volume_delta"].values.astype(np.float32)
            self.volume_delta_norm = vd / (np.std(vd) + 1e-10)
            self.volume_delta_norm = np.clip(self.volume_delta_norm, -5, 5)  # Clip outliers

            cvd = self.df["cvd"].values.astype(np.float32)
            # CVD: use rolling z-score
            cvd_diff = np.diff(cvd, prepend=cvd[0])
            self.cvd_norm = cvd_diff / (np.std(cvd_diff) + 1e-10)
            self.cvd_norm = np.clip(self.cvd_norm, -5, 5)

            logger.info("  Volume features prepared: volume, buy_volume, sell_volume, volume_delta, cvd")
        else:
            logger.warning("  Volume columns not found, using zeros")
            n = len(close)
            self.volume_norm = np.zeros(n, dtype=np.float32)
            self.buy_volume_norm = np.zeros(n, dtype=np.float32)
            self.sell_volume_norm = np.zeros(n, dtype=np.float32)
            self.volume_delta_norm = np.zeros(n, dtype=np.float32)
            self.cvd_norm = np.zeros(n, dtype=np.float32)

        # Technical indicators (already normalized in preprocessing)
        tech_cols = ["RSI", "MACD", "BB_pctB", "ATR", "MOM"]
        self.has_technical = all(col in self.df.columns for col in tech_cols)

        if self.has_technical:
            self.rsi = self.df["RSI"].values.astype(np.float32)
            self.macd = self.df["MACD"].values.astype(np.float32)
            self.bb_pctb = self.df["BB_pctB"].values.astype(np.float32)
            self.atr = self.df["ATR"].values.astype(np.float32)
            self.mom = self.df["MOM"].values.astype(np.float32)
            logger.info("  Technical indicators prepared: RSI, MACD, BB_pctB, ATR, MOM")
        else:
            logger.warning("  Technical indicator columns not found, using zeros")
            n = len(close)
            self.rsi = np.zeros(n, dtype=np.float32)
            self.macd = np.zeros(n, dtype=np.float32)
            self.bb_pctb = np.zeros(n, dtype=np.float32)
            self.atr = np.zeros(n, dtype=np.float32)
            self.mom = np.zeros(n, dtype=np.float32)

        # Store raw close for TDA
        self.close_prices = close

        # Labels (depends on mode)
        if self.mode == "classification":
            self.triggers = self.df["Trigger"].values.astype(np.float32)
            self.max_pcts = self.df["Max_Pct"].values.astype(np.float32)
            self.forward_returns = None
        elif self.mode == "regression":
            # Compute forward returns if not present in DataFrame
            return_cols = ["max_return_1h", "max_return_4h", "max_return_24h",
                          "min_return_1h", "min_return_4h", "min_return_24h"]

            if not all(col in self.df.columns for col in return_cols):
                from .preprocessing import compute_forward_returns
                logger.info("Computing forward returns for regression mode...")
                self.df = compute_forward_returns(self.df)

            # Store forward returns as numpy array [N, 6]
            self.forward_returns = self.df[return_cols].values.astype(np.float32)

            # Handle NaN values at end of data (no future data available)
            nan_mask = np.isnan(self.forward_returns).any(axis=1)
            if nan_mask.any():
                logger.info(f"  {nan_mask.sum()} samples have NaN forward returns (end of data)")
                # Fill NaN with 0 (neutral prediction target)
                self.forward_returns = np.nan_to_num(self.forward_returns, nan=0.0)

            # Keep classification labels for comparison
            if "Trigger" in self.df.columns:
                self.triggers = self.df["Trigger"].values.astype(np.float32)
                self.max_pcts = self.df["Max_Pct"].values.astype(np.float32)
            else:
                self.triggers = None
                self.max_pcts = None

            logger.info(f"  Forward returns shape: {self.forward_returns.shape}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'classification' or 'regression'")

        # Expanded complexity (6 indicators instead of 1 scalar)
        complexity_cols = ["MA_separation", "BB_width", "efficiency",
                          "support_reaction", "direction", "volume_alignment"]
        self.has_expanded_complexity = all(col in self.df.columns for col in complexity_cols)

        if self.has_expanded_complexity:
            # Use all 6 complexity indicators
            self.complexity_expanded = self.df[complexity_cols].values.astype(np.float32)
            # Fill NaN with 0.5 (neutral)
            self.complexity_expanded = np.nan_to_num(self.complexity_expanded, nan=0.5)
            self.use_expanded_complexity = True
            logger.info(f"  Using 6 expanded complexity indicators")
        elif "complexity" in self.df.columns:
            # Fall back to single complexity (replicate to 6 dims)
            self.complexity_scores = self.df["complexity"].values.astype(np.float32)
            nan_mask = np.isnan(self.complexity_scores)
            if nan_mask.any():
                logger.warning(f"Found {nan_mask.sum()} NaN complexity values, filling with placeholder")
                self.complexity_scores[nan_mask] = self.complexity_placeholder
            # Replicate to 6 dimensions
            self.complexity_expanded = np.column_stack([self.complexity_scores] * 6)
            self.use_expanded_complexity = True
            logger.info(f"  Using replicated complexity (mean={self.complexity_scores.mean():.3f})")
        else:
            # Use placeholder for all 6
            n = len(close)
            self.complexity_expanded = np.full((n, 6), self.complexity_placeholder, dtype=np.float32)
            self.use_expanded_complexity = True
            logger.info(f"  No complexity columns found, using placeholder={self.complexity_placeholder}")

    def _load_or_compute_tda(self):
        """Load TDA features from cache or compute."""
        cache_file = self.cache_dir / f"tda_features_{self.split}.npy"

        if cache_file.exists():
            logger.info(f"Loading cached TDA features from {cache_file}")
            cached_features = np.load(cache_file)

            if len(cached_features) == len(self.valid_indices):
                # Perfect match
                self.tda_features = cached_features
            elif len(cached_features) > len(self.valid_indices):
                # Cache is larger (e.g., from full dataset before split)
                # Use first N features which correspond to earlier time indices
                logger.info(
                    f"Cache has {len(cached_features)} features, "
                    f"using first {len(self.valid_indices)} for this subset"
                )
                self.tda_features = cached_features[:len(self.valid_indices)]
            else:
                # Cache is smaller, need to recompute
                logger.warning(
                    f"Cache has {len(cached_features)} features but need "
                    f"{len(self.valid_indices)}, recomputing..."
                )
                self._compute_tda_features()
                np.save(cache_file, self.tda_features)
        else:
            logger.info("Computing TDA features (this may take a while)...")
            self._compute_tda_features()
            np.save(cache_file, self.tda_features)
            logger.info(f"Saved TDA features to {cache_file}")

    def _compute_tda_features(self):
        """Compute TDA features for all valid samples."""
        from ..tda import extract_tda_features

        n_samples = len(self.valid_indices)
        feature_dim = self.config.tda_feature_dim

        self.tda_features = np.zeros((n_samples, feature_dim), dtype=np.float32)

        for i, idx in enumerate(self.valid_indices):
            if i % 1000 == 0:
                logger.info(f"  Computing TDA features: {i}/{n_samples}")

            # Extract window of close prices
            window_start = idx - self.window_size + 1
            window = self.close_prices[window_start:idx + 1]

            # Compute TDA features
            features = extract_tda_features(
                window,
                embedding_dim=self.config.tda.embedding_dim,
                time_delay=self.config.tda.time_delay,
                betti_bins=self.config.tda.betti_bins,
                landscape_layers=self.config.tda.landscape_layers,
                homology_dimensions=self.config.tda.homology_dimensions,
            )

            self.tda_features[i] = features

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample with enhanced features.

        Args:
            idx: Sample index

        Returns:
            Tuple of (depends on mode):

            Classification mode:
            - feature_seq: (seq_len, 14) sequence with OHLC + Volume + Technical
            - tda_features: (tda_dim,) TDA feature vector
            - complexity: (6,) expanded complexity indicators
            - trigger: (1,) trigger label
            - max_pct: (1,) max_pct label

            Regression mode:
            - feature_seq: (seq_len, 14) sequence with OHLC + Volume + Technical
            - tda_features: (tda_dim,) TDA feature vector
            - complexity: (6,) expanded complexity indicators
            - forward_returns: (6,) [max_1h, max_4h, max_24h, min_1h, min_4h, min_24h]

        Feature channels (14 total):
        - OHLC: 4 (open, high, low, close returns)
        - Volume: 5 (volume, buy_volume, sell_volume, volume_delta, cvd)
        - Technical: 5 (RSI, MACD, BB_pctB, ATR, MOM)
        """
        data_idx = self.valid_indices[idx]
        seq_start = data_idx - self.seq_len + 1

        # OHLC sequence (4 features)
        ohlc_seq = np.stack([
            self.open_returns[seq_start:data_idx + 1],
            self.high_returns[seq_start:data_idx + 1],
            self.low_returns[seq_start:data_idx + 1],
            self.close_returns[seq_start:data_idx + 1],
        ], axis=1).astype(np.float32)

        # Volume sequence (5 features)
        volume_seq = np.stack([
            self.volume_norm[seq_start:data_idx + 1],
            self.buy_volume_norm[seq_start:data_idx + 1],
            self.sell_volume_norm[seq_start:data_idx + 1],
            self.volume_delta_norm[seq_start:data_idx + 1],
            self.cvd_norm[seq_start:data_idx + 1],
        ], axis=1).astype(np.float32)

        # Technical indicators sequence (5 features)
        tech_seq = np.stack([
            self.rsi[seq_start:data_idx + 1],
            self.macd[seq_start:data_idx + 1],
            self.bb_pctb[seq_start:data_idx + 1],
            self.atr[seq_start:data_idx + 1],
            self.mom[seq_start:data_idx + 1],
        ], axis=1).astype(np.float32)

        # Concatenate all features: (seq_len, 14)
        feature_seq = np.concatenate([ohlc_seq, volume_seq, tech_seq], axis=1)

        # TDA features
        if self.tda_features is not None:
            tda_feat = self.tda_features[idx]
        else:
            # Compute on-the-fly (slower)
            from ..tda import extract_tda_features

            window_start = data_idx - self.window_size + 1
            window = self.close_prices[window_start:data_idx + 1]

            tda_feat = extract_tda_features(
                window,
                embedding_dim=self.config.tda.embedding_dim,
                time_delay=self.config.tda.time_delay,
                betti_bins=self.config.tda.betti_bins,
                landscape_layers=self.config.tda.landscape_layers,
                homology_dimensions=self.config.tda.homology_dimensions,
            )

        # Expanded complexity (6 dimensions)
        complexity = self.complexity_expanded[data_idx].astype(np.float32)

        # Return based on mode
        if self.mode == "classification":
            trigger = np.array([self.triggers[data_idx]], dtype=np.float32)
            max_pct = np.array([self.max_pcts[data_idx]], dtype=np.float32)

            return (
                torch.from_numpy(feature_seq),
                torch.from_numpy(tda_feat),
                torch.from_numpy(complexity),
                torch.from_numpy(trigger),
                torch.from_numpy(max_pct),
            )
        else:  # regression mode
            forward_returns = self.forward_returns[data_idx].astype(np.float32)

            return (
                torch.from_numpy(feature_seq),
                torch.from_numpy(tda_feat),
                torch.from_numpy(complexity),
                torch.from_numpy(forward_returns),
            )

    def get_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all labels for the dataset (classification mode)."""
        if self.triggers is None:
            raise ValueError("Classification labels not available in regression mode")
        triggers = np.array([self.triggers[i] for i in self.valid_indices])
        max_pcts = np.array([self.max_pcts[i] for i in self.valid_indices])
        return triggers, max_pcts

    def get_forward_returns(self) -> np.ndarray:
        """Get all forward returns for the dataset (regression mode)."""
        if self.forward_returns is None:
            raise ValueError("Forward returns not available in classification mode")
        return np.array([self.forward_returns[i] for i in self.valid_indices])

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classification."""
        if self.triggers is None:
            raise ValueError("Classification labels not available in regression mode")
        triggers, _ = self.get_labels()
        n_pos = triggers.sum()
        n_neg = len(triggers) - n_pos

        # Inverse frequency weighting
        weight_pos = len(triggers) / (2 * n_pos) if n_pos > 0 else 1.0
        weight_neg = len(triggers) / (2 * n_neg) if n_neg > 0 else 1.0

        return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
