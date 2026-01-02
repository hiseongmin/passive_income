"""
Incremental Feature Extraction Pipeline

Extracts features incrementally using only past data.
CRITICAL: No future data leakage - features computed from data[0:current_idx+1]
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm

import sys
sys.path.append('/notebooks/sa')

from features.tda_features import TDAFeatureExtractor, extract_tda_features
from features.market_microstructure import MarketMicrostructureExtractor
from .config import BacktestConfig


class IncrementalFeatureExtractor:
    """
    Extracts features incrementally using only past data.

    CRITICAL: No future data leakage - features computed from data[0:current_idx+1]
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Initialize extractors
        self.tda_extractor = TDAFeatureExtractor(
            window_size=config.tda_window_size,
            embedding_configs=[(3, 1), (5, 2), (7, 3)]
        )
        self.micro_extractor = MarketMicrostructureExtractor(
            lookback=config.micro_lookback
        )

        # Feature dimensions
        self.n_tda_features = 9   # 3 configs * 3 features
        self.n_micro_features = 12

        # Cached features
        self._tda_features: Optional[np.ndarray] = None
        self._micro_features: Optional[np.ndarray] = None

    def precompute_all_features(
        self,
        df: pd.DataFrame,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute all features for the entire dataset.

        Features at index i use only data[0:i+1].
        This allows efficient batch computation while maintaining
        the property that each feature only uses past data.

        Args:
            df: DataFrame with OHLCV data
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (tda_features, micro_features)
        """
        n_samples = len(df)
        close_prices = df['close'].values

        print("Pre-computing features for backtest...")

        # ==================== TDA Features ====================
        print("  Extracting TDA features...")
        tda_features = np.zeros((n_samples, self.n_tda_features), dtype=np.float32)

        iterator = range(self.config.tda_window_size, n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="  TDA features")

        for i in iterator:
            # Use closes from [i - window + 1] to [i] inclusive
            window = close_prices[i - self.config.tda_window_size + 1:i + 1]
            tda_features[i] = self._extract_tda_single(window)

        # ==================== Microstructure Features ====================
        print("  Extracting microstructure features...")
        # Microstructure extractor handles rolling windows internally
        # but we need to ensure it doesn't use future data
        micro_features = self.micro_extractor.extract_features(df)

        # Store for later use
        self._tda_features = tda_features
        self._micro_features = micro_features

        print(f"  TDA features shape: {tda_features.shape}")
        print(f"  Micro features shape: {micro_features.shape}")

        return tda_features, micro_features

    def _extract_tda_single(self, window: np.ndarray) -> np.ndarray:
        """
        Extract TDA features from a single window.

        Args:
            window: Price window of shape (window_size,)

        Returns:
            Feature vector of shape (n_tda_features,)
        """
        features = []

        for dim, tau in [(3, 1), (5, 2), (7, 3)]:
            try:
                entropy, amplitude, num_points = extract_tda_features(
                    window,
                    embedding_dim=dim,
                    embedding_tau=tau,
                    homology_dim=1
                )
                features.extend([entropy, amplitude, float(num_points)])
            except Exception:
                features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def get_features_at(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get TDA and microstructure features at index idx.

        Args:
            idx: Data index

        Returns:
            Tuple of (tda_features, micro_features) for the given index
        """
        if self._tda_features is None or self._micro_features is None:
            raise RuntimeError("Features not computed. Call precompute_all_features() first.")

        return self._tda_features[idx], self._micro_features[idx]

    def normalize_features(
        self,
        tda_features: np.ndarray,
        micro_features: np.ndarray,
        tda_mean: Optional[np.ndarray] = None,
        tda_std: Optional[np.ndarray] = None,
        micro_mean: Optional[np.ndarray] = None,
        micro_std: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using provided statistics or compute from data.

        For realistic backtesting, normalization params should come from
        training data, not backtest data.

        Args:
            tda_features: TDA feature array
            micro_features: Microstructure feature array
            tda_mean, tda_std: TDA normalization params (optional)
            micro_mean, micro_std: Micro normalization params (optional)

        Returns:
            Tuple of normalized (tda_features, micro_features)
        """
        # Compute stats if not provided (not recommended for true OOS testing)
        if tda_mean is None:
            tda_mean = np.mean(tda_features, axis=0)
        if tda_std is None:
            tda_std = np.std(tda_features, axis=0) + 1e-8

        if micro_mean is None:
            micro_mean = np.mean(micro_features, axis=0)
        if micro_std is None:
            micro_std = np.std(micro_features, axis=0) + 1e-8

        tda_normalized = (tda_features - tda_mean) / tda_std
        micro_normalized = (micro_features - micro_mean) / micro_std

        return tda_normalized, micro_normalized
