"""
Feature Combiner

Combines TDA features and market microstructure features into a unified feature set.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm

from .tda_features import TDAFeatureExtractor, extract_tda_features_for_dataset
from .market_microstructure import MarketMicrostructureExtractor, extract_microstructure_features


class FeatureCombiner:
    """
    Combines multiple feature sources:
    - TDA features (topological analysis)
    - Market microstructure features (trading-specific)
    """

    def __init__(
        self,
        use_tda: bool = True,
        use_microstructure: bool = True,
        tda_window_size: int = 72,
        micro_lookback: int = 20
    ):
        """
        Args:
            use_tda: Whether to extract TDA features
            use_microstructure: Whether to extract microstructure features
            tda_window_size: Window size for TDA extraction
            micro_lookback: Lookback period for microstructure features
        """
        self.use_tda = use_tda
        self.use_microstructure = use_microstructure

        if use_tda:
            self.tda_extractor = TDAFeatureExtractor(window_size=tda_window_size)
        else:
            self.tda_extractor = None

        if use_microstructure:
            self.micro_extractor = MarketMicrostructureExtractor(lookback=micro_lookback)
        else:
            self.micro_extractor = None

    @property
    def n_tda_features(self) -> int:
        """Number of TDA features."""
        return self.tda_extractor.n_features if self.tda_extractor else 0

    @property
    def n_micro_features(self) -> int:
        """Number of microstructure features."""
        return self.micro_extractor.n_features if self.micro_extractor else 0

    @property
    def n_features(self) -> int:
        """Total number of features."""
        return self.n_tda_features + self.n_micro_features

    def extract_all_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        show_progress: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract all features from DataFrame.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data (used for TDA)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (tda_features, micro_features)
            Each is ndarray of shape (n_samples, n_features) or None
        """
        tda_features = None
        micro_features = None

        if self.use_tda:
            if show_progress:
                print("Extracting TDA features...")
            series = df[price_col].values
            tda_features = self.tda_extractor.extract_all(series, show_progress=show_progress)

        if self.use_microstructure:
            if show_progress:
                print("Extracting microstructure features...")
            micro_features = self.micro_extractor.extract_features(df)

        return tda_features, micro_features

    def get_feature_names(self) -> dict:
        """Get dictionary of feature names by category."""
        names = {}

        if self.use_tda:
            names['tda'] = [
                f'tda_config{i // 3}_{"entropy" if i % 3 == 0 else "amplitude" if i % 3 == 1 else "num_points"}'
                for i in range(self.n_tda_features)
            ]

        if self.use_microstructure:
            names['microstructure'] = self.micro_extractor.get_feature_names()

        return names


def precompute_all_features(
    df: pd.DataFrame,
    output_dir: str,
    use_tda: bool = True,
    use_microstructure: bool = True,
    tda_window_size: int = 72,
    micro_lookback: int = 20
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Pre-compute all features and save to disk.

    Args:
        df: DataFrame with OHLCV data
        output_dir: Directory to save features
        use_tda: Whether to extract TDA features
        use_microstructure: Whether to extract microstructure features
        tda_window_size: Window size for TDA
        micro_lookback: Lookback for microstructure

    Returns:
        Tuple of (tda_features, micro_features)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    combiner = FeatureCombiner(
        use_tda=use_tda,
        use_microstructure=use_microstructure,
        tda_window_size=tda_window_size,
        micro_lookback=micro_lookback
    )

    tda_features, micro_features = combiner.extract_all_features(df)

    # Save features
    if tda_features is not None:
        tda_path = os.path.join(output_dir, 'tda_features.npy')
        np.save(tda_path, tda_features)
        print(f"Saved TDA features to {tda_path}")

    if micro_features is not None:
        micro_path = os.path.join(output_dir, 'micro_features.npy')
        np.save(micro_path, micro_features)
        print(f"Saved microstructure features to {micro_path}")

    return tda_features, micro_features


def load_precomputed_features(
    feature_dir: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load pre-computed features from disk.

    Args:
        feature_dir: Directory containing saved features

    Returns:
        Tuple of (tda_features, micro_features)
    """
    import os

    tda_features = None
    micro_features = None

    tda_path = os.path.join(feature_dir, 'tda_features.npy')
    if os.path.exists(tda_path):
        tda_features = np.load(tda_path)
        print(f"Loaded TDA features: {tda_features.shape}")

    micro_path = os.path.join(feature_dir, 'micro_features.npy')
    if os.path.exists(micro_path):
        micro_features = np.load(micro_path)
        print(f"Loaded microstructure features: {micro_features.shape}")

    return tda_features, micro_features


if __name__ == "__main__":
    import os

    DATA_DIR = '/notebooks/sa/data'
    data_path = os.path.join(DATA_DIR, 'BTCUSDT_perp_5m_labeled.csv')

    if os.path.exists(data_path):
        print("Loading data...")
        df = pd.read_csv(data_path)

        print("\nExtracting features...")
        tda_features, micro_features = precompute_all_features(
            df,
            output_dir=DATA_DIR,
            use_tda=True,
            use_microstructure=True
        )

        if tda_features is not None:
            print(f"\nTDA features shape: {tda_features.shape}")

        if micro_features is not None:
            print(f"Microstructure features shape: {micro_features.shape}")

        # Test loading
        print("\nTesting feature loading...")
        tda_loaded, micro_loaded = load_precomputed_features(DATA_DIR)

    else:
        print(f"Data not found at {data_path}")
        print("Run trigger_generator.py first.")
