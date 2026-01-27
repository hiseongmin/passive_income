"""
TDA Feature Preprocessing Module.

Handles feature selection, cleaning, and PCA for TDA features:
- Selects informative Betti bins (0-19)
- Removes constant features (Entropy H0)
- Clips and applies PCA to Landscape features
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Optional
import pickle
from pathlib import Path

from .config import PreprocessingConfig


class TDAPreprocessor:
    """
    Preprocessor for TDA features.

    Splits 214-dim TDA features into three functional groups:
    - Structural (H0-based): 21 dims
    - Cyclical (H1-based): 22 dims
    - Landscape (multi-scale): 20 dims (after PCA)

    Feature indices in original 214-dim TDA:
    - Betti H0: 0-49 (50 dims) -> keep 0-19
    - Betti H1: 50-99 (50 dims) -> keep 50-69
    - Entropy H0: 100 (1 dim) -> REMOVE (constant)
    - Entropy H1: 101 (1 dim) -> keep
    - Persistence H0: 102 (1 dim) -> keep
    - Persistence H1: 103 (1 dim) -> keep
    - Landscape: 104-213 (110 dims) -> clip + PCA to 20
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.landscape_pca: Optional[PCA] = None
        self.landscape_scaler: Optional[RobustScaler] = None
        self.is_fitted = False

        # Feature indices
        self.betti_h0_slice = slice(0, config.betti_bins_to_keep)  # 0-19
        self.betti_h1_slice = slice(50, 50 + config.betti_bins_to_keep)  # 50-69
        self.entropy_h1_idx = 101
        self.persistence_h0_idx = 102
        self.persistence_h1_idx = 103
        self.landscape_slice = slice(104, 214)  # 110 dims

    def fit(self, tda_features: np.ndarray) -> 'TDAPreprocessor':
        """
        Fit the preprocessor on training TDA features.

        Args:
            tda_features: (N, 214) array of TDA features

        Returns:
            self
        """
        # Extract landscape features
        landscape_raw = tda_features[:, self.landscape_slice]

        # Clip outliers
        clip_min, clip_max = self.config.landscape_clip_range
        landscape_clipped = np.clip(landscape_raw, clip_min, clip_max)

        # Fit robust scaler
        self.landscape_scaler = RobustScaler()
        landscape_scaled = self.landscape_scaler.fit_transform(landscape_clipped)

        # Fit PCA
        self.landscape_pca = PCA(n_components=self.config.landscape_pca_dims)
        self.landscape_pca.fit(landscape_scaled)

        # Log explained variance
        explained_var = self.landscape_pca.explained_variance_ratio_.sum()
        print(f"Landscape PCA: {self.config.landscape_pca_dims} components "
              f"explain {explained_var:.2%} variance")

        self.is_fitted = True
        return self

    def transform(self, tda_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform TDA features into three functional groups.

        Args:
            tda_features: (N, 214) or (214,) array of TDA features

        Returns:
            Tuple of (structural, cyclical, landscape) arrays
            - structural: (N, 21) or (21,)
            - cyclical: (N, 22) or (22,)
            - landscape: (N, 20) or (20,)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Handle single sample
        single_sample = tda_features.ndim == 1
        if single_sample:
            tda_features = tda_features.reshape(1, -1)

        # Extract structural features (H0-based)
        betti_h0 = tda_features[:, self.betti_h0_slice]  # (N, 20)
        persistence_h0 = tda_features[:, self.persistence_h0_idx:self.persistence_h0_idx+1]  # (N, 1)
        structural = np.concatenate([betti_h0, persistence_h0], axis=1)  # (N, 21)

        # Extract cyclical features (H1-based)
        betti_h1 = tda_features[:, self.betti_h1_slice]  # (N, 20)
        entropy_h1 = tda_features[:, self.entropy_h1_idx:self.entropy_h1_idx+1]  # (N, 1)
        persistence_h1 = tda_features[:, self.persistence_h1_idx:self.persistence_h1_idx+1]  # (N, 1)
        cyclical = np.concatenate([betti_h1, entropy_h1, persistence_h1], axis=1)  # (N, 22)

        # Extract and transform landscape features
        landscape_raw = tda_features[:, self.landscape_slice]  # (N, 110)
        clip_min, clip_max = self.config.landscape_clip_range
        landscape_clipped = np.clip(landscape_raw, clip_min, clip_max)
        landscape_scaled = self.landscape_scaler.transform(landscape_clipped)
        landscape = self.landscape_pca.transform(landscape_scaled)  # (N, 20)

        if single_sample:
            return structural[0], cyclical[0], landscape[0]

        return structural, cyclical, landscape

    def transform_batch(self, tda_features: np.ndarray) -> dict:
        """
        Transform and return as dictionary (convenient for DataLoader).

        Args:
            tda_features: (N, 214) array

        Returns:
            dict with 'structural', 'cyclical', 'landscape' keys
        """
        structural, cyclical, landscape = self.transform(tda_features)
        return {
            'structural': structural.astype(np.float32),
            'cyclical': cyclical.astype(np.float32),
            'landscape': landscape.astype(np.float32),
        }

    def save(self, path: str):
        """Save fitted preprocessor to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'landscape_pca': self.landscape_pca,
                'landscape_scaler': self.landscape_scaler,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'TDAPreprocessor':
        """Load fitted preprocessor from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        preprocessor = cls(data['config'])
        preprocessor.landscape_pca = data['landscape_pca']
        preprocessor.landscape_scaler = data['landscape_scaler']
        preprocessor.is_fitted = True
        return preprocessor

    def get_feature_info(self) -> dict:
        """Get information about feature dimensions."""
        return {
            'structural': {
                'dim': self.config.structural_dim,
                'components': [
                    f'betti_h0_bin_{i}' for i in range(self.config.betti_bins_to_keep)
                ] + ['persistence_h0'],
            },
            'cyclical': {
                'dim': self.config.cyclical_dim,
                'components': [
                    f'betti_h1_bin_{i}' for i in range(self.config.betti_bins_to_keep)
                ] + ['entropy_h1', 'persistence_h1'],
            },
            'landscape': {
                'dim': self.config.landscape_pca_dims,
                'components': [f'landscape_pc_{i}' for i in range(self.config.landscape_pca_dims)],
                'explained_variance': (
                    self.landscape_pca.explained_variance_ratio_.tolist()
                    if self.is_fitted else None
                ),
            },
        }


def create_preprocessor(config: PreprocessingConfig, tda_train: np.ndarray) -> TDAPreprocessor:
    """
    Create and fit a preprocessor on training data.

    Args:
        config: Preprocessing configuration
        tda_train: Training TDA features (N, 214)

    Returns:
        Fitted TDAPreprocessor
    """
    preprocessor = TDAPreprocessor(config)
    preprocessor.fit(tda_train)
    return preprocessor
