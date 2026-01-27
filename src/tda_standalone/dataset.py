"""
TDA Standalone Dataset Module.

Self-contained dataset that loads:
- TDA features from cache
- Labels (Trigger, Max_Pct) from CSV
- Pre-computed regime labels
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
from pathlib import Path

from .config import TDAStandaloneConfig, DataConfig
from .preprocessing import TDAPreprocessor
from .regime import RegimeLabeler


class TDAStandaloneDataset(Dataset):
    """
    Self-contained dataset using only cached TDA features.

    Loads:
    - TDA features from cache/*.npy
    - Labels from CSV (Trigger)
    - Pre-computed regime labels

    Returns batches with:
    - structural: (batch, 21) H0-based features
    - cyclical: (batch, 22) H1-based features
    - landscape: (batch, 20) Landscape features after PCA
    - trigger: (batch,) Binary trigger labels
    - regime: (batch,) Regime labels (0-3)
    """

    def __init__(
        self,
        tda_features: np.ndarray,
        labels: np.ndarray,
        regime_labels: np.ndarray,
        preprocessor: TDAPreprocessor,
    ):
        """
        Initialize dataset.

        Args:
            tda_features: (N, 214) raw TDA features
            labels: (N,) trigger labels
            regime_labels: (N,) regime labels
            preprocessor: Fitted TDAPreprocessor
        """
        self.tda_features = tda_features
        self.labels = labels
        self.regime_labels = regime_labels
        self.preprocessor = preprocessor

        # Pre-transform all features (faster than per-sample)
        self.structural, self.cyclical, self.landscape = preprocessor.transform(tda_features)

        # Convert to float32
        self.structural = self.structural.astype(np.float32)
        self.cyclical = self.cyclical.astype(np.float32)
        self.landscape = self.landscape.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        self.regime_labels = self.regime_labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'structural': torch.from_numpy(self.structural[idx]),
            'cyclical': torch.from_numpy(self.cyclical[idx]),
            'landscape': torch.from_numpy(self.landscape[idx]),
            'trigger': torch.tensor(self.labels[idx]),
            'regime': torch.tensor(self.regime_labels[idx]),
        }


def load_tda_from_cache(cache_dir: str, split: str) -> np.ndarray:
    """
    Load TDA features from cache.

    Args:
        cache_dir: Cache directory path
        split: 'train', 'val', or 'test'

    Returns:
        (N, 214) TDA features
    """
    cache_path = Path(cache_dir) / f"tda_features_{split}.npy"
    if not cache_path.exists():
        raise FileNotFoundError(f"TDA cache not found: {cache_path}")

    tda_features = np.load(cache_path)
    print(f"Loaded {split} TDA features: {tda_features.shape}")
    return tda_features


def load_labels_from_csv(
    csv_path: str,
    n_samples: int,
    min_idx: int,
    validation_split: float = 0.15,
    split: str = 'train',
) -> np.ndarray:
    """
    Load trigger labels from CSV and align with TDA features.

    Args:
        csv_path: Path to CSV file
        n_samples: Number of TDA samples (for alignment)
        min_idx: First valid index (window_size)
        validation_split: Fraction for validation
        split: 'train' or 'val'

    Returns:
        (N,) trigger labels
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values('open_time').reset_index(drop=True)

    # Get valid indices (same logic as TDA computation)
    all_valid_indices = list(range(min_idx, len(df)))

    # For training data, we need to compute the split
    if split in ['train', 'val']:
        # Split point
        split_idx = int(len(all_valid_indices) * (1 - validation_split))

        if split == 'train':
            valid_indices = all_valid_indices[:split_idx]
        else:  # val
            valid_indices = all_valid_indices[split_idx:]

        # Align with actual TDA samples
        valid_indices = valid_indices[:n_samples]
    else:
        # Test split uses all valid indices up to n_samples
        valid_indices = all_valid_indices[:n_samples]

    # Extract labels
    triggers = df.iloc[valid_indices]['Trigger'].values

    print(f"Loaded {split} labels: {len(triggers)} samples, "
          f"{triggers.sum():.0f} triggers ({100*triggers.mean():.1f}%)")

    return triggers


def create_data_loaders(
    config: TDAStandaloneConfig,
    preprocessor: Optional[TDAPreprocessor] = None,
    regime_labeler: Optional[RegimeLabeler] = None,
) -> Tuple[DataLoader, DataLoader, TDAPreprocessor, RegimeLabeler]:
    """
    Create train and validation data loaders.

    Args:
        config: Full configuration
        preprocessor: Optional pre-fitted preprocessor
        regime_labeler: Optional pre-fitted regime labeler

    Returns:
        Tuple of (train_loader, val_loader, preprocessor, regime_labeler)
    """
    data_config = config.data
    preproc_config = config.preprocessing
    training_config = config.training

    # Load TDA features from cache
    tda_train = load_tda_from_cache(data_config.tda_cache_dir, 'train')
    tda_val = load_tda_from_cache(data_config.tda_cache_dir, 'val')

    # Fit or use provided preprocessor
    if preprocessor is None:
        from .preprocessing import create_preprocessor
        preprocessor = create_preprocessor(preproc_config, tda_train)
    elif not preprocessor.is_fitted:
        preprocessor.fit(tda_train)

    # Fit or use provided regime labeler
    if regime_labeler is None:
        from .regime import RegimeLabeler
        regime_labeler = RegimeLabeler(
            n_regimes=config.model.num_regimes,
            random_state=config.seed,
        )
        regime_labeler.fit(tda_train)
        regime_labeler.print_regime_summary()

    # Compute regime labels
    regime_train = regime_labeler.transform(tda_train)
    regime_val = regime_labeler.transform(tda_val)

    # Load labels
    labels_train = load_labels_from_csv(
        csv_path=data_config.train_labels_path,
        n_samples=len(tda_train),
        min_idx=data_config.min_idx,
        validation_split=data_config.validation_split,
        split='train',
    )
    labels_val = load_labels_from_csv(
        csv_path=data_config.train_labels_path,
        n_samples=len(tda_val),
        min_idx=data_config.min_idx,
        validation_split=data_config.validation_split,
        split='val',
    )

    # Create datasets
    train_dataset = TDAStandaloneDataset(
        tda_features=tda_train,
        labels=labels_train,
        regime_labels=regime_train,
        preprocessor=preprocessor,
    )
    val_dataset = TDAStandaloneDataset(
        tda_features=tda_val,
        labels=labels_val,
        regime_labels=regime_val,
        preprocessor=preprocessor,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")

    return train_loader, val_loader, preprocessor, regime_labeler


def create_test_loader(
    config: TDAStandaloneConfig,
    preprocessor: TDAPreprocessor,
    regime_labeler: RegimeLabeler,
) -> DataLoader:
    """
    Create test data loader.

    Args:
        config: Full configuration
        preprocessor: Fitted preprocessor
        regime_labeler: Fitted regime labeler

    Returns:
        Test DataLoader
    """
    data_config = config.data
    training_config = config.training

    # Load test TDA features
    tda_test = load_tda_from_cache(data_config.tda_cache_dir, 'test')

    # Compute regime labels
    regime_test = regime_labeler.transform(tda_test)

    # Load test labels (from test CSV)
    df = pd.read_csv(data_config.test_labels_path)
    df = df.sort_values('open_time').reset_index(drop=True)

    # Get valid indices
    valid_indices = list(range(data_config.min_idx, len(df)))[:len(tda_test)]
    labels_test = df.iloc[valid_indices]['Trigger'].values

    print(f"Loaded test labels: {len(labels_test)} samples, "
          f"{labels_test.sum():.0f} triggers ({100*labels_test.mean():.1f}%)")

    # Create dataset
    test_dataset = TDAStandaloneDataset(
        tda_features=tda_test,
        labels=labels_test,
        regime_labels=regime_test,
        preprocessor=preprocessor,
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Test: {len(test_dataset):,} samples")

    return test_loader
