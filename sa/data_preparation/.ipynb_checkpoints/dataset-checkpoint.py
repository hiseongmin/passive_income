"""
PyTorch Dataset for Trigger Prediction Model

Handles multi-timeframe data loading and preprocessing.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split


class TriggerDataset(Dataset):
    """
    Dataset for trigger prediction with multi-timeframe support.

    Provides:
    - 5-minute OHLCV sequences (6 hours = 72 candles)
    - 1-hour OHLCV sequences (6 hours = 6 candles)
    - TDA features (pre-computed or computed on-the-fly)
    - Market microstructure features
    - Labels: TRIGGER, IMMINENCE, DIRECTION
    """

    def __init__(
        self,
        df_5m: pd.DataFrame,
        df_1h: pd.DataFrame,
        seq_len_5m: int = 72,
        seq_len_1h: int = 6,
        ohlcv_cols: List[str] = None,
        normalize: bool = True,
        tda_features: Optional[np.ndarray] = None,
        micro_features: Optional[np.ndarray] = None
    ):
        """
        Args:
            df_5m: 5-minute DataFrame with TRIGGER, IMMINENCE, DIRECTION columns
            df_1h: 1-hour DataFrame
            seq_len_5m: Number of 5-minute candles in sequence (72 = 6 hours)
            seq_len_1h: Number of 1-hour candles in sequence (6 = 6 hours)
            ohlcv_cols: OHLCV column names
            normalize: Whether to normalize OHLCV data
            tda_features: Pre-computed TDA features (n_samples, n_features)
            micro_features: Pre-computed microstructure features (n_samples, n_features)
        """
        self.df_5m = df_5m.reset_index(drop=True)
        self.df_1h = df_1h.reset_index(drop=True)
        self.seq_len_5m = seq_len_5m
        self.seq_len_1h = seq_len_1h
        self.normalize = normalize

        if ohlcv_cols is None:
            self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        else:
            self.ohlcv_cols = ohlcv_cols

        # Pre-compute hour keys for alignment
        self.df_5m['open_time'] = pd.to_datetime(self.df_5m['open_time'])
        self.df_1h['open_time'] = pd.to_datetime(self.df_1h['open_time'])

        self.df_5m['hour_key'] = self.df_5m['open_time'].dt.floor('H')
        self.df_1h['hour_key'] = self.df_1h['open_time']

        # Create hour-to-index mapping for 1h data
        self.hour_to_idx = {
            row['hour_key']: idx for idx, row in self.df_1h.iterrows()
        }

        # Store external features
        self.tda_features = tda_features
        self.micro_features = micro_features

        # Calculate valid indices (must have enough history)
        self.valid_indices = self._get_valid_indices()

        # Pre-compute normalization parameters if needed
        if self.normalize:
            self._compute_normalization_params()

    def _get_valid_indices(self) -> np.ndarray:
        """Get indices that have enough historical data."""
        # Need seq_len_5m candles of history for 5m data
        # Need seq_len_1h hours of history for 1h data
        min_5m_idx = self.seq_len_5m

        valid_mask = np.zeros(len(self.df_5m), dtype=bool)

        for i in range(min_5m_idx, len(self.df_5m)):
            # Check if we can get corresponding 1h data
            hour_key = self.df_5m.iloc[i]['hour_key']

            if hour_key in self.hour_to_idx:
                hour_idx = self.hour_to_idx[hour_key]
                if hour_idx >= self.seq_len_1h:
                    valid_mask[i] = True

        return np.where(valid_mask)[0]

    def _compute_normalization_params(self):
        """Compute mean and std for normalization."""
        ohlcv_data = self.df_5m[self.ohlcv_cols].values
        self.ohlcv_mean = ohlcv_data.mean(axis=0)
        self.ohlcv_std = ohlcv_data.std(axis=0) + 1e-8

    def _normalize_ohlcv(self, data: np.ndarray) -> np.ndarray:
        """Normalize OHLCV data using z-score."""
        if self.normalize:
            return (data - self.ohlcv_mean) / self.ohlcv_std
        return data

    def _get_5m_sequence(self, idx: int) -> np.ndarray:
        """Get 5-minute OHLCV sequence ending at idx."""
        start_idx = idx - self.seq_len_5m
        end_idx = idx

        data = self.df_5m.iloc[start_idx:end_idx][self.ohlcv_cols].values
        return self._normalize_ohlcv(data)

    def _get_1h_sequence(self, idx: int) -> np.ndarray:
        """Get 1-hour OHLCV sequence corresponding to the 5m index."""
        hour_key = self.df_5m.iloc[idx]['hour_key']
        hour_idx = self.hour_to_idx.get(hour_key, None)

        if hour_idx is None or hour_idx < self.seq_len_1h:
            # Return zeros if no valid 1h data
            return np.zeros((self.seq_len_1h, len(self.ohlcv_cols)), dtype=np.float32)

        start_idx = hour_idx - self.seq_len_1h
        end_idx = hour_idx

        data = self.df_1h.iloc[start_idx:end_idx][self.ohlcv_cols].values
        return self._normalize_ohlcv(data)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.valid_indices[idx]

        # Get sequences
        x_5m = self._get_5m_sequence(real_idx)
        x_1h = self._get_1h_sequence(real_idx)

        # Get TDA features (placeholder if not provided)
        if self.tda_features is not None:
            tda = self.tda_features[real_idx]
        else:
            tda = np.zeros(9, dtype=np.float32)  # Placeholder

        # Get microstructure features (placeholder if not provided)
        if self.micro_features is not None:
            micro = self.micro_features[real_idx]
        else:
            micro = np.zeros(12, dtype=np.float32)  # Placeholder

        # Get labels
        row = self.df_5m.iloc[real_idx]
        trigger = row['TRIGGER']
        imminence = row['IMMINENCE']
        direction = row['DIRECTION']

        return {
            'x_5m': torch.FloatTensor(x_5m),
            'x_1h': torch.FloatTensor(x_1h),
            'tda': torch.FloatTensor(tda),
            'micro': torch.FloatTensor(micro),
            'trigger': torch.FloatTensor([trigger]),
            'imminence': torch.FloatTensor([imminence]),
            'direction': torch.LongTensor([direction]).squeeze()
        }


def create_data_splits(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seq_len_5m: int = 72,
    seq_len_1h: int = 6,
    tda_features: Optional[np.ndarray] = None,
    micro_features: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Tuple[TriggerDataset, TriggerDataset, TriggerDataset]:
    """
    Create train/val/test dataset splits.

    Uses time-based splitting to avoid data leakage.

    Args:
        df_5m: Labeled 5-minute DataFrame
        df_1h: 1-hour DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seq_len_5m: 5-minute sequence length
        seq_len_1h: 1-hour sequence length
        tda_features: Pre-computed TDA features
        micro_features: Pre-computed microstructure features
        random_state: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(df_5m)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Time-based split
    df_5m_train = df_5m.iloc[:train_end].copy()
    df_5m_val = df_5m.iloc[train_end:val_end].copy()
    df_5m_test = df_5m.iloc[val_end:].copy()

    # Split features if provided
    tda_train = tda_features[:train_end] if tda_features is not None else None
    tda_val = tda_features[train_end:val_end] if tda_features is not None else None
    tda_test = tda_features[val_end:] if tda_features is not None else None

    micro_train = micro_features[:train_end] if micro_features is not None else None
    micro_val = micro_features[train_end:val_end] if micro_features is not None else None
    micro_test = micro_features[val_end:] if micro_features is not None else None

    # Create datasets
    train_dataset = TriggerDataset(
        df_5m_train, df_1h, seq_len_5m, seq_len_1h,
        tda_features=tda_train, micro_features=micro_train
    )
    val_dataset = TriggerDataset(
        df_5m_val, df_1h, seq_len_5m, seq_len_1h,
        tda_features=tda_val, micro_features=micro_val
    )
    test_dataset = TriggerDataset(
        df_5m_test, df_1h, seq_len_5m, seq_len_1h,
        tda_features=tda_test, micro_features=micro_test
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: TriggerDataset,
    val_dataset: TriggerDataset,
    test_dataset: TriggerDataset,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(dataset: TriggerDataset) -> torch.Tensor:
    """
    Compute class weights for handling class imbalance.

    Args:
        dataset: TriggerDataset instance

    Returns:
        Tensor of class weights [weight_0, weight_1]
    """
    df = dataset.df_5m.iloc[dataset.valid_indices]
    trigger_counts = df['TRIGGER'].value_counts()

    n_samples = len(df)
    n_classes = 2

    weights = []
    for cls in [0, 1]:
        count = trigger_counts.get(cls, 1)
        weight = n_samples / (n_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Test dataset creation
    import os

    DATA_DIR = '/notebooks/sa/data'

    if os.path.exists(os.path.join(DATA_DIR, 'BTCUSDT_perp_5m_labeled.csv')):
        print("Loading labeled data...")
        df_5m = pd.read_csv(os.path.join(DATA_DIR, 'BTCUSDT_perp_5m_labeled.csv'))
        df_1h = pd.read_csv(os.path.join(DATA_DIR, 'BTCUSDT_perp_1h.csv'))

        print("Creating datasets...")
        train_ds, val_ds, test_ds = create_data_splits(df_5m, df_1h)

        print(f"Train samples: {len(train_ds):,}")
        print(f"Val samples: {len(val_ds):,}")
        print(f"Test samples: {len(test_ds):,}")

        # Test single sample
        sample = train_ds[0]
        print(f"\nSample shapes:")
        print(f"  x_5m: {sample['x_5m'].shape}")
        print(f"  x_1h: {sample['x_1h'].shape}")
        print(f"  tda: {sample['tda'].shape}")
        print(f"  micro: {sample['micro'].shape}")
        print(f"  trigger: {sample['trigger']}")
        print(f"  imminence: {sample['imminence']}")
        print(f"  direction: {sample['direction']}")

        # Test class weights
        weights = compute_class_weights(train_ds)
        print(f"\nClass weights: {weights}")
    else:
        print(f"Labeled data not found in {DATA_DIR}")
        print("Run trigger_generator.py first to generate labeled data.")
