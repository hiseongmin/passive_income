"""
Data loader for TDA model.

Handles temporal train/validation/test splitting and DataLoader creation.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import Config
from .preprocessing import load_flagged_data, validate_data
from .dataset import TDADataset

logger = logging.getLogger(__name__)


def create_weighted_sampler(dataset: "TDADataset") -> WeightedRandomSampler:
    """
    Create a weighted random sampler to handle class imbalance.

    Oversamples the minority class (triggers) to balance training.

    Args:
        dataset: TDADataset with get_labels() method

    Returns:
        WeightedRandomSampler for use in DataLoader
    """
    triggers, _ = dataset.get_labels()

    # Count samples per class
    n_positive = triggers.sum()
    n_negative = len(triggers) - n_positive

    # Compute weights: inverse of class frequency
    weight_positive = len(triggers) / (2 * n_positive) if n_positive > 0 else 1.0
    weight_negative = len(triggers) / (2 * n_negative) if n_negative > 0 else 1.0

    # Assign weight to each sample based on its class
    sample_weights = np.where(triggers == 1, weight_positive, weight_negative)
    sample_weights = torch.from_numpy(sample_weights).float()

    logger.info(f"Weighted sampler: pos_weight={weight_positive:.2f}, neg_weight={weight_negative:.2f}")
    logger.info(f"  Positive samples: {n_positive}, Negative samples: {n_negative}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def split_temporal(
    df: pd.DataFrame,
    validation_split: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for train/validation.

    CRITICAL: Walk-forward only - no future data leakage.
    Training data is earlier, validation data is later.

    Args:
        df: Full training DataFrame (sorted by time)
        validation_split: Fraction of data for validation

    Returns:
        Tuple of (train_df, val_df)
    """
    # Ensure sorted by time
    df = df.sort_values("open_time").reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df) * (1 - validation_split))

    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Temporal split: train={len(train_df)}, val={len(val_df)}")
    logger.info(f"Train period: {train_df['open_time'].min()} to {train_df['open_time'].max()}")
    logger.info(f"Val period: {val_df['open_time'].min()} to {val_df['open_time'].max()}")

    return train_df, val_df


def split_stratified_temporal(
    df: pd.DataFrame,
    validation_split: float = 0.15,
    n_blocks: int = 10,
    tolerance: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into temporal blocks while maintaining trigger rate balance.

    This addresses the distribution shift problem where naive temporal split
    creates severe imbalance (e.g., 12.3% train triggers vs 1.76% val triggers).

    Strategy:
    1. Divide data into n temporal blocks
    2. Calculate trigger rate per block
    3. Select validation blocks from later periods that best match overall rate
    4. Remaining blocks become training data

    CRITICAL: Still maintains walk-forward principle - validation blocks are
    always later than training blocks within each selected set.

    Args:
        df: Full training DataFrame (sorted by time)
        validation_split: Target fraction for validation
        n_blocks: Number of temporal blocks to create
        tolerance: Acceptable difference from target trigger rate

    Returns:
        Tuple of (train_df, val_df)
    """
    # Ensure sorted by time
    df = df.sort_values("open_time").reset_index(drop=True)

    overall_trigger_rate = df["Trigger"].mean()
    target_val_size = int(len(df) * validation_split)

    logger.info(f"Stratified temporal split: overall trigger rate = {overall_trigger_rate:.4f}")

    # Divide into n temporal blocks
    block_size = len(df) // n_blocks
    blocks = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else len(df)
        block_df = df.iloc[start_idx:end_idx].copy()
        trigger_rate = block_df["Trigger"].mean()
        blocks.append({
            "index": i,
            "df": block_df,
            "trigger_rate": trigger_rate,
            "size": len(block_df),
            "start_time": block_df["open_time"].min(),
            "end_time": block_df["open_time"].max(),
        })
        logger.debug(f"Block {i}: size={len(block_df)}, trigger_rate={trigger_rate:.4f}")

    # Select validation blocks from the LATER half of data
    # to maintain walk-forward principle
    candidate_blocks = blocks[n_blocks // 2:]  # Only consider later blocks

    # Find combination of blocks that:
    # 1. Has total size close to target_val_size
    # 2. Has trigger rate close to overall_trigger_rate

    best_val_blocks = None
    best_score = float("inf")

    # Try different combinations (simple greedy approach)
    for num_val_blocks in range(1, len(candidate_blocks) + 1):
        # Try selecting the last num_val_blocks blocks
        selected = candidate_blocks[-num_val_blocks:]
        combined_size = sum(b["size"] for b in selected)
        combined_triggers = sum(b["df"]["Trigger"].sum() for b in selected)
        combined_rate = combined_triggers / combined_size if combined_size > 0 else 0

        # Score: penalize both size mismatch and rate mismatch
        size_diff = abs(combined_size - target_val_size) / target_val_size
        rate_diff = abs(combined_rate - overall_trigger_rate) / (overall_trigger_rate + 1e-6)

        score = size_diff + 2.0 * rate_diff  # Weight rate matching more heavily

        if score < best_score:
            best_score = score
            best_val_blocks = selected

    # Create train/val DataFrames
    val_indices = {b["index"] for b in best_val_blocks}
    train_blocks = [b for b in blocks if b["index"] not in val_indices]

    train_df = pd.concat([b["df"] for b in train_blocks], ignore_index=True)
    val_df = pd.concat([b["df"] for b in best_val_blocks], ignore_index=True)

    # Sort by time again
    train_df = train_df.sort_values("open_time").reset_index(drop=True)
    val_df = val_df.sort_values("open_time").reset_index(drop=True)

    # Log statistics
    train_trigger_rate = train_df["Trigger"].mean()
    val_trigger_rate = val_df["Trigger"].mean()

    logger.info(f"Stratified split complete:")
    logger.info(f"  Train: {len(train_df)} samples, trigger rate = {train_trigger_rate:.4f}")
    logger.info(f"  Val: {len(val_df)} samples, trigger rate = {val_trigger_rate:.4f}")
    logger.info(f"  Rate ratio (train/val): {train_trigger_rate / (val_trigger_rate + 1e-6):.2f}")
    logger.info(f"  Train period: {train_df['open_time'].min()} to {train_df['open_time'].max()}")
    logger.info(f"  Val period: {val_df['open_time'].min()} to {val_df['open_time'].max()}")

    # Warn if significant imbalance remains
    rate_ratio = train_trigger_rate / (val_trigger_rate + 1e-6)
    if rate_ratio > 2.0 or rate_ratio < 0.5:
        logger.warning(f"Significant trigger rate imbalance remains: ratio = {rate_ratio:.2f}")
        logger.warning("Consider using more blocks or a different validation strategy")

    return train_df, val_df


def create_data_loaders(
    config: Config,
    project_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        config: Configuration object
        project_root: Project root directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent

    # Load training data
    logger.info("Loading training data...")
    train_full_df = load_flagged_data(
        config.data.data_dir,
        config.data.train_file,
        project_root,
    )
    validate_data(train_full_df)

    # Temporal split for train/validation
    use_stratified = getattr(config.data, 'use_stratified_split', False)
    if use_stratified:
        n_blocks = getattr(config.data, 'stratified_n_blocks', 10)
        logger.info(f"Using stratified temporal split with {n_blocks} blocks")
        train_df, val_df = split_stratified_temporal(
            train_full_df,
            config.data.validation_split,
            n_blocks=n_blocks,
        )
    else:
        logger.info("Using simple temporal split")
        train_df, val_df = split_temporal(
            train_full_df,
            config.data.validation_split,
        )

    # Load test data
    logger.info("Loading test data...")
    test_df = load_flagged_data(
        config.data.data_dir,
        config.data.test_file,
        project_root,
    )
    validate_data(test_df)

    # Create cache directory
    cache_dir = project_root / config.data.cache_dir
    cache_dir.mkdir(exist_ok=True)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TDADataset(
        df=train_df,
        config=config,
        cache_dir=cache_dir,
        split="train",
    )

    val_dataset = TDADataset(
        df=val_df,
        config=config,
        cache_dir=cache_dir,
        split="val",
    )

    test_dataset = TDADataset(
        df=test_df,
        config=config,
        cache_dir=cache_dir,
        split="test",
    )

    # Create weighted sampler for class imbalance (optional - can cause distribution mismatch)
    train_sampler = None
    use_shuffle = True
    if config.training.use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
        use_shuffle = False  # Can't use both sampler and shuffle
        logger.info("Using WeightedRandomSampler for class balancing")
    else:
        logger.info("Using standard shuffle (no weighted sampling)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=use_shuffle if train_sampler is None else False,
        num_workers=config.gpu.num_workers,
        pin_memory=config.gpu.pin_memory,
        persistent_workers=config.gpu.persistent_workers if config.gpu.num_workers > 0 else False,
        prefetch_factor=config.gpu.prefetch_factor if config.gpu.num_workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.gpu.num_workers,
        pin_memory=config.gpu.pin_memory,
        persistent_workers=config.gpu.persistent_workers if config.gpu.num_workers > 0 else False,
        prefetch_factor=config.gpu.prefetch_factor if config.gpu.num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.gpu.num_workers,
        pin_memory=config.gpu.pin_memory,
        persistent_workers=config.gpu.persistent_workers if config.gpu.num_workers > 0 else False,
        prefetch_factor=config.gpu.prefetch_factor if config.gpu.num_workers > 0 else None,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_sample_batch(loader: DataLoader) -> Tuple[torch.Tensor, ...]:
    """
    Get a sample batch for debugging/verification.

    Args:
        loader: DataLoader to sample from

    Returns:
        Tuple of tensors (ohlcv_seq, tda_features, complexity, trigger, max_pct)
    """
    batch = next(iter(loader))
    return batch
