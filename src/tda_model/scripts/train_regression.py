#!/usr/bin/env python3
"""
Training script for multi-horizon return regression model.

Trains a model to predict forward returns at 1h, 4h, and 24h horizons.
Uses Huber loss for robustness to outliers (large price moves).

Usage:
    python -m scripts.train_regression
    python -m scripts.train_regression --epochs 200 --lr 0.0005
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import TDADataset
from tda_model.data.preprocessing import load_flagged_data, validate_data
from tda_model.models.regression_model import create_regression_model
from tda_model.models.losses import create_regression_loss
from tda_model.training.metrics import (
    compute_multi_horizon_metrics,
    format_multi_horizon_metrics,
    MultiHorizonRegressionMetrics,
    MultiHorizonMetricTracker,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multi-horizon return regression model"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size"
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="Early stopping patience"
    )
    parser.add_argument(
        "--horizon-weights", type=str, default="1.0,1.5,2.0",
        help="Comma-separated weights for 1h,4h,24h horizons"
    )
    return parser.parse_args()


def create_regression_data_loaders(
    config: Config,
    project_root: Path,
    batch_size: int,
    use_cached_samples: bool = True,
) -> tuple:
    """Create data loaders for regression mode using full datasets with samplers.

    Args:
        config: Configuration object
        project_root: Path to project root
        batch_size: Batch size for data loaders
        use_cached_samples: If True, limit training data to match existing TDA cache.
                           Significantly speeds up first run by avoiding TDA recomputation.
    """
    from torch.utils.data import SubsetRandomSampler

    # Load training data
    logger.info("Loading training data...")
    train_df = load_flagged_data(
        data_dir=config.data.data_dir,
        filename=config.data.train_file,
        project_root=project_root,
    )
    validate_data(train_df)

    # Load test data
    logger.info("Loading test data...")
    test_df = load_flagged_data(
        data_dir=config.data.data_dir,
        filename=config.data.test_file,
        project_root=project_root,
    )
    validate_data(test_df)

    # Cache directory for TDA features
    cache_dir = project_root / config.data.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check existing cache size to avoid recomputation
    if use_cached_samples:
        cache_file = cache_dir / "tda_features_train.npy"
        if cache_file.exists():
            import numpy as np
            cached_features = np.load(cache_file)
            cached_samples = len(cached_features)
            window_size = config.tda.window_size

            # Calculate how many rows we can use with this cache
            max_rows = cached_samples + window_size
            if len(train_df) > max_rows:
                logger.info(
                    f"Limiting training data to {max_rows} rows to use existing "
                    f"TDA cache ({cached_samples} samples)"
                )
                train_df = train_df.iloc[:max_rows].reset_index(drop=True)

    # Create single dataset from training data
    full_train_dataset = TDADataset(
        df=train_df,
        config=config,
        cache_dir=cache_dir,
        split="train",
        precompute_tda=True,
        mode="regression",
    )

    # Split indices for train/val (temporal split)
    n_samples = len(full_train_dataset)
    val_split = config.data.validation_split
    n_train = int(n_samples * (1 - val_split))

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_samples))

    logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # Create test dataset
    test_dataset = TDADataset(
        df=test_df,
        config=config,
        cache_dir=cache_dir,
        split="test",
        precompute_tda=True,
        mode="regression",
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create data loaders with samplers
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> MultiHorizonRegressionMetrics:
    """Evaluate model on a data loader."""
    model.eval()

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_loss_dict = {
        'loss_1h': 0.0,
        'loss_4h': 0.0,
        'loss_24h': 0.0,
    }
    n_batches = 0

    for batch in loader:
        ohlcv_seq, tda_features, complexity, targets = batch

        ohlcv_seq = ohlcv_seq.to(device)
        tda_features = tda_features.to(device)
        complexity = complexity.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(ohlcv_seq, tda_features, complexity)

        # Compute loss
        loss, loss_dict = loss_fn(predictions, targets)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k in total_loss_dict:
                total_loss_dict[k] += v
        n_batches += 1

        # Store predictions and targets
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Average losses
    avg_loss_dict = {
        'total_loss': total_loss / n_batches,
        'loss_1h': total_loss_dict['loss_1h'] / n_batches,
        'loss_4h': total_loss_dict['loss_4h'] / n_batches,
        'loss_24h': total_loss_dict['loss_24h'] / n_batches,
    }

    # Compute metrics
    metrics = compute_multi_horizon_metrics(
        predictions=all_predictions,
        targets=all_targets,
        loss_dict=avg_loss_dict,
    )

    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    accumulation_steps: int = 1,
) -> MultiHorizonRegressionMetrics:
    """Train for one epoch."""
    model.train()

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_loss_dict = {
        'loss_1h': 0.0,
        'loss_4h': 0.0,
        'loss_24h': 0.0,
    }
    n_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        ohlcv_seq, tda_features, complexity, targets = batch

        ohlcv_seq = ohlcv_seq.to(device)
        tda_features = tda_features.to(device)
        complexity = complexity.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(ohlcv_seq, tda_features, complexity)

        # Compute loss
        loss, loss_dict = loss_fn(predictions, targets)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        for k, v in loss_dict.items():
            if k in total_loss_dict:
                total_loss_dict[k] += v
        n_batches += 1

        # Store predictions and targets (every 10th batch to save memory)
        if i % 10 == 0:
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate stored batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Average losses
    avg_loss_dict = {
        'total_loss': total_loss / n_batches,
        'loss_1h': total_loss_dict['loss_1h'] / n_batches,
        'loss_4h': total_loss_dict['loss_4h'] / n_batches,
        'loss_24h': total_loss_dict['loss_24h'] / n_batches,
    }

    # Compute metrics
    metrics = compute_multi_horizon_metrics(
        predictions=all_predictions,
        targets=all_targets,
        loss_dict=avg_loss_dict,
    )

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent
    logger.info(f"Project root: {project_root}")

    # Setup output directory
    save_dir = project_root / "src" / "tda_model" / "models" / "regression_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_dir = project_root / "src" / "tda_model" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    # Parse horizon weights
    horizon_weights = tuple(float(w) for w in args.horizon_weights.split(","))
    logger.info(f"Horizon weights: {horizon_weights}")

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_regression_data_loaders(
        config=config,
        project_root=project_root,
        batch_size=args.batch_size,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model
    model = create_regression_model(config)
    model = model.to(device)

    trainable, total = model.get_num_parameters()
    logger.info(f"Model parameters: {trainable:,} trainable / {total:,} total")

    # Create loss function
    loss_fn = create_regression_loss(weights=horizon_weights)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")

    best_val_loss = float('inf')
    best_val_dir_acc = 0.0
    epochs_without_improvement = 0

    train_history = []
    val_history = []
    metric_tracker = MultiHorizonMetricTracker()

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        train_history.append(train_metrics.to_dict())

        # Validate
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        val_history.append(val_metrics.to_dict())

        # Update scheduler
        scheduler.step(val_metrics.total_loss)

        # Update metric tracker
        metric_tracker.update(val_metrics, epoch)

        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"Train:\n{format_multi_horizon_metrics(train_metrics, '  ')}")
        logger.info(f"Val:\n{format_multi_horizon_metrics(val_metrics, '  ')}")

        # Check for improvement
        improved = False
        if val_metrics.total_loss < best_val_loss:
            best_val_loss = val_metrics.total_loss
            improved = True

        if val_metrics.horizon_4h.direction_acc > best_val_dir_acc:
            best_val_dir_acc = val_metrics.horizon_4h.direction_acc
            improved = True

        if improved:
            epochs_without_improvement = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_dir_acc": best_val_dir_acc,
                "config": config.to_dict(),
                "metrics": val_metrics.to_dict(),
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            logger.info(f"  Saved best model (4h DirAcc={best_val_dir_acc:.2%})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= args.patience:
            logger.info(
                f"\nEarly stopping after {epoch} epochs "
                f"({epochs_without_improvement} without improvement)"
            )
            break

    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")

    # Load best model for final evaluation
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 60)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    logger.info(f"\n{format_multi_horizon_metrics(test_metrics, '')}")

    # Save results
    results = {
        "train_history": train_history,
        "val_history": val_history,
        "test_metrics": test_metrics.to_dict(),
        "best_val_loss": best_val_loss,
        "best_val_dir_acc": best_val_dir_acc,
        "total_epochs": epoch,
        "total_time_seconds": total_time,
        "horizon_weights": horizon_weights,
    }

    results_file = save_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Best model saved to: {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
