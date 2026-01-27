#!/usr/bin/env python3
"""
Training script for multi-horizon direction classification model.

Predicts price direction (UP/DOWN/NEUTRAL) at 1h, 4h, and 24h horizons.
Direction is determined by net_return = max_return + min_return.

Usage:
    python -m scripts.train_direction
    python -m scripts.train_direction --epochs 100 --lr 0.0001
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import TDADataset
from tda_model.data.preprocessing import load_flagged_data, validate_data
from tda_model.models.direction_model import (
    DirectionClassificationModel,
    WeightedCrossEntropyLoss,
    create_direction_model,
    compute_direction_labels,
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
        description="Train multi-horizon direction classification model"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.005,
        help="Threshold for UP/DOWN classification (default: 0.5%%)"
    )
    parser.add_argument(
        "--horizon-weights", type=str, default="1.0,1.5,2.0",
        help="Comma-separated weights for 1h,4h,24h horizons"
    )
    return parser.parse_args()


def create_direction_data_loaders(
    config: Config,
    project_root: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, TDADataset]:
    """Create data loaders using regression mode (we compute labels from returns)."""

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

    # Cache directory
    cache_dir = project_root / config.data.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check existing cache size
    cache_file = cache_dir / "tda_features_train.npy"
    if cache_file.exists():
        cached_features = np.load(cache_file)
        cached_samples = len(cached_features)
        window_size = config.tda.window_size
        max_rows = cached_samples + window_size
        if len(train_df) > max_rows:
            logger.info(
                f"Limiting training data to {max_rows} rows to use existing "
                f"TDA cache ({cached_samples} samples)"
            )
            train_df = train_df.iloc[:max_rows].reset_index(drop=True)

    # Create dataset in regression mode (we'll extract direction labels from returns)
    full_train_dataset = TDADataset(
        df=train_df,
        config=config,
        cache_dir=cache_dir,
        split="train",
        precompute_tda=True,
        mode="regression",
    )

    # Split indices for train/val
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

    # Create data loaders
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

    return train_loader, val_loader, test_loader, full_train_dataset


def returns_to_direction_labels(
    forward_returns: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert forward returns to direction labels.

    Args:
        forward_returns: Shape (batch, 6) - [max_1h, max_4h, max_24h, min_1h, min_4h, min_24h]
        threshold: Threshold for UP/DOWN classification

    Returns:
        Tuple of labels for (1h, 4h, 24h), each shape (batch,)
    """
    # Net return = max + min (positive = bullish bias)
    net_1h = forward_returns[:, 0] + forward_returns[:, 3]
    net_4h = forward_returns[:, 1] + forward_returns[:, 4]
    net_24h = forward_returns[:, 2] + forward_returns[:, 5]

    # Convert to labels: 0=DOWN, 1=NEUTRAL, 2=UP
    labels_1h = torch.ones_like(net_1h, dtype=torch.long)
    labels_1h[net_1h > threshold] = 2
    labels_1h[net_1h < -threshold] = 0

    labels_4h = torch.ones_like(net_4h, dtype=torch.long)
    labels_4h[net_4h > threshold] = 2
    labels_4h[net_4h < -threshold] = 0

    labels_24h = torch.ones_like(net_24h, dtype=torch.long)
    labels_24h[net_24h > threshold] = 2
    labels_24h[net_24h < -threshold] = 0

    return labels_1h, labels_4h, labels_24h


def compute_metrics(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute classification metrics."""
    correct = (pred_labels == true_labels).float()
    accuracy = correct.mean().item()

    # Per-class accuracy
    up_mask = true_labels == 2
    down_mask = true_labels == 0
    neutral_mask = true_labels == 1

    up_acc = correct[up_mask].mean().item() if up_mask.any() else 0.0
    down_acc = correct[down_mask].mean().item() if down_mask.any() else 0.0
    neutral_acc = correct[neutral_mask].mean().item() if neutral_mask.any() else 0.0

    # Class distribution
    n_up = up_mask.sum().item()
    n_down = down_mask.sum().item()
    n_neutral = neutral_mask.sum().item()
    total = len(true_labels)

    return {
        'accuracy': accuracy,
        'up_acc': up_acc,
        'down_acc': down_acc,
        'neutral_acc': neutral_acc,
        'up_pct': n_up / total if total > 0 else 0,
        'down_pct': n_down / total if total > 0 else 0,
        'neutral_pct': n_neutral / total if total > 0 else 0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate model on a data loader."""
    model.eval()

    all_preds_1h, all_preds_4h, all_preds_24h = [], [], []
    all_labels_1h, all_labels_4h, all_labels_24h = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        ohlcv_seq, tda_features, complexity, forward_returns = batch

        ohlcv_seq = ohlcv_seq.to(device)
        tda_features = tda_features.to(device)
        complexity = complexity.to(device)
        forward_returns = forward_returns.to(device)

        # Get direction labels
        labels_1h, labels_4h, labels_24h = returns_to_direction_labels(
            forward_returns, threshold
        )

        # Forward pass
        logits_1h, logits_4h, logits_24h = model(ohlcv_seq, tda_features, complexity)

        # Compute loss
        loss, _ = loss_fn(
            logits_1h, logits_4h, logits_24h,
            labels_1h, labels_4h, labels_24h,
        )

        total_loss += loss.item()
        n_batches += 1

        # Store predictions and labels
        all_preds_1h.append(torch.argmax(logits_1h, dim=1).cpu())
        all_preds_4h.append(torch.argmax(logits_4h, dim=1).cpu())
        all_preds_24h.append(torch.argmax(logits_24h, dim=1).cpu())
        all_labels_1h.append(labels_1h.cpu())
        all_labels_4h.append(labels_4h.cpu())
        all_labels_24h.append(labels_24h.cpu())

    # Concatenate
    all_preds_1h = torch.cat(all_preds_1h)
    all_preds_4h = torch.cat(all_preds_4h)
    all_preds_24h = torch.cat(all_preds_24h)
    all_labels_1h = torch.cat(all_labels_1h)
    all_labels_4h = torch.cat(all_labels_4h)
    all_labels_24h = torch.cat(all_labels_24h)

    # Compute metrics per horizon
    metrics_1h = compute_metrics(all_preds_1h, all_labels_1h)
    metrics_4h = compute_metrics(all_preds_4h, all_labels_4h)
    metrics_24h = compute_metrics(all_preds_24h, all_labels_24h)

    return {
        'loss': total_loss / n_batches,
        '1h': metrics_1h,
        '4h': metrics_4h,
        '24h': metrics_24h,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    threshold: float,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    all_preds_4h = []
    all_labels_4h = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        ohlcv_seq, tda_features, complexity, forward_returns = batch

        ohlcv_seq = ohlcv_seq.to(device)
        tda_features = tda_features.to(device)
        complexity = complexity.to(device)
        forward_returns = forward_returns.to(device)

        # Get direction labels
        labels_1h, labels_4h, labels_24h = returns_to_direction_labels(
            forward_returns, threshold
        )

        # Forward pass
        optimizer.zero_grad()
        logits_1h, logits_4h, logits_24h = model(ohlcv_seq, tda_features, complexity)

        # Compute loss
        loss, _ = loss_fn(
            logits_1h, logits_4h, logits_24h,
            labels_1h, labels_4h, labels_24h,
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Store 4h predictions for quick tracking
        all_preds_4h.append(torch.argmax(logits_4h, dim=1).cpu())
        all_labels_4h.append(labels_4h.cpu())

    # Quick metrics for 4h horizon
    all_preds_4h = torch.cat(all_preds_4h)
    all_labels_4h = torch.cat(all_labels_4h)
    metrics_4h = compute_metrics(all_preds_4h, all_labels_4h)

    return {
        'loss': total_loss / n_batches,
        '4h_accuracy': metrics_4h['accuracy'],
        '4h_up_acc': metrics_4h['up_acc'],
        '4h_down_acc': metrics_4h['down_acc'],
    }


def format_metrics(metrics: Dict, prefix: str = "") -> str:
    """Format metrics for logging."""
    lines = []
    lines.append(f"{prefix}Loss: {metrics['loss']:.4f}")

    for horizon in ['1h', '4h', '24h']:
        m = metrics[horizon]
        lines.append(
            f"{prefix}{horizon}: Acc={m['accuracy']:.1%} "
            f"(UP:{m['up_acc']:.1%}, DOWN:{m['down_acc']:.1%}, NEUT:{m['neutral_acc']:.1%}) "
            f"[{m['up_pct']:.0%}/{m['neutral_pct']:.0%}/{m['down_pct']:.0%}]"
        )

    return "\n".join(lines)


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent
    logger.info(f"Project root: {project_root}")

    # Setup output directory
    save_dir = project_root / "src" / "tda_model" / "models" / "direction_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_dir = project_root / "src" / "tda_model" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    # Parse horizon weights
    horizon_weights = tuple(float(w) for w in args.horizon_weights.split(","))
    logger.info(f"Horizon weights: {horizon_weights}")
    logger.info(f"Direction threshold: {args.threshold:.2%}")

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset = create_direction_data_loaders(
        config=config,
        project_root=project_root,
        batch_size=args.batch_size,
    )

    # Analyze class distribution
    logger.info("\nAnalyzing class distribution...")
    all_returns = train_dataset.get_forward_returns()
    net_4h = all_returns[:, 1] + all_returns[:, 4]
    n_up = np.sum(net_4h > args.threshold)
    n_down = np.sum(net_4h < -args.threshold)
    n_neutral = len(net_4h) - n_up - n_down
    logger.info(f"4h class distribution: UP={n_up/len(net_4h):.1%}, NEUTRAL={n_neutral/len(net_4h):.1%}, DOWN={n_down/len(net_4h):.1%}")

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Create model
    model = create_direction_model(config)
    model = model.to(device)

    trainable, total = model.get_num_parameters()
    logger.info(f"Model parameters: {trainable:,} trainable / {total:,} total")

    # Create loss function with class weights
    # Weight DOWN and UP higher since they're more actionable
    loss_fn = WeightedCrossEntropyLoss(
        horizon_weights=horizon_weights,
        class_weights=(1.5, 0.5, 1.5),  # DOWN, NEUTRAL, UP
        label_smoothing=0.1,
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize accuracy
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs")

    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_history = []
    val_history = []

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
            threshold=args.threshold,
        )
        train_history.append(train_metrics)

        # Validate
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=args.threshold,
        )
        val_history.append(val_metrics)

        # Update scheduler using 4h accuracy
        val_4h_acc = val_metrics['4h']['accuracy']
        scheduler.step(val_4h_acc)

        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"Train: Loss={train_metrics['loss']:.4f}, 4h Acc={train_metrics['4h_accuracy']:.1%}")
        logger.info(f"Val:\n{format_metrics(val_metrics, '  ')}")

        # Check for improvement
        if val_4h_acc > best_val_acc:
            best_val_acc = val_4h_acc
            epochs_without_improvement = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "config": config.to_dict(),
                "threshold": args.threshold,
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            logger.info(f"  Saved best model (4h Acc={best_val_acc:.1%})")
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
        threshold=args.threshold,
    )

    logger.info(f"\n{format_metrics(test_metrics, '')}")

    # Save results
    results = {
        "train_history": train_history,
        "val_history": [
            {
                'loss': v['loss'],
                '1h_acc': v['1h']['accuracy'],
                '4h_acc': v['4h']['accuracy'],
                '24h_acc': v['24h']['accuracy'],
            }
            for v in val_history
        ],
        "test_metrics": {
            'loss': test_metrics['loss'],
            '1h': test_metrics['1h'],
            '4h': test_metrics['4h'],
            '24h': test_metrics['24h'],
        },
        "best_val_acc": best_val_acc,
        "total_epochs": epoch,
        "total_time_seconds": total_time,
        "threshold": args.threshold,
        "horizon_weights": horizon_weights,
    }

    results_file = save_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Best model saved to: {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
