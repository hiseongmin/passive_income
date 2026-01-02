"""
Training Script for Trigger Prediction Model

Complete training pipeline with:
- Data loading and preprocessing
- Model training with early stopping
- Logging and checkpointing
- Evaluation metrics
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.dataset import (
    TriggerDataset, create_data_splits, create_dataloaders, compute_class_weights
)
from features.feature_combiner import precompute_all_features, load_precomputed_features
from models.trigger_model import TriggerPredictionModel, TriggerModelConfig, create_model
from training.loss import TriggerLoss, compute_class_weights_from_labels
from evaluation.metrics import TriggerEvaluator


class Trainer:
    """
    Trainer class for TriggerPredictionModel.
    """

    def __init__(
        self,
        model: TriggerPredictionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: TriggerLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        loss_components = {'trigger_loss': 0.0, 'imminence_loss': 0.0, 'direction_loss': 0.0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in pbar:
            # Move to device
            x_5m = batch['x_5m'].to(self.device)
            x_1h = batch['x_1h'].to(self.device)
            tda = batch['tda'].to(self.device)
            micro = batch['micro'].to(self.device)
            trigger_target = batch['trigger'].to(self.device)
            imminence_target = batch['imminence'].to(self.device)
            direction_target = batch['direction'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            trigger_prob, imminence, direction_logits = self.model(x_5m, x_1h, tda, micro)

            # Compute loss
            loss, loss_dict = self.criterion(
                trigger_prob, imminence, direction_logits,
                trigger_target, imminence_target, direction_target
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += loss_dict.get(k, 0.0)
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average metrics
        metrics = {
            'loss': total_loss / n_batches,
            'trigger_loss': loss_components['trigger_loss'] / n_batches,
            'imminence_loss': loss_components['imminence_loss'] / n_batches,
            'direction_loss': loss_components['direction_loss'] / n_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        evaluator = TriggerEvaluator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc='Validating', leave=False)
        for batch in pbar:
            # Move to device
            x_5m = batch['x_5m'].to(self.device)
            x_1h = batch['x_1h'].to(self.device)
            tda = batch['tda'].to(self.device)
            micro = batch['micro'].to(self.device)
            trigger_target = batch['trigger'].to(self.device)
            imminence_target = batch['imminence'].to(self.device)
            direction_target = batch['direction'].to(self.device)

            # Forward pass
            trigger_prob, imminence, direction_logits = self.model(x_5m, x_1h, tda, micro)

            # Compute loss
            loss, _ = self.criterion(
                trigger_prob, imminence, direction_logits,
                trigger_target, imminence_target, direction_target
            )

            total_loss += loss.item()
            n_batches += 1

            # Update evaluator
            evaluator.update(
                trigger_prob.cpu(),
                imminence.cpu(),
                direction_logits.cpu(),
                trigger_target.cpu(),
                imminence_target.cpu(),
                direction_target.cpu()
            )

        # Compute metrics
        metrics = evaluator.compute_metrics()
        metrics['loss'] = total_loss / n_batches

        return metrics

    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 10,
        save_every: int = 5,
        start_epoch: int = 0
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            save_every: Save checkpoint every N epochs
            start_epoch: Epoch to start from (for resuming training)

        Returns:
            Training history
        """
        remaining_epochs = epochs - start_epoch
        print(f"\nStarting training for {remaining_epochs} epochs (from epoch {start_epoch + 1} to {epochs})...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 50)

        patience_counter = 0

        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            epoch_time = time.time() - epoch_start

            # Print progress
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val F1: {val_metrics.get('trigger_f1', 0):.4f}")
            print(f"  Val AUC: {val_metrics.get('trigger_auc', 0):.4f}")
            print(f"  LR: {train_metrics['lr']:.6f}")

            # Check for improvement
            current_f1 = val_metrics.get('trigger_f1', 0)
            if current_f1 > self.best_val_f1:
                self.best_val_f1 = current_f1
                self.best_epoch = epoch
                patience_counter = 0

                # Save best model
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"  -> New best model! F1: {current_f1:.4f}")
            else:
                patience_counter += 1

            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nTraining complete!")
        print(f"Best model at epoch {self.best_epoch} with F1: {self.best_val_f1:.4f}")

        return {
            'train': self.train_history,
            'val': self.val_history
        }

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main(args):
    """Main training function."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    df_5m = pd.read_csv(args.data_5m)
    df_1h = pd.read_csv(args.data_1h)
    print(f"5m data shape: {df_5m.shape}")
    print(f"1h data shape: {df_1h.shape}")

    # Load or compute features
    tda_features, micro_features = None, None
    if args.feature_dir and os.path.exists(args.feature_dir):
        print("\nLoading pre-computed features...")
        tda_features, micro_features = load_precomputed_features(args.feature_dir)

    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_data_splits(
        df_5m, df_1h,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        seq_len_5m=args.seq_len_5m,
        seq_len_1h=args.seq_len_1h,
        tda_features=tda_features,
        micro_features=micro_features
    )

    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples: {len(val_ds):,}")
    print(f"Test samples: {len(test_ds):,}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Compute class weights
    class_weights = compute_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    # Create model
    print("\nCreating model...")
    config = TriggerModelConfig(
        seq_len_5m=args.seq_len_5m,
        seq_len_1h=args.seq_len_1h,
        tda_features=9,
        micro_features=12,
        lstm_hidden_5m=args.lstm_hidden,
        lstm_hidden_1h=args.lstm_hidden // 2,
        lstm_layers=args.lstm_layers,
        nbeats_blocks=args.nbeats_blocks,
        nbeats_hidden=args.nbeats_hidden
    )
    model = create_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create loss function
    criterion = TriggerLoss(
        trigger_weight=1.0,
        imminence_weight=0.5,
        direction_weight=0.3,
        class_weights=class_weights.to(device),
        use_focal_loss=args.use_focal_loss
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume if os.path.isabs(args.resume)
                               else os.path.join(args.checkpoint_dir, args.resume),
                               map_location=device)
        start_epoch = checkpoint.get('epoch', 0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed from epoch {start_epoch}, best F1: {checkpoint.get('best_val_f1', 0):.4f}")

    # Create scheduler for remaining epochs
    remaining_epochs = args.epochs - start_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=remaining_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Set best_val_f1 from checkpoint
    if args.resume:
        trainer.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

    # Train
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_every=args.save_every,
        start_epoch=start_epoch
    )

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    trainer.model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'))['model_state_dict']
    )

    # Create test loader for trainer
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()

    print("\nTest Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save training history
    history_path = os.path.join(args.log_dir, 'training_history.npy')
    np.save(history_path, history)
    print(f"\nTraining history saved to {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Trigger Prediction Model')

    # Data paths
    parser.add_argument('--data-5m', type=str,
                        default='/notebooks/sa/data/BTCUSDT_perp_5m_labeled.csv',
                        help='Path to 5-minute labeled data')
    parser.add_argument('--data-1h', type=str,
                        default='/notebooks/sa/data/BTCUSDT_perp_1h.csv',
                        help='Path to 1-hour data')
    parser.add_argument('--feature-dir', type=str,
                        default='/notebooks/sa/data',
                        help='Directory with pre-computed features')

    # Data splits
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)

    # Sequence lengths
    parser.add_argument('--seq-len-5m', type=int, default=72)
    parser.add_argument('--seq-len-1h', type=int, default=6)

    # Model architecture
    parser.add_argument('--lstm-hidden', type=int, default=128)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--nbeats-blocks', type=int, default=3)
    parser.add_argument('--nbeats-hidden', type=int, default=256)

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use-focal-loss', action='store_true')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='/notebooks/sa/checkpoints')
    parser.add_argument('--log-dir', type=str, default='/notebooks/sa/logs')
    parser.add_argument('--save-every', type=int, default=5)

    # Other
    parser.add_argument('--num-workers', type=int, default=4)

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
