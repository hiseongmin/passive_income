"""
TDA Standalone Training Script.

Trains the TDA Standalone model with:
- Multi-task learning (trigger + regime)
- Early stopping
- Learning rate scheduling
- Checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import argparse

from .config import TDAStandaloneConfig, load_config
from .dataset import create_data_loaders, create_test_loader
from .model import create_model
from .losses import create_loss_function, compute_class_weights
from .preprocessing import TDAPreprocessor
from .regime import RegimeLabeler


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """TDA Standalone Model Trainer."""

    def __init__(
        self,
        config: TDAStandaloneConfig,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        device: str = 'cuda',
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.training.lr_patience,
            factor=config.training.lr_factor,
            verbose=True,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.min_delta,
        )

        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_trigger_loss': [],
            'val_trigger_loss': [],
            'train_regime_loss': [],
            'val_regime_loss': [],
            'learning_rate': [],
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        trigger_loss = 0
        regime_loss = 0
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            structural = batch['structural'].to(self.device)
            cyclical = batch['cyclical'].to(self.device)
            landscape = batch['landscape'].to(self.device)
            trigger = batch['trigger'].to(self.device)
            regime = batch['regime'].to(self.device)

            # Forward pass
            outputs = self.model(structural, cyclical, landscape)

            # Compute loss
            targets = {'trigger': trigger, 'regime': regime}
            losses = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += losses['total'].item()
            trigger_loss += losses['trigger'].item()
            regime_loss += losses['regime'].item()
            n_batches += 1

        return {
            'total': total_loss / n_batches,
            'trigger': trigger_loss / n_batches,
            'regime': regime_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        trigger_loss = 0
        regime_loss = 0
        n_batches = 0

        # For metrics
        all_trigger_preds = []
        all_trigger_labels = []
        all_regime_preds = []
        all_regime_labels = []

        for batch in self.val_loader:
            structural = batch['structural'].to(self.device)
            cyclical = batch['cyclical'].to(self.device)
            landscape = batch['landscape'].to(self.device)
            trigger = batch['trigger'].to(self.device)
            regime = batch['regime'].to(self.device)

            outputs = self.model(structural, cyclical, landscape)

            targets = {'trigger': trigger, 'regime': regime}
            losses = self.loss_fn(outputs, targets)

            total_loss += losses['total'].item()
            trigger_loss += losses['trigger'].item()
            regime_loss += losses['regime'].item()
            n_batches += 1

            # Collect predictions
            trigger_prob = torch.sigmoid(outputs['trigger_logits'])
            all_trigger_preds.append(trigger_prob.cpu())
            all_trigger_labels.append(trigger.cpu())

            regime_pred = outputs['regime_logits'].argmax(dim=-1)
            all_regime_preds.append(regime_pred.cpu())
            all_regime_labels.append(regime.cpu())

        # Compute metrics
        trigger_preds = torch.cat(all_trigger_preds)
        trigger_labels = torch.cat(all_trigger_labels)
        regime_preds = torch.cat(all_regime_preds)
        regime_labels = torch.cat(all_regime_labels)

        # Trigger accuracy (at threshold 0.5)
        trigger_acc = ((trigger_preds > 0.5) == trigger_labels).float().mean().item()

        # Regime accuracy
        regime_acc = (regime_preds == regime_labels).float().mean().item()

        return {
            'total': total_loss / n_batches,
            'trigger': trigger_loss / n_batches,
            'regime': regime_loss / n_batches,
            'trigger_acc': trigger_acc,
            'regime_acc': regime_acc,
        }

    def train(self, start_epoch: int = 1) -> Dict[str, list]:
        """Full training loop."""
        if start_epoch > 1:
            print(f"\nResuming training from epoch {start_epoch}")
        else:
            print(f"\nStarting training for {self.config.training.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print("-" * 60)

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Update learning rate
            self.scheduler.step(val_losses['total'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_trigger_loss'].append(train_losses['trigger'])
            self.history['val_trigger_loss'].append(val_losses['trigger'])
            self.history['train_regime_loss'].append(train_losses['regime'])
            self.history['val_regime_loss'].append(val_losses['regime'])
            self.history['learning_rate'].append(current_lr)

            # Print progress
            print(f"Epoch {epoch:3d}/{self.config.training.epochs} | "
                  f"Train Loss: {train_losses['total']:.4f} | "
                  f"Val Loss: {val_losses['total']:.4f} | "
                  f"Trigger Acc: {val_losses['trigger_acc']:.3f} | "
                  f"Regime Acc: {val_losses['regime_acc']:.3f} | "
                  f"LR: {current_lr:.2e}")

            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pt')
                print(f"  -> New best model saved!")

            # Early stopping
            if self.early_stopping(val_losses['total']):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print("-" * 60)
        print(f"Training complete. Best model at epoch {self.best_epoch} "
              f"with val loss {self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            self.checkpoint_dir / filename,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        # Reset early stopping counter on resume
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=self.config.training.min_delta,
        )
        self.early_stopping.best_loss = self.best_val_loss
        return checkpoint['epoch']


def train_tda_standalone(
    config_path: Optional[str] = None,
    use_focal_loss: bool = False,
    resume: bool = False,
) -> Tuple[nn.Module, Dict, TDAPreprocessor, RegimeLabeler]:
    """
    Train TDA Standalone model.

    Args:
        config_path: Optional path to config YAML
        use_focal_loss: Whether to use focal loss
        resume: Whether to resume from checkpoint

    Returns:
        Tuple of (trained_model, history, preprocessor, regime_labeler)
    """
    # Load config
    config = load_config(config_path)

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, preprocessor, regime_labeler = create_data_loaders(config)

    # Compute class weights for imbalanced triggers
    all_triggers = []
    for batch in train_loader:
        all_triggers.append(batch['trigger'])
    all_triggers = torch.cat(all_triggers)
    pos_weight = compute_class_weights(all_triggers)
    print(f"Trigger positive weight: {pos_weight:.2f}")

    # Create model
    print("\nCreating model...")
    model = create_model(config.model, config.preprocessing, device)

    # Create loss function
    loss_fn = create_loss_function(
        config.training,
        pos_weight=pos_weight,
        use_focal=use_focal_loss,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    # Resume from checkpoint if requested
    start_epoch = 1
    if resume:
        checkpoint_path = Path(config.training.checkpoint_dir) / 'best_model.pt'
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from {checkpoint_path}")
            start_epoch = trainer.load_checkpoint('best_model.pt') + 1
            print(f"Resuming from epoch {start_epoch}, best val loss: {trainer.best_val_loss:.4f}")
        else:
            print("No checkpoint found, starting fresh")

    # Train
    history = trainer.train(start_epoch=start_epoch)

    # Load best model
    trainer.load_checkpoint('best_model.pt')

    # Save preprocessor and regime labeler
    preprocessor.save(str(trainer.checkpoint_dir / 'preprocessor.pkl'))
    regime_labeler.save(str(trainer.checkpoint_dir / 'regime_labeler.pkl'))

    # Save training history
    with open(trainer.checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    from .config import save_config
    save_config(config, str(trainer.checkpoint_dir / 'config.yaml'))

    print(f"\nAll artifacts saved to: {trainer.checkpoint_dir}")

    return model, history, preprocessor, regime_labeler


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    loss_fn: nn.Module,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    total_loss = 0
    n_batches = 0

    all_trigger_probs = []
    all_trigger_labels = []
    all_regime_probs = []
    all_regime_labels = []
    all_confidences = []

    for batch in test_loader:
        structural = batch['structural'].to(device)
        cyclical = batch['cyclical'].to(device)
        landscape = batch['landscape'].to(device)
        trigger = batch['trigger'].to(device)
        regime = batch['regime'].to(device)

        outputs = model(structural, cyclical, landscape)

        targets = {'trigger': trigger, 'regime': regime}
        losses = loss_fn(outputs, targets)

        total_loss += losses['total'].item()
        n_batches += 1

        # Collect outputs
        trigger_prob = torch.sigmoid(outputs['trigger_logits'])
        regime_prob = torch.softmax(outputs['regime_logits'], dim=-1)

        all_trigger_probs.append(trigger_prob.cpu())
        all_trigger_labels.append(trigger.cpu())
        all_regime_probs.append(regime_prob.cpu())
        all_regime_labels.append(regime.cpu())
        all_confidences.append(outputs['confidence'].cpu())

    # Concatenate
    trigger_probs = torch.cat(all_trigger_probs).numpy()
    trigger_labels = torch.cat(all_trigger_labels).numpy()
    regime_probs = torch.cat(all_regime_probs).numpy()
    regime_labels = torch.cat(all_regime_labels).numpy()
    confidences = torch.cat(all_confidences).numpy()

    # Compute metrics
    trigger_preds = (trigger_probs > 0.5).astype(int)
    regime_preds = regime_probs.argmax(axis=-1)

    # Trigger metrics
    trigger_acc = (trigger_preds == trigger_labels).mean()
    trigger_precision = (
        trigger_preds[trigger_labels == 1].sum() /
        max(trigger_preds.sum(), 1)
    )
    trigger_recall = (
        trigger_preds[trigger_labels == 1].sum() /
        max(trigger_labels.sum(), 1)
    )
    trigger_f1 = (
        2 * trigger_precision * trigger_recall /
        max(trigger_precision + trigger_recall, 1e-8)
    )

    # Regime metrics
    regime_acc = (regime_preds == regime_labels).mean()

    # Confidence analysis
    high_conf_mask = confidences > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_trigger_acc = (
            (trigger_preds[high_conf_mask] == trigger_labels[high_conf_mask]).mean()
        )
    else:
        high_conf_trigger_acc = 0.0

    return {
        'loss': total_loss / n_batches,
        'trigger_accuracy': float(trigger_acc),
        'trigger_precision': float(trigger_precision),
        'trigger_recall': float(trigger_recall),
        'trigger_f1': float(trigger_f1),
        'regime_accuracy': float(regime_acc),
        'mean_confidence': float(confidences.mean()),
        'high_conf_trigger_accuracy': float(high_conf_trigger_acc),
        'high_conf_samples': int(high_conf_mask.sum()),
        'total_samples': len(trigger_labels),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train TDA Standalone Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--focal', action='store_true', help='Use focal loss')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    # Train
    model, history, preprocessor, regime_labeler = train_tda_standalone(
        config_path=args.config,
        use_focal_loss=args.focal,
        resume=args.resume,
    )

    # Load config for test evaluation
    config = load_config(args.config)

    # Test evaluation
    print("\nEvaluating on test set...")
    test_loader = create_test_loader(config, preprocessor, regime_labeler)

    loss_fn = create_loss_function(config.training)
    device = config.device if torch.cuda.is_available() else 'cpu'

    metrics = evaluate_model(model, test_loader, loss_fn, device)

    print("\nTest Results:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
