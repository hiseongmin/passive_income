"""
Training loop for Hybrid Fusion multi-task model.

Features:
- Mixed precision training (AMP) for GPU optimization
- torch.compile() for faster execution
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Fusion diagnostics logging
"""

import logging
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.metrics import precision_score, recall_score

from .metrics import (
    compute_all_metrics,
    MultiTaskMetrics,
    MetricTracker,
    format_metrics,
)
from .losses import MultiTaskLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Hybrid Fusion multi-task model with GPU optimizations.
    """

    def __init__(
        self,
        model: nn.Module,
        config: "CompleteModelConfig",
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: CompleteHybridFusionModel instance
            config: CompleteModelConfig with all hyperparameters
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Optional test DataLoader
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        device = getattr(config.gpu, 'device', 'cuda:0') if hasattr(config, 'gpu') else 'cuda:0'
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Apply torch.compile if enabled (PyTorch 2.0+)
        compile_model = getattr(config.gpu, 'compile_model', False) if hasattr(config, 'gpu') else False
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")

        # Setup cuDNN
        cudnn_benchmark = getattr(config.gpu, 'cudnn_benchmark', True) if hasattr(config, 'gpu') else True
        if torch.cuda.is_available() and cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("cuDNN benchmark enabled")

        # Setup loss function
        self.loss_fn = MultiTaskLoss(
            trigger_weight=config.training.trigger_loss_weight,
            max_pct_weight=config.training.max_pct_loss_weight,
            focal_alpha=config.training.focal_alpha,
            focal_gamma=config.training.focal_gamma,
        )

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas,
        )

        # Setup learning rate scheduler
        if config.training.scheduler_type == "OneCycleLR":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.training.max_lr,
                epochs=config.training.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=config.training.pct_start,
            )
            self.scheduler_step_per_batch = True
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience,
            )
            self.scheduler_step_per_batch = False

        # Setup mixed precision training
        use_amp = getattr(config.gpu, 'mixed_precision', True) if hasattr(config, 'gpu') else True
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Mixed precision training (AMP) enabled")
        else:
            self.scaler = None

        # Gradient accumulation
        self.accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        if self.accumulation_steps > 1:
            effective_batch = config.training.batch_size * self.accumulation_steps
            logger.info(f"Gradient accumulation: {self.accumulation_steps} steps "
                       f"(effective batch = {effective_batch})")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        self.metric_tracker = MetricTracker()

        # Paths
        self.save_dir = Path(config.logging.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Fusion diagnostics logging
        self.log_diagnostics_every = config.fusion.log_diagnostics_every

    def train_epoch(self) -> MultiTaskMetrics:
        """
        Train for one epoch with gradient accumulation.

        Returns:
            Training metrics for the epoch
        """
        self.model.train()

        total_loss = 0.0
        total_trigger_loss = 0.0
        total_max_pct_loss = 0.0
        n_batches = 0

        all_trigger_true = []
        all_trigger_prob = []
        all_max_pct_true = []
        all_max_pct_pred = []

        # For diagnostics
        all_gate_weights = []

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch
            ohlcv_seq, tda_features, complexity, trigger, max_pct = batch

            # Move to device
            ohlcv_seq = ohlcv_seq.to(self.device)
            tda_features = tda_features.to(self.device)
            complexity = complexity.to(self.device)
            trigger = trigger.to(self.device)
            max_pct = max_pct.to(self.device)

            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # Get diagnostics periodically
                    return_diag = (batch_idx % self.log_diagnostics_every == 0)
                    outputs = self.model(
                        ohlcv_seq, tda_features, complexity,
                        return_diagnostics=return_diag
                    )

                    if return_diag and len(outputs) > 2:
                        trigger_prob, max_pct_pred, diagnostics = outputs
                        if 'gate_weights' in diagnostics:
                            all_gate_weights.append(diagnostics['gate_weights'].detach().cpu())
                    else:
                        trigger_prob, max_pct_pred = outputs[:2]

                    # Pass probabilities directly to loss function (no logit conversion)
                    loss, trigger_loss, max_pct_loss = self.loss_fn(
                        trigger_prob, trigger, max_pct_pred, max_pct
                    )

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(scaled_loss).backward()

                # Step optimizer every accumulation_steps or at end of epoch
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Gradient clipping
                    if self.config.training.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip,
                        )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Scheduler step (if per-batch)
                    if self.scheduler_step_per_batch:
                        self.scheduler.step()

                    # Zero gradients for next accumulation
                    self.optimizer.zero_grad()
            else:
                # Standard forward pass (no AMP)
                return_diag = (batch_idx % self.log_diagnostics_every == 0)
                outputs = self.model(
                    ohlcv_seq, tda_features, complexity,
                    return_diagnostics=return_diag
                )

                if return_diag and len(outputs) > 2:
                    trigger_prob, max_pct_pred, diagnostics = outputs
                    if 'gate_weights' in diagnostics:
                        all_gate_weights.append(diagnostics['gate_weights'].detach().cpu())
                else:
                    trigger_prob, max_pct_pred = outputs[:2]

                # Pass probabilities directly to loss function (no logit conversion)
                loss, trigger_loss, max_pct_loss = self.loss_fn(
                    trigger_prob, trigger, max_pct_pred, max_pct
                )

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.accumulation_steps

                # Backward pass
                scaled_loss.backward()

                # Step optimizer every accumulation_steps or at end of epoch
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Gradient clipping
                    if self.config.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip,
                        )

                    # Optimizer step
                    self.optimizer.step()

                    # Scheduler step (if per-batch)
                    if self.scheduler_step_per_batch:
                        self.scheduler.step()

                    # Zero gradients for next accumulation
                    self.optimizer.zero_grad()

            # Accumulate losses (use unscaled loss for logging)
            total_loss += loss.item()
            total_trigger_loss += trigger_loss.item()
            total_max_pct_loss += max_pct_loss.item()
            n_batches += 1

            # Store predictions for metrics
            with torch.no_grad():
                all_trigger_true.append(trigger.cpu().numpy())
                all_trigger_prob.append(trigger_prob.cpu().numpy())
                all_max_pct_true.append(max_pct.cpu().numpy())
                all_max_pct_pred.append(max_pct_pred.cpu().numpy())

            # Logging
            log_interval = getattr(self.config.logging, 'log_interval', 100)
            if batch_idx % log_interval == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}"
                )

        # Log gate weight statistics
        if all_gate_weights:
            gate_weights = torch.cat(all_gate_weights, dim=0)
            mean_weights = gate_weights.mean(dim=0)
            logger.info(f"Gate weights (mean): {mean_weights.numpy()}")

        # Compute epoch metrics
        all_trigger_true = np.concatenate(all_trigger_true)
        all_trigger_prob = np.concatenate(all_trigger_prob)
        all_max_pct_true = np.concatenate(all_max_pct_true)
        all_max_pct_pred = np.concatenate(all_max_pct_pred)

        trigger_pred = (all_trigger_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(
            trigger_true=all_trigger_true,
            trigger_pred=trigger_pred,
            trigger_prob=all_trigger_prob,
            max_pct_true=all_max_pct_true,
            max_pct_pred=all_max_pct_pred,
            total_loss=total_loss / n_batches,
            trigger_loss=total_trigger_loss / n_batches,
            max_pct_loss=total_max_pct_loss / n_batches,
        )

        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> MultiTaskMetrics:
        """
        Evaluate model on a data loader.

        Args:
            loader: DataLoader to evaluate on

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_trigger_loss = 0.0
        total_max_pct_loss = 0.0
        n_batches = 0

        all_trigger_true = []
        all_trigger_prob = []
        all_max_pct_true = []
        all_max_pct_pred = []

        for batch in loader:
            ohlcv_seq, tda_features, complexity, trigger, max_pct = batch

            ohlcv_seq = ohlcv_seq.to(self.device)
            tda_features = tda_features.to(self.device)
            complexity = complexity.to(self.device)
            trigger = trigger.to(self.device)
            max_pct = max_pct.to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    trigger_prob, max_pct_pred = self.model(
                        ohlcv_seq, tda_features, complexity
                    )
                    # Pass probabilities directly to loss function (no logit conversion)
                    loss, trigger_loss, max_pct_loss = self.loss_fn(
                        trigger_prob, trigger, max_pct_pred, max_pct
                    )
            else:
                trigger_prob, max_pct_pred = self.model(
                    ohlcv_seq, tda_features, complexity
                )
                # Pass probabilities directly to loss function (no logit conversion)
                loss, trigger_loss, max_pct_loss = self.loss_fn(
                    trigger_prob, trigger, max_pct_pred, max_pct
                )

            total_loss += loss.item()
            total_trigger_loss += trigger_loss.item()
            total_max_pct_loss += max_pct_loss.item()
            n_batches += 1

            all_trigger_true.append(trigger.cpu().numpy())
            all_trigger_prob.append(trigger_prob.cpu().numpy())
            all_max_pct_true.append(max_pct.cpu().numpy())
            all_max_pct_pred.append(max_pct_pred.cpu().numpy())

        # Compute metrics
        all_trigger_true = np.concatenate(all_trigger_true)
        all_trigger_prob = np.concatenate(all_trigger_prob)
        all_max_pct_true = np.concatenate(all_max_pct_true)
        all_max_pct_pred = np.concatenate(all_max_pct_pred)

        # Use provided threshold or default to 0.5
        threshold = getattr(self, '_eval_threshold', 0.5)
        trigger_pred = (all_trigger_prob >= threshold).astype(int)

        metrics = compute_all_metrics(
            trigger_true=all_trigger_true,
            trigger_pred=trigger_pred,
            trigger_prob=all_trigger_prob,
            max_pct_true=all_max_pct_true,
            max_pct_pred=all_max_pct_pred,
            total_loss=total_loss / n_batches,
            trigger_loss=total_trigger_loss / n_batches,
            max_pct_loss=total_max_pct_loss / n_batches,
        )

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with training history and best metrics
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # Get parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

        start_time = time.time()
        train_history = []
        val_history = []

        for epoch in range(1, self.config.training.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()
            train_history.append(train_metrics.to_dict())

            # Validate
            val_metrics = self.evaluate(self.val_loader)
            val_history.append(val_metrics.to_dict())

            # Update metric tracker
            self.metric_tracker.update(val_metrics, epoch)

            # Update learning rate scheduler (if per-epoch)
            if not self.scheduler_step_per_batch:
                self.scheduler.step(val_metrics.total_loss)

            # Logging
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"\nEpoch {epoch}/{self.config.training.epochs} ({epoch_time:.1f}s) LR={current_lr:.2e}")
            logger.info(f"Train:\n{format_metrics(train_metrics, '  ')}")
            logger.info(f"Val:\n{format_metrics(val_metrics, '  ')}")

            # Check for improvement
            improved = False
            if val_metrics.total_loss < self.best_val_loss:
                self.best_val_loss = val_metrics.total_loss
                improved = True

            if val_metrics.classification.f1 > self.best_val_f1:
                self.best_val_f1 = val_metrics.classification.f1
                improved = True

            if improved:
                self.epochs_without_improvement = 0
                save_best = getattr(self.config.logging, 'save_best_only', True)
                if save_best:
                    self.save_checkpoint("best_model.pt", val_metrics)
                    logger.info(f"  Saved best model (F1={self.best_val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                logger.info(
                    f"\nEarly stopping triggered after {epoch} epochs "
                    f"({self.epochs_without_improvement} epochs without improvement)"
                )
                break

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/60:.1f} minutes")

        # Load best model for final evaluation
        self.load_checkpoint("best_model.pt")

        # Find optimal threshold on validation set
        optimal_threshold, opt_precision, opt_recall = self.find_optimal_threshold(
            self.val_loader, min_recall=0.40
        )

        # Final evaluation on test set if available
        test_metrics = None
        test_metrics_optimal = None
        if self.test_loader is not None:
            # Evaluate with default threshold
            test_metrics = self.evaluate(self.test_loader)
            logger.info(f"\nTest set evaluation (threshold=0.5):\n{format_metrics(test_metrics, '  ')}")

            # Evaluate with optimal threshold
            self._eval_threshold = optimal_threshold
            test_metrics_optimal = self.evaluate(self.test_loader)
            logger.info(f"\nTest set evaluation (threshold={optimal_threshold:.2f}):\n{format_metrics(test_metrics_optimal, '  ')}")
            self._eval_threshold = 0.5  # Reset

        return {
            "train_history": train_history,
            "val_history": val_history,
            "test_metrics": test_metrics.to_dict() if test_metrics else None,
            "test_metrics_optimal": test_metrics_optimal.to_dict() if test_metrics_optimal else None,
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "total_epochs": self.current_epoch,
            "total_time_seconds": total_time,
            "optimal_threshold": optimal_threshold,
            "optimal_threshold_precision": opt_precision,
            "optimal_threshold_recall": opt_recall,
        }

    @torch.no_grad()
    def find_optimal_threshold(
        self,
        loader: DataLoader,
        min_recall: float = 0.40,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        Find optimal threshold that maximizes precision while maintaining minimum recall.

        Args:
            loader: DataLoader for threshold search
            min_recall: Minimum acceptable recall
            thresholds: Thresholds to try

        Returns:
            Tuple of (best_threshold, precision, recall)
        """
        if thresholds is None:
            thresholds = np.arange(0.30, 0.91, 0.05)

        self.model.eval()

        # Collect all predictions
        all_trigger_true = []
        all_trigger_prob = []

        for batch in loader:
            ohlcv_seq, tda_features, complexity, trigger, max_pct = batch

            ohlcv_seq = ohlcv_seq.to(self.device)
            tda_features = tda_features.to(self.device)
            complexity = complexity.to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    trigger_prob, _ = self.model(ohlcv_seq, tda_features, complexity)
            else:
                trigger_prob, _ = self.model(ohlcv_seq, tda_features, complexity)

            all_trigger_true.append(trigger.cpu().numpy())
            all_trigger_prob.append(trigger_prob.cpu().numpy())

        all_trigger_true = np.concatenate(all_trigger_true).flatten()
        all_trigger_prob = np.concatenate(all_trigger_prob).flatten()

        # Search for best threshold
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0

        logger.info("\nThreshold search (prioritizing precision):")
        logger.info(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        logger.info("-" * 45)

        for thresh in thresholds:
            preds = (all_trigger_prob >= thresh).astype(int)

            if preds.sum() == 0:
                prec, rec = 0.0, 0.0
            else:
                prec = precision_score(all_trigger_true, preds, zero_division=0)
                rec = recall_score(all_trigger_true, preds, zero_division=0)

            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            logger.info(f"{thresh:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

            if rec >= min_recall and prec > best_precision:
                best_precision = prec
                best_recall = rec
                best_threshold = thresh

        # Fallback to best F1 if no threshold meets min_recall
        if best_precision == 0.0:
            logger.warning(f"No threshold achieves min_recall={min_recall}, using best F1 instead")
            best_f1 = 0.0
            for thresh in thresholds:
                preds = (all_trigger_prob >= thresh).astype(int)
                if preds.sum() > 0:
                    prec = precision_score(all_trigger_true, preds, zero_division=0)
                    rec = recall_score(all_trigger_true, preds, zero_division=0)
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_precision = prec
                        best_recall = rec
                        best_threshold = thresh

        logger.info("-" * 45)
        logger.info(f"Best threshold: {best_threshold:.2f} (Precision: {best_precision:.4f}, Recall: {best_recall:.4f})")

        return best_threshold, best_precision, best_recall

    def save_checkpoint(self, filename: str, metrics: Optional[MultiTaskMetrics] = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
        }

        if metrics is not None:
            checkpoint["metrics"] = metrics.to_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_f1 = checkpoint["best_val_f1"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


def create_trainer(
    model: nn.Module,
    config: "CompleteModelConfig",
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
) -> Trainer:
    """
    Create a trainer instance.

    Args:
        model: CompleteHybridFusionModel instance
        config: CompleteModelConfig with all hyperparameters
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Optional test DataLoader

    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
