"""
Training module for Hybrid Fusion model.

Provides:
- Trainer class for training loop
- Metrics computation
- Loss functions
"""

from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    MultiTaskMetrics,
    MetricTracker,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_all_metrics,
    format_metrics,
)
from .losses import (
    FocalLoss,
    MaskedMSELoss,
    MultiTaskLoss,
    create_loss_function,
)
from .trainer import Trainer, create_trainer

__all__ = [
    # Metrics
    "ClassificationMetrics",
    "RegressionMetrics",
    "MultiTaskMetrics",
    "MetricTracker",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_all_metrics",
    "format_metrics",
    # Losses
    "FocalLoss",
    "MaskedMSELoss",
    "MultiTaskLoss",
    "create_loss_function",
    # Trainer
    "Trainer",
    "create_trainer",
]
