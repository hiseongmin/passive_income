"""
Evaluation Metrics for Trigger Prediction Model

Computes metrics for:
- Trigger prediction: Precision, Recall, F1, AUC-ROC
- Imminence prediction: MAE, MSE
- Direction prediction: Accuracy
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)


class TriggerEvaluator:
    """
    Evaluator for trigger prediction model.

    Accumulates predictions and computes various metrics.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for binary classification
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        self.trigger_probs = []
        self.trigger_targets = []
        self.imminence_preds = []
        self.imminence_targets = []
        self.direction_preds = []
        self.direction_targets = []

    def update(
        self,
        trigger_prob: torch.Tensor,
        imminence: torch.Tensor,
        direction_logits: torch.Tensor,
        trigger_target: torch.Tensor,
        imminence_target: torch.Tensor,
        direction_target: torch.Tensor
    ):
        """
        Update with batch predictions.

        Args:
            trigger_prob: Predicted trigger probability (batch, 1)
            imminence: Predicted imminence (batch, 1)
            direction_logits: Direction logits (batch, 3)
            trigger_target: Ground truth trigger (batch, 1)
            imminence_target: Ground truth imminence (batch, 1)
            direction_target: Ground truth direction (batch,)
        """
        # Convert to numpy
        trigger_prob = trigger_prob.squeeze().numpy()
        imminence = imminence.squeeze().numpy()
        direction_pred = torch.argmax(direction_logits, dim=1).numpy()
        trigger_target = trigger_target.squeeze().numpy()
        imminence_target = imminence_target.squeeze().numpy()
        direction_target = direction_target.numpy()

        # Accumulate
        self.trigger_probs.extend(trigger_prob.tolist() if trigger_prob.ndim > 0 else [trigger_prob.item()])
        self.trigger_targets.extend(trigger_target.tolist() if trigger_target.ndim > 0 else [trigger_target.item()])
        self.imminence_preds.extend(imminence.tolist() if imminence.ndim > 0 else [imminence.item()])
        self.imminence_targets.extend(imminence_target.tolist() if imminence_target.ndim > 0 else [imminence_target.item()])
        self.direction_preds.extend(direction_pred.tolist() if direction_pred.ndim > 0 else [direction_pred.item()])
        self.direction_targets.extend(direction_target.tolist() if direction_target.ndim > 0 else [direction_target.item()])

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Convert to numpy arrays
        trigger_probs = np.array(self.trigger_probs)
        trigger_targets = np.array(self.trigger_targets)
        trigger_preds = (trigger_probs >= self.threshold).astype(int)

        imminence_preds = np.array(self.imminence_preds)
        imminence_targets = np.array(self.imminence_targets)

        direction_preds = np.array(self.direction_preds)
        direction_targets = np.array(self.direction_targets)

        # ==================== Trigger Metrics ====================
        metrics['trigger_accuracy'] = accuracy_score(trigger_targets, trigger_preds)
        metrics['trigger_precision'] = precision_score(trigger_targets, trigger_preds, zero_division=0)
        metrics['trigger_recall'] = recall_score(trigger_targets, trigger_preds, zero_division=0)
        metrics['trigger_f1'] = f1_score(trigger_targets, trigger_preds, zero_division=0)

        # AUC-ROC (if both classes present)
        if len(np.unique(trigger_targets)) > 1:
            metrics['trigger_auc'] = roc_auc_score(trigger_targets, trigger_probs)
        else:
            metrics['trigger_auc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(trigger_targets, trigger_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['trigger_tn'] = int(tn)
            metrics['trigger_fp'] = int(fp)
            metrics['trigger_fn'] = int(fn)
            metrics['trigger_tp'] = int(tp)
            metrics['trigger_fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # ==================== Imminence Metrics (TRIGGER=1 only) ====================
        mask = trigger_targets == 1
        if mask.sum() > 0:
            metrics['imminence_mae'] = mean_absolute_error(
                imminence_targets[mask],
                imminence_preds[mask]
            )
            metrics['imminence_mse'] = mean_squared_error(
                imminence_targets[mask],
                imminence_preds[mask]
            )
            metrics['imminence_rmse'] = np.sqrt(metrics['imminence_mse'])
        else:
            metrics['imminence_mae'] = 0.0
            metrics['imminence_mse'] = 0.0
            metrics['imminence_rmse'] = 0.0

        # ==================== Direction Metrics (TRIGGER=1 only) ====================
        if mask.sum() > 0:
            # Filter for UP (0) and DOWN (1) only
            dir_mask = mask & (direction_targets < 2)
            if dir_mask.sum() > 0:
                metrics['direction_accuracy'] = accuracy_score(
                    direction_targets[dir_mask],
                    direction_preds[dir_mask]
                )
            else:
                metrics['direction_accuracy'] = 0.0
        else:
            metrics['direction_accuracy'] = 0.0

        return metrics

    def get_predictions(self) -> Dict[str, np.ndarray]:
        """
        Get all accumulated predictions.

        Returns:
            Dictionary of predictions and targets
        """
        return {
            'trigger_probs': np.array(self.trigger_probs),
            'trigger_targets': np.array(self.trigger_targets),
            'imminence_preds': np.array(self.imminence_preds),
            'imminence_targets': np.array(self.imminence_targets),
            'direction_preds': np.array(self.direction_preds),
            'direction_targets': np.array(self.direction_targets)
        }


def print_metrics_report(metrics: Dict[str, float]):
    """
    Print formatted metrics report.

    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print("\n--- Trigger Prediction ---")
    print(f"  Accuracy:  {metrics.get('trigger_accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('trigger_precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('trigger_recall', 0):.4f}")
    print(f"  F1-Score:  {metrics.get('trigger_f1', 0):.4f}")
    print(f"  AUC-ROC:   {metrics.get('trigger_auc', 0):.4f}")
    print(f"  FPR:       {metrics.get('trigger_fpr', 0):.4f}")

    if 'trigger_tp' in metrics:
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {metrics['trigger_tp']}, FP: {metrics['trigger_fp']}")
        print(f"    FN: {metrics['trigger_fn']}, TN: {metrics['trigger_tn']}")

    print("\n--- Imminence Prediction (TRIGGER=1) ---")
    print(f"  MAE:  {metrics.get('imminence_mae', 0):.4f}")
    print(f"  MSE:  {metrics.get('imminence_mse', 0):.4f}")
    print(f"  RMSE: {metrics.get('imminence_rmse', 0):.4f}")

    print("\n--- Direction Prediction (TRIGGER=1) ---")
    print(f"  Accuracy: {metrics.get('direction_accuracy', 0):.4f}")

    print("=" * 60)


def compute_trading_metrics(
    trigger_probs: np.ndarray,
    trigger_targets: np.ndarray,
    imminence_preds: np.ndarray,
    direction_preds: np.ndarray,
    direction_targets: np.ndarray,
    threshold: float = 0.5,
    imminence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute trading-specific metrics.

    Args:
        trigger_probs: Predicted trigger probabilities
        trigger_targets: Ground truth triggers
        imminence_preds: Predicted imminence scores
        direction_preds: Predicted directions
        direction_targets: Ground truth directions
        threshold: Trigger threshold
        imminence_threshold: Imminence threshold

    Returns:
        Dictionary of trading metrics
    """
    metrics = {}

    # High confidence predictions
    high_conf_mask = (trigger_probs >= threshold) & (imminence_preds >= imminence_threshold)

    if high_conf_mask.sum() > 0:
        # Precision of high confidence predictions
        high_conf_targets = trigger_targets[high_conf_mask]
        metrics['high_conf_precision'] = high_conf_targets.mean()

        # Direction accuracy of high confidence true positives
        tp_mask = high_conf_mask & (trigger_targets == 1)
        if tp_mask.sum() > 0:
            metrics['high_conf_direction_acc'] = accuracy_score(
                direction_targets[tp_mask],
                direction_preds[tp_mask]
            )
        else:
            metrics['high_conf_direction_acc'] = 0.0

        metrics['n_high_conf_signals'] = int(high_conf_mask.sum())
    else:
        metrics['high_conf_precision'] = 0.0
        metrics['high_conf_direction_acc'] = 0.0
        metrics['n_high_conf_signals'] = 0

    return metrics


if __name__ == "__main__":
    # Test metrics computation
    print("Testing TriggerEvaluator...")

    evaluator = TriggerEvaluator()

    # Create dummy data
    batch_size = 100
    for _ in range(10):
        trigger_prob = torch.sigmoid(torch.randn(batch_size, 1))
        imminence = torch.sigmoid(torch.randn(batch_size, 1))
        direction_logits = torch.randn(batch_size, 3)

        trigger_target = torch.randint(0, 2, (batch_size, 1)).float()
        imminence_target = torch.rand(batch_size, 1)
        direction_target = torch.randint(0, 3, (batch_size,))

        evaluator.update(
            trigger_prob, imminence, direction_logits,
            trigger_target, imminence_target, direction_target
        )

    metrics = evaluator.compute_metrics()
    print_metrics_report(metrics)

    # Test trading metrics
    preds = evaluator.get_predictions()
    trading_metrics = compute_trading_metrics(
        preds['trigger_probs'],
        preds['trigger_targets'],
        preds['imminence_preds'],
        preds['direction_preds'],
        preds['direction_targets']
    )

    print("\n--- Trading Metrics ---")
    for k, v in trading_metrics.items():
        print(f"  {k}: {v}")
