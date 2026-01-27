"""
Evaluation metrics for multi-task model.

Classification metrics for trigger prediction:
- Accuracy, Precision, Recall, F1, AUC-ROC

Regression metrics for max_pct prediction (on positive triggers only):
- MSE, MAE, R², MAPE
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "n_samples": self.n_samples,
        }


@dataclass
class MultiTaskMetrics:
    """Container for all metrics."""
    classification: ClassificationMetrics
    regression: RegressionMetrics
    total_loss: float
    trigger_loss: float
    max_pct_loss: float

    def to_dict(self) -> Dict[str, float]:
        result = {
            "total_loss": self.total_loss,
            "trigger_loss": self.trigger_loss,
            "max_pct_loss": self.max_pct_loss,
        }
        # Add classification metrics with prefix
        for k, v in self.classification.to_dict().items():
            result[f"cls_{k}"] = v
        # Add regression metrics with prefix
        for k, v in self.regression.to_dict().items():
            result[f"reg_{k}"] = v
        return result


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """
    Compute classification metrics for trigger prediction.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels or probabilities
        y_prob: Predicted probabilities (if y_pred is thresholded labels)
        threshold: Threshold for converting probabilities to labels

    Returns:
        ClassificationMetrics object
    """
    y_true = np.asarray(y_true).flatten()

    # If y_pred looks like probabilities, threshold it
    if y_prob is None:
        y_pred = np.asarray(y_pred).flatten()
        if y_pred.max() <= 1.0 and y_pred.min() >= 0.0 and not np.all(np.isin(y_pred, [0, 1])):
            y_prob = y_pred.copy()
            y_pred = (y_pred >= threshold).astype(int)
        else:
            y_pred = y_pred.astype(int)
    else:
        y_prob = np.asarray(y_prob).flatten()
        y_pred = np.asarray(y_pred).flatten().astype(int)

    # Compute metrics
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC-ROC requires probabilities
        if y_prob is not None and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        # Manual computation
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        auc = 0.0  # Can't compute without sklearn

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc_roc=float(auc),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> RegressionMetrics:
    """
    Compute regression metrics for max_pct prediction.

    Args:
        y_true: Ground truth max percentages
        y_pred: Predicted max percentages
        mask: Boolean mask for valid samples (positive triggers)

    Returns:
        RegressionMetrics object
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Apply mask if provided
    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    n_samples = len(y_true)

    if n_samples == 0:
        return RegressionMetrics(
            mse=0.0,
            rmse=0.0,
            mae=0.0,
            r2=0.0,
            mape=0.0,
            n_samples=0,
        )

    if SKLEARN_AVAILABLE:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if n_samples > 1 else 0.0
    else:
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    rmse = np.sqrt(mse)

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    non_zero_mask = np.abs(y_true) > 1e-8
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = 0.0

    return RegressionMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2),
        mape=float(mape),
        n_samples=int(n_samples),
    )


def compute_all_metrics(
    trigger_true: np.ndarray,
    trigger_pred: np.ndarray,
    trigger_prob: np.ndarray,
    max_pct_true: np.ndarray,
    max_pct_pred: np.ndarray,
    total_loss: float = 0.0,
    trigger_loss: float = 0.0,
    max_pct_loss: float = 0.0,
) -> MultiTaskMetrics:
    """
    Compute all metrics for multi-task evaluation.

    Args:
        trigger_true: Ground truth triggers
        trigger_pred: Predicted triggers (thresholded)
        trigger_prob: Predicted trigger probabilities
        max_pct_true: Ground truth max percentages
        max_pct_pred: Predicted max percentages
        total_loss: Total training loss
        trigger_loss: Trigger classification loss
        max_pct_loss: Max_pct regression loss

    Returns:
        MultiTaskMetrics object
    """
    # Classification metrics
    cls_metrics = compute_classification_metrics(
        y_true=trigger_true,
        y_pred=trigger_pred,
        y_prob=trigger_prob,
    )

    # Regression metrics (only on positive triggers)
    mask = np.asarray(trigger_true).flatten() > 0.5
    reg_metrics = compute_regression_metrics(
        y_true=max_pct_true,
        y_pred=max_pct_pred,
        mask=mask,
    )

    return MultiTaskMetrics(
        classification=cls_metrics,
        regression=reg_metrics,
        total_loss=float(total_loss),
        trigger_loss=float(trigger_loss),
        max_pct_loss=float(max_pct_loss),
    )


class MetricTracker:
    """
    Track metrics over training epochs.
    """

    def __init__(self):
        self.history: Dict[str, list] = {}
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: int = 0

    def update(self, metrics: MultiTaskMetrics, epoch: int):
        """Update tracker with new metrics."""
        metrics_dict = metrics.to_dict()

        for key, value in metrics_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        # Track best F1 score
        if metrics.classification.f1 > self.best_metrics.get("best_f1", 0):
            self.best_metrics["best_f1"] = metrics.classification.f1
            self.best_metrics["best_accuracy"] = metrics.classification.accuracy
            self.best_metrics["best_auc"] = metrics.classification.auc_roc
            self.best_epoch = epoch

    def get_best(self) -> Tuple[Dict[str, float], int]:
        """Get best metrics and epoch."""
        return self.best_metrics, self.best_epoch

    def get_history(self, key: str) -> list:
        """Get history for a specific metric."""
        return self.history.get(key, [])


@dataclass
class HorizonMetrics:
    """Metrics for a single horizon (1h, 4h, or 24h)."""
    mae_max: float       # MAE for max_return prediction
    mae_min: float       # MAE for min_return prediction
    rmse_max: float      # RMSE for max_return prediction
    rmse_min: float      # RMSE for min_return prediction
    r2_max: float        # R² for max_return prediction
    r2_min: float        # R² for min_return prediction
    direction_acc: float # Direction accuracy (predicted direction matches actual)
    correlation: float   # Pearson correlation between predicted and actual

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae_max": self.mae_max,
            "mae_min": self.mae_min,
            "rmse_max": self.rmse_max,
            "rmse_min": self.rmse_min,
            "r2_max": self.r2_max,
            "r2_min": self.r2_min,
            "direction_acc": self.direction_acc,
            "correlation": self.correlation,
        }


@dataclass
class MultiHorizonRegressionMetrics:
    """Container for multi-horizon regression metrics."""
    horizon_1h: HorizonMetrics
    horizon_4h: HorizonMetrics
    horizon_24h: HorizonMetrics
    total_loss: float
    loss_1h: float
    loss_4h: float
    loss_24h: float

    def to_dict(self) -> Dict[str, float]:
        result = {
            "total_loss": self.total_loss,
            "loss_1h": self.loss_1h,
            "loss_4h": self.loss_4h,
            "loss_24h": self.loss_24h,
        }
        # Add horizon metrics with prefixes
        for k, v in self.horizon_1h.to_dict().items():
            result[f"1h_{k}"] = v
        for k, v in self.horizon_4h.to_dict().items():
            result[f"4h_{k}"] = v
        for k, v in self.horizon_24h.to_dict().items():
            result[f"24h_{k}"] = v
        return result


def compute_horizon_metrics(
    y_true_max: np.ndarray,
    y_pred_max: np.ndarray,
    y_true_min: np.ndarray,
    y_pred_min: np.ndarray,
) -> HorizonMetrics:
    """
    Compute metrics for a single horizon.

    Args:
        y_true_max: Ground truth max returns
        y_pred_max: Predicted max returns
        y_true_min: Ground truth min returns (negative values)
        y_pred_min: Predicted min returns

    Returns:
        HorizonMetrics object
    """
    y_true_max = np.asarray(y_true_max).flatten()
    y_pred_max = np.asarray(y_pred_max).flatten()
    y_true_min = np.asarray(y_true_min).flatten()
    y_pred_min = np.asarray(y_pred_min).flatten()

    # Handle NaN values - filter out samples where any value is NaN
    valid_mask = (
        ~np.isnan(y_true_max) & ~np.isnan(y_pred_max) &
        ~np.isnan(y_true_min) & ~np.isnan(y_pred_min) &
        ~np.isinf(y_true_max) & ~np.isinf(y_pred_max) &
        ~np.isinf(y_true_min) & ~np.isinf(y_pred_min)
    )
    y_true_max = y_true_max[valid_mask]
    y_pred_max = y_pred_max[valid_mask]
    y_true_min = y_true_min[valid_mask]
    y_pred_min = y_pred_min[valid_mask]

    n_samples = len(y_true_max)

    if n_samples == 0:
        return HorizonMetrics(
            mae_max=0.0, mae_min=0.0,
            rmse_max=0.0, rmse_min=0.0,
            r2_max=0.0, r2_min=0.0,
            direction_acc=0.0, correlation=0.0,
        )

    # MAE
    mae_max = np.mean(np.abs(y_true_max - y_pred_max))
    mae_min = np.mean(np.abs(y_true_min - y_pred_min))

    # RMSE
    rmse_max = np.sqrt(np.mean((y_true_max - y_pred_max) ** 2))
    rmse_min = np.sqrt(np.mean((y_true_min - y_pred_min) ** 2))

    # R²
    if SKLEARN_AVAILABLE and n_samples > 1:
        r2_max = r2_score(y_true_max, y_pred_max)
        r2_min = r2_score(y_true_min, y_pred_min)
    else:
        ss_res_max = np.sum((y_true_max - y_pred_max) ** 2)
        ss_tot_max = np.sum((y_true_max - np.mean(y_true_max)) ** 2)
        r2_max = 1 - (ss_res_max / ss_tot_max) if ss_tot_max > 0 else 0.0

        ss_res_min = np.sum((y_true_min - y_pred_min) ** 2)
        ss_tot_min = np.sum((y_true_min - np.mean(y_true_min)) ** 2)
        r2_min = 1 - (ss_res_min / ss_tot_min) if ss_tot_min > 0 else 0.0

    # Direction accuracy (for max_return: is predicted > 0 when actual > 0?)
    # Net return direction: is the risk-reward ratio in the same direction?
    net_true = y_true_max + y_true_min  # Positive = bullish bias
    net_pred = y_pred_max + y_pred_min
    direction_correct = (net_true * net_pred) > 0
    direction_acc = np.mean(direction_correct)

    # Pearson correlation (using max_return as primary signal)
    if np.std(y_true_max) > 0 and np.std(y_pred_max) > 0:
        correlation = np.corrcoef(y_true_max, y_pred_max)[0, 1]
    else:
        correlation = 0.0

    return HorizonMetrics(
        mae_max=float(mae_max),
        mae_min=float(mae_min),
        rmse_max=float(rmse_max),
        rmse_min=float(rmse_min),
        r2_max=float(r2_max),
        r2_min=float(r2_min),
        direction_acc=float(direction_acc),
        correlation=float(correlation) if not np.isnan(correlation) else 0.0,
    )


def compute_multi_horizon_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    loss_dict: Optional[Dict[str, float]] = None,
) -> MultiHorizonRegressionMetrics:
    """
    Compute all multi-horizon regression metrics.

    Args:
        predictions: Predicted returns of shape (N, 6)
                    [max_1h, max_4h, max_24h, min_1h, min_4h, min_24h]
        targets: Ground truth returns of shape (N, 6)
        loss_dict: Optional dictionary with loss values per horizon

    Returns:
        MultiHorizonRegressionMetrics object
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Extract components
    pred_max_1h, pred_max_4h, pred_max_24h = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    pred_min_1h, pred_min_4h, pred_min_24h = predictions[:, 3], predictions[:, 4], predictions[:, 5]

    true_max_1h, true_max_4h, true_max_24h = targets[:, 0], targets[:, 1], targets[:, 2]
    true_min_1h, true_min_4h, true_min_24h = targets[:, 3], targets[:, 4], targets[:, 5]

    # Compute metrics for each horizon
    horizon_1h = compute_horizon_metrics(true_max_1h, pred_max_1h, true_min_1h, pred_min_1h)
    horizon_4h = compute_horizon_metrics(true_max_4h, pred_max_4h, true_min_4h, pred_min_4h)
    horizon_24h = compute_horizon_metrics(true_max_24h, pred_max_24h, true_min_24h, pred_min_24h)

    # Get losses from dict or default to 0
    if loss_dict is None:
        loss_dict = {}

    return MultiHorizonRegressionMetrics(
        horizon_1h=horizon_1h,
        horizon_4h=horizon_4h,
        horizon_24h=horizon_24h,
        total_loss=loss_dict.get('total_loss', 0.0),
        loss_1h=loss_dict.get('loss_1h', 0.0),
        loss_4h=loss_dict.get('loss_4h', 0.0),
        loss_24h=loss_dict.get('loss_24h', 0.0),
    )


class MultiHorizonMetricTracker:
    """Track multi-horizon regression metrics over training epochs."""

    def __init__(self):
        self.history: Dict[str, list] = {}
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: int = 0

    def update(self, metrics: MultiHorizonRegressionMetrics, epoch: int):
        """Update tracker with new metrics."""
        metrics_dict = metrics.to_dict()

        for key, value in metrics_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        # Track best direction accuracy (4h as primary horizon)
        current_dir_acc = metrics.horizon_4h.direction_acc
        if current_dir_acc > self.best_metrics.get("best_dir_acc_4h", 0):
            self.best_metrics["best_dir_acc_4h"] = current_dir_acc
            self.best_metrics["best_r2_4h"] = metrics.horizon_4h.r2_max
            self.best_metrics["best_mae_4h"] = metrics.horizon_4h.mae_max
            self.best_epoch = epoch

    def get_best(self) -> Tuple[Dict[str, float], int]:
        """Get best metrics and epoch."""
        return self.best_metrics, self.best_epoch

    def get_history(self, key: str) -> list:
        """Get history for a specific metric."""
        return self.history.get(key, [])


def format_multi_horizon_metrics(metrics: MultiHorizonRegressionMetrics, prefix: str = "") -> str:
    """
    Format multi-horizon regression metrics for logging.

    Args:
        metrics: MultiHorizonRegressionMetrics object
        prefix: Optional prefix for the output string

    Returns:
        Formatted string
    """
    h1 = metrics.horizon_1h
    h4 = metrics.horizon_4h
    h24 = metrics.horizon_24h

    lines = [
        f"{prefix}Loss: {metrics.total_loss:.4f} (1h: {metrics.loss_1h:.4f}, 4h: {metrics.loss_4h:.4f}, 24h: {metrics.loss_24h:.4f})",
        f"{prefix}1h:  MAE={h1.mae_max:.4f}, R²={h1.r2_max:.4f}, DirAcc={h1.direction_acc:.2%}, Corr={h1.correlation:.4f}",
        f"{prefix}4h:  MAE={h4.mae_max:.4f}, R²={h4.r2_max:.4f}, DirAcc={h4.direction_acc:.2%}, Corr={h4.correlation:.4f}",
        f"{prefix}24h: MAE={h24.mae_max:.4f}, R²={h24.r2_max:.4f}, DirAcc={h24.direction_acc:.2%}, Corr={h24.correlation:.4f}",
    ]

    return "\n".join(lines)


def format_metrics(metrics: MultiTaskMetrics, prefix: str = "") -> str:
    """
    Format metrics for logging.

    Args:
        metrics: MultiTaskMetrics object
        prefix: Optional prefix for the output string

    Returns:
        Formatted string
    """
    cls = metrics.classification
    reg = metrics.regression

    lines = [
        f"{prefix}Loss: {metrics.total_loss:.4f} (trigger: {metrics.trigger_loss:.4f}, max_pct: {metrics.max_pct_loss:.4f})",
        f"{prefix}Classification: Acc={cls.accuracy:.4f}, Prec={cls.precision:.4f}, Rec={cls.recall:.4f}, F1={cls.f1:.4f}, AUC={cls.auc_roc:.4f}",
        f"{prefix}Confusion: TP={cls.true_positives}, FP={cls.false_positives}, TN={cls.true_negatives}, FN={cls.false_negatives}",
    ]

    if reg.n_samples > 0:
        lines.append(
            f"{prefix}Regression (n={reg.n_samples}): MSE={reg.mse:.6f}, MAE={reg.mae:.6f}, R²={reg.r2:.4f}"
        )

    return "\n".join(lines)
