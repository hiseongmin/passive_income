#!/usr/bin/env python3
"""
Visualize test results for Hybrid Fusion model.

Usage:
    python -m src.hybrid_fusion.scripts.visualize_results
"""

import sys
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid_fusion.config import load_config
from src.hybrid_fusion.model import CompleteHybridFusionModel
from src.hybrid_fusion.scripts.train import create_model_from_config, create_tda_config_from_hybrid
from src.tda_model.data import create_data_loaders


def load_model_and_data(checkpoint_path: Path, config_path: Path):
    """Load trained model and test data."""
    # Load config
    config = load_config(str(config_path))

    # Create model
    model = create_model_from_config(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()

    # Load data
    project_root = Path(__file__).parent.parent.parent.parent
    tda_config = create_tda_config_from_hybrid(config)
    _, _, test_loader = create_data_loaders(tda_config, project_root)

    return model, test_loader, config


@torch.no_grad()
def get_predictions(model, test_loader):
    """Get predictions on test set."""
    all_probs = []
    all_targets = []
    all_max_pct_pred = []
    all_max_pct_true = []

    for batch in test_loader:
        ohlcv_seq, tda_features, complexity, trigger, max_pct = batch

        ohlcv_seq = ohlcv_seq.cuda()
        tda_features = tda_features.cuda()
        complexity = complexity.cuda()

        with torch.amp.autocast('cuda'):
            trigger_prob, max_pct_pred = model(ohlcv_seq, tda_features, complexity)

        all_probs.append(trigger_prob.cpu().numpy())
        all_targets.append(trigger.numpy())
        all_max_pct_pred.append(max_pct_pred.cpu().numpy())
        all_max_pct_true.append(max_pct.numpy())

    probs = np.concatenate(all_probs).flatten()
    targets = np.concatenate(all_targets).flatten()
    max_pct_pred = np.concatenate(all_max_pct_pred).flatten()
    max_pct_true = np.concatenate(all_max_pct_true).flatten()

    return probs, targets, max_pct_pred, max_pct_true


def plot_roc_curve(targets, probs, ax):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return roc_auc


def plot_precision_recall_curve(targets, probs, ax):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    pr_auc = auc(recall, precision)

    # Baseline (random classifier)
    baseline = targets.sum() / len(targets)

    ax.plot(recall, precision, 'g-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return pr_auc


def plot_confusion_matrix(targets, preds, ax, threshold=0.5):
    """Plot confusion matrix."""
    cm = confusion_matrix(targets, preds)

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    classes = ['No Trigger', 'Trigger']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix (threshold={threshold})', fontsize=14, fontweight='bold')

    # Text annotations
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                   ha='center', va='center', color=text_color, fontsize=11)


def plot_probability_distribution(targets, probs, ax):
    """Plot probability distribution for positive and negative samples."""
    pos_probs = probs[targets == 1]
    neg_probs = probs[targets == 0]

    bins = np.linspace(0, 1, 50)

    ax.hist(neg_probs, bins=bins, alpha=0.6, label=f'No Trigger (n={len(neg_probs)})',
            color='blue', density=True)
    ax.hist(pos_probs, bins=bins, alpha=0.6, label=f'Trigger (n={len(pos_probs)})',
            color='red', density=True)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    ax.axvline(x=0.3, color='green', linestyle='--', linewidth=1.5, label='Threshold=0.3')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])


def plot_threshold_analysis(targets, probs, ax):
    """Plot metrics vs threshold."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    ax.plot(thresholds, precisions, 'b-', linewidth=2, marker='o', markersize=4, label='Precision')
    ax.plot(thresholds, recalls, 'r-', linewidth=2, marker='s', markersize=4, label='Recall')
    ax.plot(thresholds, f1_scores, 'g-', linewidth=2, marker='^', markersize=4, label='F1 Score')

    ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 0.85])
    ax.set_ylim([0, 1])


def plot_max_pct_regression(max_pct_true, max_pct_pred, targets, ax):
    """Plot regression results for max_pct (only for trigger samples)."""
    mask = targets == 1

    if mask.sum() == 0:
        ax.text(0.5, 0.5, 'No trigger samples', ha='center', va='center', fontsize=14)
        return

    true_vals = max_pct_true[mask]
    pred_vals = max_pct_pred[mask]

    ax.scatter(true_vals, pred_vals, alpha=0.5, s=20, c='blue')

    # Perfect prediction line
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Compute R²
    ss_res = ((true_vals - pred_vals) ** 2).sum()
    ss_tot = ((true_vals - true_vals.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    ax.set_xlabel('True Max %', fontsize=12)
    ax.set_ylabel('Predicted Max %', fontsize=12)
    ax.set_title(f'Max % Regression (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    """Main visualization function."""
    project_root = Path(__file__).parent.parent.parent.parent
    checkpoint_path = project_root / 'checkpoints' / 'hybrid_fusion' / 'best_model.pt'
    config_path = project_root / 'src' / 'hybrid_fusion' / 'default_config.yaml'
    output_dir = project_root / 'checkpoints' / 'hybrid_fusion'

    print("Loading model and data...")
    model, test_loader, config = load_model_and_data(checkpoint_path, config_path)

    print("Getting predictions...")
    probs, targets, max_pct_pred, max_pct_true = get_predictions(model, test_loader)

    print(f"Test samples: {len(targets)}")
    print(f"Trigger rate: {targets.mean():.2%}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # ROC Curve
    ax1 = fig.add_subplot(2, 3, 1)
    roc_auc = plot_roc_curve(targets, probs, ax1)

    # Precision-Recall Curve
    ax2 = fig.add_subplot(2, 3, 2)
    pr_auc = plot_precision_recall_curve(targets, probs, ax2)

    # Probability Distribution
    ax3 = fig.add_subplot(2, 3, 3)
    plot_probability_distribution(targets, probs, ax3)

    # Confusion Matrix (threshold=0.5)
    ax4 = fig.add_subplot(2, 3, 4)
    preds_05 = (probs >= 0.5).astype(int)
    plot_confusion_matrix(targets, preds_05, ax4, threshold=0.5)

    # Confusion Matrix (threshold=0.3)
    ax5 = fig.add_subplot(2, 3, 5)
    preds_03 = (probs >= 0.3).astype(int)
    plot_confusion_matrix(targets, preds_03, ax5, threshold=0.3)

    # Threshold Analysis
    ax6 = fig.add_subplot(2, 3, 6)
    plot_threshold_analysis(targets, probs, ax6)

    plt.suptitle('Hybrid Fusion Model - Test Set Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'test_results_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("TEST SET SUMMARY")
    print("="*60)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nThreshold=0.5:")
    print(classification_report(targets, preds_05, target_names=['No Trigger', 'Trigger']))

    print("\nThreshold=0.3:")
    print(classification_report(targets, preds_03, target_names=['No Trigger', 'Trigger']))

    plt.show()


if __name__ == "__main__":
    main()
