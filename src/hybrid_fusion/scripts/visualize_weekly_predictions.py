#!/usr/bin/env python3
"""
Visualize weekly predictions for Hybrid Fusion model.

Shows predicted triggers overlaid on actual triggers grouped by week,
and provides summary statistics for different thresholds (0.6, 0.7, 0.8).

Usage:
    python -m src.hybrid_fusion.scripts.visualize_weekly_predictions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid_fusion.config import load_config
from src.hybrid_fusion.model import CompleteHybridFusionModel
from src.hybrid_fusion.scripts.train import create_model_from_config, create_tda_config_from_hybrid
from src.tda_model.data import create_data_loaders


THRESHOLDS = [0.6, 0.7, 0.8]
WINDOW_SIZE = 672  # Offset for valid predictions


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


def load_test_dataframe(project_root: Path) -> pd.DataFrame:
    """Load raw test CSV for datetime information."""
    test_path = project_root / 'data_flagged' / 'BTCUSDT_spot_last_90d_15m_flagged.csv'
    df = pd.read_csv(test_path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    return df


@torch.no_grad()
def get_predictions(model, test_loader):
    """Get predictions on test set."""
    all_probs = []
    all_targets = []

    for batch in test_loader:
        ohlcv_seq, tda_features, complexity, trigger, max_pct = batch

        ohlcv_seq = ohlcv_seq.cuda()
        tda_features = tda_features.cuda()
        complexity = complexity.cuda()

        with torch.amp.autocast('cuda'):
            trigger_prob, max_pct_pred = model(ohlcv_seq, tda_features, complexity)

        all_probs.append(trigger_prob.cpu().numpy())
        all_targets.append(trigger.numpy())

    probs = np.concatenate(all_probs).flatten()
    targets = np.concatenate(all_targets).flatten()

    return probs, targets


def compute_metrics_at_threshold(targets, probs, threshold):
    """Compute TP, FP, FN, Precision, Recall, F1 at a threshold."""
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'threshold': threshold,
        'predictions': preds.sum(),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def plot_weekly_predictions(df_test, probs, targets, output_dir: Path):
    """Create weekly visualization of predictions vs actual triggers."""
    # Get valid samples (after window_size offset)
    df_valid = df_test.iloc[WINDOW_SIZE:WINDOW_SIZE + len(probs)].copy()
    df_valid = df_valid.reset_index(drop=True)
    df_valid['prob'] = probs
    df_valid['actual_trigger'] = targets

    # Add week column
    df_valid['week'] = df_valid['open_time'].dt.isocalendar().week
    df_valid['year_week'] = df_valid['open_time'].dt.strftime('%Y-W%W')

    # Get unique weeks
    weeks = df_valid['year_week'].unique()
    n_weeks = len(weeks)

    print(f"\nTotal valid samples: {len(df_valid)}")
    print(f"Date range: {df_valid['open_time'].min()} to {df_valid['open_time'].max()}")
    print(f"Number of weeks: {n_weeks}")

    # Create figure with subplots for each week
    n_cols = 2
    n_rows = (n_weeks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    # Colors for thresholds
    colors = {0.6: 'green', 0.7: 'orange', 0.8: 'red'}

    for idx, week in enumerate(weeks):
        ax = axes[idx]
        week_data = df_valid[df_valid['year_week'] == week].copy()

        # Plot price
        ax.plot(week_data['open_time'], week_data['close'],
                'b-', linewidth=0.5, alpha=0.7, label='Price')

        # Plot actual triggers
        actual_triggers = week_data[week_data['actual_trigger'] == 1]
        ax.scatter(actual_triggers['open_time'], actual_triggers['close'],
                   marker='^', s=60, c='blue', edgecolors='black',
                   linewidth=0.5, alpha=0.8, label=f'Actual ({len(actual_triggers)})', zorder=5)

        # Plot predicted triggers at each threshold
        for thresh in THRESHOLDS:
            pred_triggers = week_data[week_data['prob'] >= thresh]
            if len(pred_triggers) > 0:
                # Offset y slightly for visibility
                offset = (thresh - 0.7) * 0.02 * (week_data['close'].max() - week_data['close'].min())
                ax.scatter(pred_triggers['open_time'], pred_triggers['close'] + offset,
                           marker='v', s=40, c=colors[thresh], alpha=0.6,
                           label=f'Pred>={thresh} ({len(pred_triggers)})', zorder=4)

        # Format
        ax.set_title(f'{week}', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price (USDT)')

    # Hide empty subplots
    for idx in range(n_weeks, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Weekly Predictions: Predicted vs Actual Triggers',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'weekly_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nWeekly visualization saved to: {output_path}")

    return df_valid


def print_summary_statistics(targets, probs):
    """Print summary statistics for each threshold."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS BY THRESHOLD")
    print("=" * 70)

    # Header
    print(f"\n{'Threshold':>10} {'Preds':>8} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    results = []
    for thresh in THRESHOLDS:
        metrics = compute_metrics_at_threshold(targets, probs, thresh)
        results.append(metrics)

        print(f"{thresh:>10.1f} {metrics['predictions']:>8} {metrics['tp']:>6} "
              f"{metrics['fp']:>6} {metrics['fn']:>6} "
              f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")

    print("-" * 70)

    # Additional context
    total_actual = int(targets.sum())
    total_samples = len(targets)
    print(f"\nTotal test samples: {total_samples}")
    print(f"Total actual triggers: {total_actual} ({100*total_actual/total_samples:.2f}%)")

    # Probability distribution
    print(f"\nProbability distribution:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (probs >= thresh).sum()
        print(f"  >= {thresh}: {count} samples ({100*count/total_samples:.2f}%)")

    return results


def save_weekly_metrics(df_valid, output_dir: Path):
    """Save weekly metrics to CSV."""
    weekly_metrics = []

    for week in df_valid['year_week'].unique():
        week_data = df_valid[df_valid['year_week'] == week]
        targets = week_data['actual_trigger'].values
        probs = week_data['prob'].values

        row = {
            'week': week,
            'start_date': week_data['open_time'].min(),
            'end_date': week_data['open_time'].max(),
            'samples': len(week_data),
            'actual_triggers': int(targets.sum()),
        }

        for thresh in THRESHOLDS:
            metrics = compute_metrics_at_threshold(targets, probs, thresh)
            row[f'preds_{thresh}'] = metrics['predictions']
            row[f'tp_{thresh}'] = metrics['tp']
            row[f'fp_{thresh}'] = metrics['fp']
            row[f'precision_{thresh}'] = metrics['precision']
            row[f'recall_{thresh}'] = metrics['recall']

        weekly_metrics.append(row)

    df_metrics = pd.DataFrame(weekly_metrics)
    output_path = output_dir / 'weekly_metrics.csv'
    df_metrics.to_csv(output_path, index=False)
    print(f"\nWeekly metrics saved to: {output_path}")

    # Print weekly breakdown
    print("\n" + "=" * 70)
    print("WEEKLY BREAKDOWN")
    print("=" * 70)
    print(f"\n{'Week':>12} {'Samples':>8} {'Actual':>8} "
          f"{'Pred@0.6':>8} {'Pred@0.7':>8} {'Pred@0.8':>8}")
    print("-" * 60)

    for _, row in df_metrics.iterrows():
        print(f"{row['week']:>12} {row['samples']:>8} {row['actual_triggers']:>8} "
              f"{row['preds_0.6']:>8} {row['preds_0.7']:>8} {row['preds_0.8']:>8}")


def main():
    """Main visualization function."""
    project_root = Path(__file__).parent.parent.parent.parent
    checkpoint_path = project_root / 'checkpoints' / 'hybrid_fusion' / 'best_model.pt'
    config_path = project_root / 'src' / 'hybrid_fusion' / 'default_config.yaml'
    output_dir = project_root / 'checkpoints' / 'hybrid_fusion'

    print("Loading model and data...")
    model, test_loader, config = load_model_and_data(checkpoint_path, config_path)

    print("Loading test dataframe for datetime info...")
    df_test = load_test_dataframe(project_root)

    print("Getting predictions...")
    probs, targets = get_predictions(model, test_loader)

    print(f"\nTest samples: {len(targets)}")
    print(f"Trigger rate: {targets.mean():.2%}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Print summary statistics
    print_summary_statistics(targets, probs)

    # Create weekly visualization
    df_valid = plot_weekly_predictions(df_test, probs, targets, output_dir)

    # Save weekly metrics
    save_weekly_metrics(df_valid, output_dir)

    plt.show()


if __name__ == "__main__":
    main()
