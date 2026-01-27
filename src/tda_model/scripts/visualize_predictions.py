#!/usr/bin/env python3
"""
Visualize model predictions on test set over Bitcoin price.

Generates weekly charts with 4 subplots:
1. Price chart with TP/FP/FN markers
2. Trigger probability timeline
3. Max_Pct predicted vs actual
4. Weekly confusion summary

Usage:
    python -m scripts.visualize_predictions
    python -m scripts.visualize_predictions --threshold 0.6
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import create_data_loaders
from tda_model.training import create_model_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Color scheme
COLORS = {
    "price": "#1a1a1a",
    "tp": "#2ECC71",  # Green - True Positive
    "fp": "#E74C3C",  # Red - False Positive
    "fn": "#F39C12",  # Orange - False Negative
    "tn": "#95A5A6",  # Gray - True Negative (not shown)
    "prob": "#3498DB",  # Blue - Probability
    "threshold": "#E74C3C",  # Red dashed
    "pred_max_pct": "#3498DB",  # Blue
    "actual_max_pct": "#2ECC71",  # Green
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize TDA model predictions on test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for charts",
    )
    return parser.parse_args()


def load_model_and_get_predictions(
    config: Config,
    project_root: Path,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load trained model and generate predictions on test set.

    Returns:
        trigger_prob: Model trigger probabilities
        max_pct_pred: Model max_pct predictions
        trigger_true: Actual trigger labels
        max_pct_true: Actual max_pct values
        test_df: Test DataFrame with timestamps and prices
    """
    logger.info("Loading model and data...")

    # Create data loaders (only need test loader)
    _, _, test_loader = create_data_loaders(
        config=config,
        project_root=project_root,
    )

    # Load model
    model = create_model_from_config(config)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = project_root / "src" / "tda_model" / config.logging.save_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Best val F1: {checkpoint.get('best_val_f1', 'N/A')}")

    # Run inference
    all_trigger_prob = []
    all_max_pct_pred = []
    all_trigger_true = []
    all_max_pct_true = []

    with torch.no_grad():
        for batch in test_loader:
            ohlcv_seq, tda_features, complexity, trigger, max_pct = batch

            ohlcv_seq = ohlcv_seq.to(device)
            tda_features = tda_features.to(device)
            complexity = complexity.to(device)

            trigger_logits, max_pct_out = model(ohlcv_seq, tda_features, complexity)
            trigger_prob = torch.sigmoid(trigger_logits)

            all_trigger_prob.append(trigger_prob.cpu().numpy())
            all_max_pct_pred.append(max_pct_out.cpu().numpy())
            all_trigger_true.append(trigger.numpy())
            all_max_pct_true.append(max_pct.numpy())

    trigger_prob = np.concatenate(all_trigger_prob).flatten()
    max_pct_pred = np.concatenate(all_max_pct_pred).flatten()
    trigger_true = np.concatenate(all_trigger_true).flatten()
    max_pct_true = np.concatenate(all_max_pct_true).flatten()

    logger.info(f"Generated {len(trigger_prob)} predictions")

    # Load raw test data for timestamps and prices
    test_file = project_root / "data" / config.data.test_file
    test_df = pd.read_csv(test_file)
    test_df["open_time"] = pd.to_datetime(test_df["open_time"])

    # The dataset uses valid indices starting from window_size
    window_size = config.tda.window_size
    valid_start = window_size
    test_df = test_df.iloc[valid_start:valid_start + len(trigger_prob)].reset_index(drop=True)

    return trigger_prob, max_pct_pred, trigger_true, max_pct_true, test_df


def compute_confusion_categories(
    trigger_prob: np.ndarray,
    trigger_true: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confusion matrix categories.

    Returns:
        tp_mask, fp_mask, fn_mask, tn_mask: Boolean masks for each category
    """
    trigger_pred = (trigger_prob >= threshold).astype(int)

    tp_mask = (trigger_pred == 1) & (trigger_true == 1)
    fp_mask = (trigger_pred == 1) & (trigger_true == 0)
    fn_mask = (trigger_pred == 0) & (trigger_true == 1)
    tn_mask = (trigger_pred == 0) & (trigger_true == 0)

    return tp_mask, fp_mask, fn_mask, tn_mask


def create_weekly_chart(
    week_num: int,
    week_start: datetime,
    week_end: datetime,
    timestamps: np.ndarray,
    prices: np.ndarray,
    trigger_prob: np.ndarray,
    max_pct_pred: np.ndarray,
    trigger_true: np.ndarray,
    max_pct_true: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Create a 4-subplot chart for one week of data."""

    # Compute confusion categories
    tp_mask, fp_mask, fn_mask, tn_mask = compute_confusion_categories(
        trigger_prob, trigger_true, threshold
    )

    # Calculate metrics
    n_tp = tp_mask.sum()
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tn = tn_mask.sum()

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1]})
    fig.suptitle(
        f"Week {week_num}: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
        fontsize=14,
        fontweight="bold",
    )

    # Subplot 1: Price chart with predictions
    ax1 = axes[0]
    ax1.plot(timestamps, prices, color=COLORS["price"], linewidth=1, label="BTC Price")

    # Plot markers
    if n_tp > 0:
        ax1.scatter(
            timestamps[tp_mask], prices[tp_mask],
            marker="^", s=100, c=COLORS["tp"], label=f"TP ({n_tp})", zorder=5
        )
    if n_fp > 0:
        ax1.scatter(
            timestamps[fp_mask], prices[fp_mask],
            marker="^", s=80, c=COLORS["fp"], label=f"FP ({n_fp})", zorder=5
        )
    if n_fn > 0:
        ax1.scatter(
            timestamps[fn_mask], prices[fn_mask],
            marker="v", s=80, c=COLORS["fn"], label=f"FN ({n_fn})", zorder=5
        )

    ax1.set_ylabel("Price (USD)")
    ax1.set_title("Bitcoin Price with Predictions")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax1.tick_params(axis="x", rotation=45)

    # Subplot 2: Probability timeline
    ax2 = axes[1]
    ax2.plot(timestamps, trigger_prob, color=COLORS["prob"], linewidth=1, label="Trigger Prob")
    ax2.axhline(y=threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")

    # Mark actual trigger events on x-axis
    trigger_events = timestamps[trigger_true == 1]
    if len(trigger_events) > 0:
        ax2.scatter(
            trigger_events, np.zeros(len(trigger_events)),
            marker="|", s=50, c=COLORS["tp"], label=f"Actual Triggers ({len(trigger_events)})"
        )

    ax2.set_ylabel("Probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Model Trigger Probability")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax2.tick_params(axis="x", rotation=45)

    # Subplot 3: Max_Pct comparison (only where Trigger=True)
    ax3 = axes[2]
    trigger_mask = trigger_true == 1
    if trigger_mask.sum() > 0:
        trigger_ts = timestamps[trigger_mask]
        ax3.scatter(
            trigger_ts, max_pct_true[trigger_mask],
            c=COLORS["actual_max_pct"], s=50, alpha=0.7, label="Actual Max_Pct"
        )
        ax3.scatter(
            trigger_ts, max_pct_pred[trigger_mask],
            c=COLORS["pred_max_pct"], s=50, alpha=0.7, marker="x", label="Predicted Max_Pct"
        )
        ax3.legend(loc="upper left", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No trigger events this week", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")

    ax3.set_ylabel("Max_Pct")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Max_Pct: Predicted vs Actual (Trigger=True only)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax3.tick_params(axis="x", rotation=45)

    # Subplot 4: Confusion summary bar chart
    ax4 = axes[3]
    categories = ["TP", "FP", "FN", "TN"]
    counts = [n_tp, n_fp, n_fn, n_tn]
    colors_bar = [COLORS["tp"], COLORS["fp"], COLORS["fn"], COLORS["tn"]]

    bars = ax4.bar(categories, counts, color=colors_bar, edgecolor="black")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax4.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(count), ha="center", va="bottom", fontsize=10
        )

    # Add metrics text
    metrics_text = f"Precision: {precision:.1%}  |  Recall: {recall:.1%}  |  F1: {f1:.3f}"
    ax4.text(
        0.5, 0.95, metrics_text, transform=ax4.transAxes,
        ha="center", va="top", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    ax4.set_ylabel("Count")
    ax4.set_title("Weekly Confusion Matrix Summary")
    ax4.grid(True, alpha=0.3, axis="y")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path.name} (TP={n_tp}, FP={n_fp}, FN={n_fn}, TN={n_tn})")


def create_summary_chart(
    timestamps: np.ndarray,
    prices: np.ndarray,
    trigger_prob: np.ndarray,
    trigger_true: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Create a condensed summary chart for the full test set."""

    # Compute confusion categories
    tp_mask, fp_mask, fn_mask, tn_mask = compute_confusion_categories(
        trigger_prob, trigger_true, threshold
    )

    n_tp = tp_mask.sum()
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tn = tn_mask.sum()

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Full Test Set Summary: {pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(timestamps[-1]).strftime('%Y-%m-%d')}",
        fontsize=14,
        fontweight="bold",
    )

    # Subplot 1: Price with all predictions
    ax1 = axes[0]
    ax1.plot(timestamps, prices, color=COLORS["price"], linewidth=0.8, alpha=0.8, label="BTC Price")

    # Subsample markers for visibility if too many
    max_markers = 200

    def subsample_mask(mask, max_n):
        indices = np.where(mask)[0]
        if len(indices) <= max_n:
            return mask
        selected = np.random.choice(indices, max_n, replace=False)
        new_mask = np.zeros_like(mask)
        new_mask[selected] = True
        return new_mask

    if n_tp > 0:
        plot_mask = subsample_mask(tp_mask, max_markers)
        ax1.scatter(
            timestamps[plot_mask], prices[plot_mask],
            marker="^", s=60, c=COLORS["tp"], label=f"TP ({n_tp})", zorder=5, alpha=0.8
        )
    if n_fp > 0:
        plot_mask = subsample_mask(fp_mask, max_markers)
        ax1.scatter(
            timestamps[plot_mask], prices[plot_mask],
            marker="^", s=40, c=COLORS["fp"], label=f"FP ({n_fp})", zorder=5, alpha=0.7
        )
    if n_fn > 0:
        plot_mask = subsample_mask(fn_mask, max_markers)
        ax1.scatter(
            timestamps[plot_mask], prices[plot_mask],
            marker="v", s=40, c=COLORS["fn"], label=f"FN ({n_fn})", zorder=5, alpha=0.7
        )

    ax1.set_ylabel("Price (USD)")
    ax1.set_title("Bitcoin Price with Model Predictions (Full Test Set)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax1.tick_params(axis="x", rotation=45)

    # Subplot 2: Probability timeline (downsampled for performance)
    ax2 = axes[1]

    # Downsample if too many points
    if len(timestamps) > 2000:
        step = len(timestamps) // 2000
        ds_ts = timestamps[::step]
        ds_prob = trigger_prob[::step]
    else:
        ds_ts = timestamps
        ds_prob = trigger_prob

    ax2.plot(ds_ts, ds_prob, color=COLORS["prob"], linewidth=0.8, alpha=0.8)
    ax2.axhline(y=threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax2.fill_between(ds_ts, ds_prob, threshold, where=ds_prob >= threshold, alpha=0.3, color=COLORS["fp"])

    ax2.set_ylabel("Probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Date")
    ax2.set_title(f"Trigger Probability | Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.3f}")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved summary: {output_path.name}")


def main():
    """Main visualization function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent
    logger.info(f"Project root: {project_root}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "src" / "tda_model" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Check for GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model and get predictions
    trigger_prob, max_pct_pred, trigger_true, max_pct_true, test_df = load_model_and_get_predictions(
        config=config,
        project_root=project_root,
        device=device,
    )

    threshold = args.threshold
    logger.info(f"Using threshold: {threshold}")

    # Extract arrays
    timestamps = test_df["open_time"].values
    prices = test_df["close"].values

    # Convert timestamps to datetime for plotting
    timestamps = pd.to_datetime(timestamps)

    # Split into weekly segments
    start_date = timestamps.min()
    end_date = timestamps.max()

    week_num = 1
    current_start = start_date

    while current_start < end_date:
        current_end = current_start + timedelta(days=7)

        # Get indices for this week
        week_mask = (timestamps >= current_start) & (timestamps < current_end)

        if week_mask.sum() > 0:
            week_timestamps = timestamps[week_mask]
            week_prices = prices[week_mask]
            week_trigger_prob = trigger_prob[week_mask]
            week_max_pct_pred = max_pct_pred[week_mask]
            week_trigger_true = trigger_true[week_mask]
            week_max_pct_true = max_pct_true[week_mask]

            output_path = output_dir / f"week_{week_num:02d}_{current_start.strftime('%Y-%m-%d')}_to_{(current_end - timedelta(days=1)).strftime('%Y-%m-%d')}.png"

            create_weekly_chart(
                week_num=week_num,
                week_start=current_start,
                week_end=current_end,
                timestamps=week_timestamps.values,
                prices=week_prices,
                trigger_prob=week_trigger_prob,
                max_pct_pred=week_max_pct_pred,
                trigger_true=week_trigger_true,
                max_pct_true=week_max_pct_true,
                threshold=threshold,
                output_path=output_path,
            )

            week_num += 1

        current_start = current_end

    # Create summary chart
    summary_path = output_dir / "summary_full_test_set.png"
    create_summary_chart(
        timestamps=timestamps.values,
        prices=prices,
        trigger_prob=trigger_prob,
        trigger_true=trigger_true,
        threshold=threshold,
        output_path=summary_path,
    )

    logger.info(f"\nVisualization complete! Generated {week_num - 1} weekly charts + 1 summary")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
