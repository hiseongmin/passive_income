"""
Visualize Trigger Predictions on Test Set
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preparation.dataset import TriggerDataset, create_data_splits
from features.feature_combiner import load_precomputed_features
from models.trigger_model import TriggerPredictionModel, TriggerModelConfig, create_model
from torch.utils.data import DataLoader


def load_model_and_predict(checkpoint_path, test_loader, device='cuda'):
    """Load model and get predictions on test set."""

    # Create model
    config = TriggerModelConfig(
        seq_len_5m=72,
        seq_len_1h=6,
        tda_features=9,
        micro_features=12,
        lstm_hidden_5m=128,
        lstm_hidden_1h=64,
        lstm_layers=2,
        nbeats_blocks=3,
        nbeats_hidden=256
    )
    model = create_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Val F1: {checkpoint.get('best_val_f1', 'unknown'):.4f}")

    # Collect predictions
    all_trigger_probs = []
    all_trigger_targets = []
    all_direction_preds = []
    all_direction_targets = []
    all_imminence_preds = []
    all_imminence_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x_5m = batch['x_5m'].to(device)
            x_1h = batch['x_1h'].to(device)
            tda = batch['tda'].to(device)
            micro = batch['micro'].to(device)

            trigger_prob, imminence, direction_logits = model(x_5m, x_1h, tda, micro)

            all_trigger_probs.append(trigger_prob.cpu().numpy())
            all_trigger_targets.append(batch['trigger'].numpy())
            all_direction_preds.append(torch.argmax(direction_logits, dim=1).cpu().numpy())
            all_direction_targets.append(batch['direction'].numpy())
            all_imminence_preds.append(imminence.cpu().numpy())
            all_imminence_targets.append(batch['imminence'].numpy())

    return {
        'trigger_probs': np.concatenate(all_trigger_probs),
        'trigger_targets': np.concatenate(all_trigger_targets),
        'direction_preds': np.concatenate(all_direction_preds),
        'direction_targets': np.concatenate(all_direction_targets),
        'imminence_preds': np.concatenate(all_imminence_preds),
        'imminence_targets': np.concatenate(all_imminence_targets)
    }


def visualize_predictions(df_5m, predictions, test_indices, threshold=0.5, save_path='trigger_predictions.png'):
    """Visualize trigger predictions on price chart."""

    # Get test data slice
    test_df = df_5m.iloc[test_indices].copy()
    test_df = test_df.reset_index(drop=True)

    # Align predictions with test data (account for sequence length offset)
    seq_len = 72  # seq_len_5m
    n_predictions = len(predictions['trigger_probs'])

    # Create prediction arrays aligned with test_df
    pred_start_idx = seq_len - 1
    pred_end_idx = pred_start_idx + n_predictions

    # Ensure we don't exceed bounds
    if pred_end_idx > len(test_df):
        pred_end_idx = len(test_df)
        n_predictions = pred_end_idx - pred_start_idx

    # Slice predictions to match
    trigger_probs = predictions['trigger_probs'][:n_predictions]
    trigger_targets = predictions['trigger_targets'][:n_predictions]
    direction_preds = predictions['direction_preds'][:n_predictions]

    # Create aligned dataframe
    plot_df = test_df.iloc[pred_start_idx:pred_end_idx].copy()
    plot_df = plot_df.reset_index(drop=True)
    plot_df['trigger_prob'] = trigger_probs
    plot_df['trigger_target'] = trigger_targets
    plot_df['trigger_pred'] = (trigger_probs >= threshold).astype(int)
    plot_df['direction_pred'] = direction_preds  # 0: down, 1: up

    # Parse datetime
    if 'open_time' in plot_df.columns:
        plot_df['datetime'] = pd.to_datetime(plot_df['open_time'])
    else:
        plot_df['datetime'] = pd.date_range(start='2024-01-01', periods=len(plot_df), freq='5min')

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Subplot 1: Price with predictions
    ax1 = axes[0]
    ax1.plot(plot_df['datetime'], plot_df['close'], 'b-', linewidth=0.8, alpha=0.8, label='Close Price')

    # True triggers (actual)
    true_triggers = plot_df[plot_df['trigger_target'] == 1]
    ax1.scatter(true_triggers['datetime'], true_triggers['close'],
                c='green', s=100, marker='^', label=f'Actual Trigger (n={len(true_triggers)})', zorder=5, alpha=0.8)

    # Predicted triggers
    pred_triggers = plot_df[plot_df['trigger_pred'] == 1]

    # Color by direction prediction
    pred_up = pred_triggers[pred_triggers['direction_pred'] == 1]
    pred_down = pred_triggers[pred_triggers['direction_pred'] == 0]

    ax1.scatter(pred_up['datetime'], pred_up['close'],
                c='red', s=80, marker='o', label=f'Pred Trigger UP (n={len(pred_up)})', zorder=4, alpha=0.7)
    ax1.scatter(pred_down['datetime'], pred_down['close'],
                c='purple', s=80, marker='o', label=f'Pred Trigger DOWN (n={len(pred_down)})', zorder=4, alpha=0.7)

    # True positives (correctly predicted)
    true_positives = plot_df[(plot_df['trigger_target'] == 1) & (plot_df['trigger_pred'] == 1)]
    ax1.scatter(true_positives['datetime'], true_positives['close'],
                c='yellow', s=150, marker='*', label=f'True Positive (n={len(true_positives)})',
                zorder=6, edgecolors='black', linewidths=1)

    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.set_title(f'BTCUSDT Trigger Predictions on Test Set (Threshold={threshold})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    # Subplot 2: Trigger probability
    ax2 = axes[1]
    ax2.fill_between(plot_df['datetime'], 0, plot_df['trigger_prob'], alpha=0.5, color='orange', label='Trigger Probability')
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    # Subplot 3: Actual triggers indicator
    ax3 = axes[2]
    ax3.fill_between(plot_df['datetime'], 0, plot_df['trigger_target'], alpha=0.7, color='green', label='Actual Trigger')
    ax3.set_ylabel('Actual', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {save_path}")

    # Print statistics
    print("\n" + "="*50)
    print("PREDICTION STATISTICS")
    print("="*50)
    print(f"Total samples: {len(plot_df)}")
    print(f"Actual triggers: {len(true_triggers)} ({100*len(true_triggers)/len(plot_df):.2f}%)")
    print(f"Predicted triggers: {len(pred_triggers)} ({100*len(pred_triggers)/len(plot_df):.2f}%)")
    print(f"  - Predicted UP: {len(pred_up)}")
    print(f"  - Predicted DOWN: {len(pred_down)}")
    print(f"True Positives: {len(true_positives)}")
    print(f"False Positives: {len(pred_triggers) - len(true_positives)}")
    print(f"False Negatives: {len(true_triggers) - len(true_positives)}")

    if len(pred_triggers) > 0:
        precision = len(true_positives) / len(pred_triggers)
        print(f"Precision: {precision:.4f}")
    if len(true_triggers) > 0:
        recall = len(true_positives) / len(true_triggers)
        print(f"Recall: {recall:.4f}")

    return plot_df


def create_zoomed_views(plot_df, save_dir='.', n_views=3):
    """Create zoomed views of interesting regions."""

    # Find regions with predictions
    pred_indices = plot_df[plot_df['trigger_pred'] == 1].index.tolist()

    if len(pred_indices) == 0:
        print("No predictions to zoom into")
        return

    # Select evenly spaced regions
    step = len(pred_indices) // n_views if len(pred_indices) >= n_views else 1
    selected_indices = pred_indices[::step][:n_views]

    for i, center_idx in enumerate(selected_indices):
        # Create window around this prediction
        window_size = 200  # 200 candles = ~16 hours
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(plot_df), center_idx + window_size // 2)

        window_df = plot_df.iloc[start_idx:end_idx].copy()

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot candlestick-like representation
        ax.plot(window_df['datetime'], window_df['close'], 'b-', linewidth=1.5, label='Close Price')

        # Highlight high/low range
        ax.fill_between(window_df['datetime'], window_df['low'], window_df['high'],
                       alpha=0.2, color='blue')

        # True triggers
        true_triggers = window_df[window_df['trigger_target'] == 1]
        ax.scatter(true_triggers['datetime'], true_triggers['close'],
                  c='green', s=200, marker='^', label=f'Actual Trigger', zorder=5)

        # Predicted triggers
        pred_triggers = window_df[window_df['trigger_pred'] == 1]
        pred_up = pred_triggers[pred_triggers['direction_pred'] == 1]
        pred_down = pred_triggers[pred_triggers['direction_pred'] == 0]

        ax.scatter(pred_up['datetime'], pred_up['close'],
                  c='red', s=150, marker='o', label='Pred UP', zorder=4)
        ax.scatter(pred_down['datetime'], pred_down['close'],
                  c='purple', s=150, marker='o', label='Pred DOWN', zorder=4)

        # True positives
        true_positives = window_df[(window_df['trigger_target'] == 1) & (window_df['trigger_pred'] == 1)]
        ax.scatter(true_positives['datetime'], true_positives['close'],
                  c='yellow', s=250, marker='*', label='True Positive', zorder=6,
                  edgecolors='black', linewidths=1.5)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price (USDT)', fontsize=12)
        ax.set_title(f'Zoomed View {i+1}: Trigger Predictions', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'trigger_zoom_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Zoomed view {i+1} saved to: {save_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    df_5m = pd.read_csv('/notebooks/sa/data/BTCUSDT_perp_5m_labeled.csv')
    df_1h = pd.read_csv('/notebooks/sa/data/BTCUSDT_perp_1h.csv')
    print(f"5m data shape: {df_5m.shape}")

    # Load features
    tda_features, micro_features = load_precomputed_features('/notebooks/sa/data')

    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_data_splits(
        df_5m, df_1h,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seq_len_5m=72,
        seq_len_1h=6,
        tda_features=tda_features,
        micro_features=micro_features
    )

    print(f"Test samples: {len(test_ds)}")

    # Create test loader
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Get test indices from dataset
    test_indices = test_ds.indices if hasattr(test_ds, 'indices') else list(range(len(df_5m) - len(train_ds) - len(val_ds), len(df_5m)))

    # Load model and predict
    print("\nLoading model and making predictions...")
    predictions = load_model_and_predict(
        '/notebooks/sa/checkpoints/best_model.pt',
        test_loader,
        device=device
    )

    # Visualize
    print("\nCreating visualization...")
    plot_df = visualize_predictions(
        df_5m,
        predictions,
        test_indices,
        threshold=0.5,
        save_path='/notebooks/sa/trigger_predictions.png'
    )

    # Create zoomed views
    print("\nCreating zoomed views...")
    create_zoomed_views(plot_df, save_dir='/notebooks/sa', n_views=3)

    print("\nDone!")


if __name__ == '__main__':
    main()
