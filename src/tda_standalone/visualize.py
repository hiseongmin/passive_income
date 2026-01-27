"""
TDA Standalone Model Visualization.
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import pickle

from .config import load_config
from .dataset import create_test_loader
from .model import create_model
from .regime import RegimeLabeler
from .preprocessing import TDAPreprocessor


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss (Trigger + Regime)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_trigger_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_trigger_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Trigger Loss (BCE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['train_regime_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_regime_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Regime Loss (Cross-Entropy)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_regime_analysis(regime_labeler, save_path: Optional[str] = None):
    """Plot regime distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get counts from regime_stats
    counts = [regime_labeler.regime_stats[r]['count'] for r in range(regime_labeler.n_regimes)]
    regime_names = [f"{regime_labeler.regime_stats[r]['name']}\n({regime_labeler.regime_stats[r]['percentage']:.1f}%)"
                    for r in range(regime_labeler.n_regimes)]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    axes[0].pie(counts, labels=regime_names, colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.02]*4)
    axes[0].set_title('Regime Distribution (Training Data)')

    bars = axes[1].bar(regime_names, counts, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Regime Counts')
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{count:,}', ha='center', fontsize=10)
    axes[1].set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


@torch.no_grad()
def plot_predictions(model, test_loader, device='cuda', save_path=None):
    """Plot model predictions on test set."""
    model.eval()
    all_trigger_probs, all_trigger_labels = [], []
    all_regime_probs, all_regime_labels = [], []
    all_confidences = []

    for batch in test_loader:
        structural = batch['structural'].to(device)
        cyclical = batch['cyclical'].to(device)
        landscape = batch['landscape'].to(device)
        outputs = model(structural, cyclical, landscape)
        all_trigger_probs.append(torch.sigmoid(outputs['trigger_logits']).cpu())
        all_trigger_labels.append(batch['trigger'])
        all_regime_probs.append(torch.softmax(outputs['regime_logits'], dim=-1).cpu())
        all_regime_labels.append(batch['regime'])
        all_confidences.append(outputs['confidence'].cpu())

    trigger_probs = torch.cat(all_trigger_probs).numpy().flatten()
    trigger_labels = torch.cat(all_trigger_labels).numpy()
    regime_probs = torch.cat(all_regime_probs).numpy()
    regime_labels = torch.cat(all_regime_labels).numpy()
    confidences = torch.cat(all_confidences).numpy().flatten()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    ax.hist(trigger_probs[trigger_labels == 0], bins=50, alpha=0.7, label='No Trigger', color='blue')
    ax.hist(trigger_probs[trigger_labels == 1], bins=50, alpha=0.7, label='Trigger', color='red')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Trigger Probability')
    ax.set_ylabel('Count')
    ax.set_title('Trigger Probability Distribution')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(confidences, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.7, color='red', linestyle='--', label='High Conf Threshold')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Model Confidence Distribution')
    ax.legend()

    ax = axes[0, 2]
    conf_bins = np.linspace(0, 1, 11)
    trigger_preds = (trigger_probs > 0.5).astype(int)
    bin_accs, bin_centers = [], []
    for i in range(len(conf_bins) - 1):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if mask.sum() > 0:
            bin_accs.append((trigger_preds[mask] == trigger_labels[mask]).mean())
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
    ax.bar(bin_centers, bin_accs, width=0.08, color='purple', alpha=0.7, edgecolor='black')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Trigger Accuracy by Confidence Bin')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    regime_preds = regime_probs.argmax(axis=1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(regime_labels, regime_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['Simple', 'Cycling', 'Chaotic', 'Complex'])
    ax.set_yticklabels(['Simple', 'Cycling', 'Chaotic', 'Complex'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Regime Confusion Matrix')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm_normalized[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    regime_names = ['Simple', 'Cycling', 'Chaotic', 'Complex']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    trigger_rates = [trigger_labels[regime_labels == r].mean() if (regime_labels == r).sum() > 0 else 0 for r in range(4)]
    bars = ax.bar(regime_names, trigger_rates, color=colors, edgecolor='black')
    ax.axhline(y=trigger_labels.mean(), color='black', linestyle='--', label=f'Overall: {trigger_labels.mean():.1%}')
    ax.set_ylabel('Trigger Rate')
    ax.set_title('Actual Trigger Rate by Regime')
    ax.legend()
    for bar, rate in zip(bars, trigger_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{rate:.1%}', ha='center', fontsize=10)

    ax = axes[1, 2]
    high_conf_mask = confidences > 0.7
    low_conf_mask = ~high_conf_mask
    high_acc = (trigger_preds[high_conf_mask] == trigger_labels[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
    low_acc = (trigger_preds[low_conf_mask] == trigger_labels[low_conf_mask]).mean() if low_conf_mask.sum() > 0 else 0
    bars = ax.bar(['High Conf (>0.7)', 'Low Conf (<=0.7)'], [high_acc, low_acc], color=['green', 'orange'], edgecolor='black')
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.set_ylabel('Accuracy')
    ax.set_title('Trigger Accuracy by Confidence Level')
    ax.set_ylim(0, 1)
    for bar, acc, n in zip(bars, [high_acc, low_acc], [high_conf_mask.sum(), low_conf_mask.sum()]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.1%}\n(n={n:,})', ha='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def visualize_results(checkpoint_dir='checkpoints/tda_standalone', config_path=None, save_plots=True):
    """Main visualization function."""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = checkpoint_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    print("Loading training history...")
    with open(checkpoint_dir / 'history.json', 'r') as f:
        history = json.load(f)
    plot_training_history(history, save_path=str(output_dir / 'training_history.png') if save_plots else None)

    print("\nLoading regime labeler...")
    regime_labeler = RegimeLabeler.load(str(checkpoint_dir / 'regime_labeler.pkl'))
    plot_regime_analysis(regime_labeler, save_path=str(output_dir / 'regime_analysis.png') if save_plots else None)

    print("\nLoading model...")
    config = load_config(config_path)
    device = config.device if torch.cuda.is_available() else 'cpu'
    preprocessor = TDAPreprocessor.load(str(checkpoint_dir / 'preprocessor.pkl'))
    model = create_model(config.model, config.preprocessing, device)
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loader = create_test_loader(config, preprocessor, regime_labeler)

    print("\nPlotting predictions...")
    plot_predictions(model, test_loader, device=device, save_path=str(output_dir / 'predictions_analysis.png') if save_plots else None)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/tda_standalone')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    visualize_results(checkpoint_dir=args.checkpoint_dir, config_path=args.config, save_plots=not args.no_save)
