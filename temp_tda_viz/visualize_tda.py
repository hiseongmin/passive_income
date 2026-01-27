"""
TDA Feature Visualization Script

Visualizes the pre-extracted TDA features from the training dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
CACHE_PATH = Path("/home/ubuntu/joo/passive_income/cache/tda_features_train.npy")
OUTPUT_DIR = Path("/home/ubuntu/joo/passive_income/temp_tda_viz")

# Feature structure (214 total dimensions)
FEATURE_GROUPS = {
    "Betti H0": (0, 50),      # Connected components
    "Betti H1": (50, 100),    # Loops/holes
    "Entropy": (100, 102),    # Persistent entropy (H0, H1)
    "Persistence": (102, 104), # Total persistence (H0, H1)
    "Landscape": (104, 214),  # Landscape L2 norms
}

def load_data():
    """Load the TDA features."""
    print(f"Loading TDA features from {CACHE_PATH}")
    data = np.load(CACHE_PATH)
    print(f"Shape: {data.shape} ({data.shape[0]:,} samples, {data.shape[1]} features)")
    return data

def plot_feature_overview(data):
    """Create box plots showing feature group distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, (start, end)) in enumerate(FEATURE_GROUPS.items()):
        ax = axes[idx]
        group_data = data[:, start:end]

        # Sample if too many features for box plot
        n_features = end - start
        if n_features > 20:
            # Show summary statistics instead
            means = group_data.mean(axis=0)
            stds = group_data.std(axis=0)
            ax.fill_between(range(n_features), means - stds, means + stds, alpha=0.3)
            ax.plot(means, 'b-', linewidth=1)
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Value (mean +/- std)")
        else:
            ax.boxplot(group_data, showfliers=False)
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Value")

        ax.set_title(f"{name} ({n_features} dims)")
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    axes[5].axis('off')

    # Add overall statistics
    stats_text = f"Total Samples: {data.shape[0]:,}\n"
    stats_text += f"Total Features: {data.shape[1]}\n\n"
    stats_text += "Feature Ranges:\n"
    for name, (start, end) in FEATURE_GROUPS.items():
        group_data = data[:, start:end]
        stats_text += f"  {name}: [{group_data.min():.3f}, {group_data.max():.3f}]\n"

    axes[5].text(0.1, 0.5, stats_text, transform=axes[5].transAxes,
                 fontsize=11, verticalalignment='center', fontfamily='monospace')

    plt.suptitle("TDA Feature Group Overview", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_feature_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_feature_overview.png")

def plot_betti_curves(data):
    """Plot average Betti curves with standard deviation bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Betti H0 (connected components)
    betti_h0 = data[:, 0:50]
    mean_h0 = betti_h0.mean(axis=0)
    std_h0 = betti_h0.std(axis=0)

    ax = axes[0]
    x = np.arange(50)
    ax.fill_between(x, mean_h0 - std_h0, mean_h0 + std_h0, alpha=0.3, color='blue')
    ax.plot(x, mean_h0, 'b-', linewidth=2, label='Mean')
    ax.set_xlabel("Filtration Step (bin)")
    ax.set_ylabel("Betti Number")
    ax.set_title("Betti Curve H0 (Connected Components)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Betti H1 (loops)
    betti_h1 = data[:, 50:100]
    mean_h1 = betti_h1.mean(axis=0)
    std_h1 = betti_h1.std(axis=0)

    ax = axes[1]
    ax.fill_between(x, mean_h1 - std_h1, mean_h1 + std_h1, alpha=0.3, color='red')
    ax.plot(x, mean_h1, 'r-', linewidth=2, label='Mean')
    ax.set_xlabel("Filtration Step (bin)")
    ax.set_ylabel("Betti Number")
    ax.set_title("Betti Curve H1 (Loops/Holes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Average Betti Curves (shaded = +/- 1 std)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_betti_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_betti_curves.png")

def plot_persistence_stats(data):
    """Plot distributions of persistence statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Persistent Entropy
    entropy_h0 = data[:, 100]
    entropy_h1 = data[:, 101]

    axes[0, 0].hist(entropy_h0, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f"Persistent Entropy H0\nmean={entropy_h0.mean():.4f}, std={entropy_h0.std():.4f}")
    axes[0, 0].set_xlabel("Entropy Value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(entropy_h0.mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()

    axes[0, 1].hist(entropy_h1, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f"Persistent Entropy H1\nmean={entropy_h1.mean():.4f}, std={entropy_h1.std():.4f}")
    axes[0, 1].set_xlabel("Entropy Value")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].axvline(entropy_h1.mean(), color='blue', linestyle='--', label='Mean')
    axes[0, 1].legend()

    # Total Persistence
    persist_h0 = data[:, 102]
    persist_h1 = data[:, 103]

    axes[1, 0].hist(persist_h0, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title(f"Total Persistence H0\nmean={persist_h0.mean():.4f}, std={persist_h0.std():.4f}")
    axes[1, 0].set_xlabel("Persistence Value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].axvline(persist_h0.mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].legend()

    axes[1, 1].hist(persist_h1, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title(f"Total Persistence H1\nmean={persist_h1.mean():.4f}, std={persist_h1.std():.4f}")
    axes[1, 1].set_xlabel("Persistence Value")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].axvline(persist_h1.mean(), color='blue', linestyle='--', label='Mean')
    axes[1, 1].legend()

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.suptitle("Persistence Statistics Distributions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_persistence_stats.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_persistence_stats.png")

def plot_landscape_heatmap(data):
    """Plot heatmap of persistence landscape L2 norms."""
    landscape_data = data[:, 104:214]  # 110 dimensions

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean landscape values
    mean_landscape = landscape_data.mean(axis=0)

    # Reshape to show structure (assuming 5 layers x 2 dims x 11 = 110)
    # Or show as 1D heatmap
    ax = axes[0]
    im = ax.imshow(mean_landscape.reshape(1, -1), aspect='auto', cmap='viridis')
    ax.set_title("Mean Landscape L2 Norms (110 dims)")
    ax.set_xlabel("Feature Index")
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Mean Value')

    # Distribution of landscape values
    ax = axes[1]
    ax.hist(landscape_data.flatten(), bins=100, color='purple', alpha=0.7, edgecolor='black')
    ax.set_title(f"Landscape L2 Norm Distribution\nAll {landscape_data.shape[0]:,} samples x 110 dims")
    ax.set_xlabel("L2 Norm Value")
    ax.set_ylabel("Count")
    ax.axvline(landscape_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {landscape_data.mean():.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Persistence Landscape Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_landscape_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_landscape_heatmap.png")

def plot_correlation_matrix(data):
    """Plot correlation matrix between feature groups."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Sample data to compute correlations (full data too slow)
    n_samples = min(10000, data.shape[0])
    idx = np.random.choice(data.shape[0], n_samples, replace=False)
    sampled_data = data[idx]

    # Compute group-level correlations (mean of each group)
    group_means = {}
    for name, (start, end) in FEATURE_GROUPS.items():
        group_means[name] = sampled_data[:, start:end].mean(axis=1)

    group_df = np.column_stack(list(group_means.values()))
    group_names = list(group_means.keys())

    corr_matrix = np.corrcoef(group_df.T)

    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(group_names)))
    ax.set_yticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_yticklabels(group_names)
    ax.set_title("Feature Group Correlations\n(mean of each group)")

    # Add correlation values
    for i in range(len(group_names)):
        for j in range(len(group_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                   color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Correlation')

    # Full correlation matrix (subsampled features)
    ax = axes[1]
    # Take every 10th feature to make visualization manageable
    feature_idx = np.arange(0, 214, 10)
    subsampled = sampled_data[:, feature_idx]
    full_corr = np.corrcoef(subsampled.T)

    im2 = ax.imshow(full_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f"Feature Correlations (every 10th feature)\n{len(feature_idx)} features shown")
    ax.set_xlabel("Feature Index (subsampled)")
    ax.set_ylabel("Feature Index (subsampled)")
    plt.colorbar(im2, ax=ax, label='Correlation')

    plt.suptitle("TDA Feature Correlations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_correlation_matrix.png")

def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("TDA Feature Visualization")
    print("=" * 60)

    # Load data
    data = load_data()
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_feature_overview(data)
    plot_betti_curves(data)
    plot_persistence_stats(data)
    plot_landscape_heatmap(data)
    plot_correlation_matrix(data)

    print()
    print("=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
