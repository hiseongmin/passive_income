"""
H1 Temporal Analysis Script

Analyzes H1 (loops/holes) features from TDA over time to understand
how cyclical patterns in Bitcoin price topology evolve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration
CACHE_PATH = Path("/home/ubuntu/joo/passive_income/cache/tda_features_train.npy")
DATA_PATH = Path("/home/ubuntu/joo/passive_income/data/BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv")
OUTPUT_DIR = Path("/home/ubuntu/joo/passive_income/temp_tda_viz/h1_temporal_analysis")

# TDA parameters (must match dataset.py)
WINDOW_SIZE = 672  # 7 days of 15-min candles
SEQ_LEN = 96       # 24 hours of 15-min candles

# H1 feature indices
H1_BETTI_START = 50
H1_BETTI_END = 100
H1_ENTROPY_IDX = 101
H1_PERSISTENCE_IDX = 103


def load_data():
    """Load TDA features and align with timestamps."""
    print("Loading TDA features...")
    tda_features = np.load(CACHE_PATH)
    n_tda_samples = len(tda_features)
    print(f"  TDA shape: {tda_features.shape}")

    print("Loading price data...")
    df = pd.read_csv(DATA_PATH)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")

    # Compute valid indices (same logic as dataset.py)
    min_idx = max(WINDOW_SIZE, SEQ_LEN)
    all_valid_indices = list(range(min_idx, len(df)))
    print(f"  All valid indices: {len(all_valid_indices)} (starting from {min_idx})")

    # The TDA features are only for training split (first N samples after valid_indices)
    # Match the number of TDA samples
    valid_indices = all_valid_indices[:n_tda_samples]
    print(f"  Using first {len(valid_indices)} indices to match TDA features")

    # Verify alignment
    assert len(valid_indices) == len(tda_features), \
        f"Mismatch: {len(valid_indices)} valid indices vs {len(tda_features)} TDA samples"

    # Extract timestamps for TDA samples
    timestamps = df.iloc[valid_indices]['open_time'].values
    close_prices = df.iloc[valid_indices]['close'].values

    print(f"  Training date range: {timestamps[0]} to {timestamps[-1]}")

    return tda_features, timestamps, close_prices, df, valid_indices


def extract_h1_features(tda_features):
    """Extract H1-specific features."""
    h1_betti = tda_features[:, H1_BETTI_START:H1_BETTI_END]  # (N, 50)
    h1_entropy = tda_features[:, H1_ENTROPY_IDX]             # (N,)
    h1_persistence = tda_features[:, H1_PERSISTENCE_IDX]     # (N,)

    # Derived metrics
    h1_betti_mean = h1_betti.mean(axis=1)
    h1_betti_max = h1_betti.max(axis=1)
    h1_betti_sum = h1_betti.sum(axis=1)

    return {
        'betti_curve': h1_betti,
        'entropy': h1_entropy,
        'persistence': h1_persistence,
        'betti_mean': h1_betti_mean,
        'betti_max': h1_betti_max,
        'betti_sum': h1_betti_sum,
    }


def plot_h1_summary_timeline(timestamps, h1_features, close_prices):
    """Plot 1: H1 summary metrics over time with price overlay."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Convert to pandas for rolling calculations
    ts = pd.to_datetime(timestamps)

    # Plot 1: Close price
    ax = axes[0]
    ax.plot(ts, close_prices, 'k-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('BTC Price (USD)')
    ax.set_title('Bitcoin Close Price')
    ax.grid(True, alpha=0.3)

    # Plot 2: H1 Betti Mean with rolling average
    ax = axes[1]
    betti_mean = h1_features['betti_mean']
    ax.plot(ts, betti_mean, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    # Rolling average (96 samples = 1 day)
    rolling = pd.Series(betti_mean).rolling(96*7, center=True).mean()
    ax.plot(ts, rolling, 'b-', linewidth=2, label='7-day MA')
    ax.set_ylabel('H1 Betti Mean')
    ax.set_title('H1 Betti Curve Mean (Loop Count)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 3: H1 Entropy
    ax = axes[2]
    entropy = h1_features['entropy']
    ax.plot(ts, entropy, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
    rolling = pd.Series(entropy).rolling(96*7, center=True).mean()
    ax.plot(ts, rolling, 'r-', linewidth=2, label='7-day MA')
    ax.set_ylabel('H1 Entropy')
    ax.set_title('H1 Persistent Entropy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 4: H1 Persistence
    ax = axes[3]
    persistence = h1_features['persistence']
    ax.plot(ts, persistence, 'g-', alpha=0.3, linewidth=0.5, label='Raw')
    rolling = pd.Series(persistence).rolling(96*7, center=True).mean()
    ax.plot(ts, rolling, 'g-', linewidth=2, label='7-day MA')
    ax.set_ylabel('H1 Persistence')
    ax.set_title('H1 Total Persistence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.suptitle('H1 Features Over Time (Training Period)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_h1_summary_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_h1_summary_timeline.png")


def plot_h1_betti_heatmap(timestamps, h1_features):
    """Plot 2: Heatmap of H1 Betti curve evolution over time."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ts = pd.to_datetime(timestamps)
    betti_curve = h1_features['betti_curve']

    # Create daily aggregation
    df_temp = pd.DataFrame({
        'date': ts.date,
    })
    for i in range(50):
        df_temp[f'bin_{i}'] = betti_curve[:, i]

    daily = df_temp.groupby('date').mean()
    dates = pd.to_datetime(daily.index)
    betti_daily = daily.values  # (n_days, 50)

    # Plot heatmap
    ax = axes[0]
    im = ax.imshow(betti_daily.T, aspect='auto', cmap='viridis',
                   extent=[0, len(dates), 49, 0])
    ax.set_ylabel('Filtration Bin')
    ax.set_title('H1 Betti Curve Evolution (Daily Averaged)')

    # Set x-axis to show dates
    n_ticks = min(12, len(dates))
    tick_positions = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in tick_positions], rotation=45, ha='right')

    plt.colorbar(im, ax=ax, label='Betti Number')

    # Plot variance per bin over time
    ax = axes[1]
    bin_variance = betti_daily.var(axis=0)
    ax.bar(range(50), bin_variance, color='purple', alpha=0.7)
    ax.set_xlabel('Filtration Bin')
    ax.set_ylabel('Variance Over Time')
    ax.set_title('H1 Betti Curve: Which Bins Vary Most Over Time?')
    ax.grid(True, alpha=0.3)

    plt.suptitle('H1 Betti Curve Temporal Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_h1_betti_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_h1_betti_heatmap.png")


def plot_h1_monthly_distribution(timestamps, h1_features):
    """Plot 3: H1 feature distributions by month."""
    ts = pd.to_datetime(timestamps)

    df_temp = pd.DataFrame({
        'month': ts.to_period('M'),
        'entropy': h1_features['entropy'],
        'persistence': h1_features['persistence'],
        'betti_mean': h1_features['betti_mean'],
    })

    months = sorted(df_temp['month'].unique())
    n_months = len(months)

    fig, axes = plt.subplots(3, min(n_months, 6), figsize=(18, 10))
    if n_months < 6:
        # Pad axes if fewer months
        pass

    # Only show first 6 months if more
    months_to_show = months[:6]

    for i, month in enumerate(months_to_show):
        month_data = df_temp[df_temp['month'] == month]

        # Entropy
        ax = axes[0, i] if n_months > 1 else axes[0]
        ax.hist(month_data['entropy'], bins=30, color='red', alpha=0.7, edgecolor='black')
        ax.set_title(f'{month}')
        if i == 0:
            ax.set_ylabel('Entropy')
        ax.set_xlim(0, 10)

        # Persistence
        ax = axes[1, i] if n_months > 1 else axes[1]
        ax.hist(month_data['persistence'], bins=30, color='green', alpha=0.7, edgecolor='black')
        if i == 0:
            ax.set_ylabel('Persistence')
        ax.set_xlim(0, 20)

        # Betti Mean
        ax = axes[2, i] if n_months > 1 else axes[2]
        ax.hist(month_data['betti_mean'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        if i == 0:
            ax.set_ylabel('Betti Mean')
        ax.set_xlim(0, 15)

    plt.suptitle('H1 Feature Distributions by Month', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_h1_monthly_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_h1_monthly_distribution.png")


def plot_h1_price_correlation(timestamps, h1_features, close_prices):
    """Plot 4: H1 features vs price metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Calculate price metrics
    returns = np.diff(close_prices) / close_prices[:-1]
    returns = np.concatenate([[0], returns])

    # Rolling volatility (96 samples = 1 day)
    volatility = pd.Series(returns).rolling(96).std().fillna(0).values

    # H1 metrics
    entropy = h1_features['entropy']
    persistence = h1_features['persistence']
    betti_mean = h1_features['betti_mean']

    # Subsample for scatter plots (too many points otherwise)
    n_sample = min(5000, len(close_prices))
    idx = np.random.choice(len(close_prices), n_sample, replace=False)

    # Row 1: vs Price
    ax = axes[0, 0]
    ax.scatter(close_prices[idx], entropy[idx], alpha=0.3, s=5)
    r, _ = pearsonr(close_prices, entropy)
    ax.set_xlabel('Close Price')
    ax.set_ylabel('H1 Entropy')
    ax.set_title(f'Entropy vs Price (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(close_prices[idx], persistence[idx], alpha=0.3, s=5, color='green')
    r, _ = pearsonr(close_prices, persistence)
    ax.set_xlabel('Close Price')
    ax.set_ylabel('H1 Persistence')
    ax.set_title(f'Persistence vs Price (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(close_prices[idx], betti_mean[idx], alpha=0.3, s=5, color='blue')
    r, _ = pearsonr(close_prices, betti_mean)
    ax.set_xlabel('Close Price')
    ax.set_ylabel('H1 Betti Mean')
    ax.set_title(f'Betti Mean vs Price (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    # Row 2: vs Volatility
    ax = axes[1, 0]
    ax.scatter(volatility[idx], entropy[idx], alpha=0.3, s=5, color='red')
    r, _ = pearsonr(volatility, entropy)
    ax.set_xlabel('Rolling Volatility (1d)')
    ax.set_ylabel('H1 Entropy')
    ax.set_title(f'Entropy vs Volatility (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(volatility[idx], persistence[idx], alpha=0.3, s=5, color='orange')
    r, _ = pearsonr(volatility, persistence)
    ax.set_xlabel('Rolling Volatility (1d)')
    ax.set_ylabel('H1 Persistence')
    ax.set_title(f'Persistence vs Volatility (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(volatility[idx], betti_mean[idx], alpha=0.3, s=5, color='purple')
    r, _ = pearsonr(volatility, betti_mean)
    ax.set_xlabel('Rolling Volatility (1d)')
    ax.set_ylabel('H1 Betti Mean')
    ax.set_title(f'Betti Mean vs Volatility (r={r:.3f})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('H1 Features vs Price Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_h1_price_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_h1_price_correlation.png")


def plot_h1_periodicity(timestamps, h1_features):
    """Plot 5: Periodicity analysis of H1 features."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = {
        'Entropy': h1_features['entropy'],
        'Persistence': h1_features['persistence'],
        'Betti Mean': h1_features['betti_mean'],
    }

    for i, (name, data) in enumerate(metrics.items()):
        # Autocorrelation (up to 7 days = 672 lags)
        ax = axes[0, i]
        max_lag = min(672, len(data) // 4)
        autocorr = [np.corrcoef(data[:-lag], data[lag:])[0, 1] for lag in range(1, max_lag)]

        # Convert lags to hours
        hours = np.arange(1, max_lag) * 0.25  # 15-min candles
        ax.plot(hours, autocorr, 'b-', linewidth=1)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(24, color='r', linestyle='--', alpha=0.5, label='1 day')
        ax.axvline(168, color='g', linestyle='--', alpha=0.5, label='1 week')
        ax.set_xlabel('Lag (hours)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'{name} Autocorrelation')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # FFT / Power spectrum
        ax = axes[1, i]
        # Detrend and compute FFT
        detrended = data - np.mean(data)
        n = len(detrended)
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(n, d=0.25)  # 15-min sampling -> hours
        power = np.abs(fft) ** 2

        # Only positive frequencies, convert to period in hours
        pos_mask = freqs > 0
        periods = 1 / freqs[pos_mask]
        power_pos = power[pos_mask]

        # Focus on periods from 1 hour to 7 days
        period_mask = (periods >= 1) & (periods <= 168)
        ax.semilogy(periods[period_mask], power_pos[period_mask], 'b-', linewidth=0.5)
        ax.axvline(24, color='r', linestyle='--', alpha=0.5, label='1 day')
        ax.axvline(168, color='g', linestyle='--', alpha=0.5, label='1 week')
        ax.set_xlabel('Period (hours)')
        ax.set_ylabel('Power')
        ax.set_title(f'{name} Power Spectrum')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 168)

    plt.suptitle('H1 Periodicity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_h1_periodicity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_h1_periodicity.png")


def plot_h1_regimes(timestamps, h1_features, close_prices):
    """Plot 6: Cluster H1 features into regimes."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ts = pd.to_datetime(timestamps)

    # Prepare features for clustering
    X = np.column_stack([
        h1_features['entropy'],
        h1_features['persistence'],
        h1_features['betti_mean'],
    ])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering (4 regimes)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(X_scaled)

    # Colors for regimes
    colors = ['blue', 'green', 'orange', 'red']
    regime_names = ['Regime 0', 'Regime 1', 'Regime 2', 'Regime 3']

    # Plot 1: Price with regime colors
    ax = axes[0]
    for r in range(n_clusters):
        mask = regimes == r
        ax.scatter(ts[mask], close_prices[mask], c=colors[r], s=1, alpha=0.5, label=regime_names[r])
    ax.set_ylabel('BTC Price (USD)')
    ax.set_title('Price Colored by H1 Regime')
    ax.legend(loc='upper right', markerscale=5)
    ax.grid(True, alpha=0.3)

    # Plot 2: Regime timeline
    ax = axes[1]
    ax.scatter(ts, regimes, c=[colors[r] for r in regimes], s=1, alpha=0.5)
    ax.set_ylabel('Regime')
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(regime_names)
    ax.set_title('Regime Over Time')
    ax.grid(True, alpha=0.3)

    # Plot 3: Regime characteristics
    ax = axes[2]
    regime_stats = []
    for r in range(n_clusters):
        mask = regimes == r
        stats = {
            'Regime': regime_names[r],
            'Count': mask.sum(),
            'Entropy': h1_features['entropy'][mask].mean(),
            'Persistence': h1_features['persistence'][mask].mean(),
            'Betti Mean': h1_features['betti_mean'][mask].mean(),
            'Price Mean': close_prices[mask].mean(),
        }
        regime_stats.append(stats)

    # Bar chart of regime characteristics
    x = np.arange(n_clusters)
    width = 0.2
    ax.bar(x - 1.5*width, [s['Entropy'] for s in regime_stats], width, label='Entropy', color='red', alpha=0.7)
    ax.bar(x - 0.5*width, [s['Persistence'] for s in regime_stats], width, label='Persistence', color='green', alpha=0.7)
    ax.bar(x + 0.5*width, [s['Betti Mean'] for s in regime_stats], width, label='Betti Mean', color='blue', alpha=0.7)
    ax.bar(x + 1.5*width, [s['Price Mean']/10000 for s in regime_stats], width, label='Price/10k', color='black', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_names)
    ax.set_ylabel('Value')
    ax.set_title('Regime Characteristics (Mean Values)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Format x-axis for timeline plots
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())

    plt.suptitle('H1 Regime Analysis (K-Means Clustering)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_h1_regimes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_h1_regimes.png")

    # Save regime stats
    stats_df = pd.DataFrame(regime_stats)
    stats_df.to_csv(OUTPUT_DIR / 'h1_statistics.csv', index=False)
    print("Saved: h1_statistics.csv")

    return regime_stats


def main():
    """Run full H1 temporal analysis."""
    print("=" * 70)
    print("H1 Temporal Analysis")
    print("=" * 70)

    # Load data
    tda_features, timestamps, close_prices, df, valid_indices = load_data()

    # Extract H1 features
    print("\nExtracting H1 features...")
    h1_features = extract_h1_features(tda_features)
    print(f"  H1 Betti curve shape: {h1_features['betti_curve'].shape}")
    print(f"  H1 Entropy range: [{h1_features['entropy'].min():.3f}, {h1_features['entropy'].max():.3f}]")
    print(f"  H1 Persistence range: [{h1_features['persistence'].min():.3f}, {h1_features['persistence'].max():.3f}]")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_h1_summary_timeline(timestamps, h1_features, close_prices)
    plot_h1_betti_heatmap(timestamps, h1_features)
    plot_h1_monthly_distribution(timestamps, h1_features)
    plot_h1_price_correlation(timestamps, h1_features, close_prices)
    plot_h1_periodicity(timestamps, h1_features)
    regime_stats = plot_h1_regimes(timestamps, h1_features, close_prices)

    # Print regime summary
    print("\n" + "=" * 70)
    print("Regime Summary:")
    print("=" * 70)
    for stats in regime_stats:
        print(f"\n{stats['Regime']}:")
        print(f"  Count: {stats['Count']:,} samples ({100*stats['Count']/len(timestamps):.1f}%)")
        print(f"  Entropy: {stats['Entropy']:.3f}")
        print(f"  Persistence: {stats['Persistence']:.3f}")
        print(f"  Betti Mean: {stats['Betti Mean']:.3f}")
        print(f"  Avg Price: ${stats['Price Mean']:,.0f}")

    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
