#!/usr/bin/env python3
"""
Complexity Training and Visualization Pipeline

1. Split 1m data into Train/Backtest1/Backtest2
2. Resample to 15m candles (OHLCV)
3. Calculate complexity on 15m intervals using all 1m data within each interval
   (preserves fine-grained patterns vs. averaging individual 1m scores)
4. Visualize with candlestick + volume + complexity overlay
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from complexity import calculate_complexity_score


def load_1m_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine 1m data files."""
    df1 = pd.read_csv(data_dir / "BTCUSDT_spot_1m_etf_to_90d_ago.csv")
    df2 = pd.read_csv(data_dir / "BTCUSDT_spot_1m_last_90d.csv")

    df = pd.concat([df1, df2], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"])
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df):,} candles")
    print(f"Date range: {df['open_time'].min()} ~ {df['open_time'].max()}")

    return df


def split_data(df: pd.DataFrame) -> dict:
    """Split data into Train/Backtest1/Backtest2."""
    today = datetime.now()
    d180 = today - timedelta(days=180)
    d90 = today - timedelta(days=90)
    d1 = today - timedelta(days=1)

    train = df[df["open_time"] < d180].copy()
    backtest1 = df[(df["open_time"] >= d180) & (df["open_time"] < d90)].copy()
    backtest2 = df[(df["open_time"] >= d90) & (df["open_time"] < d1)].copy()

    print(f"\nData split:")
    print(f"  Train: {len(train):,} candles ({train['open_time'].min().date()} ~ {train['open_time'].max().date()})")
    print(f"  Backtest1: {len(backtest1):,} candles ({backtest1['open_time'].min().date()} ~ {backtest1['open_time'].max().date()})")
    print(f"  Backtest2: {len(backtest2):,} candles ({backtest2['open_time'].min().date()} ~ {backtest2['open_time'].max().date()})")

    return {
        "train": train,
        "backtest1": backtest1,
        "backtest2": backtest2,
    }


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m data to 15m candles (OHLCV only, no complexity)."""
    df = df.copy()
    df = df.set_index("open_time")

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Add other columns if present
    for col in ["buy_volume", "sell_volume", "quote_volume", "trades"]:
        if col in df.columns:
            agg_rules[col] = "sum"

    resampled = df.resample("15min").agg(agg_rules)
    resampled = resampled.dropna(subset=["open"])
    resampled = resampled.reset_index()

    return resampled


def calculate_complexity_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate complexity for each 15-minute interval using all 1-minute data.

    This preserves fine-grained patterns by calculating complexity on the full
    15-minute window rather than averaging individual 1-minute complexity scores.
    """
    df = df.copy()
    df['time_group'] = df['open_time'].dt.floor('15min')

    complexity_results = []
    total_groups = df['time_group'].nunique()

    for i, (timestamp, group) in enumerate(df.groupby('time_group'), 1):
        if i % 500 == 0:
            print(f"    Processing group {i}/{total_groups}...")

        # Calculate complexity on the entire 15-minute window
        group_indexed = group.set_index('open_time')
        indicators, complexity_score = calculate_complexity_score(group_indexed)

        # Store results
        result = {
            'open_time': timestamp,
            'complexity': complexity_score.mean() if not complexity_score.isna().all() else 0.5
        }

        # Add individual indicators for analysis
        for col in indicators.columns:
            result[f'ind_{col}'] = indicators[col].mean() if not indicators[col].isna().all() else 0.5

        complexity_results.append(result)

    return pd.DataFrame(complexity_results)


def plot_candlestick_with_complexity(
    df: pd.DataFrame,
    title: str,
    save_path: Path,
    num_charts: int = 20,
):
    """
    Create visualization with:
    - 15m candlestick chart
    - Volume bars at bottom
    - Complexity score overlay
    """
    # Split into chunks for multiple charts
    chunk_size = len(df) // num_charts
    if chunk_size < 50:
        chunk_size = 50
        num_charts = len(df) // chunk_size

    # Create figure with subplots (2 rows per chart: price+complexity, volume)
    fig, axes = plt.subplots(num_charts * 2, 1, figsize=(20, num_charts * 4),
                             gridspec_kw={"height_ratios": [3, 1] * num_charts})

    fig.suptitle(f"{title}\n15분봉 + 복잡도 + 거래량", fontsize=16, fontweight="bold")

    for i in range(num_charts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        if len(chunk) < 10:
            continue

        ax_price = axes[i * 2]
        ax_volume = axes[i * 2 + 1]

        x = range(len(chunk))

        # Plot candlesticks
        for j in range(len(chunk)):
            row = chunk.iloc[j]
            color = "green" if row["close"] >= row["open"] else "red"

            # Candle body
            body_bottom = min(row["open"], row["close"])
            body_height = abs(row["close"] - row["open"])
            ax_price.add_patch(Rectangle(
                (j - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, edgecolor=color, alpha=0.8
            ))

            # Wicks
            ax_price.plot([j, j], [row["low"], body_bottom], color=color, linewidth=0.8)
            ax_price.plot([j, j], [body_bottom + body_height, row["high"]], color=color, linewidth=0.8)

        # Complexity overlay (secondary y-axis)
        ax_complexity = ax_price.twinx()
        complexity_color = "purple"
        ax_complexity.fill_between(x, 0, chunk["complexity"], alpha=0.2, color=complexity_color)
        ax_complexity.plot(x, chunk["complexity"], color=complexity_color, linewidth=1.5, label="Complexity")
        ax_complexity.set_ylim(0, 1)
        ax_complexity.set_ylabel("Complexity", color=complexity_color)
        ax_complexity.tick_params(axis="y", labelcolor=complexity_color)

        # Price axis settings
        ax_price.set_xlim(-1, len(chunk))
        ax_price.set_ylabel("Price (USDT)")
        ax_price.grid(True, alpha=0.3)

        # Date labels
        date_start = chunk["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M")
        date_end = chunk["open_time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        ax_price.set_title(f"Chart {i+1}: {date_start} ~ {date_end}", fontsize=10)

        # Volume bars
        volume_colors = ["green" if chunk.iloc[j]["close"] >= chunk.iloc[j]["open"] else "red"
                        for j in range(len(chunk))]
        ax_volume.bar(x, chunk["volume"], color=volume_colors, alpha=0.7, width=0.8)
        ax_volume.set_xlim(-1, len(chunk))
        ax_volume.set_ylabel("Volume")
        ax_volume.grid(True, alpha=0.3)

        # X-axis labels (show every 10th candle)
        tick_spacing = max(1, len(chunk) // 10)
        tick_positions = list(range(0, len(chunk), tick_spacing))
        tick_labels = [chunk["open_time"].iloc[k].strftime("%m/%d\n%H:%M") for k in tick_positions]
        ax_volume.set_xticks(tick_positions)
        ax_volume.set_xticklabels(tick_labels, fontsize=7)
        ax_price.set_xticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_complexity_stats(df: pd.DataFrame, name: str):
    """Print complexity statistics."""
    print(f"\n{name} Complexity Stats:")
    print(f"  Mean: {df['complexity'].mean():.4f}")
    print(f"  Std: {df['complexity'].std():.4f}")
    print(f"  Min: {df['complexity'].min():.4f}")
    print(f"  Max: {df['complexity'].max():.4f}")

    # High/Low complexity periods
    high_pct = (df["complexity"] > 0.7).sum() / len(df) * 100
    low_pct = (df["complexity"] < 0.3).sum() / len(df) * 100
    print(f"  High complexity (>0.7): {high_pct:.1f}%")
    print(f"  Low complexity (<0.3): {low_pct:.1f}%")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = base_dir / "data"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Complexity Training and Visualization Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading 1m data...")
    df = load_1m_data(data_dir)

    # 2. Split data
    print("\n[2/5] Splitting data...")
    splits = split_data(df)

    # 3. Resample to 15m (OHLCV only)
    print("\n[3/5] Resampling to 15m...")
    splits_15m = {}
    for name, split_df in splits.items():
        splits_15m[name] = resample_to_15m(split_df)
        print(f"  {name}: {len(splits_15m[name]):,} candles (15m)")

    # 4. Calculate complexity on 15m intervals using 1m granularity
    print("\n[4/5] Calculating complexity (15m intervals with 1m data)...")
    for name, split_1m in splits.items():
        print(f"  Processing {name}...")
        complexity_df = calculate_complexity_15m(split_1m)

        # Merge with 15m OHLCV data
        splits_15m[name] = pd.merge(
            splits_15m[name],
            complexity_df,
            on='open_time',
            how='left'
        )

        # Fill any missing complexity values
        splits_15m[name]['complexity'] = splits_15m[name]['complexity'].fillna(0.5)

        print_complexity_stats(splits_15m[name], name)

    # 5. Visualize
    print("\n[5/5] Creating visualizations...")

    plot_candlestick_with_complexity(
        splits_15m["train"],
        "Train Dataset (ETF Launch ~ 180 days ago)",
        output_dir / "train_complexity.png",
        num_charts=20,
    )

    plot_candlestick_with_complexity(
        splits_15m["backtest1"],
        "Backtest 1 (180 days ago ~ 90 days ago)",
        output_dir / "backtest1_complexity.png",
        num_charts=20,
    )

    plot_candlestick_with_complexity(
        splits_15m["backtest2"],
        "Backtest 2 (90 days ago ~ 1 day ago)",
        output_dir / "backtest2_complexity.png",
        num_charts=20,
    )

    # Save processed data
    print("\n[Bonus] Saving processed 15m data...")
    for name, split_df in splits_15m.items():
        save_path = output_dir / f"{name}_15m_complexity.csv"
        split_df.to_csv(save_path, index=False)
        print(f"  Saved: {save_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
