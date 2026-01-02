"""
Complexity Visualization Module

Visualizes complexity indicators on price charts for trader validation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional

from indicators import calculate_complexity_score


def plot_complexity_chart(
    df: pd.DataFrame,
    start_idx: int = 0,
    num_candles: int = 500,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot price chart with complexity indicators overlay.

    Args:
        df: DataFrame with OHLC data
        start_idx: Starting index for the view
        num_candles: Number of candles to display
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    # Slice data
    end_idx = min(start_idx + num_candles, len(df))
    df_view = df.iloc[start_idx:end_idx].copy()

    # Calculate complexity
    indicators, complexity = calculate_complexity_score(df_view)
    df_view["complexity"] = complexity

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle("Market Complexity Analysis", fontsize=14, fontweight="bold")

    # Subplot 1: Price with complexity color overlay
    ax1 = axes[0]
    x = range(len(df_view))
    dates = df_view["open_time"].values

    # Plot candlesticks simplified (just close line with complexity coloring)
    close = df_view["close"].values
    complexity_vals = df_view["complexity"].values

    # Color by complexity (green = low, red = high)
    colors = plt.cm.RdYlGn_r(complexity_vals)  # Reversed: red=high, green=low

    for i in range(len(x) - 1):
        ax1.plot([x[i], x[i + 1]], [close[i], close[i + 1]],
                 color=colors[i], linewidth=1.5, alpha=0.8)

    ax1.fill_between(x, df_view["low"], df_view["high"], alpha=0.2, color="gray")
    ax1.set_ylabel("Price (USDT)")
    ax1.set_title("BTC/USDT Price (colored by complexity: Green=Low, Red=High)")
    ax1.grid(True, alpha=0.3)

    # Add MAs for reference
    for period, color in [(20, "blue"), (50, "orange"), (100, "purple")]:
        if len(df_view) >= period:
            ma = df_view["close"].rolling(window=period).mean()
            ax1.plot(x, ma, color=color, linewidth=0.8, alpha=0.6, label=f"MA{period}")
    ax1.legend(loc="upper left", fontsize=8)

    # Subplot 2: Complexity Score
    ax2 = axes[1]
    ax2.fill_between(x, 0, complexity_vals, alpha=0.5, color="orange")
    ax2.plot(x, complexity_vals, color="darkorange", linewidth=1)
    ax2.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Complexity")
    ax2.set_ylim(0, 1)
    ax2.set_title("Complexity Score (0=Clear Trend, 1=Complex)")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Individual Indicators
    ax3 = axes[2]
    indicator_colors = {
        "ma_separation": "blue",
        "bb_width": "green",
        "price_efficiency": "red",
        "support_reaction": "purple",
        "directional_result": "brown",
    }

    for name, color in indicator_colors.items():
        ax3.plot(x, indicators[name].values, color=color, linewidth=0.8,
                 alpha=0.7, label=name.replace("_", " ").title())

    ax3.set_ylabel("Indicator Value")
    ax3.set_ylim(0, 1)
    ax3.set_title("Individual Indicators (higher = clearer trend)")
    ax3.legend(loc="upper left", fontsize=7, ncol=3)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Volume
    ax4 = axes[3]
    volume = df_view["volume"].values
    colors_vol = ["green" if df_view["close"].iloc[i] >= df_view["open"].iloc[i]
                  else "red" for i in range(len(df_view))]
    ax4.bar(x, volume, color=colors_vol, alpha=0.6, width=0.8)
    ax4.set_ylabel("Volume")
    ax4.set_xlabel("Candle Index")
    ax4.set_title("Volume")
    ax4.grid(True, alpha=0.3)

    # Format x-axis with dates for all subplots
    for ax in axes:
        # Set tick positions
        tick_spacing = max(1, len(x) // 10)
        tick_positions = list(range(0, len(x), tick_spacing))
        ax.set_xticks(tick_positions)

        # Format tick labels with dates
        tick_labels = [pd.Timestamp(dates[i]).strftime("%m/%d %H:%M")
                       for i in tick_positions]
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_complexity_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot distribution of complexity scores.

    Args:
        df: DataFrame with OHLC data
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    indicators, complexity = calculate_complexity_score(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Complexity Indicator Distributions", fontsize=14, fontweight="bold")

    # Plot complexity distribution
    ax = axes[0, 0]
    ax.hist(complexity.dropna(), bins=50, color="orange", alpha=0.7, edgecolor="black")
    ax.axvline(x=complexity.median(), color="red", linestyle="--", label=f"Median: {complexity.median():.3f}")
    ax.set_xlabel("Complexity Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Complexity Distribution")
    ax.legend()

    # Plot individual indicator distributions
    indicator_names = ["ma_separation", "bb_width", "price_efficiency",
                       "support_reaction", "directional_result"]
    colors = ["blue", "green", "red", "purple", "brown"]

    for i, (name, color) in enumerate(zip(indicator_names, colors)):
        row, col = divmod(i + 1, 3)
        ax = axes[row, col]
        data = indicators[name].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(x=data.median(), color="darkred", linestyle="--",
                   label=f"Median: {data.median():.3f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(name.replace("_", " ").title())
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Distribution chart saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def identify_extreme_periods(
    df: pd.DataFrame,
    top_n: int = 10,
    bottom_n: int = 10,
) -> dict:
    """
    Identify top/bottom complexity periods for visual validation.

    Args:
        df: DataFrame with OHLC data
        top_n: Number of high complexity periods to return
        bottom_n: Number of low complexity periods to return

    Returns:
        Dict with 'high_complexity' and 'low_complexity' DataFrames
    """
    indicators, complexity = calculate_complexity_score(df)
    df_with_complexity = df.copy()
    df_with_complexity["complexity"] = complexity

    # Add indicator columns
    for col in indicators.columns:
        df_with_complexity[f"ind_{col}"] = indicators[col]

    # Get extreme periods
    high_complexity = df_with_complexity.nlargest(top_n, "complexity")
    low_complexity = df_with_complexity.nsmallest(bottom_n, "complexity")

    return {
        "high_complexity": high_complexity,
        "low_complexity": low_complexity,
    }


def main():
    """Main function for visualization testing."""
    from pathlib import Path

    # Try to load data
    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Try 1m data first, fall back to 5m data
    data_file = data_dir / "BTCUSDT_spot_1m_etf_to_90d_ago.csv"
    if not data_file.exists():
        data_file = data_dir / "BTCUSDT_spot_etf_to_90d_ago.csv"

    if not data_file.exists():
        print(f"No data file found. Run collect_1m_data.py first.")
        return

    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    df["open_time"] = pd.to_datetime(df["open_time"])

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['open_time'].min()} ~ {df['open_time'].max()}")

    # Plot complexity chart for a sample period
    print("\nGenerating complexity chart...")
    plot_complexity_chart(
        df,
        start_idx=1000,
        num_candles=500,
        save_path=output_dir / "complexity_chart.png",
        show=False,
    )

    # Plot distribution
    print("\nGenerating distribution chart...")
    plot_complexity_distribution(
        df,
        save_path=output_dir / "complexity_distribution.png",
        show=False,
    )

    # Identify extreme periods
    print("\nIdentifying extreme complexity periods...")
    extremes = identify_extreme_periods(df)

    print("\n=== HIGH COMPLEXITY PERIODS (Complex/Choppy) ===")
    print(extremes["high_complexity"][["open_time", "close", "complexity"]].to_string())

    print("\n=== LOW COMPLEXITY PERIODS (Clear Trend) ===")
    print(extremes["low_complexity"][["open_time", "close", "complexity"]].to_string())


if __name__ == "__main__":
    main()
