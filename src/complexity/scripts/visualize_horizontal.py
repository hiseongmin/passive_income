#!/usr/bin/env python3
"""
Horizontal Complexity Visualization

Creates wide horizontal charts with:
- 15m candlestick
- Volume bars at bottom
- Complexity overlay
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_horizontal_chart(
    df: pd.DataFrame,
    title: str,
    save_path: Path,
    candles_per_chart: int = 500,
):
    """
    Create wide horizontal chart.

    Args:
        df: DataFrame with OHLCV + complexity
        title: Chart title
        save_path: Output path
        candles_per_chart: Number of candles per chart
    """
    num_charts = (len(df) + candles_per_chart - 1) // candles_per_chart

    for chart_idx in range(num_charts):
        start_idx = chart_idx * candles_per_chart
        end_idx = min((chart_idx + 1) * candles_per_chart, len(df))
        chunk = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        if len(chunk) < 10:
            continue

        # Create figure - wide horizontal layout
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, figsize=(24, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )

        x = range(len(chunk))

        # ===== Candlestick Chart =====
        for j in range(len(chunk)):
            row = chunk.iloc[j]
            is_up = row["close"] >= row["open"]
            color = "#26a69a" if is_up else "#ef5350"  # Green / Red

            # Candle body
            body_bottom = min(row["open"], row["close"])
            body_height = abs(row["close"] - row["open"])
            if body_height < 0.01:
                body_height = 0.01

            ax_price.add_patch(Rectangle(
                (j - 0.35, body_bottom), 0.7, body_height,
                facecolor=color, edgecolor=color, alpha=0.9
            ))

            # Wicks
            ax_price.plot([j, j], [row["low"], body_bottom], color=color, linewidth=0.8)
            ax_price.plot([j, j], [body_bottom + body_height, row["high"]], color=color, linewidth=0.8)

        # ===== Complexity Overlay =====
        ax_complexity = ax_price.twinx()
        complexity_vals = chunk["complexity"].values

        # Color gradient based on complexity
        ax_complexity.fill_between(x, 0, complexity_vals, alpha=0.3, color="purple")
        ax_complexity.plot(x, complexity_vals, color="purple", linewidth=1.5, alpha=0.8)

        # Mark high complexity zones (>0.7)
        high_mask = complexity_vals > 0.7
        if high_mask.any():
            ax_complexity.fill_between(x, 0, complexity_vals,
                                       where=high_mask, alpha=0.4, color="red",
                                       label="High Complexity (>0.7)")

        ax_complexity.set_ylim(0, 1)
        ax_complexity.set_ylabel("Complexity", color="purple", fontsize=10)
        ax_complexity.tick_params(axis="y", labelcolor="purple")
        ax_complexity.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax_complexity.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, linewidth=0.8)

        # Price axis
        ax_price.set_xlim(-1, len(chunk))
        ax_price.set_ylabel("Price (USDT)", fontsize=10)
        ax_price.grid(True, alpha=0.3)

        # Title with date range
        date_start = chunk["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M")
        date_end = chunk["open_time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        avg_complexity = chunk["complexity"].mean()
        ax_price.set_title(
            f"{title} | Chart {chart_idx + 1}/{num_charts} | {date_start} ~ {date_end} | Avg Complexity: {avg_complexity:.3f}",
            fontsize=12, fontweight="bold"
        )

        # ===== Volume Bars =====
        volume_colors = ["#26a69a" if chunk.iloc[j]["close"] >= chunk.iloc[j]["open"] else "#ef5350"
                        for j in range(len(chunk))]
        ax_volume.bar(x, chunk["volume"], color=volume_colors, alpha=0.7, width=0.8)
        ax_volume.set_xlim(-1, len(chunk))
        ax_volume.set_ylabel("Volume", fontsize=10)
        ax_volume.grid(True, alpha=0.3)

        # X-axis labels
        tick_spacing = max(1, len(chunk) // 20)
        tick_positions = list(range(0, len(chunk), tick_spacing))
        tick_labels = [chunk["open_time"].iloc[k].strftime("%m/%d %H:%M") for k in tick_positions]
        ax_volume.set_xticks(tick_positions)
        ax_volume.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        plt.tight_layout()

        # Save with chart index
        chart_path = save_path.parent / f"{save_path.stem}_{chart_idx + 1:02d}{save_path.suffix}"
        plt.savefig(chart_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {chart_path.name}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"

    print("=" * 60)
    print("Horizontal Complexity Visualization")
    print("=" * 60)

    # Load pre-computed 15m data
    datasets = ["train", "backtest1", "backtest2"]
    titles = {
        "train": "Train (ETF Launch ~ 180d ago)",
        "backtest1": "Backtest 1 (180d ~ 90d ago)",
        "backtest2": "Backtest 2 (90d ago ~ Now)",
    }

    for name in datasets:
        csv_path = output_dir / f"{name}_15m_complexity.csv"
        if not csv_path.exists():
            print(f"  Skipping {name} - file not found")
            continue

        print(f"\n[{name}] Loading {csv_path.name}...")
        df = pd.read_csv(csv_path)
        df["open_time"] = pd.to_datetime(df["open_time"])

        print(f"  {len(df):,} candles, creating charts...")
        plot_horizontal_chart(
            df,
            titles[name],
            output_dir / f"{name}_horizontal.png",
            candles_per_chart=400,  # ~100 hours per chart
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
