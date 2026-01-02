"""
Data Modification Script: Trigger and Max_Pct Generation

Converts 5-minute Bitcoin OHLCV data to 15-minute intervals and generates
Trigger and Max_Pct columns based on 2-hour price movement detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def resample_5min_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-minute data to 15-minute candles.

    Aggregation rules:
    - open_time: First timestamp
    - open: First open value
    - high: Maximum high
    - low: Minimum low
    - close: Last close value
    - volume, buy_volume, sell_volume, volume_delta: Sum
    - cvd, open_interest: Last value
    """
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.set_index('open_time')

    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'volume_delta': 'sum',
        'cvd': 'last',
    }

    # Add open_interest if present (only in perpetual data)
    if 'open_interest' in df.columns:
        agg_rules['open_interest'] = 'last'

    resampled = df.resample('15min').agg(agg_rules)
    resampled = resampled.dropna(subset=['open'])  # Remove incomplete candles
    resampled = resampled.reset_index()

    return resampled


def detect_triggers_and_max_pct(df: pd.DataFrame, window_size: int = 8, threshold_pct: float = 2.0) -> pd.DataFrame:
    """
    Detect 2% price movement windows and set Trigger/Max_Pct columns.

    Args:
        df: DataFrame with OHLCV data
        window_size: Number of time steps for the window (default 8 = 2 hours at 15min)
        threshold_pct: Minimum price difference percentage to trigger (default 2.0%)

    Returns:
        DataFrame with Trigger and Max_Pct columns added
    """
    df = df.copy()
    n = len(df)

    # Initialize new columns
    df['Trigger'] = False
    df['Max_Pct'] = 0.0

    # Track which windows trigger each index
    trigger_windows = {}  # Maps trigger index -> list of window start indices

    # Scan for 2% price movement windows
    for i in range(n - window_size + 1):
        window_high = df.loc[i:i + window_size - 1, 'high'].max()
        window_low = df.loc[i:i + window_size - 1, 'low'].min()

        price_diff_pct = ((window_high - window_low) / window_low) * 100

        if price_diff_pct >= threshold_pct:
            # Set Trigger=True for 3 steps before window starts
            for offset in range(1, 4):  # i-3, i-2, i-1
                trigger_idx = i - offset
                if trigger_idx >= 0:
                    df.loc[trigger_idx, 'Trigger'] = True

                    # Track which window this trigger relates to
                    if trigger_idx not in trigger_windows:
                        trigger_windows[trigger_idx] = []
                    trigger_windows[trigger_idx].append(i)

    # Calculate Max_Pct for each Trigger=True row
    for trigger_idx, window_starts in trigger_windows.items():
        # Use the nearest upcoming window (first window start)
        window_start = min(window_starts)
        window_end = window_start + window_size - 1

        # Find max close in the window
        max_close = df.loc[window_start:window_end, 'close'].max()
        current_close = df.loc[trigger_idx, 'close']

        # Calculate percentage difference
        max_pct = ((max_close - current_close) / current_close) * 100
        df.loc[trigger_idx, 'Max_Pct'] = max_pct

    return df


def process_file(input_path: Path, output_path: Path) -> dict:
    """
    Process a single file: resample and add Trigger/Max_Pct columns.

    Returns:
        Dictionary with processing statistics
    """
    print(f"Processing: {input_path.name}")

    # Load data
    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Resample 5min -> 15min
    df_15min = resample_5min_to_15min(df)
    resampled_rows = len(df_15min)

    # Detect triggers and calculate Max_Pct
    df_flagged = detect_triggers_and_max_pct(df_15min)

    # Count triggers
    trigger_count = df_flagged['Trigger'].sum()

    # Save output
    df_flagged.to_csv(output_path, index=False)

    stats = {
        'input_file': input_path.name,
        'output_file': output_path.name,
        'original_rows': original_rows,
        'resampled_rows': resampled_rows,
        'trigger_count': trigger_count,
    }

    print(f"  - Original rows: {original_rows}")
    print(f"  - Resampled rows: {resampled_rows}")
    print(f"  - Triggers found: {trigger_count}")
    print(f"  - Saved to: {output_path.name}")

    return stats


def main():
    # Define paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / 'data'
    output_dir = base_dir / 'data_flagged'

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Files to process (5-minute data only)
    input_files = [
        'BTCUSDT_perp_etf_to_90d_ago.csv',
        'BTCUSDT_perp_last_90d.csv',
        'BTCUSDT_spot_etf_to_90d_ago.csv',
        'BTCUSDT_spot_last_90d.csv',
    ]

    all_stats = []

    for filename in input_files:
        input_path = input_dir / filename

        if not input_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        # Create output filename with _15m_flagged suffix
        output_filename = filename.replace('.csv', '_15m_flagged.csv')
        output_path = output_dir / output_filename

        stats = process_file(input_path, output_path)
        all_stats.append(stats)
        print()

    # Print summary
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    for stats in all_stats:
        print(f"{stats['input_file']}: {stats['trigger_count']} triggers")


if __name__ == '__main__':
    main()
