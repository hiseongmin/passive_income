import pandas as pd
import numpy as np
from pathlib import Path

# File paths
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("./flagged_data")

FILES = [
    "BTCUSDT_perp_last_90d.csv",
    "BTCUSDT_spot_last_90d.csv",
    "BTCUSDT_perp_etf_to_90d_ago.csv",
    "BTCUSDT_spot_etf_to_90d_ago.csv",
]

def aggregate_5min_to_15min(df):
    """Aggregate 5-minute data to 15-minute data."""
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)

    # Create 15-min group index
    df['group'] = df.index // 3

    # Aggregation rules
    agg_dict = {
        'open_time': 'first',
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

    # Add open_interest if present
    if 'open_interest' in df.columns:
        agg_dict['open_interest'] = 'last'

    df_15min = df.groupby('group').agg(agg_dict).reset_index(drop=True)

    return df_15min


def add_trigger_volatility_columns(df):
    """
    Add Trigger and Volatility columns.

    - Look ahead at timesteps 5-10 (75min to 150min ahead)
    - Trigger: True if volatility > 2% in that window
    - Volatility: Max percentage volatility from current close to any high/low in the window
    """
    df = df.copy()
    n = len(df)

    trigger = np.zeros(n, dtype=bool)
    volatility = np.zeros(n, dtype=float)

    for i in range(n):
        current_close = df.loc[i, 'close']

        # Look ahead at timesteps 5 to 10 (indices i+5 to i+10)
        start_idx = i + 5
        end_idx = min(i + 11, n)  # i+10 inclusive, so i+11 exclusive

        if start_idx >= n:
            # No future data available
            trigger[i] = False
            volatility[i] = 0.0
            continue

        # Get high and low values in the lookahead window
        future_highs = df.loc[start_idx:end_idx-1, 'high'].values
        future_lows = df.loc[start_idx:end_idx-1, 'low'].values

        if len(future_highs) == 0:
            trigger[i] = False
            volatility[i] = 0.0
            continue

        # Calculate max volatility (max deviation from current close)
        max_high = np.max(future_highs)
        min_low = np.min(future_lows)

        upside_volatility = (max_high - current_close) / current_close * 100
        downside_volatility = (current_close - min_low) / current_close * 100

        max_vol = max(upside_volatility, downside_volatility)

        volatility[i] = max_vol
        trigger[i] = max_vol > 2.0

    df['Trigger'] = trigger
    df['Volatility'] = volatility

    return df


def mark_triggers_before_volatility(df):
    """
    Mark 3 timesteps RIGHT BEFORE the volatility window.

    If at row i we detect >2% volatility in window i+5 to i+10,
    then mark rows i+2, i+3, i+4 as Trigger=True (the 3 steps right before i+5).
    """
    df = df.copy()
    n = len(df)

    # Create new columns
    new_trigger = np.zeros(n, dtype=bool)
    new_volatility = np.zeros(n, dtype=float)

    for i in range(n):
        if df.loc[i, 'Trigger']:  # Original detection at row i
            vol = df.loc[i, 'Volatility']

            # Mark 3 timesteps before the volatility window (i+5)
            # That means: i+2, i+3, i+4
            for offset in [2, 3, 4]:
                mark_idx = i + offset
                if mark_idx < n:
                    new_trigger[mark_idx] = True
                    new_volatility[mark_idx] = max(new_volatility[mark_idx], vol)

    df['Trigger'] = new_trigger
    df['Volatility'] = new_volatility

    return df


def process_file(filename):
    """Process a single file."""
    print(f"Processing {filename}...")

    # Read data
    df = pd.read_csv(DATA_DIR / filename)
    print(f"  Original rows: {len(df)}")

    # Aggregate to 15-min
    df_15min = aggregate_5min_to_15min(df)
    print(f"  After 15-min aggregation: {len(df_15min)}")

    # Add trigger and volatility columns
    df_15min = add_trigger_volatility_columns(df_15min)

    # Mark 3 timesteps right before the volatility window
    df_15min = mark_triggers_before_volatility(df_15min)

    # Count triggers
    n_triggers = df_15min['Trigger'].sum()
    print(f"  Number of Trigger=True rows: {n_triggers}")

    # Save to output directory
    output_path = OUTPUT_DIR / filename
    df_15min.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    return df_15min


def main():
    print("=" * 60)
    print("Processing BTCUSDT data: 5min -> 15min with Trigger/Volatility")
    print("=" * 60)

    for filename in FILES:
        process_file(filename)
        print()

    print("All files processed successfully!")


if __name__ == "__main__":
    main()
