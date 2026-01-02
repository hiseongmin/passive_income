"""
Market Complexity Indicators

Implements 5 complexity indicators as defined in docs/complexity.md:
1. MA Separation
2. Bollinger Band Width
3. Price Efficiency
4. Support Reaction Strength
5. Directional Result per Time Unit
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_ma_separation(
    df: pd.DataFrame,
    ma_periods: list = [20, 50, 100, 200],
    normalize_window: int = 100,
) -> pd.Series:
    """
    Calculate MA Separation indicator.

    High separation = clear trend (low complexity)
    Convergence/crossing = high complexity

    Args:
        df: DataFrame with 'close' column
        ma_periods: List of MA periods to use
        normalize_window: Rolling window for normalization

    Returns:
        Series with normalized MA separation (0-1, higher = more separated)
    """
    close = df["close"]

    # Calculate MAs
    mas = {}
    for period in ma_periods:
        mas[period] = close.rolling(window=period).mean()

    # Calculate average pairwise distance between MAs
    distances = []
    for i, p1 in enumerate(ma_periods):
        for p2 in ma_periods[i + 1:]:
            dist = (mas[p1] - mas[p2]).abs() / close  # Normalize by price
            distances.append(dist)

    avg_distance = pd.concat(distances, axis=1).mean(axis=1)

    # Normalize to 0-1 range using rolling percentile
    min_val = avg_distance.rolling(window=normalize_window).min()
    max_val = avg_distance.rolling(window=normalize_window).max()
    ma_separation_norm = (avg_distance - min_val) / (max_val - min_val + 1e-8)

    return ma_separation_norm.fillna(0)


def calculate_bb_width(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    normalize_window: int = 100,
) -> pd.Series:
    """
    Calculate Bollinger Band Width indicator.

    Wide bands = low complexity (clear trend)
    Narrow bands = high complexity

    Args:
        df: DataFrame with 'close' column
        period: BB period
        std_dev: Standard deviation multiplier
        normalize_window: Rolling window for normalization

    Returns:
        Series with normalized BB width (0-1, higher = wider)
    """
    close = df["close"]

    # Calculate Bollinger Bands
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)

    # BB Width = (Upper - Lower) / Middle
    bb_width = (upper - lower) / sma

    # Normalize to 0-1 range
    min_val = bb_width.rolling(window=normalize_window).min()
    max_val = bb_width.rolling(window=normalize_window).max()
    bb_width_norm = (bb_width - min_val) / (max_val - min_val + 1e-8)

    return bb_width_norm.fillna(0)


def calculate_price_efficiency(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Calculate Price Efficiency indicator.

    |Net Movement| / Total Movement
    Close to 1 = trending (low complexity)
    Close to 0 = choppy (high complexity)

    Args:
        df: DataFrame with 'close' column
        window: Lookback window

    Returns:
        Series with price efficiency (0-1)
    """
    close = df["close"]

    # Net movement = |close[t] - close[t-window]|
    net_movement = (close - close.shift(window)).abs()

    # Total movement = sum of |close[t] - close[t-1]| over window
    price_changes = close.diff().abs()
    total_movement = price_changes.rolling(window=window).sum()

    # Efficiency ratio
    efficiency = net_movement / (total_movement + 1e-8)

    # Clip to 0-1 range
    efficiency = efficiency.clip(0, 1)

    return efficiency.fillna(0)


def calculate_support_reaction(
    df: pd.DataFrame,
    lookback: int = 20,
    reaction_window: int = 5,
    normalize_window: int = 100,
) -> pd.Series:
    """
    Calculate Support Reaction Strength indicator.

    Strong bounce from support = low complexity
    Weak reaction = high complexity

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        lookback: Window to identify support levels
        reaction_window: Candles to measure bounce after touching support
        normalize_window: Rolling window for normalization

    Returns:
        Series with normalized support reaction (0-1, higher = stronger)
    """
    low = df["low"]
    high = df["high"]
    close = df["close"]

    # Identify support levels (rolling minimum)
    support = low.rolling(window=lookback).min()

    # Check if price touched support (within 0.1%)
    touch_threshold = 0.001
    touched_support = (low - support).abs() / support < touch_threshold

    # Calculate bounce magnitude after touching support
    bounce = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if touched_support.iloc[i]:
            # Look ahead for reaction
            end_idx = min(i + reaction_window, len(df))
            if end_idx > i:
                future_high = high.iloc[i:end_idx].max()
                touch_low = low.iloc[i]
                bounce_pct = (future_high - touch_low) / touch_low
                bounce.iloc[i] = bounce_pct

    # Forward fill for non-touch candles (use last known reaction)
    bounce = bounce.ffill().fillna(0)

    # Normalize to 0-1
    min_val = bounce.rolling(window=normalize_window).min()
    max_val = bounce.rolling(window=normalize_window).max()
    reaction_norm = (bounce - min_val) / (max_val - min_val + 1e-8)

    return reaction_norm.fillna(0)


def calculate_directional_result(
    df: pd.DataFrame,
    window: int = 20,
    normalize_window: int = 100,
) -> pd.Series:
    """
    Calculate Directional Result per Time Unit indicator.

    Large displacement after N candles = low complexity
    Small displacement = high complexity

    Args:
        df: DataFrame with 'close' column
        window: Number of candles to look ahead
        normalize_window: Rolling window for normalization

    Returns:
        Series with normalized directional result (0-1, higher = more directional)
    """
    close = df["close"]

    # Price displacement after N candles (look-back version to avoid future leak)
    displacement = (close - close.shift(window)).abs() / close.shift(window)

    # Normalize to 0-1
    min_val = displacement.rolling(window=normalize_window).min()
    max_val = displacement.rolling(window=normalize_window).max()
    result_norm = (displacement - min_val) / (max_val - min_val + 1e-8)

    return result_norm.fillna(0)


def calculate_complexity_score(
    df: pd.DataFrame,
    weights: dict = None,
    ma_periods: list = [20, 50, 100, 200],
    bb_period: int = 20,
    efficiency_window: int = 20,
    support_lookback: int = 20,
    directional_window: int = 20,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate overall complexity score combining all indicators.

    Complexity formula:
    complexity = w1 * (1 - ma_separation_norm)
               + w2 * (1 - bb_width_norm)
               + w3 * (1 - price_efficiency)
               + w4 * (1 - support_reaction_norm)
               + w5 * (1 - directional_result_norm)

    Args:
        df: DataFrame with OHLC data
        weights: Dict with indicator weights (default: equal 0.2 each)
        ma_periods: MA periods for separation calculation
        bb_period: Bollinger Band period
        efficiency_window: Price efficiency window
        support_lookback: Support level lookback
        directional_window: Directional result window

    Returns:
        Tuple of (DataFrame with all indicators, Series with complexity score)
    """
    # Default equal weights
    if weights is None:
        weights = {
            "ma_separation": 0.2,
            "bb_width": 0.2,
            "price_efficiency": 0.2,
            "support_reaction": 0.2,
            "directional_result": 0.2,
        }

    # Calculate individual indicators
    indicators = pd.DataFrame(index=df.index)

    indicators["ma_separation"] = calculate_ma_separation(df, ma_periods)
    indicators["bb_width"] = calculate_bb_width(df, bb_period)
    indicators["price_efficiency"] = calculate_price_efficiency(df, efficiency_window)
    indicators["support_reaction"] = calculate_support_reaction(df, support_lookback)
    indicators["directional_result"] = calculate_directional_result(df, directional_window)

    # Calculate complexity score (inverted indicators)
    complexity = (
        weights["ma_separation"] * (1 - indicators["ma_separation"])
        + weights["bb_width"] * (1 - indicators["bb_width"])
        + weights["price_efficiency"] * (1 - indicators["price_efficiency"])
        + weights["support_reaction"] * (1 - indicators["support_reaction"])
        + weights["directional_result"] * (1 - indicators["directional_result"])
    )

    return indicators, complexity


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path

    # Try to load existing data
    data_dir = Path(__file__).parent.parent.parent / "data"
    sample_file = data_dir / "BTCUSDT_spot_etf_to_90d_ago.csv"

    if sample_file.exists():
        print(f"Loading sample data from: {sample_file}")
        df = pd.read_csv(sample_file)
        df["open_time"] = pd.to_datetime(df["open_time"])

        # Calculate complexity
        indicators, complexity = calculate_complexity_score(df)

        print("\nIndicator Statistics:")
        print(indicators.describe())

        print("\nComplexity Score Statistics:")
        print(complexity.describe())

        # Show some high/low complexity periods
        df["complexity"] = complexity
        high_complexity = df.nlargest(5, "complexity")[["open_time", "close", "complexity"]]
        low_complexity = df.nsmallest(5, "complexity")[["open_time", "close", "complexity"]]

        print("\nHigh Complexity Periods (top 5):")
        print(high_complexity)

        print("\nLow Complexity Periods (bottom 5):")
        print(low_complexity)
    else:
        print(f"Sample file not found: {sample_file}")
        print("Run collect_1m_data.py first to collect data.")
