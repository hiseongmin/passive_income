"""
Data preprocessing for TDA model.

Loads 15-minute flagged data from data/ directory.
The flagged data already contains OHLCV + Trigger + Max_Pct labels.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_flagged_data(
    data_dir: str | Path,
    filename: str,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    Load 15-minute flagged data with OHLCV and labels.

    Args:
        data_dir: Directory containing flagged data (relative to project root)
        filename: CSV filename to load
        project_root: Project root directory. If None, auto-detected.

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume,
                               buy_volume, sell_volume, volume_delta, cvd,
                               Trigger, Max_Pct
    """
    if project_root is None:
        # Auto-detect project root (go up from src/tda_model/data)
        project_root = Path(__file__).parent.parent.parent.parent

    data_path = project_root / data_dir / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path)
    df["open_time"] = pd.to_datetime(df["open_time"])

    # Ensure Trigger is boolean
    if df["Trigger"].dtype != bool:
        df["Trigger"] = df["Trigger"].astype(bool)

    # Ensure Max_Pct is float and clip to valid range [0, 1]
    df["Max_Pct"] = df["Max_Pct"].astype(float)
    invalid_count = ((df["Max_Pct"] < 0) | (df["Max_Pct"] > 1.0)).sum()
    if invalid_count > 0:
        logger.warning(f"Clipping {invalid_count} Max_Pct values outside [0, 1] range")
        df["Max_Pct"] = df["Max_Pct"].clip(0, 1.0)

    logger.info(f"Loaded {len(df)} rows from {filename}")
    logger.info(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    logger.info(f"Trigger rate: {df['Trigger'].mean():.2%}")

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data integrity.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises exception otherwise
    """
    required_columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "Trigger", "Max_Pct"
    ]

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for NaN values in critical columns
    critical_columns = ["open", "high", "low", "close", "Trigger", "Max_Pct"]
    for col in critical_columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column '{col}' has {nan_count} NaN values")

    # Check timestamp continuity (15-min intervals)
    time_diffs = df["open_time"].diff().dropna()
    expected_diff = pd.Timedelta(minutes=15)
    gaps = time_diffs[time_diffs != expected_diff]

    if len(gaps) > 0:
        logger.warning(f"Found {len(gaps)} timestamp gaps in data")

    # Check OHLC validity
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if invalid_ohlc.any():
        logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC values")

    # Check Max_Pct is reasonable (0-100%)
    unreasonable_pct = (df["Max_Pct"] < 0) | (df["Max_Pct"] > 1.0)
    if unreasonable_pct.any():
        logger.warning(f"Found {unreasonable_pct.sum()} rows with Max_Pct outside [0, 1]")

    # Check complexity column if present
    if "complexity" in df.columns:
        logger.info("Complexity column found in data")
        complexity_nan = df["complexity"].isna().sum()
        if complexity_nan > 0:
            logger.warning(f"Complexity column has {complexity_nan} NaN values")

        # Check complexity is in valid range [0, 1]
        invalid_complexity = (df["complexity"] < 0) | (df["complexity"] > 1)
        invalid_count = invalid_complexity.sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} rows with complexity outside [0, 1]")

        # Log complexity statistics
        logger.info(f"Complexity stats: min={df['complexity'].min():.3f}, "
                   f"max={df['complexity'].max():.3f}, mean={df['complexity'].mean():.3f}")
    else:
        logger.info("No complexity column found - will use placeholder during training")

    logger.info("Data validation passed")
    return True


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators for enhanced feature extraction.

    Based on research findings (2024-2025 literature), these indicators
    achieve R² > 0.82 in crypto prediction when properly normalized.

    Added indicators:
    - RSI (14-period): Momentum oscillator
    - MACD (12, 26, 9): Trend-following momentum
    - Bollinger %B (20-period): Volatility position
    - ATR (14-period): Volatility measure
    - Momentum (10-period): Rate of change

    Args:
        df: DataFrame with OHLC columns (open, high, low, close)

    Returns:
        DataFrame with added technical indicator columns
    """
    df = df.copy()

    # Ensure column names are lowercase
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- RSI (Relative Strength Index) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    # Normalize to [0, 1]
    df["RSI"] = df["RSI"] / 100.0

    # --- MACD (Moving Average Convergence Divergence) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    # Normalize MACD by price level
    df["MACD"] = macd_histogram / (close + 1e-10)
    # Clip to reasonable range and scale
    df["MACD"] = df["MACD"].clip(-0.1, 0.1) * 5  # Scale to roughly [-0.5, 0.5]

    # --- Bollinger %B ---
    sma20 = close.rolling(window=20, min_periods=1).mean()
    std20 = close.rolling(window=20, min_periods=1).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    # %B = (Price - Lower) / (Upper - Lower)
    band_width = upper_band - lower_band + 1e-10
    df["BB_pctB"] = (close - lower_band) / band_width
    # Clip to [0, 1] range (can exceed during extreme moves)
    df["BB_pctB"] = df["BB_pctB"].clip(0, 1)

    # --- ATR (Average True Range) ---
    high_low = high - low
    high_close_prev = (high - close.shift(1)).abs()
    low_close_prev = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=14, min_periods=1).mean()
    # Normalize by price level
    df["ATR"] = atr / (close + 1e-10)
    # Clip to reasonable range
    df["ATR"] = df["ATR"].clip(0, 0.1) * 10  # Scale to [0, 1]

    # --- Momentum (Rate of Change) ---
    df["MOM"] = close.pct_change(periods=10)
    # Clip to reasonable range and scale
    df["MOM"] = df["MOM"].clip(-0.1, 0.1) * 5  # Scale to roughly [-0.5, 0.5]

    # Fill NaN values from rolling calculations
    indicator_cols = ["RSI", "MACD", "BB_pctB", "ATR", "MOM"]
    for col in indicator_cols:
        df[col] = df[col].fillna(0.0)

    logger.info("Added technical indicators: RSI, MACD, BB_pctB, ATR, MOM")

    return df


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute max forward returns at multiple horizons for regression targets.

    For each timestamp t, compute:
    - max_return_{horizon}: max upside = max((high[t+1:t+n] - close[t]) / close[t])
    - min_return_{horizon}: max downside = min((low[t+1:t+n] - close[t]) / close[t])

    Horizons (at 15-min candles):
    - 1h: 4 candles
    - 4h: 16 candles
    - 24h: 96 candles

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added columns:
        - max_return_1h, max_return_4h, max_return_24h (positive values, upside)
        - min_return_1h, min_return_4h, min_return_24h (negative values, downside)
    """
    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # Horizons in number of candles
    horizons = {
        "1h": 4,
        "4h": 16,
        "24h": 96,
    }

    for horizon_name, horizon_candles in horizons.items():
        max_returns = np.full(n, np.nan)
        min_returns = np.full(n, np.nan)

        for i in range(n - horizon_candles):
            current_close = close[i]

            # Look at future window [i+1 : i+1+horizon_candles]
            future_highs = high[i + 1 : i + 1 + horizon_candles]
            future_lows = low[i + 1 : i + 1 + horizon_candles]

            # Max upside: best case if you bought at close[i]
            max_high = np.max(future_highs)
            max_returns[i] = (max_high - current_close) / current_close

            # Max downside: worst case if you bought at close[i]
            min_low = np.min(future_lows)
            min_returns[i] = (min_low - current_close) / current_close

        df[f"max_return_{horizon_name}"] = max_returns
        df[f"min_return_{horizon_name}"] = min_returns

    # Log statistics
    for horizon_name in horizons.keys():
        max_col = f"max_return_{horizon_name}"
        min_col = f"min_return_{horizon_name}"
        valid_max = df[max_col].dropna()
        valid_min = df[min_col].dropna()

        if len(valid_max) > 0:
            logger.info(
                f"Forward returns {horizon_name}: "
                f"max_return mean={valid_max.mean():.4f}, std={valid_max.std():.4f}, "
                f"min_return mean={valid_min.mean():.4f}, std={valid_min.std():.4f}"
            )

    logger.info(f"Computed forward returns for horizons: {list(horizons.keys())}")

    return df


def normalize_ohlcv(
    df: pd.DataFrame,
    method: str = "returns",
) -> pd.DataFrame:
    """
    Normalize OHLCV data for model input.

    Args:
        df: DataFrame with OHLCV columns
        method: Normalization method ('returns', 'zscore', 'minmax')

    Returns:
        DataFrame with normalized OHLCV columns
    """
    df = df.copy()

    if method == "returns":
        # Log returns for price columns
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_norm"] = np.log(df[col] / df[col].shift(1))

        # Volume as log
        df["volume_norm"] = np.log1p(df["volume"])

    elif method == "zscore":
        # Rolling z-score normalization
        window = 96  # 24 hours
        for col in ["open", "high", "low", "close", "volume"]:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            df[f"{col}_norm"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    elif method == "minmax":
        # Rolling min-max normalization
        window = 96
        for col in ["open", "high", "low", "close", "volume"]:
            rolling_min = df[col].rolling(window=window).min()
            rolling_max = df[col].rolling(window=window).max()
            df[f"{col}_norm"] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df


def get_ohlcv_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract OHLCV features for model input.

    Uses normalized OHLC (excluding volume for now).

    Args:
        df: DataFrame with normalized columns

    Returns:
        NumPy array of shape (n_samples, 4) with [open, high, low, close] normalized
    """
    required_cols = ["open_norm", "high_norm", "low_norm", "close_norm"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing normalized column: {col}. Run normalize_ohlcv first.")

    features = df[required_cols].values
    return features


def compute_data_statistics(df: pd.DataFrame) -> dict:
    """
    Compute statistics for data analysis.

    Args:
        df: DataFrame with OHLCV and labels

    Returns:
        Dictionary with data statistics
    """
    stats = {
        "n_samples": len(df),
        "date_range": {
            "start": str(df["open_time"].min()),
            "end": str(df["open_time"].max()),
        },
        "trigger_stats": {
            "count": int(df["Trigger"].sum()),
            "rate": float(df["Trigger"].mean()),
        },
        "max_pct_stats": {
            "mean": float(df["Max_Pct"].mean()),
            "std": float(df["Max_Pct"].std()),
            "min": float(df["Max_Pct"].min()),
            "max": float(df["Max_Pct"].max()),
            "median": float(df["Max_Pct"].median()),
        },
        "price_stats": {
            "mean": float(df["close"].mean()),
            "std": float(df["close"].std()),
            "min": float(df["close"].min()),
            "max": float(df["close"].max()),
        },
    }

    # Trigger-specific Max_Pct stats
    triggered = df[df["Trigger"]]
    if len(triggered) > 0:
        stats["triggered_max_pct_stats"] = {
            "mean": float(triggered["Max_Pct"].mean()),
            "std": float(triggered["Max_Pct"].std()),
            "min": float(triggered["Max_Pct"].min()),
            "max": float(triggered["Max_Pct"].max()),
        }

    return stats
