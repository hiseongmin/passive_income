#!/usr/bin/env python3
"""
Compute complexity on 5-minute data and resample to 15-minute.

Since Binance API is geo-restricted, we use existing 5-minute data
instead of collecting 1-minute data.

Usage:
    python compute_complexity_5m.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from complexity import calculate_complexity_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# File mappings: 5-min data -> 15-min flagged data
FILE_MAPPINGS = {
    "BTCUSDT_spot_etf_to_90d_ago.csv": "BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv",
    "BTCUSDT_spot_last_90d.csv": "BTCUSDT_spot_last_90d_15m_flagged.csv",
}


def load_5m_data(file_path: Path) -> pd.DataFrame:
    """Load 5-minute OHLCV data."""
    logger.info(f"Loading 5-minute data from: {file_path}")

    df = pd.read_csv(file_path)
    df["open_time"] = pd.to_datetime(df["open_time"])

    # Ensure required columns exist
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df['open_time'].min()} ~ {df['open_time'].max()}")

    return df


def compute_complexity_5m(df: pd.DataFrame) -> pd.Series:
    """
    Compute complexity on 5-minute data.

    Adjusts lookback parameters for 5-min timeframe:
    - Original 1-min: 60 bars = 1 hour
    - For 5-min: 12 bars = 1 hour
    """
    logger.info("Computing complexity on 5-minute data...")

    # Set index for time-based operations
    df_indexed = df.set_index("open_time")

    # Calculate complexity with adjusted parameters for 5-min data
    # Scale: 5-min is 5x longer than 1-min, so divide periods by 5
    # But use minimum sensible values
    indicators, complexity = calculate_complexity_score(
        df_indexed,
        ma_periods=[4, 10, 20, 40],  # ~20/50/100/200 min equivalent
        bb_period=4,  # ~20 min equivalent
        efficiency_window=12,  # ~1 hour
        support_lookback=24,  # ~2 hours
        directional_window=12,  # ~1 hour
        volume_window=12,  # ~1 hour
    )

    # Log statistics
    logger.info(f"  Complexity stats: min={complexity.min():.3f}, max={complexity.max():.3f}, mean={complexity.mean():.3f}")
    logger.info(f"  NaN count: {complexity.isna().sum()}")

    return complexity


def resample_to_15m(complexity_5m: pd.Series) -> pd.DataFrame:
    """Resample 5-minute complexity to 15-minute using mean."""
    logger.info("Resampling to 15-minute intervals (using mean)...")

    # Resample using mean aggregation (3 x 5-min = 15-min)
    complexity_15m = complexity_5m.resample("15min").mean()

    # Convert to DataFrame for merging
    df_complexity = complexity_15m.reset_index()
    df_complexity.columns = ["open_time", "complexity"]

    logger.info(f"  Resampled to {len(df_complexity):,} rows")
    logger.info(f"  Complexity stats: min={df_complexity['complexity'].min():.3f}, max={df_complexity['complexity'].max():.3f}, mean={df_complexity['complexity'].mean():.3f}")

    return df_complexity


def merge_with_flagged(
    flagged_path: Path,
    complexity_15m: pd.DataFrame,
    output_path: Path = None,
) -> pd.DataFrame:
    """Merge complexity scores with flagged data."""
    logger.info(f"Merging with flagged data: {flagged_path}")

    # Load flagged data
    df_flagged = pd.read_csv(flagged_path)
    df_flagged["open_time"] = pd.to_datetime(df_flagged["open_time"])

    original_len = len(df_flagged)

    # Check if complexity column already exists
    if "complexity" in df_flagged.columns:
        logger.warning("  'complexity' column already exists, will be overwritten")
        df_flagged = df_flagged.drop(columns=["complexity"])

    # Merge on open_time
    df_merged = pd.merge(
        df_flagged,
        complexity_15m,
        on="open_time",
        how="left",
    )

    # Check for missing values
    missing_count = df_merged["complexity"].isna().sum()
    if missing_count > 0:
        logger.warning(f"  {missing_count} rows have no complexity match, filling with 0.5")
        df_merged["complexity"] = df_merged["complexity"].fillna(0.5)

    # Verify row count unchanged
    assert len(df_merged) == original_len, "Row count changed after merge!"

    logger.info(f"  Merged successfully: {len(df_merged):,} rows")

    # Save if output path provided
    if output_path:
        df_merged.to_csv(output_path, index=False)
        logger.info(f"  Saved to: {output_path}")

    return df_merged


def process_file_pair(
    data_5m_path: Path,
    flagged_15m_path: Path,
) -> bool:
    """Process a single pair of 5-min and 15-min files."""
    try:
        # Step 1: Load 5-min data
        df_5m = load_5m_data(data_5m_path)

        # Step 2: Compute complexity
        complexity_5m = compute_complexity_5m(df_5m)

        # Step 3: Resample to 15-min
        complexity_15m = resample_to_15m(complexity_5m)

        # Step 4: Merge with flagged data
        merge_with_flagged(
            flagged_15m_path,
            complexity_15m,
            output_path=flagged_15m_path,  # Overwrite original
        )

        return True

    except Exception as e:
        logger.error(f"Failed to process: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Determine paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"

    logger.info("=" * 60)
    logger.info("Complexity Pipeline (5-minute data)")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info("=" * 60)

    # Process each file pair
    success_count = 0
    skip_count = 0
    fail_count = 0

    for src_file, dst_file in FILE_MAPPINGS.items():
        src_path = data_dir / src_file
        dst_path = data_dir / dst_file

        logger.info(f"\n--- Processing: {src_file} ---")

        if not src_path.exists():
            logger.warning(f"Skipping: 5-min data not found at {src_path}")
            skip_count += 1
            continue

        if not dst_path.exists():
            logger.warning(f"Skipping: 15-min flagged data not found at {dst_path}")
            skip_count += 1
            continue

        success = process_file_pair(src_path, dst_path)

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Failed: {fail_count}")


if __name__ == "__main__":
    main()
