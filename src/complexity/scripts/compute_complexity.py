#!/usr/bin/env python3
"""
Complexity Pipeline Script

Computes market complexity using 1-minute granular data within 15-minute intervals
for integration with the TDA model.

Pipeline:
1. Load 1-minute OHLCV data
2. Group into 15-minute intervals
3. Compute complexity on each 15-minute window using all 1-minute candles
   (preserves fine-grained patterns vs. averaging individual 1-min scores)
4. Merge with data_flagged/ files
5. Save updated files

Usage:
    python compute_complexity.py
    python compute_complexity.py --check-only  # Just check data availability
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from complexity import calculate_complexity_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# File mappings: 1-min data -> 15-min flagged data
FILE_MAPPINGS = {
    "BTCUSDT_spot_1m_etf_to_90d_ago.csv": "BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv",
    "BTCUSDT_spot_1m_last_90d.csv": "BTCUSDT_spot_last_90d_15m_flagged.csv",
}


def check_data_availability(data_dir: Path, flagged_dir: Path) -> dict:
    """Check which data files are available."""
    status = {}

    for src_file, dst_file in FILE_MAPPINGS.items():
        src_path = data_dir / src_file
        dst_path = flagged_dir / dst_file

        status[src_file] = {
            "1m_exists": src_path.exists(),
            "1m_path": src_path,
            "15m_exists": dst_path.exists(),
            "15m_path": dst_path,
            "dst_file": dst_file,
        }

    return status


def load_1m_data(file_path: Path) -> pd.DataFrame:
    """Load 1-minute OHLCV data."""
    logger.info(f"Loading 1-minute data from: {file_path}")

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


def compute_complexity_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute complexity for each 15-minute interval using all 1-minute data within that interval.

    This preserves fine-grained patterns by calculating complexity on the full 15-minute window
    rather than averaging individual 1-minute complexity scores.
    """
    logger.info("Computing complexity on 15-minute intervals (using 1-minute granularity)...")

    # Create 15-minute time groups
    df = df.copy()
    df['time_group'] = df['open_time'].dt.floor('15min')

    complexity_results = []
    total_groups = df['time_group'].nunique()

    for i, (timestamp, group) in enumerate(df.groupby('time_group'), 1):
        if i % 100 == 0:
            logger.info(f"  Processing group {i}/{total_groups}...")

        # Calculate complexity on the entire 15-minute window (15 x 1-minute candles)
        group_indexed = group.set_index('open_time')
        indicators, complexity_score = calculate_complexity_score(group_indexed)

        # Use the mean complexity across the 15-minute window
        # (could also use .iloc[-1] for the final value)
        complexity_results.append({
            'open_time': timestamp,
            'complexity': complexity_score.mean() if not complexity_score.isna().all() else 0.5
        })

    df_complexity = pd.DataFrame(complexity_results)

    # Log statistics
    logger.info(f"  Computed {len(df_complexity):,} 15-minute intervals")
    logger.info(f"  Complexity stats: min={df_complexity['complexity'].min():.3f}, max={df_complexity['complexity'].max():.3f}, mean={df_complexity['complexity'].mean():.3f}")
    logger.info(f"  NaN count: {df_complexity['complexity'].isna().sum()}")

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
    data_1m_path: Path,
    flagged_15m_path: Path,
) -> bool:
    """Process a single pair of 1-min and 15-min files."""
    try:
        # Step 1: Load 1-min data
        df_1m = load_1m_data(data_1m_path)

        # Step 2: Compute complexity on 15-min intervals using all 1-min data
        complexity_15m = compute_complexity_15m(df_1m)

        # Step 3: Merge with flagged data
        merge_with_flagged(
            flagged_15m_path,
            complexity_15m,
            output_path=flagged_15m_path,  # Overwrite original
        )

        return True

    except Exception as e:
        logger.error(f"Failed to process: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Compute complexity pipeline")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check data availability, don't process",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing 1-min data",
    )
    parser.add_argument(
        "--flagged-dir",
        type=str,
        default=None,
        help="Directory containing flagged 15-min data",
    )
    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    flagged_dir = Path(args.flagged_dir) if args.flagged_dir else project_root / "data"

    logger.info("=" * 60)
    logger.info("Complexity Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Flagged directory: {flagged_dir}")
    logger.info("=" * 60)

    # Check data availability
    status = check_data_availability(data_dir, flagged_dir)

    logger.info("\nData Availability:")
    for src_file, info in status.items():
        logger.info(f"  {src_file}:")
        logger.info(f"    1-min data: {'✓' if info['1m_exists'] else '✗ NOT FOUND'}")
        logger.info(f"    15-min flagged: {'✓' if info['15m_exists'] else '✗ NOT FOUND'}")

    if args.check_only:
        return

    # Process each file pair
    logger.info("\n" + "=" * 60)
    logger.info("Processing Files")
    logger.info("=" * 60)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for src_file, info in status.items():
        logger.info(f"\n--- Processing: {src_file} ---")

        if not info["1m_exists"]:
            logger.warning(f"Skipping: 1-min data not found")
            logger.warning(f"Run 'python collect_1m_data.py' first to collect 1-minute data")
            skip_count += 1
            continue

        if not info["15m_exists"]:
            logger.warning(f"Skipping: 15-min flagged data not found")
            skip_count += 1
            continue

        success = process_file_pair(info["1m_path"], info["15m_path"])

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

    if skip_count > 0 and success_count == 0:
        logger.info("\nTo collect 1-minute data, run:")
        logger.info("  cd src/complexity && python collect_1m_data.py")


if __name__ == "__main__":
    main()
