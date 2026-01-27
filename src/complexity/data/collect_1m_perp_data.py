"""
Binance 1-minute BTC/USDT Perpetual Futures Data Collector

Downloads 1-minute candle data from Binance Data Vision.
Period: 2024-01-11 (BTC ETF launch) ~ present
Output format: Same as spot data for consistency
"""

import pandas as pd
import requests
import zipfile
import io
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_daily_klines(date: datetime, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Download daily 1-minute klines from Binance Data Vision.

    Args:
        date: Date to download
        symbol: Trading pair symbol

    Returns:
        DataFrame with kline data
    """
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return pd.DataFrame()
        response.raise_for_status()

        # Extract CSV from zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        # Binance Data Vision kline format (with header):
        # open_time, open, high, low, close, volume, close_time,
        # quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
        df = df.rename(columns={
            "count": "trades",
            "taker_buy_volume": "taker_buy_base",
            "taker_buy_quote_volume": "taker_buy_quote"
        })

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote"]:
            df[col] = df[col].astype(float)

        df["trades"] = df["trades"].astype(int)

        # Select relevant columns (same as spot format)
        df = df[["open_time", "open", "high", "low", "close", "volume",
                 "quote_volume", "trades", "taker_buy_base"]]

        # Rename for clarity (same as spot)
        df = df.rename(columns={"taker_buy_base": "buy_volume"})
        df["sell_volume"] = df["volume"] - df["buy_volume"]

        return df

    except Exception as e:
        print(f"  Error downloading {date_str}: {e}")
        return pd.DataFrame()


def collect_historical_data(
    start_date: str = "2024-01-11",
    end_date: str = None,
    save_path: Path = None,
    max_workers: int = 5,
) -> pd.DataFrame:
    """
    Collect historical kline data from Binance Data Vision.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), defaults to yesterday
        save_path: Path to save CSV file
        max_workers: Number of parallel downloads

    Returns:
        DataFrame with all collected data
    """
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now() - timedelta(days=1)

    print(f"Collecting BTCUSDT PERP 1m data from Binance Data Vision")
    print(f"From: {start_dt.date()}")
    print(f"To: {end_dt.date()}")
    print("-" * 50)

    # Generate date list
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)

    total_days = len(dates)
    print(f"Total days to download: {total_days}")

    all_data = []
    completed = 0

    # Download in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(download_daily_klines, d): d for d in dates}

        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                df = future.result()
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"  Error for {date}: {e}")

            completed += 1
            if completed % 50 == 0 or completed == total_days:
                print(f"Progress: {completed}/{total_days} ({100*completed/total_days:.1f}%)")

    if not all_data:
        print("No data collected!")
        return pd.DataFrame()

    # Combine and sort
    print("Combining data...")
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["open_time"])

    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")

    print("-" * 50)
    print(f"Total candles: {len(df):,}")
    print(f"Date range: {df['open_time'].min()} ~ {df['open_time'].max()}")

    return df


def main():
    # Define paths
    data_dir = Path("/home/ubuntu/joo/strat/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Calculate date splits per CLAUDE.md
    today = datetime.now()
    days_90_ago = today - timedelta(days=90)
    yesterday = today - timedelta(days=1)  # Data Vision doesn't have today's data

    # Split dates
    etf_launch = "2024-01-11"
    split_date = days_90_ago.strftime("%Y-%m-%d")
    end_date = yesterday.strftime("%Y-%m-%d")

    print("=" * 60)
    print("BTC/USDT PERPETUAL 1-Minute Data Collection")
    print("(via Binance Data Vision)")
    print("=" * 60)
    print(f"ETF Launch: {etf_launch}")
    print(f"Split Date (90 days ago): {split_date}")
    print(f"End Date: {end_date}")
    print("=" * 60)

    # Collect training + validation data (ETF launch ~ 90 days ago)
    print("\n[1/2] Collecting ETF launch ~ 90 days ago data...")
    collect_historical_data(
        start_date=etf_launch,
        end_date=split_date,
        save_path=data_dir / "BTCUSDT_perp_1m_etf_to_90d_ago.csv",
        max_workers=10,
    )

    # Collect test data (90 days ago ~ yesterday)
    print("\n[2/2] Collecting last 90 days data...")
    collect_historical_data(
        start_date=split_date,
        end_date=end_date,
        save_path=data_dir / "BTCUSDT_perp_1m_last_90d.csv",
        max_workers=10,
    )

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
