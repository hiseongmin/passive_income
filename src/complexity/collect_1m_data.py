"""
Binance 1-minute BTC/USDT Data Collector

Collects 1-minute candle data from Binance API.
Period: 2024-01-11 (BTC ETF launch) ~ present
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_time: int = None,
    end_time: int = None,
    limit: int = 1000,
) -> list:
    """
    Fetch klines (candlestick) data from Binance API.

    Args:
        symbol: Trading pair symbol
        interval: Candle interval (1m, 5m, 15m, 1h, etc.)
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        limit: Number of candles to fetch (max 1000)

    Returns:
        List of kline data
    """
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """
    Convert Binance klines to DataFrame.

    Binance kline format:
    [
        open_time, open, high, low, close, volume,
        close_time, quote_volume, trades, taker_buy_base,
        taker_buy_quote, ignore
    ]
    """
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(klines, columns=columns)

    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    df["trades"] = df["trades"].astype(int)

    # Select relevant columns
    df = df[["open_time", "open", "high", "low", "close", "volume",
             "quote_volume", "trades", "taker_buy_base"]]

    # Rename for clarity
    df = df.rename(columns={"taker_buy_base": "buy_volume"})
    df["sell_volume"] = df["volume"] - df["buy_volume"]

    return df


def collect_historical_data(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_date: str = "2024-01-11",
    end_date: str = None,
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Collect historical kline data from Binance.

    Args:
        symbol: Trading pair symbol
        interval: Candle interval
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), defaults to now
        save_path: Path to save CSV file

    Returns:
        DataFrame with all collected data
    """
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Collecting {symbol} {interval} data")
    print(f"From: {start_dt}")
    print(f"To: {end_dt}")
    print("-" * 50)

    all_data = []
    current_start = start_ms

    # Calculate expected total requests for progress
    ms_per_candle = 60 * 1000  # 1 minute in ms
    total_candles = (end_ms - start_ms) // ms_per_candle
    expected_requests = total_candles // 1000 + 1
    request_count = 0

    while current_start < end_ms:
        try:
            klines = fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
                limit=1000,
            )

            if not klines:
                break

            df_batch = klines_to_dataframe(klines)
            all_data.append(df_batch)

            # Move to next batch
            last_time = klines[-1][0]
            current_start = last_time + 1

            request_count += 1

            # Progress update
            if request_count % 50 == 0:
                progress = min(100, (request_count / expected_requests) * 100)
                current_dt = datetime.fromtimestamp(last_time / 1000)
                print(f"Progress: {progress:.1f}% | Current: {current_dt}")

            # Rate limiting (Binance allows 1200 requests/min)
            time.sleep(0.05)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            print("Waiting 10 seconds before retry...")
            time.sleep(10)
            continue

    # Combine all data
    if not all_data:
        print("No data collected!")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    print("-" * 50)
    print(f"Total candles collected: {len(df):,}")
    print(f"Date range: {df['open_time'].min()} ~ {df['open_time'].max()}")

    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")

    return df


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"

    # Calculate date splits per CLAUDE.md
    today = datetime.now()
    days_90_ago = today - timedelta(days=90)

    # Split dates
    etf_launch = "2024-01-11"
    split_date = days_90_ago.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    print("=" * 60)
    print("BTC/USDT 1-Minute Data Collection")
    print("=" * 60)
    print(f"ETF Launch: {etf_launch}")
    print(f"Split Date (90 days ago): {split_date}")
    print(f"End Date: {end_date}")
    print("=" * 60)

    # Collect training + validation data (ETF launch ~ 90 days ago)
    print("\n[1/2] Collecting ETF launch ~ 90 days ago data...")
    collect_historical_data(
        symbol="BTCUSDT",
        interval="1m",
        start_date=etf_launch,
        end_date=split_date,
        save_path=data_dir / "BTCUSDT_spot_1m_etf_to_90d_ago.csv",
    )

    # Collect test data (90 days ago ~ present)
    print("\n[2/2] Collecting last 90 days data...")
    collect_historical_data(
        symbol="BTCUSDT",
        interval="1m",
        start_date=split_date,
        end_date=end_date,
        save_path=data_dir / "BTCUSDT_spot_1m_last_90d.csv",
    )

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
