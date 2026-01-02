"""
Backtest Data Loader

Loads and aligns multi-timeframe data for backtesting.
CRITICAL: Implements strict temporal alignment to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from .config import BacktestConfig


class BacktestDataLoader:
    """
    Loads and aligns multi-timeframe data for backtesting.

    CRITICAL: Implements strict temporal alignment to prevent look-ahead bias.
    """

    OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.df_5m: Optional[pd.DataFrame] = None
        self.df_1h: Optional[pd.DataFrame] = None
        self.hour_to_idx: Dict[pd.Timestamp, int] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess both timeframes.

        Returns:
            Tuple of (df_5m, df_1h)
        """
        # Load 5-minute data
        self.df_5m = pd.read_csv(self.config.data_5m_path)
        self.df_5m['open_time'] = pd.to_datetime(self.df_5m['open_time'])
        self.df_5m = self.df_5m.sort_values('open_time').reset_index(drop=True)

        # Load 1-hour data
        self.df_1h = pd.read_csv(self.config.data_1h_path)
        self.df_1h['open_time'] = pd.to_datetime(self.df_1h['open_time'])
        self.df_1h = self.df_1h.sort_values('open_time').reset_index(drop=True)

        # Build hour-to-index mapping for fast lookup
        for idx, row in self.df_1h.iterrows():
            hour_start = row['open_time'].floor('H')
            self.hour_to_idx[hour_start] = idx

        print(f"Loaded 5m data: {len(self.df_5m)} rows")
        print(f"  Date range: {self.df_5m['open_time'].min()} to {self.df_5m['open_time'].max()}")
        print(f"Loaded 1h data: {len(self.df_1h)} rows")
        print(f"  Date range: {self.df_1h['open_time'].min()} to {self.df_1h['open_time'].max()}")

        return self.df_5m, self.df_1h

    def get_warmup_end_idx(self) -> int:
        """
        Return first valid index after warmup period.

        Needs enough history for:
        - 5m sequence (seq_len_5m)
        - TDA window (tda_window_size)
        - 1h sequence alignment
        """
        return max(
            self.config.seq_len_5m,
            self.config.tda_window_size,
            self.config.micro_lookback
        ) + 1

    def _get_aligned_1h_idx(self, idx_5m: int) -> Optional[int]:
        """
        Get the 1h candle index that was CLOSED at or before the 5m candle.

        CRITICAL FOR NO LOOK-AHEAD: Only returns completed 1h candles.

        Example:
        - 5m candle at 10:25 → can only use 1h candles up to 09:00-10:00 (closed at 10:00)
        - Cannot use 10:00-11:00 candle as it's still forming

        Args:
            idx_5m: 5-minute candle index

        Returns:
            1h candle index or None if insufficient history
        """
        current_5m_time = self.df_5m.iloc[idx_5m]['open_time']

        # Find the hour that's COMPLETED (not currently forming)
        # Current 5m candle is at time T, the latest completed 1h candle
        # ended at the start of the current hour
        current_hour = current_5m_time.floor('H')

        # The last COMPLETED 1h candle started at (current_hour - 1 hour)
        # because the candle starting at current_hour is still forming
        last_completed_hour = current_hour - pd.Timedelta(hours=1)

        if last_completed_hour not in self.hour_to_idx:
            return None

        return self.hour_to_idx[last_completed_hour]

    def get_sequences_at(self, idx_5m: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get OHLCV sequences ending at idx_5m.

        CRITICAL: Only uses COMPLETED candles for prediction.
        - 5m data: candles [idx - seq_len_5m, idx) (not including idx)
        - 1h data: candles from hours that are FULLY CLOSED before idx's hour

        The current candle (idx) provides the entry price if we decide to trade.

        Args:
            idx_5m: Current 5-minute candle index

        Returns:
            Dictionary with x_5m, x_1h, current_price or None if insufficient data
        """
        if self.df_5m is None or self.df_1h is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Validate sufficient history for 5m
        if idx_5m < self.config.seq_len_5m:
            return None

        # Get 5m sequence (EXCLUDING current candle - use for prediction)
        start_5m = idx_5m - self.config.seq_len_5m
        x_5m = self.df_5m.iloc[start_5m:idx_5m][self.OHLCV_COLS].values.astype(np.float32)

        # Get aligned 1h index
        hour_idx = self._get_aligned_1h_idx(idx_5m)
        if hour_idx is None:
            return None

        # Validate sufficient history for 1h
        if hour_idx < self.config.seq_len_1h - 1:
            return None

        # Get 1h sequence (all completed hours)
        start_1h = hour_idx - self.config.seq_len_1h + 1
        x_1h = self.df_1h.iloc[start_1h:hour_idx + 1][self.OHLCV_COLS].values.astype(np.float32)

        # Current candle data (for entry price and TP/SL check)
        current_row = self.df_5m.iloc[idx_5m]

        return {
            'x_5m': x_5m,
            'x_1h': x_1h,
            'current_price': float(current_row['close']),
            'current_high': float(current_row['high']),
            'current_low': float(current_row['low']),
            'current_time': current_row['open_time'],
        }

    def get_candle_at(self, idx: int) -> Dict:
        """Get candle data at index."""
        row = self.df_5m.iloc[idx]
        return {
            'open_time': row['open_time'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
        }

    def __len__(self) -> int:
        """Return length of 5m data."""
        return len(self.df_5m) if self.df_5m is not None else 0
