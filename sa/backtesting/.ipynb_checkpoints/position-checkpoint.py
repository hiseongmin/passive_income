"""
Position and Trade Classes

Data structures for managing positions and recording trades.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass
class Position:
    """Active position state."""

    direction: str              # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    entry_price: float
    take_profit: float
    stop_loss: float
    position_size: float
    entry_idx: int

    # Signal info
    trigger_prob: float
    imminence: float
    direction_confidence: float

    def check_exit(self, high: float, low: float) -> Optional[Tuple[str, float]]:
        """
        Check if position should exit based on price action.

        IMPORTANT: Check SL first (pessimistic assumption).
        If both TP and SL are hit in same candle, assume SL hit first.

        Args:
            high: Candle high price
            low: Candle low price

        Returns:
            Tuple of (exit_reason, exit_price) or None
        """
        if self.direction == "LONG":
            # For longs: SL is below entry, TP is above
            if low <= self.stop_loss:
                return ("SL", self.stop_loss)
            if high >= self.take_profit:
                return ("TP", self.take_profit)
        else:  # SHORT
            # For shorts: SL is above entry, TP is below
            if high >= self.stop_loss:
                return ("SL", self.stop_loss)
            if low <= self.take_profit:
                return ("TP", self.take_profit)

        return None

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - current_price) * self.position_size


@dataclass
class Trade:
    """Completed trade record."""

    trade_id: int
    direction: str              # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float

    # PnL
    pnl: float                  # Realized PnL including fees
    pnl_pct: float              # Percentage return
    fees_paid: float

    # Exit info
    exit_reason: str            # "TP", "SL", "END_OF_DATA"
    duration_candles: int

    # Signal info
    trigger_prob: float
    imminence: float
    direction_confidence: float

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            'trade_id': self.trade_id,
            'direction': self.direction,
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position_size': self.position_size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'fees_paid': self.fees_paid,
            'exit_reason': self.exit_reason,
            'duration_candles': self.duration_candles,
            'trigger_prob': self.trigger_prob,
            'imminence': self.imminence,
            'direction_confidence': self.direction_confidence,
        }
