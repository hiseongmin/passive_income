"""
Portfolio Manager

Manages portfolio state, equity tracking, and trade history.
"""

import pandas as pd
from typing import List, Optional, Tuple
from .config import BacktestConfig
from .position import Position, Trade


class Portfolio:
    """
    Manages portfolio state and equity tracking.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.initial_capital = config.initial_capital
        self.cash = config.initial_capital
        self.current_position: Optional[Position] = None

        # Tracking
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.trades: List[Trade] = []
        self.signals: List[dict] = []  # All signals (entered or not)
        self.trade_counter = 0

    @property
    def equity(self) -> float:
        """Current total equity (cash only, positions close at discrete points)."""
        return self.cash

    @property
    def total_pnl(self) -> float:
        """Total realized PnL."""
        return self.cash - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Total return percentage."""
        return (self.cash / self.initial_capital - 1) * 100

    def can_open_position(self) -> bool:
        """Check if new position can be opened."""
        return self.current_position is None

    def open_position(self, position: Position) -> None:
        """
        Open a new position.

        Args:
            position: Position object to open
        """
        if not self.can_open_position():
            raise ValueError("Cannot open position: position already exists")

        self.current_position = position

    def close_position(
        self,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        fees: float,
        exit_idx: int
    ) -> Trade:
        """
        Close current position and record trade.

        Args:
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: "TP", "SL", or "END_OF_DATA"
            pnl: Realized PnL (after fees)
            fees: Total fees paid
            exit_idx: Exit candle index

        Returns:
            Completed Trade object
        """
        if self.current_position is None:
            raise ValueError("Cannot close position: no position exists")

        pos = self.current_position
        entry_notional = pos.entry_price * pos.position_size

        trade = Trade(
            trade_id=self.trade_counter,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            position_size=pos.position_size,
            pnl=pnl,
            pnl_pct=pnl / entry_notional * 100 if entry_notional > 0 else 0,
            fees_paid=fees,
            exit_reason=exit_reason,
            duration_candles=exit_idx - pos.entry_idx,
            trigger_prob=pos.trigger_prob,
            imminence=pos.imminence,
            direction_confidence=pos.direction_confidence,
        )

        # Update state
        self.cash += pnl
        self.trades.append(trade)
        self.trade_counter += 1
        self.current_position = None

        return trade

    def record_equity(self, timestamp: pd.Timestamp) -> None:
        """Record current equity for equity curve."""
        self.equity_curve.append((timestamp, self.equity))

    def record_signal(self, signal_info: dict) -> None:
        """Record signal information (for analysis)."""
        self.signals.append(signal_info)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.trades])

    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.current_position = None
        self.equity_curve = []
        self.trades = []
        self.signals = []
        self.trade_counter = 0
