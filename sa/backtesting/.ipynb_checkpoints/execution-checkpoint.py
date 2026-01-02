"""
Execution Engine

Simulates order execution with realistic market frictions:
- Slippage
- Trading fees (maker/taker)
- PnL calculation
"""

from dataclasses import dataclass
from typing import Tuple
from .config import BacktestConfig


class ExecutionEngine:
    """
    Simulates order execution with realistic market frictions.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_entry_price(self, market_price: float, direction: str) -> float:
        """
        Calculate actual entry price with slippage.
        Assumes taker order (market order) for entry.

        Slippage is always unfavorable:
        - LONG: Pay higher price
        - SHORT: Receive lower price

        Args:
            market_price: Current market price
            direction: "LONG" or "SHORT"

        Returns:
            Actual entry price after slippage
        """
        slippage = market_price * self.config.slippage_pct

        if direction == "LONG":
            return market_price + slippage
        else:  # SHORT
            return market_price - slippage

    def calculate_tp_sl_prices(
        self,
        entry_price: float,
        direction: str
    ) -> Tuple[float, float]:
        """
        Calculate Take Profit and Stop Loss prices.

        Args:
            entry_price: Entry price
            direction: "LONG" or "SHORT"

        Returns:
            Tuple of (take_profit_price, stop_loss_price)
        """
        if direction == "LONG":
            take_profit = entry_price * (1 + self.config.take_profit_pct)
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - self.config.take_profit_pct)
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)

        return take_profit, stop_loss

    def calculate_fees(self, notional: float, is_maker: bool = False) -> float:
        """
        Calculate trading fees.

        Args:
            notional: Trade notional value
            is_maker: True for limit orders (maker), False for market orders (taker)

        Returns:
            Fee amount
        """
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return notional * fee_rate

    def calculate_pnl(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        exit_is_limit: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate realized PnL and total fees.

        Args:
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            exit_is_limit: True if exit is limit order (TP/SL), False if market

        Returns:
            Tuple of (pnl_after_fees, total_fees)
        """
        entry_notional = position_size * entry_price
        exit_notional = position_size * exit_price

        # Entry fee (taker - market order)
        entry_fee = self.calculate_fees(entry_notional, is_maker=False)

        # Exit fee (maker for TP/SL limit orders, taker for forced exits)
        exit_fee = self.calculate_fees(exit_notional, is_maker=exit_is_limit)

        # Raw PnL
        if direction == "LONG":
            raw_pnl = (exit_price - entry_price) * position_size
        else:  # SHORT
            raw_pnl = (entry_price - exit_price) * position_size

        total_fees = entry_fee + exit_fee
        pnl_after_fees = raw_pnl - total_fees

        return pnl_after_fees, total_fees

    def calculate_position_size(
        self,
        trigger_prob: float,
        imminence: float,
    ) -> float:
        """
        Calculate position size based on signal confidence.

        Args:
            trigger_prob: Trigger probability
            imminence: Imminence score

        Returns:
            Position size
        """
        if self.config.use_confidence_sizing:
            confidence = trigger_prob * imminence
            size = self.config.base_position_size * confidence
            # Clamp to min/max
            size = max(self.config.min_position_size, min(size, self.config.max_position_size))
        else:
            size = self.config.base_position_size

        return size
