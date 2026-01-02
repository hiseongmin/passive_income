"""
Backtest Metrics

Calculates comprehensive trading performance metrics.
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

from .position import Trade


class BacktestMetrics:
    """
    Calculates comprehensive trading metrics.
    """

    @staticmethod
    def calculate_all(
        trades: List[Trade],
        equity_curve: List[Tuple[pd.Timestamp, float]],
        initial_capital: float,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Calculate all performance metrics.

        Args:
            trades: List of completed trades
            equity_curve: List of (timestamp, equity) tuples
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Handle empty trades
        if not trades:
            return BacktestMetrics._empty_metrics(initial_capital)

        # ==================== Basic Metrics ====================
        metrics['total_trades'] = len(trades)
        metrics['winning_trades'] = sum(1 for t in trades if t.pnl > 0)
        metrics['losing_trades'] = sum(1 for t in trades if t.pnl <= 0)
        metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])

        # ==================== PnL Metrics ====================
        pnls = [t.pnl for t in trades]
        metrics['total_pnl'] = sum(pnls)
        metrics['total_return_pct'] = metrics['total_pnl'] / initial_capital * 100
        metrics['avg_trade_pnl'] = np.mean(pnls) if pnls else 0
        metrics['final_equity'] = initial_capital + metrics['total_pnl']

        # Win/Loss analysis
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics['avg_win'] = np.mean(wins) if wins else 0
        metrics['avg_loss'] = np.mean(losses) if losses else 0
        metrics['largest_win'] = max(wins) if wins else 0
        metrics['largest_loss'] = min(losses) if losses else 0

        # ==================== Profit Factor ====================
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001  # Avoid division by zero
        metrics['profit_factor'] = gross_profit / gross_loss

        # ==================== Expectancy ====================
        metrics['expectancy'] = (
            metrics['win_rate'] * metrics['avg_win'] +
            (1 - metrics['win_rate']) * metrics['avg_loss']
        )

        # ==================== Duration ====================
        durations = [t.duration_candles for t in trades]
        metrics['avg_duration_candles'] = np.mean(durations) if durations else 0
        metrics['avg_duration_hours'] = metrics['avg_duration_candles'] * 5 / 60  # 5-min candles
        metrics['min_duration_candles'] = min(durations) if durations else 0
        metrics['max_duration_candles'] = max(durations) if durations else 0

        # ==================== Drawdown ====================
        if equity_curve:
            equity_values = [e[1] for e in equity_curve]
            metrics['max_drawdown_pct'] = BacktestMetrics._calculate_max_drawdown(equity_values)
            metrics['max_drawdown_abs'] = BacktestMetrics._calculate_max_drawdown_abs(equity_values)
        else:
            metrics['max_drawdown_pct'] = 0
            metrics['max_drawdown_abs'] = 0

        # ==================== Risk-Adjusted Returns ====================
        if equity_curve and len(equity_curve) > 1:
            equity_values = [e[1] for e in equity_curve]
            metrics['sharpe_ratio'] = BacktestMetrics._calculate_sharpe(
                equity_values, risk_free_rate
            )
            metrics['sortino_ratio'] = BacktestMetrics._calculate_sortino(
                equity_values, risk_free_rate
            )
        else:
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0

        # ==================== Exit Reason Analysis ====================
        exit_reasons = [t.exit_reason for t in trades]
        metrics['tp_exits'] = exit_reasons.count('TP')
        metrics['sl_exits'] = exit_reasons.count('SL')
        metrics['other_exits'] = len(trades) - metrics['tp_exits'] - metrics['sl_exits']
        metrics['tp_rate'] = metrics['tp_exits'] / max(1, len(trades))
        metrics['sl_rate'] = metrics['sl_exits'] / max(1, len(trades))

        # ==================== Long vs Short Analysis ====================
        long_trades = [t for t in trades if t.direction == 'LONG']
        short_trades = [t for t in trades if t.direction == 'SHORT']

        metrics['long_trades'] = len(long_trades)
        metrics['short_trades'] = len(short_trades)
        metrics['long_win_rate'] = (
            sum(1 for t in long_trades if t.pnl > 0) / max(1, len(long_trades))
        )
        metrics['short_win_rate'] = (
            sum(1 for t in short_trades if t.pnl > 0) / max(1, len(short_trades))
        )
        metrics['long_pnl'] = sum(t.pnl for t in long_trades)
        metrics['short_pnl'] = sum(t.pnl for t in short_trades)

        # ==================== Fee Analysis ====================
        metrics['total_fees'] = sum(t.fees_paid for t in trades)
        metrics['avg_fee_per_trade'] = metrics['total_fees'] / max(1, len(trades))
        metrics['fees_pct_of_pnl'] = (
            abs(metrics['total_fees'] / metrics['total_pnl']) * 100
            if metrics['total_pnl'] != 0 else 0
        )

        # ==================== Signal Quality ====================
        metrics['avg_trigger_prob'] = np.mean([t.trigger_prob for t in trades])
        metrics['avg_imminence'] = np.mean([t.imminence for t in trades])
        metrics['avg_direction_confidence'] = np.mean([t.direction_confidence for t in trades])

        return metrics

    @staticmethod
    def _empty_metrics(initial_capital: float) -> Dict:
        """Return metrics for empty trade list."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_pct': 0,
            'avg_trade_pnl': 0,
            'final_equity': initial_capital,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'avg_duration_candles': 0,
            'avg_duration_hours': 0,
            'min_duration_candles': 0,
            'max_duration_candles': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_abs': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'tp_exits': 0,
            'sl_exits': 0,
            'other_exits': 0,
            'tp_rate': 0,
            'sl_rate': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'long_pnl': 0,
            'short_pnl': 0,
            'total_fees': 0,
            'avg_fee_per_trade': 0,
            'fees_pct_of_pnl': 0,
            'avg_trigger_prob': 0,
            'avg_imminence': 0,
            'avg_direction_confidence': 0,
        }

    @staticmethod
    def _calculate_max_drawdown(equity_values: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_values:
            return 0

        peak = equity_values[0]
        max_dd = 0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd * 100

    @staticmethod
    def _calculate_max_drawdown_abs(equity_values: List[float]) -> float:
        """Calculate maximum drawdown in absolute terms."""
        if not equity_values:
            return 0

        peak = equity_values[0]
        max_dd = 0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = peak - value
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _calculate_sharpe(
        equity_values: List[float],
        risk_free_rate: float
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(equity_values) < 2:
            return 0

        returns = np.diff(equity_values) / np.array(equity_values[:-1])

        if np.std(returns) == 0:
            return 0

        # Annualize (5-min candles: ~105,120 candles per year)
        candles_per_year = 365 * 24 * 12  # 105,120
        excess_return = np.mean(returns) * candles_per_year - risk_free_rate
        vol = np.std(returns) * np.sqrt(candles_per_year)

        return excess_return / vol if vol > 0 else 0

    @staticmethod
    def _calculate_sortino(
        equity_values: List[float],
        risk_free_rate: float
    ) -> float:
        """Calculate annualized Sortino ratio."""
        if len(equity_values) < 2:
            return 0

        returns = np.diff(equity_values) / np.array(equity_values[:-1])
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return 0 if np.mean(returns) <= 0 else float('inf')

        candles_per_year = 365 * 24 * 12
        excess_return = np.mean(returns) * candles_per_year - risk_free_rate
        downside_vol = np.std(negative_returns) * np.sqrt(candles_per_year)

        return excess_return / downside_vol if downside_vol > 0 else 0
