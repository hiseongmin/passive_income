"""
Backtest Engine

Main backtesting orchestrator that processes data sequentially
to ensure no look-ahead bias.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm

from .config import BacktestConfig
from .data_loader import BacktestDataLoader
from .feature_pipeline import IncrementalFeatureExtractor
from .signal_generator import SignalGenerator
from .execution import ExecutionEngine
from .portfolio import Portfolio
from .position import Position
from .metrics import BacktestMetrics
from .visualizer import BacktestVisualizer


class BacktestEngine:
    """
    Main backtesting engine orchestrating all components.

    CRITICAL: Processes data sequentially to ensure no look-ahead bias.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Initialize components
        self.data_loader = BacktestDataLoader(config)
        self.feature_extractor = IncrementalFeatureExtractor(config)
        self.signal_generator = SignalGenerator(config)
        self.execution = ExecutionEngine(config)
        self.portfolio = Portfolio(config)
        self.visualizer = BacktestVisualizer()

        # State
        self.is_initialized = False
        self.tda_features: Optional[np.ndarray] = None
        self.micro_features: Optional[np.ndarray] = None

    def initialize(self) -> None:
        """Load data, model, and pre-compute features."""
        print("=" * 60)
        print("INITIALIZING BACKTEST")
        print("=" * 60)

        # Load data
        print("\n[1/3] Loading data...")
        self.df_5m, self.df_1h = self.data_loader.load_data()

        # Load model
        print("\n[2/3] Loading model...")
        self.signal_generator.load_model()

        # Pre-compute features
        print("\n[3/3] Pre-computing features...")
        self.tda_features, self.micro_features = \
            self.feature_extractor.precompute_all_features(self.df_5m)

        self.is_initialized = True
        print("\nInitialization complete!")

    def run(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Run backtest simulation.

        Algorithm:
        1. For each candle from start to end:
           a. Check if current position should exit (TP/SL)
           b. If no position, check for entry signal
           c. Record equity
        2. Force close any open position at end
        3. Calculate metrics

        Args:
            start_idx: Start index (default: after warmup)
            end_idx: End index (default: end of data)
            show_progress: Show progress bar

        Returns:
            Dictionary with results and metrics
        """
        if not self.is_initialized:
            self.initialize()

        # Set indices
        start_idx = start_idx or self.data_loader.get_warmup_end_idx()
        end_idx = end_idx or len(self.df_5m)

        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)
        print(f"  Period: index {start_idx} to {end_idx}")
        print(f"  Candles: {end_idx - start_idx}")
        print(f"  Initial capital: ${self.config.initial_capital:,.2f}")
        print(f"  TP: {self.config.take_profit_pct*100:.1f}%, SL: {self.config.stop_loss_pct*100:.1f}%")
        print(f"  Trigger threshold: {self.config.trigger_threshold}")
        print(f"  Imminence threshold: {self.config.imminence_threshold}")

        # Main loop
        iterator = range(start_idx, end_idx)
        if show_progress:
            iterator = tqdm(iterator, desc="Backtesting")

        for idx in iterator:
            self._process_candle(idx)

        # Force close any open position at end
        if self.portfolio.current_position is not None:
            self._force_close_position(end_idx - 1, "END_OF_DATA")

        # Compile results
        results = self._compile_results()

        return results

    def _process_candle(self, idx: int) -> None:
        """Process a single candle."""
        candle = self.data_loader.get_candle_at(idx)
        timestamp = candle['open_time']
        high, low, close = candle['high'], candle['low'], candle['close']

        # Step 1: Check existing position for exit
        if self.portfolio.current_position is not None:
            exit_result = self.portfolio.current_position.check_exit(high, low)
            if exit_result:
                exit_reason, exit_price = exit_result
                self._close_position(idx, exit_price, exit_reason)

        # Step 2: Generate signal if no position
        if self.portfolio.can_open_position():
            signal = self._generate_signal_at(idx)
            if signal is not None:
                self._open_position(idx, signal)

        # Step 3: Record equity
        self.portfolio.record_equity(timestamp)

    def _generate_signal_at(self, idx: int) -> Optional[Dict]:
        """Generate trading signal at index (no look-ahead)."""
        # Get sequences
        sequences = self.data_loader.get_sequences_at(idx)
        if sequences is None:
            return None

        # Get features
        tda = self.tda_features[idx]
        micro = self.micro_features[idx]

        # Run model inference
        prediction = self.signal_generator.predict(
            sequences['x_5m'],
            sequences['x_1h'],
            tda,
            micro
        )

        # Record signal for analysis
        signal_info = {
            'idx': idx,
            'time': sequences['current_time'],
            'price': sequences['current_price'],
            **prediction
        }
        self.portfolio.record_signal(signal_info)

        # Convert to trading signal
        signal = self.signal_generator.generate_signal(prediction)

        return signal

    def _open_position(self, idx: int, signal: Dict) -> None:
        """Open a new position."""
        candle = self.data_loader.get_candle_at(idx)
        close_price = candle['close']
        timestamp = candle['open_time']

        # Calculate entry with slippage
        entry_price = self.execution.calculate_entry_price(
            close_price, signal['direction']
        )

        # Calculate TP/SL levels
        take_profit, stop_loss = self.execution.calculate_tp_sl_prices(
            entry_price, signal['direction']
        )

        # Calculate position size
        position_size = self.execution.calculate_position_size(
            signal['trigger_prob'],
            signal['imminence']
        )

        # Create position
        position = Position(
            direction=signal['direction'],
            entry_time=timestamp,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_size=position_size,
            entry_idx=idx,
            trigger_prob=signal['trigger_prob'],
            imminence=signal['imminence'],
            direction_confidence=signal['direction_confidence'],
        )

        self.portfolio.open_position(position)

    def _close_position(self, idx: int, exit_price: float, exit_reason: str) -> None:
        """Close current position."""
        candle = self.data_loader.get_candle_at(idx)
        exit_time = candle['open_time']
        pos = self.portfolio.current_position

        # Calculate PnL
        is_limit_exit = exit_reason in ["TP", "SL"]
        pnl, fees = self.execution.calculate_pnl(
            pos.direction,
            pos.entry_price,
            exit_price,
            pos.position_size,
            exit_is_limit=is_limit_exit
        )

        # Close position
        self.portfolio.close_position(
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            fees=fees,
            exit_idx=idx
        )

    def _force_close_position(self, idx: int, reason: str) -> None:
        """Force close position at market price."""
        candle = self.data_loader.get_candle_at(idx)
        close_price = candle['close']

        # Add slippage for market exit
        pos = self.portfolio.current_position
        if pos.direction == "LONG":
            exit_price = close_price * (1 - self.config.slippage_pct)
        else:
            exit_price = close_price * (1 + self.config.slippage_pct)

        self._close_position(idx, exit_price, reason)

    def _compile_results(self) -> Dict:
        """Compile backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        # Calculate metrics
        metrics = BacktestMetrics.calculate_all(
            trades=self.portfolio.trades,
            equity_curve=self.portfolio.equity_curve,
            initial_capital=self.config.initial_capital
        )

        # Print summary
        self._print_summary(metrics)

        # Save results if configured
        if self.config.output_dir:
            self._save_results(metrics)

        return {
            'metrics': metrics,
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'signals': self.portfolio.signals,
            'config': self.config,
        }

    def _print_summary(self, metrics: Dict) -> None:
        """Print summary to console."""
        print(f"\n{'='*40}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*40}")

        print(f"\n[Returns]")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Final Equity: ${metrics['final_equity']:.2f}")

        print(f"\n[Trading]")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Total Fees: ${metrics['total_fees']:.2f}")

        print(f"\n[Risk]")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        print(f"\n[Long/Short]")
        print(f"  Long Trades: {metrics['long_trades']} (Win: {metrics['long_win_rate']*100:.1f}%)")
        print(f"  Short Trades: {metrics['short_trades']} (Win: {metrics['short_win_rate']*100:.1f}%)")

        print(f"\n[Exit Reasons]")
        print(f"  Take Profit: {metrics['tp_exits']} ({metrics['tp_rate']*100:.1f}%)")
        print(f"  Stop Loss: {metrics['sl_exits']} ({metrics['sl_rate']*100:.1f}%)")

    def _save_results(self, metrics: Dict) -> None:
        """Save results to files."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save trades CSV
        if self.config.save_trades_csv and self.portfolio.trades:
            trades_path = os.path.join(self.config.output_dir, 'trades.csv')
            trades_df = self.portfolio.get_trades_df()
            trades_df.to_csv(trades_path, index=False)
            print(f"\nSaved trades to: {trades_path}")

        # Save metrics JSON
        if self.config.save_metrics_json:
            metrics_path = os.path.join(self.config.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"Saved metrics to: {metrics_path}")

        # Save report PNG
        if self.config.save_report_png and self.portfolio.trades:
            report_path = os.path.join(self.config.output_dir, 'backtest_report.png')
            self.visualizer.create_full_report(
                df_5m=self.df_5m,
                trades=self.portfolio.trades,
                equity_curve=self.portfolio.equity_curve,
                metrics=metrics,
                output_path=report_path,
                tda_features=self.tda_features
            )
            print(f"Saved report to: {report_path}")
