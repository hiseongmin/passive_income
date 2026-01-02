#!/usr/bin/env python3
"""
Run Backtest CLI

Command-line interface for running backtests.

Usage:
    python -m backtesting.run_backtest --help
    python -m backtesting.run_backtest --checkpoint /path/to/model.pt
    python -m backtesting.run_backtest --trigger-threshold 0.7 --output-dir ./results
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, '/notebooks/sa')

from backtesting.config import BacktestConfig
from backtesting.backtest_engine import BacktestEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtest for trigger-based trading strategy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument(
        '--data-5m',
        type=str,
        default='/notebooks/data/BTCUSDT_perp_last_90d.csv',
        help='Path to 5-minute OHLCV data'
    )
    parser.add_argument(
        '--data-1h',
        type=str,
        default='/notebooks/data/BTCUSDT_perp_1h_last_90d.csv',
        help='Path to 1-hour OHLCV data'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/notebooks/sa/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )

    # Signal thresholds
    parser.add_argument(
        '--trigger-threshold',
        type=float,
        default=0.6,
        help='Minimum trigger probability for entry'
    )
    parser.add_argument(
        '--imminence-threshold',
        type=float,
        default=0.5,
        help='Minimum imminence score for entry'
    )

    # Trading parameters
    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.02,
        help='Take profit percentage (e.g., 0.02 for 2%%)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.01,
        help='Stop loss percentage (e.g., 0.01 for 1%%)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial capital in USD'
    )

    # Position sizing
    parser.add_argument(
        '--position-size',
        type=float,
        default=1.0,
        help='Base position size'
    )
    parser.add_argument(
        '--no-confidence-sizing',
        action='store_true',
        help='Disable confidence-based position sizing'
    )

    # Fees
    parser.add_argument(
        '--maker-fee',
        type=float,
        default=0.0002,
        help='Maker fee rate (e.g., 0.0002 for 0.02%%)'
    )
    parser.add_argument(
        '--taker-fee',
        type=float,
        default=0.0005,
        help='Taker fee rate (e.g., 0.0005 for 0.05%%)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./backtest_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run model on'
    )

    # Range
    parser.add_argument(
        '--start-idx',
        type=int,
        default=None,
        help='Start index (default: after warmup)'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='End index (default: end of data)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config
    config = BacktestConfig(
        data_5m_path=args.data_5m,
        data_1h_path=args.data_1h,
        model_checkpoint=args.checkpoint,
        trigger_threshold=args.trigger_threshold,
        imminence_threshold=args.imminence_threshold,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        initial_capital=args.initial_capital,
        base_position_size=args.position_size,
        use_confidence_sizing=not args.no_confidence_sizing,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        output_dir=args.output_dir if not args.no_save else None,
        save_trades_csv=not args.no_save,
        save_metrics_json=not args.no_save,
        save_report_png=not args.no_save,
        device=args.device,
    )

    # Print configuration
    print("\n" + "=" * 60)
    print("BACKTEST CONFIGURATION")
    print("=" * 60)
    print(f"  Data 5m: {config.data_5m_path}")
    print(f"  Data 1h: {config.data_1h_path}")
    print(f"  Checkpoint: {config.model_checkpoint}")
    print(f"  Trigger threshold: {config.trigger_threshold}")
    print(f"  Imminence threshold: {config.imminence_threshold}")
    print(f"  Take Profit: {config.take_profit_pct*100:.1f}%")
    print(f"  Stop Loss: {config.stop_loss_pct*100:.1f}%")
    print(f"  Initial capital: ${config.initial_capital:,.2f}")
    print(f"  Confidence sizing: {config.use_confidence_sizing}")
    print(f"  Device: {config.device}")

    # Create and run engine
    engine = BacktestEngine(config)
    results = engine.run(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        show_progress=True
    )

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETED")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
