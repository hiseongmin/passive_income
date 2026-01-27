#!/usr/bin/env python3
"""
Trading Signal Generator from Multi-Horizon Return Predictions.

Converts regression model predictions into actionable trading signals by:
1. Loading trained model and generating predictions on test data
2. Applying threshold-based rules to identify trading opportunities
3. Evaluating signal quality through backtesting metrics

Usage:
    python -m scripts.generate_signals
    python -m scripts.generate_signals --upside-threshold 0.02 --risk-ratio 1.5
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import TDADataset
from tda_model.data.preprocessing import load_flagged_data, validate_data
from tda_model.models.regression_model import MultiHorizonRegressionModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate trading signals from regression model predictions"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--upside-threshold", type=float, default=0.015,
        help="Minimum expected upside for BUY signal (default: 1.5%%)"
    )
    parser.add_argument(
        "--risk-ratio", type=float, default=1.5,
        help="Minimum upside/downside ratio (default: 1.5)"
    )
    parser.add_argument(
        "--horizon", type=str, default="4h",
        choices=["1h", "4h", "24h"],
        help="Which horizon to use for signals (default: 4h)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save signal analysis"
    )
    return parser.parse_args()


def load_model(
    model_path: Path,
    config: Config,
    device: str,
) -> MultiHorizonRegressionModel:
    """Load trained regression model."""
    model = MultiHorizonRegressionModel(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_predictions(
    model: MultiHorizonRegressionModel,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions for all samples in loader."""
    all_predictions = []
    all_targets = []

    for batch in loader:
        ohlcv_seq, tda_features, complexity, targets = batch

        ohlcv_seq = ohlcv_seq.to(device)
        tda_features = tda_features.to(device)
        complexity = complexity.to(device)

        predictions = model(ohlcv_seq, tda_features, complexity)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def generate_trading_signals(
    predictions: np.ndarray,
    horizon: str = "4h",
    upside_threshold: float = 0.015,
    risk_ratio: float = 1.5,
) -> np.ndarray:
    """
    Generate trading signals from return predictions.

    Args:
        predictions: Predicted returns of shape (N, 6)
                    [max_1h, max_4h, max_24h, min_1h, min_4h, min_24h]
        horizon: Which horizon to use ("1h", "4h", or "24h")
        upside_threshold: Minimum expected upside for BUY signal
        risk_ratio: Minimum upside/downside ratio

    Returns:
        signals: Array of shape (N,) with values:
                1 = BUY (bullish)
                -1 = SELL (bearish)
                0 = HOLD (no action)
    """
    # Get horizon index
    horizon_idx = {"1h": 0, "4h": 1, "24h": 2}[horizon]

    # Extract max (upside) and min (downside) for the horizon
    max_return = predictions[:, horizon_idx]
    min_return = predictions[:, horizon_idx + 3]

    n_samples = len(predictions)
    signals = np.zeros(n_samples, dtype=np.int32)

    # BUY signal: predicted upside exceeds threshold AND good risk/reward
    upside_condition = max_return > upside_threshold
    # Risk ratio: upside / |downside|
    risk_condition = max_return / (np.abs(min_return) + 1e-8) > risk_ratio
    buy_mask = upside_condition & risk_condition
    signals[buy_mask] = 1

    # SELL signal: predicted downside exceeds threshold AND poor upside
    downside_threshold = -upside_threshold
    downside_condition = min_return < downside_threshold
    inverse_risk = np.abs(min_return) / (max_return + 1e-8) > risk_ratio
    sell_mask = downside_condition & inverse_risk
    signals[sell_mask] = -1

    logger.info(f"Generated signals for horizon={horizon}:")
    logger.info(f"  Upside threshold: {upside_threshold:.2%}")
    logger.info(f"  Risk ratio: {risk_ratio:.1f}")
    logger.info(f"  BUY signals: {np.sum(signals == 1)} ({np.mean(signals == 1):.1%})")
    logger.info(f"  SELL signals: {np.sum(signals == -1)} ({np.mean(signals == -1):.1%})")
    logger.info(f"  HOLD signals: {np.sum(signals == 0)} ({np.mean(signals == 0):.1%})")

    return signals


def evaluate_signals(
    signals: np.ndarray,
    targets: np.ndarray,
    horizon: str = "4h",
) -> Dict[str, float]:
    """
    Evaluate trading signal quality.

    Args:
        signals: Generated signals (1=BUY, -1=SELL, 0=HOLD)
        targets: Actual returns of shape (N, 6)
        horizon: Which horizon was used

    Returns:
        Dictionary of evaluation metrics
    """
    horizon_idx = {"1h": 0, "4h": 1, "24h": 2}[horizon]

    actual_max = targets[:, horizon_idx]
    actual_min = targets[:, horizon_idx + 3]
    # Net return for direction: positive = bullish
    actual_direction = np.sign(actual_max + actual_min)

    # Buy signal evaluation
    buy_mask = signals == 1
    n_buys = np.sum(buy_mask)

    if n_buys > 0:
        # How many BUY signals had positive actual max return?
        buy_profitable = np.sum(actual_max[buy_mask] > 0)
        buy_precision = buy_profitable / n_buys

        # Average actual return when BUY signal triggered
        buy_avg_max = np.mean(actual_max[buy_mask])
        buy_avg_min = np.mean(actual_min[buy_mask])

        # Win rate: how often was direction correct?
        buy_direction_correct = np.sum(actual_direction[buy_mask] > 0)
        buy_win_rate = buy_direction_correct / n_buys
    else:
        buy_precision = 0.0
        buy_avg_max = 0.0
        buy_avg_min = 0.0
        buy_win_rate = 0.0

    # Sell signal evaluation
    sell_mask = signals == -1
    n_sells = np.sum(sell_mask)

    if n_sells > 0:
        # How many SELL signals had negative actual return?
        sell_profitable = np.sum(actual_min[sell_mask] < 0)
        sell_precision = sell_profitable / n_sells

        # Average actual return when SELL signal triggered
        sell_avg_max = np.mean(actual_max[sell_mask])
        sell_avg_min = np.mean(actual_min[sell_mask])

        # Win rate
        sell_direction_correct = np.sum(actual_direction[sell_mask] < 0)
        sell_win_rate = sell_direction_correct / n_sells
    else:
        sell_precision = 0.0
        sell_avg_max = 0.0
        sell_avg_min = 0.0
        sell_win_rate = 0.0

    # Overall metrics
    all_signals_mask = signals != 0
    n_signals = np.sum(all_signals_mask)

    if n_signals > 0:
        # Direction match: did signal direction match actual direction?
        correct_direction = (
            (signals[all_signals_mask] > 0) & (actual_direction[all_signals_mask] > 0)
        ) | (
            (signals[all_signals_mask] < 0) & (actual_direction[all_signals_mask] < 0)
        )
        overall_accuracy = np.mean(correct_direction)
    else:
        overall_accuracy = 0.0

    # Simulated P&L (simplified)
    # Assume: enter at signal, exit at end of horizon
    # Use midpoint of max and min as approximate return
    approx_returns = (actual_max + actual_min) / 2
    pnl = np.sum(signals * approx_returns)
    avg_pnl_per_trade = pnl / n_signals if n_signals > 0 else 0.0

    metrics = {
        "n_buys": int(n_buys),
        "n_sells": int(n_sells),
        "n_holds": int(np.sum(signals == 0)),
        "buy_precision": float(buy_precision),
        "buy_win_rate": float(buy_win_rate),
        "buy_avg_max_return": float(buy_avg_max),
        "buy_avg_min_return": float(buy_avg_min),
        "sell_precision": float(sell_precision),
        "sell_win_rate": float(sell_win_rate),
        "sell_avg_max_return": float(sell_avg_max),
        "sell_avg_min_return": float(sell_avg_min),
        "overall_signal_accuracy": float(overall_accuracy),
        "total_pnl": float(pnl),
        "avg_pnl_per_trade": float(avg_pnl_per_trade),
    }

    return metrics


def main():
    """Main signal generation function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent
    logger.info(f"Project root: {project_root}")

    # Model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = project_root / "src" / "tda_model" / "models" / "regression_model" / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "src" / "tda_model" / "models" / "regression_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(model_path, config, device)
    logger.info("Model loaded successfully")

    # Load test data
    logger.info("Loading test data...")
    test_df = load_flagged_data(
        data_dir=config.data.data_dir,
        filename=config.data.test_file,
        project_root=project_root,
    )
    validate_data(test_df)

    # Cache directory
    cache_dir = project_root / config.data.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create test dataset
    test_dataset = TDADataset(
        df=test_df,
        config=config,
        cache_dir=cache_dir,
        split="test",
        precompute_tda=True,
        mode="regression",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions, targets = generate_predictions(model, test_loader, device)
    logger.info(f"Generated {len(predictions)} predictions")

    # Prediction statistics
    horizon_names = ["1h", "4h", "24h"]
    logger.info("\nPrediction Statistics:")
    for i, name in enumerate(horizon_names):
        pred_max = predictions[:, i]
        pred_min = predictions[:, i + 3]
        actual_max = targets[:, i]
        actual_min = targets[:, i + 3]

        logger.info(f"  {name}:")
        logger.info(f"    Predicted max: mean={np.mean(pred_max):.4f}, std={np.std(pred_max):.4f}")
        logger.info(f"    Predicted min: mean={np.mean(pred_min):.4f}, std={np.std(pred_min):.4f}")
        logger.info(f"    Actual max:    mean={np.mean(actual_max):.4f}, std={np.std(actual_max):.4f}")
        logger.info(f"    Actual min:    mean={np.mean(actual_min):.4f}, std={np.std(actual_min):.4f}")

    # Generate signals
    logger.info(f"\nGenerating trading signals using {args.horizon} horizon...")
    signals = generate_trading_signals(
        predictions=predictions,
        horizon=args.horizon,
        upside_threshold=args.upside_threshold,
        risk_ratio=args.risk_ratio,
    )

    # Evaluate signals
    logger.info("\nEvaluating signal quality...")
    metrics = evaluate_signals(signals, targets, args.horizon)

    logger.info("\n" + "=" * 60)
    logger.info("SIGNAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Horizon: {args.horizon}")
    logger.info(f"  Upside threshold: {args.upside_threshold:.2%}")
    logger.info(f"  Risk ratio: {args.risk_ratio:.1f}")
    logger.info(f"\nSignal Distribution:")
    logger.info(f"  BUY:  {metrics['n_buys']:5d} ({metrics['n_buys']/len(signals):.1%})")
    logger.info(f"  SELL: {metrics['n_sells']:5d} ({metrics['n_sells']/len(signals):.1%})")
    logger.info(f"  HOLD: {metrics['n_holds']:5d} ({metrics['n_holds']/len(signals):.1%})")
    logger.info(f"\nBUY Signal Quality:")
    logger.info(f"  Precision (actual max > 0): {metrics['buy_precision']:.1%}")
    logger.info(f"  Win Rate (correct direction): {metrics['buy_win_rate']:.1%}")
    logger.info(f"  Avg Max Return: {metrics['buy_avg_max_return']:.2%}")
    logger.info(f"  Avg Min Return: {metrics['buy_avg_min_return']:.2%}")
    logger.info(f"\nSELL Signal Quality:")
    logger.info(f"  Precision (actual min < 0): {metrics['sell_precision']:.1%}")
    logger.info(f"  Win Rate (correct direction): {metrics['sell_win_rate']:.1%}")
    logger.info(f"  Avg Max Return: {metrics['sell_avg_max_return']:.2%}")
    logger.info(f"  Avg Min Return: {metrics['sell_avg_min_return']:.2%}")
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Signal Accuracy: {metrics['overall_signal_accuracy']:.1%}")
    logger.info(f"  Total PnL (simulated): {metrics['total_pnl']:.2%}")
    logger.info(f"  Avg PnL per trade: {metrics['avg_pnl_per_trade']:.4%}")

    # Save results
    results = {
        "config": {
            "horizon": args.horizon,
            "upside_threshold": args.upside_threshold,
            "risk_ratio": args.risk_ratio,
            "model_path": str(model_path),
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = output_dir / f"signal_analysis_{args.horizon}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Save predictions for further analysis
    np.savez(
        output_dir / f"predictions_{args.horizon}.npz",
        predictions=predictions,
        targets=targets,
        signals=signals,
    )
    logger.info(f"Predictions saved to: {output_dir / f'predictions_{args.horizon}.npz'}")


if __name__ == "__main__":
    main()
