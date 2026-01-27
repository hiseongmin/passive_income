#!/usr/bin/env python3
"""
Hyperparameter optimization for TDA multi-task model.

Performs grid search or random search over key hyperparameters.

Usage:
    python -m tda_model.scripts.hyperopt
    python -m tda_model.scripts.hyperopt --strategy random --n-trials 50
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Iterator
import random

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import create_data_loaders
from tda_model.training import create_trainer, create_model_from_config


logger = logging.getLogger(__name__)


# Default hyperparameter search space
# Focused on regularization to prevent overfitting
DEFAULT_SEARCH_SPACE = {
    # TDA parameters (reduced for speed)
    "tda.window_size": [168, 336, 504],  # 1.75, 3.5, 5.25 days at 15-min
    "tda.time_delay": [6, 12, 18],

    # Model parameters - focus on regularization
    "model.lstm_hidden_size": [64, 128],
    "model.lstm_num_layers": [1, 2],
    "model.lstm_dropout": [0.3, 0.4, 0.5],  # Higher dropout to prevent overfitting

    # Training parameters
    "training.learning_rate": [0.0005, 0.001, 0.002],
    "training.weight_decay": [0.0001, 0.001, 0.01],  # L2 regularization
    "training.batch_size": [256, 512],
}


def setup_logging(log_dir: Path) -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hyperopt_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(f"Logging to {log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for TDA model",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Base config file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Search strategy",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials for random search",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epochs per trial (reduced for speed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hyperopt_results",
        help="Directory for results",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        help="JSON file with custom search space",
    )

    return parser.parse_args()


def set_nested_attr(config: Config, key: str, value: Any) -> None:
    """Set a nested attribute on config object."""
    parts = key.split(".")
    obj = config

    for part in parts[:-1]:
        obj = getattr(obj, part)

    setattr(obj, parts[-1], value)


def grid_search_iterator(search_space: Dict[str, List]) -> Iterator[Dict[str, Any]]:
    """Generate all combinations for grid search."""
    keys = list(search_space.keys())
    values = list(search_space.values())

    for combo in product(*values):
        yield dict(zip(keys, combo))


def random_search_iterator(
    search_space: Dict[str, List],
    n_trials: int,
) -> Iterator[Dict[str, Any]]:
    """Generate random combinations for random search."""
    keys = list(search_space.keys())

    for _ in range(n_trials):
        combo = {key: random.choice(values) for key, values in search_space.items()}
        yield combo


def run_trial(
    trial_num: int,
    params: Dict[str, Any],
    base_config: Config,
    project_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single hyperparameter trial.

    Returns:
        Dictionary with trial results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial_num}")
    logger.info(f"Parameters: {params}")
    logger.info(f"{'='*60}")

    # Create config copy and apply parameters
    config = Config.from_dict(base_config.to_dict())

    for key, value in params.items():
        set_nested_attr(config, key, value)

    # Update save directory for this trial
    trial_dir = output_dir / f"trial_{trial_num:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config.logging.save_dir = str(trial_dir)

    try:
        # Create data loaders (reuse cache across trials)
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            project_root=project_root,
        )

        # Create model (LSTM or N-BEATS based on config)
        model = create_model_from_config(config)

        # Create trainer
        trainer = create_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        # Train
        results = trainer.train()

        # Compile trial results
        trial_results = {
            "trial_num": trial_num,
            "params": params,
            "best_val_loss": results["best_val_loss"],
            "best_val_f1": results["best_val_f1"],
            "total_epochs": results["total_epochs"],
            "status": "success",
        }

        if results["test_metrics"] is not None:
            trial_results["test_accuracy"] = results["test_metrics"]["cls_accuracy"]
            trial_results["test_f1"] = results["test_metrics"]["cls_f1"]
            trial_results["test_auc"] = results["test_metrics"]["cls_auc_roc"]

        logger.info(f"Trial {trial_num} completed: F1={trial_results['best_val_f1']:.4f}")

    except Exception as e:
        logger.error(f"Trial {trial_num} failed: {e}")
        trial_results = {
            "trial_num": trial_num,
            "params": params,
            "status": "failed",
            "error": str(e),
        }

    # Save trial results
    with open(trial_dir / "results.json", "w") as f:
        json.dump(trial_results, f, indent=2)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trial_results


def main():
    """Main hyperparameter optimization function."""
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    # Load base config
    base_config = load_config(args.config)
    base_config.training.epochs = args.epochs  # Reduce epochs for faster trials

    # Load search space
    if args.search_space:
        with open(args.search_space) as f:
            search_space = json.load(f)
    else:
        search_space = DEFAULT_SEARCH_SPACE

    logger.info(f"Search space: {search_space}")

    # Calculate total trials
    if args.strategy == "grid":
        total_trials = 1
        for values in search_space.values():
            total_trials *= len(values)
        logger.info(f"Grid search: {total_trials} total combinations")
        param_iterator = grid_search_iterator(search_space)
    else:
        total_trials = args.n_trials
        logger.info(f"Random search: {total_trials} trials")
        param_iterator = random_search_iterator(search_space, total_trials)

    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent

    # Run trials
    all_results = []

    for trial_num, params in enumerate(param_iterator, 1):
        logger.info(f"\nStarting trial {trial_num}/{total_trials}")

        result = run_trial(
            trial_num=trial_num,
            params=params,
            base_config=base_config,
            project_root=project_root,
            output_dir=output_dir,
        )

        all_results.append(result)

        # Save intermediate results
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Find best trial
    successful_results = [r for r in all_results if r["status"] == "success"]

    if successful_results:
        best_result = max(successful_results, key=lambda x: x.get("best_val_f1", 0))

        logger.info("\n" + "=" * 60)
        logger.info("Hyperparameter Optimization Complete")
        logger.info("=" * 60)
        logger.info(f"Total trials: {len(all_results)}")
        logger.info(f"Successful: {len(successful_results)}")
        logger.info(f"Failed: {len(all_results) - len(successful_results)}")
        logger.info(f"\nBest trial: {best_result['trial_num']}")
        logger.info(f"Best F1: {best_result['best_val_f1']:.4f}")
        logger.info(f"Best parameters: {best_result['params']}")

        # Save best config
        best_config = Config.from_dict(base_config.to_dict())
        for key, value in best_result["params"].items():
            set_nested_attr(best_config, key, value)

        import yaml
        with open(output_dir / "best_config.yaml", "w") as f:
            yaml.dump(best_config.to_dict(), f, default_flow_style=False)

        logger.info(f"\nBest config saved to {output_dir / 'best_config.yaml'}")
    else:
        logger.error("No successful trials!")


if __name__ == "__main__":
    main()
