#!/usr/bin/env python3
"""
Hyperparameter experiments for TDA model.

Runs systematic experiments to find optimal parameters for trading.
Target: Precision > 60% with Recall > 40%

KEY INSIGHT: WeightedRandomSampler was causing distribution mismatch.
- Training with ~50% positives but test only has ~10%
- This led to massive false positives
- Solution: Disable weighted sampler, use focal loss only
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import create_data_loaders
from tda_model.training import create_trainer, create_model_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_experiment(
    config: Config,
    experiment_name: str,
    project_root: Path,
    clear_cache: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment with given config."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"{'='*60}")

    # Clear TDA cache if parameters changed
    if clear_cache:
        cache_dir = project_root / config.data.cache_dir
        if cache_dir.exists():
            for f in cache_dir.glob("*.npy"):
                f.unlink()
            logger.info("Cleared TDA cache")

    # Update save directory for this experiment
    config.logging.save_dir = f"models/experiments/{experiment_name}"
    save_dir = project_root / "src" / "tda_model" / config.logging.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            project_root=project_root,
        )

        # Create model (LSTM or N-BEATS based on config)
        model = create_model_from_config(config)
        trainable, total = model.get_num_parameters()
        logger.info(f"Model parameters: {trainable:,}")

        # Create trainer and train
        trainer = create_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        results = trainer.train()

        # Extract key metrics
        test_metrics = results.get("test_metrics", {})

        experiment_result = {
            "name": experiment_name,
            "config": {
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "dropout": config.model.lstm_dropout,
                "weight_decay": config.training.weight_decay,
                "hidden_size": config.model.lstm_hidden_size,
                "num_layers": config.model.lstm_num_layers,
                "focal_alpha": config.training.focal_alpha,
                "trigger_loss_weight": config.training.trigger_loss_weight,
                "use_weighted_sampler": config.training.use_weighted_sampler,
                "tda_window_size": config.tda.window_size,
                "tda_betti_bins": config.tda.betti_bins,
            },
            "results": {
                "test_precision": test_metrics.get("classification", {}).get("precision", 0),
                "test_recall": test_metrics.get("classification", {}).get("recall", 0),
                "test_f1": test_metrics.get("classification", {}).get("f1", 0),
                "test_auc": test_metrics.get("classification", {}).get("auc_roc", 0),
                "test_accuracy": test_metrics.get("classification", {}).get("accuracy", 0),
                "best_val_f1": results.get("best_val_f1", 0),
                "optimal_threshold": results.get("optimal_threshold", 0.5),
                "total_epochs": results.get("total_epochs", 0),
            },
            "duration_seconds": time.time() - start_time,
        }

        # Save experiment result
        result_file = save_dir / "experiment_result.json"
        with open(result_file, "w") as f:
            json.dump(experiment_result, f, indent=2)

        logger.info(f"\nExperiment {experiment_name} completed:")
        logger.info(f"  Precision: {experiment_result['results']['test_precision']:.4f}")
        logger.info(f"  Recall: {experiment_result['results']['test_recall']:.4f}")
        logger.info(f"  F1: {experiment_result['results']['test_f1']:.4f}")
        logger.info(f"  AUC: {experiment_result['results']['test_auc']:.4f}")

        return experiment_result

    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": experiment_name,
            "error": str(e),
            "duration_seconds": time.time() - start_time,
        }


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments")
    parser.add_argument("--exp", type=str, default="all", help="Experiment to run (1-6 or 'all')")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_results = []

    # =====================================================
    # REVISED EXPERIMENTS (without weighted sampler)
    # =====================================================
    # Key change: use_weighted_sampler=False for all experiments
    # Rely on focal loss for class imbalance handling

    # Experiment 1: Baseline with moderate focal_alpha
    # Focus: Establish baseline without weighted sampler
    if args.exp in ["1", "all"]:
        config = load_config()
        config.training.use_weighted_sampler = False
        config.training.focal_alpha = 0.75
        config.training.trigger_loss_weight = 3.0
        config.model.lstm_dropout = 0.3
        config.training.early_stopping_patience = 15
        result = run_experiment(config, "exp1_baseline_focal75", project_root)
        all_results.append(result)

    # Experiment 2: Higher focal_alpha (more aggressive on minority class)
    if args.exp in ["2", "all"]:
        config = load_config()
        config.training.use_weighted_sampler = False
        config.training.focal_alpha = 0.85
        config.training.trigger_loss_weight = 3.0
        config.model.lstm_dropout = 0.4
        config.training.weight_decay = 0.0005
        result = run_experiment(config, "exp2_focal85", project_root)
        all_results.append(result)

    # Experiment 3: Lower focal_alpha + larger model
    # Hypothesis: More capacity might learn subtle patterns
    if args.exp in ["3", "all"]:
        config = load_config()
        config.training.use_weighted_sampler = False
        config.training.focal_alpha = 0.65
        config.model.lstm_hidden_size = 256
        config.model.lstm_num_layers = 3
        config.model.lstm_dropout = 0.4
        config.training.batch_size = 512
        config.training.weight_decay = 0.001
        result = run_experiment(config, "exp3_large_model_focal65", project_root)
        all_results.append(result)

    # Experiment 4: Very large model (fully utilize GPU)
    # A6000 has 48GB - let's use it
    if args.exp in ["4", "all"]:
        config = load_config()
        config.training.use_weighted_sampler = False
        config.training.focal_alpha = 0.75
        config.model.lstm_hidden_size = 512
        config.model.lstm_num_layers = 3
        config.model.lstm_dropout = 0.5
        config.model.shared_fc_dim = 256
        config.training.batch_size = 1024
        config.training.learning_rate = 0.0005
        config.training.weight_decay = 0.001
        config.training.early_stopping_patience = 20
        result = run_experiment(config, "exp4_very_large_model", project_root)
        all_results.append(result)

    # Experiment 5: Precision-focused (higher trigger_loss_weight)
    if args.exp in ["5", "all"]:
        config = load_config()
        config.training.use_weighted_sampler = False
        config.training.focal_alpha = 0.70
        config.training.trigger_loss_weight = 5.0  # Even more emphasis on classification
        config.model.lstm_hidden_size = 256
        config.model.lstm_dropout = 0.4
        config.training.learning_rate = 0.0005
        config.training.weight_decay = 0.0005
        result = run_experiment(config, "exp5_high_trigger_weight", project_root)
        all_results.append(result)

    # Experiment 6: Conservative prediction (lower focal_alpha)
    # Hypothesis: Model may be too aggressive, try more conservative
    if args.exp in ["6", "all"]:
        config = load_config()
        config.training.use_weighted_sampler = False
        config.training.focal_alpha = 0.55  # Close to balanced
        config.training.trigger_loss_weight = 2.0
        config.model.lstm_hidden_size = 256
        config.model.lstm_dropout = 0.3
        config.training.learning_rate = 0.001
        config.training.batch_size = 512
        result = run_experiment(config, "exp6_conservative", project_root)
        all_results.append(result)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)

    # Sort by F1 score (balance of precision and recall)
    valid_results = [r for r in all_results if "error" not in r]
    valid_results.sort(key=lambda x: x["results"]["test_f1"], reverse=True)

    logger.info(f"{'Experiment':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    logger.info("-"*75)

    for r in valid_results:
        logger.info(
            f"{r['name']:<30} "
            f"{r['results']['test_precision']:>10.4f} "
            f"{r['results']['test_recall']:>10.4f} "
            f"{r['results']['test_f1']:>10.4f} "
            f"{r['results']['test_auc']:>10.4f}"
        )

    # Save summary
    summary_file = project_root / "src" / "tda_model" / "models" / "experiments" / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {summary_file}")

    if valid_results:
        best = valid_results[0]
        logger.info(f"\nBest experiment: {best['name']}")
        logger.info(f"  Precision: {best['results']['test_precision']:.4f}")
        logger.info(f"  Recall: {best['results']['test_recall']:.4f}")
        logger.info(f"  F1: {best['results']['test_f1']:.4f}")

        # Check if meets target
        prec = best['results']['test_precision']
        rec = best['results']['test_recall']
        if prec >= 0.60 and rec >= 0.40:
            logger.info("\n✓ TARGET MET: Precision >= 60%, Recall >= 40%")
        else:
            logger.info(f"\n✗ Target not met. Need Precision >= 60% (got {prec:.1%}), Recall >= 40% (got {rec:.1%})")


if __name__ == "__main__":
    main()
