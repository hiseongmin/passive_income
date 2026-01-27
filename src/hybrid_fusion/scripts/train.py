#!/usr/bin/env python3
"""
Main training script for Hybrid Fusion multi-task model.

Usage:
    python -m hybrid_fusion.scripts.train
    python -m hybrid_fusion.scripts.train --config path/to/config.yaml
    python -m hybrid_fusion.scripts.train --epochs 100 --batch-size 1024

From project root:
    python -m src.hybrid_fusion.scripts.train --config src/hybrid_fusion/default_config.yaml
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

import torch

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hybrid_fusion.config import load_config, CompleteModelConfig
from src.hybrid_fusion.model import CompleteHybridFusionModel
from src.hybrid_fusion.training import create_trainer

# Import data loading from tda_model (reuse existing infrastructure)
from src.tda_model.data import create_data_loaders
from src.tda_model.config import Config as TDAConfig, load_config as load_tda_config


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid Fusion multi-task model for Bitcoin trigger prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing flagged data",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early stopping patience",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for fusion",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate",
    )

    # GPU arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile()",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loader workers",
    )

    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def apply_args_to_config(args: argparse.Namespace, config: CompleteModelConfig) -> CompleteModelConfig:
    """Apply command line arguments to config."""
    # Data config
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir

    # Training config
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.early_stopping is not None:
        config.training.early_stopping_patience = args.early_stopping

    # Model config
    if args.hidden_dim is not None:
        config.fusion.hidden_dim = args.hidden_dim
    if args.num_heads is not None:
        config.fusion.num_heads = args.num_heads
    if args.dropout is not None:
        config.fusion.dropout = args.dropout

    # GPU config
    if args.device is not None:
        config.gpu.device = args.device
    if args.no_amp:
        config.gpu.mixed_precision = False
    if args.compile:
        config.gpu.compile_model = True
    if args.num_workers is not None:
        config.gpu.num_workers = args.num_workers

    # Logging config
    if args.save_dir is not None:
        config.logging.save_dir = args.save_dir
    if args.log_dir is not None:
        config.logging.log_dir = args.log_dir

    return config


def print_config(config: CompleteModelConfig, logger: logging.Logger) -> None:
    """Print configuration to logger."""
    logger.info("=" * 60)
    logger.info("Hybrid Fusion Model Configuration:")
    logger.info("=" * 60)

    logger.info("\nN-BEATS Encoder:")
    logger.info(f"  input_dim: {config.nbeats.input_dim}")
    logger.info(f"  seq_length: {config.nbeats.seq_length}")
    logger.info(f"  num_stacks: {config.nbeats.num_stacks}")
    logger.info(f"  hidden_dim: {config.nbeats.hidden_dim}")
    logger.info(f"  output_dim: {config.nbeats.output_dim}")

    logger.info("\nTDA Encoder:")
    logger.info(f"  input_dim: {config.tda.input_dim}")
    logger.info(f"  hidden_dims: {config.tda.hidden_dims}")
    logger.info(f"  output_dim: {config.tda.output_dim}")

    logger.info("\nComplexity Encoder:")
    logger.info(f"  input_dim: {config.complexity.input_dim}")
    logger.info(f"  output_dim: {config.complexity.output_dim}")

    logger.info("\nHybrid Fusion:")
    logger.info(f"  hidden_dim: {config.fusion.hidden_dim}")
    logger.info(f"  regime_dim: {config.fusion.regime_dim}")
    logger.info(f"  num_heads: {config.fusion.num_heads}")
    logger.info(f"  dropout: {config.fusion.dropout}")

    logger.info("\nTraining Parameters:")
    logger.info(f"  batch_size: {config.training.batch_size}")
    logger.info(f"  learning_rate: {config.training.learning_rate}")
    logger.info(f"  epochs: {config.training.epochs}")
    logger.info(f"  early_stopping_patience: {config.training.early_stopping_patience}")
    logger.info(f"  scheduler_type: {config.training.scheduler_type}")
    logger.info(f"  trigger_loss_weight: {config.training.trigger_loss_weight}")
    logger.info(f"  max_pct_loss_weight: {config.training.max_pct_loss_weight}")

    logger.info("\nGPU Configuration:")
    logger.info(f"  device: {config.gpu.device}")
    logger.info(f"  mixed_precision: {config.gpu.mixed_precision}")
    logger.info(f"  compile_model: {config.gpu.compile_model}")
    logger.info(f"  num_workers: {config.gpu.num_workers}")

    logger.info("=" * 60)


def create_model_from_config(config: CompleteModelConfig) -> CompleteHybridFusionModel:
    """
    Create model from configuration.

    Args:
        config: CompleteModelConfig

    Returns:
        CompleteHybridFusionModel instance
    """
    return CompleteHybridFusionModel(
        # N-BEATS encoder config
        ohlcv_input_dim=config.nbeats.input_dim,
        ohlcv_seq_length=config.nbeats.seq_length,
        nbeats_hidden_dim=config.nbeats.hidden_dim,
        nbeats_output_dim=config.nbeats.output_dim,
        nbeats_num_stacks=config.nbeats.num_stacks,
        nbeats_num_blocks=config.nbeats.num_blocks_per_stack,
        nbeats_theta_dim=config.nbeats.theta_dim,
        nbeats_dropout=config.nbeats.dropout,
        # TDA encoder config
        tda_input_dim=config.tda.input_dim,
        tda_hidden_dims=config.tda.hidden_dims,
        tda_output_dim=config.tda.output_dim,
        tda_dropout=config.tda.dropout,
        # Complexity encoder config
        complexity_input_dim=config.complexity.input_dim,
        complexity_hidden_dim=config.complexity.hidden_dim,
        complexity_output_dim=config.complexity.output_dim,
        # Fusion config
        fusion_hidden_dim=config.fusion.hidden_dim,
        fusion_regime_dim=config.fusion.regime_dim,
        fusion_num_heads=config.fusion.num_heads,
        fusion_dropout=config.fusion.dropout,
        # Head config
        head_hidden_dim=config.heads.hidden_dim,
        head_dropout=config.heads.dropout,
    )


def create_tda_config_from_hybrid(config: CompleteModelConfig) -> TDAConfig:
    """
    Create TDA config for data loading compatibility.

    The data loader expects TDAConfig format, so we bridge the configs.
    """
    # Load default TDA config
    tda_config = load_tda_config()

    # Update with hybrid config values
    tda_config.data.data_dir = config.data.data_dir
    tda_config.data.train_file = config.data.train_file
    tda_config.data.test_file = config.data.test_file
    tda_config.data.cache_dir = config.data.cache_dir
    tda_config.data.validation_split = config.data.validation_split
    tda_config.data.use_stratified_split = config.data.use_stratified_split
    tda_config.data.stratified_n_blocks = config.data.stratified_n_blocks

    tda_config.training.batch_size = config.training.batch_size
    tda_config.training.use_weighted_sampler = config.training.use_weighted_sampler

    tda_config.gpu.device = config.gpu.device
    tda_config.gpu.num_workers = config.gpu.num_workers
    tda_config.gpu.pin_memory = config.gpu.pin_memory
    tda_config.gpu.persistent_workers = config.gpu.persistent_workers
    tda_config.gpu.prefetch_factor = config.gpu.prefetch_factor

    return tda_config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)
    config = apply_args_to_config(args, config)

    # Setup logging
    log_dir = Path(config.logging.log_dir)
    setup_logging(log_dir, args.log_level)
    logger = logging.getLogger(__name__)

    # Print config
    print_config(config, logger)

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
        config.gpu.device = "cpu"
        config.gpu.mixed_precision = False

    # Find project root
    project_root = Path(__file__).parent.parent.parent.parent
    logger.info(f"Project root: {project_root}")

    try:
        # Create TDA config for data loading
        tda_config = create_tda_config_from_hybrid(config)

        # Create data loaders (reuse from tda_model)
        logger.info("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config=tda_config,
            project_root=project_root,
        )

        # Create model
        logger.info("\nCreating Hybrid Fusion model...")
        model = create_model_from_config(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

        # Create trainer
        logger.info("\nCreating trainer...")
        trainer = create_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        # Train
        logger.info("\nStarting training...")
        results = trainer.train()

        # Save results
        results_file = Path(config.logging.save_dir) / "training_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {results_file}")

        # Save config
        config_file = Path(config.logging.save_dir) / "config.yaml"
        config.save_yaml(config_file)
        logger.info(f"Config saved to {config_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Summary:")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {results['total_epochs']}")
        logger.info(f"Total time: {results['total_time_seconds']/60:.1f} minutes")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")
        logger.info(f"Optimal threshold: {results['optimal_threshold']:.2f}")

        if results['test_metrics'] is not None:
            logger.info("\nTest Set Results (threshold=0.5):")
            logger.info(f"  Accuracy: {results['test_metrics']['cls_accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['test_metrics']['cls_f1']:.4f}")
            logger.info(f"  AUC-ROC: {results['test_metrics']['cls_auc_roc']:.4f}")

        if results['test_metrics_optimal'] is not None:
            logger.info(f"\nTest Set Results (threshold={results['optimal_threshold']:.2f}):")
            logger.info(f"  Accuracy: {results['test_metrics_optimal']['cls_accuracy']:.4f}")
            logger.info(f"  Precision: {results['test_metrics_optimal']['cls_precision']:.4f}")
            logger.info(f"  Recall: {results['test_metrics_optimal']['cls_recall']:.4f}")
            logger.info(f"  F1 Score: {results['test_metrics_optimal']['cls_f1']:.4f}")

        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Make sure data/ directory exists with required CSV files")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
