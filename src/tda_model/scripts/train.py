#!/usr/bin/env python3
"""
Main training script for TDA multi-task model.

Usage:
    python -m tda_model.scripts.train
    python -m tda_model.scripts.train --config path/to/config.yaml
    python -m tda_model.scripts.train --epochs 50 --batch-size 128
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tda_model.config import Config, load_config
from tda_model.data import create_data_loaders
from tda_model.training import create_trainer, create_model_from_config


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
        description="Train TDA multi-task model for Bitcoin trigger prediction",
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for TDA feature cache",
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
        "--hidden-size",
        type=int,
        default=None,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable self-attention",
    )

    # TDA arguments
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="TDA window size",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Takens embedding dimension",
    )
    parser.add_argument(
        "--time-delay",
        type=int,
        default=None,
        help="Takens time delay",
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
        "--no-compile",
        action="store_true",
        help="Disable torch.compile()",
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


def apply_args_to_config(args: argparse.Namespace, config: Config) -> Config:
    """Apply command line arguments to config."""
    # Data config
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.cache_dir is not None:
        config.data.cache_dir = args.cache_dir

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
    if args.hidden_size is not None:
        config.model.lstm_hidden_size = args.hidden_size
    if args.num_layers is not None:
        config.model.lstm_num_layers = args.num_layers
    if args.dropout is not None:
        config.model.lstm_dropout = args.dropout
    if args.no_attention:
        config.model.use_attention = False

    # TDA config
    if args.window_size is not None:
        config.tda.window_size = args.window_size
    if args.embedding_dim is not None:
        config.tda.embedding_dim = args.embedding_dim
    if args.time_delay is not None:
        config.tda.time_delay = args.time_delay

    # GPU config
    if args.device is not None:
        config.gpu.device = args.device
    if args.no_amp:
        config.gpu.mixed_precision = False
    if args.no_compile:
        config.gpu.compile_model = False
    if args.num_workers is not None:
        config.gpu.num_workers = args.num_workers

    # Logging config
    if args.save_dir is not None:
        config.logging.save_dir = args.save_dir
    if args.log_dir is not None:
        config.logging.log_dir = args.log_dir

    return config


def print_config(config: Config, logger: logging.Logger) -> None:
    """Print configuration to logger."""
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("=" * 60)

    logger.info("\nTDA Parameters:")
    logger.info(f"  window_size: {config.tda.window_size}")
    logger.info(f"  embedding_dim: {config.tda.embedding_dim}")
    logger.info(f"  time_delay: {config.tda.time_delay}")
    logger.info(f"  betti_bins: {config.tda.betti_bins}")
    logger.info(f"  landscape_layers: {config.tda.landscape_layers}")
    logger.info(f"  Feature dimension: {config.tda_feature_dim}")

    logger.info("\nModel Parameters:")
    logger.info(f"  lstm_hidden_size: {config.model.lstm_hidden_size}")
    logger.info(f"  lstm_num_layers: {config.model.lstm_num_layers}")
    logger.info(f"  lstm_dropout: {config.model.lstm_dropout}")
    logger.info(f"  use_attention: {config.model.use_attention}")
    logger.info(f"  ohlcv_sequence_length: {config.model.ohlcv_sequence_length}")

    logger.info("\nTraining Parameters:")
    logger.info(f"  batch_size: {config.training.batch_size}")
    logger.info(f"  learning_rate: {config.training.learning_rate}")
    logger.info(f"  epochs: {config.training.epochs}")
    logger.info(f"  early_stopping_patience: {config.training.early_stopping_patience}")
    logger.info(f"  trigger_loss_weight: {config.training.trigger_loss_weight}")
    logger.info(f"  max_pct_loss_weight: {config.training.max_pct_loss_weight}")

    logger.info("\nGPU Configuration:")
    logger.info(f"  device: {config.gpu.device}")
    logger.info(f"  mixed_precision: {config.gpu.mixed_precision}")
    logger.info(f"  compile_model: {config.gpu.compile_model}")
    logger.info(f"  num_workers: {config.gpu.num_workers}")

    logger.info("=" * 60)


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
        # Create data loaders
        logger.info("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            project_root=project_root,
        )

        # Create model (LSTM or N-BEATS based on config)
        logger.info("\nCreating model...")
        model = create_model_from_config(config)
        trainable, total = model.get_num_parameters()
        logger.info(f"Model parameters: {trainable:,} trainable / {total:,} total")

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
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {results_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Summary:")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {results['total_epochs']}")
        logger.info(f"Total time: {results['total_time_seconds']/60:.1f} minutes")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")

        if results['test_metrics'] is not None:
            logger.info("\nTest Set Results:")
            logger.info(f"  Accuracy: {results['test_metrics']['cls_accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['test_metrics']['cls_f1']:.4f}")
            logger.info(f"  AUC-ROC: {results['test_metrics']['cls_auc_roc']:.4f}")

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
