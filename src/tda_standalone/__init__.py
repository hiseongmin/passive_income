"""
TDA Standalone Model

A self-contained, feature-aware model that uses only TDA features
with specialized processing for each feature type based on what
it actually captures topologically.

Key Features:
- Removes useless features (Entropy H0, high Betti bins)
- Cleans problematic features (Landscape outliers via clip + PCA)
- Three specialized encoders for structural/cyclical/landscape
- Explicit regime prediction as auxiliary task
- Confidence estimation based on regime certainty
- ~15K parameters (lightweight)

Usage:
    from src.tda_standalone import (
        TDAStandaloneModel,
        TDAPreprocessor,
        TDAStandaloneDataset,
        create_model,
        train,
    )
"""

from .config import (
    ModelConfig,
    TrainingConfig,
    PreprocessingConfig,
    TDAStandaloneConfig,
    load_config,
)
from .preprocessing import TDAPreprocessor
from .regime import RegimeLabeler, compute_regime_labels
from .dataset import TDAStandaloneDataset, create_data_loaders
from .model import TDAStandaloneModel, create_model
from .losses import TDAStandaloneLoss, TDAFocalLoss, create_loss_function
from .train import train_tda_standalone, evaluate_model, Trainer

__version__ = "1.0.0"

__all__ = [
    # Config
    "ModelConfig",
    "TrainingConfig",
    "PreprocessingConfig",
    "TDAStandaloneConfig",
    "load_config",
    # Preprocessing
    "TDAPreprocessor",
    # Regime
    "RegimeLabeler",
    "compute_regime_labels",
    # Dataset
    "TDAStandaloneDataset",
    "create_data_loaders",
    # Model
    "TDAStandaloneModel",
    "create_model",
    # Losses
    "TDAStandaloneLoss",
    "TDAFocalLoss",
    "create_loss_function",
    # Training
    "train_tda_standalone",
    "evaluate_model",
    "Trainer",
]
