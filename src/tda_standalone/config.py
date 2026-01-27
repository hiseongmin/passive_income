"""
Configuration for TDA Standalone Model.

Dataclasses for model, training, and preprocessing configuration.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import yaml


@dataclass
class PreprocessingConfig:
    """Feature preprocessing configuration."""

    # Betti curve feature selection
    betti_bins_to_keep: int = 20  # Keep bins 0-19 for both H0 and H1

    # Landscape preprocessing
    landscape_clip_range: Tuple[float, float] = (-50.0, 50.0)
    landscape_pca_dims: int = 20

    # Feature removal
    remove_entropy_h0: bool = True  # Index 100 is constant

    # Resulting dimensions after preprocessing
    @property
    def structural_dim(self) -> int:
        """H0 bins + Persistence H0"""
        return self.betti_bins_to_keep + 1  # 21

    @property
    def cyclical_dim(self) -> int:
        """H1 bins + Entropy H1 + Persistence H1"""
        return self.betti_bins_to_keep + 2  # 22

    @property
    def landscape_dim(self) -> int:
        """After PCA"""
        return self.landscape_pca_dims  # 20

    @property
    def total_input_dim(self) -> int:
        """Total input dimension after preprocessing"""
        return self.structural_dim + self.cyclical_dim + self.landscape_dim  # 63


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Encoder hidden dimensions
    structural_hidden: int = 64
    cyclical_hidden: int = 64
    landscape_hidden: int = 48

    # Encoder output dimensions
    structural_embed_dim: int = 32
    cyclical_embed_dim: int = 32
    landscape_embed_dim: int = 24

    # Regime classifier
    num_regimes: int = 4
    regime_hidden: int = 16

    # Fusion
    @property
    def fusion_input_dim(self) -> int:
        return self.structural_embed_dim + self.cyclical_embed_dim + self.landscape_embed_dim

    regime_embed_dim: int = 16

    # Task head
    trigger_hidden: int = 32

    # Regularization
    dropout: float = 0.2
    landscape_dropout: float = 0.3  # Higher for noisy features


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 100

    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 0.0001

    # Loss weights
    trigger_weight: float = 1.0
    regime_weight: float = 0.3
    entropy_reg_weight: float = 0.1

    # Learning rate scheduler
    lr_scheduler: str = "reduce_on_plateau"
    lr_patience: int = 5
    lr_factor: float = 0.5

    # Validation
    val_frequency: int = 1  # Validate every N epochs

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints/tda_standalone"


@dataclass
class DataConfig:
    """Data paths configuration."""

    # TDA cache paths
    tda_cache_dir: str = "cache"
    tda_train_file: str = "tda_features_train.npy"
    tda_val_file: str = "tda_features_val.npy"
    tda_test_file: str = "tda_features_test.npy"

    # Labels path
    train_labels_path: str = "data/BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv"
    test_labels_path: str = "data/BTCUSDT_spot_last_90d_15m_flagged.csv"

    # Data split parameters (must match original TDA computation)
    window_size: int = 672  # 7 days of 15-min candles
    seq_len: int = 96  # 24 hours
    validation_split: float = 0.15

    @property
    def min_idx(self) -> int:
        return max(self.window_size, self.seq_len)


@dataclass
class TDAStandaloneConfig:
    """Complete configuration."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Device
    device: str = "cuda"

    # Random seed
    seed: int = 42

    # Logging
    log_dir: str = "logs/tda_standalone"
    experiment_name: str = "tda_standalone_v1"


def load_config(config_path: Optional[str] = None) -> TDAStandaloneConfig:
    """Load configuration from YAML file or return defaults."""
    if config_path is None:
        return TDAStandaloneConfig()

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse nested configs
    preprocessing = PreprocessingConfig(**config_dict.get('preprocessing', {}))
    model = ModelConfig(**config_dict.get('model', {}))
    training = TrainingConfig(**config_dict.get('training', {}))
    data = DataConfig(**config_dict.get('data', {}))

    return TDAStandaloneConfig(
        preprocessing=preprocessing,
        model=model,
        training=training,
        data=data,
        device=config_dict.get('device', 'cuda'),
        seed=config_dict.get('seed', 42),
        log_dir=config_dict.get('log_dir', 'logs/tda_standalone'),
        experiment_name=config_dict.get('experiment_name', 'tda_standalone_v1'),
    )


def save_config(config: TDAStandaloneConfig, path: str):
    """Save configuration to YAML file."""
    config_dict = {
        'preprocessing': {
            'betti_bins_to_keep': config.preprocessing.betti_bins_to_keep,
            'landscape_clip_range': list(config.preprocessing.landscape_clip_range),
            'landscape_pca_dims': config.preprocessing.landscape_pca_dims,
            'remove_entropy_h0': config.preprocessing.remove_entropy_h0,
        },
        'model': {
            'structural_hidden': config.model.structural_hidden,
            'cyclical_hidden': config.model.cyclical_hidden,
            'landscape_hidden': config.model.landscape_hidden,
            'structural_embed_dim': config.model.structural_embed_dim,
            'cyclical_embed_dim': config.model.cyclical_embed_dim,
            'landscape_embed_dim': config.model.landscape_embed_dim,
            'num_regimes': config.model.num_regimes,
            'regime_hidden': config.model.regime_hidden,
            'regime_embed_dim': config.model.regime_embed_dim,
            'trigger_hidden': config.model.trigger_hidden,
            'dropout': config.model.dropout,
            'landscape_dropout': config.model.landscape_dropout,
        },
        'training': {
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'epochs': config.training.epochs,
            'early_stopping_patience': config.training.early_stopping_patience,
            'trigger_weight': config.training.trigger_weight,
            'regime_weight': config.training.regime_weight,
            'entropy_reg_weight': config.training.entropy_reg_weight,
        },
        'data': {
            'tda_cache_dir': config.data.tda_cache_dir,
            'train_labels_path': config.data.train_labels_path,
            'test_labels_path': config.data.test_labels_path,
        },
        'device': config.device,
        'seed': config.seed,
        'log_dir': config.log_dir,
        'experiment_name': config.experiment_name,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
