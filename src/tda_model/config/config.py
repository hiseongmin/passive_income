"""
Configuration dataclasses for TDA model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class TDAConfig:
    """TDA feature extraction parameters."""
    window_size: int = 672  # 7 days of 15-min candles (captures weekly patterns)
    embedding_dim: int = 2
    time_delay: int = 12
    betti_bins: int = 100  # Higher resolution for better feature separation
    landscape_layers: int = 5  # More layers for finer persistence details
    homology_dimensions: List[int] = field(default_factory=lambda: [0, 1])
    stride: int = 4


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    model_type: str = "nbeats"        # Model type: "lstm" or "nbeats"
    # Scaled parameters for full GPU utilization (RTX A6000 48GB)
    lstm_hidden_size: int = 1024      # Was 128 (8× increase)
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3         # Reduced for larger model
    use_attention: bool = True
    attention_heads: int = 8          # Increased for larger model
    tda_encoder_dim: int = 256        # Was 64 (4× increase)
    complexity_encoder_dim: int = 64  # Was 16 (4× increase)
    shared_fc_dim: int = 512          # Was 128 (4× increase)
    ohlcv_sequence_length: int = 96
    # Feature dimensions
    ohlcv_features: int = 4           # OHLC (base price features)
    volume_features: int = 5          # volume, buy_volume, sell_volume, volume_delta, cvd
    technical_features: int = 5       # RSI, MACD, BB_pctB, ATR, MOM
    complexity_features: int = 6      # All 6 complexity indicators (expanded from 1)
    # N-BEATS specific parameters (scaled for larger model)
    nbeats_num_stacks: int = 4        # Was 3 (1.3× increase)
    nbeats_num_blocks: int = 6        # Was 4 (1.5× increase)
    nbeats_num_layers: int = 6        # Was 4 (1.5× increase)


@dataclass
class TrainingConfig:
    """Training parameters - scaled for full GPU utilization."""
    batch_size: int = 2048                # Was 256 (8× for GPU memory)
    learning_rate: float = 0.0001         # Further reduced to prevent NaN
    weight_decay: float = 0.01            # Increased for regularization
    trigger_loss_weight: float = 3.0      # Emphasize classification over regression
    max_pct_loss_weight: float = 0.3      # De-emphasize regression
    focal_alpha: float = 0.75             # Moderate minority class weight
    focal_gamma: float = 2.0
    use_weighted_sampler: bool = True     # Enable - needed for class balance
    inference_threshold: float = 0.6      # Higher threshold for precision
    epochs: int = 500                     # Was 100 (5× for convergence)
    early_stopping_patience: int = 30     # Was 10 (3× for larger model)
    gradient_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10          # Was 5 (2× patience)
    # Gradient accumulation for effective larger batch
    gradient_accumulation_steps: int = 4  # Effective batch = 2048 * 4 = 8192


@dataclass
class DataConfig:
    """Data loading parameters."""
    data_dir: str = "data"
    cache_dir: str = "cache"
    train_file: str = "BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv"
    test_file: str = "BTCUSDT_spot_last_90d_15m_flagged.csv"
    validation_split: float = 0.15
    use_perp_data: bool = False
    # Stratified split settings (addresses distribution shift)
    use_stratified_split: bool = True     # Use stratified temporal split
    stratified_n_blocks: int = 10         # Number of blocks for stratified split
    # Complexity settings
    complexity_column: str = "complexity"  # Column name in data files
    complexity_placeholder: float = 0.5    # Fallback if column not available


@dataclass
class GPUConfig:
    """GPU and performance parameters."""
    device: str = "cuda:0"
    mixed_precision: bool = True
    compile_model: bool = False  # Disabled: causes dtype mismatch with BatchNorm + AMP
    cudnn_benchmark: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


@dataclass
class LoggingConfig:
    """Logging parameters."""
    log_dir: str = "logs"
    save_dir: str = "models/tda_model"
    log_interval: int = 100
    save_best_only: bool = True


@dataclass
class Config:
    """Main configuration container."""
    tda: TDAConfig = field(default_factory=TDAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            yaml_config = yaml.safe_load(f)

        return cls.from_dict(yaml_config)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary."""
        tda_config = TDAConfig(**config_dict.get("tda", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        # Merge gpu and dataloader configs
        gpu_dict = config_dict.get("gpu", {})
        dataloader_dict = config_dict.get("dataloader", {})
        gpu_dict.update(dataloader_dict)
        gpu_config = GPUConfig(**gpu_dict)

        logging_config = LoggingConfig(**config_dict.get("logging", {}))

        return cls(
            tda=tda_config,
            model=model_config,
            training=training_config,
            data=data_config,
            gpu=gpu_config,
            logging=logging_config,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "tda": self.tda.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "gpu": self.gpu.__dict__,
            "logging": self.logging.__dict__,
        }

    @property
    def tda_feature_dim(self) -> int:
        """Calculate total TDA feature dimension."""
        n_homology = len(self.tda.homology_dimensions)
        betti_dim = self.tda.betti_bins * n_homology
        entropy_dim = n_homology
        persistence_dim = n_homology
        landscape_dim = self.tda.landscape_layers * n_homology
        return betti_dim + entropy_dim + persistence_dim + landscape_dim


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from file or return default.

    Args:
        config_path: Path to YAML config file. If None, loads default config.

    Returns:
        Config object with all parameters.
    """
    if config_path is None:
        # Load default config from package
        default_path = Path(__file__).parent / "default_config.yaml"
        if default_path.exists():
            return Config.from_yaml(default_path)
        return Config()

    return Config.from_yaml(config_path)
