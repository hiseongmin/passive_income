"""
Configuration dataclasses for Complete Hybrid Fusion model.

Provides configuration for:
- N-BEATS encoder
- TDA encoder
- Complexity encoder
- Hybrid fusion architecture
- Task heads
- Training parameters
- GPU settings
- Logging settings
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path

import yaml


# =============================================================================
# Encoder Configurations
# =============================================================================

@dataclass
class NBEATSEncoderConfig:
    """Configuration for N-BEATS encoder."""
    # Input
    input_dim: int = 14              # OHLCV features (open, high, low, close, volume, ...)
    seq_length: int = 96             # Sequence length

    # Stack configuration
    num_stacks: int = 4              # Number of stacks (Trend, Seasonality, Generic, Generic)
    stack_types: List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic", "generic"])

    # Block configuration
    num_blocks_per_stack: int = 3    # Blocks per stack
    hidden_dim: int = 256            # Hidden dimension in blocks
    theta_dim: int = 32              # Polynomial degree for trend/seasonality

    # Output
    output_dim: int = 256            # Output feature dimension

    # Regularization
    dropout: float = 0.1

    # Attention
    use_attention: bool = True
    num_attention_heads: int = 4


@dataclass
class TDAEncoderConfig:
    """Configuration for TDA encoder."""
    input_dim: int = 214             # TDA feature dimension
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    output_dim: int = 256            # Output feature dimension
    dropout: float = 0.2
    use_batch_norm: bool = True


@dataclass
class ComplexityEncoderConfig:
    """Configuration for Complexity encoder."""
    input_dim: int = 6               # 6 complexity indicators
    hidden_dim: int = 32
    output_dim: int = 64             # Output feature dimension
    dropout: float = 0.1


# =============================================================================
# Fusion Configuration
# =============================================================================

@dataclass
class HybridFusionConfig:
    """
    Configuration for the Hybrid Fusion module.

    This configures all stages of the fusion architecture:
    - Feature projection dimensions
    - Regime encoding
    - FiLM conditioning
    - Cross-modal attention
    - Multi-path aggregation
    - Gated fusion
    """
    # Core dimensions
    hidden_dim: int = 256           # Main hidden dimension throughout fusion
    regime_dim: int = 128           # Regime encoding dimension

    # Attention configuration
    num_heads: int = 4              # Heads for cross-modal attention

    # Regularization
    dropout: float = 0.3            # Main dropout rate

    # Logging and monitoring
    log_diagnostics_every: int = 100  # Log fusion diagnostics every N steps

    # Entropy regularization (optional)
    use_entropy_reg: bool = False     # Whether to use gate entropy regularization
    target_entropy: float = 1.0       # Target entropy for regularization
    entropy_weight: float = 0.01      # Weight of entropy loss


# =============================================================================
# Head Configuration
# =============================================================================

@dataclass
class HeadConfig:
    """Configuration for task-specific heads."""
    hidden_dim: int = 32            # Hidden dimension in heads
    dropout: float = 0.3            # Dropout in heads


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for Hybrid Fusion model."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler_type: str = "OneCycleLR"  # Options: OneCycleLR, ReduceLROnPlateau
    max_lr: float = 3e-4
    pct_start: float = 0.1
    scheduler_factor: float = 0.5       # For ReduceLROnPlateau
    scheduler_patience: int = 5         # For ReduceLROnPlateau

    # Loss weights
    trigger_loss_weight: float = 3.0
    max_pct_loss_weight: float = 0.3
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Regularization
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Training parameters
    epochs: int = 500
    batch_size: int = 2048
    early_stopping_patience: int = 30

    # Data loading
    use_weighted_sampler: bool = False


# =============================================================================
# GPU Configuration
# =============================================================================

@dataclass
class GPUConfig:
    """GPU and performance configuration."""
    device: str = "cuda:0"
    mixed_precision: bool = True
    compile_model: bool = False
    cudnn_benchmark: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "data/flagged"
    train_file: str = "btc_1m_train_flagged.csv"
    test_file: str = "btc_1m_test_flagged.csv"
    cache_dir: str = "data/tda_cache"
    validation_split: float = 0.15
    use_stratified_split: bool = True
    stratified_n_blocks: int = 10


# =============================================================================
# Logging Configuration
# =============================================================================

@dataclass
class LoggingConfig:
    """Logging and checkpoint configuration."""
    save_dir: str = "checkpoints/hybrid_fusion"
    log_dir: str = "logs/hybrid_fusion"
    log_interval: int = 100
    save_best_only: bool = True


# =============================================================================
# Complete Model Configuration
# =============================================================================

@dataclass
class CompleteModelConfig:
    """
    Complete configuration for Hybrid Fusion model with encoders.

    Combines:
    - Encoder configurations (N-BEATS, TDA, Complexity)
    - Hybrid fusion configuration
    - Head configuration
    - Training configuration
    - GPU configuration
    - Data configuration
    - Logging configuration
    """
    # Encoders
    nbeats: NBEATSEncoderConfig = field(default_factory=NBEATSEncoderConfig)
    tda: TDAEncoderConfig = field(default_factory=TDAEncoderConfig)
    complexity: ComplexityEncoderConfig = field(default_factory=ComplexityEncoderConfig)

    # Fusion
    fusion: HybridFusionConfig = field(default_factory=HybridFusionConfig)

    # Heads
    heads: HeadConfig = field(default_factory=HeadConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # GPU
    gpu: GPUConfig = field(default_factory=GPUConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CompleteModelConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CompleteModelConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        nbeats_dict = config_dict.get("nbeats", {})
        if "stack_types" in nbeats_dict and isinstance(nbeats_dict["stack_types"], str):
            nbeats_dict["stack_types"] = nbeats_dict["stack_types"].split(",")
        if "hidden_dims" in config_dict.get("tda", {}):
            tda_dict = config_dict["tda"]
            if isinstance(tda_dict["hidden_dims"], str):
                tda_dict["hidden_dims"] = [int(x) for x in tda_dict["hidden_dims"].split(",")]

        nbeats_config = NBEATSEncoderConfig(**nbeats_dict)
        tda_config = TDAEncoderConfig(**config_dict.get("tda", {}))
        complexity_config = ComplexityEncoderConfig(**config_dict.get("complexity", {}))
        fusion_config = HybridFusionConfig(**config_dict.get("fusion", {}))
        heads_config = HeadConfig(**config_dict.get("heads", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        gpu_config = GPUConfig(**config_dict.get("gpu", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))

        return cls(
            nbeats=nbeats_config,
            tda=tda_config,
            complexity=complexity_config,
            fusion=fusion_config,
            heads=heads_config,
            training=training_config,
            gpu=gpu_config,
            data=data_config,
            logging=logging_config,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(x) for x in obj]
            else:
                return obj

        return {
            "nbeats": dataclass_to_dict(self.nbeats),
            "tda": dataclass_to_dict(self.tda),
            "complexity": dataclass_to_dict(self.complexity),
            "fusion": dataclass_to_dict(self.fusion),
            "heads": dataclass_to_dict(self.heads),
            "training": dataclass_to_dict(self.training),
            "gpu": dataclass_to_dict(self.gpu),
            "data": dataclass_to_dict(self.data),
            "logging": dataclass_to_dict(self.logging),
        }

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def create_default_config() -> CompleteModelConfig:
    """
    Create default configuration for Hybrid Fusion model.

    Returns:
        CompleteModelConfig with recommended default values
    """
    return CompleteModelConfig()


def load_config(path: Optional[str | Path] = None) -> CompleteModelConfig:
    """
    Load configuration from YAML file or return default.

    Args:
        path: Path to YAML config file (optional)

    Returns:
        CompleteModelConfig instance
    """
    if path is None:
        return create_default_config()

    return CompleteModelConfig.from_yaml(path)
