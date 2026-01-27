"""
Configuration dataclasses for Cross-Modal Attention model.

Provides configuration for the cross-modal attention architecture
and the complete model.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import yaml


@dataclass
class CrossModalAttentionConfig:
    """
    Configuration for the Cross-Modal Attention fusion module.

    This configures the Transformer-based fusion:
    - Feature projection dimensions
    - Transformer architecture
    - Aggregation method
    """
    # Core dimensions
    hidden_dim: int = 256           # Transformer d_model (all modalities projected to this)
    num_heads: int = 8              # Multi-head attention heads
    num_layers: int = 2             # Transformer encoder layers
    ffn_dim: int = 1024             # Feed-forward network dimension

    # Regularization
    dropout: float = 0.1            # Dropout rate in transformer

    # Aggregation
    use_cls_token: bool = False     # If True, use CLS token; else mean pooling


@dataclass
class EncoderConfig:
    """Configuration for encoder dimensions."""
    nbeats_dim: int = 1024          # N-BEATS encoder output dimension
    tda_dim: int = 256              # TDA encoder output dimension
    complexity_dim: int = 64        # Complexity encoder output dimension


@dataclass
class HeadConfig:
    """Configuration for task-specific heads."""
    hidden_dim: int = 32            # Hidden dimension in heads
    dropout: float = 0.3            # Dropout in heads


@dataclass
class CrossModalModelConfig:
    """
    Complete configuration for Cross-Modal Attention model.

    Combines:
    - Encoder dimensions
    - Cross-modal attention fusion configuration
    - Head configuration
    """
    # Encoder configuration
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    # Fusion configuration
    fusion: CrossModalAttentionConfig = field(default_factory=CrossModalAttentionConfig)

    # Head configuration
    heads: HeadConfig = field(default_factory=HeadConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CrossModalModelConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CrossModalModelConfig":
        """Create configuration from dictionary."""
        encoder_config = EncoderConfig(**config_dict.get("encoder", {}))
        fusion_config = CrossModalAttentionConfig(**config_dict.get("fusion", {}))
        heads_config = HeadConfig(**config_dict.get("heads", {}))

        return cls(
            encoder=encoder_config,
            fusion=fusion_config,
            heads=heads_config,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "encoder": self.encoder.__dict__,
            "fusion": self.fusion.__dict__,
            "heads": self.heads.__dict__,
        }

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class TrainingConfig:
    """Training configuration for Cross-Modal Attention model."""
    # Optimizer
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Scheduler
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10

    # Loss weights
    trigger_loss_weight: float = 3.0
    max_pct_loss_weight: float = 0.3

    # Focal loss
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

    # Regularization
    gradient_clip: float = 1.0

    # Training parameters
    epochs: int = 500
    batch_size: int = 2048
    early_stopping_patience: int = 30
    gradient_accumulation_steps: int = 4


def create_default_config() -> CrossModalModelConfig:
    """
    Create default configuration for Cross-Modal Attention model.

    Returns:
        CrossModalModelConfig with recommended default values
    """
    return CrossModalModelConfig(
        encoder=EncoderConfig(
            nbeats_dim=1024,
            tda_dim=256,
            complexity_dim=64,
        ),
        fusion=CrossModalAttentionConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=2,
            ffn_dim=1024,
            dropout=0.1,
            use_cls_token=False,
        ),
        heads=HeadConfig(
            hidden_dim=32,
            dropout=0.3,
        ),
    )


# Default YAML configuration template
DEFAULT_CONFIG_YAML = """
# Cross-Modal Attention Model Configuration
# Transformer-based fusion for multi-modal features

encoder:
  nbeats_dim: 1024       # N-BEATS encoder output (from 4 stacks * 256)
  tda_dim: 256           # TDA encoder output
  complexity_dim: 64     # Complexity encoder output

fusion:
  hidden_dim: 256        # Transformer d_model
  num_heads: 8           # Multi-head attention heads
  num_layers: 2          # Transformer encoder layers
  ffn_dim: 1024          # FFN hidden dimension
  dropout: 0.1           # Transformer dropout
  use_cls_token: false   # Use mean pooling (recommended)

heads:
  hidden_dim: 32         # Hidden dim in task heads
  dropout: 0.3           # Head dropout

# Note: Encoder configurations (N-BEATS, TDA, Complexity) are defined
# in the encoders module. Only fusion-specific parameters here.
"""


def save_default_config(path: str | Path) -> None:
    """
    Save default configuration YAML to specified path.

    Args:
        path: Path to save the YAML configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(DEFAULT_CONFIG_YAML)
