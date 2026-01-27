"""
Hybrid Fusion Architecture for Multi-Modal Feature Fusion.

This package implements the recommended fusion architecture that combines:
- FiLM Conditioning: Regime-aware feature modulation
- Cross-Modal Attention: Bidirectional attention between modalities
- Gated Fusion: Adaptive weighted combination of multiple fusion paths

Architecture stages:
1. Feature Projection: Project modality features to common dimension
2. Regime Encoding: Encode complexity features into regime representation
3. FiLM Conditioning: Modulate N-BEATS and TDA based on regime
4. Cross-Modal Attention: Bidirectional attention between modalities
5. Multi-Path Aggregation: Four parallel fusion paths
6. Gated Fusion: Adaptive weighted combination
7. Output: Final normalization and dropout

Usage:
    from src.hybrid_fusion import (
        HybridFusion,
        MultiTaskNBEATSWithHybridFusion,
        HybridModelConfig,
        create_hybrid_model,
    )

    # Create with default config
    model = create_hybrid_model()

    # Or with custom config
    config = HybridModelConfig(
        fusion=HybridFusionConfig(hidden_dim=256, num_heads=4),
    )
    model = MultiTaskNBEATSWithHybridFusion.from_config(config)

    # Forward pass (with pre-encoded features)
    trigger_logits, max_pct = model(nbeats_features, tda_features, complexity_features)

    # Get predictions with sigmoid
    trigger_prob, max_pct = model.predict(nbeats_features, tda_features, complexity_features)

    # Get fusion diagnostics
    _ = model(nbeats_features, tda_features, complexity_features, return_diagnostics=True)
    diagnostics = model.get_fusion_diagnostics()
    gate_weights = diagnostics['gate_weights']  # (batch, 4) path weights

Author: Claude (Anthropic)
Reference: docs/fusion_models/hybrid_fusion.md
"""

# Core fusion module
from .fusion import HybridFusion, count_fusion_parameters

# Encoders (self-contained)
from .encoders import (
    OHLCVNBEATSEncoder,
    TDAEncoder,
    ComplexityEncoder,
    create_encoders,
    count_encoder_parameters,
)

# Complete model
from .model import (
    MultiTaskNBEATSWithHybridFusion,
    IntegratedMultiTaskNBEATS,
    CompleteHybridFusionModel,
    create_hybrid_model,
    create_complete_model,
)

# Configuration
from .config import (
    # Encoder configs
    NBEATSEncoderConfig,
    TDAEncoderConfig,
    ComplexityEncoderConfig,
    # Fusion and head configs
    HybridFusionConfig,
    HeadConfig,
    # Training and system configs
    TrainingConfig,
    GPUConfig,
    DataConfig,
    LoggingConfig,
    # Complete config
    CompleteModelConfig,
    create_default_config,
    load_config,
)

# Component modules (for advanced usage)
from .modules import (
    # FiLM
    RegimeEncoder,
    FiLMGenerator,
    FiLMLayer,
    # Cross-attention
    CrossModalAttention,
    CrossModalAttentionWithFFN,
    # Paths
    FusionPath,
    BilinearPath,
    RegimePath,
    MultiPathAggregator,
    # Gating
    GatedFusion,
    GatedFusionWithEntropy,
)

__version__ = "1.0.0"

__all__ = [
    # Main exports
    "HybridFusion",
    "CompleteHybridFusionModel",
    "MultiTaskNBEATSWithHybridFusion",
    "IntegratedMultiTaskNBEATS",
    "create_hybrid_model",
    "create_complete_model",
    # Encoders
    "OHLCVNBEATSEncoder",
    "TDAEncoder",
    "ComplexityEncoder",
    "create_encoders",
    "count_encoder_parameters",
    # Configuration
    "NBEATSEncoderConfig",
    "TDAEncoderConfig",
    "ComplexityEncoderConfig",
    "HybridFusionConfig",
    "HeadConfig",
    "TrainingConfig",
    "GPUConfig",
    "DataConfig",
    "LoggingConfig",
    "CompleteModelConfig",
    "create_default_config",
    "load_config",
    # Utilities
    "count_fusion_parameters",
    # Component modules
    "RegimeEncoder",
    "FiLMGenerator",
    "FiLMLayer",
    "CrossModalAttention",
    "CrossModalAttentionWithFFN",
    "FusionPath",
    "BilinearPath",
    "RegimePath",
    "MultiPathAggregator",
    "GatedFusion",
    "GatedFusionWithEntropy",
]


def get_module_info() -> dict:
    """
    Get information about the hybrid fusion module.

    Returns:
        Dictionary with module metadata
    """
    return {
        "name": "Hybrid Fusion",
        "version": __version__,
        "description": "Multi-modal fusion combining FiLM, Cross-Attention, and Gated Fusion",
        "estimated_parameters": "~1.54M (fusion module only)",
        "input_dimensions": {
            "nbeats": 1024,
            "tda": 256,
            "complexity": 64,
        },
        "output_dimension": 256,
        "fusion_paths": [
            "direct_film",
            "cross_attention",
            "bilinear",
            "regime_context",
        ],
    }
