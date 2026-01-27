"""
Cross-Modal Attention Transformer for Multi-Modal Fusion.

This package implements a Transformer-based cross-modal attention mechanism
that replaces simple concatenation. Each modality (N-BEATS, TDA, Complexity)
becomes a token that can attend to other modalities.

Architecture:
- Each modality (N-BEATS: 1024D, TDA: 256D, Complexity: 64D) is projected to 256D
- Modalities become tokens in a sequence (B, 3, 256)
- 2-layer Transformer encoder with 8-head self-attention
- Mean pooling aggregation → (B, 256)
- Task-specific heads for Trigger and Max_Pct

Usage:
    from src.cross_modal_attention import (
        CompleteCrossModalAttentionModel,
        MultiTaskCrossModalAttention,
        CrossModalAttentionFusion,
        CrossModalModelConfig,
        create_complete_model,
        create_cross_modal_model,
    )

    # Create complete model (with encoders)
    model = create_complete_model(model_size='medium')

    # Forward pass
    trigger_logits, max_pct = model(ohlcv_seq, tda_features, complexity)

    # Get predictions
    trigger_prob, max_pct = model.predict(ohlcv_seq, tda_features, complexity)

Reference: docs/fusion_models/cross_modal_attention.md
"""

# Core fusion module
from .fusion import CrossModalAttentionFusion, count_fusion_parameters

# Encoders (self-contained)
from .encoders import (
    OHLCVNBEATSEncoder,
    TDAEncoder,
    ComplexityEncoder,
    create_encoders,
    count_encoder_parameters,
)

# Complete models
from .model import (
    MultiTaskCrossModalAttention,
    CompleteCrossModalAttentionModel,
    create_complete_model,
    create_cross_modal_model,
)

# Configuration
from .config import (
    CrossModalAttentionConfig,
    EncoderConfig,
    HeadConfig,
    CrossModalModelConfig,
    TrainingConfig,
    create_default_config,
    save_default_config,
)

__version__ = "1.0.0"

__all__ = [
    # Main exports
    "CrossModalAttentionFusion",
    "CompleteCrossModalAttentionModel",
    "MultiTaskCrossModalAttention",
    "create_complete_model",
    "create_cross_modal_model",
    # Encoders
    "OHLCVNBEATSEncoder",
    "TDAEncoder",
    "ComplexityEncoder",
    "create_encoders",
    "count_encoder_parameters",
    # Configuration
    "CrossModalAttentionConfig",
    "EncoderConfig",
    "HeadConfig",
    "CrossModalModelConfig",
    "TrainingConfig",
    "create_default_config",
    "save_default_config",
    # Utilities
    "count_fusion_parameters",
]


def get_module_info() -> dict:
    """
    Get information about the cross-modal attention module.

    Returns:
        Dictionary with module metadata
    """
    return {
        "name": "Cross-Modal Attention",
        "version": __version__,
        "description": "Transformer-based cross-modal attention for multi-modal fusion",
        "estimated_parameters": "~2M (fusion module only), ~15.8M (complete model)",
        "input_dimensions": {
            "nbeats": 1024,
            "tda": 256,
            "complexity": 64,
        },
        "output_dimension": 256,
        "architecture": {
            "projection": "All modalities → 256D",
            "transformer": "2-layer encoder, 8-head self-attention",
            "aggregation": "Mean pooling (or CLS token)",
        },
    }
