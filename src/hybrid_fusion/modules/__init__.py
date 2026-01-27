"""
Hybrid Fusion component modules.

This package contains the building blocks for hybrid fusion:
- FiLM conditioning (regime-aware modulation)
- Cross-modal attention (bidirectional attention)
- Fusion paths (direct, bilinear, regime)
- Gated fusion (adaptive combination)
"""

from .film import RegimeEncoder, FiLMGenerator, FiLMLayer
from .cross_attention import CrossModalAttention, CrossModalAttentionWithFFN
from .paths import FusionPath, BilinearPath, RegimePath, MultiPathAggregator
from .gating import GatedFusion, GatedFusionWithEntropy

__all__ = [
    # FiLM
    "RegimeEncoder",
    "FiLMGenerator",
    "FiLMLayer",
    # Cross-attention
    "CrossModalAttention",
    "CrossModalAttentionWithFFN",
    # Paths
    "FusionPath",
    "BilinearPath",
    "RegimePath",
    "MultiPathAggregator",
    # Gating
    "GatedFusion",
    "GatedFusionWithEntropy",
]
