"""
Hybrid Fusion Architecture - Main fusion module.

Combines FiLM conditioning, cross-modal attention, and gated fusion
for sophisticated multi-modal feature fusion.

This is the RECOMMENDED fusion approach for the TDA-enhanced trading model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .modules import (
    RegimeEncoder,
    FiLMLayer,
    CrossModalAttention,
    MultiPathAggregator,
    GatedFusionWithEntropy,
)


class HybridFusion(nn.Module):
    """
    Hybrid Fusion Architecture combining FiLM, Cross-Attention, and Gated Fusion.

    Architecture stages:
    1. Feature Projection: Project modality features to common dimension
    2. Regime Encoding: Encode complexity features into regime representation
    3. FiLM Conditioning: Modulate N-BEATS and TDA based on regime
    4. Cross-Modal Attention: Bidirectional attention between modalities
    5. Multi-Path Aggregation: Four parallel fusion paths
    6. Gated Fusion: Adaptive weighted combination of paths
    7. Output: Final normalization and dropout

    Parameters: ~1.54M for fusion module (67% increase over baseline MLP)
    """

    def __init__(
        self,
        nbeats_dim: int = 1024,
        tda_dim: int = 256,
        complexity_dim: int = 64,
        hidden_dim: int = 256,
        regime_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            nbeats_dim: N-BEATS encoder output dimension
            tda_dim: TDA encoder output dimension
            complexity_dim: Complexity encoder output dimension
            hidden_dim: Hidden dimension throughout fusion
            regime_dim: Regime encoding dimension
            num_heads: Number of attention heads for cross-modal attention
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.regime_dim = regime_dim
        self.nbeats_dim = nbeats_dim
        self.tda_dim = tda_dim
        self.complexity_dim = complexity_dim

        # ═══════════════════════════════════════════════════════════════
        # Stage 1: Feature Projection
        # ═══════════════════════════════════════════════════════════════

        self.nbeats_projection = nn.Sequential(
            nn.Linear(nbeats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Lower dropout for projection
        )

        self.tda_projection = nn.Sequential(
            nn.Linear(tda_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ═══════════════════════════════════════════════════════════════
        # Stage 2: Regime Encoding
        # ═══════════════════════════════════════════════════════════════

        self.regime_encoder = RegimeEncoder(
            input_dim=complexity_dim,
            hidden_dim=regime_dim,
            output_dim=regime_dim,
            dropout=0.1,
        )

        # ═══════════════════════════════════════════════════════════════
        # Stage 3: FiLM Conditioning
        # ═══════════════════════════════════════════════════════════════

        self.film_nbeats = FiLMLayer(regime_dim, hidden_dim)
        self.film_tda = FiLMLayer(regime_dim, hidden_dim)

        # ═══════════════════════════════════════════════════════════════
        # Stage 4: Cross-Modal Attention
        # ═══════════════════════════════════════════════════════════════

        self.cross_attention = CrossModalAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
        )

        # ═══════════════════════════════════════════════════════════════
        # Stage 5: Multi-Path Aggregation
        # ═══════════════════════════════════════════════════════════════

        self.path_aggregator = MultiPathAggregator(
            hidden_dim=hidden_dim,
            regime_dim=regime_dim,
            dropout=dropout,
        )

        # ═══════════════════════════════════════════════════════════════
        # Stage 6: Gated Fusion
        # ═══════════════════════════════════════════════════════════════

        # Gate input: h_nbeats + h_tda + regime = 256 + 256 + 128 = 640
        gate_input_dim = hidden_dim + hidden_dim + regime_dim

        self.gated_fusion = GatedFusionWithEntropy(
            gate_input_dim=gate_input_dim,
            num_paths=4,
            hidden_dim=hidden_dim,
            dropout=0.1,
        )

        # ═══════════════════════════════════════════════════════════════
        # Stage 7: Output
        # ═══════════════════════════════════════════════════════════════

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

        # Storage for diagnostics
        self._diagnostics: Dict = {}

    @property
    def output_dim(self) -> int:
        """Output dimension of the fusion module."""
        return self.hidden_dim

    def forward(
        self,
        nbeats_features: torch.Tensor,
        tda_features: torch.Tensor,
        complexity_features: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through hybrid fusion network.

        Args:
            nbeats_features: (batch, nbeats_dim) from N-BEATS encoder
            tda_features: (batch, tda_dim) from TDA encoder
            complexity_features: (batch, complexity_dim) from Complexity encoder
            return_diagnostics: Whether to store intermediate values for analysis

        Returns:
            fused: (batch, hidden_dim) fused representation
        """
        # ─────────────────────────────────────────────────────────────
        # Stage 1: Project features to common dimension
        # ─────────────────────────────────────────────────────────────

        h_nbeats = self.nbeats_projection(nbeats_features)  # (B, hidden_dim)
        h_tda = self.tda_projection(tda_features)           # (B, hidden_dim)

        # ─────────────────────────────────────────────────────────────
        # Stage 2: Encode regime from complexity features
        # ─────────────────────────────────────────────────────────────

        regime = self.regime_encoder(complexity_features)   # (B, regime_dim)

        # ─────────────────────────────────────────────────────────────
        # Stage 3: FiLM conditioning
        # ─────────────────────────────────────────────────────────────

        h_n_film = self.film_nbeats(h_nbeats, regime)  # (B, hidden_dim)
        h_t_film = self.film_tda(h_tda, regime)        # (B, hidden_dim)

        # ─────────────────────────────────────────────────────────────
        # Stage 4: Cross-modal attention
        # ─────────────────────────────────────────────────────────────

        h_n_attn, h_t_attn, attn_info = self.cross_attention(h_n_film, h_t_film)

        # ─────────────────────────────────────────────────────────────
        # Stage 5: Multi-path aggregation
        # ─────────────────────────────────────────────────────────────

        paths = self.path_aggregator(
            h_n_film, h_t_film,
            h_n_attn, h_t_attn,
            regime,
        )  # (B, 4, hidden_dim)

        # ─────────────────────────────────────────────────────────────
        # Stage 6: Gated fusion
        # ─────────────────────────────────────────────────────────────

        gate_input = torch.cat([h_nbeats, h_tda, regime], dim=1)  # (B, 640)
        fused, gate_weights, gate_entropy = self.gated_fusion.forward_with_entropy(
            paths, gate_input
        )

        # ─────────────────────────────────────────────────────────────
        # Stage 7: Output normalization
        # ─────────────────────────────────────────────────────────────

        output = self.output_norm(fused)
        output = self.output_dropout(output)

        # Store diagnostics if requested
        if return_diagnostics:
            self._diagnostics = {
                'gate_weights': gate_weights.detach(),
                'gate_entropy': gate_entropy.detach(),
                'cross_attention': {
                    k: v.detach() for k, v in attn_info.items()
                },
                'regime_encoding': regime.detach(),
                'projected_features': {
                    'nbeats': h_nbeats.detach(),
                    'tda': h_tda.detach(),
                },
                'film_features': {
                    'nbeats': h_n_film.detach(),
                    'tda': h_t_film.detach(),
                },
            }

        return output

    def get_diagnostics(self) -> Dict:
        """
        Return stored diagnostic information from last forward pass.

        Returns:
            Dictionary containing:
            - gate_weights: (B, 4) weights for each fusion path
            - gate_entropy: (B,) entropy of gate distribution
            - cross_attention: dict with attention weights
            - regime_encoding: (B, regime_dim) learned regime
            - projected_features: dict with projected modality features
            - film_features: dict with FiLM-modulated features
        """
        return self._diagnostics

    def get_gate_interpretation(self, gate_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Interpret gate weights as named path contributions.

        Args:
            gate_weights: (batch, 4) gate weights. If None, uses stored diagnostics.

        Returns:
            Dictionary mapping path names to mean weights
        """
        if gate_weights is None:
            gate_weights = self._diagnostics.get('gate_weights')
            if gate_weights is None:
                raise ValueError("No gate weights available. Run forward with return_diagnostics=True first.")

        path_names = ['direct_film', 'cross_attention', 'bilinear', 'regime_context']
        mean_weights = gate_weights.mean(dim=0).tolist()

        return {name: weight for name, weight in zip(path_names, mean_weights)}

    def compute_entropy_loss(self, target_entropy: float = 1.0) -> torch.Tensor:
        """
        Compute entropy regularization loss for gate weights.

        Call this after forward() with return_diagnostics=True.

        Args:
            target_entropy: Target entropy value (default 1.0)
                           - Higher: encourage using all paths
                           - Lower: encourage specialization

        Returns:
            Scalar loss tensor
        """
        gate_weights = self._diagnostics.get('gate_weights')
        if gate_weights is None:
            raise ValueError("No gate weights available. Run forward with return_diagnostics=True first.")

        return self.gated_fusion.entropy_loss(gate_weights, target_entropy)


def count_fusion_parameters(fusion: HybridFusion) -> Dict[str, int]:
    """
    Count parameters in each stage of the fusion module.

    Args:
        fusion: HybridFusion module

    Returns:
        Dictionary with parameter counts per stage
    """
    counts = {}

    # Stage 1: Projections
    proj_params = sum(p.numel() for p in fusion.nbeats_projection.parameters())
    proj_params += sum(p.numel() for p in fusion.tda_projection.parameters())
    counts['stage1_projection'] = proj_params

    # Stage 2: Regime encoder
    counts['stage2_regime'] = sum(p.numel() for p in fusion.regime_encoder.parameters())

    # Stage 3: FiLM
    film_params = sum(p.numel() for p in fusion.film_nbeats.parameters())
    film_params += sum(p.numel() for p in fusion.film_tda.parameters())
    counts['stage3_film'] = film_params

    # Stage 4: Cross-attention
    counts['stage4_cross_attn'] = sum(p.numel() for p in fusion.cross_attention.parameters())

    # Stage 5: Paths
    counts['stage5_paths'] = sum(p.numel() for p in fusion.path_aggregator.parameters())

    # Stage 6: Gating
    counts['stage6_gating'] = sum(p.numel() for p in fusion.gated_fusion.parameters())

    # Stage 7: Output
    counts['stage7_output'] = sum(p.numel() for p in fusion.output_norm.parameters())

    # Total
    counts['total'] = sum(counts.values())

    return counts
