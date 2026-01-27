"""
Cross-Modal Attention Transformer for multi-modal fusion.

Replaces simple concatenation with transformer-based cross-attention
between N-BEATS, TDA, and Complexity modalities.

Architecture:
- Each modality (N-BEATS: 1024D, TDA: 256D, Complexity: 64D) is projected to 256D
- Modalities become tokens in a sequence (B, 3, 256)
- 2-layer Transformer encoder with 8-head self-attention
- Mean pooling aggregation → (B, 256)
- Output projection with dropout
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Transformer for multi-modal fusion.

    Replaces simple concatenation with transformer-based cross-attention
    between N-BEATS, TDA, and Complexity modalities.
    """

    def __init__(
        self,
        nbeats_dim: int = 1024,
        tda_dim: int = 256,
        complexity_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_cls_token: bool = False,
    ):
        """
        Args:
            nbeats_dim: N-BEATS encoder output dimension (1024)
            tda_dim: TDA encoder output dimension (256)
            complexity_dim: Complexity encoder output dimension (64)
            hidden_dim: Transformer hidden dimension (256)
            num_heads: Number of attention heads (8)
            num_layers: Number of transformer layers (2)
            ffn_dim: Feed-forward network dimension (1024)
            dropout: Dropout rate
            use_cls_token: Whether to use [CLS] token for aggregation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token
        self.num_modalities = 3  # N-BEATS, TDA, Complexity

        # === Projection Layers ===
        # Project each modality to same dimension
        self.proj_nbeats = nn.Linear(nbeats_dim, hidden_dim)
        self.proj_tda = nn.Linear(tda_dim, hidden_dim)
        self.proj_complexity = nn.Linear(complexity_dim, hidden_dim)

        # === Positional Encoding ===
        # Learnable positional embeddings for each modality
        num_positions = 4 if use_cls_token else 3
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_positions, hidden_dim) * 0.02
        )

        # === Optional CLS Token ===
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # === Output Projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output dimension for downstream heads
        self.output_dim = hidden_dim

        # Diagnostics storage
        self._last_attention_weights = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in [self.proj_nbeats, self.proj_tda, self.proj_complexity]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        nbeats_features: torch.Tensor,     # (B, 1024)
        tda_features: torch.Tensor,        # (B, 256)
        complexity_features: torch.Tensor,  # (B, 64)
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with cross-modal attention.

        Args:
            nbeats_features: N-BEATS encoder output (B, nbeats_dim)
            tda_features: TDA encoder output (B, tda_dim)
            complexity_features: Complexity encoder output (B, complexity_dim)
            return_diagnostics: Store attention weights for analysis

        Returns:
            Fused representation (B, hidden_dim)
        """
        batch_size = nbeats_features.size(0)

        # === Step 1: Project all modalities to same dimension ===
        nbeats_proj = self.proj_nbeats(nbeats_features)       # (B, hidden_dim)
        tda_proj = self.proj_tda(tda_features)                # (B, hidden_dim)
        complexity_proj = self.proj_complexity(complexity_features)  # (B, hidden_dim)

        # === Step 2: Stack as sequence of tokens ===
        # Each modality becomes a "token" in the sequence
        tokens = torch.stack([nbeats_proj, tda_proj, complexity_proj], dim=1)
        # Shape: (B, 3, hidden_dim)

        # === Step 3: Add CLS token if using ===
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 4, hidden_dim)

        # === Step 4: Add positional embeddings ===
        tokens = tokens + self.pos_embedding  # (B, 3 or 4, hidden_dim)

        # === Step 5: Apply Transformer Encoder ===
        # Self-attention allows each modality to attend to others
        transformed = self.transformer(tokens)  # (B, 3 or 4, hidden_dim)

        # === Step 6: Aggregate tokens ===
        if self.use_cls_token:
            # Use CLS token as aggregate representation
            aggregated = transformed[:, 0, :]  # (B, hidden_dim)
        else:
            # Mean pooling over all modality tokens
            aggregated = transformed.mean(dim=1)  # (B, hidden_dim)

        # === Step 7: Output projection ===
        output = self.output_proj(aggregated)  # (B, hidden_dim)

        return output

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information for analysis.

        Returns:
            Dictionary with fusion diagnostics
        """
        return {
            "hidden_dim": self.hidden_dim,
            "use_cls_token": self.use_cls_token,
            "num_modalities": self.num_modalities,
        }


def count_fusion_parameters(fusion: CrossModalAttentionFusion) -> Dict[str, int]:
    """
    Count parameters in the fusion module.

    Args:
        fusion: CrossModalAttentionFusion module

    Returns:
        Dictionary with parameter counts per component
    """
    counts = {}

    # Projection layers
    counts['proj_nbeats'] = sum(p.numel() for p in fusion.proj_nbeats.parameters())
    counts['proj_tda'] = sum(p.numel() for p in fusion.proj_tda.parameters())
    counts['proj_complexity'] = sum(p.numel() for p in fusion.proj_complexity.parameters())
    counts['projections_total'] = counts['proj_nbeats'] + counts['proj_tda'] + counts['proj_complexity']

    # Positional embeddings
    counts['pos_embedding'] = fusion.pos_embedding.numel()

    # CLS token
    if fusion.use_cls_token:
        counts['cls_token'] = fusion.cls_token.numel()
    else:
        counts['cls_token'] = 0

    # Transformer
    counts['transformer'] = sum(p.numel() for p in fusion.transformer.parameters())

    # Output projection
    counts['output_proj'] = sum(p.numel() for p in fusion.output_proj.parameters())

    # Total
    counts['total'] = sum(p.numel() for p in fusion.parameters())

    return counts
