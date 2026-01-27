"""
Cross-Modal Attention module for bidirectional attention between modalities.

Enables N-BEATS and TDA features to attend to each other, learning
which aspects of one modality are most relevant given the other.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention between two modalities.

    Unlike self-attention, cross-attention allows each modality to query
    the other, learning mutual dependencies between N-BEATS (technical patterns)
    and TDA (topological structure) features.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Feature dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Projections for first modality (N-BEATS)
        self.q1 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)

        # Projections for second modality (TDA)
        self.q2 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)

        # Output projections
        self.out1 = nn.Linear(dim, dim)
        self.out2 = nn.Linear(dim, dim)

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_for_attention(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reshape tensor for multi-head attention.

        Args:
            x: (batch, dim) -> (batch, num_heads, 1, head_dim)
        """
        return x.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Bidirectional cross-attention between two modalities.

        Args:
            x1: (batch, dim) first modality features (N-BEATS)
            x2: (batch, dim) second modality features (TDA)

        Returns:
            y1: (batch, dim) x1 enhanced by attending to x2
            y2: (batch, dim) x2 enhanced by attending to x1
            attn_weights: dict with attention strength metrics
        """
        batch_size = x1.size(0)

        # ═══════════════════════════════════════════════════════════════
        # x1 (N-BEATS) attends to x2 (TDA)
        # ═══════════════════════════════════════════════════════════════

        q1 = self._reshape_for_attention(self.q1(x1), batch_size)  # (B, H, 1, d)
        k2 = self._reshape_for_attention(self.k2(x2), batch_size)  # (B, H, 1, d)
        v2 = self._reshape_for_attention(self.v2(x2), batch_size)  # (B, H, 1, d)

        # Attention: Q1 queries K2, retrieves V2
        attn1 = torch.matmul(q1, k2.transpose(-2, -1)) / self.scale  # (B, H, 1, 1)
        attn1_weights = F.softmax(attn1, dim=-1)
        attn1_weights = self.dropout(attn1_weights)

        out1 = torch.matmul(attn1_weights, v2)  # (B, H, 1, d)
        out1 = out1.transpose(1, 2).contiguous().view(batch_size, -1)  # (B, dim)
        out1 = self.out1(out1)

        # ═══════════════════════════════════════════════════════════════
        # x2 (TDA) attends to x1 (N-BEATS)
        # ═══════════════════════════════════════════════════════════════

        q2 = self._reshape_for_attention(self.q2(x2), batch_size)
        k1 = self._reshape_for_attention(self.k1(x1), batch_size)
        v1 = self._reshape_for_attention(self.v1(x1), batch_size)

        attn2 = torch.matmul(q2, k1.transpose(-2, -1)) / self.scale
        attn2_weights = F.softmax(attn2, dim=-1)
        attn2_weights = self.dropout(attn2_weights)

        out2 = torch.matmul(attn2_weights, v1)
        out2 = out2.transpose(1, 2).contiguous().view(batch_size, -1)
        out2 = self.out2(out2)

        # ═══════════════════════════════════════════════════════════════
        # Residual connections and normalization
        # ═══════════════════════════════════════════════════════════════

        y1 = self.norm1(x1 + out1)
        y2 = self.norm2(x2 + out2)

        # Store attention weights for interpretability
        # Since we have single-token attention, the weights are scalars per head
        attn_info = {
            'nbeats_to_tda': attn1.squeeze(-1).squeeze(-1).mean(dim=1),  # (B,) avg across heads
            'tda_to_nbeats': attn2.squeeze(-1).squeeze(-1).mean(dim=1),  # (B,)
            'nbeats_to_tda_per_head': attn1.squeeze(-1).squeeze(-1),  # (B, H) per head
            'tda_to_nbeats_per_head': attn2.squeeze(-1).squeeze(-1),  # (B, H) per head
        }

        return y1, y2, attn_info


class CrossModalAttentionWithFFN(nn.Module):
    """
    Cross-modal attention with feed-forward network (full transformer block).

    This version includes the standard FFN after attention for additional
    non-linear transformation capacity.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.cross_attention = CrossModalAttention(dim, num_heads, dropout)

        # Feed-forward networks for each modality
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Cross-attention + FFN for both modalities.

        Args:
            x1: (batch, dim) N-BEATS features
            x2: (batch, dim) TDA features

        Returns:
            y1: (batch, dim) enhanced N-BEATS features
            y2: (batch, dim) enhanced TDA features
            attn_info: attention weights dictionary
        """
        # Cross-attention
        attn1, attn2, attn_info = self.cross_attention(x1, x2)

        # FFN with residual
        y1 = self.norm1(attn1 + self.ffn1(attn1))
        y2 = self.norm2(attn2 + self.ffn2(attn2))

        return y1, y2, attn_info
