"""
Fusion path modules for multi-path aggregation.

Each path provides a different way to combine modality features:
- DirectPath: Simple concatenation and projection
- BilinearPath: Element-wise multiplicative interaction
- RegimePath: Regime context projection
"""

import torch
import torch.nn as nn


class FusionPath(nn.Module):
    """
    A single fusion path that projects concatenated features to output dimension.

    Used for:
    - Path 1 (Direct): Concatenate FiLM outputs
    - Path 2 (Cross-Attn): Concatenate cross-attention outputs
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input dimension (typically 2 * hidden_dim for concat)
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) concatenated features

        Returns:
            output: (batch, output_dim) projected features
        """
        return self.fc(x)


class BilinearPath(nn.Module):
    """
    Bilinear interaction path using element-wise product.

    Captures multiplicative relationships between modalities:
    - If one modality is "on" and the other is "off", output is low
    - If both modalities agree (both high or both low), output is high

    This is simpler than full bilinear (W * x1 ⊗ x2) but efficient and effective.
    """

    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Feature dimension (same for both inputs)
            dropout: Dropout rate
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute bilinear interaction between two modalities.

        Args:
            x1: (batch, dim) first modality features
            x2: (batch, dim) second modality features

        Returns:
            interaction: (batch, dim) bilinear interaction features
        """
        # Element-wise product captures feature-level interactions
        interaction = x1 * x2
        return self.projection(interaction)


class RegimePath(nn.Module):
    """
    Regime context path that directly uses regime encoding.

    This path allows the model to directly incorporate market regime
    information into the fusion, separate from its role in FiLM conditioning.
    """

    def __init__(
        self,
        regime_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            regime_dim: Dimension of regime encoding
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(regime_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, regime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regime: (batch, regime_dim) regime encoding

        Returns:
            output: (batch, output_dim) projected regime context
        """
        return self.fc(regime)


class MultiPathAggregator(nn.Module):
    """
    Aggregates multiple fusion paths into a single tensor.

    Creates all four paths and stacks their outputs for gated fusion.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        regime_dim: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            hidden_dim: Hidden dimension for most paths
            regime_dim: Dimension of regime encoding
            dropout: Dropout rate for paths
        """
        super().__init__()

        # Path 1: Direct FiLM concatenation
        self.path_direct = FusionPath(hidden_dim * 2, hidden_dim, dropout)

        # Path 2: Cross-attention concatenation
        self.path_crossattn = FusionPath(hidden_dim * 2, hidden_dim, dropout)

        # Path 3: Bilinear interaction
        self.path_bilinear = BilinearPath(hidden_dim, dropout)

        # Path 4: Regime context
        self.path_regime = RegimePath(regime_dim, hidden_dim, dropout)

        self.num_paths = 4
        self.output_dim = hidden_dim

    def forward(
        self,
        h_n_film: torch.Tensor,
        h_t_film: torch.Tensor,
        h_n_attn: torch.Tensor,
        h_t_attn: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute all four fusion paths and stack them.

        Args:
            h_n_film: (batch, hidden_dim) FiLM-modulated N-BEATS
            h_t_film: (batch, hidden_dim) FiLM-modulated TDA
            h_n_attn: (batch, hidden_dim) Cross-attention N-BEATS
            h_t_attn: (batch, hidden_dim) Cross-attention TDA
            regime: (batch, regime_dim) Regime encoding

        Returns:
            paths: (batch, 4, hidden_dim) stacked path outputs
        """
        # Path 1: Direct FiLM concatenation
        path1 = self.path_direct(torch.cat([h_n_film, h_t_film], dim=1))

        # Path 2: Cross-attention concatenation
        path2 = self.path_crossattn(torch.cat([h_n_attn, h_t_attn], dim=1))

        # Path 3: Bilinear interaction
        path3 = self.path_bilinear(h_n_attn, h_t_attn)

        # Path 4: Regime context
        path4 = self.path_regime(regime)

        # Stack all paths: (batch, 4, hidden_dim)
        return torch.stack([path1, path2, path3, path4], dim=1)
