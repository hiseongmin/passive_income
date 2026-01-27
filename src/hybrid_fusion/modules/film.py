"""
FiLM (Feature-wise Linear Modulation) conditioning module.

FiLM uses complexity features as a "regime signal" to modulate other features
through learned scale (gamma) and shift (beta) parameters.

Reference: "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)
"""

import torch
import torch.nn as nn
from typing import Tuple


class RegimeEncoder(nn.Module):
    """
    Encodes complexity features into regime representation for FiLM conditioning.

    The regime encoding captures market state (trending, ranging, volatile) which
    controls how N-BEATS and TDA features should be interpreted.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Complexity encoder output dimension
            hidden_dim: Hidden layer dimension
            output_dim: Regime encoding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.output_dim = output_dim

    def forward(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            complexity: (batch, input_dim) complexity encoder output

        Returns:
            regime: (batch, output_dim) regime encoding
        """
        return self.encoder(complexity)


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from regime encoding.

    Gamma (scale) and beta (shift) are learned transformations of the regime
    that modulate feature activations: output = gamma * input + beta
    """

    def __init__(self, regime_dim: int = 128, feature_dim: int = 256):
        """
        Args:
            regime_dim: Dimension of regime encoding
            feature_dim: Dimension of features to modulate
        """
        super().__init__()

        # Generate scale (gamma) - initialize near 1 for stable training
        self.gamma_net = nn.Linear(regime_dim, feature_dim)

        # Generate shift (beta) - initialize near 0
        self.beta_net = nn.Linear(regime_dim, feature_dim)

        # Custom initialization for stable FiLM conditioning
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable FiLM conditioning."""
        # Gamma: small weights so output starts near 1.0
        nn.init.normal_(self.gamma_net.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gamma_net.bias)

        # Beta: zero initialization for identity at start
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, regime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            regime: (batch, regime_dim) regime encoding

        Returns:
            gamma: (batch, feature_dim) scale parameters (centered around 1)
            beta: (batch, feature_dim) shift parameters
        """
        gamma = self.gamma_net(regime) + 1.0  # Center around 1 for identity init
        beta = self.beta_net(regime)
        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Applies FiLM conditioning: output = gamma * x + beta

    This layer modulates input features based on regime encoding,
    allowing the model to adapt its processing based on market conditions.
    """

    def __init__(self, regime_dim: int = 128, feature_dim: int = 256):
        """
        Args:
            regime_dim: Dimension of regime encoding
            feature_dim: Dimension of features to modulate
        """
        super().__init__()

        self.film_generator = FiLMGenerator(regime_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to input features.

        Args:
            x: (batch, feature_dim) input features to modulate
            regime: (batch, regime_dim) regime encoding

        Returns:
            modulated: (batch, feature_dim) FiLM-modulated features
        """
        gamma, beta = self.film_generator(regime)
        modulated = gamma * x + beta
        return self.norm(modulated)

    def forward_with_params(
        self,
        x: torch.Tensor,
        regime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply FiLM conditioning and return parameters for interpretability.

        Args:
            x: (batch, feature_dim) input features
            regime: (batch, regime_dim) regime encoding

        Returns:
            modulated: (batch, feature_dim) FiLM-modulated features
            gamma: (batch, feature_dim) scale parameters used
            beta: (batch, feature_dim) shift parameters used
        """
        gamma, beta = self.film_generator(regime)
        modulated = gamma * x + beta
        modulated = self.norm(modulated)
        return modulated, gamma, beta
