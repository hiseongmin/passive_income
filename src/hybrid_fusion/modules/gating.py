"""
Gated Fusion module for adaptive path combination.

Learns to weight different fusion paths based on input features,
allowing the model to adapt its fusion strategy per sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatedFusion(nn.Module):
    """
    Learns adaptive weights to combine multiple fusion paths.

    The gate network takes concatenated features from all modalities
    and outputs softmax weights for each path. This allows the model
    to dynamically choose which fusion strategy to emphasize based
    on the current input.

    Interpretability: Gate weights reveal which fusion strategy
    dominates for each sample (direct, cross-attention, bilinear, or regime).
    """

    def __init__(
        self,
        gate_input_dim: int = 640,  # 256 + 256 + 128 (h_nbeats + h_tda + regime)
        num_paths: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            gate_input_dim: Dimension of concatenated gate inputs
            num_paths: Number of fusion paths to combine
            hidden_dim: Hidden dimension in gate network
            dropout: Dropout rate
        """
        super().__init__()

        self.num_paths = num_paths

        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_paths),
        )

    def forward(
        self,
        paths: torch.Tensor,
        gate_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated combination of fusion paths.

        Args:
            paths: (batch, num_paths, dim) stacked path outputs
            gate_input: (batch, gate_input_dim) concatenated features for gating

        Returns:
            fused: (batch, dim) weighted combination of paths
            gate_weights: (batch, num_paths) learned gate weights (softmax)
        """
        # Compute gate weights
        gate_logits = self.gate_network(gate_input)  # (B, num_paths)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, num_paths)

        # Weighted combination: sum of (weight * path) across paths
        # paths: (B, P, D), gate_weights: (B, P)
        fused = torch.sum(gate_weights.unsqueeze(-1) * paths, dim=1)  # (B, D)

        return fused, gate_weights

    def forward_with_temperature(
        self,
        paths: torch.Tensor,
        gate_input: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gated fusion with temperature-controlled softmax.

        Higher temperature -> more uniform weights (smoother)
        Lower temperature -> more peaked weights (sharper selection)

        Args:
            paths: (batch, num_paths, dim) stacked path outputs
            gate_input: (batch, gate_input_dim) concatenated features
            temperature: Temperature for softmax (default 1.0)

        Returns:
            fused: (batch, dim) weighted combination
            gate_weights: (batch, num_paths) temperature-scaled weights
        """
        gate_logits = self.gate_network(gate_input)
        gate_weights = F.softmax(gate_logits / temperature, dim=-1)
        fused = torch.sum(gate_weights.unsqueeze(-1) * paths, dim=1)
        return fused, gate_weights


class GatedFusionWithEntropy(GatedFusion):
    """
    Gated fusion with entropy regularization support.

    Provides methods to compute gate entropy for regularization,
    encouraging either diverse (high entropy) or specialized (low entropy)
    gate weight distributions.
    """

    def compute_entropy(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of gate weight distribution.

        Higher entropy = more uniform (using all paths equally)
        Lower entropy = more peaked (specializing to few paths)

        Args:
            gate_weights: (batch, num_paths) gate weight distribution

        Returns:
            entropy: (batch,) entropy per sample
        """
        # H = -sum(p * log(p))
        log_weights = torch.log(gate_weights + 1e-8)
        entropy = -torch.sum(gate_weights * log_weights, dim=-1)
        return entropy

    def entropy_loss(
        self,
        gate_weights: torch.Tensor,
        target_entropy: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute entropy regularization loss.

        Args:
            gate_weights: (batch, num_paths) gate weights
            target_entropy: Target entropy value to encourage
                           - Higher (e.g., log(4)=1.39): use all paths
                           - Lower (e.g., 0.5): specialize

        Returns:
            loss: Scalar MSE loss between actual and target entropy
        """
        entropy = self.compute_entropy(gate_weights)
        target = torch.full_like(entropy, target_entropy)
        return F.mse_loss(entropy, target)

    def forward_with_entropy(
        self,
        paths: torch.Tensor,
        gate_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns entropy for monitoring.

        Args:
            paths: (batch, num_paths, dim) stacked path outputs
            gate_input: (batch, gate_input_dim) concatenated features

        Returns:
            fused: (batch, dim) weighted combination
            gate_weights: (batch, num_paths) gate weights
            entropy: (batch,) entropy per sample
        """
        fused, gate_weights = self.forward(paths, gate_input)
        entropy = self.compute_entropy(gate_weights)
        return fused, gate_weights, entropy
