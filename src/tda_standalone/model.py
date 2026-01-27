"""
TDA Standalone Model Architecture.

Feature-aware model with:
- Three specialized encoders (structural, cyclical, landscape)
- Regime classification auxiliary task
- Regime-conditioned fusion
- Confidence estimation based on regime certainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import ModelConfig, PreprocessingConfig


class StructuralEncoder(nn.Module):
    """
    Encoder for H0-based structural features.

    Captures market structure coherence and trend quality.

    Input: (batch, 21) - 20 Betti H0 bins + 1 Persistence H0
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CyclicalEncoder(nn.Module):
    """
    Encoder for H1-based cyclical features.

    Captures loop structures indicating ranging/cycling markets.

    Input: (batch, 22) - 20 Betti H1 bins + Entropy H1 + Persistence H1
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LandscapeEncoder(nn.Module):
    """
    Encoder for landscape features (after PCA).

    Captures multi-scale topological patterns.
    Uses higher dropout since landscape features are noisy.

    Input: (batch, 20) - PCA-reduced landscape features
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 48,
        embed_dim: int = 24,
        dropout: float = 0.3,  # Higher for noisy features
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RegimeClassifier(nn.Module):
    """
    Auxiliary regime classifier.

    Predicts regime from cyclical (H1) embeddings.

    Input: (batch, embed_dim) cyclical embeddings
    Output: (batch, num_regimes) regime logits
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 16,
        num_regimes: int = 4,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RegimeConditionedFusion(nn.Module):
    """
    Fuses embeddings conditioned on predicted regime.

    Uses regime probabilities to modulate feature weights.

    Input:
    - structural: (batch, struct_dim)
    - cyclical: (batch, cycle_dim)
    - landscape: (batch, land_dim)
    - regime_prob: (batch, num_regimes)

    Output: (batch, fusion_dim)
    """

    def __init__(
        self,
        struct_dim: int = 32,
        cycle_dim: int = 32,
        land_dim: int = 24,
        num_regimes: int = 4,
        regime_embed_dim: int = 16,
    ):
        super().__init__()

        self.fusion_dim = struct_dim + cycle_dim + land_dim  # 88

        # Regime embedding
        self.regime_embed = nn.Linear(num_regimes, regime_embed_dim)

        # Modulation weights (project regime to fusion dim)
        self.modulation = nn.Linear(regime_embed_dim, self.fusion_dim)

    def forward(
        self,
        structural: torch.Tensor,
        cyclical: torch.Tensor,
        landscape: torch.Tensor,
        regime_prob: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate embeddings
        concat = torch.cat([structural, cyclical, landscape], dim=-1)  # (batch, 88)

        # Compute regime-based modulation
        regime_emb = self.regime_embed(regime_prob)  # (batch, 16)
        modulation = torch.tanh(self.modulation(regime_emb))  # (batch, 88)

        # Apply modulation: feature-wise scaling
        fused = concat * (1 + modulation)  # (batch, 88)

        return fused


class TriggerHead(nn.Module):
    """
    Trigger prediction head.

    Input: (batch, fusion_dim)
    Output: (batch, 1) logits
    """

    def __init__(
        self,
        input_dim: int = 88,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class TDAStandaloneModel(nn.Module):
    """
    TDA Standalone Model.

    Feature-aware architecture with:
    - Three specialized encoders for structural/cyclical/landscape features
    - Regime classification as auxiliary task
    - Regime-conditioned fusion
    - Confidence estimation from regime certainty

    Architecture:
        structural (21) → StructuralEncoder → struct_emb (32)
        cyclical (22)   → CyclicalEncoder → cycle_emb (32) → RegimeClassifier → regime (4)
        landscape (20)  → LandscapeEncoder → land_emb (24)

        [struct_emb, cycle_emb, land_emb] + regime → RegimeConditionedFusion → fused (88)

        fused → TriggerHead → trigger_logits (1)
        regime → confidence (derived from entropy)

    Total Parameters: ~15K
    """

    def __init__(self, config: ModelConfig, preproc_config: PreprocessingConfig):
        super().__init__()

        self.config = config

        # Encoders
        self.structural_encoder = StructuralEncoder(
            input_dim=preproc_config.structural_dim,
            hidden_dim=config.structural_hidden,
            embed_dim=config.structural_embed_dim,
            dropout=config.dropout,
        )
        self.cyclical_encoder = CyclicalEncoder(
            input_dim=preproc_config.cyclical_dim,
            hidden_dim=config.cyclical_hidden,
            embed_dim=config.cyclical_embed_dim,
            dropout=config.dropout,
        )
        self.landscape_encoder = LandscapeEncoder(
            input_dim=preproc_config.landscape_dim,
            hidden_dim=config.landscape_hidden,
            embed_dim=config.landscape_embed_dim,
            dropout=config.landscape_dropout,
        )

        # Regime classifier (from cyclical embeddings)
        self.regime_classifier = RegimeClassifier(
            input_dim=config.cyclical_embed_dim,
            hidden_dim=config.regime_hidden,
            num_regimes=config.num_regimes,
        )

        # Fusion
        self.fusion = RegimeConditionedFusion(
            struct_dim=config.structural_embed_dim,
            cycle_dim=config.cyclical_embed_dim,
            land_dim=config.landscape_embed_dim,
            num_regimes=config.num_regimes,
            regime_embed_dim=config.regime_embed_dim,
        )

        # Trigger head
        self.trigger_head = TriggerHead(
            input_dim=self.fusion.fusion_dim,
            hidden_dim=config.trigger_hidden,
            dropout=config.dropout,
        )

    def forward(
        self,
        structural: torch.Tensor,
        cyclical: torch.Tensor,
        landscape: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            structural: (batch, 21) H0-based features
            cyclical: (batch, 22) H1-based features
            landscape: (batch, 20) Landscape features

        Returns:
            Dict with:
            - trigger_logits: (batch,) trigger prediction logits
            - regime_logits: (batch, 4) regime classification logits
            - confidence: (batch,) confidence scores (0-1)
        """
        # Encode each feature group
        struct_emb = self.structural_encoder(structural)  # (batch, 32)
        cycle_emb = self.cyclical_encoder(cyclical)  # (batch, 32)
        land_emb = self.landscape_encoder(landscape)  # (batch, 24)

        # Predict regime from cyclical features
        regime_logits = self.regime_classifier(cycle_emb)  # (batch, 4)
        regime_prob = F.softmax(regime_logits, dim=-1)  # (batch, 4)

        # Compute confidence from regime certainty (1 - entropy)
        confidence = self._compute_confidence(regime_prob)  # (batch,)

        # Regime-conditioned fusion
        fused = self.fusion(struct_emb, cycle_emb, land_emb, regime_prob)  # (batch, 88)

        # Predict trigger
        trigger_logits = self.trigger_head(fused)  # (batch,)

        return {
            'trigger_logits': trigger_logits,
            'regime_logits': regime_logits,
            'confidence': confidence,
        }

    def _compute_confidence(self, regime_prob: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence from regime probability distribution.

        Confidence = 1 - normalized_entropy

        High confidence = model is certain about the regime
        Low confidence = uncertain regime (high entropy)
        """
        # Entropy of regime distribution
        eps = 1e-8
        entropy = -torch.sum(regime_prob * torch.log(regime_prob + eps), dim=-1)

        # Normalize by max entropy (log(num_regimes))
        max_entropy = torch.log(torch.tensor(regime_prob.size(-1), dtype=regime_prob.dtype, device=regime_prob.device))
        normalized_entropy = entropy / max_entropy

        # Confidence = 1 - normalized_entropy
        confidence = 1 - normalized_entropy

        return confidence

    def predict(
        self,
        structural: torch.Tensor,
        cyclical: torch.Tensor,
        landscape: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with probabilities.

        Returns:
            Dict with:
            - trigger_prob: (batch,) trigger probabilities (0-1)
            - regime_prob: (batch, 4) regime probabilities
            - confidence: (batch,) confidence scores (0-1)
        """
        output = self.forward(structural, cyclical, landscape)

        return {
            'trigger_prob': torch.sigmoid(output['trigger_logits']),
            'regime_prob': F.softmax(output['regime_logits'], dim=-1),
            'confidence': output['confidence'],
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder_parameters(self) -> Dict[str, int]:
        """Get parameter count per encoder."""
        return {
            'structural_encoder': sum(p.numel() for p in self.structural_encoder.parameters()),
            'cyclical_encoder': sum(p.numel() for p in self.cyclical_encoder.parameters()),
            'landscape_encoder': sum(p.numel() for p in self.landscape_encoder.parameters()),
            'regime_classifier': sum(p.numel() for p in self.regime_classifier.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'trigger_head': sum(p.numel() for p in self.trigger_head.parameters()),
        }


def create_model(
    model_config: ModelConfig,
    preproc_config: PreprocessingConfig,
    device: str = 'cuda',
) -> TDAStandaloneModel:
    """
    Create TDA Standalone model.

    Args:
        model_config: Model configuration
        preproc_config: Preprocessing configuration
        device: Device to place model on

    Returns:
        Initialized model
    """
    model = TDAStandaloneModel(model_config, preproc_config)
    model = model.to(device)

    # Print parameter counts
    print(f"\nModel Parameters:")
    print(f"  Total: {model.count_parameters():,}")
    for name, count in model.get_encoder_parameters().items():
        print(f"  {name}: {count:,}")

    return model
