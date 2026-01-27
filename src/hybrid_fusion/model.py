"""
Complete MultiTaskNBEATS model with Hybrid Fusion.

This module provides SELF-CONTAINED model architectures that include:
- Self-contained encoders (N-BEATS, TDA, Complexity)
- Hybrid Fusion module (replaces simple MLP)
- Task-specific heads (Trigger classification, Max_Pct regression)

No external dependencies on tda_model - everything needed is included.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List

from .fusion import HybridFusion, count_fusion_parameters
from .config import HybridFusionConfig
from .encoders import (
    OHLCVNBEATSEncoder,
    TDAEncoder,
    ComplexityEncoder,
    create_encoders,
    count_encoder_parameters,
)


class MultiTaskNBEATSWithHybridFusion(nn.Module):
    """
    Multi-task N-BEATS model with Hybrid Fusion for trigger prediction
    and max percentage estimation.

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  OHLCV Sequence → OHLCVNBEATSEncoder → nbeats_features (1024)   │
    │  TDA Features → TDAEncoder → tda_features (256)                 │
    │  Complexity → ComplexityEncoder → complexity_features (64)      │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    HYBRID FUSION                                 │
    │  Stage 1: Feature Projection (1024→256, 256→256)                │
    │  Stage 2: Regime Encoding (64→128)                              │
    │  Stage 3: FiLM Conditioning                                      │
    │  Stage 4: Cross-Modal Attention                                  │
    │  Stage 5: Multi-Path Aggregation (4 paths)                       │
    │  Stage 6: Gated Fusion → fused (256)                            │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  ┌─────────────────┐         ┌─────────────────┐                │
    │  │  Trigger Head   │         │  Max Pct Head   │                │
    │  │  256→32→1       │         │  256→32→1       │                │
    │  │  (sigmoid)      │         │  (linear)       │                │
    │  └─────────────────┘         └─────────────────┘                │
    └─────────────────────────────────────────────────────────────────┘

    This model is designed to be a drop-in replacement for the existing
    MultiTaskNBEATS, using the same encoder outputs but with improved
    fusion.
    """

    def __init__(
        self,
        # Encoder output dimensions
        nbeats_dim: int = 1024,
        tda_dim: int = 256,
        complexity_dim: int = 64,
        # Fusion configuration
        fusion_hidden_dim: int = 256,
        fusion_regime_dim: int = 128,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.3,
        # Head configuration
        head_hidden_dim: int = 32,
        head_dropout: float = 0.3,
    ):
        """
        Initialize model with Hybrid Fusion.

        Args:
            nbeats_dim: Output dimension of N-BEATS encoder
            tda_dim: Output dimension of TDA encoder
            complexity_dim: Output dimension of Complexity encoder
            fusion_hidden_dim: Hidden dimension in fusion
            fusion_regime_dim: Regime encoding dimension
            fusion_num_heads: Attention heads in cross-modal attention
            fusion_dropout: Dropout rate in fusion
            head_hidden_dim: Hidden dimension in task heads
            head_dropout: Dropout rate in task heads
        """
        super().__init__()

        # Store configuration
        self.nbeats_dim = nbeats_dim
        self.tda_dim = tda_dim
        self.complexity_dim = complexity_dim

        # ═══════════════════════════════════════════════════════════════
        # Hybrid Fusion (replaces shared_fc)
        # ═══════════════════════════════════════════════════════════════

        self.fusion = HybridFusion(
            nbeats_dim=nbeats_dim,
            tda_dim=tda_dim,
            complexity_dim=complexity_dim,
            hidden_dim=fusion_hidden_dim,
            regime_dim=fusion_regime_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )

        # ═══════════════════════════════════════════════════════════════
        # Task Heads
        # ═══════════════════════════════════════════════════════════════

        # Trigger head (binary classification)
        self.trigger_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        # Max_Pct head (regression)
        self.max_pct_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    @classmethod
    def from_config(cls, config: "CompleteModelConfig") -> "MultiTaskNBEATSWithHybridFusion":
        """
        Create model from configuration object.

        Args:
            config: CompleteModelConfig with all settings

        Returns:
            Initialized model
        """
        return cls(
            nbeats_dim=config.nbeats.output_dim,
            tda_dim=config.tda.output_dim,
            complexity_dim=config.complexity.output_dim,
            fusion_hidden_dim=config.fusion.hidden_dim,
            fusion_regime_dim=config.fusion.regime_dim,
            fusion_num_heads=config.fusion.num_heads,
            fusion_dropout=config.fusion.dropout,
            head_hidden_dim=config.heads.hidden_dim,
            head_dropout=config.heads.dropout,
        )

    def forward(
        self,
        nbeats_features: torch.Tensor,
        tda_features: torch.Tensor,
        complexity_features: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model.

        NOTE: This model expects pre-encoded features, not raw inputs.
        Use with existing encoders:
            nbeats_features = ohlcv_encoder(ohlcv_seq)
            tda_features = tda_encoder(tda_raw)
            complexity_features = complexity_encoder(complexity_raw)

        Args:
            nbeats_features: (batch, nbeats_dim) encoded OHLCV features
            tda_features: (batch, tda_dim) encoded TDA features
            complexity_features: (batch, complexity_dim) encoded complexity
            return_diagnostics: Store intermediate values for analysis

        Returns:
            trigger_logits: (batch, 1) raw logits for trigger classification
            max_pct_pred: (batch, 1) predicted max percentage
        """
        # Hybrid fusion
        fused = self.fusion(
            nbeats_features,
            tda_features,
            complexity_features,
            return_diagnostics=return_diagnostics,
        )  # (B, hidden_dim)

        # Task predictions
        trigger_logits = self.trigger_head(fused)   # (B, 1)
        max_pct_pred = self.max_pct_head(fused)     # (B, 1)

        return trigger_logits, max_pct_pred

    def predict(
        self,
        nbeats_features: torch.Tensor,
        tda_features: torch.Tensor,
        complexity_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with sigmoid applied to trigger.

        Args:
            nbeats_features: (batch, nbeats_dim) encoded OHLCV features
            tda_features: (batch, tda_dim) encoded TDA features
            complexity_features: (batch, complexity_dim) encoded complexity

        Returns:
            trigger_prob: (batch, 1) probability of trigger [0, 1]
            max_pct_pred: (batch, 1) predicted max percentage
        """
        trigger_logits, max_pct_pred = self.forward(
            nbeats_features, tda_features, complexity_features
        )
        trigger_prob = torch.sigmoid(trigger_logits)
        return trigger_prob, max_pct_pred

    def get_fusion_diagnostics(self) -> Dict:
        """
        Get diagnostic information from fusion module.

        Returns:
            Dictionary with gate weights, attention info, etc.
        """
        return self.fusion.get_diagnostics()

    def get_gate_interpretation(self) -> Dict[str, float]:
        """
        Get interpretation of gate weights.

        Returns:
            Dictionary mapping path names to mean weights
        """
        return self.fusion.get_gate_interpretation()

    def compute_entropy_loss(self, target_entropy: float = 1.0) -> torch.Tensor:
        """
        Compute entropy regularization loss for training.

        Must call forward with return_diagnostics=True first.

        Args:
            target_entropy: Target entropy for gate weights

        Returns:
            Scalar loss tensor
        """
        return self.fusion.compute_entropy_loss(target_entropy)

    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get number of trainable and total parameters.

        Returns:
            Tuple of (trainable, total) parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """
        Get detailed parameter count breakdown.

        Returns:
            Dictionary with parameter counts per component
        """
        breakdown = {}

        # Fusion parameters
        fusion_counts = count_fusion_parameters(self.fusion)
        for key, value in fusion_counts.items():
            breakdown[f'fusion_{key}'] = value

        # Head parameters
        breakdown['trigger_head'] = sum(p.numel() for p in self.trigger_head.parameters())
        breakdown['max_pct_head'] = sum(p.numel() for p in self.max_pct_head.parameters())

        # Total
        breakdown['total_model'] = sum(p.numel() for p in self.parameters())

        return breakdown


class IntegratedMultiTaskNBEATS(nn.Module):
    """
    Full integrated model with encoders and Hybrid Fusion.

    This is a complete model that includes the encoders, not just
    the fusion module. Use this for training from scratch with
    all components.

    NOTE: This requires the existing encoder implementations from
    tda_model.models.nbeats (OHLCVNBEATSEncoder, TDAEncoder, ComplexityEncoder)
    """

    def __init__(
        self,
        # Import from existing config
        config: Optional["Config"] = None,
        # Or specify directly
        ohlcv_encoder: Optional[nn.Module] = None,
        tda_encoder: Optional[nn.Module] = None,
        complexity_encoder: Optional[nn.Module] = None,
        # Fusion config
        fusion_config: Optional[HybridFusionConfig] = None,
    ):
        """
        Initialize integrated model.

        Can be initialized either with:
        1. Existing config + encoders
        2. Pre-built encoder modules + fusion config

        Args:
            config: Existing TDA model config (if using option 1)
            ohlcv_encoder: Pre-built N-BEATS encoder (if using option 2)
            tda_encoder: Pre-built TDA encoder
            complexity_encoder: Pre-built Complexity encoder
            fusion_config: Hybrid fusion configuration
        """
        super().__init__()

        # Store encoders (to be set externally or built from config)
        self.ohlcv_encoder = ohlcv_encoder
        self.tda_encoder = tda_encoder
        self.complexity_encoder = complexity_encoder

        # Get encoder output dimensions
        if ohlcv_encoder is not None:
            nbeats_dim = ohlcv_encoder.output_dim
        elif config is not None:
            nbeats_dim = config.model.lstm_hidden_size  # N-BEATS uses this
        else:
            nbeats_dim = 1024

        if tda_encoder is not None:
            tda_dim = tda_encoder.output_dim
        elif config is not None:
            tda_dim = config.model.tda_encoder_dim
        else:
            tda_dim = 256

        if complexity_encoder is not None:
            complexity_dim = complexity_encoder.output_dim
        elif config is not None:
            complexity_dim = config.model.complexity_encoder_dim
        else:
            complexity_dim = 64

        # Set fusion config
        if fusion_config is None:
            fusion_config = HybridFusionConfig()

        # Build fusion + heads
        self.fusion_model = MultiTaskNBEATSWithHybridFusion(
            nbeats_dim=nbeats_dim,
            tda_dim=tda_dim,
            complexity_dim=complexity_dim,
            fusion_hidden_dim=fusion_config.hidden_dim,
            fusion_regime_dim=fusion_config.regime_dim,
            fusion_num_heads=fusion_config.num_heads,
            fusion_dropout=fusion_config.dropout,
        )

    def set_encoders(
        self,
        ohlcv_encoder: nn.Module,
        tda_encoder: nn.Module,
        complexity_encoder: nn.Module,
    ) -> None:
        """
        Set encoder modules after initialization.

        Args:
            ohlcv_encoder: N-BEATS encoder for OHLCV sequences
            tda_encoder: MLP encoder for TDA features
            complexity_encoder: MLP encoder for complexity features
        """
        self.ohlcv_encoder = ohlcv_encoder
        self.tda_encoder = tda_encoder
        self.complexity_encoder = complexity_encoder

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        tda_features: torch.Tensor,
        complexity: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            ohlcv_seq: (batch, seq_len, num_channels) OHLCV sequence
            tda_features: (batch, tda_dim) raw TDA features
            complexity: (batch, complexity_dim) raw complexity features
            return_diagnostics: Store intermediate values

        Returns:
            trigger_logits: (batch, 1) raw trigger logits
            max_pct_pred: (batch, 1) max percentage prediction
        """
        if self.ohlcv_encoder is None:
            raise RuntimeError("Encoders not set. Call set_encoders() first.")

        # Encode each modality
        nbeats_encoded = self.ohlcv_encoder(ohlcv_seq)
        tda_encoded = self.tda_encoder(tda_features)
        complexity_encoded = self.complexity_encoder(complexity)

        # Fusion + heads
        return self.fusion_model(
            nbeats_encoded,
            tda_encoded,
            complexity_encoded,
            return_diagnostics=return_diagnostics,
        )

    def predict(
        self,
        ohlcv_seq: torch.Tensor,
        tda_features: torch.Tensor,
        complexity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with sigmoid applied to trigger."""
        trigger_logits, max_pct_pred = self.forward(
            ohlcv_seq, tda_features, complexity
        )
        trigger_prob = torch.sigmoid(trigger_logits)
        return trigger_prob, max_pct_pred


class CompleteHybridFusionModel(nn.Module):
    """
    SELF-CONTAINED complete model with encoders + Hybrid Fusion + heads.

    This is the recommended model for end-to-end training. It includes:
    - OHLCVNBEATSEncoder: N-BEATS based encoder for OHLCV sequences
    - TDAEncoder: MLP encoder for TDA features
    - ComplexityEncoder: MLP encoder for complexity indicators
    - HybridFusion: Advanced fusion combining FiLM, Cross-Attention, Gating
    - Task heads: Trigger classification + Max_Pct regression

    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           RAW INPUTS                                     │
    │  OHLCV Sequence (B, 96, 14)   TDA (B, 214)   Complexity (B, 6)          │
    └───────────┬─────────────────────────┬─────────────────────┬─────────────┘
                │                         │                     │
                ▼                         ▼                     ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           ENCODERS                                       │
    │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐        │
    │  │ OHLCVNBEATSEnc  │   │   TDAEncoder    │   │ ComplexityEnc   │        │
    │  │ N-BEATS Stacks  │   │   MLP 214→256   │   │   MLP 6→64      │        │
    │  │ → 1024 dims     │   │                 │   │                 │        │
    │  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘        │
    └───────────┼─────────────────────┼─────────────────────┼─────────────────┘
                │ (B, 1024)           │ (B, 256)            │ (B, 64)
                └─────────────────────┼─────────────────────┘
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        HYBRID FUSION                                     │
    │  Stage 1: Feature Projection (1024→256, 256→256)                        │
    │  Stage 2: Regime Encoding (64→128)                                      │
    │  Stage 3: FiLM Conditioning                                              │
    │  Stage 4: Cross-Modal Attention                                          │
    │  Stage 5: Multi-Path Aggregation (4 paths)                               │
    │  Stage 6: Gated Fusion → fused (256)                                    │
    └─────────────────────────────────┬───────────────────────────────────────┘
                                      │ (B, 256)
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           TASK HEADS                                     │
    │  ┌─────────────────────┐              ┌─────────────────────┐           │
    │  │    Trigger Head     │              │    Max Pct Head     │           │
    │  │    256 → 32 → 1     │              │    256 → 32 → 1     │           │
    │  │    (sigmoid)        │              │    (linear)         │           │
    │  └─────────────────────┘              └─────────────────────┘           │
    └─────────────────────────────────────────────────────────────────────────┘

    Total Parameters: ~18.5M (encoders ~17M + fusion ~1.5M)
    """

    def __init__(
        self,
        # N-BEATS Encoder config
        ohlcv_input_dim: int = 14,
        ohlcv_seq_length: int = 96,
        nbeats_hidden_dim: int = 256,
        nbeats_output_dim: int = 256,
        nbeats_num_stacks: int = 4,
        nbeats_num_blocks: int = 3,
        nbeats_theta_dim: int = 32,
        nbeats_dropout: float = 0.1,
        nbeats_use_attention: bool = True,
        # TDA Encoder config
        tda_input_dim: int = 214,
        tda_hidden_dims: List[int] = None,
        tda_output_dim: int = 256,
        tda_dropout: float = 0.2,
        # Complexity Encoder config
        complexity_input_dim: int = 6,
        complexity_hidden_dim: int = 32,
        complexity_output_dim: int = 64,
        complexity_dropout: float = 0.1,
        # Fusion config
        fusion_hidden_dim: int = 256,
        fusion_regime_dim: int = 128,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.3,
        # Head config
        head_hidden_dim: int = 32,
        head_dropout: float = 0.3,
    ):
        """
        Initialize complete self-contained model.

        Args:
            ohlcv_input_dim: Number of OHLCV features per timestep
            ohlcv_seq_length: Length of input OHLCV sequence
            nbeats_hidden_dim: Hidden size for N-BEATS encoder
            nbeats_output_dim: N-BEATS encoder output dimension
            nbeats_num_stacks: Number of N-BEATS stacks
            nbeats_num_blocks: Number of blocks per stack
            nbeats_theta_dim: Polynomial degree for basis expansion
            nbeats_dropout: Dropout for OHLCV encoder
            nbeats_use_attention: Whether to use attention in OHLCV encoder
            tda_input_dim: TDA feature dimension (Betti + entropy + etc.)
            tda_hidden_dims: TDA encoder hidden layer dimensions (list)
            tda_output_dim: TDA encoder output dimension
            tda_dropout: Dropout for TDA encoder
            complexity_input_dim: Number of complexity indicators
            complexity_hidden_dim: Complexity encoder hidden dimension
            complexity_output_dim: Complexity encoder output dimension
            complexity_dropout: Dropout for complexity encoder
            fusion_hidden_dim: Fusion hidden dimension
            fusion_regime_dim: Regime encoding dimension
            fusion_num_heads: Cross-attention heads
            fusion_dropout: Fusion dropout
            head_hidden_dim: Task head hidden dimension
            head_dropout: Task head dropout
        """
        super().__init__()

        # Default hidden dims for TDA encoder
        if tda_hidden_dims is None:
            tda_hidden_dims = [512, 256]

        # Store config
        self.ohlcv_seq_length = ohlcv_seq_length
        self.ohlcv_input_dim = ohlcv_input_dim
        self.tda_input_dim = tda_input_dim
        self.complexity_input_dim = complexity_input_dim

        # ═══════════════════════════════════════════════════════════════
        # ENCODERS (Self-Contained)
        # ═══════════════════════════════════════════════════════════════

        self.ohlcv_encoder = OHLCVNBEATSEncoder(
            input_size=ohlcv_seq_length,
            hidden_size=nbeats_hidden_dim,
            num_stacks=nbeats_num_stacks,
            num_blocks=nbeats_num_blocks,
            num_layers=4,  # Fixed for N-BEATS
            dropout=nbeats_dropout,
            num_channels=ohlcv_input_dim,
            use_attention=nbeats_use_attention,
        )

        self.tda_encoder = TDAEncoder(
            input_dim=tda_input_dim,
            output_dim=tda_output_dim,
            hidden_dim=tda_hidden_dims[0] if tda_hidden_dims else 256,
            dropout=tda_dropout,
        )

        self.complexity_encoder = ComplexityEncoder(
            input_dim=complexity_input_dim,
            output_dim=complexity_output_dim,
            hidden_dim=complexity_hidden_dim,
            dropout=complexity_dropout,
        )

        # Get encoder output dimensions
        nbeats_dim = self.ohlcv_encoder.output_dim
        tda_dim = self.tda_encoder.output_dim
        complexity_dim = self.complexity_encoder.output_dim

        # ═══════════════════════════════════════════════════════════════
        # HYBRID FUSION
        # ═══════════════════════════════════════════════════════════════

        self.fusion = HybridFusion(
            nbeats_dim=nbeats_dim,
            tda_dim=tda_dim,
            complexity_dim=complexity_dim,
            hidden_dim=fusion_hidden_dim,
            regime_dim=fusion_regime_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )

        # ═══════════════════════════════════════════════════════════════
        # TASK HEADS
        # ═══════════════════════════════════════════════════════════════

        self.trigger_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        self.max_pct_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        tda_features: torch.Tensor,
        complexity: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            ohlcv_seq: (batch, seq_len, num_channels) OHLCV sequence
            tda_features: (batch, tda_input_dim) raw TDA features
            complexity: (batch, complexity_input_dim) raw complexity indicators
            return_diagnostics: Store intermediate values for analysis

        Returns:
            trigger_logits: (batch, 1) raw logits for trigger classification
            max_pct_pred: (batch, 1) predicted max percentage
        """
        # Encode each modality
        nbeats_encoded = self.ohlcv_encoder(ohlcv_seq)       # (B, 256)
        tda_encoded = self.tda_encoder(tda_features)         # (B, 256)
        complexity_encoded = self.complexity_encoder(complexity)  # (B, 64)

        # Hybrid fusion
        fused = self.fusion(
            nbeats_encoded,
            tda_encoded,
            complexity_encoded,
            return_diagnostics=return_diagnostics,
        )  # (B, 256)

        # Task predictions
        trigger_logits = self.trigger_head(fused)   # (B, 1)
        max_pct_pred = self.max_pct_head(fused)     # (B, 1)

        return trigger_logits, max_pct_pred

    def predict(
        self,
        ohlcv_seq: torch.Tensor,
        tda_features: torch.Tensor,
        complexity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with sigmoid applied to trigger.

        Args:
            ohlcv_seq: (batch, seq_len, num_channels) OHLCV sequence
            tda_features: (batch, tda_input_dim) raw TDA features
            complexity: (batch, complexity_input_dim) raw complexity indicators

        Returns:
            trigger_prob: (batch, 1) probability of trigger [0, 1]
            max_pct_pred: (batch, 1) predicted max percentage
        """
        trigger_logits, max_pct_pred = self.forward(
            ohlcv_seq, tda_features, complexity
        )
        trigger_prob = torch.sigmoid(trigger_logits)
        return trigger_prob, max_pct_pred

    def get_fusion_diagnostics(self) -> Dict:
        """Get diagnostic information from fusion module."""
        return self.fusion.get_diagnostics()

    def get_gate_interpretation(self) -> Dict[str, float]:
        """Get interpretation of gate weights."""
        return self.fusion.get_gate_interpretation()

    def compute_entropy_loss(self, target_entropy: float = 1.0) -> torch.Tensor:
        """Compute entropy regularization loss."""
        return self.fusion.compute_entropy_loss(target_entropy)

    def get_num_parameters(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get detailed parameter count breakdown."""
        breakdown = {}

        # Encoder parameters
        breakdown['ohlcv_encoder'] = count_encoder_parameters(self.ohlcv_encoder)
        breakdown['tda_encoder'] = count_encoder_parameters(self.tda_encoder)
        breakdown['complexity_encoder'] = count_encoder_parameters(self.complexity_encoder)
        breakdown['encoders_total'] = sum([
            breakdown['ohlcv_encoder'],
            breakdown['tda_encoder'],
            breakdown['complexity_encoder'],
        ])

        # Fusion parameters
        fusion_counts = count_fusion_parameters(self.fusion)
        breakdown['fusion_total'] = fusion_counts['total']

        # Head parameters
        breakdown['trigger_head'] = sum(p.numel() for p in self.trigger_head.parameters())
        breakdown['max_pct_head'] = sum(p.numel() for p in self.max_pct_head.parameters())
        breakdown['heads_total'] = breakdown['trigger_head'] + breakdown['max_pct_head']

        # Total
        breakdown['total_model'] = sum(p.numel() for p in self.parameters())

        return breakdown


def create_complete_model(
    # Quick config options
    ohlcv_seq_length: int = 96,
    ohlcv_num_channels: int = 14,
    tda_input_dim: int = 214,
    complexity_input_dim: int = 6,
    # Model size: 'small', 'medium', 'large'
    model_size: str = 'medium',
    dropout: float = 0.3,
) -> CompleteHybridFusionModel:
    """
    Factory function to create complete self-contained model.

    Args:
        ohlcv_seq_length: OHLCV sequence length
        ohlcv_num_channels: Number of OHLCV channels
        tda_input_dim: TDA feature dimension
        complexity_input_dim: Number of complexity indicators
        model_size: 'small', 'medium', or 'large'
        dropout: Global dropout rate

    Returns:
        Initialized CompleteHybridFusionModel
    """
    # Size configurations
    size_configs = {
        'small': {
            'ohlcv_hidden_size': 128,
            'ohlcv_num_stacks': 2,
            'ohlcv_num_blocks': 2,
            'tda_output_dim': 128,
            'fusion_hidden_dim': 128,
            'fusion_regime_dim': 64,
        },
        'medium': {
            'ohlcv_hidden_size': 256,
            'ohlcv_num_stacks': 4,
            'ohlcv_num_blocks': 4,
            'tda_output_dim': 256,
            'fusion_hidden_dim': 256,
            'fusion_regime_dim': 128,
        },
        'large': {
            'ohlcv_hidden_size': 512,
            'ohlcv_num_stacks': 4,
            'ohlcv_num_blocks': 6,
            'tda_output_dim': 512,
            'fusion_hidden_dim': 512,
            'fusion_regime_dim': 256,
        },
    }

    config = size_configs.get(model_size, size_configs['medium'])

    model = CompleteHybridFusionModel(
        ohlcv_seq_length=ohlcv_seq_length,
        ohlcv_num_channels=ohlcv_num_channels,
        ohlcv_hidden_size=config['ohlcv_hidden_size'],
        ohlcv_num_stacks=config['ohlcv_num_stacks'],
        ohlcv_num_blocks=config['ohlcv_num_blocks'],
        ohlcv_dropout=dropout * 0.3,  # Lower for encoder
        tda_input_dim=tda_input_dim,
        tda_output_dim=config['tda_output_dim'],
        tda_dropout=dropout,
        complexity_input_dim=complexity_input_dim,
        complexity_output_dim=64,
        complexity_dropout=dropout * 0.3,
        fusion_hidden_dim=config['fusion_hidden_dim'],
        fusion_regime_dim=config['fusion_regime_dim'],
        fusion_dropout=dropout,
        head_dropout=dropout,
    )

    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return model


def create_hybrid_model(
    config: Optional["CompleteModelConfig"] = None,
    **kwargs,
) -> MultiTaskNBEATSWithHybridFusion:
    """
    Factory function to create Hybrid Fusion model.

    Args:
        config: CompleteModelConfig (if None, uses defaults + kwargs)
        **kwargs: Override any config parameter

    Returns:
        Initialized MultiTaskNBEATSWithHybridFusion model
    """
    if config is None:
        from .config import create_default_config
        config = create_default_config()

    # Apply any overrides
    if kwargs:
        config_dict = config.to_dict()
        for key, value in kwargs.items():
            if key in config_dict.get('fusion', {}):
                config_dict['fusion'][key] = value
            elif key in config_dict.get('nbeats', {}):
                config_dict['nbeats'][key] = value
            elif key in config_dict.get('tda', {}):
                config_dict['tda'][key] = value
            elif key in config_dict.get('heads', {}):
                config_dict['heads'][key] = value
        from .config import CompleteModelConfig
        config = CompleteModelConfig.from_dict(config_dict)

    model = MultiTaskNBEATSWithHybridFusion.from_config(config)

    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return model
