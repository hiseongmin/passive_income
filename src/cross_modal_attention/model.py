"""
Complete Cross-Modal Attention model with encoders and task heads.

This module provides SELF-CONTAINED model architectures that include:
- Self-contained encoders (N-BEATS, TDA, Complexity)
- Cross-Modal Attention Transformer fusion
- Task-specific heads (Trigger classification, Max_Pct regression)

No external dependencies - everything needed is included.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from .fusion import CrossModalAttentionFusion, count_fusion_parameters
from .config import CrossModalModelConfig, CrossModalAttentionConfig
from .encoders import (
    OHLCVNBEATSEncoder,
    TDAEncoder,
    ComplexityEncoder,
    create_encoders,
    count_encoder_parameters,
)


class MultiTaskCrossModalAttention(nn.Module):
    """
    Multi-task model with Cross-Modal Attention fusion.

    Takes pre-encoded features as input. Use when you have existing
    encoders or pre-computed encoder outputs.

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  nbeats_features (1024) ──┐                                     │
    │  tda_features (256) ──────┼──→ CrossModalAttentionFusion        │
    │  complexity_features (64) ┘    → (B, 256)                       │
    └────────────────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  ┌─────────────────┐         ┌─────────────────┐                │
    │  │  Trigger Head   │         │  Max Pct Head   │                │
    │  │  256→32→1       │         │  256→32→1       │                │
    │  └─────────────────┘         └─────────────────┘                │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        # Encoder output dimensions
        nbeats_dim: int = 1024,
        tda_dim: int = 256,
        complexity_dim: int = 64,
        # Fusion configuration
        fusion_hidden_dim: int = 256,
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        fusion_ffn_dim: int = 1024,
        fusion_dropout: float = 0.1,
        fusion_use_cls_token: bool = False,
        # Head configuration
        head_hidden_dim: int = 32,
        head_dropout: float = 0.3,
    ):
        """
        Initialize model with Cross-Modal Attention fusion.

        Args:
            nbeats_dim: Output dimension of N-BEATS encoder
            tda_dim: Output dimension of TDA encoder
            complexity_dim: Output dimension of Complexity encoder
            fusion_hidden_dim: Hidden dimension in fusion (transformer d_model)
            fusion_num_heads: Number of attention heads
            fusion_num_layers: Number of transformer encoder layers
            fusion_ffn_dim: Feed-forward network dimension
            fusion_dropout: Dropout rate in fusion
            fusion_use_cls_token: Use CLS token (False = mean pooling)
            head_hidden_dim: Hidden dimension in task heads
            head_dropout: Dropout rate in task heads
        """
        super().__init__()

        # Store configuration
        self.nbeats_dim = nbeats_dim
        self.tda_dim = tda_dim
        self.complexity_dim = complexity_dim

        # ═══════════════════════════════════════════════════════════════
        # Cross-Modal Attention Fusion
        # ═══════════════════════════════════════════════════════════════

        self.fusion = CrossModalAttentionFusion(
            nbeats_dim=nbeats_dim,
            tda_dim=tda_dim,
            complexity_dim=complexity_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            ffn_dim=fusion_ffn_dim,
            dropout=fusion_dropout,
            use_cls_token=fusion_use_cls_token,
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
    def from_config(cls, config: CrossModalModelConfig) -> "MultiTaskCrossModalAttention":
        """
        Create model from configuration object.

        Args:
            config: CrossModalModelConfig with all settings

        Returns:
            Initialized model
        """
        return cls(
            nbeats_dim=config.encoder.nbeats_dim,
            tda_dim=config.encoder.tda_dim,
            complexity_dim=config.encoder.complexity_dim,
            fusion_hidden_dim=config.fusion.hidden_dim,
            fusion_num_heads=config.fusion.num_heads,
            fusion_num_layers=config.fusion.num_layers,
            fusion_ffn_dim=config.fusion.ffn_dim,
            fusion_dropout=config.fusion.dropout,
            fusion_use_cls_token=config.fusion.use_cls_token,
            head_hidden_dim=config.heads.hidden_dim,
            head_dropout=config.heads.dropout,
        )

    def forward(
        self,
        nbeats_features: torch.Tensor,
        tda_features: torch.Tensor,
        complexity_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model.

        NOTE: This model expects pre-encoded features, not raw inputs.

        Args:
            nbeats_features: (batch, nbeats_dim) encoded OHLCV features
            tda_features: (batch, tda_dim) encoded TDA features
            complexity_features: (batch, complexity_dim) encoded complexity

        Returns:
            trigger_logits: (batch, 1) raw logits for trigger classification
            max_pct_pred: (batch, 1) predicted max percentage
        """
        # Cross-modal attention fusion
        fused = self.fusion(
            nbeats_features,
            tda_features,
            complexity_features,
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

    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get number of trainable and total parameters.

        Returns:
            Tuple of (trainable, total) parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


class CompleteCrossModalAttentionModel(nn.Module):
    """
    SELF-CONTAINED complete model with encoders + Cross-Modal Attention + heads.

    This is the recommended model for end-to-end training. It includes:
    - OHLCVNBEATSEncoder: N-BEATS based encoder for OHLCV sequences
    - TDAEncoder: MLP encoder for TDA features
    - ComplexityEncoder: MLP encoder for complexity indicators
    - CrossModalAttentionFusion: Transformer-based multi-modal fusion
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
    │                    CROSS-MODAL ATTENTION FUSION                          │
    │  Step 1: Project all to 256D                                            │
    │  Step 2: Stack as tokens (B, 3, 256)                                    │
    │  Step 3: Add positional embeddings                                      │
    │  Step 4: 2-layer Transformer Encoder (8-head self-attention)            │
    │  Step 5: Mean pooling → (B, 256)                                        │
    │  Step 6: Output projection                                              │
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

    Total Parameters: ~15.8M (encoders ~13.8M + fusion ~2M)
    """

    def __init__(
        self,
        # OHLCV Encoder config
        ohlcv_seq_length: int = 96,
        ohlcv_num_channels: int = 14,
        ohlcv_hidden_size: int = 256,
        ohlcv_num_stacks: int = 4,
        ohlcv_num_blocks: int = 4,
        ohlcv_num_layers: int = 4,
        ohlcv_dropout: float = 0.1,
        ohlcv_use_attention: bool = True,
        # TDA Encoder config
        tda_input_dim: int = 214,
        tda_hidden_dim: int = 128,
        tda_output_dim: int = 256,
        tda_dropout: float = 0.3,
        # Complexity Encoder config
        complexity_input_dim: int = 6,
        complexity_hidden_dim: int = 64,
        complexity_output_dim: int = 64,
        complexity_dropout: float = 0.1,
        # Fusion config
        fusion_hidden_dim: int = 256,
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        fusion_ffn_dim: int = 1024,
        fusion_dropout: float = 0.1,
        fusion_use_cls_token: bool = False,
        # Head config
        head_hidden_dim: int = 32,
        head_dropout: float = 0.3,
    ):
        """
        Initialize complete self-contained model.

        Args:
            ohlcv_seq_length: Length of input OHLCV sequence
            ohlcv_num_channels: Number of channels (OHLC + Volume + Technical)
            ohlcv_hidden_size: Hidden size for N-BEATS encoder
            ohlcv_num_stacks: Number of N-BEATS stacks
            ohlcv_num_blocks: Number of blocks per stack
            ohlcv_num_layers: Number of FC layers per block
            ohlcv_dropout: Dropout for OHLCV encoder
            ohlcv_use_attention: Whether to use attention in OHLCV encoder
            tda_input_dim: TDA feature dimension
            tda_hidden_dim: TDA encoder hidden dimension
            tda_output_dim: TDA encoder output dimension
            tda_dropout: Dropout for TDA encoder
            complexity_input_dim: Number of complexity indicators
            complexity_hidden_dim: Complexity encoder hidden dimension
            complexity_output_dim: Complexity encoder output dimension
            complexity_dropout: Dropout for complexity encoder
            fusion_hidden_dim: Fusion hidden dimension (transformer d_model)
            fusion_num_heads: Cross-modal attention heads
            fusion_num_layers: Transformer encoder layers
            fusion_ffn_dim: FFN dimension
            fusion_dropout: Fusion dropout
            fusion_use_cls_token: Use CLS token (False = mean pooling)
            head_hidden_dim: Task head hidden dimension
            head_dropout: Task head dropout
        """
        super().__init__()

        # Store config
        self.ohlcv_seq_length = ohlcv_seq_length
        self.ohlcv_num_channels = ohlcv_num_channels
        self.tda_input_dim = tda_input_dim
        self.complexity_input_dim = complexity_input_dim

        # ═══════════════════════════════════════════════════════════════
        # ENCODERS (Self-Contained)
        # ═══════════════════════════════════════════════════════════════

        self.ohlcv_encoder = OHLCVNBEATSEncoder(
            input_size=ohlcv_seq_length,
            hidden_size=ohlcv_hidden_size,
            num_stacks=ohlcv_num_stacks,
            num_blocks=ohlcv_num_blocks,
            num_layers=ohlcv_num_layers,
            dropout=ohlcv_dropout,
            num_channels=ohlcv_num_channels,
            use_attention=ohlcv_use_attention,
        )

        self.tda_encoder = TDAEncoder(
            input_dim=tda_input_dim,
            output_dim=tda_output_dim,
            hidden_dim=tda_hidden_dim,
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
        # CROSS-MODAL ATTENTION FUSION
        # ═══════════════════════════════════════════════════════════════

        self.fusion = CrossModalAttentionFusion(
            nbeats_dim=nbeats_dim,
            tda_dim=tda_dim,
            complexity_dim=complexity_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            ffn_dim=fusion_ffn_dim,
            dropout=fusion_dropout,
            use_cls_token=fusion_use_cls_token,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            ohlcv_seq: (batch, seq_len, num_channels) OHLCV sequence
            tda_features: (batch, tda_input_dim) raw TDA features
            complexity: (batch, complexity_input_dim) raw complexity indicators

        Returns:
            trigger_logits: (batch, 1) raw logits for trigger classification
            max_pct_pred: (batch, 1) predicted max percentage
        """
        # Encode each modality
        nbeats_encoded = self.ohlcv_encoder(ohlcv_seq)       # (B, 256)
        tda_encoded = self.tda_encoder(tda_features)         # (B, 256)
        complexity_encoded = self.complexity_encoder(complexity)  # (B, 64)

        # Cross-modal attention fusion
        fused = self.fusion(
            nbeats_encoded,
            tda_encoded,
            complexity_encoded,
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
    dropout: float = 0.1,
) -> CompleteCrossModalAttentionModel:
    """
    Factory function to create complete self-contained model.

    Args:
        ohlcv_seq_length: OHLCV sequence length
        ohlcv_num_channels: Number of OHLCV channels
        tda_input_dim: TDA feature dimension
        complexity_input_dim: Number of complexity indicators
        model_size: 'small', 'medium', or 'large'
        dropout: Global dropout rate for fusion

    Returns:
        Initialized CompleteCrossModalAttentionModel
    """
    # Size configurations
    size_configs = {
        'small': {
            'ohlcv_hidden_size': 128,
            'ohlcv_num_stacks': 2,
            'ohlcv_num_blocks': 2,
            'tda_output_dim': 128,
            'fusion_hidden_dim': 128,
            'fusion_ffn_dim': 512,
        },
        'medium': {
            'ohlcv_hidden_size': 256,
            'ohlcv_num_stacks': 4,
            'ohlcv_num_blocks': 4,
            'tda_output_dim': 256,
            'fusion_hidden_dim': 256,
            'fusion_ffn_dim': 1024,
        },
        'large': {
            'ohlcv_hidden_size': 512,
            'ohlcv_num_stacks': 4,
            'ohlcv_num_blocks': 6,
            'tda_output_dim': 512,
            'fusion_hidden_dim': 512,
            'fusion_ffn_dim': 2048,
        },
    }

    config = size_configs.get(model_size, size_configs['medium'])

    model = CompleteCrossModalAttentionModel(
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
        fusion_ffn_dim=config['fusion_ffn_dim'],
        fusion_dropout=dropout,
        head_dropout=dropout * 3,  # Higher for heads
    )

    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return model


def create_cross_modal_model(
    config: Optional[CrossModalModelConfig] = None,
    **kwargs,
) -> MultiTaskCrossModalAttention:
    """
    Factory function to create Cross-Modal Attention model (fusion only).

    Args:
        config: CrossModalModelConfig (if None, uses defaults + kwargs)
        **kwargs: Override any config parameter

    Returns:
        Initialized MultiTaskCrossModalAttention model
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
            elif key in config_dict.get('encoder', {}):
                config_dict['encoder'][key] = value
            elif key in config_dict.get('heads', {}):
                config_dict['heads'][key] = value
        config = CrossModalModelConfig.from_dict(config_dict)

    model = MultiTaskCrossModalAttention.from_config(config)

    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return model
