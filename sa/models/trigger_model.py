"""
Trigger Prediction Model

LSTM + N-BEATS + TDA architecture for predicting trading triggers.

Outputs:
- trigger_prob: Probability of trigger occurrence (0~1)
- imminence: How soon the trigger will occur (0~1, 1=imminent)
- direction_logits: Direction prediction (UP/DOWN/NONE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .nbeats_blocks import NBeatsBlock, NBeatsStack


class TriggerPredictionModel(nn.Module):
    """
    Multi-input model for trigger prediction.

    Inputs:
    - 5-minute OHLCV sequence (batch, seq_len_5m, 5)
    - 1-hour OHLCV sequence (batch, seq_len_1h, 5)
    - TDA features (batch, n_tda_features)
    - Microstructure features (batch, n_micro_features)

    Outputs:
    - trigger_prob: (batch, 1) - Trigger probability
    - imminence: (batch, 1) - Imminence score
    - direction_logits: (batch, 3) - Direction logits (UP, DOWN, NONE)
    """

    def __init__(
        self,
        # Sequence dimensions
        seq_len_5m: int = 72,
        seq_len_1h: int = 6,
        ohlcv_features: int = 5,

        # Feature dimensions
        tda_features: int = 9,
        micro_features: int = 12,

        # LSTM configuration
        lstm_hidden_5m: int = 128,
        lstm_hidden_1h: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,

        # Encoder configuration
        encoder_hidden: int = 64,
        encoder_dropout: float = 0.1,

        # N-BEATS configuration
        nbeats_blocks: int = 3,
        nbeats_hidden: int = 256,
        nbeats_layers: int = 4,
        theta_size: int = 32,
        nbeats_dropout: float = 0.1,

        # Output configuration
        num_directions: int = 3,  # UP, DOWN, NONE
    ):
        super().__init__()

        self.seq_len_5m = seq_len_5m
        self.seq_len_1h = seq_len_1h
        self.lstm_hidden_5m = lstm_hidden_5m
        self.lstm_hidden_1h = lstm_hidden_1h
        self.lstm_layers = lstm_layers

        # ==================== LSTM Encoders ====================
        self.lstm_5m = nn.LSTM(
            input_size=ohlcv_features,
            hidden_size=lstm_hidden_5m,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.lstm_1h = nn.LSTM(
            input_size=ohlcv_features,
            hidden_size=lstm_hidden_1h,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        # LSTM output sizes (bidirectional)
        lstm_5m_out_size = lstm_hidden_5m * 2  # 256
        lstm_1h_out_size = lstm_hidden_1h * 2  # 128

        # ==================== Feature Encoders ====================
        self.tda_encoder = nn.Sequential(
            nn.Linear(tda_features, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.GELU()
        )

        self.micro_encoder = nn.Sequential(
            nn.Linear(micro_features, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.GELU()
        )

        # ==================== Temporal Fusion ====================
        # Total: 256 (5m) + 128 (1h) + 64 (tda) + 64 (micro) = 512
        fusion_input_size = lstm_5m_out_size + lstm_1h_out_size + encoder_hidden * 2

        self.temporal_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, nbeats_hidden),
            nn.LayerNorm(nbeats_hidden),
            nn.GELU(),
            nn.Dropout(lstm_dropout)
        )

        # ==================== N-BEATS Blocks ====================
        self.nbeats_stack = NBeatsStack(
            n_blocks=nbeats_blocks,
            input_size=nbeats_hidden,
            hidden_size=nbeats_hidden,
            n_layers=nbeats_layers,
            theta_size=theta_size,
            dropout=nbeats_dropout
        )

        # ==================== Output Heads ====================
        # Trigger probability head (Sigmoid output)
        self.trigger_head = nn.Sequential(
            nn.Linear(nbeats_hidden, nbeats_hidden // 2),
            nn.GELU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(nbeats_hidden // 2, 1),
            nn.Sigmoid()
        )

        # Imminence score head (Sigmoid output)
        self.imminence_head = nn.Sequential(
            nn.Linear(nbeats_hidden, encoder_hidden),
            nn.GELU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(encoder_hidden, 1),
            nn.Sigmoid()
        )

        # Direction head (Softmax via CrossEntropy)
        self.direction_head = nn.Sequential(
            nn.Linear(nbeats_hidden, encoder_hidden),
            nn.GELU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(encoder_hidden, num_directions)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _extract_lstm_hidden(
        self,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        num_layers: int,
        hidden_size: int
    ) -> torch.Tensor:
        """
        Extract and concatenate bidirectional LSTM hidden states.

        Args:
            hidden: Tuple of (h_n, c_n) from LSTM
            num_layers: Number of LSTM layers
            hidden_size: Hidden size per direction

        Returns:
            Concatenated hidden state of shape (batch, hidden_size * 2)
        """
        h_n = hidden[0]  # (num_layers * 2, batch, hidden_size)

        # Reshape to (num_layers, 2, batch, hidden_size)
        h_n = h_n.view(num_layers, 2, -1, hidden_size)

        # Take last layer, concatenate forward and backward
        h_forward = h_n[-1, 0]  # (batch, hidden_size)
        h_backward = h_n[-1, 1]  # (batch, hidden_size)

        return torch.cat([h_forward, h_backward], dim=-1)

    def forward(
        self,
        x_5m: torch.Tensor,
        x_1h: torch.Tensor,
        tda_features: torch.Tensor,
        micro_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_5m: 5-minute OHLCV sequence (batch, seq_len_5m, 5)
            x_1h: 1-hour OHLCV sequence (batch, seq_len_1h, 5)
            tda_features: TDA features (batch, n_tda)
            micro_features: Microstructure features (batch, n_micro)

        Returns:
            Tuple of:
            - trigger_prob: (batch, 1) - Trigger probability
            - imminence: (batch, 1) - Imminence score
            - direction_logits: (batch, 3) - Direction logits
        """
        batch_size = x_5m.size(0)

        # ==================== LSTM Encoding ====================
        _, hidden_5m = self.lstm_5m(x_5m)
        h_5m = self._extract_lstm_hidden(hidden_5m, self.lstm_layers, self.lstm_hidden_5m)

        _, hidden_1h = self.lstm_1h(x_1h)
        h_1h = self._extract_lstm_hidden(hidden_1h, self.lstm_layers, self.lstm_hidden_1h)

        # ==================== Feature Encoding ====================
        tda_encoded = self.tda_encoder(tda_features)
        micro_encoded = self.micro_encoder(micro_features)

        # ==================== Temporal Fusion ====================
        fused = torch.cat([h_5m, h_1h, tda_encoded, micro_encoded], dim=-1)
        fused = self.temporal_fusion(fused)

        # ==================== N-BEATS Processing ====================
        residual, forecast = self.nbeats_stack(fused)

        # Combine fused representation with N-BEATS output
        final_repr = fused + residual

        # ==================== Output Heads ====================
        trigger_prob = self.trigger_head(final_repr)
        imminence = self.imminence_head(final_repr)
        direction_logits = self.direction_head(final_repr)

        return trigger_prob, imminence, direction_logits


class TriggerModelConfig:
    """Configuration class for TriggerPredictionModel."""

    def __init__(
        self,
        seq_len_5m: int = 72,
        seq_len_1h: int = 6,
        ohlcv_features: int = 5,
        tda_features: int = 9,
        micro_features: int = 12,
        lstm_hidden_5m: int = 128,
        lstm_hidden_1h: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        encoder_hidden: int = 64,
        encoder_dropout: float = 0.1,
        nbeats_blocks: int = 3,
        nbeats_hidden: int = 256,
        nbeats_layers: int = 4,
        theta_size: int = 32,
        nbeats_dropout: float = 0.1,
        num_directions: int = 3
    ):
        self.seq_len_5m = seq_len_5m
        self.seq_len_1h = seq_len_1h
        self.ohlcv_features = ohlcv_features
        self.tda_features = tda_features
        self.micro_features = micro_features
        self.lstm_hidden_5m = lstm_hidden_5m
        self.lstm_hidden_1h = lstm_hidden_1h
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.encoder_hidden = encoder_hidden
        self.encoder_dropout = encoder_dropout
        self.nbeats_blocks = nbeats_blocks
        self.nbeats_hidden = nbeats_hidden
        self.nbeats_layers = nbeats_layers
        self.theta_size = theta_size
        self.nbeats_dropout = nbeats_dropout
        self.num_directions = num_directions

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> 'TriggerModelConfig':
        return cls(**d)


def create_model(config: Optional[TriggerModelConfig] = None) -> TriggerPredictionModel:
    """
    Create TriggerPredictionModel from config.

    Args:
        config: Model configuration (uses defaults if None)

    Returns:
        TriggerPredictionModel instance
    """
    if config is None:
        config = TriggerModelConfig()

    return TriggerPredictionModel(**config.to_dict())


if __name__ == "__main__":
    # Test model
    print("Testing TriggerPredictionModel...")

    config = TriggerModelConfig()
    model = create_model(config)

    # Create dummy inputs
    batch_size = 32
    x_5m = torch.randn(batch_size, config.seq_len_5m, config.ohlcv_features)
    x_1h = torch.randn(batch_size, config.seq_len_1h, config.ohlcv_features)
    tda = torch.randn(batch_size, config.tda_features)
    micro = torch.randn(batch_size, config.micro_features)

    # Forward pass
    model.eval()
    with torch.no_grad():
        trigger_prob, imminence, direction_logits = model(x_5m, x_1h, tda, micro)

    print(f"\nInput shapes:")
    print(f"  x_5m: {x_5m.shape}")
    print(f"  x_1h: {x_1h.shape}")
    print(f"  tda: {tda.shape}")
    print(f"  micro: {micro.shape}")

    print(f"\nOutput shapes:")
    print(f"  trigger_prob: {trigger_prob.shape}")
    print(f"  imminence: {imminence.shape}")
    print(f"  direction_logits: {direction_logits.shape}")

    print(f"\nSample outputs:")
    print(f"  trigger_prob: {trigger_prob[0].item():.4f}")
    print(f"  imminence: {imminence[0].item():.4f}")
    print(f"  direction_logits: {direction_logits[0].tolist()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
