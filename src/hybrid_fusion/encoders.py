"""
Self-contained encoder implementations for Hybrid Fusion model.

This module provides all encoders needed for end-to-end training:
- OHLCVNBEATSEncoder: N-BEATS based encoder for OHLCV sequences
- TDAEncoder: MLP encoder for TDA features
- ComplexityEncoder: MLP encoder for complexity indicators

These are adapted from the original tda_model to make hybrid_fusion self-contained.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# N-BEATS COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════


class NBEATSBlock(nn.Module):
    """
    Basic N-BEATS block with fully connected layers.

    Each block produces:
    - backcast: Reconstruction of input (for residual connection)
    - forecast: Prediction output (used as features)
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size

        # Build FC stack
        layers = []
        current_size = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size

        self.fc_stack = nn.Sequential(*layers)
        self.theta_b_fc = nn.Linear(hidden_size, theta_size)
        self.theta_f_fc = nn.Linear(hidden_size, theta_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(x)
        theta_b = self.theta_b_fc(h)
        theta_f = self.theta_f_fc(h)
        return theta_b, theta_f


class GenericBlock(NBEATSBlock):
    """Generic N-BEATS block with learnable basis expansion."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        theta_size: int = 32,
    ):
        super().__init__(
            input_size=input_size,
            theta_size=theta_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.output_size = output_size
        self.backcast_fc = nn.Linear(theta_size, input_size)
        self.forecast_fc = nn.Linear(theta_size, output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_b, theta_f = super().forward(x)
        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)
        return backcast, forecast


class TrendBlock(NBEATSBlock):
    """Trend N-BEATS block with polynomial basis expansion."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        degree: int = 3,
    ):
        super().__init__(
            input_size=input_size,
            theta_size=degree + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.output_size = output_size
        self.degree = degree

        # Pre-compute polynomial basis
        backcast_t = torch.linspace(0, 1, input_size).unsqueeze(0)
        forecast_t = torch.linspace(0, 1, output_size).unsqueeze(0)

        backcast_basis = torch.stack([backcast_t ** i for i in range(degree + 1)], dim=1)
        forecast_basis = torch.stack([forecast_t ** i for i in range(degree + 1)], dim=1)

        self.register_buffer('backcast_basis', backcast_basis.squeeze(0))
        self.register_buffer('forecast_basis', forecast_basis.squeeze(0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_b, theta_f = super().forward(x)
        backcast = torch.mm(theta_b, self.backcast_basis)
        forecast = torch.mm(theta_f, self.forecast_basis)
        return backcast, forecast


class SeasonalityBlock(NBEATSBlock):
    """Seasonality N-BEATS block with Fourier basis expansion."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_harmonics: int = 5,
    ):
        super().__init__(
            input_size=input_size,
            theta_size=2 * num_harmonics,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.output_size = output_size
        self.num_harmonics = num_harmonics

        # Pre-compute Fourier basis
        backcast_t = torch.linspace(0, 2 * math.pi, input_size).unsqueeze(0)
        forecast_t = torch.linspace(0, 2 * math.pi, output_size).unsqueeze(0)

        backcast_basis = []
        forecast_basis = []
        for i in range(1, num_harmonics + 1):
            backcast_basis.extend([torch.cos(i * backcast_t), torch.sin(i * backcast_t)])
            forecast_basis.extend([torch.cos(i * forecast_t), torch.sin(i * forecast_t)])

        self.register_buffer('backcast_basis', torch.cat(backcast_basis, dim=0))
        self.register_buffer('forecast_basis', torch.cat(forecast_basis, dim=0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_b, theta_f = super().forward(x)
        backcast = torch.mm(theta_b, self.backcast_basis)
        forecast = torch.mm(theta_f, self.forecast_basis)
        return backcast, forecast


class NBEATSStack(nn.Module):
    """Stack of N-BEATS blocks with doubly residual connections."""

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        **block_kwargs,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.input_size = input_size
        self.output_size = output_size

        blocks = []
        for _ in range(num_blocks):
            if block_type == 'generic':
                block = GenericBlock(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    theta_size=block_kwargs.get('theta_size', 32),
                )
            elif block_type == 'trend':
                block = TrendBlock(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    degree=block_kwargs.get('degree', 3),
                )
            elif block_type == 'seasonality':
                block = SeasonalityBlock(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    num_harmonics=block_kwargs.get('num_harmonics', 5),
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        forecast = torch.zeros(x.size(0), self.output_size, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return residual, forecast


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for feature weighting."""

    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Ensure divisibility
        if input_dim % num_heads != 0:
            for nh in [8, 4, 2, 1]:
                if input_dim % nh == 0:
                    num_heads = nh
                    break

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        out = self.norm(x + attn_out)

        if squeeze_output:
            out = out.squeeze(1)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENCODERS
# ═══════════════════════════════════════════════════════════════════════════════


class OHLCVNBEATSEncoder(nn.Module):
    """
    N-BEATS encoder for OHLCV + Volume + Technical Indicator sequences.

    Processes multi-channel time series through N-BEATS stacks with
    trend, seasonality, and generic decomposition.

    Input channels (14 total):
    - OHLC: 4 (open, high, low, close)
    - Volume: 5 (volume, buy_volume, sell_volume, volume_delta, cvd)
    - Technical: 5 (RSI, MACD, BB_pctB, ATR, MOM)
    """

    def __init__(
        self,
        input_size: int = 96,
        output_size: int = 1,
        hidden_size: int = 256,
        num_stacks: int = 4,
        num_blocks: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        stack_types: Optional[List[str]] = None,
        num_channels: int = 14,
        use_attention: bool = True,
    ):
        super().__init__()

        if stack_types is None:
            stack_types = ['trend', 'seasonality', 'generic', 'generic']

        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size

        # Feature normalization
        self.feature_norm = nn.BatchNorm1d(num_channels)

        # Flattened input size
        flat_input_size = input_size * num_channels

        # Create stacks
        stacks = []
        for stack_type in stack_types:
            stack = NBEATSStack(
                block_type=stack_type,
                num_blocks=num_blocks,
                input_size=flat_input_size,
                output_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            stacks.append(stack)

        self.stacks = nn.ModuleList(stacks)
        self.num_stacks = len(stack_types)

        # Combined dimension
        combined_dim = hidden_size * self.num_stacks

        # Attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(combined_dim, num_heads=8, dropout=dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_channels) input sequence

        Returns:
            encoded: (batch, output_dim) encoded features
        """
        batch_size = x.size(0)

        # Normalize: (B, seq, channels) -> (B, channels, seq) -> normalize -> back
        x = x.permute(0, 2, 1)
        x = self.feature_norm(x)
        x = x.permute(0, 2, 1)

        # Flatten: (B, seq, channels) -> (B, seq * channels)
        x_flat = x.reshape(batch_size, -1)

        # Process through stacks
        forecasts = []
        residual = x_flat
        for stack in self.stacks:
            residual, forecast = stack(residual)
            forecasts.append(forecast)

        # Concatenate stack outputs
        combined = torch.cat(forecasts, dim=1)

        # Apply attention
        if self.use_attention:
            combined = self.attention(combined)

        # Project to output dimension
        return self.output_proj(combined)


class TDAEncoder(nn.Module):
    """
    MLP encoder for TDA (Topological Data Analysis) features.

    Encodes topological features including:
    - Betti curves
    - Persistent entropy
    - Persistence statistics
    - Persistence landscapes
    """

    def __init__(
        self,
        input_dim: int = 214,
        output_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) TDA features

        Returns:
            encoded: (batch, output_dim) encoded TDA features
        """
        return self.encoder(x)


class ComplexityEncoder(nn.Module):
    """
    MLP encoder for market complexity indicators.

    Encodes 6 complexity indicators:
    1. MA Separation - Trend direction/strength
    2. Bollinger Width - Volatility regime
    3. Price Efficiency - Mean reversion signals
    4. Support Reaction - S/R strength
    5. Directional Result - Momentum
    6. Volume-Price Alignment - Confirmation signals
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) complexity indicators

        Returns:
            encoded: (batch, output_dim) encoded complexity features
        """
        return self.encoder(x)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def count_encoder_parameters(encoder: nn.Module) -> int:
    """Count trainable parameters in an encoder."""
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)


def create_encoders(
    # OHLCV encoder config
    ohlcv_seq_length: int = 96,
    ohlcv_num_channels: int = 14,
    ohlcv_hidden_size: int = 256,
    ohlcv_num_stacks: int = 4,
    ohlcv_num_blocks: int = 4,
    ohlcv_dropout: float = 0.1,
    # TDA encoder config
    tda_input_dim: int = 214,
    tda_output_dim: int = 256,
    tda_dropout: float = 0.3,
    # Complexity encoder config
    complexity_input_dim: int = 6,
    complexity_output_dim: int = 64,
    complexity_dropout: float = 0.1,
) -> Tuple[OHLCVNBEATSEncoder, TDAEncoder, ComplexityEncoder]:
    """
    Create all three encoders with specified configurations.

    Returns:
        Tuple of (ohlcv_encoder, tda_encoder, complexity_encoder)
    """
    ohlcv_encoder = OHLCVNBEATSEncoder(
        input_size=ohlcv_seq_length,
        hidden_size=ohlcv_hidden_size,
        num_stacks=ohlcv_num_stacks,
        num_blocks=ohlcv_num_blocks,
        dropout=ohlcv_dropout,
        num_channels=ohlcv_num_channels,
    )

    tda_encoder = TDAEncoder(
        input_dim=tda_input_dim,
        output_dim=tda_output_dim,
        dropout=tda_dropout,
    )

    complexity_encoder = ComplexityEncoder(
        input_dim=complexity_input_dim,
        output_dim=complexity_output_dim,
        dropout=complexity_dropout,
    )

    return ohlcv_encoder, tda_encoder, complexity_encoder
