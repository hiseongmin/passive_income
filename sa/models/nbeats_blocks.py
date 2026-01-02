"""
N-BEATS Blocks for Classification

Adapted from N-BEATS (Neural Basis Expansion Analysis for Time Series)
for use in classification tasks.

Original paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
"""

import torch
import torch.nn as nn
from typing import Tuple


class NBeatsBlock(nn.Module):
    """
    N-BEATS block adapted for classification.

    Uses fully connected layers with residual learning.
    Produces backcast (residual) and forecast (representation) outputs.
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        n_layers: int = 4,
        theta_size: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            n_layers: Number of fully connected layers
            theta_size: Dimension of theta (basis coefficients)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.theta_size = theta_size

        # Build fully connected stack
        layers = []
        in_dim = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.fc_stack = nn.Sequential(*layers)

        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

        # Basis expansion (linear projection back to input size)
        self.backcast_basis = nn.Linear(theta_size, input_size)
        self.forecast_basis = nn.Linear(theta_size, input_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            Tuple of (backcast, forecast) each of shape (batch, input_size)
        """
        # Apply FC stack
        h = self.fc_stack(x)

        # Compute theta coefficients
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Basis expansion
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)

        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    Stack of N-BEATS blocks with residual connections.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        input_size: int = 256,
        hidden_size: int = 256,
        n_layers: int = 4,
        theta_size: int = 32,
        dropout: float = 0.1,
        share_weights: bool = False
    ):
        """
        Args:
            n_blocks: Number of N-BEATS blocks
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            n_layers: Number of FC layers per block
            theta_size: Theta dimension
            dropout: Dropout probability
            share_weights: Whether to share weights across blocks
        """
        super().__init__()

        self.n_blocks = n_blocks
        self.share_weights = share_weights

        if share_weights:
            # Single block with shared weights
            self.blocks = nn.ModuleList([
                NBeatsBlock(input_size, hidden_size, n_layers, theta_size, dropout)
            ])
        else:
            # Separate blocks
            self.blocks = nn.ModuleList([
                NBeatsBlock(input_size, hidden_size, n_layers, theta_size, dropout)
                for _ in range(n_blocks)
            ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            Tuple of (final_residual, accumulated_forecast)
        """
        residual = x
        forecast_sum = torch.zeros_like(x)

        for i in range(self.n_blocks):
            block_idx = 0 if self.share_weights else i
            backcast, forecast = self.blocks[block_idx](residual)

            # Update residual (subtract backcast)
            residual = residual - backcast

            # Accumulate forecasts
            forecast_sum = forecast_sum + forecast

        return residual, forecast_sum


class GenericNBeatsBlock(nn.Module):
    """
    Generic N-BEATS block with learnable basis functions.
    """

    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 256,
        hidden_size: int = 256,
        n_layers: int = 4,
        theta_size: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: Input dimension
            output_size: Output dimension
            hidden_size: Hidden layer dimension
            n_layers: Number of FC layers
            theta_size: Theta dimension
            dropout: Dropout probability
        """
        super().__init__()

        # FC stack
        layers = []
        in_dim = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.fc_stack = nn.Sequential(*layers)

        # Output projection
        self.theta = nn.Linear(hidden_size, theta_size)
        self.basis = nn.Linear(theta_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        h = self.fc_stack(x)
        theta = self.theta(h)
        output = self.basis(theta)
        return output


if __name__ == "__main__":
    # Test N-BEATS blocks
    batch_size = 32
    input_size = 256

    print("Testing N-BEATS blocks...")

    # Test single block
    block = NBeatsBlock(input_size=input_size)
    x = torch.randn(batch_size, input_size)
    backcast, forecast = block(x)

    print(f"Single block:")
    print(f"  Input shape: {x.shape}")
    print(f"  Backcast shape: {backcast.shape}")
    print(f"  Forecast shape: {forecast.shape}")

    # Test stack
    stack = NBeatsStack(n_blocks=3, input_size=input_size)
    residual, forecast_sum = stack(x)

    print(f"\nStack (3 blocks):")
    print(f"  Input shape: {x.shape}")
    print(f"  Final residual shape: {residual.shape}")
    print(f"  Forecast sum shape: {forecast_sum.shape}")

    # Test generic block
    generic = GenericNBeatsBlock(input_size=input_size, output_size=128)
    output = generic(x)

    print(f"\nGeneric block:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nSingle block parameters: {total_params:,}")

    total_params = sum(p.numel() for p in stack.parameters())
    print(f"Stack parameters: {total_params:,}")
