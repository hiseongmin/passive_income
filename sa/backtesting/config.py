"""
Backtest Configuration

All configuration parameters for backtesting.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Master configuration for backtesting."""

    # Data paths
    data_5m_path: str = "/notebooks/data/BTCUSDT_perp_last_90d.csv"
    data_1h_path: str = "/notebooks/data/BTCUSDT_perp_1h_last_90d.csv"
    model_checkpoint: str = "/notebooks/sa/checkpoints/best_model.pt"

    # Model sequence lengths (must match model config)
    seq_len_5m: int = 72   # 6 hours of 5-min data
    seq_len_1h: int = 6    # 6 hours of 1h data

    # Feature extraction
    tda_window_size: int = 72
    micro_lookback: int = 20

    # Signal thresholds
    trigger_threshold: float = 0.6
    imminence_threshold: float = 0.5

    # Trading parameters
    take_profit_pct: float = 0.02   # 2% TP
    stop_loss_pct: float = 0.01     # 1% SL

    # Fee structure (Binance Futures)
    maker_fee: float = 0.0002       # 0.02%
    taker_fee: float = 0.0005       # 0.05%
    slippage_pct: float = 0.0001    # 0.01% average slippage

    # Position sizing
    base_position_size: float = 1.0
    use_confidence_sizing: bool = True  # size = base * trigger_prob * imminence
    max_position_size: float = 1.0
    min_position_size: float = 0.1

    # Capital
    initial_capital: float = 10000.0

    # Output
    output_dir: str = "./backtest_results"
    save_trades_csv: bool = True
    save_metrics_json: bool = True
    save_report_png: bool = True

    # Device
    device: str = "cuda"  # or "cpu"

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.trigger_threshold <= 1, "trigger_threshold must be in (0, 1]"
        assert 0 < self.imminence_threshold <= 1, "imminence_threshold must be in (0, 1]"
        assert self.take_profit_pct > 0, "take_profit_pct must be positive"
        assert self.stop_loss_pct > 0, "stop_loss_pct must be positive"
        assert self.initial_capital > 0, "initial_capital must be positive"
