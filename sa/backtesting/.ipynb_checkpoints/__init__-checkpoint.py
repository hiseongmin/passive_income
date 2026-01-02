"""
Backtesting Module for Trigger-Based Trading Strategy

Components:
- BacktestConfig: Configuration dataclass
- BacktestDataLoader: Data loading and temporal alignment
- IncrementalFeatureExtractor: Feature extraction (no look-ahead)
- SignalGenerator: Model inference
- Position, Trade: Trade management
- ExecutionEngine: Order execution with fees
- Portfolio: State management
- BacktestEngine: Main orchestrator
- BacktestMetrics: Performance calculation
- BacktestVisualizer: Charts and reports
"""

from .config import BacktestConfig
from .position import Position, Trade
from .execution import ExecutionEngine
from .data_loader import BacktestDataLoader
from .feature_pipeline import IncrementalFeatureExtractor
from .signal_generator import SignalGenerator
from .portfolio import Portfolio
from .backtest_engine import BacktestEngine
from .metrics import BacktestMetrics
from .visualizer import BacktestVisualizer

__all__ = [
    'BacktestConfig',
    'BacktestDataLoader',
    'IncrementalFeatureExtractor',
    'SignalGenerator',
    'Position',
    'Trade',
    'ExecutionEngine',
    'Portfolio',
    'BacktestEngine',
    'BacktestMetrics',
    'BacktestVisualizer',
]
