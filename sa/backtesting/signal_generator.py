"""
Signal Generator

Generates trading signals from model predictions.
"""

import numpy as np
import torch
from typing import Dict, Optional

import sys
sys.path.append('/notebooks/sa')

from models.trigger_model import TriggerPredictionModel, TriggerModelConfig, create_model
from .config import BacktestConfig


class SignalGenerator:
    """
    Generates trading signals from model predictions.
    """

    DIRECTION_MAP = {0: "LONG", 1: "SHORT", 2: "NONE"}

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.device = config.device
        self.model: Optional[TriggerPredictionModel] = None
        self.model_config: Optional[TriggerModelConfig] = None

        # Normalization params (should be loaded from training)
        self.ohlcv_mean: Optional[np.ndarray] = None
        self.ohlcv_std: Optional[np.ndarray] = None

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint (uses config if None)
        """
        path = checkpoint_path or self.config.model_checkpoint

        print(f"Loading model from {path}...")

        checkpoint = torch.load(path, map_location=self.device)

        # Load model config if saved
        if 'model_config' in checkpoint:
            self.model_config = TriggerModelConfig.from_dict(checkpoint['model_config'])
        else:
            self.model_config = TriggerModelConfig()

        # Create and load model
        self.model = create_model(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization params if available
        if 'normalization' in checkpoint:
            norm = checkpoint['normalization']
            self.ohlcv_mean = norm.get('ohlcv_mean')
            self.ohlcv_std = norm.get('ohlcv_std')

        print(f"  Model loaded successfully")
        print(f"  Device: {self.device}")

    def _normalize_ohlcv(self, x: np.ndarray) -> np.ndarray:
        """Normalize OHLCV data."""
        if self.ohlcv_mean is not None and self.ohlcv_std is not None:
            return (x - self.ohlcv_mean) / (self.ohlcv_std + 1e-8)

        # Simple normalization: scale by first close price
        # This is a common approach for price data
        scale = x[0, 3] if x[0, 3] != 0 else 1.0  # Close price
        return x / scale - 1.0

    @torch.no_grad()
    def predict(
        self,
        x_5m: np.ndarray,
        x_1h: np.ndarray,
        tda_features: np.ndarray,
        micro_features: np.ndarray
    ) -> Dict:
        """
        Run model inference for a single timestep.

        Args:
            x_5m: 5-minute OHLCV sequence (seq_len_5m, 5)
            x_1h: 1-hour OHLCV sequence (seq_len_1h, 5)
            tda_features: TDA features (9,)
            micro_features: Microstructure features (12,)

        Returns:
            Dictionary with:
            - trigger_prob: float (0-1)
            - imminence: float (0-1)
            - direction: str ("LONG", "SHORT", "NONE")
            - direction_probs: List[float] (softmax probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Normalize OHLCV
        x_5m_norm = self._normalize_ohlcv(x_5m)
        x_1h_norm = self._normalize_ohlcv(x_1h)

        # Convert to tensors and add batch dimension
        x_5m_t = torch.tensor(x_5m_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_1h_t = torch.tensor(x_1h_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        tda_t = torch.tensor(tda_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        micro_t = torch.tensor(micro_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass
        trigger_prob, imminence, direction_logits = self.model(
            x_5m_t, x_1h_t, tda_t, micro_t
        )

        # Extract values
        trigger_prob_val = trigger_prob[0, 0].item()
        imminence_val = imminence[0, 0].item()
        direction_probs = torch.softmax(direction_logits[0], dim=0).cpu().numpy()
        direction_idx = int(np.argmax(direction_probs))

        return {
            'trigger_prob': trigger_prob_val,
            'imminence': imminence_val,
            'direction': self.DIRECTION_MAP[direction_idx],
            'direction_idx': direction_idx,
            'direction_probs': direction_probs.tolist(),
        }

    def generate_signal(self, prediction: Dict) -> Optional[Dict]:
        """
        Convert prediction to trading signal based on thresholds.

        Args:
            prediction: Output from predict()

        Returns:
            Signal dict with direction and confidence, or None (no signal)
        """
        # Check trigger threshold
        if prediction['trigger_prob'] < self.config.trigger_threshold:
            return None

        # Check imminence threshold
        if prediction['imminence'] < self.config.imminence_threshold:
            return None

        # Check direction
        direction = prediction['direction']
        if direction == "NONE":
            return None

        # Get direction confidence
        direction_idx = 0 if direction == "LONG" else 1
        direction_confidence = prediction['direction_probs'][direction_idx]

        return {
            'direction': direction,
            'trigger_prob': prediction['trigger_prob'],
            'imminence': prediction['imminence'],
            'direction_confidence': direction_confidence,
        }
