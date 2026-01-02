"""
Market Microstructure Feature Extraction

Extracts trading-specific features from OHLCV and order flow data:
- Volatility measures (ATR, standard deviation)
- Volume analysis (relative volume, buy/sell imbalance)
- Momentum indicators
- Market regime classification
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from tqdm import tqdm


class MarketMicrostructureExtractor:
    """
    Extracts market microstructure features from OHLCV data.

    Features:
    - ATR percentage
    - Volatility (standard deviation)
    - Candle range MA
    - Relative volume
    - Volume intensity
    - Buy/sell imbalance
    - Momentum
    - Efficiency ratio
    - Velocity score
    - Body ratio
    - Thickness score
    - Market regime
    """

    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: Lookback period for rolling calculations
        """
        self.lookback = lookback
        self.n_features = 12

    def compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Compute Average True Range percentage."""
        n = len(high)
        tr = np.zeros(n)

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        atr = pd.Series(tr).rolling(self.lookback, min_periods=1).mean().values
        atr_pct = atr / (close + 1e-10) * 100

        return atr_pct

    def compute_volatility(self, close: np.ndarray) -> np.ndarray:
        """Compute rolling standard deviation of returns."""
        returns = np.zeros(len(close))
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)

        volatility = pd.Series(returns).rolling(self.lookback, min_periods=1).std().values
        return volatility

    def compute_candle_range_ma(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Compute moving average of candle range as percentage."""
        candle_range = (high - low) / (close + 1e-10) * 100
        range_ma = pd.Series(candle_range).rolling(self.lookback, min_periods=1).mean().values
        return range_ma

    def compute_relative_volume(self, volume: np.ndarray) -> np.ndarray:
        """Compute volume relative to rolling average."""
        vol_ma = pd.Series(volume).rolling(self.lookback, min_periods=1).mean().values
        relative_vol = volume / (vol_ma + 1e-10)
        return relative_vol

    def compute_volume_intensity(self, volume: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Compute volume intensity (volume per price range)."""
        price_range = high - low + 1e-10
        intensity = volume / price_range
        intensity_normalized = intensity / (pd.Series(intensity).rolling(self.lookback, min_periods=1).mean().values + 1e-10)
        return intensity_normalized

    def compute_buy_sell_imbalance(self, buy_volume: np.ndarray, sell_volume: np.ndarray) -> np.ndarray:
        """Compute buy/sell volume imbalance."""
        total_volume = buy_volume + sell_volume + 1e-10
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance

    def compute_momentum(self, close: np.ndarray) -> np.ndarray:
        """Compute price momentum."""
        momentum = np.zeros(len(close))
        momentum[self.lookback:] = (close[self.lookback:] - close[:-self.lookback]) / (close[:-self.lookback] + 1e-10)
        return momentum

    def compute_efficiency_ratio(self, close: np.ndarray) -> np.ndarray:
        """Compute Kaufman's efficiency ratio."""
        n = len(close)
        er = np.zeros(n)

        for i in range(self.lookback, n):
            direction = abs(close[i] - close[i - self.lookback])
            volatility = np.sum(np.abs(np.diff(close[i - self.lookback:i + 1])))
            er[i] = direction / (volatility + 1e-10)

        return er

    def compute_velocity_score(self, close: np.ndarray) -> np.ndarray:
        """Compute price velocity (rate of change smoothed)."""
        roc = np.zeros(len(close))
        roc[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)

        velocity = pd.Series(roc).rolling(self.lookback, min_periods=1).mean().values
        return velocity

    def compute_body_ratio(self, open_: np.ndarray, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Compute candle body ratio (body size / total range)."""
        body = np.abs(close - open_)
        total_range = high - low + 1e-10
        body_ratio = body / total_range
        return body_ratio

    def compute_thickness_score(self, volume: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Compute market thickness score."""
        thickness = volume / (high - low + 1e-10)
        thickness_ma = pd.Series(thickness).rolling(self.lookback, min_periods=1).mean().values
        thickness_score = thickness / (thickness_ma + 1e-10)
        return thickness_score

    def compute_market_regime(self, close: np.ndarray) -> np.ndarray:
        """
        Classify market regime.

        Returns:
            0: Ranging
            1: Trending up
            2: Trending down
        """
        n = len(close)
        regime = np.zeros(n)

        # Use simple moving averages
        sma_short = pd.Series(close).rolling(self.lookback // 2, min_periods=1).mean().values
        sma_long = pd.Series(close).rolling(self.lookback, min_periods=1).mean().values

        # Trend threshold
        threshold = 0.001

        for i in range(n):
            diff = (sma_short[i] - sma_long[i]) / (sma_long[i] + 1e-10)

            if diff > threshold:
                regime[i] = 1  # Trending up
            elif diff < -threshold:
                regime[i] = 2  # Trending down
            else:
                regime[i] = 0  # Ranging

        return regime

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract all microstructure features from DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume, buy_volume, sell_volume

        Returns:
            Array of shape (n_samples, n_features)
        """
        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float32)

        # Extract columns
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        # Handle optional columns
        if 'buy_volume' in df.columns:
            buy_volume = df['buy_volume'].values
        else:
            buy_volume = volume * 0.5

        if 'sell_volume' in df.columns:
            sell_volume = df['sell_volume'].values
        else:
            sell_volume = volume * 0.5

        # Compute features
        features[:, 0] = self.compute_atr(high, low, close)
        features[:, 1] = self.compute_volatility(close)
        features[:, 2] = self.compute_candle_range_ma(high, low, close)
        features[:, 3] = self.compute_relative_volume(volume)
        features[:, 4] = self.compute_volume_intensity(volume, high, low)
        features[:, 5] = self.compute_buy_sell_imbalance(buy_volume, sell_volume)
        features[:, 6] = self.compute_momentum(close)
        features[:, 7] = self.compute_efficiency_ratio(close)
        features[:, 8] = self.compute_velocity_score(close)
        features[:, 9] = self.compute_body_ratio(open_, close, high, low)
        features[:, 10] = self.compute_thickness_score(volume, high, low)
        features[:, 11] = self.compute_market_regime(close)

        # Handle NaN and Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'atr_pct',
            'volatility_std',
            'candle_range_ma',
            'relative_volume',
            'volume_intensity',
            'buy_sell_imbalance',
            'momentum',
            'efficiency_ratio',
            'velocity_score',
            'body_ratio',
            'thickness_score',
            'market_regime'
        ]


def extract_microstructure_features(
    df: pd.DataFrame,
    lookback: int = 20
) -> np.ndarray:
    """
    Extract microstructure features from DataFrame.

    Args:
        df: DataFrame with OHLCV data
        lookback: Lookback period for rolling calculations

    Returns:
        Array of shape (n_samples, 12)
    """
    extractor = MarketMicrostructureExtractor(lookback=lookback)
    features = extractor.extract_features(df)
    return features


if __name__ == "__main__":
    # Test microstructure feature extraction
    import os

    DATA_DIR = '/notebooks/sa/data'
    data_path = os.path.join(DATA_DIR, 'BTCUSDT_perp_5m_labeled.csv')

    if os.path.exists(data_path):
        print("Loading data...")
        df = pd.read_csv(data_path)

        print("Extracting microstructure features...")
        extractor = MarketMicrostructureExtractor()
        features = extractor.extract_features(df)

        print(f"\nFeatures shape: {features.shape}")
        print(f"Feature names: {extractor.get_feature_names()}")
        print(f"\nSample features (first 5 rows):")
        print(features[:5])

        print(f"\nFeature statistics:")
        for i, name in enumerate(extractor.get_feature_names()):
            print(f"  {name}: mean={features[:, i].mean():.4f}, std={features[:, i].std():.4f}")
    else:
        print(f"Data not found at {data_path}")
        print("Run trigger_generator.py first.")
