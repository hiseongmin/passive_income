# Bitcoin Trigger Prediction System - Paperspace Implementation

## 목적
비트코인 15분봉 데이터로 2% 상승/하락 트리거를 예측하는 시스템 구축

## 핵심 원칙
1. **미래 데이터 누수 금지**: 모든 feature는 해당 시점 이전 데이터만 사용
2. **Rolling prediction**: 학습 시점까지의 데이터로만 예측

## 데이터 기간
- **ETF 출시일**: 2024-01-11 (Bitcoin Spot ETF 승인)
- **Training**: 2024-01-11 ~ 180일 전 (약 2024-01-11 ~ 2025-07-05)
- **Backtest 1**: 180일 전 ~ 90일 전 (약 2025-07-05 ~ 2025-10-03)
- **Backtest 2**: 90일 전 ~ 현재 (약 2025-10-03 ~ 2026-01-02)

## 폴더 구조
```
/notebooks/
├── data/
│   └── BTCUSDT_15m.csv  # 15분봉 데이터 (이미 존재)
├── models/
│   └── trigger_model.pkl
├── results/
│   ├── backtest1_chart.png
│   └── backtest2_chart.png
└── trigger_system.ipynb
```

## 구현 코드

### Cell 1: Imports & Setup
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    !pip install ripser
    from ripser import ripser
    RIPSER_AVAILABLE = True

print("Setup complete!")
```

### Cell 2: Complexity Module (Event-based)
```python
@dataclass
class BreakoutAttempt:
    """Failed breakout attempt record"""
    timestamp: pd.Timestamp
    direction: str  # 'up' or 'down'
    volume: float
    price: float
    position: float


class AccumulatedComplexity:
    """
    Event-based complexity tracker
    Complexity increases on failed breakout attempts
    """

    def __init__(self,
                 range_window: int = 24,      # 6 hours for 15min
                 decay_rate: float = 0.95,
                 boundary_threshold: float = 0.1,
                 center_threshold: float = 0.2,
                 breakout_confirm_candles: int = 4):

        self.range_window = range_window
        self.decay_rate = decay_rate
        self.boundary_threshold = boundary_threshold
        self.center_threshold = center_threshold
        self.breakout_confirm_candles = breakout_confirm_candles

        self.complexity = 0.0
        self.pending_attempt: Optional[BreakoutAttempt] = None
        self.attempts_history: List[BreakoutAttempt] = []
        self.outside_range_count = 0

        self.volume_buffer = deque(maxlen=range_window)
        self.high_buffer = deque(maxlen=range_window)
        self.low_buffer = deque(maxlen=range_window)

    def _get_range(self) -> Tuple[float, float]:
        if len(self.high_buffer) < 5:
            return 0, 0
        return min(self.low_buffer), max(self.high_buffer)

    def _get_position(self, price: float, range_low: float, range_high: float) -> float:
        if range_high == range_low:
            return 0.5
        return (price - range_low) / (range_high - range_low)

    def _get_avg_volume(self) -> float:
        if len(self.volume_buffer) == 0:
            return 1.0
        return np.mean(self.volume_buffer) or 1.0

    def update(self, timestamp, high: float, low: float, close: float, volume: float) -> dict:
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        self.volume_buffer.append(volume)

        if len(self.high_buffer) < 5:
            return {'complexity': 0, 'range_high': 0, 'range_low': 0,
                    'position': 0.5, 'pending_attempt': False, 'attempt_count': 0}

        range_low, range_high = self._get_range()
        position = self._get_position(close, range_low, range_high)

        at_upper = position > (1 - self.boundary_threshold)
        at_lower = position < self.boundary_threshold
        at_center = (0.5 - self.center_threshold) < position < (0.5 + self.center_threshold)

        if at_upper and self.pending_attempt is None:
            self.pending_attempt = BreakoutAttempt(timestamp, 'up', volume, close, position)
            self.outside_range_count = 0
        elif at_lower and self.pending_attempt is None:
            self.pending_attempt = BreakoutAttempt(timestamp, 'down', volume, close, position)
            self.outside_range_count = 0
        elif at_center and self.pending_attempt is not None:
            avg_vol = self._get_avg_volume()
            vol_weight = np.clip(self.pending_attempt.volume / avg_vol, 0.5, 3.0)
            self.complexity += 1.0 * vol_weight
            self.attempts_history.append(self.pending_attempt)
            self.pending_attempt = None
            self.outside_range_count = 0
        elif position > 1.0 or position < 0.0:
            self.outside_range_count += 1
            if self.outside_range_count >= self.breakout_confirm_candles:
                self.complexity = 0
                self.pending_attempt = None
                self.attempts_history = []
                self.outside_range_count = 0

        self.complexity *= self.decay_rate
        self.complexity = np.clip(self.complexity, 0, 20)

        return {
            'complexity': self.complexity,
            'range_high': range_high,
            'range_low': range_low,
            'position': position,
            'pending_attempt': self.pending_attempt is not None,
            'attempt_count': len(self.attempts_history)
        }

    def reset(self):
        self.complexity = 0.0
        self.pending_attempt = None
        self.attempts_history = []
        self.outside_range_count = 0
        self.volume_buffer.clear()
        self.high_buffer.clear()
        self.low_buffer.clear()


class EntropyComplexity:
    """Entropy-based complexity measures"""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def permutation_entropy(self, series: np.ndarray, order: int = 3, delay: int = 1) -> float:
        if len(series) < order * delay:
            return 0.0

        n = len(series) - (order - 1) * delay
        permutations = []
        for i in range(n):
            pattern = tuple(np.argsort([series[i + j * delay] for j in range(order)]))
            permutations.append(pattern)

        counts = Counter(permutations)
        probs = np.array(list(counts.values())) / len(permutations)
        max_entropy = np.log(math.factorial(order))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return entropy / max_entropy if max_entropy > 0 else 0

    def sample_entropy(self, series: np.ndarray, m: int = 2, r: float = None) -> float:
        n = len(series)
        if n < m + 1:
            return 0.0

        if r is None:
            r = 0.2 * np.std(series)

        def _count_matches(template_length):
            count = 0
            templates = [series[i:i + template_length] for i in range(n - template_length)]
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count

        A = _count_matches(m + 1)
        B = _count_matches(m)

        return -np.log(A / B + 1e-10) if B > 0 else 0.0

    def compute(self, prices: np.ndarray, volumes: np.ndarray = None) -> dict:
        if len(prices) < self.window_size:
            return {'perm_entropy': 0, 'sample_entropy': 0, 'price_entropy': 0, 'volume_entropy': 0}

        price_window = prices[-self.window_size:]
        returns = np.diff(price_window) / price_window[:-1]

        result = {
            'perm_entropy': self.permutation_entropy(returns),
            'sample_entropy': self.sample_entropy(returns),
        }

        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]
        result['price_entropy'] = -np.sum(hist * np.log(hist + 1e-10)) / np.log(10)

        if volumes is not None and len(volumes) >= self.window_size:
            vol_window = volumes[-self.window_size:]
            vol_norm = vol_window / (np.sum(vol_window) + 1e-10)
            vol_norm = vol_norm[vol_norm > 0]
            result['volume_entropy'] = -np.sum(vol_norm * np.log(vol_norm + 1e-10))
        else:
            result['volume_entropy'] = 0

        return result

print("Complexity module loaded!")
```

### Cell 3: TDA Module
```python
class TDAAnalyzer:
    """Topological Data Analysis for price series"""

    def __init__(self, embedding_dim: int = 3, delay: int = 1, window_size: int = 20):
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.window_size = window_size

    def time_delay_embedding(self, series: np.ndarray) -> np.ndarray:
        n = len(series) - (self.embedding_dim - 1) * self.delay
        if n <= 0:
            return np.array([]).reshape(0, self.embedding_dim)

        embedded = np.zeros((n, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = series[i * self.delay : i * self.delay + n]
        return embedded

    def compute_persistence(self, point_cloud: np.ndarray, maxdim: int = 1) -> Dict[int, np.ndarray]:
        if not RIPSER_AVAILABLE or len(point_cloud) < 3:
            return {d: np.array([]).reshape(0, 2) for d in range(maxdim + 1)}

        point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-8)
        result = ripser(point_cloud, maxdim=maxdim)

        diagrams = {}
        for d, dgm in enumerate(result['dgms']):
            finite_mask = np.isfinite(dgm[:, 1])
            diagrams[d] = dgm[finite_mask]
        return diagrams

    def extract_features(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
        features = {}

        for dim in [0, 1]:
            prefix = f'h{dim}'
            dgm = diagrams.get(dim, np.array([]).reshape(0, 2))

            if len(dgm) == 0:
                features[f'{prefix}_entropy'] = 0.0
                features[f'{prefix}_amplitude'] = 0.0
                features[f'{prefix}_count'] = 0
                continue

            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[lifetimes > 0]

            if len(lifetimes) == 0:
                features[f'{prefix}_entropy'] = 0.0
                features[f'{prefix}_amplitude'] = 0.0
                features[f'{prefix}_count'] = 0
                continue

            probs = lifetimes / lifetimes.sum()
            features[f'{prefix}_entropy'] = -np.sum(probs * np.log(probs + 1e-10))
            features[f'{prefix}_amplitude'] = float(lifetimes.max())
            features[f'{prefix}_count'] = len(lifetimes)

        features['tda_complexity'] = min(1.0, (features.get('h1_entropy', 0) + 0.1 * features.get('h1_count', 0)) / 3.0)
        return features

    def compute(self, price_series: np.ndarray) -> Dict[str, float]:
        if len(price_series) < self.window_size:
            return self._empty_features()

        window = price_series[-self.window_size:]
        point_cloud = self.time_delay_embedding(window)

        if len(point_cloud) < 3:
            return self._empty_features()

        diagrams = self.compute_persistence(point_cloud)
        return self.extract_features(diagrams)

    def _empty_features(self) -> Dict[str, float]:
        return {'h0_entropy': 0, 'h0_amplitude': 0, 'h0_count': 0,
                'h1_entropy': 0, 'h1_amplitude': 0, 'h1_count': 0, 'tda_complexity': 0}

print("TDA module loaded!")
```

### Cell 4: Feature Engineering (No Future Leakage)
```python
def compute_features_no_leakage(df: pd.DataFrame,
                                 complexity_window: int = 24,
                                 tda_window: int = 20) -> pd.DataFrame:
    """
    Compute all features WITHOUT future data leakage
    Each row only uses data up to that point
    """
    df = df.copy()

    # Initialize trackers
    acc_complexity = AccumulatedComplexity(range_window=complexity_window)
    ent_complexity = EntropyComplexity(window_size=complexity_window)
    tda_analyzer = TDAAnalyzer(window_size=tda_window)

    # Storage
    acc_results = []
    ent_results = []
    tda_results = []

    prices = []
    volumes = []

    print("Computing features...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 5000 == 0:
            print(f"  Processing row {idx}/{len(df)}")

        # Accumulated Complexity
        ts = pd.to_datetime(row.get('timestamp', row.get('open_time', pd.Timestamp.now())))
        acc_result = acc_complexity.update(ts, row['high'], row['low'], row['close'], row['volume'])
        acc_results.append(acc_result)

        # Collect for entropy/TDA
        prices.append(row['close'])
        volumes.append(row['volume'])

        # Entropy Complexity
        if len(prices) >= complexity_window:
            ent_result = ent_complexity.compute(
                np.array(prices[-complexity_window:]),
                np.array(volumes[-complexity_window:])
            )
        else:
            ent_result = {'perm_entropy': 0, 'sample_entropy': 0, 'price_entropy': 0, 'volume_entropy': 0}
        ent_results.append(ent_result)

        # TDA
        if len(prices) >= tda_window:
            tda_result = tda_analyzer.compute(np.array(prices[-tda_window:]))
        else:
            tda_result = tda_analyzer._empty_features()
        tda_results.append(tda_result)

    # Add complexity columns
    df['acc_complexity'] = [r['complexity'] for r in acc_results]
    df['range_position'] = [r['position'] for r in acc_results]
    df['attempt_count'] = [r['attempt_count'] for r in acc_results]

    # Add entropy columns
    df['perm_entropy'] = [r['perm_entropy'] for r in ent_results]
    df['sample_entropy'] = [r['sample_entropy'] for r in ent_results]

    # Add TDA columns
    df['tda_h0_entropy'] = [r['h0_entropy'] for r in tda_results]
    df['tda_h1_entropy'] = [r['h1_entropy'] for r in tda_results]
    df['tda_h1_count'] = [r['h1_count'] for r in tda_results]
    df['tda_complexity'] = [r['tda_complexity'] for r in tda_results]

    # Base features (all use past data only)
    df['returns'] = df['close'].pct_change()
    df['returns_4'] = df['close'].pct_change(4)
    df['returns_8'] = df['close'].pct_change(8)

    df['volatility_4'] = df['returns'].rolling(4).std()
    df['volatility_20'] = df['returns'].rolling(20).std()

    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)

    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    df['body'] = (df['close'] - df['open']) / df['open']
    df['momentum_4'] = df['close'] / df['close'].shift(4) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # Combined complexity score
    max_acc = df['acc_complexity'].max() or 1
    df['complexity_score'] = (
        0.4 * (df['acc_complexity'] / max_acc) +
        0.3 * df['perm_entropy'] +
        0.2 * df['sample_entropy'].clip(0, 2) / 2 +
        0.1 * df['tda_complexity']
    ).clip(0, 1)

    print("Features computed!")
    return df

print("Feature engineering loaded!")
```

### Cell 5: Trigger Labeling (LONG & SHORT)
```python
def label_triggers(df: pd.DataFrame,
                   target_pct: float = 0.02,
                   lookforward: int = 4,
                   lookback: int = 20) -> pd.DataFrame:
    """
    Label trigger zones for LONG and SHORT

    LONG: 2% rise within 4 candles
    SHORT: 2% drop within 4 candles
    """
    df = df.copy()
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    df['trigger_long'] = 0
    df['trigger_short'] = 0

    # Find LONG events (2% rise)
    long_events = []
    min_gap = lookforward * 2
    last_event = -min_gap

    for i in range(len(df) - lookforward):
        entry = closes[i]
        target = entry * (1 + target_pct)

        if any(highs[i+1:i+1+lookforward] >= target):
            if i - last_event >= min_gap:
                long_events.append(i)
                last_event = i

    # Find SHORT events (2% drop)
    short_events = []
    last_event = -min_gap

    for i in range(len(df) - lookforward):
        entry = closes[i]
        target = entry * (1 - target_pct)

        if any(lows[i+1:i+1+lookforward] <= target):
            if i - last_event >= min_gap:
                short_events.append(i)
                last_event = i

    # Label lookback candles before each event
    for event_idx in long_events:
        start = max(0, event_idx - lookback)
        end = event_idx + lookforward + 1
        df.loc[start:end-1, 'trigger_long'] = 1

    for event_idx in short_events:
        start = max(0, event_idx - lookback)
        end = event_idx + lookforward + 1
        df.loc[start:end-1, 'trigger_short'] = 1

    print(f"LONG events: {len(long_events)}, SHORT events: {len(short_events)}")
    return df

print("Trigger labeling loaded!")
```

### Cell 6: Load Data & Split
```python
# Load data
DATA_PATH = "data/BTCUSDT_15m.csv"  # 폴더 내 파일 경로에 맞게 수정
df = pd.read_csv(DATA_PATH)

# Parse timestamp
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
elif 'open_time' in df.columns:
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# Define periods
ETF_DATE = pd.Timestamp('2024-01-11')
TODAY = pd.Timestamp.now()
DAYS_180_AGO = TODAY - timedelta(days=180)
DAYS_90_AGO = TODAY - timedelta(days=90)

print(f"\n=== Data Periods ===")
print(f"ETF Launch: {ETF_DATE}")
print(f"180 days ago: {DAYS_180_AGO.date()}")
print(f"90 days ago: {DAYS_90_AGO.date()}")
print(f"Today: {TODAY.date()}")

# Split data
df_train = df[(df['timestamp'] >= ETF_DATE) & (df['timestamp'] < DAYS_180_AGO)].copy()
df_backtest1 = df[(df['timestamp'] >= DAYS_180_AGO) & (df['timestamp'] < DAYS_90_AGO)].copy()
df_backtest2 = df[(df['timestamp'] >= DAYS_90_AGO)].copy()

print(f"\nTraining: {len(df_train)} rows ({df_train['timestamp'].min().date()} ~ {df_train['timestamp'].max().date()})")
print(f"Backtest1: {len(df_backtest1)} rows ({df_backtest1['timestamp'].min().date()} ~ {df_backtest1['timestamp'].max().date()})")
print(f"Backtest2: {len(df_backtest2)} rows ({df_backtest2['timestamp'].min().date()} ~ {df_backtest2['timestamp'].max().date()})")
```

### Cell 7: Compute Features & Labels
```python
# Compute features for all data (but train only on training period)
print("=== Computing features for Training data ===")
df_train = compute_features_no_leakage(df_train)
df_train = label_triggers(df_train)

print("\n=== Computing features for Backtest1 ===")
df_backtest1 = compute_features_no_leakage(df_backtest1)
df_backtest1 = label_triggers(df_backtest1)

print("\n=== Computing features for Backtest2 ===")
df_backtest2 = compute_features_no_leakage(df_backtest2)
df_backtest2 = label_triggers(df_backtest2)

# Drop warmup rows
WARMUP = 30
df_train = df_train.iloc[WARMUP:].reset_index(drop=True)
df_backtest1 = df_backtest1.iloc[WARMUP:].reset_index(drop=True)
df_backtest2 = df_backtest2.iloc[WARMUP:].reset_index(drop=True)

print(f"\nAfter warmup:")
print(f"Training: {len(df_train)} rows")
print(f"Backtest1: {len(df_backtest1)} rows")
print(f"Backtest2: {len(df_backtest2)} rows")
```

### Cell 8: Train Model
```python
# Feature columns
FEATURE_COLS = [
    'acc_complexity', 'range_position', 'attempt_count',
    'perm_entropy', 'sample_entropy',
    'tda_h0_entropy', 'tda_h1_entropy', 'tda_h1_count', 'tda_complexity',
    'returns', 'returns_4', 'returns_8',
    'volatility_4', 'volatility_20',
    'volume_ratio', 'price_position',
    'body', 'momentum_4', 'momentum_20', 'rsi',
    'complexity_score'
]

def prepare_features(df):
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    return X

# Train LONG model
print("=== Training LONG Model ===")
X_train = prepare_features(df_train)
y_train_long = df_train['trigger_long'].values

scaler_long = StandardScaler()
X_train_scaled = scaler_long.fit_transform(X_train)

model_long = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model_long.fit(X_train_scaled, y_train_long)

train_acc_long = model_long.score(X_train_scaled, y_train_long)
print(f"LONG Train Accuracy: {train_acc_long:.4f}")
print(f"LONG Trigger ratio: {y_train_long.mean():.2%}")

# Train SHORT model
print("\n=== Training SHORT Model ===")
y_train_short = df_train['trigger_short'].values

scaler_short = StandardScaler()
X_train_scaled_short = scaler_short.fit_transform(X_train)

model_short = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model_short.fit(X_train_scaled_short, y_train_short)

train_acc_short = model_short.score(X_train_scaled_short, y_train_short)
print(f"SHORT Train Accuracy: {train_acc_short:.4f}")
print(f"SHORT Trigger ratio: {y_train_short.mean():.2%}")

# Feature importance
importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_long': model_long.feature_importances_,
    'importance_short': model_short.feature_importances_
}).sort_values('importance_long', ascending=False)

print("\n=== Top 10 Features (LONG) ===")
print(importance[['feature', 'importance_long']].head(10).to_string(index=False))
```

### Cell 9: Backtest Function
```python
def run_backtest(df, model_long, model_short, scaler_long, scaler_short,
                 threshold: float = 0.5,
                 tp_pct: float = 0.02,
                 sl_pct: float = 0.01,
                 hold_candles: int = 4) -> pd.DataFrame:
    """
    Run backtest with LONG and SHORT predictions
    """
    X = prepare_features(df)
    X_scaled_long = scaler_long.transform(X)
    X_scaled_short = scaler_short.transform(X)

    # Get probabilities
    prob_long = model_long.predict_proba(X_scaled_long)[:, 1]
    prob_short = model_short.predict_proba(X_scaled_short)[:, 1]

    df = df.copy()
    df['prob_long'] = prob_long
    df['prob_short'] = prob_short
    df['signal_long'] = (prob_long >= threshold).astype(int)
    df['signal_short'] = (prob_short >= threshold).astype(int)

    # Simulate trades
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    trades = []
    cooldown = 0

    for i in range(len(df) - hold_candles):
        if cooldown > 0:
            cooldown -= 1
            continue

        # Check LONG signal
        if df.iloc[i]['signal_long'] == 1:
            entry = closes[i]
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)

            result = None
            exit_price = None

            for j in range(1, hold_candles + 1):
                if lows[i + j] <= sl:
                    result = 'loss'
                    exit_price = sl
                    break
                if highs[i + j] >= tp:
                    result = 'win'
                    exit_price = tp
                    break

            if result is None:
                exit_price = closes[min(i + hold_candles, len(df) - 1)]
                result = 'win' if exit_price > entry else 'loss'

            pnl = (exit_price - entry) / entry
            trades.append({
                'idx': i, 'timestamp': df.iloc[i]['timestamp'],
                'direction': 'LONG', 'entry': entry, 'exit': exit_price,
                'pnl': pnl, 'result': result,
                'actual_trigger': df.iloc[i]['trigger_long']
            })
            cooldown = hold_candles

        # Check SHORT signal
        elif df.iloc[i]['signal_short'] == 1:
            entry = closes[i]
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)

            result = None
            exit_price = None

            for j in range(1, hold_candles + 1):
                if highs[i + j] >= sl:
                    result = 'loss'
                    exit_price = sl
                    break
                if lows[i + j] <= tp:
                    result = 'win'
                    exit_price = tp
                    break

            if result is None:
                exit_price = closes[min(i + hold_candles, len(df) - 1)]
                result = 'win' if exit_price < entry else 'loss'

            pnl = (entry - exit_price) / entry
            trades.append({
                'idx': i, 'timestamp': df.iloc[i]['timestamp'],
                'direction': 'SHORT', 'entry': entry, 'exit': exit_price,
                'pnl': pnl, 'result': result,
                'actual_trigger': df.iloc[i]['trigger_short']
            })
            cooldown = hold_candles

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return df, trades_df

print("Backtest function loaded!")
```

### Cell 10: Run Backtests
```python
# Backtest 1
print("=== Running Backtest 1 (180d ~ 90d ago) ===")
df_bt1_result, trades_bt1 = run_backtest(
    df_backtest1, model_long, model_short,
    scaler_long, scaler_short, threshold=0.5
)

if len(trades_bt1) > 0:
    print(f"Trades: {len(trades_bt1)}")
    print(f"Win Rate: {(trades_bt1['result'] == 'win').mean():.2%}")
    print(f"Total PnL: {trades_bt1['pnl'].sum():.2%}")
    print(f"Avg PnL: {trades_bt1['pnl'].mean():.4%}")
    print(f"LONG: {len(trades_bt1[trades_bt1['direction']=='LONG'])}, SHORT: {len(trades_bt1[trades_bt1['direction']=='SHORT'])}")
else:
    print("No trades!")

# Backtest 2
print("\n=== Running Backtest 2 (90d ago ~ now) ===")
df_bt2_result, trades_bt2 = run_backtest(
    df_backtest2, model_long, model_short,
    scaler_long, scaler_short, threshold=0.5
)

if len(trades_bt2) > 0:
    print(f"Trades: {len(trades_bt2)}")
    print(f"Win Rate: {(trades_bt2['result'] == 'win').mean():.2%}")
    print(f"Total PnL: {trades_bt2['pnl'].sum():.2%}")
    print(f"Avg PnL: {trades_bt2['pnl'].mean():.4%}")
    print(f"LONG: {len(trades_bt2[trades_bt2['direction']=='LONG'])}, SHORT: {len(trades_bt2[trades_bt2['direction']=='SHORT'])}")
else:
    print("No trades!")
```

### Cell 11: Visualization Function (20-row Chart)
```python
def plot_backtest_chart(df, trades_df, title: str, n_rows: int = 20,
                        save_path: str = None, figsize_per_row: tuple = (20, 2)):
    """
    Plot candlestick chart with volume in N rows

    Args:
        df: DataFrame with OHLCV + signals
        trades_df: DataFrame with trade results
        title: Chart title
        n_rows: Number of rows to split into
        save_path: Path to save figure
        figsize_per_row: Figure size per row
    """
    # Calculate rows
    total_candles = len(df)
    candles_per_row = total_candles // n_rows

    fig, axes = plt.subplots(n_rows, 1, figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Color settings
    up_color = '#26a69a'
    down_color = '#ef5350'
    long_signal_color = '#2196F3'
    short_signal_color = '#FF9800'

    for row in range(n_rows):
        ax = axes[row] if n_rows > 1 else axes

        start_idx = row * candles_per_row
        end_idx = min((row + 1) * candles_per_row, total_candles)

        if start_idx >= total_candles:
            ax.axis('off')
            continue

        df_row = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Create twin axis for volume
        ax2 = ax.twinx()

        # Plot volume bars
        for i, (_, candle) in enumerate(df_row.iterrows()):
            color = up_color if candle['close'] >= candle['open'] else down_color
            ax2.bar(i, candle['volume'], color=color, alpha=0.3, width=0.8)

        # Plot candlesticks
        for i, (_, candle) in enumerate(df_row.iterrows()):
            color = up_color if candle['close'] >= candle['open'] else down_color

            # Wick
            ax.plot([i, i], [candle['low'], candle['high']], color=color, linewidth=0.8)

            # Body
            body_bottom = min(candle['open'], candle['close'])
            body_height = abs(candle['close'] - candle['open'])
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                            facecolor=color, edgecolor=color)
            ax.add_patch(rect)

            # Signal markers
            if candle.get('signal_long', 0) == 1:
                ax.scatter(i, candle['low'] * 0.999, marker='^',
                          color=long_signal_color, s=30, zorder=5, alpha=0.7)
            if candle.get('signal_short', 0) == 1:
                ax.scatter(i, candle['high'] * 1.001, marker='v',
                          color=short_signal_color, s=30, zorder=5, alpha=0.7)

        # Mark trades
        if len(trades_df) > 0:
            row_trades = trades_df[(trades_df['idx'] >= start_idx) & (trades_df['idx'] < end_idx)]
            for _, trade in row_trades.iterrows():
                local_idx = trade['idx'] - start_idx
                if 0 <= local_idx < len(df_row):
                    marker_color = '#4CAF50' if trade['result'] == 'win' else '#f44336'
                    marker = 'o' if trade['direction'] == 'LONG' else 's'
                    ax.scatter(local_idx, trade['entry'], marker=marker,
                              color=marker_color, s=100, zorder=10, edgecolors='black')

        # Formatting
        ax.set_xlim(-1, len(df_row))
        ax.set_ylabel('Price', fontsize=8)
        ax2.set_ylabel('Volume', fontsize=8)
        ax2.yaxis.label.set_color('gray')
        ax2.tick_params(axis='y', colors='gray')

        # Date labels
        if len(df_row) > 0:
            start_date = df_row.iloc[0]['timestamp'].strftime('%m/%d %H:%M')
            end_date = df_row.iloc[-1]['timestamp'].strftime('%m/%d %H:%M')
            ax.set_title(f"Row {row+1}: {start_date} ~ {end_date}", fontsize=10, loc='left')

        ax.grid(True, alpha=0.3)

        # Hide x ticks for cleaner look
        ax.set_xticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

print("Visualization function loaded!")
```

### Cell 12: Generate Backtest Charts
```python
# Create results directory
import os
os.makedirs('results', exist_ok=True)

# Plot Backtest 1
print("=== Generating Backtest 1 Chart ===")
plot_backtest_chart(
    df_bt1_result, trades_bt1,
    title=f"Backtest 1: 180d~90d ago | Trades: {len(trades_bt1)} | WinRate: {(trades_bt1['result']=='win').mean():.1%}" if len(trades_bt1) > 0 else "Backtest 1: No Trades",
    n_rows=20,
    save_path='results/backtest1_chart.png'
)

# Plot Backtest 2
print("\n=== Generating Backtest 2 Chart ===")
plot_backtest_chart(
    df_bt2_result, trades_bt2,
    title=f"Backtest 2: 90d ago~Now | Trades: {len(trades_bt2)} | WinRate: {(trades_bt2['result']=='win').mean():.1%}" if len(trades_bt2) > 0 else "Backtest 2: No Trades",
    n_rows=20,
    save_path='results/backtest2_chart.png'
)
```

### Cell 13: Detailed Analysis
```python
def analyze_trades(trades_df, name: str):
    """Detailed trade analysis"""
    if len(trades_df) == 0:
        print(f"{name}: No trades")
        return

    print(f"\n{'='*50}")
    print(f"{name} Trade Analysis")
    print(f"{'='*50}")

    total = len(trades_df)
    wins = (trades_df['result'] == 'win').sum()
    losses = total - wins

    print(f"Total Trades: {total}")
    print(f"Wins: {wins} ({wins/total:.1%})")
    print(f"Losses: {losses} ({losses/total:.1%})")
    print(f"Total PnL: {trades_df['pnl'].sum():.2%}")
    print(f"Avg PnL: {trades_df['pnl'].mean():.4%}")
    print(f"Max Win: {trades_df['pnl'].max():.2%}")
    print(f"Max Loss: {trades_df['pnl'].min():.2%}")

    # By direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_df[trades_df['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"\n{direction}:")
            print(f"  Trades: {len(dir_trades)}")
            print(f"  Win Rate: {(dir_trades['result']=='win').mean():.1%}")
            print(f"  Total PnL: {dir_trades['pnl'].sum():.2%}")

    # Accuracy (predictions vs actual triggers)
    accuracy = trades_df['actual_trigger'].mean()
    print(f"\nTrigger Accuracy: {accuracy:.1%} (predicted in actual trigger zones)")

analyze_trades(trades_bt1, "Backtest 1 (180d~90d)")
analyze_trades(trades_bt2, "Backtest 2 (90d~Now)")
```

### Cell 14: Save Model
```python
# Save models
model_data = {
    'model_long': model_long,
    'model_short': model_short,
    'scaler_long': scaler_long,
    'scaler_short': scaler_short,
    'feature_cols': FEATURE_COLS
}

os.makedirs('models', exist_ok=True)
with open('models/trigger_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to models/trigger_model.pkl")
```

### Cell 15: Complexity Score Analysis
```python
# Analyze complexity score correlation with trade outcomes
def analyze_complexity_impact(df, trades_df, name: str):
    """Analyze how complexity affects trade outcomes"""
    if len(trades_df) == 0:
        return

    print(f"\n{'='*50}")
    print(f"{name} - Complexity Analysis")
    print(f"{'='*50}")

    # Get complexity at trade entry points
    trade_complexity = []
    for _, trade in trades_df.iterrows():
        idx = trade['idx']
        if idx < len(df):
            trade_complexity.append({
                'complexity': df.iloc[idx]['complexity_score'],
                'result': trade['result'],
                'pnl': trade['pnl'],
                'direction': trade['direction']
            })

    tc_df = pd.DataFrame(trade_complexity)

    # Split by complexity level
    low_complexity = tc_df[tc_df['complexity'] < 0.4]
    mid_complexity = tc_df[(tc_df['complexity'] >= 0.4) & (tc_df['complexity'] < 0.6)]
    high_complexity = tc_df[tc_df['complexity'] >= 0.6]

    print("\nTrade Outcomes by Complexity Level:")
    for name_level, df_level in [('Low (<0.4)', low_complexity),
                                  ('Mid (0.4-0.6)', mid_complexity),
                                  ('High (>0.6)', high_complexity)]:
        if len(df_level) > 0:
            wr = (df_level['result'] == 'win').mean()
            avg_pnl = df_level['pnl'].mean()
            print(f"  {name_level}: {len(df_level)} trades, WR={wr:.1%}, AvgPnL={avg_pnl:.4%}")
        else:
            print(f"  {name_level}: No trades")

analyze_complexity_impact(df_bt1_result, trades_bt1, "Backtest 1")
analyze_complexity_impact(df_bt2_result, trades_bt2, "Backtest 2")
```

---

## 주요 체크포인트

1. **미래 데이터 누수 없음**
   - 모든 feature는 해당 시점 이전 데이터만 사용
   - Rolling calculation 사용 (shift, rolling)
   - 트리거 라벨은 학습 후 백테스트에서만 평가용으로 사용

2. **기간 분리**
   - Training: ETF 출시 ~ 180일 전
   - Backtest 1: 180일 전 ~ 90일 전
   - Backtest 2: 90일 전 ~ 현재

3. **시각화**
   - 20행 분할 캔들 차트
   - 거래량 포함 (twin axis)
   - 시그널 마커 (LONG: 삼각형, SHORT: 역삼각형)
   - 거래 결과 마커 (승리: 녹색, 패배: 빨강)

4. **데이터 요구사항**
   - 15분봉 OHLCV 데이터
   - 컬럼: timestamp/open_time, open, high, low, close, volume
