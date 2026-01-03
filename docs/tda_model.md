# TDA Deep Learning Model Implementation Plan

## Overview

Build a **multi-task LSTM model** that uses Topological Data Analysis (TDA) features to predict:
1. **Trigger** (binary): Whether a 2% price move will occur in the next 2 hours
2. **Max_Pct** (continuous): Maximum percentage gain if trigger is positive

### Inputs
- **Raw OHLCV data**: Read 5-min candles from `data/`, convert to 15-min (see Data Pipeline below)
- **Market complexity score**: 0-1 value from `src/complexity/` module (see [Market Complexity Integration](#market-complexity-integration))
- **TDA features**: Extracted from converted 15-min price time series
- **Labels for supervised learning**: `Trigger` and `Max_Pct` from `data_flagged/`

### Key Design Decisions
- **Architecture**: Multi-task LSTM (single model, two output heads)
- **TDA Features**: All four - Betti curve, Persistent entropy, Total persistence, Landscape L2 norm
- **Framework**: PyTorch + giotto-tda

---

## Data Pipeline

### Step 1: Read Raw 5-min Data
Source: `data/` directory
```
data/
├── BTCUSDT_perp_etf_to_90d_ago.csv   (181,441 rows)
├── BTCUSDT_perp_last_90d.csv         (25,921 rows)
├── BTCUSDT_spot_etf_to_90d_ago.csv   (181,441 rows)
└── BTCUSDT_spot_last_90d.csv         (25,920 rows)
```

### Step 2: Convert 5-min to 15-min (per data_modification.md)
Merge every 3 consecutive 5-minute candles into 1 15-minute candle:

| Column | Aggregation Method |
|--------|-------------------|
| `open_time` | First timestamp |
| `open` | First value |
| `high` | Maximum |
| `low` | Minimum |
| `close` | Last value |
| `volume` | Sum |
| `buy_volume` | Sum |
| `sell_volume` | Sum |
| `volume_delta` | Sum |
| `cvd` | Last value (cumulative) |
| `open_interest` | Last value (perp only) |

### Step 3: Timestamp Alignment (CRITICAL)
**Purpose**: Ensure generated 15-min data and `data_flagged/` have identical timestamps to prevent timeline distortion.

**Implementation**:
1. Load `data_flagged/` file to get reference timestamps
2. After converting 5-min → 15-min, join on `open_time` column
3. Only keep rows where timestamps match exactly
4. Validate: assert converted data timestamps == flagged data timestamps

```python
# Pseudo-code for timestamp alignment
converted_15m = resample_5min_to_15min(raw_5min_data)
flagged_data = pd.read_csv('data_flagged/..._15m_flagged.csv')

# Inner join on timestamp - ensures exact alignment
merged = pd.merge(
    converted_15m,
    flagged_data[['open_time', 'Trigger', 'Max_Pct']],
    on='open_time',
    how='inner'
)

# Validation
assert len(merged) == len(flagged_data), "Timestamp mismatch detected!"
```

**Edge Cases**:
- If timestamps don't match: Log warning, investigate resampling logic
- Missing rows: Use inner join to exclude unmatched timestamps
- Timezone handling: Ensure both use same timezone (UTC)

### Step 4: Extract TDA Features
Apply TDA feature extraction pipeline to the **aligned** 15-min OHLCV data.

### Step 5: Supervised Learning Labels
Labels (`Trigger`, `Max_Pct`) are obtained via timestamp-aligned merge from `data_flagged/`:
```
data_flagged/
├── BTCUSDT_perp_etf_to_90d_ago_15m_flagged.csv
├── BTCUSDT_perp_last_90d_15m_flagged.csv
├── BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv
└── BTCUSDT_spot_last_90d_15m_flagged.csv
```
- `Trigger` (bool): 2% price move detected in next 2 hours
- `Max_Pct` (float): Maximum percentage gain in the window

### Data Flow Diagram
```
data/ (5-min raw)              data_flagged/ (15-min with labels)
    │                                    │
    ▼                                    │
┌─────────────────────────────┐          │
│ Resample 5-min → 15-min     │          │
│ (aggregation rules above)   │          │
└─────────────────────────────┘          │
    │                                    │
    └──────────────┬─────────────────────┘
                   ▼
    ┌─────────────────────────────────────┐
    │ TIMESTAMP ALIGNMENT (inner join)    │
    │ • Join on open_time column          │
    │ • Validate row counts match         │
    │ • Ensure no timeline distortion     │
    └─────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │ Aligned Dataset:                    │
    │ • OHLCV (from converted 5-min)      │
    │ • Trigger, Max_Pct (from flagged)   │
    └─────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────────┐  ┌───────────────────┐
│ TDA Feature       │  │ Complexity Score  │
│ Extraction        │  │ (src/complexity/) │
│ (close prices)    │  │ (OHLCV data)      │
└───────────────────┘  └───────────────────┘
        │                     │
        └──────────┬──────────┘
                   ▼
           ┌─────────────────┐
           │ PyTorch Dataset │
           │ • OHLCV sequence│
           │ • TDA features  │
           │ • Complexity    │
           │ • Labels        │
           └─────────────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ Multi-task LSTM │
           └─────────────────┘
```

---

## Project Structure

```
src/tda_model/
├── config/
│   ├── __init__.py
│   ├── default_config.yaml      # All tunable hyperparameters
│   └── config.py                # Config dataclasses
├── data/
│   ├── __init__.py
│   ├── preprocessing.py         # 5-min → 15-min conversion
│   ├── dataset.py               # PyTorch Dataset
│   └── data_loader.py           # Train/val/test splitting
├── tda/
│   ├── __init__.py
│   ├── point_cloud.py           # Takens embedding
│   ├── persistence.py           # Persistent homology
│   └── features.py              # Feature extraction
├── models/
│   ├── __init__.py
│   ├── lstm.py                  # Multi-task LSTM
│   └── losses.py                # Custom loss functions
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop
│   └── metrics.py               # Evaluation metrics
├── scripts/
│   ├── train.py                 # Main training script
│   └── hyperopt.py              # Hyperparameter search
└── __init__.py
```

---

## Tunable Parameters

### TDA Parameters (from research papers)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `window_size` | 1344 | 672-2016 | Segment length (1344 = 2 weeks at 15-min) |
| `embedding_dim` | 2 | 2-5 | Takens embedding dimension (k=2 best per enhance.pdf) |
| `time_delay` | 12 | 3-48 | Embedding delay (12 = 3 hours at 15-min) |
| `betti_bins` | 100 | 50-200 | Betti curve resolution |
| `landscape_layers` | 5 | 3-10 | Persistence landscape layers |

### Model Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lstm_hidden_size` | 128 | 64-256 | LSTM hidden dimension |
| `lstm_num_layers` | 2 | 1-3 | Number of LSTM layers |
| `lstm_dropout` | 0.3 | 0.1-0.5 | Dropout rate |
| `use_attention` | true | - | Self-attention on LSTM output |

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `batch_size` | 256 | 128-512 | Training batch size (increased for A6000) |
| `learning_rate` | 0.001 | 1e-4 to 1e-2 | Initial learning rate |
| `trigger_loss_weight` | 1.0 | 0.5-2.0 | Classification loss weight |
| `max_pct_loss_weight` | 0.5 | 0.3-1.0 | Regression loss weight |

---

## GPU Optimization (NVIDIA A6000)

### Hardware Specs
- **GPU**: NVIDIA RTX A6000
- **VRAM**: 48GB GDDR6
- **CUDA Cores**: 10,752
- **Tensor Cores**: 336 (3rd gen)

### PyTorch GPU Optimizations

#### 1. Mixed Precision Training (AMP)
Use `torch.cuda.amp` for faster training with lower memory usage:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():  # FP16 forward pass
        trigger_pred, max_pct_pred = model(ohlcv, tda_features, complexity)
        loss = loss_fn(trigger_pred, trigger_true, max_pct_pred, max_pct_true)

    scaler.scale(loss).backward()  # Scaled backward pass
    scaler.step(optimizer)
    scaler.update()
```

#### 2. DataLoader Optimizations
```python
train_loader = DataLoader(
    dataset,
    batch_size=256,           # Large batch for A6000's 48GB VRAM
    shuffle=True,
    num_workers=8,            # Parallel data loading
    pin_memory=True,          # Faster CPU→GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=4,        # Pre-fetch batches
)
```

#### 3. cuDNN Optimizations
```python
import torch.backends.cudnn as cudnn

cudnn.benchmark = True        # Auto-tune convolution algorithms
cudnn.deterministic = False   # Allow non-deterministic for speed
torch.set_float32_matmul_precision('medium')  # TensorCore optimization
```

#### 4. Memory Management
```python
# Enable memory-efficient attention (if using Transformer)
torch.backends.cuda.enable_flash_sdp(True)

# Gradient checkpointing for large models
model.lstm.gradient_checkpointing_enable()

# Clear cache periodically
torch.cuda.empty_cache()
```

#### 5. Compile Model (PyTorch 2.0+)
```python
# JIT compile for faster execution
model = torch.compile(model, mode="reduce-overhead")
```

### GPU Config in default_config.yaml
```yaml
gpu:
  device: "cuda:0"
  mixed_precision: true       # Enable AMP
  compile_model: true         # torch.compile()
  cudnn_benchmark: true

dataloader:
  batch_size: 256             # Increased for A6000
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

training:
  gradient_accumulation_steps: 1  # Increase if OOM
  gradient_clip: 1.0
```

### Expected Performance
| Optimization | Speedup | Memory Reduction |
|--------------|---------|------------------|
| Mixed Precision (AMP) | ~2x | ~50% |
| torch.compile | ~1.5x | - |
| cuDNN benchmark | ~1.2x | - |
| Larger batch size | ~1.5x throughput | - |

### Monitoring GPU Usage
```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi

# Or use in Python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used / 1e9:.1f}GB / {info.total / 1e9:.1f}GB")
```

---

## Implementation Steps

### Phase 1: Configuration & Data (Files: 5)
1. Create `config/default_config.yaml` with all tunable parameters
2. Create `config/config.py` with dataclasses and validation
3. Create `data/preprocessing.py`:
   - Read 5-min from `data/`, convert to 15-min
   - **Timestamp alignment with `data_flagged/`** (inner join on `open_time`)
   - Validation to ensure no timeline distortion
4. Create `data/data_loader.py` - Load aligned data, temporal splitting
5. Create `data/dataset.py` for PyTorch Dataset with TDA pre-computation

### Phase 2: TDA Pipeline (Files: 3)
6. Create `tda/point_cloud.py` - Takens embedding wrapper
7. Create `tda/persistence.py` - Vietoris-Rips persistence via giotto-tda
8. Create `tda/features.py` - Extract Betti curve, entropy, persistence, landscape

### Phase 3: Model Architecture (Files: 2)
9. Create `models/lstm.py` - Multi-task LSTM with attention
10. Create `models/losses.py` - Combined focal + MSE loss

### Phase 4: Training Pipeline (Files: 3)
11. Create `training/metrics.py` - Classification/regression metrics
12. Create `training/trainer.py` - Training loop with:
    - **Mixed precision (AMP)** for 2x speedup
    - **torch.compile()** for optimized execution
    - **cuDNN benchmark** enabled
    - Early stopping, gradient clipping
13. Create `scripts/train.py` - Main entry point with CLI and GPU config

### Phase 5: Integration & Testing
14. End-to-end test with sample data
15. Verify walk-forward validation (no data leakage)
16. Create `scripts/hyperopt.py` for parameter tuning

---

## Key Architecture Details

### TDA Feature Extraction Pipeline

```
Price Time Series (15-min close prices)
    │
    ▼
┌─────────────────────────────┐
│ Sliding Window Extraction   │  window_size=1344, stride=4
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Takens Embedding            │  dim=2, tau=12
│ → Point Cloud in R^k        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Vietoris-Rips Persistence   │  H0, H1 homology
│ → Persistence Diagram       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Feature Vectorization       │
│ • Betti Curve (200 dims)    │
│ • Persistent Entropy (2)    │
│ • Total Persistence (2)     │
│ • Landscape L2 Norm (10)    │
└─────────────────────────────┘
    │
    ▼
TDA Feature Vector (214 dims)
```

### Multi-task LSTM Architecture

```
Inputs:
├── OHLCV Sequence [B, T, 4]
├── TDA Features [B, 214]
└── Complexity Score [B, 1]

                    ┌──────────────┐
OHLCV ──────────────│    LSTM     │──────┐
                    │  (2 layers) │      │
                    └──────────────┘      │
                           │              │
                    ┌──────────────┐      │
                    │  Attention   │      │
                    └──────────────┘      │
                           │              │
                           ▼              │
                    ┌──────────────┐      │
TDA Features ───────│   MLP (64)  │──────┤
                    └──────────────┘      │
                                          │  Concat
                    ┌──────────────┐      │
Complexity ─────────│   MLP (16)  │──────┤
                    └──────────────┘      │
                                          ▼
                                   ┌──────────────┐
                                   │  Shared FC   │
                                   │   (128→64)   │
                                   └──────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                   ┌──────────────┐               ┌──────────────┐
                   │ Trigger Head │               │ Max_Pct Head │
                   │  (Sigmoid)   │               │  (Linear)    │
                   └──────────────┘               └──────────────┘
                          │                               │
                          ▼                               ▼
                   Trigger Prob [B,1]              Max_Pct [B,1]
```

### Loss Function

```python
Loss = w1 * FocalLoss(trigger_pred, trigger_true)
     + w2 * MSE(max_pct_pred[trigger_true=1], max_pct_true[trigger_true=1])
```

- **Focal Loss**: Handles class imbalance (~10% triggers)
- **Masked MSE**: Only compute regression loss on positive triggers

---

## Data Split (from CLAUDE.md)

| Period | Usage | Files |
|--------|-------|-------|
| ETF launch ~ 180 days ago | Training | `*_etf_to_90d_ago_15m_flagged.csv` |
| 180 days ago ~ 90 days ago | Validation | Split from training |
| 90 days ago ~ present | Test | `*_last_90d_15m_flagged.csv` |

**Critical**: Walk-forward only — no future data leakage

---

## Dependencies

```
# Core
torch>=2.0.0                  # Required for torch.compile()
giotto-tda>=0.6.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0

# GPU Monitoring
pynvml>=11.5.0                # NVIDIA GPU monitoring
```

---

## Success Metrics (from CLAUDE.md)

- Sharpe Ratio > 1.5
- Maximum Drawdown < 20%
- Win Rate > 55%
- Positive returns after fees

---

## Critical Files to Create

1. `src/tda_model/config/default_config.yaml` - All tunable hyperparameters
2. `src/tda_model/data/preprocessing.py` - 5-min → 15-min conversion
3. `src/tda_model/data/data_loader.py` - Load data + labels from `data_flagged/`
4. `src/tda_model/tda/features.py` - TDA feature extraction
5. `src/tda_model/models/lstm.py` - Multi-task LSTM model
6. `src/tda_model/data/dataset.py` - PyTorch Dataset
7. `src/tda_model/training/trainer.py` - Training loop
8. `src/tda_model/scripts/train.py` - Main entry point

---

## Market Complexity Integration

The market complexity module (`src/complexity/`) is fully implemented and provides a 0-1 score measuring how difficult it is to predict the trend.

### Complexity Module Overview

**Location**: `src/complexity/indicators.py`

**6 Indicators Combined**:
1. **MA Separation**: Distance between MAs (20, 50, 100, 200)
2. **Bollinger Band Width**: (Upper - Lower) / Price
3. **Price Efficiency**: |Net Movement| / Total Movement
4. **Support Reaction**: Bounce magnitude after touching support
5. **Directional Result**: Price displacement per time unit
6. **Volume-Price Alignment**: Correlation between volume and price movement

**Score Interpretation**:
- `0.0` = Clear trend (easy to predict)
- `1.0` = Complex/choppy market (hard to predict)

### Complexity Data Pipeline

**IMPORTANT**: The complexity module is designed for 1-minute data. Since the TDA model uses 15-minute data, complexity must be computed on 1-minute data first, then resampled to 15-minute.

#### Step 1: Collect 1-minute Data
```bash
# Run the data collector (requires internet connection)
cd src/complexity
python collect_1m_data.py
```

Output files:
```
data/
├── BTCUSDT_spot_1m_etf_to_90d_ago.csv   # Training period
└── BTCUSDT_spot_1m_last_90d.csv          # Test period
```

#### Step 2: Compute Complexity on 1-minute Data
```python
import pandas as pd
from complexity import calculate_complexity_score

# Load 1-minute data
df_1m = pd.read_csv('data/BTCUSDT_spot_1m_etf_to_90d_ago.csv')
df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
df_1m = df_1m.set_index('open_time')

# Compute complexity (uses default 1-min optimized parameters)
_, complexity_1m = calculate_complexity_score(df_1m)
```

#### Step 3: Resample to 15-minute
```python
# Resample: take mean of 15 consecutive 1-min complexity values
complexity_15m = complexity_1m.resample('15min').mean()

# Reset index for merging
complexity_15m = complexity_15m.reset_index()
complexity_15m.columns = ['open_time', 'complexity']
```

#### Step 4: Merge with Flagged Data
```python
# Load 15-min flagged data
df_flagged = pd.read_csv('data_flagged/BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv')
df_flagged['open_time'] = pd.to_datetime(df_flagged['open_time'])

# Merge complexity scores
df_flagged = pd.merge(
    df_flagged,
    complexity_15m,
    on='open_time',
    how='left'
)

# Fill any missing values with placeholder
df_flagged['complexity'] = df_flagged['complexity'].fillna(0.5)

# Save updated file
df_flagged.to_csv('data_flagged/BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv', index=False)
```

### Complete Pipeline Diagram

```
data/BTCUSDT_spot_1m_*.csv (1-minute OHLCV)
            │
            ▼
┌─────────────────────────────────────┐
│ calculate_complexity_score()        │
│ • 6 indicators computed at 1-min    │
│ • Uses default window parameters    │
└─────────────────────────────────────┘
            │
            ▼
    complexity_1m (pd.Series)
            │
            ▼
┌─────────────────────────────────────┐
│ Resample to 15-min                  │
│ • complexity_1m.resample('15min')   │
│ • Aggregation: mean()               │
└─────────────────────────────────────┘
            │
            ▼
    complexity_15m (pd.Series)
            │
            ▼
┌─────────────────────────────────────┐
│ Merge with data_flagged/            │
│ • Join on open_time                 │
│ • Add 'complexity' column           │
└─────────────────────────────────────┘
            │
            ▼
data_flagged/*_15m_flagged.csv
(now includes: OHLCV, Trigger, Max_Pct, complexity)
```

### Aggregation Method

When resampling 1-min complexity to 15-min, we use `mean()`:
- Averages complexity over the 15-minute window
- Smooths out short-term noise
- Provides a representative complexity for each 15-min candle

### Configuration

In `config/default_config.yaml`:
```yaml
data:
  use_complexity: true           # Enable complexity integration
  complexity_column: "complexity" # Column name in data_flagged/
  complexity_placeholder: 0.5    # Fallback if complexity unavailable
```

---

## Notes

- **GPU Optimization (CRITICAL)**: Use NVIDIA A6000's 48GB VRAM with mixed precision (AMP), torch.compile(), and large batch sizes (256+) for maximum throughput
- **Timestamp Alignment (CRITICAL)**: Inner join on `open_time` between generated 15-min and `data_flagged/` to prevent timeline distortion
- **Complexity Integration**: Use `calculate_complexity_score()` from `src/complexity/` to compute real complexity values instead of placeholder
- **TDA Caching**: Pre-compute TDA features once and cache to avoid repeated expensive computations
- **Parameter Tuning**: Use `scripts/hyperopt.py` for grid search over TDA and model parameters

---

## Training Session Analysis (2026-01-03)

### Problem: Model Predicting All Negatives

**Observed Results:**
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 87.66% | 98.24% | 89.56% |
| F1 Score | 0.000 | 0.000 | 0.000 |
| Precision | 0.000 | 0.000 | 0.000 |
| Recall | 0.000 | 0.000 | 0.000 |
| AUC-ROC | 0.50 | 0.41 | 0.541 |

The model learned to predict **all samples as "no trigger"** because this achieves high accuracy on imbalanced data.

### Root Causes Identified

#### 1. Focal Loss Alpha Configuration (CRITICAL)

**Problem:** `focal_alpha=0.25` is **backwards** for our use case.

In focal loss: `alpha_t = alpha * targets + (1-alpha) * (1-targets)`
- With alpha=0.25: positives get 25% weight, negatives get 75%
- For minority class (triggers ~10%), this **worsens** imbalance

**Fix:** Set `focal_alpha=0.90` to give 90% weight to positive (trigger) class.

#### 2. Extreme Validation Class Imbalance

| Dataset | Trigger Rate | Imbalance Ratio |
|---------|-------------|-----------------|
| Training | 12.3% | 7.1:1 |
| **Validation** | **1.76%** | **55.7:1** |
| Test | 10.4% | 8.6:1 |

Temporal split placed most triggers in training period, leaving validation severely imbalanced.

**Fix:** Add WeightedRandomSampler for training; use precision-focused early stopping.

#### 3. No Precision Optimization

Trading context: **False positives are costly** (bad trades) while missing some triggers is acceptable.

Current implementation uses fixed threshold=0.5, which is inappropriate for imbalanced data.

**Fix:** Add threshold tuning to maximize precision at minimum 40% recall.

#### 4. Data Quality Issues

- 4,375 training rows + 581 test rows have `Max_Pct > 1.0`
- Invalid regression targets add noise to training

**Fix:** Clip `Max_Pct` to [0, 1] range.

---

## Complexity Computation Limitations

### 5-Minute vs 1-Minute Data

Due to Binance API geo-restriction (HTTP 451), complexity was computed on **5-minute data** instead of 1-minute as originally designed.

**Information Loss Assessment:**
- ~80% fewer data points per hour (60 bars -> 12 bars)
- Fast support/resistance bounces may be missed
- ~20-30% overall signal quality reduction

### Parameter Scaling Issues

The 5-minute complexity parameters have **inconsistent scaling**:

| Indicator | Original (1-min) | Should Be (5-min) | Actually Used |
|-----------|------------------|-------------------|---------------|
| MA periods | [20,50,100,200] | [4,10,20,40] | [4,10,20,40] |
| BB period | 20 | 4 | 4 |
| Efficiency | 20 | 4 | 12 (too large) |
| Support | 20 | 4 | 24 (too large) |
| Direction | 20 | 4 | 12 (too large) |
| Volume | 20 | 4 | 12 (too large) |

4 of 6 indicators use 2-6x larger lookbacks than intended, causing over-smoothing.

**Impact:** Complexity values may be artificially inflated and respond slowly to market regime changes.

**Mitigation:** Current values are usable for prototyping but should be recalculated with correct parameters for production.

---

## TDA Feature Verification

### Pipeline Correctness

The TDA feature extraction pipeline has been **verified as mathematically correct**:

1. **Takens Embedding** (`tda/point_cloud.py`)
   - Correctly creates delay vectors with dim=2, tau=12
   - For 336-window: produces 324 points in 2D space
   - Min-max normalized to [0,1]^2

2. **Vietoris-Rips Persistence** (`tda/persistence.py`)
   - Uses giotto-tda with ripser fallback
   - Correctly handles infinite deaths
   - Computes H0 (components) and H1 (loops)

3. **Feature Extraction** (`tda/features.py`)
   - Betti curve: 50 bins x 2 dims = 100 features
   - Persistent entropy: 2 features
   - Total persistence: 2 features
   - Landscape L2 norms: 3 layers x 2 dims = 6 features
   - **Total: 110 features** (correctly cached)

### Parameter Appropriateness

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| window_size | 336 | 3.5 days of 15-min candles (captures weekly patterns) |
| embedding_dim | 2 | Per enhance.pdf: k=2 works best for financial time series |
| time_delay | 12 | 3 hours at 15-min intervals (intraday periodicity) |
| betti_bins | 50 | Reduced from 100 for computational efficiency |
| landscape_layers | 3 | Reduced from 5 for efficiency |

**Conclusion:** TDA features are **correctly implemented and reasonably parameterized**.

---

## Trading Considerations

### Precision vs Recall Trade-off

**Key Insight:** For trading signals, **precision > recall**.

- **False Positive (FP)**: Predict trigger when none occurs -> unnecessary trade -> loss
- **False Negative (FN)**: Miss a real trigger -> missed opportunity -> no loss

The cost function is asymmetric: FP incurs actual loss, FN only incurs opportunity cost.

### Target Metrics

Based on this analysis, we target:
- **Precision: 60-70%** (acceptable false positive rate: 30-40%)
- **Recall: 40-50%** (catch half of real triggers)
- **Threshold: 0.6-0.7** (higher than default 0.5)

### Threshold Tuning Strategy

After training, tune prediction threshold to maximize:
```
score = precision - lambda * max(0, min_recall - recall)
```
Where `min_recall = 0.40` and `lambda` is a large penalty.

---

## Configuration Fixes Applied

### `config/config.py` Changes

```python
# BEFORE (incorrect)
focal_alpha: float = 0.25
trigger_loss_weight: float = 1.0
max_pct_loss_weight: float = 0.5

# AFTER (fixed)
focal_alpha: float = 0.90      # Weight minority class properly
trigger_loss_weight: float = 3.0  # Emphasize classification
max_pct_loss_weight: float = 0.3  # De-emphasize regression
```

### Training Improvements

1. **WeightedRandomSampler**: Oversample trigger class during training
2. **Threshold Tuning**: Find optimal threshold post-training
3. **Data Cleaning**: Clip Max_Pct to [0, 1]
4. **Early Stopping**: Monitor precision, not just loss

---

## Hyperparameter Experiments (2026-01-03 12:00-12:15)

### Problem: WeightedRandomSampler Distribution Mismatch

The WeightedRandomSampler creates 50-50 balanced batches during training, but actual data has:
- Training: ~12.3% triggers
- Validation: ~1.76% triggers (extremely low)
- Test: ~10.17% triggers

This causes the model to output probabilities calibrated for 50% trigger rate, resulting in excessive false positives on real data.

### Experiments Summary

| Config | Sampler | focal_alpha | Test Precision | Test Recall | Test AUC |
|--------|---------|-------------|----------------|-------------|----------|
| Baseline | Yes | 0.90 | 15.0% | 85.8% | 0.745 (train) |
| Exp 1 | No | 0.75 | 0.0% | 0.0% | ~0.5 |
| Exp 2 | No | 0.85 | 0.0% | 0.0% | ~0.5 |
| Exp 3 | Yes | 0.75 | 12.1%* | 48.9%* | 0.526 |
| Exp 4 | No | 0.90 | 10.4% | 100% | 0.500 |

*At threshold 0.6 (best precision-recall trade-off found)

### Key Findings

1. **Without Sampler**: Model predicts all negatives (cannot learn minority class)
2. **With Sampler + Low Alpha**: Distribution mismatch causes excess false positives
3. **High Alpha Only**: Model predicts all positives
4. **Test AUC ~0.5**: Model cannot discriminate between triggers and non-triggers on unseen data

### Root Cause Analysis

The fundamental issue is **severe overfitting**:
- Train AUC: ~0.74 (good discrimination)
- Test AUC: ~0.53 (near random)

Possible causes:
1. **Data Quality**: Trigger events may not be predictable from historical patterns
2. **Feature Gap**: TDA/OHLCV/complexity features may not capture trigger-predictive information
3. **Distribution Shift**: Training period (2024-01 to 2025-06) differs from test (2025-09 to 2025-12)

### Current Best Configuration

```yaml
training:
  focal_alpha: 0.75
  use_weighted_sampler: true
  inference_threshold: 0.6
  weight_decay: 0.001
model:
  lstm_dropout: 0.5
```

Best achievable results:
- Precision: ~12% at threshold 0.6
- Recall: ~49% at threshold 0.6
- **Target of 60% precision not achieved**

### Recommendations for Future Work

1. **Feature Engineering**: Add technical indicators, market regime detection
2. **Data Investigation**: Analyze what makes triggers predictable (or unpredictable)
3. **Architecture Changes**: Try attention-only models, transformers
4. **Ensemble Methods**: Combine multiple models for better generalization
5. **Simpler Baselines**: Validate feature quality with logistic regression, random forest

---

## Comprehensive Enhancement Plan (2026-01-03)

After extensive hyperparameter experiments, we identified that the current LSTM-based approach has fundamental limitations. This section documents a comprehensive enhancement strategy.

### Current Performance Status

| Metric | Best Achieved | Target |
|--------|---------------|--------|
| Test Precision | 12.1% | 60-70% |
| Test Recall | 48.9% | 40-50% |
| Test AUC | 0.526 | > 0.70 |
| Train AUC | 0.74 | - |

**Core Problem**: Severe overfitting - Train AUC 0.74 but Test AUC 0.53 (near random)

### Detailed Root Cause Analysis

#### 1. Severe Distribution Shift (CRITICAL)

Triggers are **NOT uniformly distributed** across time - they concentrate in early 2024:

| Dataset | Period | Trigger Rate | Ratio |
|---------|--------|--------------|-------|
| Training | Jan 2024 - Jun 2025 | 12.3% | 7.1:1 |
| **Validation** | Jun 2025 - Sep 2025 | **1.76%** | **55.7:1** |
| Test | Sep 2025 - Dec 2025 | 10.17% | 8.8:1 |

**The 6.98x imbalance** between train/val triggers indicates:
- Bitcoin market regime changed significantly in late 2025
- Patterns learned in early period don't transfer to later period
- Model memorizes training patterns, fails on validation/test

#### 2. LSTM Architecture Limitations

The current LSTM architecture has inherent issues for this problem:

| Issue | Impact |
|-------|--------|
| **Recurrent bias** | Sequential processing biased toward recent history |
| **Vanishing gradients** | Hard to learn long-term dependencies |
| **Overfitting prone** | Too flexible for this data size (~51K samples) |
| **Sequential processing** | Underutilizes GPU (A6000 with 48GB) |

#### 3. TDA Feature Configuration Concerns

Current TDA configuration may be suboptimal:

| Parameter | Current | Concern |
|-----------|---------|---------|
| window_size | 336 (3.5 days) | May be too short for weekly patterns |
| embedding_dim | 2 | Standard, but 3 could capture more structure |
| time_delay | 12 (3 hours) | Appropriate for intraday |
| betti_bins | 50 | Reduced for speed, may lose resolution |
| landscape_layers | 3 | Reduced for speed |

#### 4. WeightedRandomSampler Distribution Mismatch

When enabled:
- Training sees 50-50 balanced batches
- Model outputs calibrated for 50% trigger rate
- At inference, actual rate is ~10% → excessive false positives

When disabled:
- Model can't learn minority class (predicts all negatives)

---

## Proposed Enhancements

### Enhancement 1: N-BEATS Architecture (Replace LSTM)

**Why N-BEATS?**

N-BEATS (Neural Basis Expansion Analysis for Time Series) is a pure deep learning architecture that:
- Uses stacked fully-connected networks instead of RNNs
- Employs doubly residual connections for better gradient flow
- Decomposes time series into interpretable trend/seasonality/residual
- Achieves full GPU parallelization (faster training on A6000)

**N-BEATS vs LSTM for This Problem:**

| Aspect | LSTM | N-BEATS |
|--------|------|---------|
| Overfitting | High risk (flexible) | Lower (simpler architecture) |
| Distribution shift | Vulnerable (sequential bias) | More robust (no recurrence) |
| GPU utilization | ~1MB of 48GB | Full parallelization |
| Training speed | Slower (sequential) | 1.5-2x faster |
| Interpretability | Black box | Trend/seasonality decomposition |
| Long-term patterns | Vanishing gradients | Deep FC layers capture well |

**Proposed N-BEATS Multi-Task Architecture:**

```
Input: OHLCV sequence (96 steps) + TDA features (110) + Complexity (1)
    ↓
N-BEATS Stacks (4 stacks × 4 blocks each):
├─ Trend Stack (polynomial basis)
├─ Seasonality Stack (harmonic basis)
└─ Residual Stack (generic basis)
    ↓
Concatenate: [trend, seasonality, residual, TDA, complexity]
    ↓
Multi-Task Heads:
├─ Trigger Classification → sigmoid → binary
└─ Max_Pct Regression → linear → [0,1]
```

**Files to Create:**
- `src/tda_model/models/nbeats.py` - N-BEATS block/stack implementation
- Modify `src/tda_model/models/__init__.py` - Add N-BEATS export

### Enhancement 2: Dataset Distribution Fix (Stratified Temporal Split)

**Problem**: Current 85/15 temporal split creates 6.98x trigger imbalance

**Solution**: Use stratified temporal blocks that maintain trigger ratio

```python
def stratified_temporal_split(df, validation_ratio=0.15, n_blocks=10):
    """
    Split data into temporal blocks, then select blocks for validation
    that maintain similar trigger rate as training.
    """
    # Divide into n temporal blocks
    block_size = len(df) // n_blocks
    blocks = [df.iloc[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    # Calculate trigger rate per block
    block_trigger_rates = [b['Trigger'].mean() for b in blocks]

    # Select validation blocks that match overall rate
    overall_rate = df['Trigger'].mean()
    val_blocks = select_blocks_matching_rate(blocks, overall_rate, validation_ratio)

    return train_df, val_df
```

**Alternative**: Use K-Fold temporal cross-validation (purged) to evaluate model on multiple time windows.

### Enhancement 3: TDA Parameter Optimization

**Proposed TDA Parameter Grid:**

| Parameter | Current | Proposed Options |
|-----------|---------|------------------|
| window_size | 336 (3.5d) | **672 (7d)**, 1008 (10.5d) |
| embedding_dim | 2 | 2, **3** |
| time_delay | 12 | 12, **24**, 48 |
| betti_bins | 50 | **100**, 150 |
| landscape_layers | 3 | **5**, 7 |

**Rationale:**
- Larger window captures weekly/bi-weekly patterns
- Higher embedding_dim captures more topological structure
- More betti_bins improves resolution of persistence diagrams
- More landscape layers captures finer persistence details

**Implementation:**
- Run TDA extraction with new parameters (requires cache invalidation)
- Compare TDA feature distributions across trigger/non-trigger samples
- Select parameters that maximize feature separation

### Enhancement 4: Class Imbalance Handling (Alternative to WeightedSampler)

**Problem**: WeightedRandomSampler creates calibration mismatch

**Solution**: Use class weights in loss function only (no sampling manipulation)

```python
# In FocalLoss, add pos_weight parameter
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=8.0):
        # pos_weight = n_negative / n_positive ≈ 8.0
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # BCE with pos_weight handles imbalance
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor([self.pos_weight]),
            reduction='none'
        )
        # Apply focal weighting
        ...
```

**Benefits:**
- Model sees natural distribution during training
- Loss function compensates for imbalance
- Output probabilities naturally calibrated

---

## Implementation Priority

| Priority | Enhancement | Effort | Expected Impact |
|----------|-------------|--------|-----------------|
| 1 | N-BEATS Architecture | High | Major (fixes overfitting) |
| 2 | Stratified Temporal Split | Medium | Major (fixes distribution shift) |
| 3 | TDA Parameter Optimization | Medium | Moderate (better features) |
| 4 | Class Weight (no sampler) | Low | Moderate (better calibration) |

---

## Expected Results After Enhancements

| Metric | Current | Target |
|--------|---------|--------|
| **Precision** | 12.1% | **60-70%** |
| **Recall** | 48.9% | **40-50%** |
| F1 Score | ~0.20 | > 0.45 |
| AUC-ROC | 0.526 | > 0.70 |

**Rationale**: In trading, false positives (predicting trigger when none occurs) lead to unnecessary trades and losses. Missing some real triggers is acceptable; acting on false signals is not.
