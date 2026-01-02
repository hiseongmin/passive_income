# TDA Deep Learning Model Implementation Plan

## Overview

Build a **multi-task LSTM model** that uses Topological Data Analysis (TDA) features to predict:
1. **Trigger** (binary): Whether a 2% price move will occur in the next 2 hours
2. **Max_Pct** (continuous): Maximum percentage gain if trigger is positive

### Inputs
- **Raw OHLCV data**: Read 5-min candles from `data/`, convert to 15-min (see Data Pipeline below)
- **Market complexity score**: 0-1 value from dev_ye branch (placeholder until implemented)
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
                   ▼
    ┌─────────────────────────────────────┐
    │ TDA Feature Extraction              │
    │ (on aligned close prices)           │
    └─────────────────────────────────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ PyTorch Dataset │
           │ (features + labels)
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

## Notes

- **GPU Optimization (CRITICAL)**: Use NVIDIA A6000's 48GB VRAM with mixed precision (AMP), torch.compile(), and large batch sizes (256+) for maximum throughput
- **Timestamp Alignment (CRITICAL)**: Inner join on `open_time` between generated 15-min and `data_flagged/` to prevent timeline distortion
- **Complexity Score Placeholder**: Until dev_ye implements the complexity module, use a dummy value (0.5) or skip the complexity encoder
- **TDA Caching**: Pre-compute TDA features once and cache to avoid repeated expensive computations
- **Parameter Tuning**: Use `scripts/hyperopt.py` for grid search over TDA and model parameters
