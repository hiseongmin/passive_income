# TDA Model - Current Status Presentation

**Date:** 2026-01-04
**Project:** Passive Income via Bitcoin Trading - TDA Signal Module

---

## 1. Project Structure Overview

```
tda_model/
├── config/
│   ├── config.py              # Configuration dataclass
│   └── default_config.yaml    # Hyperparameters
├── data/
│   ├── data_loader.py         # Data loading utilities
│   ├── dataset.py             # PyTorch Dataset
│   └── preprocessing.py       # Technical indicators
├── models/
│   ├── lstm.py                # LSTM architecture (legacy)
│   ├── nbeats.py              # N-BEATS architecture (current)
│   ├── losses.py              # Focal loss + MSE
│   └── experiments/           # Experiment results
├── tda/
│   ├── features.py            # TDA feature extraction
│   ├── persistence.py         # Persistence diagram computation
│   └── point_cloud.py         # Time series embedding
├── training/
│   ├── trainer.py             # Training loop
│   └── metrics.py             # Evaluation metrics
├── scripts/
│   ├── train.py               # Main training script
│   ├── run_experiments.py     # Hyperparameter experiments
│   └── hyperopt.py            # Bayesian optimization
└── cache/                     # TDA feature cache (NPY files)
```

---

## 2. Model Architecture

### 2.1 Current Model: Multi-Task N-BEATS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-TASK N-BEATS MODEL                             │
│                     (Neural Basis Expansion Analysis)                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              INPUT LAYER
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  OHLCV + Volume + Tech      TDA Features        Complexity         │
    │    (96 × 14 tensor)         (214 dims)          (6 dims)           │
    │                                                                     │
    └──────────┬────────────────────┬────────────────────┬───────────────┘
               │                    │                    │
               ▼                    ▼                    ▼
    ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
    │  OHLCVEncoder    │  │   TDAEncoder    │  │ ComplexityEncoder   │
    │  (N-BEATS)       │  │   (MLP)         │  │      (MLP)          │
    │                  │  │                 │  │                     │
    │  ┌────────────┐  │  │  214 → 128     │  │   6 → 64 → 64       │
    │  │TrendStack │  │  │  128 → 256     │  │                     │
    │  │(Polynomial)│  │  │                 │  │                     │
    │  └────────────┘  │  └────────┬────────┘  └──────────┬──────────┘
    │  ┌────────────┐  │           │                      │
    │  │Seasonality │  │           │                      │
    │  │ (Fourier)  │  │           │                      │
    │  └────────────┘  │           │                      │
    │  ┌────────────┐  │           │                      │
    │  │ Generic×2  │  │           │                      │
    │  │(Learnable) │  │           │                      │
    │  └────────────┘  │           │                      │
    │        │         │           │                      │
    │        ▼         │           │                      │
    │  ┌────────────┐  │           │                      │
    │  │ Attention  │  │           │                      │
    │  │(8-head MHA)│  │           │                      │
    │  └────────────┘  │           │                      │
    │        │         │           │                      │
    │  Output: 1024    │  Output: 256       Output: 64    │
    └──────────┬───────┘           │                      │
               │                   │                      │
               └───────────────────┼──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │     CONCATENATE      │
                        │   (1024 + 256 + 64)  │
                        │      = 1344 dims     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Shared FC Layers    │
                        │  1344 → 512 → 256    │
                        │  (ReLU + Dropout)    │
                        └──────────┬───────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               │                                       │
               ▼                                       ▼
    ┌────────────────────┐              ┌────────────────────┐
    │   TRIGGER HEAD     │              │   MAX_PCT HEAD     │
    │   (Classification) │              │   (Regression)     │
    │   256 → 32 → 1     │              │   256 → 32 → 1     │
    │                    │              │                    │
    │   Output: sigmoid  │              │   Output: linear   │
    │   P(trigger=1)     │              │   Max % movement   │
    └────────────────────┘              └────────────────────┘
```

### 2.2 Layer Count Summary

| Component              | Layers | Parameters (Est.) |
|------------------------|--------|-------------------|
| N-BEATS Stacks (×4)    | 24     | ~15M              |
| Attention Block        | 3      | ~1M               |
| TDA Encoder            | 4      | ~50K              |
| Complexity Encoder     | 4      | ~5K               |
| Shared FC              | 4      | ~800K             |
| Trigger Head           | 3      | ~10K              |
| Max_Pct Head           | 3      | ~10K              |
| **Total**              | **~45**| **~17M**          |

---

## 3. Input/Output Specification

### 3.1 Input Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT FEATURES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SEQUENCE DATA (96 timesteps × 14 channels)                         │   │
│  │                                                                      │   │
│  │  OHLC (4 features):          Volume (5 features):                   │   │
│  │  ├─ open  (log returns)       ├─ volume                             │   │
│  │  ├─ high  (log returns)       ├─ buy_volume                         │   │
│  │  ├─ low   (log returns)       ├─ sell_volume                        │   │
│  │  └─ close (log returns)       ├─ volume_delta                       │   │
│  │                               └─ cvd (cumulative volume delta)      │   │
│  │                                                                      │   │
│  │  Technical Indicators (5 features):                                 │   │
│  │  ├─ RSI (14-period)                                                 │   │
│  │  ├─ MACD (12,26,9)                                                  │   │
│  │  ├─ Bollinger %B                                                    │   │
│  │  ├─ ATR (14-period)                                                 │   │
│  │  └─ Momentum (10-period)                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TDA FEATURES (214 dimensions)                                      │   │
│  │                                                                      │   │
│  │  From 672-sample window (7 days @ 15-min):                          │   │
│  │  ├─ Betti Curve H0:     100 bins  ─────────┐                        │   │
│  │  ├─ Betti Curve H1:     100 bins           │                        │   │
│  │  ├─ Persistent Entropy:   2 dims           ├─ Total: 214 dims       │   │
│  │  ├─ Total Persistence:    2 dims           │                        │   │
│  │  └─ Landscape L2 Norms:  10 dims  ─────────┘                        │   │
│  │      (5 layers × 2 homology dims)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  COMPLEXITY INDICATORS (6 dimensions)                               │   │
│  │                                                                      │   │
│  │  ├─ MA Separation (20/50/100/200)                                   │   │
│  │  ├─ Bollinger Band Width                                            │   │
│  │  ├─ Price Efficiency                                                │   │
│  │  ├─ Support Reaction Strength                                       │   │
│  │  ├─ Directional Result per Time                                     │   │
│  │  └─ Volume-Price Alignment                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Output Predictions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT PREDICTIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────┐    ┌────────────────────────────┐          │
│  │      TRIGGER SIGNAL        │    │       MAX PERCENTAGE       │          │
│  │                            │    │                            │          │
│  │  Type: Binary (0 or 1)     │    │  Type: Continuous [0, 1]   │          │
│  │                            │    │                            │          │
│  │  Threshold: 0.5 (default)  │    │  Interpretation:           │          │
│  │              0.6 (optimal) │    │  Expected max % move       │          │
│  │                            │    │  within prediction window  │          │
│  │  0 = No trigger expected   │    │                            │          │
│  │  1 = Trigger expected      │    │  0.0 = No significant move │          │
│  │      (entry opportunity)   │    │  1.0 = Large move expected │          │
│  └────────────────────────────┘    └────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Training Configuration

### 4.1 Current Hyperparameters

| Category | Parameter | Value | Description |
|----------|-----------|-------|-------------|
| **Model** | model_type | N-BEATS | Architecture type |
| | lstm_hidden_size | 1024 | Hidden dimension (8× scaled) |
| | lstm_num_layers | 2 | Number of LSTM layers |
| | lstm_dropout | 0.3 | Dropout rate |
| | attention_heads | 8 | Multi-head attention |
| | tda_encoder_dim | 256 | TDA encoder output |
| | complexity_encoder_dim | 64 | Complexity encoder output |
| | shared_fc_dim | 512 | Shared FC dimension |
| | nbeats_num_stacks | 4 | Number of N-BEATS stacks |
| | nbeats_num_blocks | 6 | Blocks per stack |
| | nbeats_num_layers | 6 | FC layers per block |
| **Training** | batch_size | 2048 | Batch size (8× scaled) |
| | epochs | 500 | Maximum epochs |
| | learning_rate | 0.0005 | Initial LR |
| | weight_decay | 0.01 | L2 regularization |
| | gradient_clip | 1.0 | Gradient clipping |
| | early_stopping_patience | 30 | Early stopping |
| | gradient_accumulation_steps | 4 | Effective batch: 8192 |
| **Loss** | focal_alpha | 0.75 | Focal loss alpha |
| | focal_gamma | 2.0 | Focal loss gamma |
| | trigger_loss_weight | 3.0 | Classification weight |
| | max_pct_loss_weight | 0.3 | Regression weight |
| **TDA** | window_size | 672 | 7 days @ 15-min |
| | embedding_dim | 2 | Takens embedding |
| | time_delay | 12 | 3 hours delay |
| | betti_bins | 100 | Betti curve resolution |
| | landscape_layers | 5 | Persistence landscape |
| **GPU** | device | cuda:0 | NVIDIA RTX A6000 |
| | mixed_precision | true | FP16 training |
| | compile_model | true | torch.compile() |

### 4.2 Dataset Split

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATASET STATISTICS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Timeline:  2024-01-09 (ETF Launch) ─────────────────────────► 2025-12-29  │
│                                                                             │
│  ┌───────────────────────────┬───────────────────┬───────────────────────┐ │
│  │        TRAINING           │   VALIDATION      │         TEST          │ │
│  │    (50,064 samples)       │  (9,073 samples)  │    (8,641 samples)    │ │
│  │                           │                   │                       │ │
│  │  2024-01-09 ~ 2025-06-28  │ 2025-06-28~09-30  │  2025-09-30~12-29     │ │
│  │                           │                   │                       │ │
│  │  Trigger Rate: 12.3%      │  Trigger: 1.76%   │   Trigger: 10.17%     │ │
│  │  (With WeightedSampler)   │  (Very imbalanced)│   (Test reality)      │ │
│  └───────────────────────────┴───────────────────┴───────────────────────┘ │
│                                                                             │
│  Total: 67,778 samples (15-minute candles)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Results Analysis

### 5.1 Best Model Performance (Fixed Training Session)

| Metric | Training | Validation | Test | Target |
|--------|----------|------------|------|--------|
| **Precision** | 55% | 3.8% | **15.0%** | 60-70% |
| **Recall** | 86% | 50% | **85.8%** | 40-50% |
| **F1 Score** | 68% | 7.0% | **25.6%** | 50-60% |
| **AUC-ROC** | 81% | 72% | **74.5%** | >70% |
| **Accuracy** | 58% | 77% | **47.9%** | - |

### 5.2 Training Progress (26 epochs, early stopped at 16)

```
Loss Curve:

  0.22 ┤
       │▓
  0.20 ┤ ▓
       │  ▓
  0.18 ┤   ▓▓▓
       │      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  0.16 ┤                           ▓▓▓▓▓ ← Train Loss
       │
  0.14 ┤
       │
  0.12 ┤  ───────────────────────────── ← Val Loss
       │
  0.10 ┤
       │
  0.08 ┤  ○ Best Model (Epoch 16)
       └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──►
          2  4  6  8  10 12 14 16 18 20 22 24  Epoch

AUC-ROC Progress:

  0.85 ┤                        ○○○○○ ← Train AUC
       │                   ○○○○○
  0.80 ┤              ○○○○○
       │         ○○○○○
  0.75 ┤    ○○○○○
       │  ○○
  0.70 ┤ ○
       │  ─────────────────────── ← Val AUC (~0.70)
  0.65 ┤
       │
  0.60 ┤
       │
  0.55 ┤○
       └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──►
          2  4  6  8  10 12 14 16 18 20   Epoch
```

### 5.3 Confusion Matrix (Test Set)

```
                    Predicted
                 ┌─────────────────────┐
                 │    0     │    1     │
    ┌────────────┼──────────┼──────────┤
    │     0      │   3231   │   4207   │  ← Actual Negatives
    │  Actual    │   (TN)   │   (FP)   │     (7438 total)
    │────────────┼──────────┼──────────┤
    │     1      │    123   │   744    │  ← Actual Positives
    │  Actual    │   (FN)   │   (TP)   │     (867 total)
    └────────────┴──────────┴──────────┘

    Total Test Samples: 8,305

    Key Insights:
    ├─ High Recall (85.8%): Catches most actual triggers
    ├─ Low Precision (15.0%): Many false alarms
    └─ Trade-off: Prefer missing trades to losing money
```

---

## 6. Estimated Training Time by Steps

### 6.1 TDA Feature Extraction (One-time)

| Phase | Samples | Time/Sample | Total Time |
|-------|---------|-------------|------------|
| Training set | 50,064 | ~0.14 sec | ~2 hours |
| Validation set | 11,425 | ~0.14 sec | ~26 min |
| Test set | 7,969 | ~0.14 sec | ~18 min |
| **Total** | **~70K** | - | **~3 hours** |

*Note: Cached to `cache/tda_features_{train,val,test}.npy` - only needed once*

### 6.2 Model Training (Per Run)

| Step | GPU Utilization | Time (Est.) |
|------|-----------------|-------------|
| Data Loading | 10% | ~30 sec |
| Forward Pass (epoch) | 80-90% | ~2 sec |
| Backward Pass (epoch) | 80-90% | ~3 sec |
| Validation (epoch) | 60% | ~1 sec |
| **Total per Epoch** | - | **~6 sec** |
| **Full Training (500 epochs)** | - | **~50 min** |
| **With Early Stop (30 patience)** | - | **~3-10 min** |

### 6.3 GPU Resource Usage (RTX A6000)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GPU UTILIZATION (NVIDIA RTX A6000)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VRAM Usage:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 1-5 GB / 48 GB  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  (Currently underutilized - model could be scaled further)                 │
│                                                                             │
│  Compute Usage During Training:                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │████████████████████████████████████████░░░░░░░░░░│ ~80% peak        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Power Draw: 26W / 300W (idle), ~150W during training                      │
│  Temperature: 43°C (idle), ~65°C during training                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Observations & Issues

### 7.1 Current Challenges

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            IDENTIFIED ISSUES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SEVERE OVERFITTING                                                      │
│     ├─ Train Precision: 55% → Test Precision: 15%                          │
│     ├─ Train AUC: 0.81 → Test AUC: 0.74                                    │
│     └─ Gap indicates model memorizes training patterns                     │
│                                                                             │
│  2. VALIDATION SET IMBALANCE                                                │
│     ├─ Training trigger rate: 12.3%                                        │
│     ├─ Validation trigger rate: 1.76%  (6.98× lower!)                      │
│     └─ Causes threshold calibration issues                                 │
│                                                                             │
│  3. DISTRIBUTION SHIFT                                                      │
│     ├─ 2024-2025 market dynamics differ from test period                   │
│     ├─ Triggers may not be equally predictable across time                 │
│     └─ TDA features might capture different patterns                       │
│                                                                             │
│  4. LOW PRECISION (HIGH FALSE POSITIVE RATE)                                │
│     ├─ 85% of trigger predictions are wrong                                │
│     ├─ Trading costs would eat profits                                     │
│     └─ Need precision > 50% for profitable trading                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Experiment Results Summary

| Experiment | Sampler | Focal Alpha | Test Precision | Test Recall | Test AUC |
|------------|---------|-------------|----------------|-------------|----------|
| Baseline (failed) | Yes | 0.25 | 0% | 0% | 0.54 |
| **Best (fixed)** | **Yes** | **0.90** | **15%** | **86%** | **0.74** |
| Focal only | No | 0.75 | 0% | 0% | ~0.5 |
| Focal only | No | 0.85 | 0% | 0% | ~0.5 |
| Moderate | Yes | 0.75 | 10-12% | 50-90% | 0.53 |
| High focal only | No | 0.90 | 10% | 100% | 0.50 |

---

## 8. Next Steps & Recommendations

### 8.1 Short-term Improvements

1. **Stratified Temporal Split** (Implemented)
   - Balance trigger rates across train/val/test
   - Reduce distribution mismatch

2. **Feature Engineering**
   - Add RSI, MACD, Bollinger %B, ATR, Momentum (Implemented)
   - Expand complexity from 1 → 6 indicators (Implemented)

3. **Model Scaling** (Implemented)
   - Hidden size: 128 → 1024
   - Batch size: 256 → 2048
   - Longer training: 100 → 500 epochs

### 8.2 Medium-term Research

1. **Threshold Optimization**
   - Use validation set for threshold tuning
   - Consider precision-recall trade-off curve

2. **Ensemble Methods**
   - Combine LSTM + N-BEATS predictions
   - Bagging/boosting for robustness

3. **Feature Importance Analysis**
   - SHAP values for TDA features
   - Identify most predictive features

### 8.3 Target Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Precision | 15% | 60-70% | 45-55% |
| Recall | 86% | 40-50% | (over) |
| F1 Score | 26% | 50-60% | 24-34% |
| AUC-ROC | 74% | >80% | 6% |

---

## 9. Quick Reference Commands

```bash
# Activate environment
conda activate passive_income

# Run training
cd /home/ubuntu/joo/passive_income/src/tda_model
python -m scripts.train

# Monitor GPU
nvidia-smi -l 1

# Check training progress
tail -f logs/train_*.log

# Run experiments
python -m scripts.run_experiments

# Hyperparameter optimization
python -m scripts.hyperopt
```

---

## Appendix: TDA Feature Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TDA FEATURE COMPOSITION (214 dims)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. BETTI CURVES (200 dims)                                                 │
│     └─ Counts topological features at each filtration value                │
│        ├─ H0 (connected components): 100 bins                              │
│        └─ H1 (loops/cycles): 100 bins                                      │
│                                                                             │
│  2. PERSISTENT ENTROPY (2 dims)                                             │
│     └─ Measures complexity/disorder of persistence diagram                 │
│        ├─ H0 entropy: 1 value                                              │
│        └─ H1 entropy: 1 value                                              │
│                                                                             │
│  3. TOTAL PERSISTENCE (2 dims)                                              │
│     └─ L2 norm of all lifetimes (death - birth)                            │
│        ├─ H0 amplitude: 1 value                                            │
│        └─ H1 amplitude: 1 value                                            │
│                                                                             │
│  4. PERSISTENCE LANDSCAPE L2 NORMS (10 dims)                                │
│     └─ Functional summary capturing shape of diagram                       │
│        ├─ H0 layers 1-5: 5 values                                          │
│        └─ H1 layers 1-5: 5 values                                          │
│                                                                             │
│  Pipeline: Time Series → Takens Embedding → VR Complex → Persistence       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Document auto-generated from TDA model codebase analysis*
