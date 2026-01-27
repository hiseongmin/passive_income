# Plan: TDA-Standalone Model with Feature-Aware Architecture

## Objective
Build a self-contained model that uses ONLY TDA features from cache, with specialized processing for each feature type based on what it actually captures topologically.

---

## Part 1: Understanding Each TDA Feature

### Feature Inventory (214 total dimensions)

| Index | Feature | Dims | What It Captures | Variance | Verdict |
|-------|---------|------|------------------|----------|---------|
| 0-49 | Betti H0 | 50 | Connected components (fragmentation) | High at 0-15, low at 30-49 | **Keep 0-19** |
| 50-99 | Betti H1 | 50 | Loops/cycles (mean-reversion) | High at 0-15, low at 30-49 | **Keep 0-19** |
| 100 | Entropy H0 | 1 | Component diversity | **CONSTANT at 1.0** | **REMOVE** |
| 101 | Entropy H1 | 1 | Loop diversity (chaos indicator) | Good variance | **Keep** |
| 102 | Persistence H0 | 1 | Total structural activity | Good variance | **Keep** |
| 103 | Persistence H1 | 1 | Total cyclical activity | Good variance | **Keep** |
| 104-213 | Landscape | 110 | Multi-scale topology | Extreme outliers | **Keep with preprocessing** |

### What Each Feature Group Tells Us

**Group A: Structural Features (H0-based)**
```
Betti H0 curve shape:
  - Steep decay → Clean trend, simple structure
  - Slow decay → Fragmented market, many disconnected movements
  - High early values → More initial complexity

Persistence H0:
  - High → Lots of structural "activity" (components forming/merging)
  - Low → Simple, coherent price movement
```
**Use**: Trend coherence indicator, market structure quality

**Group B: Cyclical Features (H1-based)**
```
Betti H1 curve:
  - High values → Many loops forming (ranging/cycling market)
  - Low values → Few loops (trending market)

Entropy H1:
  - High → Many diverse loops (chaotic, unpredictable)
  - Low → Few dominant loops (predictable cycles)

Persistence H1:
  - High → Strong, long-lasting cycles (mean-reversion opportunity)
  - Low → Weak cycles (trend-following opportunity)
```
**Use**: Regime detection, strategy selection

**Group C: Landscape Features**
```
Persistence Landscapes:
  - Multi-resolution summary of persistence diagram
  - Captures features at different "importance" levels
  - Currently has EXTREME outliers (need preprocessing)
```
**Use**: Additional discriminative power after cleanup

---

## Part 2: Feature Engineering Strategy

### Step 1: Feature Selection & Cleaning

```python
# From 214 dims → ~65 useful dims

# Structural (H0)
betti_h0_selected = tda[:, 0:20]      # 20 dims (bins 0-19, drop 20-49)
persistence_h0 = tda[:, 102:103]       # 1 dim
# Total: 21 dims

# Cyclical (H1)
betti_h1_selected = tda[:, 50:70]      # 20 dims (bins 0-19, drop 20-49)
entropy_h1 = tda[:, 101:102]           # 1 dim
persistence_h1 = tda[:, 103:104]       # 1 dim
# Total: 22 dims

# Landscape (cleaned)
landscape_raw = tda[:, 104:214]        # 110 dims
# → Clip to [-50, 50]
# → RobustScaler
# → PCA to 20 dims
# Total: 20 dims

# REMOVED:
# - Entropy H0 (index 100) - constant, zero information
# - Betti H0 bins 20-49 - near-zero variance
# - Betti H1 bins 20-49 - near-zero variance

# Final: 21 + 22 + 20 = 63 dims (down from 214)
```

### Step 2: Derived Features

**From Betti Curves:**
```python
# Betti H0 derived
h0_mean = betti_h0.mean(axis=1)           # Overall structural complexity
h0_max = betti_h0.max(axis=1)             # Peak fragmentation
h0_decay_rate = fit_exponential_decay()   # How fast structure simplifies
h0_auc = betti_h0.sum(axis=1)             # Area under curve

# Betti H1 derived
h1_mean = betti_h1.mean(axis=1)           # Average loop count
h1_max = betti_h1.max(axis=1)             # Peak cyclicality
h1_peak_bin = betti_h1.argmax(axis=1)     # Where loops peak (scale information)
h1_auc = betti_h1.sum(axis=1)             # Total loop activity
```

### Step 3: Pre-computed Regime Labels

From our K-means analysis, we identified 4 regimes:
```python
# Pre-compute regime labels (self-supervised)
regime_features = [entropy_h1, persistence_h1, h1_mean]
kmeans = KMeans(n_clusters=4)
regime_labels = kmeans.fit_predict(regime_features)

# Regime interpretations:
# 0: "Simple" - low entropy, low persistence, low betti (trending)
# 1: "Chaotic" - high entropy (avoid trading)
# 2: "Cycling" - high persistence (mean-reversion)
# 3: "Complex" - high betti count (consolidation)
```

---

## Part 3: Model Architecture

### Design Philosophy
1. **Separate encoders** for each feature group (they capture different things)
2. **Explicit regime prediction** as auxiliary task
3. **Regime-conditioned fusion** for final prediction
4. **Confidence output** based on regime certainty

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│                     TDA Features from Cache (214 dims)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING MODULE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Select H0 bins  │  │ Select H1 bins  │  │ Landscape: Clip + Scale     │  │
│  │ [0:20] + Pers   │  │ [0:20]+Ent+Pers │  │ + PCA (110→20)              │  │
│  │ Remove Ent H0   │  │                 │  │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │ (21 dims)          │ (22 dims)                │ (20 dims)       │
└───────────┼────────────────────┼──────────────────────────┼─────────────────┘
            │                    │                          │
            ▼                    ▼                          ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────────┐
│  STRUCTURAL       │  │  CYCLICAL         │  │  LANDSCAPE                    │
│  ENCODER          │  │  ENCODER          │  │  ENCODER                      │
│                   │  │                   │  │                               │
│  Linear(21→64)    │  │  Linear(22→64)    │  │  Linear(20→48)                │
│  LayerNorm        │  │  LayerNorm        │  │  LayerNorm                    │
│  GELU             │  │  GELU             │  │  GELU                         │
│  Dropout(0.2)     │  │  Dropout(0.2)     │  │  Dropout(0.3) [noisy input]   │
│  Linear(64→32)    │  │  Linear(64→32)    │  │  Linear(48→24)                │
│  LayerNorm        │  │  LayerNorm        │  │  LayerNorm                    │
│                   │  │                   │  │                               │
│  Output: 32 dims  │  │  Output: 32 dims  │  │  Output: 24 dims              │
│  "struct_emb"     │  │  "cycle_emb"      │  │  "landscape_emb"              │
└─────────┬─────────┘  └─────────┬─────────┘  └───────────────┬───────────────┘
          │                      │                            │
          │                      ├──────────────┐             │
          │                      │              ▼             │
          │                      │    ┌─────────────────────┐ │
          │                      │    │  REGIME CLASSIFIER  │ │
          │                      │    │  (Auxiliary Task)   │ │
          │                      │    │                     │ │
          │                      │    │  Linear(32→16)      │ │
          │                      │    │  GELU               │ │
          │                      │    │  Linear(16→4)       │ │
          │                      │    │  Softmax            │ │
          │                      │    │                     │ │
          │                      │    │  Output: regime_prob│ │
          │                      │    │  (4 classes)        │ │
          │                      │    └──────────┬──────────┘ │
          │                      │               │            │
          ▼                      ▼               ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REGIME-CONDITIONED FUSION                            │
│                                                                              │
│  Concatenate: [struct_emb(32) + cycle_emb(32) + landscape_emb(24)] = 88     │
│                                                                              │
│  Regime Embedding: regime_prob → Linear(4→16) → regime_emb                  │
│                                                                              │
│  Modulation: fused = concat_features * (1 + tanh(regime_emb_expanded))      │
│              [Feature-wise scaling based on regime]                          │
│                                                                              │
│  Output: fused_features (88 dims)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TASK HEADS                                      │
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │      TRIGGER HEAD           │    │      CONFIDENCE HEAD                │ │
│  │                             │    │      (Based on Regime Certainty)    │ │
│  │  Linear(88→32)              │    │                                     │ │
│  │  GELU                       │    │  confidence = 1 - entropy(regime)   │ │
│  │  Dropout(0.2)               │    │  [High regime certainty → high conf]│ │
│  │  Linear(32→1)               │    │                                     │ │
│  │  Sigmoid                    │    │  Output: confidence_score (0-1)     │ │
│  │                             │    │                                     │ │
│  │  Output: trigger_prob (0-1) │    └─────────────────────────────────────┘ │
│  └─────────────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘

OUTPUTS:
  1. trigger_prob: Probability of trigger event (primary task)
  2. regime_prob: 4-class regime distribution (auxiliary task)
  3. confidence: How confident the model is (derived from regime certainty)
```

### Parameter Count Estimate
```
Structural Encoder:  21×64 + 64×32 = 1,344 + 2,048 = ~3.5K
Cyclical Encoder:    22×64 + 64×32 = 1,408 + 2,048 = ~3.5K
Landscape Encoder:   20×48 + 48×24 = 960 + 1,152 = ~2.1K
Regime Classifier:   32×16 + 16×4 = 512 + 64 = ~0.6K
Fusion Modulator:    4×16 + 16×88 = 64 + 1,408 = ~1.5K
Trigger Head:        88×32 + 32×1 = 2,816 + 32 = ~2.8K
LayerNorms + biases: ~1K

Total: ~15K parameters (very lightweight!)
```

---

## Part 4: Training Strategy

### Loss Function

```python
# Multi-task loss
loss = (
    1.0 * BCE(trigger_pred, trigger_label) +      # Primary task
    0.3 * CE(regime_pred, regime_label) +          # Auxiliary (self-supervised)
    0.1 * entropy_regularization(regime_pred)      # Encourage confident regimes
)
```

### Self-Supervised Regime Labels

```python
# Pre-compute regime labels before training (one-time)
def compute_regime_labels(tda_features):
    # Extract H1-based features for clustering
    h1_betti_mean = tda_features[:, 50:70].mean(axis=1)
    h1_entropy = tda_features[:, 101]
    h1_persistence = tda_features[:, 103]

    X = np.column_stack([h1_betti_mean, h1_entropy, h1_persistence])
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    regime_labels = kmeans.fit_predict(X_scaled)

    return regime_labels, kmeans.cluster_centers_
```

### Training Configuration

```yaml
# config/tda_standalone.yaml
model:
  structural_hidden: 64
  cyclical_hidden: 64
  landscape_hidden: 48
  fusion_dim: 88
  dropout: 0.2
  landscape_dropout: 0.3  # Higher for noisy features

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.01
  epochs: 100
  early_stopping_patience: 15

  # Loss weights
  trigger_weight: 1.0
  regime_weight: 0.3
  entropy_reg_weight: 0.1

preprocessing:
  betti_bins_to_keep: 20  # 0-19 for both H0 and H1
  landscape_clip_range: [-50, 50]
  landscape_pca_dims: 20
  remove_entropy_h0: true

data:
  tda_cache_path: "cache/tda_features_{split}.npy"
  labels_path: "data/BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv"
```

---

## Part 5: File Structure (Self-Contained)

```
src/tda_standalone/
├── __init__.py
├── config.py                 # Configuration dataclasses
├── preprocessing.py          # Feature selection, cleaning, PCA
├── dataset.py               # TDA-only dataset (loads from cache + labels)
├── model.py                 # TDAStandaloneModel architecture
├── regime.py                # Regime clustering and label generation
├── losses.py                # Multi-task loss functions
├── train.py                 # Training script
├── evaluate.py              # Evaluation and metrics
└── configs/
    └── default.yaml         # Default configuration
```

### Key Files Description

**preprocessing.py**
```python
class TDAPreprocessor:
    """
    Handles all TDA feature preprocessing:
    - Feature selection (keep useful bins)
    - Remove constant features (Entropy H0)
    - Landscape cleaning (clip, scale, PCA)
    - Derived feature computation
    """
    def __init__(self, config):
        self.betti_bins = config.betti_bins_to_keep
        self.landscape_pca = PCA(n_components=config.landscape_pca_dims)

    def fit(self, tda_features):
        # Fit PCA on landscape features
        landscape = tda_features[:, 104:214]
        landscape_clipped = np.clip(landscape, -50, 50)
        self.landscape_pca.fit(landscape_clipped)

    def transform(self, tda_features):
        # Returns: structural (21), cyclical (22), landscape (20)
        ...
```

**dataset.py**
```python
class TDAStandaloneDataset(Dataset):
    """
    Self-contained dataset using only cached TDA features.
    Loads:
    - TDA features from cache/*.npy
    - Labels from CSV (Trigger, Max_Pct)
    - Pre-computed regime labels
    """
    def __init__(self, split, config, preprocessor, regime_labels):
        self.tda = np.load(f"cache/tda_features_{split}.npy")
        self.labels = self._load_labels(split)
        self.regime_labels = regime_labels
        self.preprocessor = preprocessor

    def __getitem__(self, idx):
        tda_raw = self.tda[idx]
        structural, cyclical, landscape = self.preprocessor.transform(tda_raw)

        return {
            'structural': structural,
            'cyclical': cyclical,
            'landscape': landscape,
            'trigger': self.labels['trigger'][idx],
            'regime': self.regime_labels[idx],
        }
```

**model.py**
```python
class TDAStandaloneModel(nn.Module):
    """
    Feature-aware TDA model with:
    - Separate encoders for structural/cyclical/landscape
    - Regime classification auxiliary task
    - Regime-conditioned fusion
    - Confidence estimation
    """
    def __init__(self, config):
        self.structural_encoder = StructuralEncoder(21, config.structural_hidden)
        self.cyclical_encoder = CyclicalEncoder(22, config.cyclical_hidden)
        self.landscape_encoder = LandscapeEncoder(20, config.landscape_hidden)
        self.regime_classifier = RegimeClassifier(config.cyclical_hidden, 4)
        self.fusion = RegimeConditionedFusion(config.fusion_dim, 4)
        self.trigger_head = TriggerHead(config.fusion_dim)

    def forward(self, structural, cyclical, landscape):
        # Encode each feature group
        struct_emb = self.structural_encoder(structural)
        cycle_emb = self.cyclical_encoder(cyclical)
        land_emb = self.landscape_encoder(landscape)

        # Predict regime from cyclical features
        regime_logits = self.regime_classifier(cycle_emb)
        regime_prob = F.softmax(regime_logits, dim=-1)

        # Compute confidence from regime certainty
        confidence = 1 - self._entropy(regime_prob)

        # Regime-conditioned fusion
        fused = self.fusion(struct_emb, cycle_emb, land_emb, regime_prob)

        # Predict trigger
        trigger_logits = self.trigger_head(fused)

        return {
            'trigger_logits': trigger_logits,
            'regime_logits': regime_logits,
            'confidence': confidence,
        }
```

---

## Part 6: What Makes This Different

### vs. Current TDA Usage (in tda_model/hybrid_fusion)

| Aspect | Current Approach | This Approach |
|--------|------------------|---------------|
| Feature treatment | All 214 dims treated equally | Specialized by function |
| Entropy H0 | Included (but useless) | Removed |
| Betti bins | All 50 used | Only 0-19 (informative) |
| Landscape | Raw (with outliers) | Clipped + PCA |
| Regime | Implicit in fusion | Explicit auxiliary task |
| Confidence | Not computed | Derived from regime certainty |
| Architecture | Single encoder | Three specialized encoders |

### Key Innovations

1. **Feature-Function Alignment**: Each encoder processes features that capture related information
2. **Explicit Regime Awareness**: Model knows what regime it's in
3. **Self-Supervised Regime Labels**: No manual labeling needed
4. **Confidence Estimation**: Know when NOT to trust the prediction
5. **Lightweight**: ~15K params vs millions in full model

---

## Part 7: Expected Outcomes

### What This Model Should Learn

| Feature Group | Expected Learning |
|---------------|-------------------|
| Structural (H0) | "Is the market structurally coherent?" → Trend quality |
| Cyclical (H1) | "What regime is the market in?" → Strategy selection |
| Landscape | "Additional discriminative patterns" → Fine-tuning |

### Predictions

1. **Trigger Prediction**: When to enter a trade
2. **Regime Prediction**: What type of market we're in
3. **Confidence**: How much to trust the trigger signal

### Use Case

```python
output = model(structural, cyclical, landscape)

if output['trigger_prob'] > 0.5 and output['confidence'] > 0.7:
    # High-confidence trigger - take the trade
    execute_trade()
elif output['trigger_prob'] > 0.5 and output['confidence'] < 0.4:
    # Low-confidence trigger - skip or reduce size
    skip_or_reduce()
```

---

## Part 8: Implementation Order

1. **Create folder structure** - `src/tda_standalone/`
2. **Implement preprocessing.py** - Feature selection, cleaning, PCA fitting
3. **Implement regime.py** - K-means clustering, regime label generation
4. **Implement dataset.py** - Load TDA cache + labels + regimes
5. **Implement model.py** - Three encoders + fusion + heads
6. **Implement losses.py** - Multi-task loss
7. **Implement train.py** - Training loop with validation
8. **Implement evaluate.py** - Metrics, confusion matrices, regime analysis
9. **Create default config** - YAML configuration
10. **Test end-to-end** - Verify no crashes, reasonable outputs

---

## Summary

This plan creates a **self-contained, feature-aware TDA model** that:

1. **Removes useless features** (Entropy H0, high Betti bins)
2. **Cleans problematic features** (Landscape outliers)
3. **Specializes processing** by topological meaning
4. **Explicitly models regime** as auxiliary task
5. **Provides confidence** based on regime certainty
6. **Stays lightweight** (~15K parameters)
7. **Doesn't interfere** with existing models

The key insight: **TDA features tell us about market STATE, not price DIRECTION**. This architecture embraces that by using TDA for regime conditioning rather than direct prediction.
