# Cross-Modal Attention Transformer Fusion

> **SELF-CONTAINED IMPLEMENTATION**: This specification includes all components needed for end-to-end training. No external dependencies required.

## 0. Self-Contained Encoders

This model includes its own encoder implementations for complete independence from external code.

### 0.1 Input Specifications

| Input | Shape | Description |
|-------|-------|-------------|
| OHLCV Sequence | (B, 96, 14) | 96 timesteps × 14 channels (OHLC + Volume + Technical) |
| TDA Features | (B, 214) | Topological features (Betti curves, entropy, landscapes) |
| Complexity | (B, 6) | Market complexity indicators |

### 0.2 Encoder Architecture Summary

```
RAW INPUTS
├── OHLCV (B, 96, 14)
│   └── OHLCVNBEATSEncoder
│       ├── BatchNorm1d(14)
│       ├── Flatten → (B, 1344)
│       ├── NBEATSStack[Trend, Seasonality, Generic, Generic]
│       ├── Concat → (B, 1024)
│       ├── AttentionBlock(1024)
│       └── Output → (B, 1024)
│
├── TDA (B, 214)
│   └── TDAEncoder: Linear(214→128→256) → (B, 256)
│
└── Complexity (B, 6)
    └── ComplexityEncoder: Linear(6→64→64) → (B, 64)

ENCODER OUTPUTS → FUSION INPUT
├── nbeats_features: (B, 1024)
├── tda_features: (B, 256)
└── complexity_features: (B, 64)
```

### 0.3 Encoder Parameter Counts

| Encoder | Parameters |
|---------|------------|
| OHLCVNBEATSEncoder | ~13.7M |
| TDAEncoder | ~61K |
| ComplexityEncoder | ~5K |
| **Total Encoders** | **~13.8M** |

---

## 1. Architecture Overview

### Concept
Replace simple concatenation with a Transformer-based cross-modal attention mechanism. Each modality (N-BEATS, TDA, Complexity) becomes a "token" that can attend to other modalities, learning which features from one modality are relevant given the others.

### Rationale
- **Problem**: Current architecture concatenates features without learning inter-modal relationships
- **Solution**: Cross-attention allows TDA topology features to "query" relevant price patterns from N-BEATS
- **Benefit**: Model learns dynamic feature importance based on current market state

### Key Innovation
Instead of treating encoded features as independent vectors to concatenate, we treat them as tokens in a sequence and apply self-attention, enabling:
- TDA features to attend to relevant N-BEATS patterns
- N-BEATS to attend to relevant topological structures
- Complexity to modulate attention weights

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-MODAL ATTENTION TRANSFORMER                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS (from existing encoders):                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│  │  N-BEATS    │   │    TDA      │   │ Complexity  │                       │
│  │  (B, 1024)  │   │  (B, 256)   │   │   (B, 64)   │                       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                       │
│         │                 │                 │                               │
│         ▼                 ▼                 ▼                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│  │ Project     │   │ Project     │   │ Project     │                       │
│  │ 1024→256    │   │ 256→256     │   │ 64→256      │                       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                       │
│         │                 │                 │                               │
│         └─────────────────┼─────────────────┘                               │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │  Stack Tokens   │                                        │
│                  │  (B, 3, 256)    │  ← 3 modality tokens                   │
│                  └────────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │ + Positional    │                                        │
│                  │   Embedding     │  (optional, learnable)                 │
│                  └────────┬────────┘                                        │
│                           │                                                 │
│              ┌────────────┼────────────┐                                    │
│              │            │            │                                    │
│              ▼            ▼            ▼                                    │
│         ┌─────────────────────────────────┐                                 │
│         │    TRANSFORMER ENCODER BLOCK 1   │                                │
│         │  ┌───────────────────────────┐  │                                 │
│         │  │ Multi-Head Self-Attention │  │  8 heads, d_k=32               │
│         │  │ Q, K, V all from tokens   │  │                                 │
│         │  └─────────────┬─────────────┘  │                                 │
│         │                │                │                                 │
│         │         ┌──────┴──────┐         │                                 │
│         │         │ Add & Norm  │         │  Residual + LayerNorm          │
│         │         └──────┬──────┘         │                                 │
│         │                │                │                                 │
│         │  ┌─────────────┴─────────────┐  │                                 │
│         │  │    Feed-Forward Network   │  │  256→1024→256                  │
│         │  └─────────────┬─────────────┘  │                                 │
│         │                │                │                                 │
│         │         ┌──────┴──────┐         │                                 │
│         │         │ Add & Norm  │         │  Residual + LayerNorm          │
│         │         └─────────────┘         │                                 │
│         └─────────────────────────────────┘                                 │
│                           │                                                 │
│                           ▼                                                 │
│         ┌─────────────────────────────────┐                                 │
│         │    TRANSFORMER ENCODER BLOCK 2   │  (same structure)              │
│         └─────────────────────────────────┘                                 │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │   Aggregation   │                                        │
│                  │  Mean Pool or   │                                        │
│                  │  [CLS] token    │                                        │
│                  │  → (B, 256)     │                                        │
│                  └────────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │ Output Project  │                                        │
│                  │ 256→256 + ReLU  │                                        │
│                  │ + Dropout(0.3)  │                                        │
│                  └────────┬────────┘                                        │
│                           │                                                 │
│           ┌───────────────┴───────────────┐                                 │
│           ▼                               ▼                                 │
│   ┌──────────────┐               ┌──────────────┐                           │
│   │ Trigger Head │               │ Max_Pct Head │                           │
│   │  256→32→1    │               │  256→32→1    │                           │
│   └──────────────┘               └──────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer-by-Layer Specification

| Layer | Input Shape | Output Shape | Parameters | Notes |
|-------|-------------|--------------|------------|-------|
| proj_nbeats | (B, 1024) | (B, 256) | 262,400 | Linear projection |
| proj_tda | (B, 256) | (B, 256) | 65,792 | Linear projection |
| proj_complexity | (B, 64) | (B, 256) | 16,640 | Linear projection |
| pos_embedding | (3, 256) | (3, 256) | 768 | Learnable positions |
| transformer_layer_1.self_attn | (B, 3, 256) | (B, 3, 256) | 263,168 | 8-head attention |
| transformer_layer_1.ffn | (B, 3, 256) | (B, 3, 256) | 525,312 | 256→1024→256 |
| transformer_layer_1.norm1 | (B, 3, 256) | (B, 3, 256) | 512 | LayerNorm |
| transformer_layer_1.norm2 | (B, 3, 256) | (B, 3, 256) | 512 | LayerNorm |
| transformer_layer_2.* | (B, 3, 256) | (B, 3, 256) | 789,504 | Same as layer 1 |
| output_proj | (B, 256) | (B, 256) | 65,792 | Final projection |
| **Total Fusion Module** | - | - | **~1.99M** | - |

---

## 4. Full PyTorch Pseudocode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Transformer for multi-modal fusion.

    Replaces simple concatenation with transformer-based cross-attention
    between N-BEATS, TDA, and Complexity modalities.
    """

    def __init__(
        self,
        nbeats_dim: int = 1024,
        tda_dim: int = 256,
        complexity_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.3,
        use_cls_token: bool = False,
    ):
        """
        Args:
            nbeats_dim: N-BEATS encoder output dimension (1024)
            tda_dim: TDA encoder output dimension (256)
            complexity_dim: Complexity encoder output dimension (64)
            hidden_dim: Transformer hidden dimension (256)
            num_heads: Number of attention heads (8)
            num_layers: Number of transformer layers (2)
            ffn_dim: Feed-forward network dimension (1024)
            dropout: Dropout rate (0.3)
            use_cls_token: Whether to use [CLS] token for aggregation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token
        self.num_modalities = 3  # N-BEATS, TDA, Complexity

        # === Projection Layers ===
        # Project each modality to same dimension
        self.proj_nbeats = nn.Linear(nbeats_dim, hidden_dim)
        self.proj_tda = nn.Linear(tda_dim, hidden_dim)
        self.proj_complexity = nn.Linear(complexity_dim, hidden_dim)

        # === Positional Encoding ===
        # Learnable positional embeddings for each modality
        num_positions = 4 if use_cls_token else 3
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_positions, hidden_dim) * 0.02
        )

        # === Optional CLS Token ===
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # === Output Projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output dimension for downstream heads
        self.output_dim = hidden_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in [self.proj_nbeats, self.proj_tda, self.proj_complexity]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        nbeats_features: torch.Tensor,    # (B, 1024)
        tda_features: torch.Tensor,       # (B, 256)
        complexity_features: torch.Tensor, # (B, 64)
    ) -> torch.Tensor:
        """
        Forward pass with cross-modal attention.

        Args:
            nbeats_features: N-BEATS encoder output (B, 1024)
            tda_features: TDA encoder output (B, 256)
            complexity_features: Complexity encoder output (B, 64)

        Returns:
            Fused representation (B, hidden_dim=256)
        """
        batch_size = nbeats_features.size(0)

        # === Step 1: Project all modalities to same dimension ===
        nbeats_proj = self.proj_nbeats(nbeats_features)  # (B, 256)
        tda_proj = self.proj_tda(tda_features)            # (B, 256)
        complexity_proj = self.proj_complexity(complexity_features)  # (B, 256)

        # === Step 2: Stack as sequence of tokens ===
        # Each modality becomes a "token" in the sequence
        tokens = torch.stack([nbeats_proj, tda_proj, complexity_proj], dim=1)
        # Shape: (B, 3, 256)

        # === Step 3: Add CLS token if using ===
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, 256)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 4, 256)

        # === Step 4: Add positional embeddings ===
        tokens = tokens + self.pos_embedding  # (B, 3 or 4, 256)

        # === Step 5: Apply Transformer Encoder ===
        # Self-attention allows each modality to attend to others
        transformed = self.transformer(tokens)  # (B, 3 or 4, 256)

        # === Step 6: Aggregate tokens ===
        if self.use_cls_token:
            # Use CLS token as aggregate representation
            aggregated = transformed[:, 0, :]  # (B, 256)
        else:
            # Mean pooling over all modality tokens
            aggregated = transformed.mean(dim=1)  # (B, 256)

        # === Step 7: Output projection ===
        output = self.output_proj(aggregated)  # (B, 256)

        return output


class MultiTaskCrossModalAttention(nn.Module):
    """
    Complete multi-task model with Cross-Modal Attention fusion.

    Drop-in replacement for MultiTaskNBEATS with improved fusion.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config

        # === Existing Encoders (unchanged) ===
        # Import from existing nbeats.py
        from .nbeats import OHLCVNBEATSEncoder, TDAEncoder, ComplexityEncoder

        num_channels = (
            config.model.ohlcv_features +      # 4
            config.model.volume_features +      # 5
            config.model.technical_features     # 5
        )  # Total: 14

        self.ohlcv_encoder = OHLCVNBEATSEncoder(
            input_size=config.model.ohlcv_sequence_length,
            output_size=1,
            hidden_size=config.model.lstm_hidden_size,
            num_stacks=config.model.nbeats_num_stacks,
            num_blocks=config.model.nbeats_num_blocks,
            num_layers=config.model.nbeats_num_layers,
            dropout=config.model.lstm_dropout,
            stack_types=['trend', 'seasonality', 'generic', 'generic'],
            num_channels=num_channels,
            use_attention=config.model.use_attention,
        )

        self.tda_encoder = TDAEncoder(
            input_dim=config.tda_feature_dim,
            output_dim=config.model.tda_encoder_dim,
            dropout=config.model.lstm_dropout,
        )

        self.complexity_encoder = ComplexityEncoder(
            input_dim=config.model.complexity_features,
            output_dim=config.model.complexity_encoder_dim,
        )

        # === NEW: Cross-Modal Attention Fusion ===
        self.fusion = CrossModalAttentionFusion(
            nbeats_dim=config.model.lstm_hidden_size,      # 1024
            tda_dim=config.model.tda_encoder_dim,          # 256
            complexity_dim=config.model.complexity_encoder_dim,  # 64
            hidden_dim=config.model.fusion_hidden_dim,     # 256 (new config)
            num_heads=config.model.fusion_num_heads,       # 8 (new config)
            num_layers=config.model.fusion_num_layers,     # 2 (new config)
            ffn_dim=config.model.fusion_ffn_dim,           # 1024 (new config)
            dropout=config.model.lstm_dropout,
            use_cls_token=config.model.fusion_use_cls_token,  # False (new config)
        )

        # === Task-Specific Heads (unchanged) ===
        fusion_out_dim = self.fusion.output_dim  # 256

        self.trigger_head = nn.Sequential(
            nn.Linear(fusion_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.max_pct_head = nn.Sequential(
            nn.Linear(fusion_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ohlcv_seq: torch.Tensor,
        tda_features: torch.Tensor,
        complexity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            ohlcv_seq: OHLCV sequence (B, seq_len, 14)
            tda_features: TDA features (B, 214)
            complexity: Complexity indicators (B, 6)

        Returns:
            trigger_logits: (B, 1)
            max_pct_pred: (B, 1)
        """
        # === Encode each modality ===
        ohlcv_encoded = self.ohlcv_encoder(ohlcv_seq)      # (B, 1024)
        tda_encoded = self.tda_encoder(tda_features)       # (B, 256)
        complexity_encoded = self.complexity_encoder(complexity)  # (B, 64)

        # === Cross-Modal Attention Fusion ===
        fused = self.fusion(ohlcv_encoded, tda_encoded, complexity_encoded)
        # Shape: (B, 256)

        # === Task-Specific Heads ===
        trigger_logits = self.trigger_head(fused)  # (B, 1)
        max_pct_pred = self.max_pct_head(fused)    # (B, 1)

        return trigger_logits, max_pct_pred

    def predict(
        self,
        ohlcv_seq: torch.Tensor,
        tda_features: torch.Tensor,
        complexity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with sigmoid applied to trigger."""
        trigger_logits, max_pct_pred = self.forward(ohlcv_seq, tda_features, complexity)
        trigger_prob = torch.sigmoid(trigger_logits)
        return trigger_prob, max_pct_pred

    def get_num_parameters(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total
```

---

## 5. Integration Points

### 5.1 Replace in `models/nbeats.py`

```python
# OLD CODE (lines 645-653):
# self.shared_fc = nn.Sequential(
#     nn.Linear(total_dim, config.model.shared_fc_dim),
#     nn.ReLU(),
#     nn.Dropout(config.model.lstm_dropout),
#     nn.Linear(config.model.shared_fc_dim, config.model.shared_fc_dim // 2),
#     nn.ReLU(),
#     nn.Dropout(config.model.lstm_dropout),
# )

# NEW CODE:
from .cross_modal_attention import CrossModalAttentionFusion

self.fusion = CrossModalAttentionFusion(
    nbeats_dim=self.ohlcv_encoder.output_dim,
    tda_dim=self.tda_encoder.output_dim,
    complexity_dim=self.complexity_encoder.output_dim,
    hidden_dim=config.model.fusion_hidden_dim,
    num_heads=config.model.fusion_num_heads,
    num_layers=config.model.fusion_num_layers,
    ffn_dim=config.model.fusion_ffn_dim,
    dropout=config.model.lstm_dropout,
)
```

### 5.2 Update `forward()` method

```python
# OLD CODE (lines 695-700):
# combined = torch.cat([ohlcv_encoded, tda_encoded, complexity_encoded], dim=1)
# shared_out = self.shared_fc(combined)

# NEW CODE:
fused = self.fusion(ohlcv_encoded, tda_encoded, complexity_encoded)

# Update head input dimension
trigger_logits = self.trigger_head(fused)
max_pct_pred = self.max_pct_head(fused)
```

---

## 6. Configuration Parameters

### Add to `config/config.py` (ModelConfig dataclass):

```python
@dataclass
class ModelConfig:
    # ... existing parameters ...

    # Cross-Modal Attention Fusion parameters
    fusion_type: str = "cross_modal_attention"  # Options: "concat", "cross_modal_attention", etc.
    fusion_hidden_dim: int = 256      # Transformer hidden dimension
    fusion_num_heads: int = 8         # Number of attention heads
    fusion_num_layers: int = 2        # Number of transformer layers
    fusion_ffn_dim: int = 1024        # Feed-forward network dimension
    fusion_use_cls_token: bool = False  # Whether to use [CLS] token
```

### Add to `config/default_config.yaml`:

```yaml
model:
  # ... existing parameters ...

  # Fusion configuration
  fusion_type: "cross_modal_attention"
  fusion_hidden_dim: 256
  fusion_num_heads: 8
  fusion_num_layers: 2
  fusion_ffn_dim: 1024
  fusion_use_cls_token: false
```

---

## 7. Complexity Analysis

### Parameter Count Comparison

| Component | Current (Concat) | Cross-Modal Attention |
|-----------|------------------|----------------------|
| Projection layers | 0 | 344,832 |
| Positional embedding | 0 | 768 |
| Transformer layers (×2) | 0 | 1,579,008 |
| Output projection | 0 | 65,792 |
| shared_fc | 820,736 | 0 (replaced) |
| **Fusion Total** | **820,736** | **1,990,400** |
| **Delta** | - | **+1,169,664 (+142%)** |

### Computational Complexity

| Metric | Current | Cross-Modal Attention |
|--------|---------|----------------------|
| FLOPs per sample | ~1.6M | ~4.2M |
| Memory (batch=2048) | ~3.2 GB | ~4.8 GB |
| Training time/epoch | ~6 sec | ~8-10 sec |

### Memory Breakdown (batch_size=2048)
- Token storage: 2048 × 3 × 256 × 4 bytes = 6.3 MB
- Attention matrices: 2048 × 8 × 3 × 3 × 4 bytes = 0.6 MB per layer
- FFN activations: 2048 × 3 × 1024 × 4 bytes = 25 MB per layer
- **Total fusion overhead**: ~100-150 MB (well within RTX A6000 48GB)

---

## 8. Expected Benefits

### 8.1 Improved Cross-Modal Learning
- **Before**: Features from different modalities are simply concatenated
- **After**: Attention mechanism learns which TDA features are relevant for specific price patterns

### 8.2 Dynamic Feature Weighting
- Attention weights change based on input, allowing model to focus on different features for different market conditions

### 8.3 Better Generalization
- Transformer's pre-LN and residual connections improve gradient flow
- Less prone to overfitting than simple MLP fusion

### 8.4 Interpretability
- Attention weights can be visualized to understand which modalities are important for each prediction

### 8.5 Expected Metric Improvements
| Metric | Current | Expected |
|--------|---------|----------|
| Test Precision | 15% | 25-35% |
| Test AUC-ROC | 0.74 | 0.78-0.82 |
| Train-Test Gap | 40% | 20-25% |

---

## 9. Training Tips

1. **Learning Rate**: Use lower learning rate (0.0001-0.0003) for transformer layers
2. **Warmup**: Apply learning rate warmup for first 5-10% of training
3. **Layer Freezing**: Consider freezing encoders for first few epochs to stabilize fusion learning
4. **Gradient Clipping**: Keep gradient clipping at 1.0 for stability
5. **Regularization**: Attention dropout already included; may reduce weight decay

---

## 10. File Location

**Implemented as standalone module**: `src/cross_modal_attention/`

---

## 11. Implementation Details

### 11.1 Project Structure

```
src/cross_modal_attention/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── config.py                # Config with fusion-specific params
│   └── default_config.yaml      # Default hyperparameters
├── data/
│   ├── __init__.py
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── dataset.py               # PyTorch dataset
│   └── data_loader.py           # DataLoader creation
├── tda/
│   ├── __init__.py
│   ├── point_cloud.py           # Takens embedding
│   ├── persistence.py           # Persistence diagram computation
│   └── features.py              # TDA feature extraction
├── models/
│   ├── __init__.py
│   ├── encoders.py              # N-BEATS, TDA, Complexity encoders
│   ├── cross_modal_attention.py # Cross-modal attention fusion
│   └── losses.py                # MultiTaskLoss (Focal + MSE)
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop with GPU optimizations
│   └── metrics.py               # Evaluation metrics
└── scripts/
    ├── __init__.py
    └── train.py                 # Main entry point
```

### 11.2 Training Command

```bash
cd /home/ubuntu/joo/passive_income

# Train with default config
python -m src.cross_modal_attention.scripts.train \
    --cache-dir cache \
    --data-dir data_flagged \
    --log-level INFO

# Train with custom parameters
python -m src.cross_modal_attention.scripts.train \
    --epochs 100 \
    --batch-size 1024 \
    --lr 0.0003 \
    --fusion-hidden-dim 256 \
    --fusion-num-heads 8 \
    --fusion-num-layers 2
```

### 11.3 Configuration Parameters

```yaml
# src/cross_modal_attention/config/default_config.yaml

tda:
  window_size: 672          # 7 days of 15-min candles
  embedding_dim: 2
  time_delay: 12
  betti_bins: 100
  landscape_layers: 5
  homology_dimensions: [0, 1]

fusion:
  fusion_type: "cross_modal_attention"
  hidden_dim: 256           # Transformer d_model
  num_heads: 8              # Multi-head attention heads
  num_layers: 2             # Transformer encoder layers
  ffn_dim: 1024             # FFN hidden dimension
  dropout: 0.1
  use_cls_token: false      # Use mean pooling instead
  output_dim: 256

model:
  model_type: "cross_modal_attention"
  lstm_hidden_size: 1024    # N-BEATS output dim
  tda_encoder_dim: 256
  complexity_encoder_dim: 64
  nbeats_num_stacks: 4
  nbeats_num_blocks: 6
  nbeats_num_layers: 6

training:
  batch_size: 2048
  learning_rate: 0.0005
  weight_decay: 0.01
  epochs: 500
  early_stopping_patience: 30
  gradient_accumulation_steps: 4

logging:
  save_dir: "models/cross_modal_attention"
```

### 11.4 Key Implementation Notes

1. **Standalone Module**: The cross-modal attention model is implemented as a completely standalone module that doesn't modify any existing tda_model code.

2. **TDA Feature Caching**: Uses the same TDA cache (`cache/tda_features_*.npy`) as the existing tda_model since TDA parameters are unchanged.

3. **GPU Optimizations**: Retains all RTX A6000 optimizations:
   - Mixed precision training (AMP)
   - torch.compile() for faster execution
   - Large batch sizes (2048)
   - Gradient accumulation (4 steps)
   - cuDNN benchmark mode

4. **Pre-LN Transformer**: Uses Pre-LayerNorm configuration (`norm_first=True`) for better training stability.

5. **GELU Activation**: Uses GELU instead of ReLU in transformer layers for smoother gradients.

6. **Aggregation**: Default uses mean pooling over modality tokens (no CLS token) for simpler architecture.

### 11.5 Architecture Summary

```
Inputs:
├── OHLCV Sequence (B, 96, 14)
├── TDA Features (B, 214)
└── Complexity (B, 6)

                    ┌───────────────────┐
OHLCV ──────────────│ OHLCVNBEATSEncoder │──── (B, 1024)
                    └───────────────────┘
                              │
                    ┌───────────────────┐
TDA ────────────────│    TDAEncoder     │──── (B, 256)
                    └───────────────────┘
                              │
                    ┌───────────────────┐
Complexity ─────────│ ComplexityEncoder │──── (B, 64)
                    └───────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │      CrossModalAttentionFusion        │
          ├───────────────────────────────────────┤
          │  Project to 256D each                 │
          │  Stack as 3 tokens (B, 3, 256)        │
          │  + Positional embeddings              │
          │  → 2-layer Transformer Encoder        │
          │  → Mean pooling → (B, 256)            │
          │  → Output projection                  │
          └───────────────────────────────────────┘
                              │
                        (B, 256)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌────────────┐               ┌────────────┐
       │Trigger Head│               │Max_Pct Head│
       │ 256→32→1   │               │  256→32→1  │
       └────────────┘               └────────────┘
```

### 11.6 Model Output Location

Training artifacts are saved to: `models/cross_modal_attention/`
- `best_model.pt` - Best model checkpoint
- `training_results.json` - Training metrics and history
