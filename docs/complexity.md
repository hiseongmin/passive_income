# Market Complexity Module (dev_ye)

## Overview
Measures **market complexity** — defined as "how difficult it is to predict the trend."

## Core Concept
- **Low complexity (clear trend)**: MAs separated, BB wide, strong support reactions, efficient price movement, volume confirms direction
- **High complexity**: MAs tangled, BB narrow, weak reactions, price moves a lot but goes nowhere, volume diverges from price

### Clear Trend vs Complex Market

| Aspect | Clear Trend | Complex Market |
|--------|-------------|----------------|
| Moving Averages | Separated (aligned) | Tangled, converged |
| Bollinger Bands | Wide | Narrow |
| Support Reaction | Strong bounce | Weak, unclear |
| Time to Result | Fast | Slow, no result |
| Position Sizing | Time-based OK | Price-based only |
| Volume | Confirms direction | Diverges or spikes without result |

## Indicators

### 1. MA Separation
- Calculate distance between MAs (20, 50, 100, 200)
- High separation = clear trend
- Convergence/crossing = high complexity

### 2. Bollinger Band Width
- `(Upper - Lower) / Price`
- Wide = low complexity
- Narrow = high complexity

### 3. Price Efficiency
- `|Net Movement| / Total Movement`
- Close to 1 = trending (low complexity)
- Close to 0 = choppy (high complexity)

### 4. Support Reaction Strength
- Bounce magnitude within N minutes after touching support
- Strong = low complexity
- Weak = high complexity

### 5. Directional Result per Time Unit
- Price displacement after N candles
- Large = low complexity
- Small despite time = high complexity

### 6. Volume-Price Alignment
- Measures if volume confirms price movement
- `correlation(price_change, volume)` over rolling window
- High positive correlation = volume confirms trend (low complexity)
- Low/negative correlation = volume diverges from price (high complexity)
- Additional: Volume spike with clear direction vs volume spike with no result

## Complexity Score Formula

```python
complexity = w1 * (1 - ma_separation_norm)
           + w2 * (1 - bb_width_norm)
           + w3 * (1 - price_efficiency)
           + w4 * (1 - support_reaction_norm)
           + w5 * (1 - directional_result_norm)
           + w6 * (1 - volume_price_alignment_norm)
```

Initial weights: equal (~0.167 each) → tune after visual validation

## Labeling Strategy

1. Calculate complexity score using formula above
2. Auto-extract "complex" vs "simple" periods
3. Sample and validate visually (top/bottom 10%)
4. Adjust formula if needed
5. Later: re-validate with actual strategy results

## Model

- **Framework**: PyTorch
- **Target**: Complexity as continuous value (regression)
- **Architecture**: LSTM (initial) → N-BEATS or Transformer (later)
- **Input**: Raw features + TDA signals (after dev_sa integration)

## Workflow

1. Collect 1-minute candle data from Binance
2. Implement each indicator individually
3. Visualize on chart for trader validation
4. Tune weights based on visual inspection
5. Build PyTorch dataset with sequential loading
6. Train LSTM model
7. Integrate with TDA signals from dev_sa
