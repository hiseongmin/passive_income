# Project: Passive Income via Bitcoin Trading

## Objectives
1. **Signal Generation**: Apply Topological Data Analysis (TDA) to identify optimal long/short trigger points in Bitcoin price movements
2. **Strategy Development**: Construct an automated trading strategy combining TDA signals with portfolio optimization techniques
3. **Validation**: Conduct comprehensive backtesting with realistic trading fees and slippage, visualize performance metrics

## Scope
### In Scope
- Bitcoin spot trading only (initially)
- TDA-based feature engineering for time series prediction
- Risk-adjusted portfolio construction
- Performance visualization and reporting

### Out of Scope (for now)
- Altcoin trading
- Derivatives/leverage products
- High-frequency trading strategies
- Live deployment (backtest phase first)

## Module Responsibilities

### TDA Signal Module (dev_sa)
See: `docs/tda_signal.md`

### Market Complexity Module (dev_ye)
See: `docs/complexity.md`

## Key References

### TDA in Financial Markets
1. **Stock Index Prediction with TDA**
   `docs/enhance.pdf`
   Focus: Practical applications to equity indices

2. **Time Series Classification via TDA**
   `docs/12_228.pdf`
   Focus: Classification methodologies applicable to trading signals

3. **TDA for Chronological Data**
   `docs/paper15.pdf`
   Focus: Temporal analysis techniques

4. **Extreme Event Detection**
   `docs/2405.16052v1.pdf`
   Focus: Identifying market crashes/rallies using topology

5. **Financial Time Series Forecasting Enhancement**
   `docs/enhance2.pdf`
   Focus: Improving forecast accuracy with TDA features

## Technical Stack
- **TDA Libraries**: giotto-tda, ripser, persim
- **ML Framework**: PyTorch
- **Data Analysis**: pandas, numpy
- **Backtesting**: backtrader, vectorbt
- **Visualization**: matplotlib, plotly
- **Data Source**: Binance API

## Data Requirements
- Source: Binance API (BTC/USDT)
- Timeframe: 1-minute candles
- Period: 2024.01.11 (BTC ETF launch) ~ present

### Data Split
| Period | Usage |
|--------|-------|
| ETF launch ~ 180 days ago | Training |
| 180 days ago ~ 90 days ago | Validation |
| 90 days ago ~ present | Test |

**Critical**: Walk-forward only — no future data leakage

## Success Metrics
- Sharpe Ratio > 1.5
- Maximum Drawdown < 20%
- Win Rate > 55%
- Positive returns after fees over 2+ year backtest period

## Next Steps
1. Literature review of listed papers
2. Collect 1-minute candle data from Binance
3. Proof-of-concept: TDA feature extraction on sample Bitcoin data
4. Implement complexity indicators and validate visually
5. Baseline strategy implementation
6. Integrate TDA signals + complexity module
7. Iterative refinement based on backtest results

---

# Development Guidelines

## Package Installation
Always use uv install

## Python Environment
Always use conda environment named passive_income
