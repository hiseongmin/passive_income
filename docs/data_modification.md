# Data Modification Plan: Trigger and Max_Pct Generation

## Overview
Convert 5-minute Bitcoin OHLCV data to 15-minute intervals and generate `Trigger` and `Max_Pct` columns based on 2-hour price movement detection.

## Input Files (from `data/`)
1. `BTCUSDT_perp_etf_to_90d_ago.csv` (181,441 rows)
2. `BTCUSDT_perp_last_90d.csv` (25,921 rows)
3. `BTCUSDT_spot_etf_to_90d_ago.csv` (181,441 rows)
4. `BTCUSDT_spot_last_90d.csv` (25,920 rows)

## Output
- New folder: `data_flagged/`
- Same filenames with `_15m_flagged` suffix
- Documentation: `docs/data_modification.md`

---

## Step 1: Convert 5-min to 15-min Data

Merge every 3 consecutive 5-minute candles into 1 15-minute candle:

| Column | Aggregation Method |
|--------|-------------------|
| `open_time` | First timestamp of the 3 candles |
| `open` | First `open` value |
| `high` | Maximum `high` across 3 candles |
| `low` | Minimum `low` across 3 candles |
| `close` | Last `close` value |
| `volume` | Sum of all 3 `volume` values |
| `buy_volume` | Sum of all 3 `buy_volume` values |
| `sell_volume` | Sum of all 3 `sell_volume` values |
| `volume_delta` | Sum of all 3 `volume_delta` values |
| `cvd` | Last `cvd` value (cumulative) |
| `open_interest` | Last `open_interest` value (perp only) |

**Grouping Logic**: Group by `floor(row_index / 3)` or resample by 15-minute intervals using `open_time`.

---

## Step 2: Initialize New Columns

Add two new columns to the 15-min dataframe:
- `Trigger`: Boolean, default `False`
- `Max_Pct`: Float, default `0.0`

---

## Step 3: Detect 2% Price Movement Windows

For each position `i` in the data, look at the **next 8 time steps** (2-hour window):
- Window: rows `[i, i+1, i+2, ..., i+7]`
- Find `max_high` = maximum `high` value within the window
- Find `min_low` = minimum `low` value within the window
- Calculate: `price_diff_pct = ((max_high - min_low) / min_low) * 100`
- If `price_diff_pct >= 2.0`: Mark this as a "trigger window"

**Note**: This captures the maximum price swing within the 2-hour window, regardless of direction.

---

## Step 4: Set Trigger Flags

When a trigger window is detected starting at index `i`:
- Set `Trigger = True` for indices: `i-3`, `i-2`, `i-1` (3 time steps **before** window starts)
- Guard against negative indices (skip if `i < 3`)

```
Timeline visualization:
... [i-3] [i-2] [i-1] | [i] [i+1] [i+2] [i+3] [i+4] [i+5] [i+6] [i+7] ...
    ↑     ↑     ↑     |  ←─────────── 2-hour window ──────────────→
    Trigger=True      |  (8 steps where >= 2% price change detected)
```

---

## Step 5: Calculate Max_Pct Values

For each row where `Trigger = True`:
1. Identify which 2-hour window(s) this trigger relates to
2. Find the **maximum close price** within that 8-step window
3. Calculate: `Max_Pct = ((max_close - current_close) / current_close) * 100`

**Note**: A trigger row may relate to multiple overlapping windows. Use the nearest upcoming window.

---

## Implementation Details

### File Structure
```
passive_income/
├── data/                          # Input (unchanged)
├── data_flagged/                  # Output (new)
│   ├── BTCUSDT_perp_etf_to_90d_ago_15m_flagged.csv
│   ├── BTCUSDT_perp_last_90d_15m_flagged.csv
│   ├── BTCUSDT_spot_etf_to_90d_ago_15m_flagged.csv
│   └── BTCUSDT_spot_last_90d_15m_flagged.csv
└── docs/
    └── data_modification.md       # Documentation
```

### Python Script Structure
```python
# Main script: process_data.py (or similar)
1. Load CSV with pandas
2. Resample 5min -> 15min using groupby or resample
3. Initialize Trigger=False, Max_Pct=0.0
4. Loop through data to detect 2% windows
5. Set Trigger flags for 3 steps before each window
6. Calculate Max_Pct for each Trigger=True row
7. Save to data_flagged/
```

### Edge Cases to Handle
- First 3 rows cannot be triggers (no preceding rows)
- Last 7 rows cannot start a window (insufficient future data)
- Overlapping trigger windows: ensure all relevant triggers are marked
- Missing data handling: skip incomplete 15-min groups

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create `data_flagged/` and `docs/` directories |
| 2 | Load each 5-min CSV file |
| 3 | Resample to 15-min candles |
| 4 | Scan for 2% price movements (max_high vs min_low in window) |
| 5 | Set Trigger=True for 3 rows before each detected window |
| 6 | Calculate Max_Pct = % diff between trigger price and window max |
| 7 | Save output CSV files |
| 8 | Generate documentation in `docs/data_modification.md` |
