# ticker_technical_ml

A Python/pandas helper library for technical analysis and machine learning features on trading data. Provides normalized RSI, CCI, WaveTrend, ADX, filters, color helpers, smoothing transforms, and a simple calibration backtest helper.

## Install locally

### Setup virtual environment (recommended)

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install the library

From this folder:

```bash
pip install -e .
```

Or from the zip after extracting it:

```bash
cd ticker_technical_ml
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Basic usage

```python
import pandas as pd
import ticker_technical_ml as ttm

# df must have columns: open, high, low, close
# Example:
# df = pd.read_csv("ohlc.csv", parse_dates=["date"], index_col="date")

rsi_feature = ttm.n_rsi(df["close"], 14, 1)
cci_feature = ttm.n_cci(df["close"], 20, 1)
wt_feature = ttm.n_wt((df["high"] + df["low"] + df["close"]) / 3, 10, 11)
adx_feature = ttm.n_adx(df["high"], df["low"], df["close"], 20)

vol_ok = ttm.filter_volatility(df["high"], df["low"], df["close"], 1, 10, True)
regime_ok = ttm.regime_filter(df["close"], df["high"], df["low"], threshold=-0.1, use_regime_filter=True)
adx_ok = ttm.filter_adx(df["high"], df["low"], df["close"], 14, 20, True)

all_filters_ok = vol_ok & regime_ok & adx_ok
```


## Backtest helper

```python
stats = ttm.backtest(
    high=df["high"],
    low=df["low"],
    open_=df["open"],
    start_long_trade=start_long,
    end_long_trade=end_long,
    start_short_trade=start_short,
    end_short_trade=end_short,
    is_early_signal_flip=early_flip,
    max_bars_back_index=200,
    src=df["close"],
    use_worst_case=False,
)

print(stats.tail())
```

## Module Reference

### `indicators`
Provides normalized technical indicators for feature engineering:
- `n_rsi(src, n1, n2)` - Normalized Relative Strength Index (0-1 range)
- `n_cci(src, n1, n2)` - Normalized Commodity Channel Index (0-1 range)
- `n_wt(src, n1, n2)` - Normalized WaveTrend Classic oscillator (0-1 range)
- `n_adx(high, low, close, n1)` - Normalized Average Directional Index (0-1 range)

All return normalized values between 0 and 1, making them suitable for machine learning models.

### `filters`
Market condition filters to improve signal quality:
- `filter_volatility(high, low, close, ...)` - Filters out low volatility periods
- `filter_adx(high, low, close, ...)` - Trend strength filter using ADX
- `regime_filter(src, high, low, ...)` - Identifies trending vs. ranging markets

Use these to validate signals before trading, reducing false positives in sideways markets.

### `transforms`
Signal transformation and smoothing utilities:
- `normalize_deriv(src, quadratic_mean_length)` - Normalized rate of change
- `dual_pole_filter(src, lookback)` - Dual-pole IIR smoothing filter
- `normalize(src, new_min, new_max)` - Min-max normalization using expanding range
- `rescale(src, old_min, old_max, new_min, new_max)` - Rescale between ranges
- `tanh(src)` - Sigmoid-like compression to roughly [-1, 1]

### `utils`
Core numerical utilities for calculations:
- `sma(src, length)` - Simple moving average
- `ema(src, length)` - Exponential moving average
- `rma(src, length)` - Wilder's moving average (used by RSI/ADX)
- `rsi(src, length)` - Relative Strength Index
- `atr(high, low, close, length)` - Average True Range
- `true_range(high, low, close)` - True range for volatility calculations
- `nz(values, replacement)` - Replace NaN values
- `rolling_sum(src, length)` - Rolling sum over a period

### `colors`
Helper functions for visualization and signal strength:
- `color_green(prediction)` - Returns green color hex codes based on bullish strength (0-10 scale)
- `color_red(prediction)` - Returns red color hex codes based on bearish strength (0-10 scale)

Useful for charting library integrations.

### `backtest`
Simple calibration backtester for strategy evaluation:
- `backtest(high, low, open_, start_long, end_long, start_short, end_short, is_early_signal_flip, max_bars_back_index, src)` - Runs a basic backtest

Returns a DataFrame with trade statistics including entry/exit prices, P&L, and win/loss ratios. This is a simple simulator for parameter tuning, not a production backtester.

## Notes and limitations

- This is for research purpose. Use it with caution if trading.
- The backtest helper does not account for commissions, slippage, or realistic market conditions.
- Normalized indicators are designed to feed machine learning models; scale appropriately for your use case.

