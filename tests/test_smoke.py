import numpy as np
import pandas as pd

import mlextensions as ml


def sample_ohlc(n=300):
    rng = np.random.default_rng(7)
    close = pd.Series(100 + rng.normal(0, 1, n).cumsum())
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close.shift(1).fillna(close.iloc[0]) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_indicator_shapes():
    df = sample_ohlc()
    assert len(ml.n_rsi(df.close, 14, 1)) == len(df)
    assert len(ml.n_cci(df.close, 20, 1)) == len(df)
    assert len(ml.n_wt((df.high + df.low + df.close) / 3, 10, 11)) == len(df)
    assert len(ml.n_adx(df.high, df.low, df.close, 14)) == len(df)


def test_filters_are_boolean_series():
    df = sample_ohlc()
    out = ml.filter_volatility(df.high, df.low, df.close)
    assert out.dtype == bool
    out = ml.filter_adx(df.high, df.low, df.close, use_adx_filter=False)
    assert out.all()
    out = ml.regime_filter(df.close, df.high, df.low, use_regime_filter=False)
    assert out.all()


def test_backtest_returns_stats():
    df = sample_ohlc()
    start_long = df.close > df.close.rolling(20).mean()
    end_long = df.close < df.close.rolling(20).mean()
    false = pd.Series(False, index=df.index)
    stats = ml.backtest(df.high, df.low, df.open, start_long, end_long, false, false, false, 20, df.close)
    assert "total_trades" in stats.columns
    assert len(stats) == len(df)
