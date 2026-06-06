"""Normalized technical indicators used by the Lorentzian Classification script."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import normalize, rescale
from .utils import EPSILON, align_like, ema, rma, sma, to_series, true_range


def rsi(src, length: int = 14) -> pd.Series:
    """RSI using Wilder/RMA smoothing."""
    series = to_series(src)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain.fillna(0), length)
    avg_loss = rma(loss.fillna(0), length)
    rs = avg_gain / avg_loss.where(avg_loss.abs() > EPSILON, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.where(avg_loss.abs() > EPSILON, 100.0)
    value = value.where(avg_gain.abs() > EPSILON, 0.0)
    both_zero = (avg_gain.abs() <= EPSILON) & (avg_loss.abs() <= EPSILON)
    value = value.where(~both_zero, 50.0)
    return value.rename("rsi")


def cci(src, length: int = 20) -> pd.Series:
    """Commodity Channel Index (CCI) indicator."""
    series = to_series(src)
    basis = sma(series, length)
    mean_dev = series.rolling(length, min_periods=length).apply(
        lambda window: np.mean(np.abs(window - np.mean(window))), raw=True
    )
    return ((series - basis) / (0.015 * mean_dev.where(mean_dev.abs() > EPSILON, np.nan))).rename("cci")


def n_rsi(src, n1: int = 14, n2: int = 1) -> pd.Series:
    """Normalized RSI indicator."""
    return rescale(ema(rsi(src, n1), n2), 0, 100, 0, 1).rename("n_rsi")


def n_cci(src, n1: int = 20, n2: int = 1) -> pd.Series:
    """Normalized CCI indicator."""
    return normalize(ema(cci(src, n1), n2), 0, 1).rename("n_cci")


def n_wt(src, n1: int = 10, n2: int = 11) -> pd.Series:
    """Normalized WaveTrend Classic indicator."""
    series = to_series(src)
    ema1 = ema(series, n1)
    ema2 = ema((series - ema1).abs(), n1)
    ci = (series - ema1) / (0.015 * ema2.where(ema2.abs() > EPSILON, np.nan))
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    return normalize(wt1 - wt2, 0, 1).rename("n_wt")


def _wilder_sum(src: pd.Series, length: int) -> pd.Series:
    """Recursive smoother for ADX calculation."""
    out = np.zeros(len(src), dtype="float64")
    values = src.fillna(0).to_numpy(dtype="float64")
    prev = 0.0
    for i, value in enumerate(values):
        prev = prev - (prev / length) + value
        out[i] = prev
    return pd.Series(out, index=src.index)


def _adx_raw(high, low, close, length: int = 14) -> pd.Series:
    high_s = to_series(high, name="high")
    low_s = align_like(high_s, low, name="low")
    close_s = align_like(high_s, close, name="close")

    tr = true_range(high_s, low_s, close_s).fillna(0)
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    dm_plus = pd.Series(dm_plus, index=high_s.index)
    dm_minus = pd.Series(dm_minus, index=high_s.index)

    tr_smooth = _wilder_sum(tr, length)
    plus_smooth = _wilder_sum(dm_plus, length)
    minus_smooth = _wilder_sum(dm_minus, length)

    di_plus = plus_smooth / tr_smooth.where(tr_smooth.abs() > EPSILON, np.nan) * 100
    di_minus = minus_smooth / tr_smooth.where(tr_smooth.abs() > EPSILON, np.nan) * 100
    denom = (di_plus + di_minus).where((di_plus + di_minus).abs() > EPSILON, np.nan)
    dx = (di_plus - di_minus).abs() / denom * 100
    return rma(dx.fillna(0), length).rename("adx")


def n_adx(high, low, close, n1: int = 14) -> pd.Series:
    """Normalized ADX indicator."""
    return rescale(_adx_raw(high, low, close, n1), 0, 100, 0, 1).rename("n_adx")


def feature_series_from(feature_name: str, close, high=None, low=None, hlc3=None, param_a: int = 14, param_b: int = 1) -> pd.Series:
    """Convenience dispatcher matching the Lorentzian script's ``series_from`` helper."""
    name = feature_name.upper()
    if name == "RSI":
        return n_rsi(close, param_a, param_b)
    if name == "WT":
        if hlc3 is None:
            if high is None or low is None:
                raise ValueError("WT requires hlc3 or high/low/close inputs")
            close_s = to_series(close)
            hlc3 = (align_like(close_s, high) + align_like(close_s, low) + close_s) / 3
        return n_wt(hlc3, param_a, param_b)
    if name == "CCI":
        return n_cci(close, param_a, param_b)
    if name == "ADX":
        if high is None or low is None:
            raise ValueError("ADX requires high, low, and close")
        return n_adx(high, low, close, param_a)
    raise ValueError("feature_name must be one of: RSI, WT, CCI, ADX")
