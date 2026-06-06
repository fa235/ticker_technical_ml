"""Signal filters for technical indicator analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .indicators import _adx_raw
from .utils import EPSILON, align_like, atr, ema, to_series


def filter_volatility(high, low, close, min_length: int = 1, max_length: int = 10, use_volatility_filter: bool = True) -> pd.Series:
    """Return True when short ATR is greater than longer ATR, or all True if disabled."""
    close_s = to_series(close)
    if not use_volatility_filter:
        return pd.Series(True, index=close_s.index, name="filter_volatility")
    recent_atr = atr(high, low, close_s, min_length)
    historical_atr = atr(high, low, close_s, max_length)
    return (recent_atr > historical_atr).rename("filter_volatility")


def filter_adx(high, low, close, length: int = 14, adx_threshold: int = 20, use_adx_filter: bool = True) -> pd.Series:
    """Return True when ADX is above threshold, or all True if disabled."""
    close_s = to_series(close)
    if not use_adx_filter:
        return pd.Series(True, index=close_s.index, name="filter_adx")
    adx = _adx_raw(high, low, close_s, length)
    return (adx > adx_threshold).rename("filter_adx")


def regime_filter(src, high, low, threshold: float = -0.1, use_regime_filter: bool = True) -> pd.Series:
    """Trend/range regime filter for market condition analysis.

    Identifies whether the market is in a trending or ranging regime.
    The high and low series must be explicitly provided.
    """
    src_s = to_series(src, name="src")
    high_s = align_like(src_s, high, name="high")
    low_s = align_like(src_s, low, name="low")

    if not use_regime_filter:
        return pd.Series(True, index=src_s.index, name="regime_filter")

    value1 = np.zeros(len(src_s), dtype="float64")
    value2 = np.zeros(len(src_s), dtype="float64")
    klmf = np.zeros(len(src_s), dtype="float64")

    src_values = src_s.to_numpy(dtype="float64")
    high_values = high_s.to_numpy(dtype="float64")
    low_values = low_s.to_numpy(dtype="float64")

    for i in range(len(src_s)):
        prev_src = src_values[i - 1] if i > 0 and not np.isnan(src_values[i - 1]) else src_values[i]
        price_change = 0.0 if np.isnan(src_values[i]) or np.isnan(prev_src) else src_values[i] - prev_src
        bar_range = 0.0 if np.isnan(high_values[i]) or np.isnan(low_values[i]) else high_values[i] - low_values[i]
        prev_value1 = value1[i - 1] if i > 0 else 0.0
        prev_value2 = value2[i - 1] if i > 0 else 0.0
        value1[i] = 0.2 * price_change + 0.8 * prev_value1
        value2[i] = 0.1 * bar_range + 0.8 * prev_value2
        omega = abs(value1[i] / value2[i]) if abs(value2[i]) > EPSILON else 0.0
        alpha = (-omega**2 + math.sqrt(omega**4 + 16 * omega**2)) / 8 if omega != 0 else 0.0
        prev_klmf = klmf[i - 1] if i > 0 else 0.0
        klmf[i] = np.nan if np.isnan(src_values[i]) else alpha * src_values[i] + (1 - alpha) * prev_klmf

    klmf_s = pd.Series(klmf, index=src_s.index)
    abs_curve_slope = (klmf_s - klmf_s.shift(1)).abs()
    avg_abs_curve_slope = ema(abs_curve_slope.fillna(0), 200)
    normalized_slope_decline = (abs_curve_slope - avg_abs_curve_slope) / avg_abs_curve_slope.where(avg_abs_curve_slope.abs() > EPSILON, np.nan)
    return (normalized_slope_decline >= threshold).fillna(False).rename("regime_filter")
