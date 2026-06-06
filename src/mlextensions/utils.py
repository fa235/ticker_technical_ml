"""Core numeric utilities for ML technical indicator analysis.

Provides helper functions for calculating normalized indicators across various tickers.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd

SeriesLike = Union[pd.Series, np.ndarray, Iterable[float], float, int]
EPSILON = 1e-10


def to_series(values: SeriesLike, *, name: str | None = None) -> pd.Series:
    """Return *values* as a pandas Series.

    Scalars are converted to a one-element Series. Existing Series preserve their
    original index unless a new name is supplied.
    """
    if isinstance(values, pd.Series):
        return values.rename(name) if name is not None else values.copy()
    if np.isscalar(values):
        return pd.Series([values], name=name, dtype="float64")
    return pd.Series(values, name=name, dtype="float64")


def align_like(reference: pd.Series, values: SeriesLike, *, name: str | None = None) -> pd.Series:
    """Convert values to Series and align/index it like *reference* when possible."""
    series = to_series(values, name=name)
    if len(series) == len(reference):
        series.index = reference.index
    return series


def safe_divide(numerator: SeriesLike, denominator: SeriesLike, *, fill_value: float = np.nan) -> pd.Series:
    """Vectorized division that avoids zero-denominator explosions."""
    numerator_s = to_series(numerator)
    denominator_s = align_like(numerator_s, denominator)
    denominator_s = denominator_s.where(denominator_s.abs() > EPSILON, np.nan)
    result = numerator_s / denominator_s
    if not np.isnan(fill_value):
        result = result.fillna(fill_value)
    return result


def nz(values: SeriesLike, replacement: float | pd.Series = 0.0) -> pd.Series:
    """Replace NaN values with a replacement value."""
    series = to_series(values)
    if isinstance(replacement, pd.Series):
        return series.where(series.notna(), replacement)
    return series.fillna(replacement)


def ema(src: SeriesLike, length: int) -> pd.Series:
    """Exponential moving average using pandas ewm."""
    if length <= 0:
        raise ValueError("length must be positive")
    series = to_series(src)
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def rma(src: SeriesLike, length: int) -> pd.Series:
    """Wilder's moving average, used by RSI/ATR/ADX."""
    if length <= 0:
        raise ValueError("length must be positive")
    series = to_series(src)
    return series.ewm(alpha=1 / length, adjust=False, min_periods=1).mean()


def sma(src: SeriesLike, length: int) -> pd.Series:
    """Simple moving average."""
    if length <= 0:
        raise ValueError("length must be positive")
    return to_series(src).rolling(length, min_periods=length).mean()


def rolling_sum(src: SeriesLike, length: int) -> pd.Series:
    """Rolling sum with length validation."""
    if length <= 0:
        raise ValueError("length must be positive")
    return to_series(src).rolling(length, min_periods=length).sum()


def true_range(high: SeriesLike, low: SeriesLike, close: SeriesLike) -> pd.Series:
    """True range used by ATR/ADX."""
    high_s = to_series(high, name="high")
    low_s = align_like(high_s, low, name="low")
    close_s = align_like(high_s, close, name="close")
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            high_s - low_s,
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: SeriesLike, low: SeriesLike, close: SeriesLike, length: int) -> pd.Series:
    """Average True Range using Wilder's RMA."""
    return rma(true_range(high, low, close), length)
