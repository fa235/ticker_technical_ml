"""Normalization and smoothing transforms for ML-style technical features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import EPSILON, ema, rolling_sum, to_series


def normalize_deriv(src, quadratic_mean_length: int) -> pd.Series:
    """Return a normalized two-bar derivative."""
    series = to_series(src)
    deriv = series - series.shift(2)
    quadratic_mean = np.sqrt(rolling_sum(deriv.pow(2), quadratic_mean_length) / quadratic_mean_length)
    return deriv / quadratic_mean.where(quadratic_mean.abs() > EPSILON, np.nan)


def normalize(src, new_min: float = 0.0, new_max: float = 1.0) -> pd.Series:
    """Rescale an unbounded series to ``[new_min, new_max]`` using expanding extrema."""
    series = to_series(src)
    historic_min = series.expanding(min_periods=1).min()
    historic_max = series.expanding(min_periods=1).max()
    denom = (historic_max - historic_min).where((historic_max - historic_min).abs() > EPSILON, EPSILON)
    return new_min + (new_max - new_min) * (series - historic_min) / denom


def rescale(src, old_min: float, old_max: float, new_min: float, new_max: float) -> pd.Series:
    """Rescale a bounded series from one fixed range to another."""
    series = to_series(src)
    denom = max(abs(old_max - old_min), EPSILON)
    return new_min + (new_max - new_min) * (series - old_min) / denom


def tanh(src) -> pd.Series:
    """Sigmoid-like hyperbolic tangent transform; compresses to roughly [-1, 1]."""
    series = to_series(src)
    return -1 + 2 / (1 + np.exp(-2 * series))


def dual_pole_filter(src, lookback: int) -> pd.Series:
    """Return a dual-pole smoothed signal."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    series = to_series(src)
    omega = -99 * math.pi / (70 * lookback)
    alpha = math.exp(omega)
    beta = -(alpha**2)
    gamma = math.cos(omega) * 2 * alpha
    delta = 1 - gamma - beta

    prev_src = series.shift(1).fillna(series)
    sliding_avg = 0.5 * (series + prev_src)

    out = np.full(len(series), np.nan, dtype="float64")
    for i, value in enumerate(sliding_avg.to_numpy(dtype="float64")):
        prev1 = 0.0 if i < 1 or np.isnan(out[i - 1]) else out[i - 1]
        prev2 = 0.0 if i < 2 or np.isnan(out[i - 2]) else out[i - 2]
        out[i] = np.nan if np.isnan(value) else (delta * value) + gamma * prev1 + beta * prev2
    return pd.Series(out, index=series.index, name="dual_pole_filter")


def tanh_transform(src, smoothing_frequency: int, quadratic_mean_length: int) -> pd.Series:
    """Normalize derivative -> tanh -> dual-pole smoothing."""
    return dual_pole_filter(tanh(normalize_deriv(src, quadratic_mean_length)), smoothing_frequency)
