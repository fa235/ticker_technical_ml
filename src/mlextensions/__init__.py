"""ML extensions to analyze technical indicators for various tickers.
"""

from .backtest import backtest
from .colors import color_green, color_red, get_color_shades, get_prediction_color
from .filters import filter_adx, filter_volatility, regime_filter
from .indicators import cci, feature_series_from, n_adx, n_cci, n_rsi, n_wt, rsi
from .transforms import dual_pole_filter, normalize, normalize_deriv, rescale, tanh, tanh_transform
from .utils import atr, ema, rma, sma, true_range

# Pine-style aliases
normalizeDeriv = normalize_deriv
dualPoleFilter = dual_pole_filter
tanhTransform = tanh_transform
getColorShades = get_color_shades
getPredictionColor = get_prediction_color

__all__ = [
    "atr",
    "backtest",
    "cci",
    "color_green",
    "color_red",
    "dual_pole_filter",
    "dualPoleFilter",
    "ema",
    "feature_series_from",
    "filter_adx",
    "filter_volatility",
    "get_color_shades",
    "get_prediction_color",
    "getColorShades",
    "getPredictionColor",
    "n_adx",
    "n_cci",
    "n_rsi",
    "n_wt",
    "normalize",
    "normalize_deriv",
    "normalizeDeriv",
    "rescale",
    "rma",
    "rsi",
    "sma",
    "tanh",
    "tanh_transform",
    "tanhTransform",
    "true_range",
    "regime_filter",
]

__version__ = "0.1.0"
