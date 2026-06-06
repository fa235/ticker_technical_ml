"""Simple calibration backtest helper for strategy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import align_like, to_series


def backtest(
    high,
    low,
    open_,
    start_long_trade,
    end_long_trade,
    start_short_trade,
    end_short_trade,
    is_early_signal_flip,
    max_bars_back_index: int,
    src,
    use_worst_case: bool = False,
) -> pd.DataFrame:
    """Run a simple cumulative win/loss calibration backtest.

    A basic backtester for strategy evaluation. Does not account for
    commissions, slippage, partial fills, margin, or realistic execution.
    """
    high_s = to_series(high, name="high")
    low_s = align_like(high_s, low, name="low")
    open_s = align_like(high_s, open_, name="open")
    src_s = align_like(high_s, src, name="src")

    start_long = align_like(high_s, start_long_trade).fillna(False).astype(bool)
    end_long = align_like(high_s, end_long_trade).fillna(False).astype(bool)
    start_short = align_like(high_s, start_short_trade).fillna(False).astype(bool)
    end_short = align_like(high_s, end_short_trade).fillna(False).astype(bool)
    early_flip = align_like(high_s, is_early_signal_flip).fillna(False).astype(bool)

    market_price = src_s if use_worst_case else (high_s + low_s + open_s + open_s) / 4

    start_long_price = market_price.iloc[0] if len(market_price) else np.nan
    start_short_price = market_price.iloc[0] if len(market_price) else np.nan

    wins = np.zeros(len(high_s), dtype="int64")
    losses = np.zeros(len(high_s), dtype="int64")
    trades = np.zeros(len(high_s), dtype="int64")
    early = np.zeros(len(high_s), dtype="int64")
    long_profit = np.zeros(len(high_s), dtype="float64")
    short_profit = np.zeros(len(high_s), dtype="float64")

    prices = market_price.to_numpy(dtype="float64")
    for i in range(len(high_s)):
        if i <= max_bars_back_index:
            continue

        if start_long.iloc[i]:
            start_short_price = 0.0
            early[i] = 1 if early_flip.iloc[i] else 0
            start_long_price = prices[i]
            trades[i] = 1

        if end_long.iloc[i]:
            delta = prices[i] - start_long_price
            wins[i] = 1 if delta > 0 else 0
            losses[i] = 1 if delta < 0 else 0
            long_profit[i] = delta

        if start_short.iloc[i]:
            start_long_price = 0.0
            start_short_price = prices[i]
            trades[i] = 1

        if end_short.iloc[i]:
            early[i] = 1 if early_flip.iloc[i] else early[i]
            delta = start_short_price - prices[i]
            wins[i] = 1 if delta > 0 else wins[i]
            losses[i] = 1 if delta < 0 else losses[i]
            short_profit[i] = delta

    out = pd.DataFrame(index=high_s.index)
    out["long_profit"] = pd.Series(long_profit, index=high_s.index).cumsum()
    out["short_profit"] = pd.Series(short_profit, index=high_s.index).cumsum()
    out["total_profit"] = out["long_profit"] + out["short_profit"]
    out["total_early_signal_flips"] = pd.Series(early, index=high_s.index).cumsum()
    out["total_wins"] = pd.Series(wins, index=high_s.index).cumsum()
    out["total_losses"] = pd.Series(losses, index=high_s.index).cumsum()
    out["total_trades"] = pd.Series(wins + losses, index=high_s.index).cumsum()
    out["win_loss_ratio"] = out["total_wins"] / out["total_trades"].replace(0, np.nan)
    out["win_rate"] = out["total_wins"] / (out["total_wins"] + out["total_losses"]).replace(0, np.nan)
    return out
