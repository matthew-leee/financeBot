"""
FinRobot daily report filter + Active Pivot hedge selection.

This module is read-only and defensive. It loads a daily sentiment report
(produced out-of-band by FinRobot/an LLM) and provides two things:

  1. is_trade_allowed(): does a symbol clear the daily sentiment threshold?
  2. select_hedge_asset(): when sentiment is weak, dynamically choose the inverse
     ETF most negatively correlated to the target (Dynamic Correlation Hedging).

Behavior change (Active Pivot): a weak/missing sentiment score no longer means
"do nothing". The execution loop consults is_trade_allowed() to decide whether
to buy the target directly or PIVOT into a hedge chosen by select_hedge_asset().
The selection itself still fails safe: if no correlation can be computed, it
returns None and the caller declines to trade.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config


def load_sentiment(path: str | None = None) -> dict:
    """Load the daily sentiment JSON. Returns an empty dict on any failure."""
    path = path or config.DAILY_SENTIMENT_PATH
    if not os.path.exists(path):
        print(f"[sentiment] No report at {path}; weak-sentiment pivot will apply.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            print("[sentiment] Report is not a JSON object; ignoring.")
            return {}
        return data
    except Exception as exc:  # noqa: BLE001 -- never crash the loop on a bad report
        print(f"[sentiment] Failed to parse {path}: {exc}")
        return {}


def get_score(report: dict, symbol: str) -> float | None:
    """
    Extract the numeric sentiment score for a symbol from a loaded report.

    Sentiment is keyed by the equity ticker (e.g. "SPY"). Crypto pairs like
    "BTC/USD" fall back to their base asset. Returns None if unavailable.
    """
    key = symbol.split("/")[0]
    entry = report.get(key) or report.get(symbol)
    if entry is None:
        return None

    # Accept either {"SPY": 7.2} or {"SPY": {"score": 7.2, ...}}.
    if isinstance(entry, dict):
        raw = entry.get("score")
    else:
        raw = entry

    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def is_trade_allowed(
    report: dict,
    symbol: str,
    min_score: float = config.SENTIMENT_MIN_SCORE,
) -> bool:
    """
    True if the symbol clears the daily sentiment threshold.

    Missing/invalid scores follow ``config.SENTIMENT_MISSING_IS_PASS``:
      * True (default) -> NEUTRAL-PASS: trade the target normally, so a
        partial daily report no longer forces pivot-heavy behavior.
      * False -> legacy semantics: treat as weak and let the caller pivot.

    An EXPLICIT score below min_score always blocks/pivots regardless of the
    flag -- only absence is neutral, never bad news.
    """
    score = get_score(report, symbol)
    if score is None:
        return bool(getattr(config, "SENTIMENT_MISSING_IS_PASS", False))
    return score >= min_score


# ---------------------------------------------------------------------------
# Dynamic Correlation Hedging
# ---------------------------------------------------------------------------

def daily_returns(bars: pd.DataFrame) -> pd.Series:
    """
    Trailing daily returns from a bar frame''s close.

    Resamples to calendar-day closes first so hourly and daily feeds align on a
    common index before correlation. Empty/short input -> empty series.
    """
    if bars is None or bars.empty or "close" not in bars.columns:
        return pd.Series(dtype="float64")
    close = bars["close"].copy()
    try:
        # Collapse intraday bars to one close per day (no-op for daily data).
        close = close.resample("1D").last().dropna()
    except (TypeError, ValueError):
        # Non-datetime index -> use as-is (already one point per period).
        pass
    return close.pct_change().dropna()


def select_hedge_asset(
    target_symbol: str,
    safe_list: list[str] | None = None,
    bar_fetcher=None,
    lookback_days: int | None = None,
) -> str | None:
    """
    Pick the inverse ETF most negatively correlated to the target.

    Fetches trailing daily returns for the target and each candidate in
    safe_list, computes the Pearson correlation on the overlapping dates, and
    returns the ETF with the STRONGEST (most negative) correlation.

    Returns None if no candidate yields a computable correlation -- the caller
    then declines the trade instead of hedging blindly.
    """
    safe_list = safe_list if safe_list is not None else config.INVERSE_SAFE_LIST
    lookback_days = lookback_days or config.HEDGE_CORR_LOOKBACK_DAYS

    if bar_fetcher is None:
        from src.data import fetch_bars as bar_fetcher  # lazy import, testable

    target_bars = bar_fetcher(target_symbol, lookback_days=lookback_days)
    target_ret = daily_returns(target_bars)
    if target_ret.empty:
        print(f"[hedge] No target returns for {target_symbol}; cannot hedge.")
        return None

    best_symbol: str | None = None
    best_corr = np.inf  # we minimize correlation (most negative wins)

    for etf in safe_list:
        if etf == target_symbol:
            continue
        etf_ret = daily_returns(bar_fetcher(etf, lookback_days=lookback_days))
        if etf_ret.empty:
            continue

        aligned = pd.concat([target_ret, etf_ret], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            continue

        # A zero-variance leg has undefined correlation; skip it cleanly instead
        # of letting numpy emit a divide-by-zero RuntimeWarning.
        if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
            continue

        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if corr is None or pd.isna(corr):
            continue

        if corr < best_corr:
            best_corr = corr
            best_symbol = etf

    if best_symbol is not None:
        print(f"[hedge] {target_symbol} -> {best_symbol} (corr={best_corr:.3f}).")
    else:
        print(f"[hedge] No hedge candidate found for {target_symbol}.")
    return best_symbol

