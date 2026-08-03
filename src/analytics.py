"""
Shared performance analytics.

Pure, dependency-light (pandas/numpy only) helpers used by BOTH the read-only
Streamlit dashboard and the backtester, so the two can never disagree on how
PnL, win rate, drawdown, or Sharpe are computed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import config
from src.fifo import FIFOInventory
from src.trade_log import TRADE_LOG_COLUMNS

# Column order for the dashboard's Open Positions table.
INVENTORY_COLUMNS: list[str] = [
    "ticker",
    "position_type",
    "open_qty",
    "avg_price",
    "last_price",
    "unrealized_pnl",
]


def classify_position(ticker: str) -> str:
    """
    Label a held ticker as a hedge vs. a direct hold.

    A position is a "Hedge" if its base asset is one of the inverse ETFs in
    config.INVERSE_SAFE_LIST (these are what the Active Pivot buys); everything
    else is a "Direct Hold".
    """
    base = str(ticker).split("/")[0]
    return "Hedge" if base in config.INVERSE_SAFE_LIST else "Direct Hold"


def load_trades(path: str) -> pd.DataFrame:
    """
    Load a trades CSV into a normalized DataFrame.

    Always returns a frame with TRADE_LOG_COLUMNS, even when the file is missing,
    empty, or header-only. This is what lets the dashboard render safely with no
    data instead of crashing.
    """
    empty = pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    try:
        if not path:
            return empty
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return empty
    except Exception:  # noqa: BLE001 -- a corrupt log should not crash the viewer
        return empty

    if df.empty:
        return empty

    # Guarantee all expected columns exist and are ordered consistently.
    for col in TRADE_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[TRADE_LOG_COLUMNS]

    for col in ("price", "size", "pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with a cumulative PnL column (empty-safe)."""
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "pnl", "cumulative_pnl"])
    out = trades.copy()
    out["cumulative_pnl"] = out["pnl"].cumsum()
    return out[["timestamp", "pnl", "cumulative_pnl"]]


def compute_metrics(trades: pd.DataFrame) -> dict:
    """
    Summary metrics from a trades frame. Safe on empty input (returns zeros).

    Win rate counts only PnL-bearing (closing) trades: entries logged with
    pnl == 0 are neutral and excluded from the win/loss denominator.
    """
    total_trades = int(len(trades))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    pnl = trades["pnl"].astype(float)
    realized = pnl[pnl != 0.0]

    wins = int((realized > 0).sum())
    win_rate = float(wins / len(realized)) if len(realized) > 0 else 0.0

    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    # Sharpe on per-trade PnL (not annualized): mean / std. Zero if no variance.
    if len(realized) > 1 and realized.std(ddof=1) > 0:
        sharpe = float(realized.mean() / realized.std(ddof=1))
    else:
        sharpe = 0.0

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(float(pnl.sum()), 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe": round(sharpe, 4),
    }


def realized_pnl_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Split realized PnL by position type (Hedge vs Direct Hold).

    Classifies every trade row by its ticker (via classify_position) and
    aggregates realized PnL, trade count, and win rate per bucket. Only
    PnL-bearing rows (closing SELLs, pnl != 0) count toward win rate; BUY
    entries with pnl == 0 are neutral. Empty-safe.

    Returns a frame with columns:
      position_type, realized_pnl, trades, wins, win_rate
    """
    columns = ["position_type", "realized_pnl", "trades", "wins", "win_rate"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=columns)

    df = trades.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    df["position_type"] = df["ticker"].map(classify_position)

    rows: list[dict] = []
    for ptype in ("Direct Hold", "Hedge"):
        subset = df[df["position_type"] == ptype]
        if subset.empty:
            continue
        realized = subset["pnl"][subset["pnl"] != 0.0]
        wins = int((realized > 0).sum())
        win_rate = float(wins / len(realized)) if len(realized) > 0 else 0.0
        rows.append(
            {
                "position_type": ptype,
                "realized_pnl": round(float(subset["pnl"].sum()), 6),
                "trades": int(len(subset)),
                "wins": wins,
                "win_rate": round(win_rate, 4),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def load_inventory(path: str | None = None) -> FIFOInventory:
    """Load the persisted FIFO inventory save-state (empty/flat on any failure)."""
    return FIFOInventory.load(path or config.INVENTORY_STATE_PATH)


def open_inventory_summary(
    inventory: FIFOInventory,
    price_lookup: Callable[[str], float | None] | None = None,
) -> pd.DataFrame:
    """
    Build the Open Positions table with unrealized PnL.

    price_lookup(ticker) -> current price (or None). When a price is unavailable
    we fall back to the average entry price, which yields an unrealized PnL of 0
    rather than a crash or a misleading number.

    Unrealized PnL per position = open_qty * (last_price - avg_price).
    """
    positions = inventory.positions()
    if not positions:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    rows: list[dict] = []
    for ticker, pos in positions.items():
        qty = float(pos["open_qty"])
        avg_price = float(pos["avg_price"])

        last_price = None
        if price_lookup is not None:
            try:
                last_price = price_lookup(ticker)
            except Exception:  # noqa: BLE001 -- a bad price feed must not crash the view
                last_price = None
        if last_price is None or (isinstance(last_price, float) and np.isnan(last_price)):
            last_price = avg_price  # fallback -> unrealized 0

        last_price = float(last_price)
        unrealized = qty * (last_price - avg_price)
        rows.append(
            {
                "ticker": ticker,
                "position_type": classify_position(ticker),
                "open_qty": round(qty, 6),
                "avg_price": round(avg_price, 6),
                "last_price": round(last_price, 6),
                "unrealized_pnl": round(unrealized, 6),
            }
        )

    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


