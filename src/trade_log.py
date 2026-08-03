"""
Trade log writer with FIFO realized-PnL tracking.

Quietly appends every executed trade to a CSV so the decoupled Streamlit
dashboard (read-only) and the backtester share one schema. SELL PnL is not
caller-provided: it is calculated from a persisted strict FIFO inventory queue.

This makes the live log auditable:
  BUY  -> appends a lot, records pnl=0
  SELL -> matches oldest lots first, records realized FIFO pnl

Writing is fully defensive: a logging failure must never interrupt the trading
loop after an order has already executed.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import config
from src.fifo import FIFOInventory

# Canonical column order shared by live logging, the dashboard, and backtests.
TRADE_LOG_COLUMNS: list[str] = [
    "timestamp",
    "ticker",
    "side",
    "price",
    "size",
    "pnl",
]


def _inventory_path(path: str | None = None) -> str:
    """Inventory path is injectable for tests, config-backed for live runs."""
    return path or config.INVENTORY_STATE_PATH


def _calculate_fifo_pnl(
    ticker: str,
    side: str,
    price: float,
    size: float,
    inventory_path: str | None = None,
) -> float:
    """
    Apply the trade to persisted FIFO inventory and return realized PnL.

    BUY rows always record zero realized PnL. SELL rows consume oldest lots and
    return the exact FIFO realized PnL. State is saved after every valid trade so
    restarts preserve cost basis.
    """
    inventory = FIFOInventory.load(_inventory_path(inventory_path))
    normalized_side = side.lower()

    if normalized_side == "buy":
        inventory.add_buy(ticker, size, price)
        realized_pnl = 0.0
    elif normalized_side == "sell":
        realized_pnl = inventory.add_sell(ticker, size, price)
    else:
        # Unknown sides should never happen; log zero and avoid mutating state.
        print(f"[trade_log] Unknown side '{side}' for {ticker}; FIFO not updated.")
        return 0.0

    inventory.save(_inventory_path(inventory_path))
    return round(realized_pnl, 6)


def append_trade(
    ticker: str,
    side: str,
    price: float,
    size: float,
    pnl: float | None = None,
    timestamp: str | None = None,
    path: str | None = None,
    inventory_path: str | None = None,
) -> float:
    """
    Append one trade row to the CSV, writing a header if the file is new.

    The `pnl` argument is kept for backward compatibility but intentionally
    ignored for BUY/SELL rows. Recorded PnL is always FIFO-derived here.
    Returns the realized PnL that was written (0 on logging failure).
    """
    path = path or config.TRADES_LOG_PATH
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    try:
        realized_pnl = _calculate_fifo_pnl(ticker, side, price, size, inventory_path)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=TRADE_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "side": side.lower(),
                    "price": round(float(price), 6),
                    "size": round(float(size), 6),
                    "pnl": round(float(realized_pnl), 6),
                }
            )
        return realized_pnl
    except Exception as exc:  # noqa: BLE001 -- logging must not kill the loop
        print(f"[trade_log] Failed to append trade for {ticker}: {exc}")
        return 0.0
