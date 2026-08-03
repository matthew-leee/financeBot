"""
FIFO (First-In, First-Out) realized-PnL inventory.

A SELL is matched against the OLDEST open lots first. Realized PnL is the sum
over matched lots of:  matched_qty * (sell_price - lot_buy_price).

Design notes:
  * Per-ticker queues -- inventory of one symbol never crosses into another.
  * Long-only: sells are matched only against existing long lots. If a sell
    exceeds inventory (should not happen with our guardrails), the surplus is
    ignored for PnL (we never fabricate a short lot).
  * Optional JSON persistence so the LIVE loop keeps a correct cost basis across
    process restarts. The in-memory queue alone is enough for the backtester.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass

# Float tolerance for treating a residual quantity as "flat".
_EPS = 1e-9


@dataclass
class Lot:
    qty: float
    price: float


class FIFOInventory:
    """Per-ticker FIFO lot queues with strict realized-PnL matching."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[Lot]] = {}

    # -- mutation ------------------------------------------------------------

    def add_buy(self, ticker: str, qty: float, price: float) -> None:
        """Push a new long lot onto the back of the ticker''s queue."""
        if qty <= _EPS:
            return
        self._queues.setdefault(ticker, deque()).append(Lot(float(qty), float(price)))

    def add_sell(self, ticker: str, qty: float, price: float) -> float:
        """
        Match a sell against the oldest lots and return realized PnL.

        Partial fills consume part of the front lot (leaving the remainder in
        place); full exits pop lots until qty is satisfied or inventory is empty.
        """
        remaining = float(qty)
        if remaining <= _EPS:
            return 0.0

        queue = self._queues.get(ticker)
        if not queue:
            # Selling with no inventory -> no realized PnL (never invent a short).
            return 0.0

        realized = 0.0
        while remaining > _EPS and queue:
            lot = queue[0]
            matched = min(remaining, lot.qty)
            realized += matched * (price - lot.price)
            lot.qty -= matched
            remaining -= matched
            if lot.qty <= _EPS:
                queue.popleft()  # lot fully consumed

        return round(realized, 6)

    # -- introspection -------------------------------------------------------

    def open_qty(self, ticker: str) -> float:
        """Total open quantity currently held for a ticker."""
        queue = self._queues.get(ticker)
        if not queue:
            return 0.0
        return round(sum(lot.qty for lot in queue), 6)

    def average_price(self, ticker: str) -> float:
        """Quantity-weighted average entry price of the open lots (0 if flat)."""
        queue = self._queues.get(ticker)
        if not queue:
            return 0.0
        total_qty = sum(lot.qty for lot in queue)
        if total_qty <= _EPS:
            return 0.0
        weighted = sum(lot.qty * lot.price for lot in queue)
        return round(weighted / total_qty, 6)

    def positions(self) -> dict[str, dict[str, float]]:
        """
        Snapshot of all non-empty positions.

        Returns {ticker: {"open_qty": q, "avg_price": p}} -- exactly what the
        dashboard needs to render open inventory + cost basis.
        """
        out: dict[str, dict[str, float]] = {}
        for ticker in self._queues:
            qty = self.open_qty(ticker)
            if qty > _EPS:
                out[ticker] = {"open_qty": qty, "avg_price": self.average_price(ticker)}
        return out

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            ticker: [[lot.qty, lot.price] for lot in queue]
            for ticker, queue in self._queues.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FIFOInventory":
        inv = cls()
        for ticker, lots in (data or {}).items():
            inv._queues[ticker] = deque(Lot(float(q), float(p)) for q, p in lots)
        return inv

    def save(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh)
        except Exception as exc:  # noqa: BLE001 -- persistence must not kill the loop
            print(f"[fifo] Failed to save inventory: {exc}")

    @classmethod
    def load(cls, path: str) -> "FIFOInventory":
        if not path or not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except Exception as exc:  # noqa: BLE001 -- corrupt state -> start flat
            print(f"[fifo] Failed to load inventory ({exc}); starting flat.")
            return cls()

