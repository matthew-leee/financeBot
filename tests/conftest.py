"""Shared pytest setup for local imports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Dual-Horizon Engine test fixtures (Phase 0)
# ===========================================================================
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def utc_now() -> datetime:
    return datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def shuffled_panel() -> pd.DataFrame:
    """Multi-symbol panel with a 5-day forward label, rows shuffled.

    Used to prove that panel folds split by calendar time (not row order) and
    are deterministic regardless of input ordering.
    """
    rng = np.random.default_rng(7)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symbols = ["AAA", "BBB", "CCC"]
    horizon = 5
    rows = []
    n_days = 60
    for d in range(n_days):
        ts = start + timedelta(days=d)
        for sym in symbols:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "label_end_time": ts + timedelta(days=horizon),
                    "feat_a": float(rng.normal()),
                    "feat_b": float(rng.normal()),
                    "target": float(rng.normal()),
                }
            )
    panel = pd.DataFrame(rows)
    return panel.sample(frac=1.0, random_state=123).reset_index(drop=True)


@dataclass
class FakeBrokerPosition:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0


class MockBroker:
    """Deterministic, offline broker double. Never touches a real API."""

    def __init__(
        self,
        *,
        equity: float = 1000.0,
        cash: float = 1000.0,
        positions: list[FakeBrokerPosition] | None = None,
        fills: list | None = None,
        last_prices: dict | None = None,
    ) -> None:
        self.equity = equity
        self.cash = cash
        self._positions = positions or []
        self._fills = fills or []
        self._last_prices = last_prices or {}
        self.submitted: list = []
        self.cancel_all_called = 0

    def submit_order_intent(self, intent, client_order_id):
        self.submitted.append((intent, client_order_id))
        return f"broker-{client_order_id}"

    def list_recent_fills(self):
        return list(self._fills)

    def set_fills(self, fills: list) -> None:
        self._fills = list(fills)

    def get_all_positions(self):
        return list(self._positions)

    def set_positions(self, positions: list) -> None:
        self._positions = list(positions)

    def get_equity(self):
        return self.equity

    def get_cash(self):
        return self.cash

    def get_last_price(self, symbol):
        return self._last_prices.get(symbol, 100.0)

    def cancel_all_open_orders(self):
        self.cancel_all_called += 1


@pytest.fixture
def mock_broker() -> MockBroker:
    return MockBroker()


@pytest.fixture
def make_fill():
    """Factory building BrokerFill objects for fill-processing tests."""
    from src.portfolio_manager import BrokerFill

    def _make(
        *,
        fill_id: str,
        order_id: str = "o1",
        symbol: str = "AAPL",
        side: str = "buy",
        qty: float = 1.0,
        price: float = 100.0,
        fees: float = 0.0,
        at: datetime | None = None,
    ) -> "BrokerFill":
        return BrokerFill(
            fill_id=fill_id,
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_qty=qty,
            filled_price=price,
            filled_at=at or datetime(2026, 6, 1, tzinfo=timezone.utc),
            fees=fees,
            liquidity_flag=None,
        )

    return _make
