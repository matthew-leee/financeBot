"""Cash-account guards: T+1 settled-funds gate + dual-executor turnover dampener.

All broker doubles are local and offline; nothing touches Alpaca.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

import config
from src.portfolio_manager import PortfolioManager
from src.tactical_executor import (
    OrderIntent,
    TacticalExecutor,
    apply_turnover_dampening,
)

UTC = timezone.utc


def _intent(
    *,
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float = 2.0,
    price: float = 100.0,
    reduce_only: bool = False,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        quantity=qty,
        order_style="limit" if price else "market",
        limit_price=price,
        reduce_only=reduce_only,
        target_weight_after=0.1,
        reason="test",
    )


class WithdrawableBroker:
    """Offline double exposing explicit settled-cash truth."""

    def __init__(self, withdrawable: float | None = 500.0) -> None:
        self.withdrawable = withdrawable
        self.submitted: list = []
        self._fills: list = []

    def get_withdrawable_cash(self):
        return self.withdrawable

    def submit_order_intent(self, intent, client_order_id):
        self.submitted.append((intent, client_order_id))
        return f"broker-{client_order_id}"

    def set_fills(self, fills) -> None:
        self._fills = list(fills)

    def list_recent_fills(self):
        return list(self._fills)

    def get_all_positions(self):
        return []

    def get_equity(self):
        return 1000.0

    def get_last_price(self, symbol):
        return 100.0


class CashOnlyBroker(WithdrawableBroker):
    """No withdrawable method -- PM must fall back to raw cash."""

    def __init__(self, cash: float) -> None:
        super().__init__(withdrawable=None)
        self.cash = cash
        del self.withdrawable

    def get_withdrawable_cash(self):  # removed from this double
        raise AttributeError

    def get_cash(self):
        return self.cash


class BlindBroker(WithdrawableBroker):
    """No cash information at all -- buys must fail closed."""

    def __init__(self) -> None:
        super().__init__(withdrawable=None)
        del self.withdrawable

    def get_withdrawable_cash(self):
        raise AttributeError

    def get_cash(self):
        raise AttributeError


def _pm(tmp_path, broker) -> PortfolioManager:
    return PortfolioManager(
        broker=broker,
        state_path=str(tmp_path / "ps.json"),
        orders_log_path=str(tmp_path / "orders_log.csv"),
        fills_log_path=str(tmp_path / "fills_log.csv"),
    )


def _blocked_rows(orders_log) -> list[dict]:
    with open(orders_log, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["status"] == "blocked_settled_cash"]


# ---------------------------------------------------------------------------
# Settled-funds gate (PortfolioManager)
# ---------------------------------------------------------------------------


def test_buy_within_settled_cash_submits_and_records_commitment(tmp_path) -> None:
    broker = WithdrawableBroker(withdrawable=500.0)
    pm = _pm(tmp_path, broker)
    oid = pm.submit_order_intent(_intent(qty=2.0, price=100.0))  # $200 notional
    assert oid is not None
    assert len(broker.submitted) == 1
    # Commitment mirrored under both ids.
    assert len(pm._pending_buy_notional) == 2


def test_buy_exceeding_settled_cash_is_blocked_with_audit_row(tmp_path) -> None:
    broker = WithdrawableBroker(withdrawable=150.0)
    log = tmp_path / "orders_log.csv"
    pm = PortfolioManager(
        broker=broker,
        state_path=str(tmp_path / "ps.json"),
        orders_log_path=str(log),
        fills_log_path=str(tmp_path / "fills_log.csv"),
    )
    oid = pm.submit_order_intent(_intent(qty=2.0, price=100.0))  # $200 > $150
    assert oid is None
    assert broker.submitted == []
    rows = _blocked_rows(log)
    assert len(rows) == 1 and rows[0]["symbol"] == "AAPL"


def test_unknown_cash_fails_closed_for_buys(tmp_path) -> None:
    pm = _pm(tmp_path, BlindBroker())
    assert pm.submit_order_intent(_intent()) is None


def test_fallback_to_raw_cash_when_no_withdrawable_method(tmp_path) -> None:
    broker = CashOnlyBroker(cash=300.0)
    pm = _pm(tmp_path, broker)
    assert pm.submit_order_intent(_intent(qty=2.0, price=100.0)) is not None  # 200 <= 300
    assert pm.submit_order_intent(_intent(qty=4.0, price=100.0)) is None  # 400 > 300 - 200


def test_sells_are_exempt_from_the_gate(tmp_path) -> None:
    from tests.conftest import FakeBrokerPosition

    broker = BlindBroker()
    broker.get_all_positions = lambda: [
        FakeBrokerPosition(symbol="AAPL", qty=1.0, market_value=100.0)
    ]
    pm = _pm(tmp_path, broker)
    intent = _intent(side="sell", qty=1.0, price=100.0, reduce_only=True)
    assert pm.submit_order_intent(intent) is not None


def test_fill_releases_pending_commitment_so_next_buy_fits(tmp_path, make_fill) -> None:
    broker = WithdrawableBroker(withdrawable=300.0)
    pm = _pm(tmp_path, broker)

    first_id = pm.submit_order_intent(_intent(qty=2.0, price=100.0))  # commits 200
    assert first_id == "broker-" + broker.submitted[-1][1]
    assert pm.submit_order_intent(_intent(qty=2.0, price=100.0)) is None  # only 100 free

    # The buy fills -> commitment must release (both mirrored ids).
    broker.set_fills([make_fill(fill_id="f1", order_id=first_id, side="buy", qty=2.0)])
    pm.poll_and_apply_fills()
    assert not pm._pending_buy_notional

    assert pm.submit_order_intent(_intent(qty=2.0, price=100.0)) is not None


def test_gate_can_be_disabled_via_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ENFORCE_SETTLED_CASH_GATE", False)
    pm = _pm(tmp_path, BlindBroker())
    assert pm.submit_order_intent(_intent()) is not None


# ---------------------------------------------------------------------------
# Turnover dampener (dual executor, pure function + wiring)
# ---------------------------------------------------------------------------

NOW = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)


def test_reductions_never_dampened_even_inside_cooldown() -> None:
    damped, blocked = apply_turnover_dampening(
        -0.05,
        last_increase_at=NOW - timedelta(seconds=10),
        now=NOW,
        factor=0.25,
        cooldown_seconds=900.0,
    )
    assert damped == -0.05 and not blocked


def test_increase_scaled_by_factor() -> None:
    damped, blocked = apply_turnover_dampening(
        0.08, last_increase_at=None, now=NOW, factor=0.25, cooldown_seconds=900.0
    )
    assert abs(damped - 0.02) < 1e-12 and not blocked


def test_increase_blocked_within_cooldown() -> None:
    damped, blocked = apply_turnover_dampening(
        0.08,
        last_increase_at=NOW - timedelta(seconds=60),
        now=NOW,
        factor=0.25,
        cooldown_seconds=900.0,
    )
    assert damped == 0.0 and blocked


def test_increase_allowed_after_cooldown_elapsed() -> None:
    damped, blocked = apply_turnover_dampening(
        0.08,
        last_increase_at=NOW - timedelta(seconds=901),
        now=NOW,
        factor=0.25,
        cooldown_seconds=900.0,
    )
    assert abs(damped - 0.02) < 1e-12 and not blocked


def test_disabled_dampener_passes_everything_through() -> None:
    damped, blocked = apply_turnover_dampening(
        0.08,
        last_increase_at=NOW - timedelta(seconds=1),
        now=NOW,
        enabled=False,
        factor=0.25,
        cooldown_seconds=900.0,
    )
    assert damped == 0.08 and not blocked


def test_executor_carries_bookkeeping_state() -> None:
    ex = TacticalExecutor(
        strategist=None,
        portfolio_manager=None,
        feature_store=None,
        risk_machine=None,
        broker=None,
    )
    assert ex._last_increase_ts == {}
