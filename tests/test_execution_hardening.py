from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src import execution
from src.broker import BrokerRiskSnapshot
from src.guardrails import RiskPolicy


class Model:
    def __init__(self, prob_up: float) -> None:
        self.prob_up = prob_up

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        return self.prob_up


class Broker:
    def __init__(self) -> None:
        self.orders: list[tuple[str, float, str]] = []
        self.cancel_all_called = 0

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        self.orders.append((symbol, qty, side))
        return True

    def cancel_all_open_orders(self) -> bool:
        self.cancel_all_called += 1
        return True


def _bars(rows: int = 40, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    price = float(start)
    closes: list[float] = []
    for i in range(rows):
        # Alternating gains/losses keep RSI and rolling-volatility features finite.
        price += 0.8 if i % 3 else -0.6
        closes.append(price)
    close = pd.Series(closes, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000 + (i % 7) * 25 for i in range(rows)],
        },
        index=index,
    )


def _pos(symbol="AAPL", qty=1.0, market_value=100.0):
    return SimpleNamespace(symbol=symbol, qty=str(qty), market_value=str(market_value), avg_entry_price="100")


def _order(symbol="AAPL", side="buy", qty=0.01, limit_price=100.0):
    return SimpleNamespace(symbol=symbol, side=side, qty=str(qty), limit_price=str(limit_price))


def _snap(*, equity=1000.0, positions=None, orders=None, equity_ok=True, positions_ok=True, open_orders_ok=True):
    return BrokerRiskSnapshot(
        equity=equity,
        positions=list(positions or []),
        open_orders=list(orders or []),
        equity_ok=equity_ok,
        positions_ok=positions_ok,
        open_orders_ok=open_orders_ok,
    )


def _policy(**kw):
    data = dict(
        profile="test",
        max_position_pct=None,
        max_gross_exposure_pct=None,
        daily_loss_pct=None,
        max_position_size_abs=5.0,
        daily_loss_limit_abs=10.0,
        max_open_positions=3,
    )
    data.update(kw)
    return RiskPolicy(**data)


def _ctx(*, snapshot=None, market_open=True, policy=None, active=("AAPL",), sentiment=None):
    return execution._Pass(
        policy=policy or _policy(),
        snapshot=snapshot or _snap(),
        market_open=market_open,
        sentiment=sentiment or {"AAPL": {"score": 8.0}, "BTC": {"score": 8.0}},
        active_target_keys={execution._norm_key(s) for s in active},
    )


# --- market-hours gating ---------------------------------------------------

def test_closed_equity_market_skips_equity_bars_and_orders(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: calls.append(symbol) or _bars())
    broker = Broker()

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(market_open=False))

    assert calls == []
    assert broker.orders == []


def test_crypto_remains_eligible_while_equity_market_closed(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: calls.append(symbol) or _bars())
    broker = Broker()

    execution.process_symbol_hardened("BTC/USD", Model(0.99), broker, _ctx(market_open=False, active=("BTC/USD",)))

    assert calls == ["BTC/USD"]
    assert len(broker.orders) == 1
    assert broker.orders[0][0] == "BTC/USD"


def test_clock_failure_fails_closed_for_equities(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: calls.append(symbol) or _bars())
    broker = Broker()

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(market_open=None))

    assert calls == []
    assert broker.orders == []


# --- broker truth and pre-trade risk gate ----------------------------------

def test_unknown_equity_allows_verified_reduction_but_blocks_buy(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    broker = Broker()
    unknown_equity = _snap(equity=None, equity_ok=False, positions=[_pos("AAPL", qty=1.0, market_value=100.0)])

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(snapshot=unknown_equity))
    assert broker.orders == []

    execution.process_symbol_hardened("AAPL", Model(0.01), broker, _ctx(snapshot=unknown_equity))
    assert broker.orders == [("AAPL", 1.0, "sell")]


@pytest.mark.parametrize("positions_ok,open_orders_ok", [(False, True), (True, False)])
def test_unknown_positions_or_open_orders_blocks_new_orders(monkeypatch, positions_ok, open_orders_ok) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    broker = Broker()
    snap = _snap(positions_ok=positions_ok, open_orders_ok=open_orders_ok)

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(snapshot=snap))

    assert broker.orders == []


def test_repeated_buys_cannot_exceed_total_symbol_cap(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars(start=100.0))
    broker = Broker()
    snap = _snap(orders=[_order("AAPL", "buy", qty=0.04, limit_price=100.0)])  # pending $4

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(snapshot=snap))

    assert broker.orders == []


def test_pending_buys_count_toward_open_position_limit(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    broker = Broker()
    policy = _policy(max_open_positions=1)
    snap = _snap(orders=[_order("MSFT", "buy", qty=0.01, limit_price=100.0)])

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(snapshot=snap, policy=policy))

    assert broker.orders == []


def test_pending_buys_count_toward_gross_exposure_limit(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    broker = Broker()
    policy = _policy(max_position_size_abs=100.0, max_gross_exposure_pct=0.10)
    snap = _snap(equity=100.0, orders=[_order("MSFT", "buy", qty=0.08, limit_price=100.0)])  # pending $8

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(snapshot=snap, policy=policy))

    assert broker.orders == []


def test_oversized_sell_cannot_open_short(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    broker = Broker()
    snap = _snap(positions=[_pos("AAPL", qty=0.25, market_value=25.0)])

    execution.process_symbol_hardened("AAPL", Model(0.01), broker, _ctx(snapshot=snap))

    assert broker.orders == [("AAPL", 0.25, "sell")]


def test_gross_exposure_blocks_increase_while_exit_allowed(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    policy = _policy(max_position_size_abs=100.0, max_gross_exposure_pct=0.10)
    over_gross = _snap(equity=100.0, positions=[_pos("MSFT", qty=1.0, market_value=15.0)])

    buy_broker = Broker()
    execution.process_symbol_hardened("AAPL", Model(0.99), buy_broker, _ctx(snapshot=over_gross, policy=policy))
    assert buy_broker.orders == []

    sell_broker = Broker()
    execution.process_symbol_hardened("MSFT", Model(0.01), sell_broker, _ctx(snapshot=over_gross, policy=policy, active=("AAPL",)))
    assert sell_broker.orders == [("MSFT", 1.0, "sell")]


def test_non_target_held_symbol_cannot_increase_but_can_sell(monkeypatch) -> None:
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    snap = _snap(positions=[_pos("MSFT", qty=1.0, market_value=100.0)])

    buy_broker = Broker()
    execution.process_symbol_hardened("MSFT", Model(0.99), buy_broker, _ctx(snapshot=snap, active=("AAPL",)))
    assert buy_broker.orders == []

    sell_broker = Broker()
    execution.process_symbol_hardened("MSFT", Model(0.01), sell_broker, _ctx(snapshot=snap, active=("AAPL",)))
    assert sell_broker.orders == [("MSFT", 1.0, "sell")]


# --- hedges, cache, and breaker -------------------------------------------

def test_hedge_opens_only_through_existing_selection_and_risk_gate(monkeypatch) -> None:
    def fake_select(symbol, bar_fetcher):
        return "PSQ"

    monkeypatch.setattr(execution, "select_hedge_asset", fake_select)
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: _bars())
    broker = Broker()

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(sentiment={"AAPL": {"score": 1.0}}))

    assert broker.orders and broker.orders[0][0] == "PSQ"


def test_each_symbol_bars_fetched_at_most_once_per_pass(monkeypatch) -> None:
    calls: dict[str, int] = {}

    def fake_fetch(symbol, lookback_days):
        calls[symbol] = calls.get(symbol, 0) + 1
        return _bars(start=100.0 if symbol != "PSQ" else 50.0)

    def fake_select(symbol, bar_fetcher):
        # Simulates existing hedge-selection calls; the selected hedge is reused
        # by _hardened_pivot instead of fetched again.
        bar_fetcher(symbol, lookback_days=30)
        bar_fetcher("PSQ", lookback_days=30)
        return "PSQ"

    monkeypatch.setattr(execution, "fetch_bars", fake_fetch)
    monkeypatch.setattr(execution, "select_hedge_asset", fake_select)
    broker = Broker()

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, _ctx(sentiment={"AAPL": {"score": 1.0}}))

    assert calls["AAPL"] == 1
    assert calls["PSQ"] == 1


def test_circuit_breaker_shutdown_cancels_open_orders(tmp_path, monkeypatch) -> None:
    import config
    from src import guardrails

    monkeypatch.setattr(config, "STATE_PATH", str(tmp_path / "bot_state.json"))
    guardrails.record_equity_anchor(100.0)
    broker = Broker()
    policy = _policy(daily_loss_limit_abs=10.0)
    snap = _snap(equity=89.0)

    with pytest.raises(SystemExit) as exc:
        execution._enforce_circuit_breaker(broker, policy, snap)

    assert exc.value.code == 1
    assert broker.cancel_all_called == 1
