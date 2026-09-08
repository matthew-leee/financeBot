"""Stale-conviction exit + on-call strategist consultation tests (offline)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import config
from src import execution

UTC = timezone.utc
NOW = datetime(2026, 9, 4, 15, 0, tzinfo=UTC)


class Model:
    def __init__(self, prob_up: float) -> None:
        self.prob_up = prob_up

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        return self.prob_up


class Broker:
    def __init__(self) -> None:
        self.orders: list[tuple[str, float, str]] = []

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        self.orders.append((symbol, qty, side))
        return True


def _bars(rows: int, start: float) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="1D", tz="UTC")
    close = pd.Series([start + (i % 5) * 0.3 for i in range(rows)], index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000.0 + (i % 7) * 25 for i in range(rows)],
        },
        index=index,
    )


def _pos(symbol="AAPL", qty=1.0, market_value=100.0):
    return SimpleNamespace(
        symbol=symbol, qty=str(qty), market_value=str(market_value), avg_entry_price="100"
    )


def _snap(positions) -> SimpleNamespace:
    from src.broker import BrokerRiskSnapshot

    return BrokerRiskSnapshot(
        equity=1000.0,
        positions=list(positions),
        open_orders=[],
        equity_ok=True,
        positions_ok=True,
        open_orders_ok=True,
    )


def _policy():
    from src.guardrails import RiskPolicy

    return RiskPolicy(
        profile="test",
        max_position_pct=None,
        max_gross_exposure_pct=None,
        daily_loss_pct=None,
        max_position_size_abs=5.0,
        daily_loss_limit_abs=10.0,
        max_open_positions=12,
    )


def _ctx(sentiment, active=("AAPL",), positions=None):
    return execution._Pass(
        policy=_policy(),
        snapshot=_snap(positions if positions is not None else [_pos()]),
        market_open=True,
        sentiment=sentiment,
        active_target_keys={execution._norm_key(s) for s in active},
        now=NOW,
    )


def _seed_trades_log(tmp_path, symbol="AAPL", days_ago=10):
    path = tmp_path / "trades_log.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "ticker", "side", "price", "size", "pnl"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
                "ticker": symbol,
                "side": "buy",
                "price": 100.0,
                "size": 1.0,
                "pnl": 0.0,
            }
        )
    return str(path)


# --- age parser -------------------------------------------------------------


def test_position_age_from_trades_log(tmp_path):
    log = _seed_trades_log(tmp_path, days_ago=10)
    orig = config.TRADES_LOG_PATH
    config.TRADES_LOG_PATH = log
    try:
        age = execution._position_age_days("AAPL", now=NOW)
        assert age is not None and 9.9 <= age <= 10.1
        assert execution._position_age_days("MSFT", now=NOW) is None
    finally:
        config.TRADES_LOG_PATH = orig


# --- verdict cache ----------------------------------------------------------


def test_verdict_cache_round_trip_and_corrupt(tmp_path):
    path = str(tmp_path / "v.json")
    execution._cache_verdict({}, "AAPL", "2026-09-04", "keep", "thesis intact", path)
    v = execution._load_verdicts(path)
    assert v["AAPL"] == {"date": "2026-09-04", "decision": "keep", "reason": "thesis intact"}

    open(path, "w").write("{broken")
    assert execution._load_verdicts(path) == {}


# --- consult ----------------------------------------------------------------


def test_consult_discard_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "dummy-key-for-tests")
    path = str(tmp_path / "v.json")
    monkeypatch.setattr(config, "STRATEGIST_VERDICTS_PATH", path)
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "t.csv"))

    def transport(url, *, headers, body):
        user = body["messages"][-1]["content"]
        assert "news_trajectory" in user and "held_days" in user
        return {"choices": [{"message": {"content": json.dumps(
            {"decision": "discard", "confidence": 0.8, "reason": "thesis deteriorated"}
        )}}]}

    decision, reason = execution._strategist_consult(
        "AAPL", held_days=9.0, prob_up=0.50, sentiment_report={},
        transport=transport, fetcher=lambda s, lookback_days: _bars(100, 100.0),
        now=NOW, verdicts_path=path,
    )
    assert decision == "discard"
    v = execution._load_verdicts(path)
    assert v["AAPL"]["decision"] == "discard"


def test_consult_garbage_defaults_to_keep(tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "dummy-key-for-tests")
    path = str(tmp_path / "v.json")
    monkeypatch.setattr(config, "STRATEGIST_VERDICTS_PATH", path)

    def transport(url, *, headers, body):
        return {"choices": [{"message": {"content": "total garbage"}}]}

    decision, reason = execution._strategist_consult(
        "AAPL", held_days=9.0, prob_up=0.50, sentiment_report={},
        transport=transport, fetcher=lambda s, lookback_days: _bars(100, 100.0),
        now=NOW, verdicts_path=path,
    )
    assert decision == "keep" and "unavailable" in reason
    assert execution._load_verdicts(path)["AAPL"]["decision"] == "keep"


def test_consult_cached_same_day_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "dummy-key-for-tests")
    path = str(tmp_path / "v.json")
    execution._cache_verdict({}, "AAPL", NOW.date().isoformat(), "keep", "cached", path)
    monkeypatch.setattr(config, "STRATEGIST_VERDICTS_PATH", path)

    def transport(url, *, headers, body):  # pragma: no cover
        raise AssertionError("must not call LLM when cached today")

    decision, reason = execution._strategist_consult(
        "AAPL", held_days=9.0, prob_up=0.50, sentiment_report={},
        transport=transport, fetcher=lambda s, lookback_days: _bars(100, 100.0),
        now=NOW, verdicts_path=path,
    )
    assert decision == "keep" and reason == "cached"


# --- hook end-to-end --------------------------------------------------------


def _wire_hook(tmp_path, monkeypatch, *, verdict, prob, days_ago=10, held=True):
    monkeypatch.setattr(config, "STRATEGIST_VERDICTS_PATH", str(tmp_path / "v.json"))
    monkeypatch.setattr(config, "TRADES_LOG_PATH", _seed_trades_log(tmp_path, days_ago=days_ago))
    monkeypatch.setattr(
        execution, "fetch_bars", lambda sym, lookback_days: _bars(100, 100.0)
    )
    consult_calls: list[str] = []

    def fake_consult(symbol, **kw):
        consult_calls.append(symbol)
        return verdict, f"{verdict} reason"

    monkeypatch.setattr(execution, "_strategist_consult", fake_consult)
    broker = Broker()
    positions = [_pos("AAPL")] if held else []
    ctx = _ctx(sentiment={}, active=("AAPL",), positions=positions)
    if held:
        ctx.ledger.reserve_buy("AAPL", 5.0)
    return broker, ctx, consult_calls


def test_stale_exit_discard_sells(tmp_path, monkeypatch):
    broker, ctx, calls = _wire_hook(tmp_path, monkeypatch, verdict="discard", prob=0.50)
    execution.process_symbol_hardened("AAPL", Model(0.50), broker, ctx)
    assert len(calls) == 1
    assert broker.orders == [("AAPL", 1.0, "sell")]


def test_stale_exit_keep_holds_without_orders(tmp_path, monkeypatch):
    broker, ctx, calls = _wire_hook(tmp_path, monkeypatch, verdict="keep", prob=0.50)
    execution.process_symbol_hardened("AAPL", Model(0.50), broker, ctx)
    assert len(calls) == 1
    assert broker.orders == []  # strategist said keep; nothing happens


def test_young_position_never_consults(tmp_path, monkeypatch):
    broker, ctx, calls = _wire_hook(tmp_path, monkeypatch, verdict="discard", prob=0.50, days_ago=2)
    execution.process_symbol_hardened("AAPL", Model(0.50), broker, ctx)
    assert calls == [] and broker.orders == []


def test_conviction_outside_deadzone_never_consults(tmp_path, monkeypatch):
    broker, ctx, calls = _wire_hook(tmp_path, monkeypatch, verdict="discard", prob=0.59)
    # P=0.59 -> action=buy, already long -> rotation not applicable (no pair
    # origin), buy path skips as already long. No stale consult either.
    execution.process_symbol_hardened("AAPL", Model(0.59), broker, ctx)
    assert calls == []
    assert broker.orders == []


# --- slots follow universe --------------------------------------------------


def test_slots_follow_universe_replaces_policy():
    import dataclasses

    policy = _policy()
    universe = ["A", "B", "C"]
    updated = dataclasses.replace(policy, max_open_positions=len(universe))
    assert updated.max_open_positions == 3
    assert policy.max_open_positions == 12  # original immutable


# --- mid-run universe refresh -----------------------------------------------


def test_refresh_picks_up_pool_change_and_reslots(tmp_path, monkeypatch):
    uni = tmp_path / "active_universe.json"
    uni.write_text(
        json.dumps({"version": 1, "as_of": "2026-09-08T00:00:00Z", "symbols": ["AAA", "BBB"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", str(uni))
    monkeypatch.setattr(config, "SLOTS_FOLLOW_UNIVERSE", True)

    policy = _policy()  # 12 slots
    state: dict = {"fingerprint": execution._universe_file_fingerprint()}
    p1, t1, k1, changed = execution._maybe_refresh_universe(
        policy, ["OLD"], {execution._norm_key("OLD")}, state
    )
    assert changed is False  # first sighting: baseline, no change event
    assert t1 == ["OLD"]

    # Curator writes a new pool (content + mtime change).
    import os as _os
    import time as _time

    _time.sleep(0.01)
    uni.write_text(
        json.dumps({"version": 1, "as_of": "2026-09-09T00:00:00Z", "symbols": ["CCC", "DDD", "EEE"]}),
        encoding="utf-8",
    )
    _os.utime(uni, None)
    p2, t2, k2, changed = execution._maybe_refresh_universe(policy, t1, k1, state)
    assert changed is True
    assert sorted(t2) == ["CCC", "DDD", "EEE"]
    assert p2.max_open_positions == 3  # slots follow universe
    # Third call: no further change event.
    _, t3, _, changed3 = execution._maybe_refresh_universe(p2, t2, k2, state)
    assert changed3 is False and t3 == t2


def test_refresh_failure_keeps_previous_targets(tmp_path, monkeypatch):
    uni = tmp_path / "active_universe.json"
    uni.write_text(
        json.dumps({"version": 1, "as_of": "2026-09-08T00:00:00Z", "symbols": ["AAA"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", str(uni))
    state: dict = {"fingerprint": execution._universe_file_fingerprint()}
    p0, t0, k0, _ = execution._maybe_refresh_universe(_policy(), ["OLD"], set(), state)

    # Corrupt the file AND force strict mode so resolution raises.
    uni.write_text("{broken", encoding="utf-8")
    import os as _os
    _os.utime(uni, None)
    monkeypatch.setattr(config, "PAPER", False)  # strict on
    monkeypatch.setenv("FINANCEBOT_UNIVERSE_STRICT", "true")

    p1, t1, k1, changed = execution._maybe_refresh_universe(p0, t0, k0, state)
    assert changed is False
    assert t1 == ["OLD"]  # fail-conservative


# --- orphan review (ex-targets dropped by pool churn) ------------------------


def test_pool_dropped_ex_target_gets_strategist_review(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", str(tmp_path / "h.json"))
    monkeypatch.setattr(config, "STRATEGIST_VERDICTS_PATH", str(tmp_path / "v.json"))
    monkeypatch.setattr(config, "TRADES_LOG_PATH", _seed_trades_log(tmp_path, symbol="MSFT", days_ago=10))
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: _bars(100, 100.0))
    monkeypatch.setattr(execution, "_strategist_consult", lambda sym, **kw: ("discard", "dropped from pool"))
    broker = Broker()
    # MSFT held but NOT in active targets (pool churn dropped it); in origins.
    ctx = _ctx(sentiment={}, active=("OTHER",), positions=[_pos("MSFT")])
    ctx.ledger.reserve_buy("MSFT", 5.0)
    ctx.now = NOW

    execution.process_symbol_hardened("MSFT", Model(0.50), broker, ctx)

    assert broker.orders == [("MSFT", 1.0, "sell")]


def test_hedges_are_exempt_from_orphan_review(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", str(tmp_path / "h.json"))
    monkeypatch.setattr(config, "STRATEGIST_VERDICTS_PATH", str(tmp_path / "v.json"))
    monkeypatch.setattr(config, "TRADES_LOG_PATH", _seed_trades_log(tmp_path, symbol="RWM", days_ago=10))
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: _bars(100, 100.0))
    monkeypatch.setattr(execution, "_strategist_consult", lambda sym, **kw: (_ for _ in ()).throw(AssertionError("hedge consulted")))
    broker = Broker()
    execution._save_hedge_pairs({"RWM": "BTC/USD"}, {}, set(), str(tmp_path / "h.json"))
    ctx = _ctx(sentiment={}, active=("OTHER",), positions=[_pos("RWM")])
    ctx.ledger.reserve_buy("RWM", 5.0)
    ctx.now = NOW

    execution.process_symbol_hardened("RWM", Model(0.50), broker, ctx)

    assert broker.orders == []  # hedge lifecycle governs, no strategist consult


def test_target_buy_registers_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", str(tmp_path / "h.json"))
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "t.csv"))
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: _bars(60, 100.0))
    broker = Broker()
    ctx = _ctx(sentiment={"AAPL": {"score": 8.0}}, active=("AAPL",), positions=[])

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, ctx)

    assert len(broker.orders) == 1
    _, _, origins = execution._load_hedge_pairs(str(tmp_path / "h.json"))
    assert "AAPL" in origins
