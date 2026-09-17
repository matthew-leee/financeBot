"""The Gardener: conviction/mood/inverse-vol sizing + continuous re-evaluation."""

from __future__ import annotations

import json

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import config
from src import execution

UTC = timezone.utc
NOW = datetime(2026, 9, 17, 15, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _sizing_mode_on(monkeypatch):
    """The sizing suite opts into conviction_vol (conftest defaults to flat)."""
    monkeypatch.setattr(config, "SIZING_MODE", "conviction_vol")


def _bars_daily(rows: int, start: float, day_move: float) -> pd.DataFrame:
    """Daily bars with alternating +-day_move returns (controllable vol)."""
    index = pd.date_range("2026-06-01", periods=rows, freq="1D", tz="UTC")
    close = pd.Series([start] * rows, index=index, dtype="float64")
    for i in range(1, rows):
        close.iloc[i] = close.iloc[i - 1] * (1.0 + day_move * ((-1) ** i))
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


def _bars_hourly(days: int, start: float, day_move: float) -> pd.DataFrame:
    """
    Hourly bars interpolated between daily closes that alternate +-day_move.
    Daily-resampled vol is therefore ~ day_move (controllable), unlike naive
    alternating intraday steps which collapse to ~zero daily movement.
    """
    per_day = 6
    index = pd.date_range("2026-06-01", periods=days * per_day, freq="4h", tz="UTC")
    closes: list[float] = []
    prev = start
    for d in range(days):
        target = prev * (1.0 + day_move * ((-1) ** d))
        for k in range(per_day):
            closes.append(prev + (target - prev) * (k + 1) / per_day)
        prev = target
    close = pd.Series(closes, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000.0 + (i % 7) * 25 for i in range(len(close))],
        },
        index=index,
    )


def _size(symbol="X", *, cap=5000.0, prob=0.64, mood=None, bars=None):
    return execution._determine_position_size(
        symbol, cap=cap, prob_up=prob, sentiment_report=mood or {}, bars=bars,
    )


# --- formula -----------------------------------------------------------------


def test_conviction_maps_thresholds():
    assert _size(prob=0.58)["conviction_mult"] == 0.60
    mid = _size(prob=0.64)["conviction_mult"]
    assert 0.60 < mid < 1.00
    assert _size(prob=0.75)["conviction_mult"] == 1.00


def test_mood_maps_and_missing_is_neutral():
    assert _size(mood={"X": {"score": 5.0}})["sentiment_mult"] == 0.80
    assert _size(mood={"X": {"score": 10.0}})["sentiment_mult"] == 1.00
    assert _size(mood={})["sentiment_mult"] == 0.90


def test_inverse_vol_clamps():
    calm = _bars_daily(60, 100.0, 0.002)   # ~3% ann -> clamps HIGH
    jumpy = _bars_daily(60, 100.0, 0.063)  # ~100% ann -> clamps LOW
    r_calm = _size(bars=calm)
    r_jumpy = _size(bars=jumpy)
    assert r_calm["invvol_mult"] == 3.0
    assert r_jumpy["invvol_mult"] == 0.33
    # Calm+high conviction product exceeds cap -> clamped to cap.
    assert r_calm["notional"] == 5000.0
    # Jumpy: mid conviction x neutral mood x 0.33.
    # conv(0.64)=0.80, mood 0.90, invvol 0.33
    assert r_jumpy["notional"] == pytest.approx(5000.0 * 0.80 * 0.90 * 0.33)


def test_degenerate_bars_fall_back_to_flat_mult():
    r = _size(bars=pd.DataFrame({"close": [1.0]}))  # unusable vol
    assert r["invvol_mult"] == 1.0
    # conv(0.64)=0.80, mood 0.90, invvol 1.0 fallback
    assert r["notional"] == pytest.approx(5000.0 * 0.80 * 0.90)


def test_hedge_conviction_fixed():
    r = _size(prob=None, mood={})  # no bars -> invvol 1.0
    assert r["conviction_mult"] == 0.75 and r["sentiment_mult"] == 0.90
    assert r["notional"] == pytest.approx(5000.0 * 0.75 * 0.90)


def test_min_fraction_floor_binds_when_configured(monkeypatch):
    monkeypatch.setattr(config, "SIZE_MIN_FRACTION", 0.25)
    jumpy = _bars_daily(60, 100.0, 0.063)
    r = _size(prob=0.58, mood={"X": {"score": 5.0}}, bars=jumpy)
    # product = 5000*0.6*0.8*0.33 = 792 < 1250 floor -> floored
    assert r["notional"] == 1250.0


def test_flat_mode_passthrough(monkeypatch):
    monkeypatch.setattr(config, "SIZING_MODE", "flat")
    r = _size(prob=0.99, mood={"X": {"score": 10}}, bars=_bars_daily(60, 100.0, 0.063))
    assert r["mode"] == "flat" and r["notional"] == 5000.0


# --- wiring: new buys + gardener ---------------------------------------------


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


def _ctx(sentiment, active=("AAA",), positions=None):
    from src.broker import BrokerRiskSnapshot
    from src.guardrails import RiskPolicy

    policy = RiskPolicy(
        profile="test", max_position_pct=None, max_gross_exposure_pct=None,
        daily_loss_pct=None, max_position_size_abs=5000.0,
        daily_loss_limit_abs=10.0, max_open_positions=12,
    )
    return execution._Pass(
        policy=policy,
        snapshot=BrokerRiskSnapshot(
            equity=100000.0,
            positions=list(positions or []),
            open_orders=[],
            equity_ok=True,
            positions_ok=True,
            open_orders_ok=True,
        ),
        market_open=True,
        sentiment=sentiment,
        active_target_keys={execution._norm_key(s) for s in active},
        now=NOW,
    )


def test_new_buy_uses_sized_notional(tmp_path, monkeypatch):
    # 100%-vol toy -> invvol 0.33 -> sized far below the flat cap.
    universe = {"AAA": _bars_hourly(40, 100.0, 0.063)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe[sym])
    monkeypatch.setattr(config, "SIZING_LOG_PATH", str(tmp_path / "sizing_log.csv"))
    broker = Broker()
    ctx = _ctx(sentiment={}, active=("AAA",))

    execution.process_symbol_hardened("AAA", Model(0.65), broker, ctx)

    assert len(broker.orders) == 1
    _, qty, _ = broker.orders[0]
    last = float(universe["AAA"]["close"].iloc[-1])
    assert float(qty) * last <= 5000.0 * 0.7  # well under flat cap for a jumpy toy
    log = (tmp_path / "sizing_log.csv").read_text()
    assert "AAA" in log


def test_gardener_tops_up_undersized_active_target(tmp_path, monkeypatch):
    # Calm toy (invvol 1.0 at ~20% vol): sized target ~2700 >> held $100.
    monkeypatch.setattr(config, "SIZING_LOG_PATH", str(tmp_path / "sizing_log.csv"))
    monkeypatch.setattr(config, "RESIZE_STATE_PATH", str(tmp_path / "state.json"))
    universe = {"AAA": _bars_hourly(40, 100.0, 0.0126)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    pos = SimpleNamespace(symbol="AAA", qty="1.0", market_value="100.0", avg_entry_price="100")
    ctx = _ctx(sentiment={}, active=("AAA",), positions=[pos])
    ctx.ledger.reserve_buy("AAA", 100.0)

    execution.process_symbol_hardened("AAA", Model(0.60), broker, ctx)

    assert broker.orders and broker.orders[-1][2] == "buy"
    log = (tmp_path / "sizing_log.csv").read_text()
    assert "topup" in log


def test_gardener_trims_oversized_position(tmp_path, monkeypatch):
    # Jumpy toy held at $5000 notional: sized target ~1100 -> trim the excess.
    monkeypatch.setattr(config, "SIZING_LOG_PATH", str(tmp_path / "sizing_log.csv"))
    monkeypatch.setattr(config, "RESIZE_STATE_PATH", str(tmp_path / "state.json"))
    universe = {"BBB": _bars_hourly(40, 100.0, 0.063)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    pos = SimpleNamespace(symbol="BBB", qty="50.0", market_value="5000.0", avg_entry_price="100")
    ctx = _ctx(sentiment={}, active=("BBB",), positions=[pos])
    ctx.ledger.reserve_buy("BBB", 5000.0)

    execution.process_symbol_hardened("BBB", Model(0.60), broker, ctx)

    assert broker.orders and broker.orders[-1][2] == "sell"
    _, qty, _ = broker.orders[-1]
    # Trim sells the excess above ~1100: 5000 - 1100 = 3900 notional.
    assert float(qty) * 100.0 >= 3000.0
    log = (tmp_path / "sizing_log.csv").read_text()
    assert "trim" in log


def test_gardener_respects_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIZING_LOG_PATH", str(tmp_path / "sizing_log.csv"))
    monkeypatch.setattr(config, "RESIZE_STATE_PATH", str(tmp_path / "state.json"))
    state = {"cooldown": {"AAA": NOW.isoformat()}}  # adjusted seconds ago
    (tmp_path / "state.json").write_text(json.dumps(state))
    universe = {"AAA": _bars_hourly(40, 100.0, 0.0126)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    pos = SimpleNamespace(symbol="AAA", qty="1.0", market_value="100.0", avg_entry_price="100")
    ctx = _ctx(sentiment={}, active=("AAA",), positions=[pos])
    ctx.ledger.reserve_buy("AAA", 100.0)

    execution.process_symbol_hardened("AAA", Model(0.60), broker, ctx)

    assert broker.orders == []  # hand-off rule: adjusted moments ago


def test_gardener_no_topup_for_non_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIZING_LOG_PATH", str(tmp_path / "sizing_log.csv"))
    monkeypatch.setattr(config, "RESIZE_STATE_PATH", str(tmp_path / "state.json"))
    universe = {"AAA": _bars_hourly(40, 100.0, 0.0126)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    pos = SimpleNamespace(symbol="AAA", qty="1.0", market_value="100.0", avg_entry_price="100")
    # AAA dropped from the pool -> exit-only: top-up must NOT fire.
    ctx = _ctx(sentiment={}, active=("OTHER",), positions=[pos])
    ctx.ledger.reserve_buy("AAA", 100.0)

    execution.process_symbol_hardened("AAA", Model(0.60), broker, ctx)

    assert broker.orders == []
