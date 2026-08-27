"""Pivot guards: closed-market hedge deferral + non-fractionable sizing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from src import execution


class Model:
    def __init__(self, prob_up: float) -> None:
        self.prob_up = prob_up

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        return self.prob_up


class Broker:
    """Offline double with fractionability control."""

    def __init__(self, *, fractionable: bool | None = True) -> None:
        self.orders: list[tuple[str, float, str]] = []
        self._fractionable = fractionable

    def is_fractionable(self, symbol: str):
        return self._fractionable

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        self.orders.append((symbol, qty, side))
        return True


def _bars(rows: int, start: float, drift: float, *, freq: str = "1D") -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq=freq, tz="UTC")
    closes = [start]
    for i in range(1, rows):
        closes.append(closes[-1] * (1.0 + drift + 0.001 * ((-1) ** i)))
    # Index MUST match the frame's index or DataFrame construction
    # label-aligns these columns into all-NaN.
    close = pd.Series(closes, index=index, dtype="float64", name="close")
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


def _correlation_universe(target: str = "BTC/USD") -> dict[str, pd.DataFrame]:
    """RWM is the unique strongest negative hedge (it is in INVERSE_SAFE_LIST)."""
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.01, 60)
    return {
        target: _bars_from_returns(r, start=100.0),
        "RWM": _neg_bars(r),
        # Decoys: uncorrelated / positively correlated.
        "PSQ": _bars(60, 25.0, 0.0),
        "DOG": _pos_bars(r),
    }


def _neg_bars(r: np.ndarray) -> pd.DataFrame:
    return _bars_from_returns(-r, start=50.0)


def _pos_bars(r: np.ndarray) -> pd.DataFrame:
    return _bars_from_returns(r.copy(), start=20.0)


def _bars_from_returns(returns: np.ndarray, *, start: float, freq: str = "1D") -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(returns), freq=freq, tz="UTC")
    close = pd.Series(start * np.cumprod(1.0 + returns), index=index, dtype="float64")
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


def _ctx(market_open: bool, sentiment: dict, active: tuple[str, ...]) -> execution._Pass:
    return execution._Pass(
        policy=_policy(),
        snapshot=_snap(),
        market_open=market_open,
        sentiment=sentiment,
        active_target_keys={execution._norm_key(s) for s in active},
    )


def _snap():
    from src.broker import BrokerRiskSnapshot

    return BrokerRiskSnapshot(
        equity=1000.0,
        positions=[],
        open_orders=[],
        equity_ok=True,
        positions_ok=True,
        open_orders_ok=True,
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
    from src.guardrails import RiskPolicy

    return RiskPolicy(**data)


# --- closed-market hedge deferral -------------------------------------------


def test_pivot_to_equity_hedge_deferred_when_market_closed(monkeypatch) -> None:
    universe = _correlation_universe("BTC/USD")
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    ctx = _ctx(market_open=False, sentiment={"BTC": {"score": 1.0}}, active=("BTC/USD",))

    execution.process_symbol_hardened("BTC/USD", Model(0.99), broker, ctx)

    assert broker.orders == []  # deferred, not queued at a stale price
    assert ctx.telemetry.skipped.get("hedge_market_closed") == 1


def test_pivot_proceeds_when_hedge_market_open(monkeypatch) -> None:
    universe = _correlation_universe("BTC/USD")
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    ctx = _ctx(market_open=True, sentiment={"BTC": {"score": 1.0}}, active=("BTC/USD",))

    execution.process_symbol_hardened("BTC/USD", Model(0.99), broker, ctx)

    assert len(broker.orders) == 1
    symbol, qty, side = broker.orders[0]
    assert symbol == "RWM" and side == "buy"
    assert qty > 0


# --- non-fractionable whole-share sizing -------------------------------------


def test_non_fractionable_uses_whole_shares(monkeypatch) -> None:
    universe = {"AAPL": _bars(40, start=2.5, drift=0.0)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker(fractionable=False)
    ctx = _ctx(market_open=True, sentiment={"AAPL": {"score": 8.0}}, active=("AAPL",))

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, ctx)

    assert len(broker.orders) == 1
    _, qty, side = broker.orders[0]
    assert side == "buy"
    assert float(qty) == 2.0  # floor($5 cap / $2.5) = 2 whole shares


def test_non_fractionable_below_one_share_skips(monkeypatch) -> None:
    universe = {"SETH": _bars(40, start=33.15, drift=0.0)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker(fractionable=False)
    ctx = _ctx(market_open=True, sentiment={"SETH": {"score": 8.0}}, active=("SETH",))

    execution.process_symbol_hardened("SETH", Model(0.99), broker, ctx)

    assert broker.orders == []
    assert ctx.telemetry.skipped.get("non_fractionable_too_expensive") == 1


def test_fractionable_unknown_falls_back_to_legacy_sizing(monkeypatch) -> None:
    universe = {"AAPL": _bars(40, start=100.0, drift=0.0)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    broker.is_fractionable = None  # simulate "unknown": getattr -> not callable
    ctx = _ctx(market_open=True, sentiment={"AAPL": {"score": 8.0}}, active=("AAPL",))

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, ctx)

    assert len(broker.orders) == 1
    _, qty, _ = broker.orders[0]
    last_close = float(universe["AAPL"]["close"].iloc[-1])
    assert float(qty) * last_close <= config.MAX_POSITION_SIZE + 1e-6


# --- hedge pair lifecycle: unwind / rotation --------------------------------


def _pos(symbol="BITI", qty=0.25, market_value=5.0):
    return SimpleNamespace(
        symbol=symbol,
        qty=str(qty),
        market_value=str(market_value),
        avg_entry_price="19.3",
    )


def _snap_with_positions(positions) -> SimpleNamespace:
    from src.broker import BrokerRiskSnapshot

    return BrokerRiskSnapshot(
        equity=1000.0,
        positions=list(positions),
        open_orders=[],
        equity_ok=True,
        positions_ok=True,
        open_orders_ok=True,
    )


def test_pair_state_round_trip_and_corrupt_tolerance(tmp_path) -> None:
    path = str(tmp_path / "pairs.json")
    execution._save_hedge_pairs({"RWM": "BTC/USD"}, {"AAPL": "2026-08-27"}, {"BTC/USD"}, path)
    pairs, stamps, origins = execution._load_hedge_pairs(path)
    assert pairs == {"RWM": "BTC/USD"}
    assert stamps == {"AAPL": "2026-08-27"}
    assert origins == {"BTC/USD"}

    (tmp_path / "pairs.json").write_text("{broken", encoding="utf-8")
    pairs, stamps, origins = execution._load_hedge_pairs(str(tmp_path / "pairs.json"))
    assert pairs == {} and stamps == {} and origins == set()


def test_pivot_success_records_pair(tmp_path, monkeypatch, mock_broker=None) -> None:
    universe = _correlation_universe("BTC/USD")
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    ctx = _ctx(market_open=True, sentiment={"BTC": {"score": 1.0}}, active=("BTC/USD",))
    path = str(tmp_path / "hedge_pairs.json")
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", path)

    execution.process_symbol_hardened("BTC/USD", Model(0.99), broker, ctx)

    pairs, _, _ = execution._load_hedge_pairs(path)
    assert pairs.get("RWM") == "BTC/USD"


def test_unwind_fires_when_target_is_clean_buy(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "hedge_pairs.json")
    execution._save_hedge_pairs({"RWM": "BTC/USD"}, {}, {"BTC/USD"}, path)
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", path)

    # Universe includes the held hedge AND its paired target (for readiness).
    universe = dict(_correlation_universe("BTC/USD"))
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))

    class HeldBroker(Broker):
        def __init__(self):
            super().__init__()

    broker = Broker()
    positions = [_pos("RWM")]
    snap = _snap_with_positions(positions)
    ctx = execution._Pass(
        policy=_policy(),
        snapshot=snap,
        market_open=True,
        sentiment={"BTC": {"score": 8.0}},  # explicit strong -> clean buy
        active_target_keys={execution._norm_key("BTC/USD")},
    )
    # Seed the in-memory ledger with the held hedge (mirrors real loop state).
    ctx.ledger.reserve_buy("RWM", 5.0)

    execution.process_symbol_hardened("RWM", Model(0.99), broker, ctx)

    assert len(broker.orders) == 1
    sym, qty, side = broker.orders[0]
    assert sym == "RWM" and side == "sell"
    pairs, _, _ = execution._load_hedge_pairs(path)
    assert "RWM" not in pairs  # consumed


def test_unwind_holds_when_sentiment_for_target_is_bad(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "hedge_pairs.json")
    execution._save_hedge_pairs({"RWM": "BTC/USD"}, {}, {"BTC/USD"}, path)
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", path)

    universe = dict(_correlation_universe("BTC/USD"))
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    snap = _snap_with_positions([_pos("RWM")])
    ctx = execution._Pass(
        policy=_policy(),
        snapshot=snap,
        market_open=True,
        sentiment={"BTC": {"score": 2.0}},  # explicitly bad -> keep hedging
        active_target_keys={execution._norm_key("BTC/USD")},
    )
    ctx.ledger.reserve_buy("RWM", 5.0)

    execution.process_symbol_hardened("RWM", Model(0.99), broker, ctx)

    assert broker.orders == []
    pairs, _, _ = execution._load_hedge_pairs(path)
    assert pairs == {"RWM": "BTC/USD"}  # retained


def test_rotation_releases_direct_leg_on_bad_news(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "hedge_pairs.json")
    # Historical pivot origin recorded for AAPL.
    execution._save_hedge_pairs({}, {}, {"AAPL"}, path)
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", path)

    universe = {"AAPL": _bars(60, 300.0, 0.0)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    snap = _snap_with_positions([_pos("AAPL", qty=1.0, market_value=100.0)])
    ctx = execution._Pass(
        policy=_policy(max_position_size_abs=5000.0),
        snapshot=snap,
        market_open=True,
        sentiment={"AAPL": {"score": 2.0}},  # explicitly bad news
        active_target_keys={execution._norm_key("AAPL")},
    )

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, ctx)

    assert len(broker.orders) == 1
    sym, qty, side = broker.orders[0]
    assert sym == "AAPL" and side == "sell"
    _, stamps, _ = execution._load_hedge_pairs(path)
    today = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).date().isoformat()
    assert stamps["AAPL"] == today


def test_rotation_damped_once_per_day(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "hedge_pairs.json")
    today = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).date().isoformat()
    execution._save_hedge_pairs({}, {"AAPL": today}, {"AAPL"}, path)
    monkeypatch.setattr(config, "HEDGE_PAIRS_PATH", path)

    universe = {"AAPL": _bars(60, 300.0, 0.0)}
    monkeypatch.setattr(execution, "fetch_bars", lambda sym, lookback_days: universe.get(sym, pd.DataFrame()))
    broker = Broker()
    snap = _snap_with_positions([_pos("AAPL")])
    ctx = execution._Pass(
        policy=_policy(),
        snapshot=snap,
        market_open=True,
        sentiment={"AAPL": {"score": 2.0}},
        active_target_keys={execution._norm_key("AAPL")},
    )

    execution.process_symbol_hardened("AAPL", Model(0.99), broker, ctx)

    assert broker.orders == []  # damped: already transitioned today
