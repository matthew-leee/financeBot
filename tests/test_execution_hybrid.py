from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from src import execution
from src.trade_log import TRADE_LOG_COLUMNS


class FakeModel:
    def __init__(self, prob_up: float) -> None:
        self.prob_up = prob_up

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        assert len(features_row) == 1
        return self.prob_up


class FakeBroker:
    def __init__(self, held_qty: float = 0.0) -> None:
        self.held_qty = held_qty
        self.orders: list[tuple[str, float, str]] = []

    def get_position_qty(self, symbol: str) -> float:
        return self.held_qty

    def get_open_positions(self) -> list:
        return []

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        self.orders.append((symbol, qty, side))
        return True


def _bars_from_returns(returns: np.ndarray) -> pd.DataFrame:
    """Build daily OHLCV bars whose close pct_change reproduces `returns`."""
    n = len(returns)
    index = pd.date_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc), periods=n, freq="D"
    )
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": [1000 + (i % 9) * 30 for i in range(n)],
        },
        index=index,
    )


def _correlation_universe(n: int = 40) -> dict[str, pd.DataFrame]:
    """
    Deterministic price universe where PSQ is the strongest negative hedge.

      AAPL : base returns
      PSQ  : exact inverse of AAPL  -> corr = -1.0 (selected)
      SH   : partial inverse + noise -> corr in (-1, 0)
      BITI : same as AAPL           -> corr = +1.0
      SARK : flat (zero variance)   -> corr NaN -> skipped
    """
    rng = np.random.default_rng(0)
    r_aapl = rng.normal(0.0, 0.01, n)
    return {
        "AAPL": _bars_from_returns(r_aapl),
        "PSQ": _bars_from_returns(-r_aapl),
        "SH": _bars_from_returns(-0.3 * r_aapl + rng.normal(0.0, 0.01, n)),
        "BITI": _bars_from_returns(r_aapl.copy()),
        "SARK": _bars_from_returns(np.zeros(n)),
    }


def _install_mock_fetch(monkeypatch, universe: dict[str, pd.DataFrame]) -> None:
    def _fetch(symbol, lookback_days):  # noqa: ANN001 - mirrors fetch_bars signature
        return universe.get(symbol, pd.DataFrame())

    monkeypatch.setattr(execution, "fetch_bars", _fetch)


def test_weak_sentiment_pivots_to_most_negatively_correlated_etf(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    _install_mock_fetch(monkeypatch, _correlation_universe())

    broker = FakeBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.05)  # strong BUY
    weak_sentiment = {"AAPL": {"score": 4.0}}  # below the 5.0 gate -> pivot

    execution.process_symbol("AAPL", model, broker, weak_sentiment)

    # Pivoted: exactly one order, and it is the inverse ETF PSQ (not AAPL).
    assert len(broker.orders) == 1
    ordered_ticker, _, side = broker.orders[0]
    assert ordered_ticker == "PSQ"
    assert side == "buy"

    # FIFO log records the trade under the INVERSE ticker for dashboard accuracy.
    logged = pd.read_csv(tmp_path / "trades_log.csv")
    assert list(logged.columns) == TRADE_LOG_COLUMNS
    assert logged.iloc[0]["ticker"] == "PSQ"
    assert logged.iloc[0]["side"] == "buy"


def test_missing_sentiment_report_is_neutral_pass(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    _install_mock_fetch(monkeypatch, _correlation_universe())

    broker = FakeBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.05)

    # Empty report -> no score -> NEUTRAL-PASS -> buy the target directly.
    execution.process_symbol("AAPL", model, broker, {})

    assert len(broker.orders) == 1
    assert broker.orders[0][0] == "AAPL"


def test_explicit_weak_sentiment_still_pivots(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    _install_mock_fetch(monkeypatch, _correlation_universe())

    broker = FakeBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.05)

    # Explicit LOW score is real bad news -> Active Pivot (not neutral).
    execution.process_symbol("AAPL", model, broker, {"AAPL": {"score": 3.0}})

    assert len(broker.orders) == 1
    assert broker.orders[0][0] == "PSQ"


def test_strong_sentiment_buys_target_directly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    _install_mock_fetch(monkeypatch, _correlation_universe())

    broker = FakeBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.05)
    strong_sentiment = {"AAPL": {"score": 7.0}}  # clears the gate -> direct buy

    execution.process_symbol("AAPL", model, broker, strong_sentiment)

    assert len(broker.orders) == 1
    assert broker.orders[0][0] == "AAPL"
    assert broker.orders[0][2] == "buy"

    logged = pd.read_csv(tmp_path / "trades_log.csv")
    assert logged.iloc[0]["ticker"] == "AAPL"


def test_pivot_declines_when_no_hedge_price_available(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))

    # Only the target has data; every inverse ETF is empty -> no computable hedge.
    universe = {"AAPL": _correlation_universe()["AAPL"]}
    _install_mock_fetch(monkeypatch, universe)

    broker = FakeBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.05)
    weak_sentiment = {"AAPL": {"score": 3.0}}

    execution.process_symbol("AAPL", model, broker, weak_sentiment)

    # No hedge candidate -> decline the trade (no order, no log).
    assert broker.orders == []
    assert not os.path.exists(tmp_path / "trades_log.csv")
