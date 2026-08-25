"""
End-to-end integration test for the live execution loop (run_bot.py -> run()).

Drives the REAL src.execution.run() loop with a stub broker, stub model, and a
mocked correlation universe -- no Alpaca, no network. A patched time.sleep breaks
the otherwise-infinite loop after exactly one pass so we can assert on behavior.

This permanently guards the Active Pivot routing against regressions:
  * weak sentiment on a BUY -> pivot into the most negatively correlated inverse
    ETF and log it under the inverse ticker,
  * strong sentiment -> buy the target directly,
  * a per-symbol error never crashes the whole loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import execution


class _StopLoop(Exception):
    """Sentinel raised from a patched sleep to end the infinite loop cleanly."""


class StubModel:
    def __init__(self, prob_up: float) -> None:
        self.prob_up = prob_up

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        return self.prob_up


class StubBroker:
    """Minimal broker: records orders, never touches a network.

    Reports a trusted, flat snapshot with the equity market open so the hardened
    loop's pre-trade risk gate permits the asserted single order per pass.
    """

    def __init__(self, equity: float = 100_000.0) -> None:
        self.orders: list[tuple[str, float, str]] = []
        self.equity = equity
        self.cancelled = 0

    def get_equity(self):
        return self.equity

    def get_equity_market_open(self):
        return True

    def get_risk_snapshot(self):
        from src.broker import BrokerRiskSnapshot

        return BrokerRiskSnapshot(
            equity=self.equity,
            positions=[],
            open_orders=[],
            equity_ok=True,
            positions_ok=True,
            open_orders_ok=True,
        )

    def get_api_error_rate(self) -> float:
        return 0.0

    def get_position_qty(self, symbol: str) -> float:
        return 0.0

    def get_open_positions(self) -> list:
        return []

    def cancel_all_open_orders(self) -> bool:
        self.cancelled += 1
        return True

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        self.orders.append((symbol, qty, side))
        return True


def _bars_from_returns(returns: np.ndarray) -> pd.DataFrame:
    n = len(returns)
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
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
    """PSQ is the unique strongest negative hedge for the target AAPL."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, n)
    return {
        "AAPL": _bars_from_returns(r),
        "PSQ": _bars_from_returns(-r),                                  # corr -1
        "SH": _bars_from_returns(-0.3 * r + rng.normal(0, 0.01, n)),
        "BITI": _bars_from_returns(r.copy()),                          # corr +1
        "SARK": _bars_from_returns(np.zeros(n)),                       # NaN skip
        "SETH": _bars_from_returns(0.5 * r + rng.normal(0, 0.01, n)),
        "RWM": _bars_from_returns(0.1 * r + rng.normal(0, 0.01, n)),
        "DOG": _bars_from_returns(-0.2 * r + rng.normal(0, 0.01, n)),
    }


def _wire_loop(monkeypatch, tmp_path, *, sentiment, universe, prob_up):
    """Common harness: isolate paths, stub deps, and break after one pass."""
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    monkeypatch.setattr(config, "STATE_PATH", str(tmp_path / "bot_state.json"))
    # Hermetic model artifacts: run() fail-fasts on missing files before
    # reaching the stubbed load_model(), so point it at tiny placeholders.
    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "feature_meta.json"
    model_path.write_text("{}")
    meta_path.write_text("{}")
    monkeypatch.setattr(config, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta_path))
    monkeypatch.setattr(config, "EQUITY_SYMBOLS", ["AAPL"])
    monkeypatch.setattr(config, "CRYPTO_SYMBOLS", [])
    # Resolve the live universe deterministically so an operator-local
    # active_universe.json can never make this integration test nondeterministic.
    monkeypatch.setattr(execution, "resolve_live_universe", lambda: ["AAPL"])

    broker = StubBroker()
    monkeypatch.setattr(execution, "load_model", lambda: StubModel(prob_up))
    monkeypatch.setattr(execution, "Broker", lambda: broker)
    monkeypatch.setattr(
        execution, "fetch_bars",
        lambda symbol, lookback_days: universe.get(symbol, pd.DataFrame()),
    )
    monkeypatch.setattr(execution, "load_sentiment", lambda: sentiment)

    def _raise_stop(_seconds):
        raise _StopLoop()

    monkeypatch.setattr(execution.time, "sleep", _raise_stop)
    return broker


def test_run_loop_pivots_to_hedge_on_weak_sentiment(monkeypatch, tmp_path) -> None:
    broker = _wire_loop(
        monkeypatch,
        tmp_path,
        sentiment={"AAPL": {"score": 3.0}},  # weak -> pivot
        universe=_correlation_universe(),
        prob_up=config.BUY_THRESHOLD + 0.05,
    )

    with pytest.raises(_StopLoop):
        execution.run()

    assert len(broker.orders) == 1
    ticker, qty, side = broker.orders[0]
    assert ticker in config.INVERSE_SAFE_LIST  # pivoted into a hedge
    assert ticker != "AAPL"
    assert side == "buy"
    # Guardrail still enforced on the hedge notional.
    hedge_price = float(_correlation_universe()[ticker]["close"].iloc[-1])
    assert qty * hedge_price <= config.MAX_POSITION_SIZE

    logged = pd.read_csv(tmp_path / "trades_log.csv")
    assert logged.iloc[0]["ticker"] == ticker
    assert logged.iloc[0]["side"] == "buy"


def test_run_loop_direct_buy_on_strong_sentiment(monkeypatch, tmp_path) -> None:
    broker = _wire_loop(
        monkeypatch,
        tmp_path,
        sentiment={"AAPL": {"score": 8.0}},  # strong -> direct
        universe=_correlation_universe(),
        prob_up=config.BUY_THRESHOLD + 0.05,
    )

    with pytest.raises(_StopLoop):
        execution.run()

    assert len(broker.orders) == 1
    assert broker.orders[0][0] == "AAPL"
    assert broker.orders[0][2] == "buy"


def test_run_loop_survives_symbol_error(monkeypatch, tmp_path) -> None:
    """A fetch failure on one symbol must not crash the whole loop."""
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    monkeypatch.setattr(config, "STATE_PATH", str(tmp_path / "bot_state.json"))
    # Hermetic model artifacts (same rationale as _wire_loop above).
    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "feature_meta.json"
    model_path.write_text("{}")
    meta_path.write_text("{}")
    monkeypatch.setattr(config, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta_path))
    monkeypatch.setattr(config, "EQUITY_SYMBOLS", ["AAPL"])
    monkeypatch.setattr(config, "CRYPTO_SYMBOLS", [])
    monkeypatch.setattr(execution, "resolve_live_universe", lambda: ["AAPL"])

    def _boom(symbol, lookback_days):
        raise RuntimeError("simulated data outage")

    broker = StubBroker()
    monkeypatch.setattr(execution, "load_model", lambda: StubModel(config.BUY_THRESHOLD + 0.05))
    monkeypatch.setattr(execution, "Broker", lambda: broker)
    monkeypatch.setattr(execution, "fetch_bars", _boom)
    monkeypatch.setattr(execution, "load_sentiment", lambda: {"AAPL": {"score": 8.0}})

    def _raise_stop(_seconds):
        raise _StopLoop()

    monkeypatch.setattr(execution.time, "sleep", _raise_stop)

    # The loop should reach its sleep (one clean pass) despite the symbol error.
    with pytest.raises(_StopLoop):
        execution.run()

    assert broker.orders == []
