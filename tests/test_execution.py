from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pandas as pd

import config
from src import execution
from src.broker import Broker
from src.guardrails import clamp_position_size


class FakeModel:
    def __init__(self, prob_up: float) -> None:
        self.prob_up = prob_up

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        assert len(features_row) == 1
        return self.prob_up


class FakeBroker:
    def __init__(self, held_qty: float = 0.0, open_positions: list | None = None) -> None:
        self.held_qty = held_qty
        self.open_positions = open_positions or []
        self.orders: list[tuple[str, float, str]] = []

    def get_position_qty(self, symbol: str) -> float:
        return self.held_qty

    def get_open_positions(self) -> list:
        return self.open_positions

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        self.orders.append((symbol, qty, side))
        return True


def _sample_bars(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        periods=rows,
        freq="h",
    )
    price = 100.0
    closes: list[float] = []
    for i in range(rows):
        # Alternate gains/losses so RSI and rolling volatility are well-defined.
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


def test_process_symbol_submits_guarded_buy_order(monkeypatch, tmp_path) -> None:
    bars = _sample_bars()
    broker = FakeBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.01)

    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: bars)

    execution.process_symbol("BTC/USD", model, broker, {"BTC": {"score": 8.0}})

    latest_price = float(bars["close"].iloc[-1])
    expected_qty = clamp_position_size(config.MAX_POSITION_SIZE, latest_price)
    assert broker.orders == [("BTC/USD", expected_qty, "buy")]
    assert expected_qty * latest_price <= config.MAX_POSITION_SIZE


def test_process_symbol_handles_broker_order_rejection_without_crashing(monkeypatch) -> None:
    class RejectingBroker(FakeBroker):
        def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
            self.orders.append((symbol, qty, side))
            return False  # mirrors Broker.submit_market_order on HTTP 429/order rejection

    bars = _sample_bars()
    broker = RejectingBroker(held_qty=0.0)
    model = FakeModel(prob_up=config.BUY_THRESHOLD + 0.01)

    monkeypatch.setattr(execution, "fetch_bars", lambda symbol, lookback_days: bars)

    execution.process_symbol("BTC/USD", model, broker, {"BTC": {"score": 8.0}})

    assert len(broker.orders) == 1


def _install_fake_trading_alpaca(monkeypatch, *, fail_submit: bool) -> None:
    class FakeAccount:
        equity = "123.45"

    class FakeOrder:
        id = "fake-order-id"

    class FakeTradingClient:
        def __init__(self, key: str, secret: str, paper: bool) -> None:
            assert key == "test-key"
            assert secret == "test-secret"
            assert paper is config.PAPER

        def get_account(self):
            if fail_submit:
                raise TimeoutError("HTTP 429 rate limit")
            return FakeAccount()

        def get_all_positions(self):
            if fail_submit:
                raise TimeoutError("HTTP 429 rate limit")
            return []

        def submit_order(self, request):  # noqa: ANN001 - mirrors SDK call shape
            if fail_submit:
                raise TimeoutError("HTTP 429 rate limit")
            return FakeOrder()

    class FakeMarketOrderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    alpaca = types.ModuleType("alpaca")
    trading_pkg = types.ModuleType("alpaca.trading")
    client = types.ModuleType("alpaca.trading.client")
    enums = types.ModuleType("alpaca.trading.enums")
    requests = types.ModuleType("alpaca.trading.requests")

    client.TradingClient = FakeTradingClient
    enums.OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
    enums.TimeInForce = types.SimpleNamespace(GTC="gtc", DAY="day")
    requests.MarketOrderRequest = FakeMarketOrderRequest

    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests)


def test_broker_wrapper_swallows_http_429_style_failures(monkeypatch) -> None:
    _install_fake_trading_alpaca(monkeypatch, fail_submit=True)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setattr(config, "API_CALL_DELAY_SECONDS", 0.0)

    broker = Broker()

    assert broker.get_equity() is None
    assert broker.get_open_positions() == []
    assert broker.submit_market_order("BTC/USD", 0.001, "buy") is False


def test_broker_wrapper_returns_success_for_mocked_order(monkeypatch) -> None:
    _install_fake_trading_alpaca(monkeypatch, fail_submit=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setattr(config, "API_CALL_DELAY_SECONDS", 0.0)

    broker = Broker()

    assert broker.get_equity() == 123.45
    assert broker.get_open_positions() == []
    assert broker.submit_market_order("BTC/USD", 0.001, "buy") is True



