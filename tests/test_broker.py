from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone

import config
from src.broker import Broker


@dataclass
class OrderIntentStub:
    symbol: str = "AAPL"
    side: str = "buy"
    quantity: float = 0.01
    order_style: str = "market"
    limit_price: float | None = None


class FakeAccount:
    equity = "123.45"
    cash = "67.89"


class FakePosition:
    symbol = "AAPL"
    qty = "0.5"
    market_value = "75.0"
    avg_entry_price = "100.0"
    unrealized_pl = "1.23"


class FakeOrder:
    id = "fake-order-id"


class FakeTrade:
    price = "101.25"


class FakeClock:
    def __init__(self, is_open: bool = True) -> None:
        self.is_open = is_open


class FakeTradingClient:
    instances: list["FakeTradingClient"] = []
    fail_next: bool = False
    fail_on: set[str] = set()
    positions_payload: list | None = None
    orders_payload: list | None = None
    clock_open: bool = True

    def __init__(self, key: str, secret: str, paper: bool) -> None:
        assert key == "test-key"
        assert secret == "test-secret"
        assert paper is config.PAPER
        self.submitted_requests: list[object] = []
        self.cancel_orders_called = 0
        self.get_orders_called = 0
        self.raw_gets: list[tuple[str, object]] = []
        FakeTradingClient.instances.append(self)

    def _maybe_fail(self, name: str) -> None:
        if FakeTradingClient.fail_next or name in FakeTradingClient.fail_on:
            FakeTradingClient.fail_next = False
            raise TimeoutError("HTTP 429 rate limit")

    def get_account(self):
        self._maybe_fail("account")
        return FakeAccount()

    def get_all_positions(self):
        self._maybe_fail("positions")
        if FakeTradingClient.positions_payload is not None:
            return FakeTradingClient.positions_payload
        return [FakePosition()]

    def get_open_position(self, symbol):
        self._maybe_fail("position")
        assert symbol == "AAPL"
        return FakePosition()

    def get_clock(self):
        self._maybe_fail("clock")
        return FakeClock(FakeTradingClient.clock_open)

    def submit_order(self, request):  # noqa: ANN001 - mirrors SDK call shape
        self._maybe_fail("submit")
        self.submitted_requests.append(request)
        return FakeOrder()

    def get_orders(self, filter=None):  # noqa: ANN001 - mirrors SDK call shape
        self._maybe_fail("orders")
        self.get_orders_called += 1
        self.last_order_filter = filter
        if FakeTradingClient.orders_payload is not None:
            return FakeTradingClient.orders_payload
        return [FakeOrder(), FakeOrder()]

    def cancel_orders(self):
        self._maybe_fail("cancel")
        self.cancel_orders_called += 1
        return [FakeOrder()]

    def get(self, path, data=None, **kwargs):  # noqa: ANN001 - mirrors SDK call shape
        self._maybe_fail("raw_get")
        self.raw_gets.append((path, data))
        return [
            {
                "id": "fill-1",
                "order_id": "order-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "0.25",
                "price": "101.5",
                "transaction_time": "2026-06-01T12:00:00Z",
                "type": "fill",
            }
        ]


class FakeStockHistoricalDataClient:
    def __init__(self, api_key: str, secret_key: str) -> None:
        assert api_key == "test-key"
        assert secret_key == "test-secret"

    def get_stock_latest_trade(self, request):  # noqa: ANN001
        assert request.symbol_or_symbols == "AAPL"
        return {"AAPL": FakeTrade()}


class FakeCryptoHistoricalDataClient:
    def get_crypto_latest_trade(self, request):  # noqa: ANN001
        assert request.symbol_or_symbols == "BTC/USD"
        return {"BTC/USD": FakeTrade()}


class FakeRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_fake_alpaca(monkeypatch) -> None:
    FakeTradingClient.instances.clear()
    FakeTradingClient.fail_next = False
    FakeTradingClient.fail_on = set()
    FakeTradingClient.positions_payload = None
    FakeTradingClient.orders_payload = None
    FakeTradingClient.clock_open = True

    alpaca = types.ModuleType("alpaca")
    trading_pkg = types.ModuleType("alpaca.trading")
    trading_client = types.ModuleType("alpaca.trading.client")
    trading_enums = types.ModuleType("alpaca.trading.enums")
    trading_requests = types.ModuleType("alpaca.trading.requests")
    data_pkg = types.ModuleType("alpaca.data")
    data_historical = types.ModuleType("alpaca.data.historical")
    data_requests = types.ModuleType("alpaca.data.requests")

    trading_client.TradingClient = FakeTradingClient
    trading_enums.OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
    trading_enums.TimeInForce = types.SimpleNamespace(GTC="gtc", DAY="day")
    trading_enums.QueryOrderStatus = types.SimpleNamespace(OPEN="open")
    trading_requests.MarketOrderRequest = FakeRequest
    trading_requests.LimitOrderRequest = FakeRequest
    trading_requests.GetOrdersRequest = FakeRequest
    data_historical.StockHistoricalDataClient = FakeStockHistoricalDataClient
    data_historical.CryptoHistoricalDataClient = FakeCryptoHistoricalDataClient
    data_requests.StockLatestTradeRequest = FakeRequest
    data_requests.CryptoLatestTradeRequest = FakeRequest

    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", trading_client)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", trading_enums)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", trading_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data", data_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.data.historical", data_historical)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", data_requests)


def _broker(monkeypatch) -> Broker:
    _install_fake_alpaca(monkeypatch)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setattr(config, "API_CALL_DELAY_SECONDS", 0.0)
    return Broker()


def test_config_env_defaults_are_safe(monkeypatch) -> None:
    monkeypatch.delenv("FINANCEBOT_PAPER", raising=False)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    monkeypatch.delenv("FINANCEBOT_ENGINE", raising=False)
    reloaded = importlib.reload(config)
    assert reloaded.PAPER is True
    assert reloaded.ENGINE == "legacy"


def test_config_env_invalid_paper_value_stays_safe(monkeypatch) -> None:
    monkeypatch.setenv("FINANCEBOT_PAPER", "flase")
    reloaded = importlib.reload(config)
    assert reloaded.PAPER is True
    monkeypatch.delenv("FINANCEBOT_PAPER", raising=False)
    importlib.reload(config)


def test_config_env_can_select_dual_and_live(monkeypatch) -> None:
    monkeypatch.setenv("FINANCEBOT_PAPER", "false")
    monkeypatch.setenv("FINANCEBOT_ENGINE", "dual")
    reloaded = importlib.reload(config)
    assert reloaded.PAPER is False
    assert reloaded.ENGINE == "dual"
    monkeypatch.delenv("FINANCEBOT_PAPER", raising=False)
    monkeypatch.delenv("FINANCEBOT_ENGINE", raising=False)
    importlib.reload(config)


def test_dual_broker_account_position_and_price_methods_are_offline(monkeypatch) -> None:
    broker = _broker(monkeypatch)

    assert broker.get_equity() == 123.45
    assert broker.get_cash() == 67.89
    assert broker.get_all_positions()[0].symbol == "AAPL"
    assert broker.get_open_positions()[0].symbol == "AAPL"
    assert broker.get_position_qty("AAPL") == 0.5
    assert broker.get_last_price("AAPL") == 101.25
    assert broker.get_last_price("BTC/USD") == 101.25
    assert broker.is_available() is True
    assert broker.get_api_error_rate() == 0.0


def test_dual_broker_list_recent_fills_normalizes_activity_payload(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    fills = broker.list_recent_fills(limit=7)

    assert len(fills) == 1
    assert fills[0]["fill_id"] == "fill-1"
    assert fills[0]["order_id"] == "order-1"
    assert fills[0]["symbol"] == "AAPL"
    assert fills[0]["filled_qty"] == 0.25
    assert fills[0]["filled_price"] == 101.5
    assert fills[0]["filled_at"] == datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    client = FakeTradingClient.instances[-1]
    assert client.raw_gets == [("/account/activities/FILL", {"page_size": 7})]


def test_dual_broker_submit_order_intent_supports_market_and_limit(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    client = FakeTradingClient.instances[-1]

    market_id = broker.submit_order_intent(OrderIntentStub(), "cid-market")
    limit_id = broker.submit_order_intent(
        OrderIntentStub(order_style="limit", limit_price=100.12), "cid-limit"
    )

    assert market_id == "fake-order-id"
    assert limit_id == "fake-order-id"
    assert client.submitted_requests[0].client_order_id == "cid-market"
    assert client.submitted_requests[0].symbol == "AAPL"
    assert not hasattr(client.submitted_requests[0], "limit_price")
    assert client.submitted_requests[1].client_order_id == "cid-limit"
    assert client.submitted_requests[1].limit_price == 100.12


def test_dual_broker_open_order_count_and_cancel_are_wrapped(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    client = FakeTradingClient.instances[-1]

    assert len(broker.list_open_orders()) == 2
    assert broker.get_open_order_count() == 2
    assert broker.cancel_all_open_orders() is True
    assert client.cancel_orders_called == 1


def test_dual_broker_failures_degrade_into_health_telemetry(monkeypatch) -> None:
    broker = _broker(monkeypatch)

    FakeTradingClient.fail_next = True
    assert broker.get_equity() is None

    assert broker.is_available() is False
    assert broker.get_api_error_rate() == 1.0

    assert broker.get_cash() == 67.89
    assert broker.is_available() is True
    assert 0.0 < broker.get_api_error_rate() < 1.0



def test_broker_market_clock_open_closed_and_failure(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    assert broker.get_equity_market_open() is True

    FakeTradingClient.clock_open = False
    assert broker.get_equity_market_open() is False

    FakeTradingClient.fail_on = {"clock"}
    assert broker.get_equity_market_open() is None


def test_risk_snapshot_preserves_successful_empty_positions_and_orders(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    FakeTradingClient.positions_payload = []
    FakeTradingClient.orders_payload = []

    snap = broker.get_risk_snapshot()

    assert snap.equity == 123.45
    assert snap.equity_ok is True
    assert snap.positions == []
    assert snap.positions_ok is True
    assert snap.open_orders == []
    assert snap.open_orders_ok is True


def test_risk_snapshot_distinguishes_position_fetch_failure(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    FakeTradingClient.fail_on = {"positions"}
    FakeTradingClient.orders_payload = []

    snap = broker.get_risk_snapshot()

    assert snap.equity_ok is True
    assert snap.positions == []
    assert snap.positions_ok is False
    assert snap.open_orders_ok is True


def test_risk_snapshot_distinguishes_open_order_fetch_failure(monkeypatch) -> None:
    broker = _broker(monkeypatch)
    FakeTradingClient.positions_payload = []
    FakeTradingClient.fail_on = {"orders"}

    snap = broker.get_risk_snapshot()

    assert snap.equity_ok is True
    assert snap.positions_ok is True
    assert snap.open_orders == []
    assert snap.open_orders_ok is False
