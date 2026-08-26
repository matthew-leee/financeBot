"""
Alpaca broker wrapper.

Two jobs, done paranoidly:
  1. Force global rate limiting -- every outbound call goes through _throttle()
     so no code path can burst past Alpaca's limits.
  2. Wrap EVERY API call in strict try/except so a network timeout, HTTP 429,
     or order rejection degrades gracefully instead of crashing the loop.

Secrets are read from Windows environment variables ONLY:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY

Legacy single-horizon methods (get_equity, get_open_positions, get_position_qty,
submit_market_order) are preserved verbatim in behavior. The remaining methods
complete the dual-horizon adapter contract consumed by PortfolioManager and the
TacticalExecutor (get_all_positions, get_cash, get_last_price, list_recent_fills,
submit_order_intent, list_open_orders, get_open_order_count,
cancel_all_open_orders) plus defensive health telemetry (get_api_error_rate,
is_available) fed to the RiskStateMachine.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config

# Rolling window (number of most-recent API calls) used to estimate the live
# API error rate fed to the risk state machine. Small + bounded on purpose.
_API_ERROR_WINDOW: int = 20


@dataclass(frozen=True)
class BrokerRiskSnapshot:
    """
    Immutable, single-pass snapshot of broker truth for the risk gate.

    The ``*_ok`` flags preserve the crucial distinction between a SUCCESSFUL
    fetch that happens to be empty (``positions == []`` with ``positions_ok is
    True``) and a FAILED fetch (``positions == []`` with ``positions_ok is
    False``). A failed snapshot must never be mistaken for a flat account.
    """

    equity: float | None
    positions: list = field(default_factory=list)
    open_orders: list = field(default_factory=list)
    equity_ok: bool = False
    positions_ok: bool = False
    open_orders_ok: bool = False
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _parse_iso(value: object) -> "datetime | None":
    """Best-effort parse of an Alpaca ISO-8601 timestamp into aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


class Broker:
    def __init__(self) -> None:
        # Fail fast + loud if secrets are not present. Never hard-code them.
        try:
            self._key = os.environ["APCA_API_KEY_ID"]
            self._secret = os.environ["APCA_API_SECRET_KEY"]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing env var {exc}. Set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY as Windows environment variables."
            ) from exc

        from alpaca.trading.client import TradingClient

        self._client = TradingClient(self._key, self._secret, paper=config.PAPER)
        self._last_call_ts = 0.0
        # Defensive health telemetry (consumed by the risk state machine).
        self._call_log: deque[int] = deque(maxlen=_API_ERROR_WINDOW)  # 1=error 0=ok
        self._available: bool = True

    # -- rate limiting -------------------------------------------------------

    def _throttle(self) -> None:
        """Block until at least API_CALL_DELAY_SECONDS since the last call."""
        elapsed = time.time() - self._last_call_ts
        wait = config.API_CALL_DELAY_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    # -- health telemetry ----------------------------------------------------

    def _note(self, ok: bool, *, health: bool) -> None:
        """Record a call outcome. `health` calls also flip availability."""
        self._call_log.append(0 if ok else 1)
        if health:
            self._available = ok

    def get_api_error_rate(self) -> float:
        """Fraction of recent API calls that failed (0.0..1.0)."""
        if not self._call_log:
            return 0.0
        return sum(self._call_log) / len(self._call_log)

    def is_available(self) -> bool:
        """True while recent broker-truth (account/positions) calls succeed."""
        return self._available

    # -- account / positions -------------------------------------------------

    def get_equity(self) -> "float | None":
        """Current account equity, or None on failure."""
        self._throttle()
        try:
            acct = self._client.get_account()
            self._note(True, health=True)
            return float(acct.equity)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] get_equity failed: {exc}")
            return None

    def get_cash(self) -> "float | None":
        """Current account cash balance, or None on failure."""
        self._throttle()
        try:
            acct = self._client.get_account()
            self._note(True, health=True)
            return float(acct.cash)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] get_cash failed: {exc}")
            return None

    def get_withdrawable_cash(self) -> "float | None":
        """
        Settled/spendable cash, or None when unknown (callers fail closed).

        Preference ladder, first available wins:
          1. ``cash_withdrawable`` -- explicit withdrawable balance when the
             account payload exposes it,
          2. ``non_marginable_buying_power`` -- on cash accounts this reflects
             SETTLED funds only (T+1 settlement aware),
          3. ``cash`` -- last resort, may include unsettled proceeds.

        On a T+1 cash account this is THE number buys must respect; unsettled
        sell proceeds are invisible here by construction.
        """
        self._throttle()
        try:
            acct = self._client.get_account()
            self._note(True, health=True)
            for attr in ("cash_withdrawable", "non_marginable_buying_power", "cash"):
                try:
                    raw = getattr(acct, attr, None)
                except Exception:  # noqa: BLE001 -- odd payloads degrade to next attr
                    continue
                if raw is None:
                    continue
                val = float(raw)
                if math.isfinite(val):
                    return val
            print("[broker] get_withdrawable_cash: no usable cash attribute.")
            return None
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] get_withdrawable_cash failed: {exc}")
            return None

    def is_cash_account(self) -> "bool | None":
        """True/False when knowable, None when the payload does not say."""
        self._throttle()
        try:
            acct = self._client.get_account()
            self._note(True, health=True)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] is_cash_account failed: {exc}")
            return None

        account_type = str(getattr(acct, "account_type", "") or "").strip().lower()
        if account_type in ("cash", "margin"):
            return account_type == "cash"
        shorting = getattr(acct, "shorting_enabled", None)
        if shorting is None:
            return None
        return not bool(shorting)

    def is_fractionable(self, symbol: str) -> "bool | None":
        """
        Whether the asset supports fractional/notional orders.

        Cached per process (asset rules are effectively static intraday).
        Returns None when unknown -- callers keep their current behavior and
        may still see a broker-side rejection. This pre-check exists because
        several inverse ETFs reject fractional quantities outright
        (e.g. Alpaca 40310000 "asset is not fractionable").
        """
        key = str(symbol).upper()
        cached = getattr(self, "_fractional_cache", None)
        if cached is None:
            cached = {}
            self._fractional_cache = cached
        if key in cached:
            return cached[key]

        self._throttle()
        try:
            asset = self._client.get_asset(key.replace("/", ""))
            raw = getattr(asset, "fractionable", None)
            if raw is None:
                raw = getattr(asset, "fractional_enabled", None)
            result = bool(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001 -- unknown stays unknown
            self._note(False, health=False)
            print(f"[broker] is_fractionable({symbol}) failed: {exc}")
            return None
        self._note(True, health=False)
        # Cache only definite answers; failures stay uncached so a transient
        # network error never poisons the whole day.
        cached[key] = result
        return result

    def get_open_positions(self) -> list:
        """List of open positions; empty list on failure."""
        self._throttle()
        try:
            positions = list(self._client.get_all_positions())
            self._note(True, health=True)
            return positions
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] get_open_positions failed: {exc}")
            return []

    def get_all_positions(self) -> list:
        """Dual-engine alias for get_open_positions() (blueprint contract)."""
        return self.get_open_positions()

    def get_position_qty(self, symbol: str) -> float:
        """Signed quantity held for a symbol (0.0 if flat or on error)."""
        self._throttle()
        try:
            pos = self._client.get_open_position(symbol.replace("/", ""))
            return float(pos.qty)
        except Exception:  # noqa: BLE001 -- "no position" also raises; treat as flat
            return 0.0

    def get_last_price(self, symbol: str) -> "float | None":
        """Latest trade price for a symbol via read-only market data, or None."""
        self._throttle()
        is_crypto = "/" in symbol
        try:
            if is_crypto:
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoLatestTradeRequest

                client = CryptoHistoricalDataClient()
                resp = client.get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=symbol)
                )
            else:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestTradeRequest

                client = StockHistoricalDataClient(
                    api_key=self._key, secret_key=self._secret
                )
                resp = client.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=symbol)
                )
            trade = resp[symbol] if isinstance(resp, dict) else resp
            self._note(True, health=False)
            return float(trade.price)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=False)
            print(f"[broker] get_last_price failed for {symbol}: {exc}")
            return None

    # -- fills ---------------------------------------------------------------

    def list_recent_fills(self, limit: int = 100) -> list:
        """
        Return recent per-fill execution events as normalized dicts.

        Sourced from Alpaca's account activities (FILL) feed, which gives one
        row per (partial) fill with a unique id -- exactly what the portfolio
        manager needs for idempotent, fill-truth FIFO accounting. Each dict is
        pre-shaped to the PortfolioManager.normalize_broker_fill contract.
        """
        self._throttle()
        try:
            raw = self._client.get(
                "/account/activities/FILL", data={"page_size": int(limit)}
            )
            self._note(True, health=True)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] list_recent_fills failed: {exc}")
            return []

        if isinstance(raw, list):
            activities = raw
        elif isinstance(raw, dict):
            activities = raw.get("activities", [])
        else:
            activities = []

        fills: list = []
        for act in activities:
            if isinstance(act, dict):
                get = act.get
            else:
                get = lambda k, d=None, _a=act: getattr(_a, k, d)  # noqa: E731
            try:
                fills.append(
                    {
                        "fill_id": str(get("id")),
                        "order_id": str(get("order_id") or ""),
                        "symbol": str(get("symbol")),
                        "side": str(get("side") or "buy").lower(),
                        "filled_qty": float(get("qty") or 0.0),
                        "filled_price": float(get("price") or 0.0),
                        "filled_at": _parse_iso(get("transaction_time")),
                        "fees": 0.0,
                        "liquidity_flag": get("type"),
                    }
                )
            except (TypeError, ValueError) as exc:
                print(f"[broker] skipping malformed fill activity: {exc}")
                continue
        return fills

    # -- order routing -------------------------------------------------------

    def submit_market_order(self, symbol: str, qty: float, side: str) -> bool:
        """
        Submit a market order. Returns True on apparent success, False otherwise.
        Any exception (429, timeout, rejection) is swallowed and logged.
        """
        if qty <= 0:
            print(f"[broker] Refusing zero/negative qty for {symbol}.")
            return False

        self._throttle()
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            # Crypto trades ~24/7 with GTC; equities use DAY. Keep it simple.
            tif = TimeInForce.GTC if "/" in symbol else TimeInForce.DAY

            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            order = self._client.submit_order(req)
            self._note(True, health=False)
            print(f"[broker] {side.upper()} {qty} {symbol} -> id={order.id}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=False)
            print(f"[broker] submit_market_order failed for {symbol}: {exc}")
            return False

    def submit_order_intent(self, intent: object, client_order_id: str) -> "str | None":
        """
        Route a dual-engine OrderIntent to Alpaca and return the broker order id.

        The intent quantity is already hard-capped to MAX_POSITION_SIZE upstream
        in the tactical executor (convert_weight_delta_to_order_intent). This
        method just translates the intent into the correct Alpaca request and
        tags it with the caller's client_order_id for reconciliation.
        """
        symbol = getattr(intent, "symbol", None)
        side = str(getattr(intent, "side", "")).lower()
        qty = float(getattr(intent, "quantity", 0.0) or 0.0)
        order_style = str(getattr(intent, "order_style", "market") or "market").lower()
        limit_price = getattr(intent, "limit_price", None)

        if not symbol or qty <= 0 or side not in {"buy", "sell"}:
            print(f"[broker] Refusing invalid order intent for {symbol!r}.")
            return None

        self._throttle()
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            tif = TimeInForce.GTC if "/" in symbol else TimeInForce.DAY

            if order_style == "limit" and limit_price is not None:
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=float(limit_price),
                    client_order_id=client_order_id,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    client_order_id=client_order_id,
                )
            order = self._client.submit_order(req)
            self._note(True, health=False)
            print(f"[broker] intent {side.upper()} {qty} {symbol} -> id={order.id}")
            return str(order.id)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=False)
            print(f"[broker] submit_order_intent failed for {symbol}: {exc}")
            return None

    # -- open-order management ----------------------------------------------

    def list_open_orders(self) -> list:
        """List currently open orders; empty list on failure."""
        self._throttle()
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = list(self._client.get_orders(filter=req))
            self._note(True, health=True)
            return orders
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] list_open_orders failed: {exc}")
            return []

    def get_open_order_count(self) -> int:
        """Number of currently open orders (0 on failure)."""
        return len(self.list_open_orders())

    def cancel_all_open_orders(self) -> bool:
        """
        Cancel every open order (used by the risk kill switch). Returns True on
        apparent success, False otherwise. Never raises into the caller.
        """
        self._throttle()
        try:
            self._client.cancel_orders()
            self._note(True, health=False)
            print("[broker] cancelled all open orders.")
            return True
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=False)
            print(f"[broker] cancel_all_open_orders failed: {exc}")
            return False


    # -- market-hours gating -------------------------------------------------

    def get_equity_market_open(self) -> "bool | None":
        """
        Return the equity market open/closed state via Alpaca's trading clock.

        * True  -> equity market is open.
        * False -> Alpaca reports it closed.
        * None  -> the clock request failed (caller must fail closed for equities).

        Fetched once per loop pass, never once per symbol. Recorded in broker
        health telemetry like any other API call.
        """
        self._throttle()
        try:
            clock = self._client.get_clock()
            self._note(True, health=False)
            return bool(clock.is_open)
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=False)
            print(f"[broker] get_equity_market_open failed: {exc}")
            return None

    # -- trusted single-pass risk snapshot -----------------------------------

    def get_risk_snapshot(self) -> "BrokerRiskSnapshot":
        """
        Fetch equity, positions, and open orders ONCE for the current pass.

        Every call is throttled and wrapped. Success-empty is preserved as
        distinct from retrieval failure via the per-field ``*_ok`` flags. Equity
        is validated finite; a non-finite equity is treated as unknown.
        """
        equity: "float | None" = None
        equity_ok = False
        self._throttle()
        try:
            acct = self._client.get_account()
            self._note(True, health=True)
            raw_equity = float(acct.equity)
            if math.isfinite(raw_equity):
                equity = raw_equity
                equity_ok = True
            else:
                print("[broker] snapshot: non-finite equity; treating as unknown.")
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] snapshot equity failed: {exc}")

        positions: list = []
        positions_ok = False
        self._throttle()
        try:
            positions = list(self._client.get_all_positions())
            self._note(True, health=True)
            positions_ok = True
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] snapshot positions failed: {exc}")
            positions = []

        open_orders: list = []
        open_orders_ok = False
        self._throttle()
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = list(self._client.get_orders(filter=req))
            self._note(True, health=True)
            open_orders_ok = True
        except Exception as exc:  # noqa: BLE001
            self._note(False, health=True)
            print(f"[broker] snapshot open orders failed: {exc}")
            open_orders = []

        return BrokerRiskSnapshot(
            equity=equity,
            positions=positions,
            open_orders=open_orders,
            equity_ok=equity_ok,
            positions_ok=positions_ok,
            open_orders_ok=open_orders_ok,
            fetched_at=datetime.now(timezone.utc),
        )
