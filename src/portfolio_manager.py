"""
Portfolio Manager -- broker truth, fills, FIFO inventory, and reconciliation.

This module is the ONLY live writer of orders, fills, and inventory state in the
dual-horizon engine. Its defining rule: live PnL and FIFO lots are derived from
actual BrokerFill events, never from requested order size or a stale bar close.
That fixes four legacy bugs at once:

  * FIFO based on requested size instead of actual fill.
  * Trade log using stale bar close instead of the true execution price.
  * No partial-fill handling.
  * No broker/internal reconciliation, slippage, or fee accounting.

Fills are applied exactly once (idempotent by fill_id) so a broker that replays
recent fills can never double-count inventory or PnL.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import config
from src.fifo import FIFOInventory

_DUST_THRESHOLD = 1e-6

ORDERS_LOG_COLUMNS: list[str] = [
    "submitted_at",
    "client_order_id",
    "broker_order_id",
    "symbol",
    "side",
    "requested_qty",
    "order_type",
    "limit_price",
    "reduce_only",
    "reason",
    "status",
]

FILLS_LOG_COLUMNS: list[str] = [
    "filled_at",
    "fill_id",
    "broker_order_id",
    "symbol",
    "side",
    "actual_qty",
    "actual_price",
    "fees",
    "realized_pnl",
    "arrival_price",
    "slippage_bps",
    "liquidity_flag",
]


@dataclass(frozen=True)
class BrokerFill:
    """Actual broker execution event."""

    fill_id: str
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    filled_qty: float
    filled_price: float
    filled_at: datetime
    fees: float
    liquidity_flag: str | None = None


@dataclass(frozen=True)
class BrokerPosition:
    """Broker-reported position."""

    symbol: str
    quantity: float
    market_value: float
    avg_entry_price: float
    unrealized_pnl: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Reconciled portfolio state."""

    as_of: datetime
    equity: float
    cash: float
    positions: dict[str, BrokerPosition]
    internal_weights: dict[str, float]
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float

    def weight(self, symbol: str) -> float:
        """Return market_value / equity for symbol, or 0 if flat."""
        pos = self.positions.get(symbol)
        if pos is None or self.equity <= 0:
            return 0.0
        return pos.market_value / self.equity


@dataclass(frozen=True)
class ReconciliationDiff:
    """Difference between broker truth and internal FIFO state."""

    symbol: str
    broker_qty: float
    internal_qty: float
    qty_diff: float
    broker_avg_price: float | None
    internal_avg_price: float | None
    severity: Literal["none", "minor", "major"]


def _to_utc(ts: datetime | None) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _get(obj: object, name: str, default=None):
    """Read attribute or dict key from a broker payload."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PortfolioManager:
    """Owns order lifecycle, fill processing, FIFO inventory, and reconciliation."""

    def __init__(
        self,
        *,
        broker: object,
        fifo_inventory: FIFOInventory | None = None,
        state_path: str | None = None,
        orders_log_path: str | None = None,
        fills_log_path: str | None = None,
    ) -> None:
        self.broker = broker
        self.fifo = fifo_inventory or FIFOInventory()
        self.state_path = state_path or config.PORTFOLIO_STATE_PATH
        self.orders_log_path = orders_log_path or config.ORDERS_LOG_PATH
        self.fills_log_path = fills_log_path or config.FILLS_LOG_PATH

        self.processed_fill_ids: set[str] = set()
        self.realized_pnl: float = 0.0
        # order_id -> arrival/reference price for slippage attribution.
        self._order_ref_price: dict[str, float] = {}
        self._load_state()

    # -- order submission ----------------------------------------------------

    def submit_order_intent(self, intent: object) -> str | None:
        """Submit an order intent to the broker and log it."""
        quantity = float(_get(intent, "quantity", 0.0))
        if quantity <= 0:
            print("[pm] Refusing order intent with non-positive quantity.")
            return None

        symbol = _get(intent, "symbol")
        side = _get(intent, "side")
        reduce_only = bool(_get(intent, "reduce_only", False))

        if reduce_only:
            broker_qty = self._broker_position_qty(symbol)
            if abs(broker_qty) <= _DUST_THRESHOLD:
                print(f"[pm] reduce_only order for flat {symbol}; skipping.")
                return None

        client_order_id = str(uuid.uuid4())
        order_style = _get(intent, "order_style", "market")
        limit_price = _get(intent, "limit_price")
        reason = _get(intent, "reason", "")

        broker_order_id = self._broker_submit(intent, client_order_id)
        status = "submitted" if broker_order_id else "rejected"

        # Remember the reference price for later slippage math.
        ref_price = limit_price if limit_price else self._reference_price(symbol)
        if broker_order_id and ref_price:
            self._order_ref_price[str(broker_order_id)] = float(ref_price)
            self._order_ref_price[client_order_id] = float(ref_price)

        self._append_row(
            self.orders_log_path,
            ORDERS_LOG_COLUMNS,
            {
                "submitted_at": _to_utc(None).isoformat(),
                "client_order_id": client_order_id,
                "broker_order_id": broker_order_id or "",
                "symbol": symbol,
                "side": side,
                "requested_qty": round(quantity, 6),
                "order_type": order_style,
                "limit_price": limit_price if limit_price is not None else "",
                "reduce_only": reduce_only,
                "reason": reason,
                "status": status,
            },
        )
        self._save_state()
        return broker_order_id

    # -- fill processing -----------------------------------------------------

    def poll_and_apply_fills(self) -> list[BrokerFill]:
        """Poll the broker for new fills and apply each exactly once."""
        try:
            raw_fills = list(self.broker.list_recent_fills())
        except Exception as exc:  # noqa: BLE001 -- polling must not kill the loop
            print(f"[pm] Failed to list recent fills: {exc}")
            return []

        applied: list[BrokerFill] = []
        for raw in raw_fills:
            fill = self.normalize_broker_fill(raw)
            if fill.fill_id in self.processed_fill_ids:
                continue  # idempotency: never double-count a replayed fill
            self.apply_fill(fill)
            self.processed_fill_ids.add(fill.fill_id)
            applied.append(fill)

        if applied:
            self._save_state()
        return applied

    def normalize_broker_fill(self, raw: object) -> BrokerFill:
        """Coerce a raw broker fill payload into a BrokerFill."""
        if isinstance(raw, BrokerFill):
            return raw
        side = str(_get(raw, "side", "buy")).lower()
        return BrokerFill(
            fill_id=str(_get(raw, "fill_id") or _get(raw, "id")),
            order_id=str(_get(raw, "order_id") or _get(raw, "broker_order_id") or ""),
            symbol=str(_get(raw, "symbol")),
            side="buy" if side == "buy" else "sell",
            filled_qty=float(_get(raw, "filled_qty") or _get(raw, "qty") or 0.0),
            filled_price=float(_get(raw, "filled_price") or _get(raw, "price") or 0.0),
            filled_at=_to_utc(_get(raw, "filled_at") or _get(raw, "timestamp")),
            fees=float(_get(raw, "fees") or 0.0),
            liquidity_flag=_get(raw, "liquidity_flag"),
        )

    def apply_fill(self, fill: BrokerFill) -> float:
        """Apply an actual fill to FIFO, write the fill log, return realized PnL."""
        if fill.side == "buy":
            self.fifo.add_buy(fill.symbol, fill.filled_qty, fill.filled_price)
            realized_pnl = -fill.fees
        else:
            gross = self.fifo.add_sell(fill.symbol, fill.filled_qty, fill.filled_price)
            realized_pnl = gross - fill.fees

        # Slippage vs. the order arrival/reference price (never a stale bar).
        arrival_price = self._order_ref_price.get(fill.order_id)
        slippage_bps: float | str = ""
        if arrival_price and arrival_price > 0:
            if fill.side == "buy":
                slippage = fill.filled_price - arrival_price
            else:
                slippage = arrival_price - fill.filled_price
            slippage_bps = slippage / arrival_price * 10000.0

        self.realized_pnl += realized_pnl
        self._append_row(
            self.fills_log_path,
            FILLS_LOG_COLUMNS,
            {
                "filled_at": _to_utc(fill.filled_at).isoformat(),
                "fill_id": fill.fill_id,
                "broker_order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side,
                "actual_qty": round(fill.filled_qty, 6),
                "actual_price": round(fill.filled_price, 6),
                "fees": round(fill.fees, 6),
                "realized_pnl": round(realized_pnl, 6),
                "arrival_price": round(arrival_price, 6) if arrival_price else "",
                "slippage_bps": (
                    round(slippage_bps, 4)
                    if isinstance(slippage_bps, float)
                    else ""
                ),
                "liquidity_flag": fill.liquidity_flag or "",
            },
        )
        return realized_pnl

    # -- reconciliation ------------------------------------------------------

    def reconcile_with_broker(self) -> list[ReconciliationDiff]:
        """Compare broker positions to internal FIFO positions."""
        broker_positions = self._broker_positions()
        internal = self.fifo.positions()

        all_symbols = set(broker_positions) | set(internal)
        minor_threshold = max(config.MAX_RECONCILIATION_QTY_DIFF, _DUST_THRESHOLD)

        diffs: list[ReconciliationDiff] = []
        for symbol in sorted(all_symbols):
            bpos = broker_positions.get(symbol)
            broker_qty = float(bpos.quantity) if bpos else 0.0
            internal_qty = float(internal.get(symbol, {}).get("open_qty", 0.0))
            qty_diff = broker_qty - internal_qty
            abs_diff = abs(qty_diff)

            if abs_diff <= _DUST_THRESHOLD:
                severity: Literal["none", "minor", "major"] = "none"
            elif abs_diff <= minor_threshold:
                severity = "minor"
            else:
                severity = "major"

            diffs.append(
                ReconciliationDiff(
                    symbol=symbol,
                    broker_qty=broker_qty,
                    internal_qty=internal_qty,
                    qty_diff=qty_diff,
                    broker_avg_price=(float(bpos.avg_entry_price) if bpos else None),
                    internal_avg_price=(
                        float(internal.get(symbol, {}).get("avg_price", 0.0))
                        if symbol in internal
                        else None
                    ),
                    severity=severity,
                )
            )

        majors = [d for d in diffs if d.severity == "major"]
        if majors:
            # Never fabricate FIFO lots silently -- surface the break loudly.
            print(
                f"[pm] MAJOR reconciliation break on: "
                f"{', '.join(d.symbol for d in majors)}"
            )
        return diffs

    def major_break_count(self, diffs: list[ReconciliationDiff]) -> int:
        return sum(1 for d in diffs if d.severity == "major")

    # -- snapshot ------------------------------------------------------------

    def snapshot(self) -> PortfolioSnapshot:
        """Return a broker-truth portfolio snapshot."""
        positions = self._broker_positions()
        equity = self._broker_equity()
        if equity is None:
            # Missing broker equity while positions exist is a risk event; fall
            # back to summed market value so downstream weights stay sane.
            equity = sum(p.market_value for p in positions.values())
        cash = self._broker_cash()
        if cash is None:
            cash = equity - sum(p.market_value for p in positions.values())

        gross = sum(abs(p.market_value) for p in positions.values())
        net = sum(p.market_value for p in positions.values())
        unrealized = sum(p.unrealized_pnl for p in positions.values())
        internal_weights = {
            sym: (p.market_value / equity if equity else 0.0)
            for sym, p in positions.items()
        }

        return PortfolioSnapshot(
            as_of=_to_utc(None),
            equity=float(equity),
            cash=float(cash),
            positions=positions,
            internal_weights=internal_weights,
            gross_exposure=(gross / equity if equity else 0.0),
            net_exposure=(net / equity if equity else 0.0),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=float(unrealized),
        )

    def current_position_symbols(self) -> set[str]:
        """Symbols with a nonzero broker position."""
        return {
            sym
            for sym, p in self._broker_positions().items()
            if abs(p.quantity) > _DUST_THRESHOLD
        }

    def internal_fifo_symbols(self) -> set[str]:
        """Symbols with nonzero FIFO inventory."""
        return {
            sym
            for sym, info in self.fifo.positions().items()
            if abs(info.get("open_qty", 0.0)) > _DUST_THRESHOLD
        }

    # -- broker adapters (defensive/getattr-based) ---------------------------

    def _broker_submit(self, intent: object, client_order_id: str) -> str | None:
        try:
            if hasattr(self.broker, "submit_order_intent"):
                result = self.broker.submit_order_intent(intent, client_order_id)
            elif hasattr(self.broker, "submit_order"):
                result = self.broker.submit_order(intent)
            else:
                ok = self.broker.submit_market_order(
                    _get(intent, "symbol"),
                    float(_get(intent, "quantity", 0.0)),
                    str(_get(intent, "side")),
                )
                result = client_order_id if ok else None
            if result is None or result is False:
                return None
            if result is True:
                return client_order_id
            return str(_get(result, "id", result))
        except Exception as exc:  # noqa: BLE001 -- broker errors degrade gracefully
            print(f"[pm] Order submission failed: {exc}")
            return None

    def _broker_positions(self) -> dict[str, BrokerPosition]:
        try:
            raw_positions = list(self.broker.get_all_positions())
        except Exception as exc:  # noqa: BLE001
            print(f"[pm] Failed to fetch broker positions: {exc}")
            return {}
        out: dict[str, BrokerPosition] = {}
        for raw in raw_positions:
            if isinstance(raw, BrokerPosition):
                out[raw.symbol] = raw
                continue
            symbol = str(_get(raw, "symbol"))
            qty = float(_get(raw, "qty") or _get(raw, "quantity") or 0.0)
            market_value = float(
                _get(raw, "market_value")
                if _get(raw, "market_value") is not None
                else qty * float(_get(raw, "current_price") or 0.0)
            )
            out[symbol] = BrokerPosition(
                symbol=symbol,
                quantity=qty,
                market_value=market_value,
                avg_entry_price=float(_get(raw, "avg_entry_price") or 0.0),
                unrealized_pnl=float(_get(raw, "unrealized_pl") or _get(raw, "unrealized_pnl") or 0.0),
            )
        return out

    def _broker_position_qty(self, symbol: str) -> float:
        pos = self._broker_positions().get(symbol)
        return float(pos.quantity) if pos else 0.0

    def _broker_equity(self) -> float | None:
        for name in ("get_equity", "equity"):
            fn = getattr(self.broker, name, None)
            if callable(fn):
                try:
                    val = fn()
                    return float(val) if val is not None else None
                except Exception:  # noqa: BLE001
                    return None
        return None

    def _broker_cash(self) -> float | None:
        fn = getattr(self.broker, "get_cash", None)
        if callable(fn):
            try:
                val = fn()
                return float(val) if val is not None else None
            except Exception:  # noqa: BLE001
                return None
        return None

    def _reference_price(self, symbol: str) -> float | None:
        fn = getattr(self.broker, "get_last_price", None)
        if callable(fn):
            try:
                return float(fn(symbol))
            except Exception:  # noqa: BLE001
                return None
        pos = self._broker_positions().get(symbol)
        if pos and pos.quantity:
            return pos.market_value / pos.quantity
        return None

    # -- persistence ---------------------------------------------------------

    def _append_row(self, path: str, columns: list[str], row: dict) -> None:
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            file_exists = os.path.exists(path) and os.path.getsize(path) > 0
            with open(path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:  # noqa: BLE001 -- logging must not kill the loop
            print(f"[pm] Failed to append log row to {path}: {exc}")

    def _load_state(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.processed_fill_ids = set(data.get("processed_fill_ids", []))
            self.realized_pnl = float(data.get("realized_pnl", 0.0))
            self._order_ref_price = {
                k: float(v) for k, v in data.get("order_ref_price", {}).items()
            }
            if "fifo" in data:
                self.fifo = FIFOInventory.from_dict(data["fifo"])
        except Exception:  # noqa: BLE001 -- corrupt state -> start conservative
            pass

    def _save_state(self) -> None:
        try:
            directory = os.path.dirname(self.state_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "processed_fill_ids": sorted(self.processed_fill_ids),
                        "realized_pnl": self.realized_pnl,
                        "order_ref_price": self._order_ref_price,
                        "fifo": self.fifo.to_dict(),
                    },
                    fh,
                    indent=2,
                )
        except Exception as exc:  # noqa: BLE001 -- persistence must not crash run
            print(f"[pm] Failed to persist portfolio state: {exc}")
