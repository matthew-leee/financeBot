"""
Tactical Executor -- continuous intraday execution inside strategist envelopes.

The executor is the *fast* brain. It consumes the daily AllocationMatrix emitted
by the strategist and nudges real positions toward each symbol''s target weight,
timing entries/exits with intraday microstructure features. It is deliberately
subordinate to the strategist and the risk machine:

  * It may trade only inside the strategist''s per-symbol envelopes.
  * It re-applies the hard MAX_POSITION_SIZE cap on every order -- model logic
    can never widen the per-order notional guardrail.
  * Its processing universe always includes core, hedge, allocation, broker, and
    internal FIFO symbols, so orphaned hedges are always seen and exited.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd

import config
from src.guardrails import RiskState
from src.strategist import AllocationMatrix

TacticalAction = Literal["buy", "sell", "hold"]
OrderStyle = Literal["market", "limit", "notional"]

_EPS = 1e-12
_REQUIRED_LIVE_MODEL_ROLES = ("expected_return", "regime")


@dataclass(frozen=True)
class TacticalSignal:
    """Intraday execution timing signal."""

    symbol: str
    action: TacticalAction
    urgency: float
    confidence: float
    limit_price: float | None
    reason: str


@dataclass(frozen=True)
class OrderIntent:
    """Pre-broker order request (positive quantity; risk-checked upstream)."""

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_style: OrderStyle
    limit_price: float | None
    reduce_only: bool
    target_weight_after: float
    reason: str


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


def compute_tactical_signal(
    *,
    symbol: str,
    intraday_features: pd.Series,
) -> TacticalSignal:
    """Convert intraday microstructure features into execution timing."""

    def _feat(name: str, default: float) -> float:
        try:
            val = float(intraday_features.get(name, default))
        except (TypeError, ValueError):
            return default
        return default if pd.isna(val) else val

    spread_bps = _feat("spread_bps", 0.0)
    vwap_distance = _feat("vwap_distance", 0.0)
    order_flow = _feat("order_flow_imbalance", 0.0)
    volume_pct = _feat("volume_percentile", 0.5)

    if spread_bps > config.MAX_SPREAD_BPS:
        return TacticalSignal(
            symbol=symbol,
            action="hold",
            urgency=0.0,
            confidence=1.0,
            limit_price=None,
            reason="spread too wide",
        )

    spread_z = spread_bps / max(config.MAX_SPREAD_BPS, _EPS)
    buy_score = (
        -0.25 * vwap_distance
        + 0.20 * order_flow
        + 0.15 * volume_pct
        - 0.10 * spread_z
    )
    sell_score = (
        0.25 * vwap_distance
        - 0.20 * order_flow
        + 0.15 * volume_pct
        - 0.10 * spread_z
    )
    threshold = 0.05

    if buy_score > threshold and buy_score >= sell_score:
        return TacticalSignal(
            symbol=symbol,
            action="buy",
            urgency=_sigmoid(buy_score),
            confidence=min(1.0, 0.5 + abs(buy_score)),
            limit_price=None,
            reason="favorable buy microstructure",
        )
    if sell_score > threshold:
        return TacticalSignal(
            symbol=symbol,
            action="sell",
            urgency=_sigmoid(sell_score),
            confidence=min(1.0, 0.5 + abs(sell_score)),
            limit_price=None,
            reason="favorable sell microstructure",
        )
    return TacticalSignal(
        symbol=symbol,
        action="hold",
        urgency=0.0,
        confidence=1.0,
        limit_price=None,
        reason="no timing edge",
    )


def convert_weight_delta_to_order_intent(
    *,
    symbol: str,
    delta_weight: float,
    snapshot: object,
    latest_price: float,
    signal: TacticalSignal,
) -> OrderIntent | None:
    """Convert a desired weight change into a hard-capped order intent."""
    from src.guardrails import clamp_position_size

    if latest_price is None or latest_price <= 0 or math.isnan(latest_price):
        return None
    equity = float(getattr(snapshot, "equity", 0.0))
    if equity <= 0:
        return None

    desired_notional = abs(delta_weight) * equity
    # MANDATORY: re-apply the hard per-order notional cap. Strategist weights are
    # model logic and may never override MAX_POSITION_SIZE.
    capped_notional = min(desired_notional, config.MAX_POSITION_SIZE)
    quantity = clamp_position_size(capped_notional, latest_price)
    if quantity <= 0:
        return None

    side: Literal["buy", "sell"] = "buy" if delta_weight > 0 else "sell"
    current_weight = snapshot.weight(symbol)
    sign = 1.0 if delta_weight > 0 else -1.0
    target_weight_after = current_weight + sign * capped_notional / equity
    reduce_only = abs(target_weight_after) < abs(current_weight)

    return OrderIntent(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_style=signal.limit_price and "limit" or "market",
        limit_price=signal.limit_price,
        reduce_only=reduce_only,
        target_weight_after=target_weight_after,
        reason=f"{signal.reason} | delta_w={delta_weight:.4f}",
    )


class TacticalExecutor:
    """Continuous intraday executor bound to strategist + risk envelopes."""

    def __init__(
        self,
        *,
        strategist: object,
        portfolio_manager: object,
        feature_store: object,
        risk_machine: object,
        broker: object,
        loop_interval_seconds: float = config.LOOP_INTERVAL_SECONDS,
        allocation_refresh_interval: timedelta | None = None,
    ) -> None:
        self.strategist = strategist
        self.portfolio_manager = portfolio_manager
        self.feature_store = feature_store
        self.risk_machine = risk_machine
        self.broker = broker
        self.loop_interval_seconds = loop_interval_seconds
        self.allocation_refresh_interval = allocation_refresh_interval or timedelta(
            seconds=config.ALLOCATION_STALE_SECONDS
        )
        self._allocation: AllocationMatrix | None = None

    # -- main loop -----------------------------------------------------------

    def run_forever(self) -> None:
        """Run the continuous execution loop until a kill decision is issued."""
        while True:
            now = datetime.now(timezone.utc)
            self.run_once(now=now)
            time.sleep(self.loop_interval_seconds)

    def run_once(self, *, now: datetime) -> list[OrderIntent]:
        """One full execution pass. Returns the order intents that were submitted."""
        pm = self.portfolio_manager
        pm.poll_and_apply_fills()
        reconciliation = pm.reconcile_with_broker()
        snapshot = pm.snapshot()

        metrics = self.build_risk_metrics(snapshot, reconciliation, now=now)
        risk_decision = self.risk_machine.evaluate(metrics)

        if risk_decision.kill_process:
            self._cancel_all_open_orders()
            raise SystemExit(1)

        allocation = self.refresh_allocation_if_due(now=now)
        symbols = self.build_processing_universe(allocation)

        submitted: list[OrderIntent] = []
        for symbol in sorted(symbols):
            try:
                intent = self.process_symbol(
                    symbol=symbol,
                    allocation=allocation,
                    risk_decision=risk_decision,
                    now=now,
                )
                if intent is not None:
                    submitted.append(intent)
            except Exception as exc:  # noqa: BLE001 -- one symbol must not kill loop
                print(f"[executor] error processing {symbol}: {exc}")
                continue
        return submitted

    def build_risk_metrics(self, snapshot, reconciliation, *, now: datetime):
        """Assemble live RiskMetrics from portfolio + dependency telemetry."""
        from src.guardrails import RiskMetrics

        majors = sum(1 for d in reconciliation if getattr(d, "severity", "none") == "major")
        equity = float(getattr(snapshot, "equity", 0.0))
        hwm = self._high_watermark_equity(equity)
        drawdown = (equity / hwm - 1.0) if hwm > _EPS else 0.0
        largest = 0.0
        weights = getattr(snapshot, "internal_weights", {}) or {}
        if weights:
            largest = max(abs(w) for w in weights.values())
        return RiskMetrics(
            as_of=now,
            equity=equity,
            high_watermark_equity=hwm,
            rolling_24h_pnl=self._rolling_pnl(snapshot),
            drawdown=drawdown,
            gross_exposure=float(getattr(snapshot, "gross_exposure", 0.0)),
            net_exposure=float(getattr(snapshot, "net_exposure", 0.0)),
            largest_symbol_weight=largest,
            data_staleness_seconds=self._data_staleness_seconds(snapshot, now),
            major_reconciliation_breaks=majors,
            open_order_count=self._open_order_count(),
            api_error_rate=self._api_error_rate(),
            model_artifacts_valid=self._model_artifacts_valid(),
            broker_available=self._broker_available(snapshot),
            positions_open=bool(getattr(snapshot, "positions", {})),
        )

    def _high_watermark_equity(self, equity: float) -> float:
        existing = getattr(self.risk_machine, "high_watermark_equity", None)
        try:
            existing_hwm = float(existing) if existing is not None else 0.0
        except (TypeError, ValueError):
            existing_hwm = 0.0
        return max(existing_hwm, equity, 0.0)

    def _rolling_pnl(self, snapshot) -> float:
        realized = float(getattr(snapshot, "realized_pnl", 0.0) or 0.0)
        unrealized = float(getattr(snapshot, "unrealized_pnl", 0.0) or 0.0)
        return realized + unrealized

    def _data_staleness_seconds(self, snapshot, now: datetime) -> int:
        candidates: list[datetime] = []
        if self._allocation is not None:
            self._append_timestamp(candidates, getattr(self._allocation, "as_of", None))

        latest_data_at = self._latest_feature_store_timestamp(now)
        self._append_timestamp(candidates, latest_data_at)
        if not candidates:
            return 0
        newest = max(candidates)
        return max(0, int((self._to_utc(now) - newest).total_seconds()))

    def _latest_feature_store_timestamp(self, now: datetime) -> datetime | None:
        for name in ("latest_available_at", "max_available_at", "last_updated_at"):
            attr = getattr(self.feature_store, name, None)
            try:
                value = attr() if callable(attr) else attr
            except Exception:  # noqa: BLE001 -- telemetry must not kill trading loop
                continue
            ts = self._coerce_timestamp(value)
            if ts is not None:
                return ts

        records = getattr(self.feature_store, "_by_key", None)
        if isinstance(records, dict):
            eligible: list[datetime] = []
            now_utc = self._to_utc(now)
            for record in records.values():
                available_at = self._coerce_timestamp(getattr(record, "available_at", None))
                if available_at is not None and available_at <= now_utc:
                    eligible.append(available_at)
            if eligible:
                return max(eligible)
        return None

    def _open_order_count(self) -> int:
        value = self._read_broker_value(("open_order_count", "get_open_order_count"))
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
        list_open_orders = getattr(self.broker, "list_open_orders", None)
        if callable(list_open_orders):
            try:
                return len(list(list_open_orders()))
            except Exception:  # noqa: BLE001 -- telemetry failure handled by api error rate
                return 0
        return 0

    def _api_error_rate(self) -> float:
        value = self._read_broker_value(("api_error_rate", "get_api_error_rate"))
        if value is None:
            value = getattr(self.portfolio_manager, "api_error_rate", 0.0)
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _model_artifacts_valid(self) -> bool:
        registry = getattr(self.strategist, "model_registry", None)
        if registry is None:
            return True
        verify = getattr(registry, "verify", None)
        if not callable(verify):
            return True
        try:
            verify(_REQUIRED_LIVE_MODEL_ROLES)
            return True
        except Exception as exc:  # noqa: BLE001 -- invalid artifacts are a risk input
            print(f"[executor] model registry invalid: {exc}")
            return False

    def _broker_available(self, snapshot) -> bool:
        value = self._read_broker_value(("broker_available", "is_available", "health_check"))
        if value is not None:
            return bool(value)
        value = getattr(self.portfolio_manager, "broker_available", None)
        if value is not None:
            return bool(value() if callable(value) else value)
        return float(getattr(snapshot, "equity", 0.0) or 0.0) > 0.0

    def _read_broker_value(self, names: tuple[str, ...]):
        for name in names:
            attr = getattr(self.broker, name, None)
            if attr is None:
                continue
            try:
                return attr() if callable(attr) else attr
            except Exception:  # noqa: BLE001 -- telemetry failure should degrade safely
                continue
        return None

    def _append_timestamp(self, out: list[datetime], value) -> None:
        ts = self._coerce_timestamp(value)
        if ts is not None:
            out.append(ts)

    def _coerce_timestamp(self, value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return self._to_utc(value)
        return None

    def _to_utc(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def refresh_allocation_if_due(self, *, now: datetime) -> AllocationMatrix:
        """Refresh the daily allocation when missing or stale."""
        needs_refresh = self._allocation is None
        if not needs_refresh and self._allocation is not None:
            age = now - self._allocation.as_of
            if age >= self.allocation_refresh_interval:
                needs_refresh = True
        if needs_refresh:
            self._allocation = self.strategist.generate_allocation(as_of=now)
        return self._allocation

    def build_processing_universe(self, allocation: AllocationMatrix | None) -> set[str]:
        """Build the complete execution universe (prevents orphaned hedges)."""
        symbols: set[str] = set(config.CORE_UNIVERSE) | set(config.HEDGE_UNIVERSE)
        if allocation is not None:
            symbols |= allocation.symbols()
        symbols |= self.portfolio_manager.current_position_symbols()
        symbols |= self.portfolio_manager.internal_fifo_symbols()
        return symbols

    def process_symbol(
        self,
        *,
        symbol: str,
        allocation: AllocationMatrix | None,
        risk_decision: object,
        now: datetime,
    ) -> OrderIntent | None:
        """Process one symbol toward its allocation target inside risk rules."""
        target = allocation.get(symbol) if allocation is not None else None
        if target is None:
            target_weight = 0.0
            min_weight = 0.0
            max_weight = 0.0
        else:
            target_weight = target.target_weight
            min_weight = target.min_weight
            max_weight = target.max_weight

        snapshot = self.portfolio_manager.snapshot()
        current_weight = snapshot.weight(symbol)

        intraday = self.feature_store.build_intraday_snapshot(
            symbols=[symbol],
            as_of=now,
            lookback_minutes=config.INTRADAY_LOOKBACK_MINUTES,
        )
        if symbol not in intraday.frame.index:
            return None
        features = intraday.frame.loc[symbol]
        signal = compute_tactical_signal(symbol=symbol, intraday_features=features)

        desired_delta_weight = target_weight - current_weight

        if current_weight < min_weight or current_weight > max_weight:
            urgency = 1.0  # out of envelope -> act regardless of timing edge
        else:
            urgency = signal.urgency

        allowed_delta = self.apply_risk_state_rules(
            desired_delta_weight=desired_delta_weight,
            current_weight=current_weight,
            target_weight=target_weight,
            risk_decision=risk_decision,
        )
        allowed_delta *= urgency * signal.confidence

        if abs(allowed_delta) < config.MINIMUM_TRADE_WEIGHT:
            return None

        latest_price = float(features.get("last_price", float("nan")))
        intent = convert_weight_delta_to_order_intent(
            symbol=symbol,
            delta_weight=allowed_delta,
            snapshot=snapshot,
            latest_price=latest_price,
            signal=signal,
        )
        if intent is None:
            return None

        if not self.risk_machine.is_order_allowed(
            state=getattr(risk_decision, "state", RiskState.NORMAL),
            current_weight=current_weight,
            target_weight_after=intent.target_weight_after,
            side=intent.side,
            reduce_only=intent.reduce_only,
        ):
            return None

        self.portfolio_manager.submit_order_intent(intent)
        return intent

    def apply_risk_state_rules(
        self,
        *,
        desired_delta_weight: float,
        current_weight: float,
        target_weight: float,
        risk_decision: object,
    ) -> float:
        """Clamp the desired weight delta to what the risk state permits."""
        if getattr(risk_decision, "kill_process", False):
            return 0.0
        if getattr(risk_decision, "force_liquidation", False):
            # LIQUIDATE_ONLY: drive every position to flat.
            return -current_weight

        exposure_before = abs(current_weight)
        exposure_after = abs(current_weight + desired_delta_weight)
        increases = exposure_after > exposure_before + _EPS

        if increases and not getattr(risk_decision, "allow_increase_exposure", True):
            return 0.0
        if (
            not getattr(risk_decision, "allow_new_entries", True)
            and exposure_before <= _EPS
            and abs(desired_delta_weight) > _EPS
        ):
            return 0.0
        return desired_delta_weight

    def _cancel_all_open_orders(self) -> None:
        fn = getattr(self.broker, "cancel_all_open_orders", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[executor] cancel_all_open_orders failed: {exc}")
