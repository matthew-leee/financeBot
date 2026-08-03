"""
Guardrail enforcement -- the non-negotiable safety layer.

Nothing in the signal/model path is allowed to bypass these functions. They are
deliberately simple and boring: clamp size, count positions, and trip the daily
loss circuit breaker with a hard sys.exit().
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

import config


def clamp_position_size(
    desired_notional: float,
    price: float,
    *,
    max_notional: float | None = None,
) -> float:
    """
    Return an order quantity whose notional never exceeds the effective cap.

    The clamp is applied to notional first (the thing we actually care about),
    then converted to quantity. AI/model logic can *request* anything; it gets
    capped here, full stop.

    ``max_notional`` is an additive, backward-compatible override: when omitted
    the hard-coded ``config.MAX_POSITION_SIZE`` is used, so existing callers keep
    their exact behavior. The hardened live loop passes a resolved risk-policy
    position cap instead.
    """
    if price <= 0:
        return 0.0
    cap = config.MAX_POSITION_SIZE if max_notional is None else float(max_notional)
    capped_notional = min(float(desired_notional), cap)
    if capped_notional <= 0:
        return 0.0
    raw_qty = capped_notional / price
    # Floor instead of round: rounding up can exceed the hard notional cap by a
    # few micros, which violates the guardrail in exactly the wrong direction.
    return math.floor(raw_qty * 1_000_000) / 1_000_000


def can_open_new_position(
    open_position_count: int,
    *,
    max_positions: int | None = None,
) -> bool:
    """Refuse to exceed the hard cap on concurrent positions.

    ``max_positions`` is an additive override; omitting it preserves the legacy
    ``config.MAX_OPEN_POSITIONS`` behavior exactly.
    """
    cap = config.MAX_OPEN_POSITIONS if max_positions is None else int(max_positions)
    return open_position_count < cap


# ---------------------------------------------------------------------------
# Daily loss circuit breaker (rolling 24h)
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if os.path.exists(config.STATE_PATH):
        try:
            with open(config.STATE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001 -- corrupt state should not crash startup
            pass
    return {}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(config.STATE_PATH), exist_ok=True)
    with open(config.STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def record_equity_anchor(current_equity: float) -> None:
    """
    Store the account equity that anchors the rolling 24h window. Reset the
    anchor whenever the stored one is older than 24h.
    """
    now = datetime.now(timezone.utc)
    state = _load_state()

    anchor_ts = state.get("anchor_ts")
    needs_reset = True
    if anchor_ts is not None:
        age = now - datetime.fromisoformat(anchor_ts)
        needs_reset = age.total_seconds() >= 24 * 3600

    if needs_reset:
        state["anchor_ts"] = now.isoformat()
        state["anchor_equity"] = float(current_equity)
        _save_state(state)


def check_circuit_breaker(
    current_equity: float,
    *,
    loss_limit: float | None = None,
) -> None:
    """
    Compare current equity to the 24h anchor. If the drawdown breaches the loss
    limit, shut the whole process down immediately.

    sys.exit() is intentional and required by the guardrail spec: a tripped
    breaker means "stop trading NOW", not "log and keep going".

    ``loss_limit`` is an additive override (a NEGATIVE threshold). Omitting it
    preserves the legacy ``config.DAILY_LOSS_LIMIT`` behavior exactly. The
    hardened live loop passes a dynamically resolved, anchor-relative threshold.
    """
    record_equity_anchor(current_equity)
    state = _load_state()
    anchor_equity = state.get("anchor_equity")
    if anchor_equity is None:
        return

    limit = config.DAILY_LOSS_LIMIT if loss_limit is None else float(loss_limit)
    pnl = float(current_equity) - float(anchor_equity)
    if pnl <= limit:
        print(
            f"[CIRCUIT BREAKER] 24h PnL {pnl:.2f} <= limit "
            f"{limit:.2f}. Shutting down."
        )
        sys.exit(1)


# ===========================================================================
# ADDITIVE: Multi-tier Risk State Machine for the Dual-Horizon Engine
# ===========================================================================
# The legacy check_circuit_breaker() above is a blunt, single-trigger sys.exit()
# and is intentionally preserved for existing tests and the legacy loop. The
# state machine below adds graduated risk states and per-order permissions so
# the dual engine can freeze, de-risk, liquidate, or kill instead of only dying.

import dataclasses
from datetime import timedelta
from enum import Enum
from typing import Literal


class RiskState(str, Enum):
    """Risk state controlling permitted trading behavior."""

    NORMAL = "NORMAL"
    FREEZE_NEW_ENTRIES = "FREEZE_NEW_ENTRIES"
    REDUCE_RISK = "REDUCE_RISK"
    LIQUIDATE_ONLY = "LIQUIDATE_ONLY"
    KILL_PROCESS = "KILL_PROCESS"


# Severity order used for escalation and one-step-at-a-time recovery.
_SEVERITY: list[RiskState] = [
    RiskState.NORMAL,
    RiskState.FREEZE_NEW_ENTRIES,
    RiskState.REDUCE_RISK,
    RiskState.LIQUIDATE_ONLY,
    RiskState.KILL_PROCESS,
]

_RISK_EPS = 1e-9


@dataclasses.dataclass(frozen=True)
class RiskLimits:
    """Hard and soft portfolio risk limits."""

    max_symbol_weight: float
    max_gross_exposure: float
    max_net_exposure: float
    warn_drawdown: float
    reduce_drawdown: float
    liquidate_drawdown: float
    kill_drawdown: float
    max_spread_bps: float
    max_data_staleness_seconds: int
    max_reconciliation_qty_diff: float
    cooldown: timedelta

    @classmethod
    def from_config(cls) -> "RiskLimits":
        return cls(
            max_symbol_weight=config.MAX_SYMBOL_WEIGHT,
            max_gross_exposure=config.MAX_GROSS_EXPOSURE,
            max_net_exposure=config.MAX_NET_EXPOSURE,
            warn_drawdown=config.RISK_WARN_DRAWDOWN,
            reduce_drawdown=config.RISK_REDUCE_DRAWDOWN,
            liquidate_drawdown=config.RISK_LIQUIDATE_DRAWDOWN,
            kill_drawdown=config.RISK_KILL_DRAWDOWN,
            max_spread_bps=config.MAX_SPREAD_BPS,
            max_data_staleness_seconds=config.MAX_DATA_STALENESS_SECONDS,
            max_reconciliation_qty_diff=config.MAX_RECONCILIATION_QTY_DIFF,
            cooldown=timedelta(seconds=config.RISK_COOLDOWN_SECONDS),
        )


@dataclasses.dataclass(frozen=True)
class RiskMetrics:
    """Current risk telemetry fed to the state machine."""

    as_of: datetime
    equity: float
    high_watermark_equity: float
    rolling_24h_pnl: float
    drawdown: float
    gross_exposure: float
    net_exposure: float
    largest_symbol_weight: float
    data_staleness_seconds: int
    major_reconciliation_breaks: int
    open_order_count: int
    api_error_rate: float
    model_artifacts_valid: bool
    broker_available: bool
    positions_open: bool = False


@dataclasses.dataclass(frozen=True)
class RiskDecision:
    """Risk-state output consumed by the tactical executor."""

    state: RiskState
    previous_state: RiskState
    reason: str
    allow_new_entries: bool
    allow_increase_exposure: bool
    allow_reduce_exposure: bool
    force_liquidation: bool
    kill_process: bool


def _severity_index(state: RiskState) -> int:
    return _SEVERITY.index(state)


class RiskStateMachine:
    """Multi-tier risk guardrail engine with graduated states and cooldown."""

    def __init__(self, *, limits: RiskLimits, state_path: str | None = None) -> None:
        self.limits = limits
        self.state_path = state_path or config.RISK_STATE_PATH
        self.state = RiskState.NORMAL
        self.high_watermark_equity: float | None = None
        self._last_change_ts: datetime | None = None
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.state = RiskState(data.get("state", RiskState.NORMAL.value))
            hwm = data.get("high_watermark_equity")
            self.high_watermark_equity = float(hwm) if hwm is not None else None
            ts = data.get("last_change_ts")
            self._last_change_ts = datetime.fromisoformat(ts) if ts else None
        except Exception:  # noqa: BLE001 -- corrupt state -> conservative NORMAL
            self.state = RiskState.NORMAL

    def _save(self) -> None:
        try:
            directory = os.path.dirname(self.state_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "state": self.state.value,
                        "high_watermark_equity": self.high_watermark_equity,
                        "last_change_ts": (
                            self._last_change_ts.isoformat()
                            if self._last_change_ts
                            else None
                        ),
                    },
                    fh,
                    indent=2,
                )
        except Exception as exc:  # noqa: BLE001 -- persistence must not crash run
            print(f"[risk] Failed to persist risk state: {exc}")

    # -- evaluation ----------------------------------------------------------

    def _required_state(self, m: RiskMetrics) -> tuple[RiskState, str]:
        lim = self.limits
        # KILL_PROCESS: catastrophic / integrity failures.
        if m.drawdown <= lim.kill_drawdown:
            return RiskState.KILL_PROCESS, f"drawdown {m.drawdown:.3f} <= kill"
        if not m.model_artifacts_valid:
            return RiskState.KILL_PROCESS, "model artifacts invalid during live trading"
        if not m.broker_available and m.positions_open:
            return RiskState.KILL_PROCESS, "broker unavailable while positions open"

        # LIQUIDATE_ONLY.
        if m.drawdown <= lim.liquidate_drawdown:
            return RiskState.LIQUIDATE_ONLY, f"drawdown {m.drawdown:.3f} <= liquidate"
        if m.major_reconciliation_breaks > 0:
            return RiskState.LIQUIDATE_ONLY, "unresolved major reconciliation break"
        if m.data_staleness_seconds > lim.max_data_staleness_seconds * 3:
            return RiskState.LIQUIDATE_ONLY, "data critically stale"
        if m.api_error_rate >= 0.5 and m.positions_open:
            return RiskState.LIQUIDATE_ONLY, "api error rate critically high"

        # REDUCE_RISK.
        if m.drawdown <= lim.reduce_drawdown:
            return RiskState.REDUCE_RISK, f"drawdown {m.drawdown:.3f} <= reduce"
        if m.gross_exposure > lim.max_gross_exposure + _RISK_EPS:
            return RiskState.REDUCE_RISK, "gross exposure over limit"
        if abs(m.net_exposure) > lim.max_net_exposure + _RISK_EPS:
            return RiskState.REDUCE_RISK, "net exposure over limit"
        if m.largest_symbol_weight > lim.max_symbol_weight + _RISK_EPS:
            return RiskState.REDUCE_RISK, "symbol weight over limit"

        # FREEZE_NEW_ENTRIES.
        if m.drawdown <= lim.warn_drawdown:
            return RiskState.FREEZE_NEW_ENTRIES, f"drawdown {m.drawdown:.3f} <= warn"
        if 0 < m.data_staleness_seconds > lim.max_data_staleness_seconds:
            return RiskState.FREEZE_NEW_ENTRIES, "data stale (non-critical)"

        return RiskState.NORMAL, "all clear"

    def evaluate(self, metrics: RiskMetrics) -> RiskDecision:
        """Evaluate state transitions with escalate-now / recover-slowly rules."""
        previous = self.state

        # Track high-watermark equity for observability.
        if self.high_watermark_equity is None:
            self.high_watermark_equity = metrics.high_watermark_equity
        else:
            self.high_watermark_equity = max(
                self.high_watermark_equity, metrics.high_watermark_equity
            )

        required, reason = self._required_state(metrics)
        now = metrics.as_of

        if previous == RiskState.KILL_PROCESS:
            # Terminal state -- never auto-recovers.
            new_state = RiskState.KILL_PROCESS
            reason = "kill state is terminal"
        elif _severity_index(required) >= _severity_index(previous):
            new_state = required  # escalate immediately
        else:
            # Recovery: at most one severity level down, and only after cooldown.
            cooldown_ok = (
                self._last_change_ts is None
                or (now - self._last_change_ts) >= self.limits.cooldown
            )
            if cooldown_ok:
                new_state = _SEVERITY[max(0, _severity_index(previous) - 1)]
                reason = f"recovering toward {required.value}"
            else:
                new_state = previous
                reason = "cooldown active; holding state"

        if new_state != previous:
            self._last_change_ts = now
        self.state = new_state
        self._save()

        decision = self.permissions_for_state(new_state)
        return dataclasses.replace(decision, previous_state=previous, reason=reason)

    def permissions_for_state(self, state: RiskState) -> RiskDecision:
        """Return the permission matrix for a state (previous_state == state)."""
        flags = {
            RiskState.NORMAL: (True, True, True, False, False),
            RiskState.FREEZE_NEW_ENTRIES: (False, False, True, False, False),
            RiskState.REDUCE_RISK: (False, False, True, False, False),
            RiskState.LIQUIDATE_ONLY: (False, False, True, True, False),
            RiskState.KILL_PROCESS: (False, False, False, False, True),
        }[state]
        allow_new, allow_inc, allow_red, force_liq, kill = flags
        return RiskDecision(
            state=state,
            previous_state=state,
            reason=state.value,
            allow_new_entries=allow_new,
            allow_increase_exposure=allow_inc,
            allow_reduce_exposure=allow_red,
            force_liquidation=force_liq,
            kill_process=kill,
        )

    def is_order_allowed(
        self,
        *,
        state: RiskState,
        current_weight: float,
        target_weight_after: float,
        side: Literal["buy", "sell"],
        reduce_only: bool,
    ) -> bool:
        """Validate a single order against the current risk state."""
        if state == RiskState.KILL_PROCESS:
            return False

        exposure_before = abs(current_weight)
        exposure_after = abs(target_weight_after)
        increases = exposure_after > exposure_before + _RISK_EPS

        if state == RiskState.NORMAL:
            return True
        if state == RiskState.FREEZE_NEW_ENTRIES:
            return not increases
        if state == RiskState.REDUCE_RISK:
            return exposure_after < exposure_before - _RISK_EPS
        if state == RiskState.LIQUIDATE_ONLY:
            # Must reduce exposure AND move toward zero (no side flips).
            moves_toward_zero = (
                abs(target_weight_after) < abs(current_weight)
                and current_weight * target_weight_after >= -_RISK_EPS
            )
            return exposure_after < exposure_before - _RISK_EPS and moves_toward_zero
        return False


# ===========================================================================
# ADDITIVE: Immutable runtime risk policy (legacy constants above unchanged)
# ===========================================================================
# The legacy MAX_POSITION_SIZE / DAILY_LOSS_LIMIT / MAX_OPEN_POSITIONS constants
# remain the authoritative hard floor for the "research" profile. This layer adds
# named, immutable profiles that an operator can select (and only ever TIGHTEN)
# via environment variables. It is resolved ONCE at process startup and never
# re-read per order.

_RISK_PCT_FIELDS = ("max_position_pct", "max_gross_exposure_pct", "daily_loss_pct")


class RiskConfigurationError(Exception):
    """Raised when the risk profile/config is unknown, invalid, or loosening."""


@dataclasses.dataclass(frozen=True)
class RiskPolicy:
    """Immutable resolved risk policy for a single process lifetime."""

    profile: str
    max_position_pct: float | None
    max_gross_exposure_pct: float | None
    daily_loss_pct: float | None
    max_position_size_abs: float
    daily_loss_limit_abs: float
    max_open_positions: int


# Code-defined profiles. "research" resolves to the exact legacy $5 / -$10 / 3
# behavior. Absolute values are POSITIVE magnitudes here (sign applied later).
_RISK_PROFILES: dict[str, RiskPolicy] = {
    "research": RiskPolicy(
        profile="research",
        max_position_pct=None,
        max_gross_exposure_pct=None,
        daily_loss_pct=None,
        max_position_size_abs=5.00,
        daily_loss_limit_abs=10.00,
        max_open_positions=3,
    ),
    "micro_live": RiskPolicy(
        profile="micro_live",
        max_position_pct=0.02,
        max_gross_exposure_pct=0.25,
        daily_loss_pct=0.005,
        max_position_size_abs=100.00,
        daily_loss_limit_abs=50.00,
        max_open_positions=3,
    ),
    "small_live": RiskPolicy(
        profile="small_live",
        max_position_pct=0.05,
        max_gross_exposure_pct=0.50,
        daily_loss_pct=0.01,
        max_position_size_abs=1000.00,
        daily_loss_limit_abs=250.00,
        max_open_positions=5,
    ),
}


def _env_raw(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _parse_pct_override(name: str, profile_value: float | None) -> float | None:
    """Parse a percentage override (0 < v <= 1). Reject invalid; clamp loosening."""
    raw = _env_raw(name)
    if raw is None:
        return profile_value
    try:
        value = float(raw)
    except ValueError as exc:
        raise RiskConfigurationError(f"{name} is not a number: {raw!r}") from exc
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise RiskConfigurationError(f"{name} must be a fraction in (0, 1], got {raw!r}.")
    if profile_value is not None and value > profile_value:
        # Loosening a defined ceiling: clamp down to the profile with a warning.
        print(
            f"[risk] WARNING: {name}={value} loosens profile ({profile_value}); "
            f"clamping to {profile_value}."
        )
        return profile_value
    return value


def _parse_abs_override(name: str, profile_value: float) -> float:
    """Parse a positive absolute override. Reject invalid; clamp loosening."""
    raw = _env_raw(name)
    if raw is None:
        return profile_value
    try:
        value = float(raw)
    except ValueError as exc:
        raise RiskConfigurationError(f"{name} is not a number: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise RiskConfigurationError(f"{name} must be a positive magnitude, got {raw!r}.")
    if value > profile_value:
        print(
            f"[risk] WARNING: {name}={value} loosens profile ({profile_value}); "
            f"clamping to {profile_value}."
        )
        return profile_value
    return value


def _parse_int_override(name: str, profile_value: int) -> int:
    """Parse a positive integer override. Reject invalid; clamp loosening."""
    raw = _env_raw(name)
    if raw is None:
        return profile_value
    try:
        value = int(raw)
    except ValueError as exc:
        raise RiskConfigurationError(f"{name} is not an integer: {raw!r}") from exc
    if value < 1:
        raise RiskConfigurationError(f"{name} must be >= 1, got {raw!r}.")
    if value > profile_value:
        print(
            f"[risk] WARNING: {name}={value} loosens profile ({profile_value}); "
            f"clamping to {profile_value}."
        )
        return profile_value
    return value


def resolve_risk_policy() -> RiskPolicy:
    """
    Resolve the immutable risk policy ONCE at startup.

    Selects ``FINANCEBOT_RISK_PROFILE`` (default ``research``) then applies any
    tightening environment overrides. Unknown profiles and invalid/non-finite/
    out-of-range overrides raise :class:`RiskConfigurationError` and must stop
    startup before broker initialization. Overrides may only tighten a profile:
    invalid values are rejected; merely-loosening values are clamped down to the
    profile ceiling with a warning.
    """
    name = _env_raw("FINANCEBOT_RISK_PROFILE") or "research"
    name = name.lower()
    base = _RISK_PROFILES.get(name)
    if base is None:
        raise RiskConfigurationError(
            f"Unknown risk profile {name!r}. Choose one of "
            f"{sorted(_RISK_PROFILES)}."
        )

    return RiskPolicy(
        profile=base.profile,
        max_position_pct=_parse_pct_override(
            "FINANCEBOT_MAX_POSITION_PCT", base.max_position_pct
        ),
        max_gross_exposure_pct=_parse_pct_override(
            "FINANCEBOT_MAX_GROSS_EXPOSURE_PCT", base.max_gross_exposure_pct
        ),
        daily_loss_pct=_parse_pct_override(
            "FINANCEBOT_DAILY_LOSS_LIMIT_PCT", base.daily_loss_pct
        ),
        max_position_size_abs=_parse_abs_override(
            "FINANCEBOT_MAX_POSITION_SIZE_ABS", base.max_position_size_abs
        ),
        daily_loss_limit_abs=_parse_abs_override(
            "FINANCEBOT_DAILY_LOSS_LIMIT_ABS", base.daily_loss_limit_abs
        ),
        max_open_positions=_parse_int_override(
            "FINANCEBOT_MAX_OPEN_POSITIONS", base.max_open_positions
        ),
    )


def resolve_position_cap(policy: RiskPolicy, equity: float | None) -> float:
    """
    Resolve the per-symbol TOTAL position notional cap.

        position_cap = min(max_position_size_abs, equity * max_position_pct)

    When ``max_position_pct`` is None the cap is the absolute magnitude. When
    equity is unavailable/invalid we fall back to the absolute magnitude (the
    pre-trade gate independently blocks exposure increases on unknown equity).
    """
    abs_cap = float(policy.max_position_size_abs)
    if policy.max_position_pct is None:
        return abs_cap
    if equity is None or not math.isfinite(float(equity)) or float(equity) <= 0:
        return abs_cap
    return min(abs_cap, float(equity) * policy.max_position_pct)


def resolve_daily_loss_threshold(
    policy: RiskPolicy,
    anchor_equity: float | None,
) -> float:
    """
    Resolve the NEGATIVE rolling-24h daily-loss threshold.

        daily_loss_threshold = -min(daily_loss_limit_abs, anchor_equity * daily_loss_pct)

    When ``daily_loss_pct`` is None the threshold is ``-daily_loss_limit_abs``.
    The result is always negative (stricter-limit = negative min of positive
    magnitudes -- never max() on negatives).
    """
    abs_limit = float(policy.daily_loss_limit_abs)
    if policy.daily_loss_pct is None:
        return -abs_limit
    if (
        anchor_equity is None
        or not math.isfinite(float(anchor_equity))
        or float(anchor_equity) <= 0
    ):
        return -abs_limit
    return -min(abs_limit, float(anchor_equity) * policy.daily_loss_pct)
