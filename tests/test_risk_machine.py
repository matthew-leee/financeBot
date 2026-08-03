from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import config
from src.guardrails import (
    RiskLimits,
    RiskMetrics,
    RiskState,
    RiskStateMachine,
    check_circuit_breaker,
    record_equity_anchor,
)


def _machine(tmp_path) -> RiskStateMachine:
    return RiskStateMachine(
        limits=RiskLimits.from_config(), state_path=str(tmp_path / "risk_state.json")
    )


def _metrics(**overrides) -> RiskMetrics:
    base = dict(
        as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
        equity=1000.0,
        high_watermark_equity=1000.0,
        rolling_24h_pnl=0.0,
        drawdown=0.0,
        gross_exposure=0.5,
        net_exposure=0.3,
        largest_symbol_weight=0.1,
        data_staleness_seconds=0,
        major_reconciliation_breaks=0,
        open_order_count=0,
        api_error_rate=0.0,
        model_artifacts_valid=True,
        broker_available=True,
        positions_open=True,
    )
    base.update(overrides)
    return RiskMetrics(**base)


def test_normal_when_all_clear(tmp_path) -> None:
    m = _machine(tmp_path)
    decision = m.evaluate(_metrics())
    assert decision.state == RiskState.NORMAL
    assert decision.allow_new_entries is True


def test_freeze_new_entries_blocks_increases(tmp_path) -> None:
    m = _machine(tmp_path)
    decision = m.evaluate(_metrics(drawdown=config.RISK_WARN_DRAWDOWN - 0.001))
    assert decision.state == RiskState.FREEZE_NEW_ENTRIES
    assert decision.allow_increase_exposure is False
    # Increasing exposure is blocked; reducing is allowed.
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.1, target_weight_after=0.2, side="buy", reduce_only=False
    ) is False
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.2, target_weight_after=0.1, side="sell", reduce_only=True
    ) is True


def test_reduce_risk_allows_only_reductions(tmp_path) -> None:
    m = _machine(tmp_path)
    decision = m.evaluate(_metrics(drawdown=config.RISK_REDUCE_DRAWDOWN - 0.001))
    assert decision.state == RiskState.REDUCE_RISK
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.2, target_weight_after=0.1, side="sell", reduce_only=True
    ) is True
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.1, target_weight_after=0.1, side="buy", reduce_only=False
    ) is False


def test_liquidate_only_targets_zero(tmp_path) -> None:
    m = _machine(tmp_path)
    decision = m.evaluate(_metrics(drawdown=config.RISK_LIQUIDATE_DRAWDOWN - 0.001))
    assert decision.state == RiskState.LIQUIDATE_ONLY
    assert decision.force_liquidation is True
    # Reduce toward zero allowed; flipping sign is not.
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.2, target_weight_after=0.0, side="sell", reduce_only=True
    ) is True
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.1, target_weight_after=-0.2, side="sell", reduce_only=False
    ) is False


def test_kill_process_blocks_all_orders(tmp_path) -> None:
    m = _machine(tmp_path)
    decision = m.evaluate(_metrics(drawdown=config.RISK_KILL_DRAWDOWN - 0.001))
    assert decision.state == RiskState.KILL_PROCESS
    assert decision.kill_process is True
    assert m.is_order_allowed(
        state=decision.state, current_weight=0.2, target_weight_after=0.0, side="sell", reduce_only=True
    ) is False


def test_major_reconciliation_break_escalates(tmp_path) -> None:
    m = _machine(tmp_path)
    decision = m.evaluate(_metrics(major_reconciliation_breaks=1))
    assert decision.state == RiskState.LIQUIDATE_ONLY


def test_recovery_is_gradual_one_level(tmp_path) -> None:
    m = _machine(tmp_path)
    # Escalate to LIQUIDATE_ONLY.
    m.evaluate(_metrics(drawdown=config.RISK_LIQUIDATE_DRAWDOWN - 0.001))
    # All-clear later, past cooldown -> steps down only one level, not to NORMAL.
    later = datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(seconds=config.RISK_COOLDOWN_SECONDS + 1)
    decision = m.evaluate(_metrics(as_of=later))
    assert decision.state == RiskState.REDUCE_RISK


def test_legacy_circuit_breaker_still_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "STATE_PATH", str(tmp_path / "bot_state.json"))
    record_equity_anchor(100.0)
    with pytest.raises(SystemExit) as exc:
        check_circuit_breaker(100.0 + config.DAILY_LOSS_LIMIT - 0.01)
    assert exc.value.code == 1
