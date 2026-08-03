from __future__ import annotations

import pytest

import config
from src.guardrails import (
    can_open_new_position,
    check_circuit_breaker,
    clamp_position_size,
    record_equity_anchor,
)


def test_position_sizing_clamps_oversized_trade_to_max_notional() -> None:
    requested_notional = 10.00
    price = 2.00

    qty = clamp_position_size(requested_notional, price)

    assert qty == 2.5
    assert qty * price == config.MAX_POSITION_SIZE


def test_position_sizing_rejects_invalid_prices() -> None:
    assert clamp_position_size(10.00, 0.0) == 0.0
    assert clamp_position_size(10.00, -1.0) == 0.0


def test_max_open_positions_guard() -> None:
    assert can_open_new_position(config.MAX_OPEN_POSITIONS - 1) is True
    assert can_open_new_position(config.MAX_OPEN_POSITIONS) is False


def test_daily_loss_circuit_breaker_triggers_sys_exit(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "bot_state.json"
    monkeypatch.setattr(config, "STATE_PATH", str(state_path))

    record_equity_anchor(100.00)

    with pytest.raises(SystemExit) as exc_info:
        check_circuit_breaker(100.00 + config.DAILY_LOSS_LIMIT - 0.01)

    assert exc_info.value.code == 1


def test_daily_loss_circuit_breaker_allows_non_breach(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "bot_state.json"
    monkeypatch.setattr(config, "STATE_PATH", str(state_path))

    record_equity_anchor(100.00)

    check_circuit_breaker(100.00 + config.DAILY_LOSS_LIMIT + 0.01)
