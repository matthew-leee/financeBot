from __future__ import annotations

import config


def test_guardrail_constants_are_present_and_typed() -> None:
    assert hasattr(config, "MAX_POSITION_SIZE")
    assert isinstance(config.MAX_POSITION_SIZE, float)
    assert config.MAX_POSITION_SIZE == 5.00

    assert hasattr(config, "DAILY_LOSS_LIMIT")
    assert isinstance(config.DAILY_LOSS_LIMIT, float)
    assert config.DAILY_LOSS_LIMIT < 0

    assert hasattr(config, "API_CALL_DELAY_SECONDS")
    assert isinstance(config.API_CALL_DELAY_SECONDS, float)
    assert config.API_CALL_DELAY_SECONDS >= 1.0

    assert hasattr(config, "MAX_OPEN_POSITIONS")
    assert isinstance(config.MAX_OPEN_POSITIONS, int)
    assert config.MAX_OPEN_POSITIONS > 0


def test_live_decision_thresholds_are_sane() -> None:
    assert isinstance(config.BUY_THRESHOLD, float)
    assert isinstance(config.SELL_THRESHOLD, float)
    assert 0.0 < config.SELL_THRESHOLD < config.BUY_THRESHOLD < 1.0


def test_secret_values_are_not_hardcoded_in_config() -> None:
    assert not hasattr(config, "APCA_API_KEY_ID")
    assert not hasattr(config, "APCA_API_SECRET_KEY")
