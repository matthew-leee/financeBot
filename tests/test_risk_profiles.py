from __future__ import annotations

import pytest

import config
from src import guardrails
from src.guardrails import (
    RiskConfigurationError,
    can_open_new_position,
    check_circuit_breaker,
    clamp_position_size,
    record_equity_anchor,
    resolve_daily_loss_threshold,
    resolve_position_cap,
    resolve_risk_policy,
)

_RISK_ENV = (
    "FINANCEBOT_RISK_PROFILE",
    "FINANCEBOT_MAX_POSITION_PCT",
    "FINANCEBOT_MAX_GROSS_EXPOSURE_PCT",
    "FINANCEBOT_DAILY_LOSS_LIMIT_PCT",
    "FINANCEBOT_MAX_POSITION_SIZE_ABS",
    "FINANCEBOT_DAILY_LOSS_LIMIT_ABS",
    "FINANCEBOT_MAX_OPEN_POSITIONS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _RISK_ENV:
        monkeypatch.delenv(name, raising=False)


# --- profiles resolve exactly ---------------------------------------------

def test_research_resolves_to_legacy_5_10_3():
    policy = resolve_risk_policy()
    assert policy.profile == "research"
    assert resolve_position_cap(policy, 10_000.0) == 5.00
    assert resolve_daily_loss_threshold(policy, 10_000.0) == -10.00
    assert policy.max_open_positions == 3


def test_micro_live_exact_formulas(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    policy = resolve_risk_policy()
    assert resolve_position_cap(policy, 10_000.0) == 100.00
    assert resolve_daily_loss_threshold(policy, 10_000.0) == -50.00
    assert policy.max_open_positions == 3


def test_small_live_exact_formulas(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "small_live")
    policy = resolve_risk_policy()
    assert resolve_position_cap(policy, 10_000.0) == 500.00
    assert resolve_daily_loss_threshold(policy, 10_000.0) == -100.00
    assert policy.max_open_positions == 5


def test_percentage_limit_binds_for_small_equity(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    policy = resolve_risk_policy()
    # 1000 * 0.02 = 20 < 100 abs -> pct binds.
    assert resolve_position_cap(policy, 1_000.0) == 20.0


def test_absolute_limit_binds_for_large_equity(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    policy = resolve_risk_policy()
    # 10_000_000 * 0.02 = 200_000 >> 100 abs -> abs binds.
    assert resolve_position_cap(policy, 10_000_000.0) == 100.0


def test_daily_loss_threshold_is_always_negative(monkeypatch):
    for profile in ("research", "micro_live", "small_live"):
        monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", profile)
        policy = resolve_risk_policy()
        assert resolve_daily_loss_threshold(policy, 10_000.0) < 0
        assert resolve_daily_loss_threshold(policy, None) < 0


# --- overrides: tighten / clamp / reject -----------------------------------

def test_override_can_tighten_profile(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    monkeypatch.setenv("FINANCEBOT_MAX_POSITION_SIZE_ABS", "50")
    monkeypatch.setenv("FINANCEBOT_MAX_OPEN_POSITIONS", "1")
    policy = resolve_risk_policy()
    assert policy.max_position_size_abs == 50.0
    assert policy.max_open_positions == 1
    assert resolve_position_cap(policy, 10_000.0) == 50.0  # min(50, 200)


def test_loosening_override_is_clamped(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    monkeypatch.setenv("FINANCEBOT_MAX_POSITION_SIZE_ABS", "500")  # > 100 profile
    monkeypatch.setenv("FINANCEBOT_MAX_POSITION_PCT", "0.9")       # > 0.02 profile
    policy = resolve_risk_policy()
    assert policy.max_position_size_abs == 100.0
    assert policy.max_position_pct == 0.02


@pytest.mark.parametrize(
    "name,value",
    [
        ("FINANCEBOT_MAX_POSITION_PCT", "2"),
        ("FINANCEBOT_MAX_POSITION_PCT", "-0.1"),
        ("FINANCEBOT_MAX_POSITION_PCT", "0"),
        ("FINANCEBOT_MAX_POSITION_SIZE_ABS", "-5"),
        ("FINANCEBOT_MAX_POSITION_SIZE_ABS", "notnum"),
        ("FINANCEBOT_MAX_OPEN_POSITIONS", "0"),
        ("FINANCEBOT_DAILY_LOSS_LIMIT_PCT", "1.5"),
    ],
)
def test_invalid_override_rejected(monkeypatch, name, value):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    monkeypatch.setenv(name, value)
    with pytest.raises(RiskConfigurationError):
        resolve_risk_policy()


def test_unknown_profile_fails_startup(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "bogus")
    with pytest.raises(RiskConfigurationError):
        resolve_risk_policy()


# --- backward-compatible guardrail signatures ------------------------------

def test_legacy_guardrail_calls_retain_old_behavior():
    assert clamp_position_size(10.0, 2.0) == 2.5
    assert can_open_new_position(config.MAX_OPEN_POSITIONS - 1) is True
    assert can_open_new_position(config.MAX_OPEN_POSITIONS) is False


def test_dynamic_breach_raises_system_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_PATH", str(tmp_path / "bot_state.json"))
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "micro_live")
    policy = resolve_risk_policy()
    record_equity_anchor(10_000.0)
    threshold = resolve_daily_loss_threshold(policy, 10_000.0)  # -50
    with pytest.raises(SystemExit) as exc:
        check_circuit_breaker(10_000.0 + threshold - 0.01, loss_limit=threshold)
    assert exc.value.code == 1


# --- growth_live: sized for $10k-30k equity, evidence-gated -----------------

def test_growth_live_exact_formulas(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "growth_live")
    policy = resolve_risk_policy()
    # $10k anchor: pct binds (0.12 * 10k = 1200 < 5000 abs).
    assert resolve_position_cap(policy, 10_000.0) == 1200.00
    # $25k anchor: still pct-bound (3000 < 5000); loss = -min(750, 750).
    assert resolve_position_cap(policy, 25_000.0) == 3000.00
    assert resolve_daily_loss_threshold(policy, 25_000.0) == -750.00
    # Abs cap binds only above ~$41.7k equity.
    assert resolve_position_cap(policy, 1_000_000.0) == 5000.00
    assert policy.max_gross_exposure_pct == 0.85
    assert policy.max_open_positions == 8


def test_growth_live_threshold_negative_for_all_equities(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "growth_live")
    policy = resolve_risk_policy()
    for equity in (None, 0.0, -5.0, 1_000.0, 250_000.0):
        assert resolve_daily_loss_threshold(policy, equity) < 0
        assert resolve_position_cap(policy, equity) > 0


def test_growth_live_overrides_may_only_tighten(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "growth_live")
    # Loosening abs cap clamps to profile ceiling with a warning.
    monkeypatch.setenv("FINANCEBOT_MAX_POSITION_SIZE_ABS", "999999")
    policy = resolve_risk_policy()
    assert policy.max_position_size_abs == 5000.00
    # Loosening position count clamps too.
    monkeypatch.setenv("FINANCEBOT_MAX_OPEN_POSITIONS", "50")
    policy = resolve_risk_policy()
    assert policy.max_open_positions == 8
    # Tightening works.
    monkeypatch.setenv("FINANCEBOT_MAX_POSITION_SIZE_ABS", "1000")
    policy = resolve_risk_policy()
    assert policy.max_position_size_abs == 1000.00


def test_all_profiles_listed_and_gated_set_matches():
    names = set(guardrails._RISK_PROFILES)
    assert {"research", "micro_live", "small_live", "growth_live"} <= names
    assert guardrails._EVIDENCE_GATED_PROFILES == frozenset({"growth_live"})


def test_soak_profile_exact_formulas(monkeypatch):
    monkeypatch.setenv("FINANCEBOT_RISK_PROFILE", "soak")
    policy = resolve_risk_policy()
    # Research sizing preserved: $5/order, -$10/day -- at ANY equity.
    for equity in (1_000.0, 100_000.0):
        assert resolve_position_cap(policy, equity) == 5.00
        assert resolve_daily_loss_threshold(policy, equity) == -10.00
    assert policy.max_open_positions == 50  # ceiling; effective = universe size
    assert policy.max_gross_exposure_pct == 0.05
    assert guardrails._EVIDENCE_GATED_PROFILES == frozenset({"growth_live"})
