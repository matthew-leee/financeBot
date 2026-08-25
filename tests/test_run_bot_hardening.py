from __future__ import annotations

import pytest

import run_bot


def test_run_dual_empty_feature_store_exits_before_broker(monkeypatch) -> None:
    import src.broker as broker_mod
    import src.data as data_mod

    class EmptyFeatureStore:
        def __init__(self, storage_path=None):
            self.storage_path = storage_path

        def is_empty(self) -> bool:
            return True

    def broker_should_not_init():
        raise AssertionError("Broker must not initialize when dual feature store is empty")

    monkeypatch.setattr(data_mod, "PointInTimeFeatureStore", EmptyFeatureStore)
    monkeypatch.setattr(broker_mod, "Broker", broker_should_not_init)

    with pytest.raises(SystemExit) as exc:
        run_bot.run_dual()

    assert exc.value.code == 1


# --- promotion evidence gate wiring (legacy path) ---------------------------


def test_legacy_growth_live_blocked_without_evidence(monkeypatch) -> None:
    import pytest

    import src.guardrails as guardrails_mod

    def _no_broker():
        raise AssertionError("Broker must not initialize when evidence gate fails")

    monkeypatch.setattr("src.broker.Broker", _no_broker)
    monkeypatch.setattr(guardrails_mod, "resolve_risk_policy", lambda: type(
        "P", (), {"profile": "growth_live"}
    )())
    monkeypatch.setattr(
        guardrails_mod,
        "verify_promotion_evidence",
        lambda profile, **kw: (_ for _ in ()).throw(
            guardrails_mod.PromotionEvidenceError("no trade history found")
        ),
    )
    # run_bot imports these lazily inside _gate_legacy_profile from
    # src.guardrails -- patching the module attributes covers it.
    monkeypatch.setattr("sys.argv", ["run_bot.py", "--engine", "legacy"])

    with pytest.raises(SystemExit) as exc:
        run_bot.main()
    assert exc.value.code == 1


def test_legacy_lower_profile_runs_without_gate(monkeypatch) -> None:
    import src.guardrails as guardrails_mod

    calls = {"gate": [], "legacy": False}
    monkeypatch.setattr(
        run_bot, "run_legacy", lambda: calls.__setitem__("legacy", True)
    )
    monkeypatch.setattr(
        guardrails_mod,
        "resolve_risk_policy",
        lambda: type("P", (), {"profile": "research"})(),
    )

    def _record(profile, **kw):
        calls["gate"].append(profile)

    monkeypatch.setattr(guardrails_mod, "verify_promotion_evidence", _record)
    monkeypatch.setattr("sys.argv", ["run_bot.py", "--engine", "legacy"])

    run_bot.main()
    assert calls["legacy"] is True
    assert calls["gate"] == ["research"]  # consulted, but never blocks
