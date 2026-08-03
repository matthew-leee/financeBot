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
