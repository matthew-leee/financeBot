from __future__ import annotations

import pandas as pd
import pytest

from src.model_io import ModelArtifactError, ModelRegistry


class _EchoModel:
    """Returns the first feature column so column order is observable."""

    def predict(self, x: pd.DataFrame):
        return x.iloc[:, 0].to_numpy()


def test_registry_rejects_missing_required_role() -> None:
    registry = ModelRegistry(registry_dir="does/not/exist")
    with pytest.raises(ModelArtifactError) as exc:
        registry.verify(("expected_return",))
    assert "expected_return" in str(exc.value)


def test_registry_verify_passes_for_registered_role() -> None:
    registry = ModelRegistry(registry_dir="does/not/exist")
    registry.register(
        "expected_return", _EchoModel(), feature_columns=("a", "b", "c")
    )
    registry.verify(("expected_return",))  # should not raise


def test_registry_enforces_feature_column_order() -> None:
    registry = ModelRegistry(registry_dir="does/not/exist")
    registry.register("expected_return", _EchoModel(), feature_columns=("a", "b"))

    # Columns supplied out of order; registry must reorder to ["a", "b"] so the
    # echo model returns column "a", not column "b".
    features = pd.DataFrame({"b": [9.0, 9.0], "a": [1.0, 2.0]}, index=["X", "Y"])
    out = registry.predict("expected_return", features)
    assert list(out.values) == [1.0, 2.0]
    assert list(out.index) == ["X", "Y"]


def test_registry_missing_feature_column_raises() -> None:
    registry = ModelRegistry(registry_dir="does/not/exist")
    registry.register("expected_return", _EchoModel(), feature_columns=("a", "z"))
    features = pd.DataFrame({"a": [1.0]}, index=["X"])
    with pytest.raises(ModelArtifactError):
        registry.predict("expected_return", features)
