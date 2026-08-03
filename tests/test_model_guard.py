from __future__ import annotations

import pytest

import config
from src import execution
from src.model_io import ModelArtifactError, verify_model_artifacts


def _write(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_verify_passes_for_valid_sized_artifacts(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.json"
    meta = tmp_path / "feature_meta.json"
    _write(model, '{"learner": {"gradient_booster": {}}}')
    _write(meta, '{"feature_columns": ["ret_1"]}')

    monkeypatch.setattr(config, "MODEL_PATH", str(model))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta))

    # Should not raise.
    verify_model_artifacts()


def test_verify_raises_when_model_missing(tmp_path, monkeypatch) -> None:
    meta = tmp_path / "feature_meta.json"
    _write(meta, '{"feature_columns": ["ret_1"]}')

    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta))

    with pytest.raises(ModelArtifactError) as exc:
        verify_model_artifacts()
    assert "MISSING" in str(exc.value)
    assert "train.py" in str(exc.value)


def test_verify_raises_when_model_is_one_byte_placeholder(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.json"
    meta = tmp_path / "feature_meta.json"
    _write(model, "0")  # 1-byte placeholder
    _write(meta, '{"feature_columns": ["ret_1"]}')

    monkeypatch.setattr(config, "MODEL_PATH", str(model))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta))

    with pytest.raises(ModelArtifactError) as exc:
        verify_model_artifacts()
    assert "placeholder" in str(exc.value)


def test_verify_raises_when_meta_empty(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.json"
    meta = tmp_path / "feature_meta.json"
    _write(model, '{"learner": {}}')
    _write(meta, "")  # 0 bytes

    monkeypatch.setattr(config, "MODEL_PATH", str(model))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta))

    with pytest.raises(ModelArtifactError):
        verify_model_artifacts()


def test_run_fails_fast_and_exits_on_placeholder_artifacts(tmp_path, monkeypatch, capsys) -> None:
    """run() must sys.exit(1) with a friendly message, before building a broker."""
    model = tmp_path / "model.json"
    meta = tmp_path / "feature_meta.json"
    _write(model, "0")  # placeholder
    _write(meta, "0")

    monkeypatch.setattr(config, "MODEL_PATH", str(model))
    monkeypatch.setattr(config, "FEATURE_META_PATH", str(meta))

    # If the guard fails to trip, this would blow up loudly instead of exiting.
    def _should_not_run():
        raise AssertionError("Broker must not be constructed when artifacts are invalid")

    monkeypatch.setattr(execution, "Broker", _should_not_run)

    with pytest.raises(SystemExit) as exc:
        execution.run()
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "train.py" in out
