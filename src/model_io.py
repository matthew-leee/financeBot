"""
Model (de)serialization.

XGBoost has first-class JSON serialization via Booster.save_model("*.json"),
which is portable, human-inspectable, and version-stable. We pair it with a
small sidecar meta file that pins the exact feature order + thresholds so the
live path can never silently drift from what was trained.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from xgboost import XGBClassifier

import config
from src.data import FEATURE_COLUMNS


# A real XGBoost JSON model is multi-KB and the sidecar meta is a few hundred
# bytes. Anything <= 1 byte is a missing/empty/placeholder artifact.
_MIN_ARTIFACT_BYTES = 2


class ModelArtifactError(RuntimeError):
    """Raised when the trained model artifacts are missing/empty/placeholder."""


def verify_model_artifacts(
    model_path: str | None = None,
    meta_path: str | None = None,
) -> None:
    """
    Fail-fast validation of the trained model artifacts.

    Checks that both the model JSON and the feature-metadata sidecar exist and
    are not empty or a 1-byte placeholder. Raises ModelArtifactError with a
    clear, actionable message otherwise. Call this at startup BEFORE any capital
    is put at risk.
    """
    model_path = model_path or config.MODEL_PATH
    meta_path = meta_path or config.FEATURE_META_PATH

    problems: list[str] = []
    for label, path in (("Model", model_path), ("Feature metadata", meta_path)):
        if not os.path.exists(path):
            problems.append(f"  - {label} file is MISSING: {path}")
            continue
        size = os.path.getsize(path)
        if size < _MIN_ARTIFACT_BYTES:
            problems.append(
                f"  - {label} file is EMPTY/placeholder ({size} byte(s)): {path}"
            )

    if problems:
        raise ModelArtifactError(
            "Trained model artifacts are not ready:\n"
            + "\n".join(problems)
            + "\n\n  >> Run `python train.py` to train and save a model before "
            "starting the\n     live loop (run_bot.py) or the backtester "
            "(backtest.py)."
        )


def save_model(model: XGBClassifier, report_summary: dict) -> None:
    """Persist the model to JSON and write the feature/threshold sidecar."""
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # Native XGBoost JSON -> zero-dependency, forward-compatible artifact.
    model.save_model(config.MODEL_PATH)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "buy_threshold": config.BUY_THRESHOLD,
        "sell_threshold": config.SELL_THRESHOLD,
        "bar_timeframe": config.BAR_TIMEFRAME,
        "validation": report_summary,
    }
    with open(config.FEATURE_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[model_io] Saved model -> {config.MODEL_PATH}")
    print(f"[model_io] Saved meta  -> {config.FEATURE_META_PATH}")


class LoadedModel:
    """Thin, deterministic wrapper around a loaded booster + its metadata."""

    def __init__(self, model: XGBClassifier, meta: dict) -> None:
        self._model = model
        self.feature_columns: list[str] = meta["feature_columns"]
        self.buy_threshold: float = meta["buy_threshold"]
        self.sell_threshold: float = meta["sell_threshold"]

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        """
        Return P(next bar up) for a single feature row.

        Enforces exact column order so a reordered/renamed feature can never be
        fed to the model unnoticed.
        """
        ordered = features_row[self.feature_columns]
        proba = self._model.predict_proba(ordered)[:, 1]
        return float(proba[-1])


def load_model() -> LoadedModel:
    """Load the JSON model + sidecar meta for the live loop."""
    # Fail fast on missing/empty/placeholder artifacts with a clear message.
    verify_model_artifacts()

    model = XGBClassifier()
    model.load_model(config.MODEL_PATH)

    with open(config.FEATURE_META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    return LoadedModel(model, meta)



# ===========================================================================
# ADDITIVE: Multi-role Model Registry for the Dual-Horizon Engine
# ===========================================================================
# The legacy load_model()/verify_model_artifacts()/LoadedModel above are the
# single-model contract used by the legacy engine and are intentionally left
# untouched. The registry below layers a role-keyed multi-model loader on top so
# the strategist can ask for "expected_return", "regime", etc. independently.

from dataclasses import dataclass
from typing import Literal, Protocol

ModelRole = Literal[
    "legacy_direction",
    "expected_return",
    "regime",
    "covariance",
    "tactical",
]


class InferenceModel(Protocol):
    """Minimal inference contract used by the registry and strategist."""

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Return model output indexed like x."""
        ...


@dataclass(frozen=True)
class ModelArtifact:
    """Role-keyed trained artifact metadata."""

    role: ModelRole
    model_path: str
    feature_columns: tuple[str, ...]
    horizon: Literal["interday", "intraday"]
    validation: dict
    trained_at: str


class _OrderedModel:
    """
    Wrapper enforcing exact feature-column order before inference.

    This is the registry analogue of LoadedModel.predict_up_proba''s column
    guard: a reordered or missing feature can never silently reach the model.
    """

    def __init__(self, model: object, feature_columns: tuple[str, ...]) -> None:
        self._model = model
        self.feature_columns: tuple[str, ...] = tuple(feature_columns)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_columns if c not in x.columns]
        if missing:
            raise ModelArtifactError(
                "Feature matrix is missing required columns for model: "
                f"{missing}"
            )
        ordered = x[list(self.feature_columns)]
        raw = self._model.predict(ordered)
        if isinstance(raw, pd.Series):
            return raw.reindex(x.index)
        return pd.Series(list(raw), index=x.index)


def _load_registry_model_file(model_path: str) -> object:
    """Load a role model artifact from disk (native XGBoost JSON)."""
    verify_model_artifacts(model_path=model_path, meta_path=model_path)
    model = XGBClassifier()
    model.load_model(model_path)
    return model


class ModelRegistry:
    """
    Multi-model loader keyed by role.

    Metadata lives in <registry_dir>/registry.json:

        {
          "expected_return": {
            "model_path": "models/registry/expected_return.json",
            "feature_columns": ["ret_20", "vol_63", ...],
            "horizon": "interday",
            "validation": {...},
            "trained_at": "2026-01-01T00:00:00Z"
          },
          ...
        }

    In-memory models can also be registered directly (used heavily in tests and
    by deterministic fallback logic) via register().
    """

    def __init__(self, registry_dir: str | None = None) -> None:
        self.registry_dir = registry_dir or config.MODEL_REGISTRY_DIR
        self._artifacts: dict[str, ModelArtifact] = {}
        self._loaded: dict[str, _OrderedModel] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        path = os.path.join(self.registry_dir, "registry.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:  # noqa: BLE001 -- corrupt registry -> empty
            print(f"[model_io] Failed to read registry metadata: {exc}")
            return
        for role, meta in (raw or {}).items():
            self._artifacts[role] = ModelArtifact(
                role=role,
                model_path=meta.get("model_path", ""),
                feature_columns=tuple(meta.get("feature_columns", ())),
                horizon=meta.get("horizon", "interday"),
                validation=meta.get("validation", {}),
                trained_at=meta.get("trained_at", ""),
            )

    def register(
        self,
        role: ModelRole,
        model: object,
        *,
        feature_columns: tuple[str, ...],
        horizon: Literal["interday", "intraday"] = "interday",
        validation: dict | None = None,
        trained_at: str = "",
    ) -> None:
        """Register an in-memory model (predict-capable) under a role."""
        self._artifacts[role] = ModelArtifact(
            role=role,
            model_path="<in-memory>",
            feature_columns=tuple(feature_columns),
            horizon=horizon,
            validation=validation or {},
            trained_at=trained_at,
        )
        self._loaded[role] = _OrderedModel(model, tuple(feature_columns))

    def roles(self) -> tuple[str, ...]:
        """Return all known artifact roles."""
        return tuple(self._artifacts)

    def artifact(self, role: ModelRole) -> ModelArtifact:
        """Return artifact metadata for a role or raise if absent."""
        art = self._artifacts.get(role)
        if art is None:
            raise ModelArtifactError(f"No registered model artifact for role '{role}'.")
        return art

    def verify(self, required_roles: tuple[ModelRole, ...]) -> None:
        """Raise ModelArtifactError if any required role is missing or invalid."""
        problems: list[str] = []
        for role in required_roles:
            art = self._artifacts.get(role)
            if art is None:
                problems.append(f"  - Required model role is MISSING: '{role}'")
                continue
            if not art.feature_columns:
                problems.append(
                    f"  - Model role '{role}' declares no feature_columns."
                )
            if role not in self._loaded and art.model_path not in ("", "<in-memory>"):
                if not os.path.exists(art.model_path):
                    problems.append(
                        f"  - Model role '{role}' file is MISSING: {art.model_path}"
                    )
                elif os.path.getsize(art.model_path) < _MIN_ARTIFACT_BYTES:
                    problems.append(
                        f"  - Model role '{role}' file is EMPTY/placeholder: "
                        f"{art.model_path}"
                    )
        if problems:
            raise ModelArtifactError(
                "Model registry is not ready:\n" + "\n".join(problems)
            )

    def load(self, role: ModelRole) -> _OrderedModel:
        """Load a role model, enforcing feature-column order at inference."""
        if role in self._loaded:
            return self._loaded[role]
        art = self.artifact(role)
        model = _load_registry_model_file(art.model_path)
        wrapped = _OrderedModel(model, art.feature_columns)
        self._loaded[role] = wrapped
        return wrapped

    def predict(self, role: ModelRole, features: pd.DataFrame) -> pd.Series:
        """Order features per artifact metadata, then call the role model."""
        return self.load(role).predict(features)


def save_registry_artifact(
    role: str,
    model: object,
    *,
    feature_columns: tuple[str, ...] | list[str],
    horizon: Literal["interday", "intraday"] = "interday",
    validation: dict | None = None,
    registry_dir: str | None = None,
) -> str:
    """
    Persist a trained role model + register it in <registry_dir>/registry.json.

    Additive training-side helper: writes the native XGBoost artifact to
    <registry_dir>/<role>.json and merges the metadata entry into registry.json
    without touching unrelated roles. Returns the written model path.

    The live ModelRegistry picks this up on next construction, so a freshly
    trained strategist model is verified and loaded with strict column order.
    """
    import datetime as _dt

    directory = registry_dir or config.MODEL_REGISTRY_DIR
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, f"{role}.json")
    save_fn = getattr(model, "save_model", None)
    if not callable(save_fn):
        raise ModelArtifactError(
            f"Model for role '{role}' has no save_model() -- cannot persist."
        )
    save_fn(model_path)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < _MIN_ARTIFACT_BYTES:
        raise ModelArtifactError(
            f"Model artifact for role '{role}' was not written correctly."
        )

    entry = {
        "model_path": model_path,
        "feature_columns": list(feature_columns),
        "horizon": horizon,
        "validation": validation or {},
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    registry_path = os.path.join(directory, "registry.json")
    registry: dict = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                registry = json.load(fh) or {}
        except Exception as exc:  # noqa: BLE001 -- rebuild rather than crash training
            print(f"[model_io] Registry unreadable, rebuilding: {exc}")
            registry = {}
    registry[role] = entry
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    print(f"[model_io] Saved '{role}' artifact -> {model_path}")
    return model_path
