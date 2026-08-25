"""Smoke tests for the dual-engine training panel builder (train_dual.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from src.data import PointInTimeFeatureStore, PointInTimeRecord
from train_dual import build_panel

UTC = timezone.utc


def _seed_store(store: PointInTimeFeatureStore, symbols: list[str], days: int = 320):
    base = datetime(2025, 6, 1, tzinfo=UTC)
    for i in range(days):
        et = base + timedelta(days=i)
        for j, sym in enumerate(symbols):
            store.upsert_records(
                [
                    PointInTimeRecord(
                        event_time=et,
                        available_at=et,
                        source="alpaca",
                        entity_id=sym,
                        field="close",
                        value=100.0 + i * 0.2 + j * 7.0,
                    )
                ]
            )


def test_build_panel_is_leakage_safe_and_complete(tmp_path) -> None:
    store = PointInTimeFeatureStore(str(tmp_path / "fs"))
    # Keep the universe small so the snapshot grid stays fast.
    symbols = ["AAA"]
    original_core = config.CORE_UNIVERSE
    original_hedge = config.HEDGE_UNIVERSE
    config.CORE_UNIVERSE = ("AAA",)
    config.HEDGE_UNIVERSE = ()
    try:
        _seed_store(store, symbols)
        panel, feature_cols = build_panel(
            store, lookback_days=400, date_step_days=20, horizon=5
        )
    finally:
        config.CORE_UNIVERSE = original_core
        config.HEDGE_UNIVERSE = original_hedge

    assert not panel.empty
    for col in ("timestamp", "symbol", "fwd_return_5", "label_end_time_5"):
        assert col in panel.columns

    labels = panel.dropna(subset=["fwd_return_5"])
    assert len(labels) > 0
    # Every label's end must lie strictly beyond its decision timestamp...
    assert (labels["label_end_time_5"] > labels["timestamp"]).all()
    # ...and within stored history (no fabricated future). The bound is
    # DERIVED from the seed so it can never drift from the fixture.
    base = datetime(2025, 6, 1, tzinfo=UTC)
    days = 320  # must match _seed_store(days=...)
    last_stored_close = pd.Timestamp(base + timedelta(days=days - 1))
    assert (labels["label_end_time_5"] <= last_stored_close).all()
    assert all(c not in feature_cols for c in ("timestamp", "symbol"))
