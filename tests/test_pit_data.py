from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.data import (
    NewsEvent,
    PointInTimeFeatureStore,
    PointInTimeRecord,
    build_news_embedding_state,
)


def _rec(entity, field, value, event_day, avail_day, revision=None):
    et = datetime(2026, 1, event_day, tzinfo=timezone.utc)
    av = datetime(2026, 1, avail_day, tzinfo=timezone.utc)
    return PointInTimeRecord(
        event_time=et,
        available_at=av,
        source="test",
        entity_id=entity,
        field=field,
        value=value,
        revision_id=revision,
    )


def test_available_after_asof_is_excluded() -> None:
    store = PointInTimeFeatureStore(":mem:")
    store.upsert_records([_rec("US10Y", "yield", 4.0, event_day=10, avail_day=10)])
    # A value that only becomes available on the 20th must not be visible on the 15th.
    store.upsert_records([_rec("US10Y", "yield", 9.9, event_day=15, avail_day=20)])

    as_of = datetime(2026, 1, 15, tzinfo=timezone.utc)
    frame = store.query_asof(
        entity_ids=["US10Y"], fields=["yield"], as_of=as_of
    )
    assert frame.loc["US10Y", "yield"] == 4.0


def test_macro_revision_uses_correct_vintage() -> None:
    store = PointInTimeFeatureStore(":mem:")
    # Same event_time, two vintages. Original known on the 5th, revision on the 20th.
    store.upsert_records([_rec("CPI", "cpi_yoy", 3.0, event_day=1, avail_day=5, revision="v1")])
    store.upsert_records([_rec("CPI", "cpi_yoy", 3.4, event_day=1, avail_day=20, revision="v2")])

    early = store.query_asof(
        entity_ids=["CPI"], fields=["cpi_yoy"], as_of=datetime(2026, 1, 10, tzinfo=timezone.utc)
    )
    late = store.query_asof(
        entity_ids=["CPI"], fields=["cpi_yoy"], as_of=datetime(2026, 1, 25, tzinfo=timezone.utc)
    )
    assert early.loc["CPI", "cpi_yoy"] == 3.0  # only the original vintage was known
    assert late.loc["CPI", "cpi_yoy"] == 3.4  # revision now available


def test_missing_field_creates_indicator_no_backfill() -> None:
    store = PointInTimeFeatureStore(":mem:")
    store.upsert_records([_rec("AAA", "close", 100.0, event_day=1, avail_day=1)])
    frame = store.query_asof(
        entity_ids=["AAA", "BBB"], fields=["close"], as_of=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )
    assert "close_missing" in frame.columns
    assert frame.loc["AAA", "close_missing"] == 0.0
    assert frame.loc["BBB", "close_missing"] == 1.0
    # No future backfill: BBB has no value, stays NaN in the raw column.
    assert np.isnan(frame.loc["BBB", "close"])


def test_news_decay_direction_and_intensity() -> None:
    as_of = datetime(2026, 2, 1, tzinfo=timezone.utc)
    tau = timedelta(days=7)
    embedding = np.array([1.0, 0.0], dtype=float)

    recent = NewsEvent(
        event_id="n1",
        event_time=as_of - timedelta(days=1),
        available_at=as_of - timedelta(days=1),
        source="wire",
        text_hash="h1",
        tickers=("AAA",),
        themes=("earnings",),
        embedding=embedding,
        source_weight=1.0,
        event_weight=1.0,
    )
    old = NewsEvent(
        event_id="n2",
        event_time=as_of - timedelta(days=30),
        available_at=as_of - timedelta(days=30),
        source="wire",
        text_hash="h2",
        tickers=("AAA",),
        themes=("earnings",),
        embedding=embedding,
        source_weight=1.0,
        event_weight=1.0,
    )

    direction, intensity = build_news_embedding_state(
        [recent, old], symbol="AAA", as_of=as_of, tau=tau, embedding_dim=2
    )
    # Direction is a unit vector; intensity positive; recent event dominates mass.
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-9
    assert intensity > 0.0

    # Only-old event yields lower intensity than only-recent event (decay works).
    _, intensity_recent_only = build_news_embedding_state(
        [recent], symbol="AAA", as_of=as_of, tau=tau, embedding_dim=2
    )
    _, intensity_old_only = build_news_embedding_state(
        [old], symbol="AAA", as_of=as_of, tau=tau, embedding_dim=2
    )
    assert intensity_recent_only > intensity_old_only


def test_news_future_event_excluded() -> None:
    as_of = datetime(2026, 2, 1, tzinfo=timezone.utc)
    future = NewsEvent(
        event_id="future",
        event_time=as_of + timedelta(days=1),
        available_at=as_of + timedelta(days=1),
        source="wire",
        text_hash="h",
        tickers=("AAA",),
        themes=(),
        embedding=np.array([1.0, 1.0]),
        source_weight=1.0,
        event_weight=1.0,
    )
    direction, intensity = build_news_embedding_state(
        [future], symbol="AAA", as_of=as_of, tau=timedelta(days=7), embedding_dim=2
    )
    assert intensity == 0.0
    assert np.allclose(direction, 0.0)


def test_interday_snapshot_builds_returns() -> None:
    store = PointInTimeFeatureStore(":mem:")
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(300):
        et = base + timedelta(days=i)
        store.upsert_records(
            [
                PointInTimeRecord(
                    event_time=et,
                    available_at=et,
                    source="alpaca",
                    entity_id="AAA",
                    field="close",
                    value=100.0 + i,
                )
            ]
        )
    as_of = base + timedelta(days=299)
    snap = store.build_interday_snapshot(
        universe=["AAA"], as_of=as_of, news_tau=timedelta(days=7)
    )
    assert snap.horizon == "interday"
    assert "ret_20" in snap.frame.columns
    assert snap.frame.loc["AAA", "ret_20"] > 0.0
