"""Persistence + telemetry tests for the point-in-time feature store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.data import (
    NewsEvent,
    PointInTimeFeatureStore,
    PointInTimeRecord,
)

UTC = timezone.utc


def _store_with_records(path) -> PointInTimeFeatureStore:
    store = PointInTimeFeatureStore(str(path))
    base = datetime(2026, 1, 1, tzinfo=UTC)
    recs = []
    for i in range(10):
        et = base + timedelta(days=i)
        recs.append(
            PointInTimeRecord(
                event_time=et,
                available_at=et,
                source="alpaca",
                entity_id="AAA",
                field="close",
                value=100.0 + i,
            )
        )
    # A late-arriving revision must survive the round trip.
    recs.append(
        PointInTimeRecord(
            event_time=base + timedelta(days=4),
            available_at=base + timedelta(days=9),
            source="fred",
            entity_id="US10Y",
            field="yield",
            value=4.25,
            revision_id="v2",
        )
    )
    store.upsert_records(recs)
    return store


def test_round_trip_preserves_query_results(tmp_path) -> None:
    original = _store_with_records(tmp_path / "fs")
    written = original.save_to_disk()
    assert written == original.record_count == 11

    restored = PointInTimeFeatureStore(str(tmp_path / "fs"))
    loaded = restored.load_from_disk()
    assert loaded == 11
    assert not restored.is_empty()

    as_of = datetime(2026, 1, 8, tzinfo=UTC)
    a = original.query_asof(entity_ids=["AAA"], fields=["close"], as_of=as_of)
    b = restored.query_asof(entity_ids=["AAA"], fields=["close"], as_of=as_of)
    assert a.loc["AAA", "close"] == b.loc["AAA", "close"]

    # Vintage gating survives persistence: the Jan-5 revision is invisible
    # before its available_at but visible after.
    early = restored.query_asof(
        entity_ids=["US10Y"], fields=["yield"], as_of=datetime(2026, 1, 7, tzinfo=UTC)
    )
    late = restored.query_asof(
        entity_ids=["US10Y"], fields=["yield"], as_of=datetime(2026, 1, 10, tzinfo=UTC)
    )
    assert np.isnan(early.loc["US10Y", "yield"])
    assert late.loc["US10Y", "yield"] == 4.25


def test_news_round_trip(tmp_path) -> None:
    store = PointInTimeFeatureStore(str(tmp_path / "fs"))
    ev = NewsEvent(
        event_id="n1",
        event_time=datetime(2026, 2, 1, tzinfo=UTC),
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        source="wire",
        text_hash="abc",
        tickers=("AAA", "BBB"),
        themes=("rates",),
        embedding=np.array([0.5, -0.25, 2.0]),
        source_weight=0.9,
        event_weight=1.5,
    )
    store.upsert_news([ev])
    assert store.save_to_disk() == 0  # records file empty; news persisted anyway

    restored = PointInTimeFeatureStore(str(tmp_path / "fs"))
    restored.load_from_disk()
    assert "n1" in restored._news
    back = restored._news["n1"]
    assert back.tickers == ("AAA", "BBB")
    assert np.allclose(back.embedding, [0.5, -0.25, 2.0])


def test_corrupt_file_fails_closed(tmp_path) -> None:
    root = tmp_path / "fs"
    root.mkdir()
    (root / "records.jsonl").write_text("{not json at all\n", encoding="utf-8")
    store = PointInTimeFeatureStore(str(root))
    loaded = store.load_from_disk()
    assert loaded == 0
    assert store.is_empty()


def test_latest_available_at_respects_contents_and_empty_store(tmp_path) -> None:
    empty = PointInTimeFeatureStore(":mem:")
    assert empty.latest_available_at() is None

    store = _store_with_records(tmp_path / "fs2")
    expected = datetime(2026, 1, 10, tzinfo=UTC)  # last alpaca close
    assert store.latest_available_at() == expected
