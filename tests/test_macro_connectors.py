"""Tests for political-economical PIT connectors (src/macro.py).

Every HTTP transport and sleep is injected -- these tests NEVER touch the
network (AGENTS.md rule).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from src.data import PointInTimeFeatureStore
from src.macro import (
    BlsConnector,
    FredConnector,
    SecEdgarConnector,
    populate_feature_store,
)

UTC = timezone.utc


def _sleep_recorder():
    calls: list[float] = []
    return calls.append, calls


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def test_fred_parses_csv_with_publication_lag() -> None:
    csv_body = (
        b"observation_date,CPIAUCSL\n"
        b"2025-01-01,315.601\n"
        b"2025-02-01,316.500\n"
        b"2025-03-01,.\n"  # missing observation must be skipped
    )
    sleeps, sleep_calls = _sleep_recorder()
    conn = FredConnector(
        http_get=lambda **kw: csv_body, sleeper=sleeps, delay_seconds=0.001
    )
    recs = conn.fetch_series(
        series_ids=("CPIAUCSL",),
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )
    assert not conn.result.errors
    assert len(recs) == 2
    jan = [r for r in recs if r.event_time.month == 1][0]
    assert jan.entity_id == "CPI"
    assert jan.field == "index"
    assert jan.value == 315.601
    assert jan.source == "fred"
    # Vintage safety: available_at strictly lags event_time by the configured lag.
    assert jan.available_at == jan.event_time + timedelta(
        days=float(config.FRED_SERIES["CPIAUCSL"][2])
    )
    # Rate limiting engaged before every provider call.
    assert len(sleep_calls) == 1


def test_fred_unknown_series_is_non_fatal() -> None:
    conn = FredConnector(http_get=lambda **kw: b"", sleeper=lambda s: None, delay_seconds=0)
    recs = conn.fetch_series(
        series_ids=("NOPE_NOT_REAL",),
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 2, 1, tzinfo=UTC),
    )
    assert recs == []
    assert conn.result.errors


def test_fred_http_failure_degrades_not_raises() -> None:
    def boom(**kwargs):
        raise OSError("network down")

    conn = FredConnector(http_get=boom, sleeper=lambda s: None, delay_seconds=0)
    recs = conn.fetch_series(
        series_ids=("DGS10",),
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 2, 1, tzinfo=UTC),
    )
    assert recs == []
    assert conn.result.errors


# ---------------------------------------------------------------------------
# BLS
# ---------------------------------------------------------------------------


def test_bls_parses_monthly_vintages_and_skips_annual_average() -> None:
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "UNRATE",
                    "data": [
                        {"year": "2025", "period": "M13", "value": "4.2"},
                        {"year": "2025", "period": "M06", "value": "4.1"},
                        {"year": "2025", "period": "M05", "value": "4.0"},
                        {"year": "2025", "period": "M05", "value": "-"},
                    ],
                }
            ]
        },
    }
    config.BLS_SERIES["UNRATE"] = ("UNEMPLOYMENT", "rate", 38.0)
    try:
        conn = BlsConnector(
            http_get=lambda **kw: json.dumps(payload).encode(),
            sleeper=lambda s: None,
            delay_seconds=0,
            )
        recs = conn.fetch_series(
            series_ids=("UNRATE",),
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 12, 31, tzinfo=UTC),
        )
    finally:
        config.BLS_SERIES.pop("UNRATE", None)

    assert not conn.result.errors
    assert len(recs) == 2  # M13 skipped, dash skipped
    may, june = sorted(recs, key=lambda r: r.event_time)
    assert may.entity_id == "UNEMPLOYMENT"
    assert may.value == 4.0
    assert may.available_at == may.event_time + timedelta(days=38.0)
    assert june.value == 4.1


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------


def _edgar_doubles(tickers_body: bytes, submissions_body: bytes):
    def http_get(*, url, headers=None, timeout=30.0, method="GET", body=None):
        if "company_tickers" in url:
            return tickers_body
        return submissions_body

    return http_get


def test_edgar_emits_filing_records_with_conservative_availability() -> None:
    tickers = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE"}})
    submissions = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "UPLOAD"],
                    "filingDate": ["2025-02-01", "2025-03-05", "2025-03-06"],
                    "accessionNumber": ["0001", "0002", "0003"],
                }
            }
        }
    )
    conn = SecEdgarConnector(
        http_get=_edgar_doubles(tickers.encode(), submissions.encode()),
        sleeper=lambda s: None,
        delay_seconds=0,
    )
    recs = conn.fetch_filings(
        symbols=["AAPL"],
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )
    assert [r.field for r in recs] == ["filing_date", "filing_date"]
    tenk, eightk = recs
    assert tenk.entity_id == "AAPL"
    assert tenk.value == "0001"
    # Conservative lag: filing only knowable later the same UTC day.
    assert tenk.available_at == tenk.event_time + timedelta(hours=18)


def test_edgar_untracked_form_and_missing_cik_are_skipped_cleanly() -> None:
    tickers = json.dumps({"0": {"cik_str": 789029, "ticker": "PSQ", "title": "PROSHARES"}})
    submissions = json.dumps({"filings": {"recent": {"form": [], "filingDate": [], "accessionNumber": []}}})
    conn = SecEdgarConnector(
        http_get=_edgar_doubles(tickers.encode(), submissions.encode()),
        sleeper=lambda s: None,
        delay_seconds=0,
    )
    recs = conn.fetch_filings(
        symbols=["PSQ", "BTC/USD"],
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )
    assert recs == []
    assert any("BTC/USD" in e for e in conn.result.errors)


# ---------------------------------------------------------------------------
# Orchestration into the feature store
# ---------------------------------------------------------------------------


class _FakeFred:
    source_name = "fred"

    def __init__(self) -> None:
        self.result = type("R", (), {"errors": []})()

    def fetch_series(self, *, series_ids, start, end):
        from src.data import PointInTimeRecord

        return [
            PointInTimeRecord(
                event_time=start,
                available_at=start,
                source="fred",
                entity_id="US10Y",
                field="yield",
                value=4.5,
            )
        ]


class _FakeSec:
    source_name = "sec"

    def __init__(self) -> None:
        self.result = type("R", (), {"errors": []})()

    def fetch_filings(self, *, symbols, start, end):
        return []


def test_populate_feature_store_ingests_bars_and_macro(tmp_path, monkeypatch) -> None:
    # Bars must live INSIDE the lookback window, which is anchored to the real
    # clock: end at the current hour and span 72h so >= 2 daily closes always
    # exist regardless of the wall-clock time this test runs at.
    end = pd.Timestamp.now(tz="UTC").floor("h")
    idx = pd.date_range(end=end, periods=72, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0 + i * 0.1 for i in range(len(idx))],
            "volume": 1000.0,
        },
        index=idx,
    )
    monkeypatch.setattr("src.data.fetch_bars", lambda symbol, lookback_days=None: bars)

    store = PointInTimeFeatureStore(str(tmp_path / "feature_store"))
    summary = populate_feature_store(
        store,
        symbols=["AAPL"],
        lookback_days=60,
        include_edgar=True,
        connectors={"fred": _FakeFred(), "sec": _FakeSec()},
    )

    assert summary["by_source"]["alpaca"] >= 2  # 48h -> >= 2 daily closes
    assert summary["by_source"]["fred"] == 1
    assert summary["persisted"] == summary["records"]
    assert store.latest_available_at() is not None
