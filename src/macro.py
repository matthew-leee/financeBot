"""
Political-economical data connectors: FRED, BLS, SEC EDGAR.

These connectors turn public macro/filing releases into point-in-time
``PointInTimeRecord`` objects so the interday Strategist can consume long-horizon
political-economical trends through the anti-lookahead feature store boundary:

    provider HTTP ──► vintage-safe available_at ──► PointInTimeRecord
                          (event_time + publication lag)      │
                                                              ▼
                                              PointInTimeFeatureStore.upsert_records()

Hard rules (see AGENTS.md / Architecture.txt):
  * Every record carries event_time AND available_at. available_at is the
    earliest moment the public could know the value -- never earlier.
  * No secrets hardcoded: optional API keys come from environment variables.
  * Strict try/except around every network call; failures degrade to empty
    results + error strings so research/live loops never die on one provider.
  * Rate limited: every outbound call sleeps config.API_CALL_DELAY_SECONDS.
  * Fully mockable: HTTP transport and sleep are injected seams. Tests patch
    these and NEVER hit the network.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import config
from src.data import PointInTimeRecord

HttpGetter = Callable[..., bytes]


def _to_utc(ts: datetime) -> datetime:
    """Normalize any datetime to timezone-aware UTC (naive assumed UTC)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _default_http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    method: str = "GET",
    body: bytes | None = None,
) -> bytes:
    """Real HTTP transport used only outside tests."""
    req = urllib.request.Request(url, headers=headers or {}, method=method, data=body)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


class ConnectorResult:
    """Outcome of one connector pass: records plus non-fatal error strings."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.records: list[PointInTimeRecord] = []
        self.errors: list[str] = []

    def add(self, record: PointInTimeRecord) -> None:
        self.records.append(record)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


class MacroConnectorBase:
    """Shared defensive plumbing: throttling, transport seam, error capture."""

    source_name: str = "macro"

    def __init__(
        self,
        *,
        http_get: HttpGetter | None = None,
        sleeper: Callable[[float], None] | None = None,
        delay_seconds: float | None = None,
    ) -> None:
        self._http_get = http_get or _default_http_get
        self._sleep = sleeper or time.sleep
        self._delay = (
            config.API_CALL_DELAY_SECONDS if delay_seconds is None else delay_seconds
        )
        self.result = ConnectorResult(self.source_name)

    def _throttle(self) -> None:
        """Enforce the global minimum delay between provider calls."""
        if self._delay > 0:
            try:
                self._sleep(self._delay)
            except Exception:  # noqa: BLE001 -- a broken sleeper must not kill us
                pass

    def _fetch_bytes(self, **kwargs) -> bytes | None:
        try:
            self._throttle()
            return self._http_get(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- provider failure is not fatal
            self.result.fail(f"{type(exc).__name__}: {exc}")
            return None


# ---------------------------------------------------------------------------
# FRED -- rates, inflation, policy rate (keyless CSV endpoint by default)
# ---------------------------------------------------------------------------


class FredConnector(MacroConnectorBase):
    """
    Fetch FRED series as vintage-safe records.

    Uses the keyless ``fredgraph.csv`` endpoint by default so no credential is
    required. Per-series publication lags from ``config.FRED_SERIES`` become
    ``available_at = observation_date + lag``, which is what keeps revised
    macro history from leaking backward into features.
    """

    source_name = "fred"

    FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch_series(
        self,
        *,
        series_ids: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> list[PointInTimeRecord]:
        self.result = ConnectorResult(self.source_name)
        start_d = _to_utc(start).date()
        end_d = _to_utc(end).date()

        for sid in series_ids:
            spec = config.FRED_SERIES.get(sid)
            if spec is None:
                self.result.fail(f"Unknown FRED series '{sid}' (not in config.FRED_SERIES)")
                continue
            entity_id, field_name, lag_days = spec
            url = (
                f"{self.FRED_CSV_URL}?id={sid}"
                f"&cosd={start_d.isoformat()}&coed={end_d.isoformat()}"
            )
            raw = self._fetch_bytes(url=url)
            if raw is None:
                continue
            parsed = self._parse_csv(raw, sid=sid, lag_days=lag_days)
            if not parsed:
                self.result.fail(f"FRED series '{sid}' returned no usable observations")
                continue
            self.result.records.extend(parsed)
        return list(self.result.records)

    def _parse_csv(
        self, raw: bytes, *, sid: str, lag_days: float
    ) -> list[PointInTimeRecord]:
        out: list[PointInTimeRecord] = []
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            self.result.fail(f"FRED decode failure for {sid}: {exc}")
            return out

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if len(rows) < 2:
            return out

        # Header may be "DATE,<SID>" or "observation_date,<SID>"; column order
        # is stable (date first), which is all we rely on.
        for row in rows[1:]:
            if len(row) < 2:
                continue
            date_raw, value_raw = row[0].strip(), row[1].strip()
            if not date_raw or not value_raw or value_raw in {".", "NA", ""}:
                continue
            try:
                event_time = datetime.fromisoformat(date_raw).replace(tzinfo=timezone.utc)
                value = float(value_raw)
            except (ValueError, TypeError):
                continue
            out.append(
                PointInTimeRecord(
                    event_time=event_time,
                    available_at=event_time + timedelta(days=float(lag_days)),
                    source="fred",
                    entity_id=config.FRED_SERIES[sid][0],
                    field=config.FRED_SERIES[sid][1],
                    value=value,
                    revision_id=f"{sid}:{event_time.date().isoformat()}",
                )
            )

        # Monthly series repeat the same month-start date across revisions;
        # dedupe identical event times keeping the LAST row (latest vintage).
        deduped: dict[datetime, PointInTimeRecord] = {}
        for rec in out:
            deduped[rec.event_time] = rec
        return sorted(deduped.values(), key=lambda r: r.event_time)


# ---------------------------------------------------------------------------
# BLS -- labor market statistics (optional; empty default series map)
# ---------------------------------------------------------------------------


class BlsConnector(MacroConnectorBase):
    """Fetch BLS series via public API v2 (with key) or v1 (without)."""

    source_name = "bls"

    BLS_V2_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
    BLS_V1_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data"

    def _api_key(self) -> str:
        env = getattr(config, "BLS_API_KEY_ENV", "BLS_API_KEY")
        return os.environ.get(env, "").strip()

    def fetch_series(
        self,
        *,
        series_ids: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> list[PointInTimeRecord]:
        self.result = ConnectorResult(self.source_name)
        ids = [s for s in series_ids if s in config.BLS_SERIES]
        unknown = [s for s in series_ids if s not in config.BLS_SERIES]
        for sid in unknown:
            self.result.fail(f"Unknown BLS series '{sid}' (not in config.BLS_SERIES)")
        if not ids:
            return []

        payload = {
            "seriesid": ids,
            "startyear": str(_to_utc(start).year),
            "endyear": str(_to_utc(end).year),
        }
        key = self._api_key()
        headers = {"Content-Type": "application/json"}
        url = self.BLS_V1_URL
        if key:
            payload["registrationkey"] = key
            url = self.BLS_V2_URL

        body = json.dumps(payload).encode("utf-8")
        raw = self._fetch_bytes(url=url, headers=headers, method="POST", body=body)
        if raw is None:
            return []
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.result.fail(f"BLS JSON parse failure: {exc}")
            return []
        if str(data.get("status", "")).upper() != "REQUEST_SUCCEEDED":
            self.result.fail(f"BLS status: {data.get('status', 'unknown')}")
        return self._parse_payload(data)

    def _parse_payload(self, data: dict) -> list[PointInTimeRecord]:
        out: list[PointInTimeRecord] = []
        for series in (data.get("Results", {}) or {}).get("series", []) or []:
            sid = str(series.get("seriesID", ""))
            spec = config.BLS_SERIES.get(sid)
            if spec is None:
                continue
            entity_id, field_name, lag_days = spec
            best_by_period: dict[str, tuple[datetime, float]] = {}
            for obs in series.get("data", []) or []:
                year = str(obs.get("year", ""))
                period = str(obs.get("period", ""))
                value_raw = str(obs.get("value", "")).strip()
                # Monthly periods only (M01..M12); skip M13 annual averages.
                if not year or period not in {f"M{m:02d}" for m in range(1, 13)}:
                    continue
                if not value_raw or value_raw in {"-", "NA"}:
                    continue
                try:
                    month = int(period[1:])
                    event_time = datetime(
                        int(year), month, 1, tzinfo=timezone.utc
                    ) + timedelta(days=31)
                    event_time = event_time.replace(day=1) - timedelta(days=1)
                    value = float(value_raw)
                except (ValueError, TypeError):
                    continue
                # Keep the most recent update per period (latest vintage).
                prev = best_by_period.get(period)
                if prev is None:
                    best_by_period[period] = (event_time, value)

            for event_time, value in sorted(best_by_period.values()):
                out.append(
                    PointInTimeRecord(
                        event_time=event_time,
                        available_at=event_time + timedelta(days=float(lag_days)),
                        source="bls",
                        entity_id=entity_id,
                        field=field_name,
                        value=value,
                        revision_id=f"{sid}:{event_time.date().isoformat()}",
                    )
                )
        return out


# ---------------------------------------------------------------------------
# SEC EDGAR -- corporate filing timeline (filing-age features)
# ---------------------------------------------------------------------------


class SecEdgarConnector(MacroConnectorBase):
    """
    Fetch filing timelines per ticker from data.sec.gov.

    Two-step flow, both cached-friendly GETs:
      1. company_tickers.json  -> ticker -> CIK mapping
      2. submissions/CIK##########.json -> recent filings arrays

    A filing becomes ``available_at`` filing_date + a conservative lag
    (filings are mostly accepted after US market close; see config).
    """

    source_name = "sec"

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

    TRACKED_FORMS = {
        "10-K",
        "10-Q",
        "8-K",
        "6-K",
        "20-F",
        "S-1",
        "S-3",
        "424B5",
        "DEF 14A",
        "13D",
        "13G",
    }

    def _user_agent(self) -> str:
        env = getattr(config, "SEC_USER_AGENT_EMAIL_ENV", "SEC_CONTACT_EMAIL")
        email = os.environ.get(env, "").strip()
        contact = email or "financebot-operator@example.com"
        return f"financeBot ({contact})"

    def _ticker_to_cik(self, symbols: Iterable[str]) -> dict[str, int]:
        wanted = {s.upper() for s in symbols}
        raw = self._fetch_bytes(
            url=self.TICKERS_URL, headers={"User-Agent": self._user_agent()}
        )
        if raw is None:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.result.fail(f"EDGAR ticker map parse failure: {exc}")
            return {}
        mapping: dict[str, int] = {}
        for entry in (data or {}).values():
            try:
                ticker = str(entry.get("ticker", "")).upper()
                cik = int(entry.get("cik_str"))
            except (TypeError, ValueError, AttributeError):
                continue
            if ticker in wanted:
                mapping[ticker] = cik
        return mapping

    def fetch_filings(
        self,
        *,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> list[PointInTimeRecord]:
        self.result = ConnectorResult(self.source_name)
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        lag = timedelta(days=float(getattr(config, "SEC_FILING_AVAILABLE_LAG_DAYS", 0.75)))

        mapping = self._ticker_to_cik(symbols)
        missing = [s.upper() for s in symbols if s.upper() not in mapping]
        for sym in missing:
            # Not fatal: many tradables (ETFs, crypto pairs) have no CIK.
            self.result.fail(f"No CIK found for '{sym}' (skipped)")

        headers = {"User-Agent": self._user_agent()}
        for sym in sorted(mapping):
            cik = mapping[sym]
            raw = self._fetch_bytes(
                url=self.SUBMISSIONS_URL.format(cik=cik), headers=headers
            )
            if raw is None:
                continue
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.result.fail(f"EDGAR submissions parse failure for {sym}: {exc}")
                continue
            records = self._parse_submissions(
                data,
                symbol=sym,
                start=start_utc,
                end=end_utc,
                lag=lag,
            )
            if not records:
                self.result.fail(f"EDGAR returned no tracked filings for {sym}")
            self.result.records.extend(records)
        return list(self.result.records)

    def _parse_submissions(
        self,
        data: dict,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
        lag: timedelta,
    ) -> list[PointInTimeRecord]:
        """Emit one filing_date record per tracked recent filing."""
        out: list[PointInTimeRecord] = []
        recent = (data or {}).get("filings", {}).get("recent", {}) or {}
        forms = recent.get("form", []) or []
        dates = recent.get("filingDate", []) or []
        accessions = recent.get("accessionNumber", []) or []
        for form, filing_date_raw, accession in zip(forms, dates, accessions):
            try:
                # filingDate is ISO "YYYY-MM-DD" (sometimes with time).
                event_time = datetime.fromisoformat(str(filing_date_raw)[:10]).replace(
                    tzinfo=timezone.utc
                )
            except (ValueError, TypeError):
                continue
            if event_time < start or event_time > end:
                continue
            if form not in self.TRACKED_FORMS:
                continue
            out.append(
                PointInTimeRecord(
                    event_time=event_time,
                    available_at=event_time + lag,
                    source="sec",
                    entity_id=symbol,
                    field="filing_date",
                    value=str(accession),
                    revision_id=str(accession),
                )
            )
        return sorted(out, key=lambda r: r.event_time)


# ---------------------------------------------------------------------------
# Orchestration: bars + macro + filings -> PointInTimeFeatureStore
# ---------------------------------------------------------------------------


def _daily_bar_records(
    *,
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
) -> tuple[list[PointInTimeRecord], list[str]]:
    """
    Convert Alpaca hourly bars into PIT daily-close records (source='alpaca').

    Anti-lookahead rule: each day's record is stamped at the COMPLETION of its
    last hourly bar (event_time = last bar start + 1h) and available_at equals
    that moment, so an intraday query can never see the current day's close
    early. Reuses the legacy ``fetch_bars`` HTTP path (rate-limited there).
    """
    import pandas as pd

    from src.data import fetch_bars

    records: list[PointInTimeRecord] = []
    errors: list[str] = []
    for symbol in symbols:
        try:
            bars = fetch_bars(symbol)
        except Exception as exc:  # noqa: BLE001 -- data fetch must not crash build
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        if bars is None or getattr(bars, "empty", True):
            errors.append(f"{symbol}: no bars returned")
            continue
        try:
            if not isinstance(bars.index, pd.DatetimeIndex):
                errors.append(f"{symbol}: non-datetime bar index")
                continue
            idx = bars.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            closes = pd.Series(
                bars["close"].to_numpy(dtype="float64"), index=idx
            ).dropna()
            if closes.empty:
                errors.append(f"{symbol}: no usable closes")
                continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: normalization failed: {exc}")
            continue

        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        for _, day_series in closes.groupby(closes.index.date):
            if day_series.empty:
                continue
            last_bar_start = day_series.index[-1].to_pydatetime()
            event_time = last_bar_start + timedelta(hours=1)  # bar completion
            if event_time < start_utc or event_time > end_utc:
                continue
            records.append(
                PointInTimeRecord(
                    event_time=event_time,
                    available_at=event_time,
                    source="alpaca",
                    entity_id=symbol,
                    field="close",
                    value=float(day_series.iloc[-1]),
                    revision_id=event_time.date().isoformat(),
                )
            )
    return records, errors


def populate_feature_store(
    store,
    *,
    symbols: Iterable[str],
    lookback_days: int = 400,
    include_edgar: bool = True,
    connectors: dict | None = None,
) -> dict:
    """
    Populate a PointInTimeFeatureStore from all providers (offline research path).

    Every connector failure is captured, never raised: the store ends up with
    whatever succeeded and the summary reports what did not. Returns a summary
    dict {records, by_source, errors}.

    ``connectors`` allows tests to inject doubles; production builds defaults.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(lookback_days))
    symbols = [s for s in symbols]

    by_source: dict[str, int] = {}
    all_errors: list[str] = []

    def _absorb(records: list[PointInTimeRecord], source: str, errors: list[str]) -> None:
        accepted = store.upsert_records(records)
        by_source[source] = by_source.get(source, 0) + accepted
        all_errors.extend(f"[{source}] {e}" for e in errors)

    # 1) Market bars (Alpaca via legacy fetch path).
    bar_records, bar_errors = _daily_bar_records(symbols=symbols, start=start, end=now)
    _absorb(bar_records, "alpaca", bar_errors)

    # 2) FRED macro series.
    fred = (
        connectors.get("fred") if connectors else None
    ) or FredConnector()
    fred_records = fred.fetch_series(
        series_ids=tuple(config.FRED_SERIES), start=start, end=now
    )
    _absorb(fred_records, "fred", list(fred.result.errors))

    # 3) BLS labor series (empty default map -> zero network calls).
    bls = (connectors.get("bls") if connectors else None) or BlsConnector()
    bls_records = bls.fetch_series(
        series_ids=tuple(config.BLS_SERIES), start=start, end=now
    )
    _absorb(bls_records, "bls", list(bls.result.errors))

    # 4) SEC EDGAR filing timelines for equity tickers.
    if include_edgar:
        sec = (connectors.get("sec") if connectors else None) or SecEdgarConnector()
        equity_symbols = [s for s in symbols if "/" not in s]
        sec_records = sec.fetch_filings(
            symbols=equity_symbols, start=start, end=now
        )
        _absorb(sec_records, "sec", list(sec.result.errors))

    persisted = store.save_to_disk() if hasattr(store, "save_to_disk") else 0
    return {
        "records": sum(by_source.values()),
        "by_source": by_source,
        "persisted": persisted,
        "errors": all_errors,
    }
