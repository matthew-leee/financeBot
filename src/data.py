"""
Data engineering: fetch historical bars from Alpaca and build a leakage-free
feature matrix + label.

Design rules that protect predictive validity:
  * Every feature is computed from information available *at or before* the bar
    it is attached to (no look-ahead).
  * The label looks strictly into the FUTURE (next bar return) and is therefore
    only known in hindsight -- it is used for training, never as a feature.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Canonical, ordered list of feature columns. The live path MUST reproduce these
# exact names in this exact order, so we persist it with the model too.
FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "sma_ratio_12",
    "sma_ratio_24",
    "rsi_14",
    "vol_12",
    "range_pct",
    "volume_z_24",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder-ish RSI. Uses only past data (rolling mean of gains/losses)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Rolling means are strictly backward-looking => no leakage.
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a raw OHLCV frame into the model feature matrix.

    Expects columns: open, high, low, close, volume (indexed by timestamp).
    Returns a frame containing exactly FEATURE_COLUMNS (plus close for labeling).
    """
    df = bars.copy()

    # Simple momentum features -- percentage returns over several horizons.
    # Why: short-horizon momentum/mean-reversion is the bread and butter of
    # tabular price models; multiple horizons let the tree pick the useful one.
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)

    # Price relative to its own moving averages (trend context, scale-free).
    df["sma_ratio_12"] = df["close"] / df["close"].rolling(12).mean() - 1.0
    df["sma_ratio_24"] = df["close"] / df["close"].rolling(24).mean() - 1.0

    # Momentum oscillator.
    df["rsi_14"] = _rsi(df["close"], 14)

    # Realized volatility of recent returns (regime signal).
    df["vol_12"] = df["ret_1"].rolling(12).std()

    # Intrabar range as a fraction of open (proxy for intrabar volatility).
    df["range_pct"] = (df["high"] - df["low"]) / df["open"].replace(0.0, np.nan)

    # Volume z-score vs its own recent history (is this bar unusually active?).
    vol_mean = df["volume"].rolling(24).mean()
    vol_std = df["volume"].rolling(24).std()
    df["volume_z_24"] = (df["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    keep = FEATURE_COLUMNS + ["close"]
    return df[keep]


def build_label(features: pd.DataFrame) -> pd.Series:
    """
    Binary label: 1 if the NEXT bar''s close is higher than this bar''s close.

    shift(-1) pulls a future value onto the current row -> this is the thing we
    predict. It is intentionally future-looking and is dropped from X.
    """
    future_ret = features["close"].shift(-1) / features["close"] - 1.0
    label = (future_ret > 0.0).astype("float64")
    # The final bar has no future close, so keep its label as NaN. This ensures
    # make_dataset() drops it instead of silently treating "unknown" as class 0.
    label[future_ret.isna()] = np.nan
    return label


def make_dataset(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble a clean, aligned (X, y) with all NaN rows removed."""
    feats = build_features(bars)
    label = build_label(feats)

    data = feats.copy()
    data["y"] = label
    # Drop warm-up NaNs (rolling windows) and the final row (no future label).
    data = data.dropna()

    x = data[FEATURE_COLUMNS].astype("float64")
    y = data["y"].astype("int64")
    return x, y


# ---------------------------------------------------------------------------
# Alpaca historical data fetch (wrapped defensively)
# ---------------------------------------------------------------------------

def fetch_bars(symbol: str, lookback_days: int = config.TRAIN_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Pull historical bars for one symbol. Returns an OHLCV DataFrame indexed by
    timestamp. Returns an empty frame on any failure (caller decides what to do).

    Note: uses the data clients directly (read-only market data). Order routing
    lives in src/broker.py.
    """
    from datetime import datetime, timedelta, timezone

    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    is_crypto = "/" in symbol

    try:
        if is_crypto:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame

            client = CryptoHistoricalDataClient()
            req = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
            )
            # Rate limit even read calls to stay a good API citizen.
            time.sleep(config.API_CALL_DELAY_SECONDS)
            bars = client.get_crypto_bars(req)
        else:
            import os

            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            client = StockHistoricalDataClient(
                api_key=os.environ["APCA_API_KEY_ID"],
                secret_key=os.environ["APCA_API_SECRET_KEY"],
            )
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
            )
            time.sleep(config.API_CALL_DELAY_SECONDS)
            bars = client.get_stock_bars(req)

        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()

        # Multi-index (symbol, timestamp) -> flatten to single symbol frame.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    except KeyError as exc:
        # Missing env var -> surface loudly; secrets are mandatory.
        print(f"[data] Missing required environment variable: {exc}")
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001 -- defensive: never let data kill the run
        print(f"[data] Failed to fetch bars for {symbol}: {exc}")
        return pd.DataFrame()


# ===========================================================================
# ADDITIVE: Point-in-Time (PIT) Data Layer for the Dual-Horizon Engine
# ===========================================================================
# Everything above (FEATURE_COLUMNS, fetch_bars, build_features, make_dataset)
# is the legacy single-horizon contract and is intentionally unchanged. The PIT
# layer below is the anti-lookahead boundary: every feature query enforces
# available_at <= as_of so revised/late-arriving data can never leak backward.

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Protocol

AssetClass = Literal["equity", "crypto", "etf", "inverse_etf", "macro_series"]
FeatureHorizon = Literal["interday", "intraday"]

# Trading-day spacing used to translate calendar lookbacks into row counts.
_TRADING_DAYS_PER_YEAR = 252


def _to_utc(ts: datetime) -> datetime:
    """Normalize any datetime to timezone-aware UTC (naive assumed UTC)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class PointInTimeRecord:
    """Atomic point-in-time input record (see Architecture Module A)."""

    event_time: datetime
    available_at: datetime
    source: str
    entity_id: str
    field: str
    value: Any
    revision_id: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class NewsEvent:
    """Entity-linked news / filing / alt-data text event with an embedding."""

    event_id: str
    event_time: datetime
    available_at: datetime
    source: str
    text_hash: str
    tickers: tuple[str, ...]
    themes: tuple[str, ...]
    embedding: np.ndarray
    source_weight: float
    event_weight: float


@dataclass(frozen=True)
class FeatureSnapshot:
    """Immutable model input matrix generated as of one decision timestamp."""

    frame: pd.DataFrame
    as_of: datetime
    horizon: FeatureHorizon
    lineage: dict[str, str]


class DataConnector(Protocol):
    """Provider boundary for external data. Implementations must be mockable."""

    def fetch_raw(
        self,
        *,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> list[PointInTimeRecord]:
        """Fetch raw provider records with event_time and available_at set."""
        ...


def build_news_embedding_state(
    events: Iterable[NewsEvent],
    *,
    symbol: str,
    as_of: datetime,
    tau: timedelta,
    embedding_dim: int,
) -> tuple[np.ndarray, float]:
    """
    Build a time-decayed news state without lookahead.

        raw_state = Σ_s exp(-(t - available_at_s)/τ) * sw_s * ew_s * embedding_s
        mass      = Σ_s |exp(-(t - available_at_s)/τ) * sw_s * ew_s|
        direction = raw_state / ||raw_state||  (zeros if norm == 0)
        intensity = log1p(mass)

    Only events with symbol in tickers and available_at <= as_of contribute.
    """
    as_of_utc = _to_utc(as_of)
    tau_seconds = max(tau.total_seconds(), 1e-9)
    raw_state = np.zeros(int(embedding_dim), dtype="float64")
    mass = 0.0

    for ev in events:
        if symbol not in ev.tickers:
            continue
        available = _to_utc(ev.available_at)
        if available > as_of_utc:
            continue  # not knowable yet -> hard anti-lookahead gate
        age_seconds = (as_of_utc - available).total_seconds()
        decay = np.exp(-age_seconds / tau_seconds)
        weight = decay * float(ev.source_weight) * float(ev.event_weight)
        emb = np.asarray(ev.embedding, dtype="float64")
        if emb.shape[0] != raw_state.shape[0]:
            # Defensive: skip mis-sized embeddings rather than corrupt the state.
            continue
        raw_state += weight * emb
        mass += abs(weight)

    norm = float(np.linalg.norm(raw_state))
    direction = raw_state / norm if norm > 0 else np.zeros_like(raw_state)
    intensity = float(np.log1p(mass))
    return direction, intensity


class PointInTimeFeatureStore:
    """
    Local point-in-time feature store and anti-lookahead boundary.

    Records are held in memory keyed by their dedupe tuple and optionally
    persisted under storage_path. No model, strategist, executor, validator, or
    backtester may read external data directly once this exists.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self.storage_path = storage_path or config.FEATURE_STORE_PATH
        # Dedupe key -> record. Key = (source, entity_id, field, event_time, revision_id)
        self._by_key: dict[tuple, PointInTimeRecord] = {}
        self._news: dict[str, NewsEvent] = {}

    # -- read-only introspection --------------------------------------------

    @property
    def record_count(self) -> int:
        """Number of stored point-in-time records (usable for allocations)."""
        return len(self._by_key)

    def is_empty(self) -> bool:
        """True when there are no usable point-in-time records nor news events.

        The dual engine uses this as a fail-closed precondition: with no
        persisted point-in-time data it can only emit empty allocations, so it
        must refuse to start rather than trade on nothing.
        """
        return not self._by_key and not self._news

    # -- ingestion -----------------------------------------------------------

    def upsert_records(self, records: Iterable[PointInTimeRecord]) -> int:
        """Insert/update records, deduplicated by vintage key. Returns accepted."""
        accepted = 0
        for rec in records:
            conf = float(rec.confidence)
            if not (0.0 <= conf <= 1.0):
                print(f"[pit] Rejecting record with confidence {conf} out of [0,1].")
                continue
            norm = PointInTimeRecord(
                event_time=_to_utc(rec.event_time),
                available_at=_to_utc(rec.available_at),
                source=rec.source,
                entity_id=rec.entity_id,
                field=rec.field,
                value=rec.value,
                revision_id=rec.revision_id,
                confidence=conf,
            )
            key = (
                norm.source,
                norm.entity_id,
                norm.field,
                norm.event_time,
                norm.revision_id,
            )
            self._by_key[key] = norm
            accepted += 1
        return accepted

    def upsert_news(self, events: Iterable[NewsEvent]) -> int:
        """Insert/update news events keyed by event_id."""
        accepted = 0
        for ev in events:
            self._news[ev.event_id] = ev
            accepted += 1
        return accepted

    # -- eligibility helpers -------------------------------------------------

    def _eligible(
        self,
        *,
        entity_ids: set[str],
        fields: set[str] | None,
        as_of: datetime,
        lookback: timedelta | None,
    ) -> list[PointInTimeRecord]:
        as_of_utc = _to_utc(as_of)
        floor = as_of_utc - lookback if lookback is not None else None
        out: list[PointInTimeRecord] = []
        for rec in self._by_key.values():
            if rec.available_at > as_of_utc:
                continue
            if rec.entity_id not in entity_ids:
                continue
            if fields is not None and rec.field not in fields:
                continue
            if floor is not None and rec.event_time < floor:
                continue
            out.append(rec)
        return out

    def _series(
        self,
        *,
        entity_id: str,
        field: str,
        as_of: datetime,
        lookback: timedelta | None = None,
    ) -> pd.Series:
        """Point-in-time value series for one entity/field (latest vintage per event)."""
        recs = self._eligible(
            entity_ids={entity_id}, fields={field}, as_of=as_of, lookback=lookback
        )
        # Per event_time, keep the latest available vintage as of as_of.
        by_event: dict[datetime, PointInTimeRecord] = {}
        for rec in recs:
            cur = by_event.get(rec.event_time)
            if cur is None or rec.available_at > cur.available_at:
                by_event[rec.event_time] = rec
        if not by_event:
            return pd.Series(dtype="float64")
        items = sorted(by_event.items(), key=lambda kv: kv[0])
        idx = [k for k, _ in items]
        vals = [pd.to_numeric(v.value, errors="coerce") for _, v in items]
        return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype="float64")

    def query_asof(
        self,
        *,
        entity_ids: Iterable[str],
        fields: Iterable[str] | None,
        as_of: datetime,
        lookback: timedelta | None = None,
    ) -> pd.DataFrame:
        """
        Return latest eligible observation per entity/field as of `as_of`.

        Missing fields become explicit `<field>_missing` indicator columns; no
        value is ever backfilled from the future.
        """
        entity_set = set(entity_ids)
        field_set = set(fields) if fields is not None else None
        recs = self._eligible(
            entity_ids=entity_set, fields=field_set, as_of=as_of, lookback=lookback
        )

        # (entity, field) -> latest by (event_time, available_at).
        best: dict[tuple[str, str], PointInTimeRecord] = {}
        for rec in recs:
            key = (rec.entity_id, rec.field)
            cur = best.get(key)
            if cur is None or (rec.event_time, rec.available_at) > (
                cur.event_time,
                cur.available_at,
            ):
                best[key] = rec

        columns = sorted(field_set) if field_set is not None else sorted(
            {f for _, f in best}
        )
        index = sorted(entity_set)
        frame = pd.DataFrame(index=index, columns=columns, dtype="float64")
        for (entity, field), rec in best.items():
            if field in frame.columns and entity in frame.index:
                frame.loc[entity, field] = pd.to_numeric(rec.value, errors="coerce")

        # Missing indicators BEFORE any fill: 1.0 where absent, else 0.0.
        for col in list(frame.columns):
            frame[f"{col}_missing"] = frame[col].isna().astype("float64")
        return frame

    # -- snapshot builders ---------------------------------------------------

    def build_interday_snapshot(
        self,
        *,
        universe: Iterable[str],
        as_of: datetime,
        news_tau: timedelta,
    ) -> FeatureSnapshot:
        """Build daily strategic features for the strategist (see Module A)."""
        as_of_utc = _to_utc(as_of)
        universe = list(universe)
        price_lookback = timedelta(days=400)
        news_events = list(self._news.values())
        embedding_dim = (
            int(np.asarray(news_events[0].embedding).shape[0]) if news_events else 1
        )

        rows: dict[str, dict[str, float]] = {}
        for symbol in universe:
            close = self._series(
                entity_id=symbol,
                field="close",
                as_of=as_of_utc,
                lookback=price_lookback,
            ).dropna()
            feats: dict[str, float] = {}
            if not close.empty:
                feats["ret_20"] = _horizon_return(close, 20)
                feats["ret_63"] = _horizon_return(close, 63)
                feats["ret_126"] = _horizon_return(close, 126)
                feats["ret_252"] = _horizon_return(close, 252)
                daily = close.pct_change().dropna()
                feats["vol_20"] = _annualized_vol(daily, 20)
                feats["vol_63"] = _annualized_vol(daily, 63)
                feats["vol_126"] = _annualized_vol(daily, 126)
                feats["drawdown_252"] = _drawdown(close, 252)

            # Point-in-time macro / yield curve (shared across symbols).
            y3m = self._latest_value(symbol="US3M", field="yield", as_of=as_of_utc)
            y2 = self._latest_value(symbol="US2Y", field="yield", as_of=as_of_utc)
            y5 = self._latest_value(symbol="US5Y", field="yield", as_of=as_of_utc)
            y10 = self._latest_value(symbol="US10Y", field="yield", as_of=as_of_utc)
            if y10 is not None and y2 is not None:
                feats["curve_10y_2y"] = y10 - y2
            if y10 is not None and y3m is not None:
                feats["curve_10y_3m"] = y10 - y3m
            if y10 is not None and y2 is not None and y5 is not None:
                feats["curve_curvature"] = 2.0 * y5 - y2 - y10

            # SEC filing age (days since latest filing available as of as_of).
            filing_age = self._filing_age_days(symbol=symbol, as_of=as_of_utc)
            if filing_age is not None:
                feats["filing_age_days"] = filing_age

            _direction, intensity = build_news_embedding_state(
                news_events,
                symbol=symbol,
                as_of=as_of_utc,
                tau=news_tau,
                embedding_dim=embedding_dim,
            )
            feats["news_intensity"] = intensity

            rows[symbol] = feats

        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame = frame.sort_index()
        frame = _add_missing_indicators(frame)

        lineage = {
            "as_of": as_of_utc.isoformat(),
            "horizon": "interday",
            "n_symbols": str(len(universe)),
            "store": self.storage_path,
        }
        return FeatureSnapshot(
            frame=frame, as_of=as_of_utc, horizon="interday", lineage=lineage
        )

    def build_intraday_snapshot(
        self,
        *,
        symbols: Iterable[str],
        as_of: datetime,
        lookback_minutes: int,
    ) -> FeatureSnapshot:
        """Build intraday tactical execution-timing features (see Module A)."""
        as_of_utc = _to_utc(as_of)
        symbols = list(symbols)
        lookback = timedelta(minutes=lookback_minutes)

        rows: dict[str, dict[str, float]] = {}
        for symbol in symbols:
            close = self._series(
                entity_id=symbol, field="close", as_of=as_of_utc, lookback=lookback
            ).dropna()
            vwap = self._latest_value(symbol=symbol, field="vwap", as_of=as_of_utc)
            bid = self._latest_value(symbol=symbol, field="bid", as_of=as_of_utc)
            ask = self._latest_value(symbol=symbol, field="ask", as_of=as_of_utc)
            order_flow = self._latest_value(
                symbol=symbol, field="order_flow_imbalance", as_of=as_of_utc
            )
            volume_pct = self._latest_value(
                symbol=symbol, field="volume_percentile", as_of=as_of_utc
            )

            feats: dict[str, float] = {}
            last_price = float(close.iloc[-1]) if not close.empty else float("nan")
            feats["last_price"] = last_price
            if len(close) >= 2:
                feats["ret_1m"] = float(close.iloc[-1] / close.iloc[-2] - 1.0)
            if len(close) >= 6:
                feats["ret_5m"] = float(close.iloc[-1] / close.iloc[-6] - 1.0)
            if len(close) >= 2:
                minute_ret = close.pct_change().dropna()
                feats["realized_vol_30m"] = _annualized_vol(minute_ret, 30, 390)
                feats["realized_vol_60m"] = _annualized_vol(minute_ret, 60, 390)
            if vwap is not None and vwap > 0 and not np.isnan(last_price):
                feats["vwap_distance"] = last_price / vwap - 1.0
            if bid is not None and ask is not None and (bid + ask) > 0:
                midpoint = (bid + ask) / 2.0
                feats["spread_bps"] = (ask - bid) / midpoint * 10000.0
            if order_flow is not None:
                feats["order_flow_imbalance"] = order_flow
            if volume_pct is not None:
                feats["volume_percentile"] = volume_pct

            rows[symbol] = feats

        frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        frame = _add_missing_indicators(frame, skip={"last_price"})
        lineage = {
            "as_of": as_of_utc.isoformat(),
            "horizon": "intraday",
            "lookback_minutes": str(lookback_minutes),
        }
        return FeatureSnapshot(
            frame=frame, as_of=as_of_utc, horizon="intraday", lineage=lineage
        )

    # -- small point-in-time helpers ----------------------------------------

    def _latest_value(
        self, *, symbol: str, field: str, as_of: datetime
    ) -> float | None:
        series = self._series(entity_id=symbol, field=field, as_of=as_of)
        if series.empty:
            return None
        val = float(series.iloc[-1])
        return None if np.isnan(val) else val

    def _filing_age_days(self, *, symbol: str, as_of: datetime) -> float | None:
        recs = self._eligible(
            entity_ids={symbol}, fields={"filing_date"}, as_of=as_of, lookback=None
        )
        if not recs:
            return None
        latest = max(recs, key=lambda r: r.event_time)
        return (_to_utc(as_of) - latest.event_time).total_seconds() / 86400.0

    def return_panel(
        self, *, universe: Iterable[str], as_of: datetime, lookback_days: int = 400
    ) -> pd.DataFrame:
        """Wide point-in-time daily-return panel (columns=symbols) up to as_of."""
        cols: dict[str, pd.Series] = {}
        for symbol in universe:
            close = self._series(
                entity_id=symbol,
                field="close",
                as_of=as_of,
                lookback=timedelta(days=lookback_days),
            ).dropna()
            if not close.empty:
                cols[symbol] = close.pct_change().dropna()
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()


def _horizon_return(close: pd.Series, horizon: int) -> float:
    if len(close) <= horizon:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - horizon] - 1.0)


def _annualized_vol(
    returns: pd.Series, window: int, periods_per_year: int = _TRADING_DAYS_PER_YEAR
) -> float:
    if len(returns) < 2:
        return float("nan")
    window_returns = returns.iloc[-window:]
    if len(window_returns) < 2:
        return float("nan")
    return float(window_returns.std(ddof=1) * np.sqrt(periods_per_year))


def _drawdown(close: pd.Series, window: int) -> float:
    if close.empty:
        return float("nan")
    window_close = close.iloc[-window:]
    peak = float(window_close.max())
    if peak <= 0:
        return float("nan")
    return float(window_close.iloc[-1] / peak - 1.0)


def _add_missing_indicators(
    frame: pd.DataFrame, skip: set[str] | None = None
) -> pd.DataFrame:
    """Append `<col>_missing` indicators, then fill NaNs with 0.0 (no backfill)."""
    skip = skip or set()
    if frame.empty:
        return frame
    for col in list(frame.columns):
        if col in skip or col.endswith("_missing"):
            continue
        frame[f"{col}_missing"] = frame[col].isna().astype("float64")
    fill_cols = [c for c in frame.columns if c not in skip]
    frame[fill_cols] = frame[fill_cols].fillna(0.0)
    return frame
