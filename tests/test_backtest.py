from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest
import config


class StubModel:
    """Deterministic model: always emits a strong BUY probability."""

    def predict_up_proba(self, features_row: pd.DataFrame) -> float:
        return config.BUY_THRESHOLD + 0.1


def _bars_over_days(days: int = 5, per_day: int = 6) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start, periods=days * per_day, freq="h")
    price = 100.0
    closes: list[float] = []
    for i in range(len(index)):
        price += 0.8 if i % 3 else -0.6
        closes.append(price)
    close = pd.Series(closes, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000 + (i % 7) * 25 for i in range(len(index))],
        },
        index=index,
    )


def _sentiment_frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "ticker": ["SPY", "SPY", "SPY"],
            "score": [9.0, 2.0, 8.0],  # pass, block, pass
        }
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw["ticker"] = raw["ticker"].astype(str)
    raw["score"] = pd.to_numeric(raw["score"])
    return raw


def test_merge_sentiment_aligns_by_date_without_misalignment() -> None:
    bars = _bars_over_days()
    sentiment = _sentiment_frame()

    scores = backtest.merge_sentiment(bars, sentiment, "SPY")

    # 1:1 alignment with bars index -- no shifting or reindex drift.
    assert list(scores.index) == list(bars.index)
    assert len(scores) == len(bars)

    # Every bar maps to its OWN calendar day''s score.
    for ts in bars.index:
        day = pd.Timestamp(ts).date()
        expected = {
            pd.Timestamp("2026-01-01").date(): 9.0,
            pd.Timestamp("2026-01-02").date(): 2.0,
            pd.Timestamp("2026-01-03").date(): 8.0,
        }.get(day, np.nan)
        actual = scores.loc[ts]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == expected


def test_simulate_respects_historical_sentiment_gate() -> None:
    bars = _bars_over_days()
    sentiment = _sentiment_frame()
    model = StubModel()

    trades = backtest.simulate(bars, model, sentiment, "SPY", min_score=5.0)

    assert list(trades.columns) == ["timestamp", "ticker", "side", "price", "size", "pnl"]

    # No buy may ever be logged on a blocked (score < 5) calendar day.
    buys = trades[trades["side"] == "buy"]
    for ts in pd.to_datetime(buys["timestamp"]):
        assert ts.date() != pd.Timestamp("2026-01-02").date()


def test_simulate_crypto_symbol_uses_base_asset_sentiment() -> None:
    bars = _bars_over_days()
    sentiment = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01").date()],
            "ticker": ["BTC"],
            "score": [9.0],
        }
    )
    scores = backtest.merge_sentiment(bars, sentiment, "BTC/USD")
    day1 = [ts for ts in bars.index if pd.Timestamp(ts).date() == pd.Timestamp("2026-01-01").date()]
    assert all(scores.loc[ts] == 9.0 for ts in day1)


def test_load_historical_sentiment_reads_bundled_csv() -> None:
    df = backtest.load_historical_sentiment(config.HISTORICAL_SENTIMENT_PATH)
    assert {"date", "ticker", "score"}.issubset(df.columns)
    assert (df["score"].between(1, 10)).all()


def test_generate_historical_sentiment_aligns_to_bar_dates(tmp_path) -> None:
    bars = _bars_over_days(days=5, per_day=6)
    out = tmp_path / "historical_sentiment.csv"

    df = backtest.generate_historical_sentiment(
        {"BTC/USD": bars}, path=str(out), seed=7
    )

    # Base-asset keying: 'BTC/USD' -> 'BTC'.
    assert set(df["ticker"].unique()) == {"BTC"}

    # One row per unique calendar date in the bars -- exact alignment.
    bar_dates = sorted({pd.Timestamp(ts).date().isoformat() for ts in bars.index})
    assert sorted(df["date"].tolist()) == bar_dates

    # Scores are within the requested 1.0..10.0 band.
    assert df["score"].between(1.0, 10.0).all()

    # File persisted and reloads cleanly.
    reloaded = backtest.load_historical_sentiment(str(out))
    assert len(reloaded) == len(bar_dates)


def test_generate_is_deterministic_with_seed(tmp_path) -> None:
    bars = _bars_over_days()
    a = backtest.generate_historical_sentiment(
        {"AAPL": bars}, path=str(tmp_path / "a.csv"), seed=42
    )
    b = backtest.generate_historical_sentiment(
        {"AAPL": bars}, path=str(tmp_path / "b.csv"), seed=42
    )
    assert a["score"].tolist() == b["score"].tolist()


def test_generated_sentiment_yields_alignable_gate(tmp_path) -> None:
    """End-to-end: aligned sentiment must actually intersect the bars (no misses)."""
    bars = _bars_over_days(days=6, per_day=6)
    out = tmp_path / "hist.csv"
    backtest.generate_historical_sentiment({"SPY": bars}, path=str(out), seed=1)
    sentiment = backtest.load_historical_sentiment(str(out))

    scores = backtest.merge_sentiment(bars, sentiment, "SPY")
    # Every bar now maps to a real score -- zero NaNs (the original bug).
    assert scores.notna().all()


def test_generate_handles_multiple_symbols(tmp_path) -> None:
    bars = _bars_over_days()
    out = tmp_path / "multi.csv"
    df = backtest.generate_historical_sentiment(
        {"AAPL": bars, "MSFT": bars, "BTC/USD": bars, "ETH/USD": bars},
        path=str(out),
        seed=3,
    )
    assert set(df["ticker"].unique()) == {"AAPL", "MSFT", "BTC", "ETH"}
