from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data import FEATURE_COLUMNS, build_features, fetch_bars, make_dataset


def _install_fake_crypto_alpaca(monkeypatch, bars_df: pd.DataFrame) -> None:
    """Install just enough fake alpaca-py modules for fetch_bars() to import."""

    class FakeBars:
        df = bars_df

    class FakeCryptoHistoricalDataClient:
        def get_crypto_bars(self, request):  # noqa: ANN001 - mirrors SDK call shape
            assert request.symbol_or_symbols == "BTC/USD"
            return FakeBars()

    class FakeCryptoBarsRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    alpaca = types.ModuleType("alpaca")
    data_pkg = types.ModuleType("alpaca.data")
    historical = types.ModuleType("alpaca.data.historical")
    requests = types.ModuleType("alpaca.data.requests")
    timeframe = types.ModuleType("alpaca.data.timeframe")

    historical.CryptoHistoricalDataClient = FakeCryptoHistoricalDataClient
    requests.CryptoBarsRequest = FakeCryptoBarsRequest
    timeframe.TimeFrame = types.SimpleNamespace(Hour="1Hour")

    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.data", data_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.data.historical", historical)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", timeframe)


def _sample_bars(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        periods=rows,
        freq="h",
    )
    price = 100.0
    closes: list[float] = []
    for i in range(rows):
        # Alternate gains/losses so RSI and rolling volatility are well-defined.
        price += 0.8 if i % 3 else -0.6
        closes.append(price)
    close = pd.Series(closes, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000 + (i % 7) * 25 for i in range(rows)],
        },
        index=index,
    )


def test_fetch_bars_normalizes_mocked_crypto_multiindex(monkeypatch) -> None:
    base = _sample_bars(3)
    multi_index = pd.MultiIndex.from_product(
        [["BTC/USD"], base.index], names=["symbol", "timestamp"]
    )
    mocked_df = base.copy()
    mocked_df.index = multi_index

    _install_fake_crypto_alpaca(monkeypatch, mocked_df)
    monkeypatch.setattr("time.sleep", lambda _: None)

    result = fetch_bars("BTC/USD", lookback_days=1)

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert not isinstance(result.index, pd.MultiIndex)
    assert result.index.is_monotonic_increasing
    assert result.iloc[-1]["close"] == base.iloc[-1]["close"]


def test_make_dataset_excludes_future_values_from_features() -> None:
    bars = _sample_bars(40)
    x, y = make_dataset(bars)

    feature_frame = build_features(bars).dropna()
    expected_rows = len(feature_frame) - 1  # the final row has no future label

    assert list(x.columns) == FEATURE_COLUMNS
    assert "close" not in x.columns
    assert "y" not in x.columns
    assert len(x) == expected_rows
    assert len(y) == expected_rows

    first_feature_ts = x.index[0]
    current_close = bars.loc[first_feature_ts, "close"]
    previous_close = bars.shift(1).loc[first_feature_ts, "close"]
    next_close = bars.shift(-1).loc[first_feature_ts, "close"]

    assert x.iloc[0]["ret_1"] == (current_close / previous_close) - 1.0
    assert y.iloc[0] == int(next_close > current_close)

