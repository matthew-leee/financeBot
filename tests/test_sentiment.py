from __future__ import annotations

import json

import config
from src.sentiment import get_score, is_trade_allowed, load_sentiment


def test_load_sentiment_missing_file_returns_empty(tmp_path) -> None:
    missing = tmp_path / "nope.json"
    assert load_sentiment(str(missing)) == {}


def test_load_sentiment_bad_json_returns_empty(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert load_sentiment(str(bad)) == {}


def test_get_score_supports_nested_and_flat_shapes() -> None:
    nested = {"SPY": {"score": 7.2}}
    flat = {"SPY": 6.1}
    assert get_score(nested, "SPY") == 7.2
    assert get_score(flat, "SPY") == 6.1
    assert get_score({}, "SPY") is None


def test_get_score_crypto_falls_back_to_base_asset() -> None:
    report = {"BTC": {"score": 8.0}}
    assert get_score(report, "BTC/USD") == 8.0


def test_is_trade_allowed_threshold_and_fail_closed() -> None:
    report = {"SPY": {"score": 5.0}, "AAPL": {"score": 4.99}}
    assert is_trade_allowed(report, "SPY", min_score=5.0) is True
    assert is_trade_allowed(report, "AAPL", min_score=5.0) is False
    # Missing symbol must fail CLOSED (blocked), never open.
    assert is_trade_allowed(report, "MSFT", min_score=5.0) is False


def test_bundled_daily_sentiment_file_is_valid() -> None:
    with open(config.DAILY_SENTIMENT_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "SPY" in data
    score = get_score(data, "SPY")
    assert score is not None
    assert 1.0 <= score <= 10.0


def _corr_bars(returns):
    import numpy as np
    import pandas as pd

    n = len(returns)
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.cumprod(1.0 + np.asarray(returns)), index=index, dtype="float64")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": [1000 + i for i in range(n)]},
        index=index,
    )


def test_daily_returns_empty_on_bad_input() -> None:
    import pandas as pd

    from src.sentiment import daily_returns

    assert daily_returns(pd.DataFrame()).empty
    assert daily_returns(None).empty


def test_select_hedge_asset_picks_strongest_negative_correlation() -> None:
    import numpy as np

    from src.sentiment import select_hedge_asset

    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, 40)
    universe = {
        "AAPL": _corr_bars(r),
        "PSQ": _corr_bars(-r),                       # corr -1 -> selected
        "SH": _corr_bars(-0.4 * r + rng.normal(0, 0.01, 40)),
        "BITI": _corr_bars(r.copy()),                # corr +1
        "SARK": _corr_bars(np.zeros(40)),            # NaN -> skipped
    }

    def fetch(symbol, lookback_days):
        import pandas as pd
        return universe.get(symbol, pd.DataFrame())

    chosen = select_hedge_asset(
        "AAPL", safe_list=["SH", "PSQ", "BITI", "SARK"], bar_fetcher=fetch
    )
    assert chosen == "PSQ"


def test_select_hedge_asset_returns_none_when_no_candidate_data() -> None:
    import pandas as pd

    from src.sentiment import select_hedge_asset

    def fetch(symbol, lookback_days):
        if symbol == "AAPL":
            return _corr_bars([0.01, -0.01, 0.02, -0.02, 0.01, 0.0, 0.03])
        return pd.DataFrame()

    assert select_hedge_asset("AAPL", bar_fetcher=fetch) is None


def test_select_hedge_asset_none_when_target_missing() -> None:
    import pandas as pd

    from src.sentiment import select_hedge_asset

    assert select_hedge_asset("AAPL", bar_fetcher=lambda s, lookback_days: pd.DataFrame()) is None
