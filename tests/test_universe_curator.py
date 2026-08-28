"""Universe curator + news ingestion tests (all transports mocked, no network)."""

from __future__ import annotations

import json

import pytest

import config
from src.news import NewsItem, _parse_feed, fetch_policy_news, fetch_symbol_news
from curate_universe import (
    SECTOR_GROUPS,
    _sector_of,
    _validate_selection,
)

NOW = __import__("datetime").datetime(2026, 8, 28, tzinfo=__import__("datetime").timezone.utc)


# ---------------------------------------------------------------------------
# RSS parsing + news connectors
# ---------------------------------------------------------------------------

RSS_SAMPLE = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>NVDA surges on datacenter demand</title>
<pubDate>Thu, 27 Aug 2026 14:00:00 GMT</pubDate>
<link>https://example.com/a</link></item>
<item><title>NVDA old headline</title>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
<link>https://example.com/b</link></item>
</channel></rss>"""


def test_parse_feed_rss_extracts_items():
    items = _parse_feed(RSS_SAMPLE)
    assert len(items) == 2
    assert items[0].title.startswith("NVDA surges")
    assert items[0].published is not None and items[0].published.year == 2026


def test_symbol_news_filters_stale_and_caps(tmp_path):
    sleeps: list[float] = []
    items = fetch_symbol_news(
        "NVDA",
        lookback_days=7,
        max_items=5,
        http_get=lambda url, timeout=20.0: RSS_SAMPLE,
        sleeper=sleeps.append,
        now=NOW,
    )
    assert len(items) == 1  # the 2024 headline is filtered out
    assert items[0].source == "news:NVDA"
    assert len(sleeps) == 1  # rate limiting engaged


def test_dead_feed_degrades_to_empty_not_raise():
    def boom(url, timeout=20.0):
        raise OSError("down")

    assert fetch_symbol_news(
        "NVDA", http_get=boom, sleeper=lambda s: None, now=NOW
    ) == []
    assert fetch_policy_news(http_get=boom, sleeper=lambda s: None, now=NOW) == []


# ---------------------------------------------------------------------------
# Deterministic post-validation
# ---------------------------------------------------------------------------

def _stats(sym: str, dollar_vol: float = 500_000_000) -> dict:
    return {
        "avg_dollar_vol": dollar_vol,
        "ret_20d_pct": 1.0,
        "ret_60d_pct": 2.0,
        "ann_vol_pct": 20.0,
        "drawdown_60d_pct": -3.0,
        "corr_spy": 0.5,
    }


@pytest.fixture(autouse=True)
def _small_settings(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE_CURATOR_TARGET_SIZE", 32)
    monkeypatch.setattr(config, "CURATOR_LIQUIDITY_FLOOR_USD", 50_000_000)
    monkeypatch.setattr(config, "CURATOR_MAX_PER_SECTOR", 6)


def test_hallucinated_and_malformed_symbols_dropped():
    raw = ["NVDA", "FAKECOIN", "nvda", "x!"]
    pool, audit = _validate_selection(
        raw, {"NVDA": _stats("NVDA")}, target_size=32
    )
    assert pool[2] == "NVDA"  # after force-included BTC/ETH
    assert any("FAKECOIN" in a for a in audit)


def test_liquidity_floor_enforced_by_code():
    raw = ["NVDA", "SPY"]
    quant = {"NVDA": _stats("NVDA", 10_000_000), "SPY": _stats("SPY", 20_000_000_000)}
    pool, _ = _validate_selection(raw, quant, target_size=32)
    assert pool[2] == "SPY"  # NVDA floor-dropped; crypto force-kept


def test_sector_cap_forces_breadth(monkeypatch):
    monkeypatch.setattr(config, "CURATOR_MAX_PER_SECTOR", 3)
    raw = ["XLK", "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "NFLX", "SPY"]
    quant = {s: _stats(s) for s in raw}
    # 6 tech candidates vs cap 3 -> LLM-order priority keeps the first three.
    pool, audit = _validate_selection(raw, quant, target_size=32)
    tech = [s for s in pool if SECTOR_GROUPS.get(s) == "tech"]
    assert tech == ["XLK", "AAPL", "MSFT"]
    assert "NVDA" not in pool
    assert any("sector" in a for a in audit)
    assert "SPY" in pool


def test_crypto_always_included_and_first():
    raw = ["SPY", "NVDA"]
    quant = {s: _stats(s) for s in raw}
    # BTC/ETH absent from the model's proposal -> code force-includes them.
    pool, _ = _validate_selection(raw, quant, target_size=32)
    assert pool[0] == "BTC/USD" and pool[1] == "ETH/USD"


def test_truncation_to_target_size():
    raw = [s for s in SECTOR_GROUPS]  # every mapped symbol
    quant = {s: _stats(s) for s in raw}
    pool, _ = _validate_selection(raw, quant, target_size=12)
    assert len(pool) == 12
    assert pool[0] == "BTC/USD"


def test_sector_of_unknown_is_other():
    assert _sector_of("ZZZZ") == "other"
    assert _sector_of("NVDA") == "tech"
