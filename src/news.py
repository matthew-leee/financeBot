"""
News ingestion for the universe curator: keyless RSS feeds.

Two corpus layers feed the weekly political-economical research prompt:

  * PER-SYMBOL market news  -- Google News RSS ("SYMBOL stock"), last N days
  * POLICY/MACRO news       -- Fed press releases, Treasury, BLS feeds

Hard rules (AGENTS.md):
  * stdlib-only (urllib + xml.etree) -- no feedparser dependency
  * every feed wrapped in strict try/except: one dead feed degrades to [] and
    a journal line, never an exception
  * rate limited like every other network path (sleep between feeds)
  * transport + sleeper are injected seams; tests never touch the network
  * stale news is FILTERED by lookback so the corpus is always "latest"
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

import config

HttpGet = Callable[..., bytes]


def _default_http_get(url: str, *, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "financeBot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


POLICY_FEEDS: tuple[tuple[str, str], ...] = (
    ("federalreserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("treasury", "https://home.treasury.gov/news/press-releases/feed"),
    ("bls", "https://www.bls.gov/feed/news_release.rss"),
)


@dataclass(frozen=True)
class NewsItem:
    """One dated headline ready for the research prompt."""

    source: str
    title: str
    published: datetime | None
    link: str

    def age_days(self, now: datetime) -> float:
        if self.published is None:
            return float("inf")
        return (now - self.published).total_seconds() / 86400.0


def _parse_pubdate(raw: str | None) -> datetime | None:
    """RFC-822 (RSS) or ISO-8601 (Atom) -> aware UTC; None when unparseable."""
    if not raw:
        return None
    text = raw.strip()
    try:
        dt = parsedate_to_datetime(text)
        if dt is None:
            raise ValueError
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _local_name(tag: str) -> str:
    """Namespace-agnostic tag name (RSS <item> vs Atom <entry>)."""
    return tag.rsplit("}", 1)[-1].lower()


def _parse_feed(xml_bytes: bytes) -> list[NewsItem]:
    """Extract dated headlines from RSS 2.0 or Atom; defensive to oddities."""
    root = ET.fromstring(xml_bytes)
    items: list[NewsItem] = []
    for entry in root.iter():
        tag = _local_name(entry.tag)
        if tag not in ("item", "entry"):
            continue
        title = link = published_raw = ""
        for child in entry.iter():
            ctag = _local_name(child.tag)
            if ctag == "title" and not title:
                title = (child.text or "").strip()
            elif ctag == "link":
                href = child.attrib.get("href", "")
                link = (child.text or "").strip() or href or link
            elif ctag in ("pubdate", "published", "updated", "date") and not published_raw:
                published_raw = (child.text or "").strip()
        if title:
            items.append(
                NewsItem(
                    source="",
                    title=title,
                    published=_parse_pubdate(published_raw),
                    link=link,
                )
            )
    return items


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    out: list[NewsItem] = []
    for it in items:
        key = hashlib.sha1(it.title.lower().encode()).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _fetch_feed(
    url: str,
    *,
    source: str,
    lookback_days: float,
    max_items: int,
    now: datetime,
    http_get: HttpGet,
    sleeper: Callable[[float], None],
    delay: float,
) -> list[NewsItem]:
    try:
        if delay > 0:
            sleeper(delay)
        raw = http_get(url)
        items = _parse_feed(raw)
    except Exception as exc:  # noqa: BLE001 -- one dead feed is not fatal
        print(f"[news] {source} feed failed: {type(exc).__name__}: {exc}")
        return []
    fresh = []
    for it in items:
        if it.age_days(now) <= lookback_days:
            fresh.append(
                NewsItem(source=source, title=it.title, published=it.published, link=it.link)
            )
    fresh.sort(key=lambda it: it.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return _dedupe(fresh)[:max_items]


def fetch_symbol_news(
    symbol: str,
    *,
    lookback_days: int | None = None,
    max_items: int = 5,
    http_get: HttpGet | None = None,
    sleeper: Callable[[float], None] | None = None,
    now: datetime | None = None,
) -> list[NewsItem]:
    """Latest market headlines for one symbol via Google News RSS."""
    lookback = float(lookback_days or config.CURATOR_NEWS_LOOKBACK_DAYS)
    return _fetch_feed(
        (
            "https://news.google.com/rss/search?q="
            f"{symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        ),
        source=f"news:{symbol}",
        lookback_days=lookback,
        max_items=max_items,
        now=now or datetime.now(timezone.utc),
        http_get=http_get or _default_http_get,
        sleeper=sleeper or time.sleep,
        delay=config.API_CALL_DELAY_SECONDS,
    )


def fetch_policy_news(
    *,
    lookback_days: int | None = None,
    max_items_per_feed: int = 6,
    http_get: HttpGet | None = None,
    sleeper: Callable[[float], None] | None = None,
    now: datetime | None = None,
) -> list[NewsItem]:
    """Central-bank / fiscal-policy headlines (Fed, Treasury, BLS)."""
    out: list[NewsItem] = []
    for source, url in POLICY_FEEDS:
        out.extend(
            _fetch_feed(
                url,
                source=source,
                lookback_days=float(lookback_days or config.CURATOR_NEWS_LOOKBACK_DAYS),
                max_items=max_items_per_feed,
                now=now or datetime.now(timezone.utc),
                http_get=http_get or _default_http_get,
                sleeper=sleeper or time.sleep,
                delay=config.API_CALL_DELAY_SECONDS,
            )
        )
    return _dedupe(out)
