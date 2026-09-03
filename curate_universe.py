"""
Weekly universe curator: the political-economical research engine.

Turns the static 47-symbol candidate pool into a fresh, macro-aware tradable
pool every week:

    1. QUANT SCREEN   per candidate: liquidity (avg $ volume), momentum,
                      volatility, drawdown, correlation to SPY
    2. MACRO BRIEF    FRED keyless series: curve level/slope, CPI YoY,
                      policy rate
    3. NEWS CORPUS    dated headlines: per-symbol Google News RSS + Fed /
                      Treasury / BLS policy feeds (src/news.py)
    4. GEMINI         one research call with an explicit epistemics frame:
                      training data declared stale, provided news wins
    5. CODE DISPOSES  deterministic post-validation -- pool membership,
                      liquidity floor, sector-cap breadth, BTC/ETH kept,
                      target size. The model proposes; arithmetic disposes.

Outputs (atomic, fail-closed -- any error keeps last week's pool trading):
    active_universe.json           schema v1, fresh as_of
    models/universe_rationale.json macro note + per-name rationale (audit)

Run: weekly via financebot-curate.timer (Sun 22:00 UTC) or manually:
    python curate_universe.py [--dry-run] [--size 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import config
from src.universe import FULL_CANDIDATE_UNIVERSE, normalize_symbols

# ---------------------------------------------------------------------------
# Sector grouping for breadth enforcement (LLM order kept; code enforces caps)
# ---------------------------------------------------------------------------

SECTOR_GROUPS: dict[str, str] = {
    # broad indices
    "SPY": "broad", "QQQ": "broad", "IWM": "broad", "DIA": "broad",
    # SPDR sectors (ETF + representative names share a group)
    "XLK": "tech", "AAPL": "tech", "MSFT": "tech", "NVDA": "tech",
    "AVGO": "tech", "AMD": "tech",
    "XLF": "financials", "JPM": "financials", "V": "financials", "MA": "financials",
    "XLE": "energy",
    "XLY": "cons_disc", "AMZN": "cons_disc", "TSLA": "cons_disc", "HD": "cons_disc",
    "XLP": "cons_staples", "COST": "cons_staples", "PG": "cons_staples",
    "KO": "cons_staples", "PEP": "cons_staples",
    "XLV": "health", "UNH": "health",
    "XLI": "industrials",
    "XLU": "utilities",
    "XLB": "materials",
    "XLRE": "reit", "VNQ": "reit",
    "XLC": "comm", "GOOGL": "comm", "GOOG": "comm", "NFLX": "comm",
    # rates & credit
    "TLT": "rates_credit", "IEF": "rates_credit", "SHY": "rates_credit",
    "HYG": "rates_credit", "LQD": "rates_credit",
    # metals / intl
    "GLD": "metals", "SLV": "metals",
    "EFA": "intl", "EEM": "intl",
}
CRYPTO = ("BTC/USD", "ETH/USD")


def _finite(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and abs(v) != float("inf")


# ---------------------------------------------------------------------------
# Layer 1: quant screen
# ---------------------------------------------------------------------------

def _quant_screen(symbols, *, fetcher=None) -> dict[str, dict]:
    """Liquidity / momentum / vol / drawdown / corr-to-SPY per symbol."""
    import numpy as np
    import pandas as pd

    from src.data import fetch_bars

    fetch = fetcher or fetch_bars
    stats: dict[str, dict] = {}
    spy_ret = None
    for sym in symbols:
        try:
            bars = fetch(sym, lookback_days=60)
        except Exception as exc:  # noqa: BLE001
            print(f"[curate] {sym}: bars fetch failed ({exc})")
            continue
        if bars is None or getattr(bars, "empty", True) or len(bars) < 30:
            print(f"[curate] {sym}: insufficient bars")
            continue
        close = bars["close"].astype("float64")
        vol = bars["volume"].astype("float64")
        # TRUE daily dollar volume: hourly bars undercount ~7x if averaged
        # directly; sum to calendar days first (exact), then average 20d.
        dollar_series = close * vol
        if isinstance(dollar_series.index, pd.DatetimeIndex):
            dollar_vol = float(dollar_series.resample("1D").sum().tail(20).mean())
        else:
            dollar_vol = float(dollar_series.tail(20).mean())
        ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0
        ret_60d = float(close.iloc[-1] / close.iloc[0] - 1.0)
        daily = close.pct_change().dropna()
        vol20 = float(daily.tail(20).std(ddof=1) * (252 ** 0.5)) if len(daily) >= 3 else 0.0
        dd = float(close.iloc[-1] / close.tail(60).max() - 1.0)
        row = {
            "avg_dollar_vol": round(dollar_vol),
            "ret_20d_pct": round(ret_20d * 100, 2),
            "ret_60d_pct": round(ret_60d * 100, 2),
            "ann_vol_pct": round(vol20 * 100, 2),
            "drawdown_60d_pct": round(dd * 100, 2),
        }
        if sym == "SPY":
            spy_ret = daily.tail(30)
            row["corr_spy"] = 1.0
        elif spy_ret is not None:
            aligned = pd.concat([daily.tail(30), spy_ret], axis=1, join="inner").dropna()
            if len(aligned) >= 10 and aligned.iloc[:, 0].std() > 0 and aligned.iloc[:, 1].std() > 0:
                row["corr_spy"] = round(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])), 2)
        if all(_finite(v) for v in row.values()):
            stats[sym] = row
    return stats


# ---------------------------------------------------------------------------
# Layer 2: macro brief (keyless FRED)
# ---------------------------------------------------------------------------

def _macro_brief(*, connector=None) -> dict:
    """Curve level/slope + CPI YoY + policy rate, vintage-lagged connectors."""
    fred = connector or __import__("src.macro", fromlist=["FredConnector"]).FredConnector()
    try:
        recs = fred.fetch_series(
            series_ids=("DGS2", "DGS10", "CPIAUCSL", "FEDFUNDS"),
            start=datetime.now(timezone.utc) - timedelta(days=400),
            end=datetime.now(timezone.utc),
        )
    except Exception as exc:  # noqa: BLE001
        return {"note": f"macro fetch failed: {exc}"}
    series: dict[str, list[tuple[datetime, float]]] = {}
    for r in recs:
        try:
            series.setdefault(r.entity_id, []).append((r.event_time, float(r.value)))
        except (TypeError, ValueError):
            continue
    out: dict = {}
    for ent, pts in series.items():
        pts.sort()
        out[ent] = pts[-1][1]
    try:
        if "US10Y" in out and "US2Y" in out:
            out["curve_10y_2y"] = round(out["US10Y"] - out["US2Y"], 3)
    except (KeyError, TypeError):
        pass
    out["as_of"] = datetime.now(timezone.utc).date().isoformat()
    return out


# ---------------------------------------------------------------------------
# Layer 3: news corpus
# ---------------------------------------------------------------------------

def _news_corpus(symbols, *, now=None) -> dict:
    from src import news as news_mod

    now = now or datetime.now(timezone.utc)
    corpus: dict = {"policy": [], "symbols": {}}
    policy = news_mod.fetch_policy_news(now=now)
    corpus["policy"] = [
        {"source": it.source, "title": it.title, "age_days": round(it.age_days(now), 1)}
        for it in policy
    ]
    for sym in symbols:
        items = news_mod.fetch_symbol_news(sym, now=now)
        if items:
            corpus["symbols"][sym] = [
                {"title": it.title, "age_days": round(it.age_days(now), 1)}
                for it in items
            ]
    return corpus


# ---------------------------------------------------------------------------
# Layer 4: the research call
# ---------------------------------------------------------------------------

def _build_prompt(quant: dict, macro: dict, corpus: dict, target_size: int) -> tuple[str, str]:
    system = (
        "You are a cautious macro research analyst curating a weekly tradable "
        "universe for a micro-size systematic trader. EPISTEMICS: today is "
        f"{datetime.now(timezone.utc).date().isoformat()}. Your training data is "
        "OUTDATED and must be treated as irrelevant background. Your ONLY "
        "research inputs are the quant statistics, FRED macro series, and DATED "
        "news headlines provided here. If a headline conflicts with anything "
        "you 'remember', the headline wins. Cite specific headlines in your "
        f"rationale. Select EXACTLY {target_size} symbols from the candidate "
        "list (BTC/USD and ETH/USD must be included). Respect the stated "
        "max-per-sector breadth cap. Respond with STRICT JSON only: "
        '{"symbols": [...], "rationale": {"SYMBOL": "<=20 words citing news"}, '
        '"macro_note": "<=60 words"}.'
    )
    user = json.dumps(
        {
            "macro": macro,
            "policy_news": corpus.get("policy", []),
            "candidates": [
                {
                    "symbol": sym,
                    **quant[sym],
                    "sector": _sector_of(sym),
                    "recent_news": corpus.get("symbols", {}).get(sym, []),
                }
                for sym in sorted(quant)
            ],
            "target_size": target_size,
            "max_per_sector": config.CURATOR_MAX_PER_SECTOR,
        }
    )
    return system, user


def _sector_of(sym: str) -> str:
    return SECTOR_GROUPS.get(sym.upper(), "other")


def _gemini_select(system: str, user: str, *, transport=None) -> dict:
    """One layered-parse research call (strict -> sanitize -> single retry)."""
    from generate_sentiment import _extract_json

    api_key = os.environ.get(config.LLM_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {config.LLM_API_KEY_ENV}; refusing to curate.")
    post = transport or _http_post_json
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Exception | None = None
    for attempt in range(1, max(int(config.SENTIMENT_LLM_MAX_ATTEMPTS), 1) + 1):
        if attempt > 1 and last_err is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid JSON "
                        f"({type(last_err).__name__}). Respond again with ONLY "
                        f'the JSON object: {{"symbols":[...],"rationale":{{...}},'
                        f'"macro_note":"..."}}.'
                    ),
                }
            )
        try:
            payload = post(
                f"{config.LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "financeBot",
                },
                body={
                    "model": config.LLM_MODEL,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_tokens": int(config.LLM_MAX_TOKENS),
                    # GLM hybrids REQUIRE reasoning (enabled:false -> HTTP 400);
                    # effort=low is the cross-model supported shape.
                    "reasoning_effort": "low",
                },
            )
            return _extract_json(str(payload["choices"][0]["message"]["content"]))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[curate] research attempt {attempt} failed: {exc}")
    raise RuntimeError(f"Research call failed after retries ({last_err})")


def _http_post_json(url: str, *, headers: dict, body: dict) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_SECONDS) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Layer 5: deterministic validation (the model proposes; code disposes)
# ---------------------------------------------------------------------------

def _validate_selection(
    raw_symbols, quant: dict, *, target_size: int
) -> tuple[list[str], list[str]]:
    """Apply hard constraints. Returns (final_pool, audit_lines)."""
    audit: list[str] = []
    allowed = {s.upper() for s in FULL_CANDIDATE_UNIVERSE}
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in raw_symbols or []:
        try:
            sym = normalize_symbols([raw])[0]
        except Exception:  # noqa: BLE001 -- UniverseError on garbage
            audit.append(f"drop malformed {raw!r}")
            continue
        if sym not in allowed:
            audit.append(f"drop {sym}: not in candidate pool")
            continue
        if sym in seen:
            continue
        seen.add(sym)
        ordered.append(sym)

    # Liquidity floor (quant, not model). Crypto majors are exempt: they are
    # structural members (24/7 market, strategy symmetry) and are force-kept.
    final: list[str] = []
    for sym in ordered:
        if sym in CRYPTO:
            final.append(sym)
            continue
        stats = quant.get(sym)
        if stats is None:
            audit.append(f"drop {sym}: no quant data")
            continue
        if stats["avg_dollar_vol"] < config.CURATOR_LIQUIDITY_FLOOR_USD:
            audit.append(
                f"drop {sym}: liquidity {stats['avg_dollar_vol']:,} < floor"
            )
            continue
        final.append(sym)

    # Sector breadth cap (crypto exempt).
    per_sector: dict[str, int] = {}
    breadth: list[str] = []
    for sym in final:
        if sym in CRYPTO:
            breadth.append(sym)
            continue
        sector = _sector_of(sym)
        if per_sector.get(sector, 0) >= config.CURATOR_MAX_PER_SECTOR:
            audit.append(f"drop {sym}: sector '{sector}' at cap")
            continue
        per_sector[sector] = per_sector.get(sector, 0) + 1
        breadth.append(sym)

    # Crypto force-include at the front (even if the model omitted them),
    # then truncate to target size.
    pool = list(CRYPTO) + [s for s in breadth if s not in CRYPTO]
    pool = pool[:target_size]
    for line in audit:
        print(f"[curate] {line}")
    return pool, audit


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def curate(*, target_size: int | None = None, transport=None, dry_run: bool = False) -> dict:
    """Full pipeline. Returns the audit dict; writes files unless dry-run."""
    target_size = int(target_size or config.UNIVERSE_CURATOR_TARGET_SIZE)
    candidates = [s for s in FULL_CANDIDATE_UNIVERSE]

    print(f"[curate] quant screen over {len(candidates)} candidates ...")
    quant = _quant_screen(candidates)
    print(f"[curate] {len(quant)} candidates passed data checks.")

    print("[curate] macro brief ...")
    macro = _macro_brief()

    print("[curate] news corpus (policy + per-symbol) ...")
    corpus = _news_corpus(candidates)

    system, user = _build_prompt(quant, macro, corpus, target_size)
    print("[curate] research call ...")
    selection = _gemini_select(system, user, transport=transport)

    pool, audit = _validate_selection(
        selection.get("symbols"), quant, target_size=target_size
    )
    if len(pool) < max(int(target_size * 0.6), 8):
        raise RuntimeError(
            f"Validated pool too small ({len(pool)}/{target_size}); keeping "
            f"previous universe (fail-closed)."
        )

    result = {
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "macro_note": str(selection.get("macro_note", ""))[:300],
        "rationale": {
            str(k).upper(): str(v)[:200] for k, v in (selection.get("rationale") or {}).items()
        },
        "audit": audit,
        "pool": pool,
    }

    if dry_run:
        print(f"[curate] DRY RUN -- pool ({len(pool)}): {pool}")
        return result

    # Atomic writes: universe file + rationale artifact.
    uni_tmp = f"{config.ACTIVE_UNIVERSE_PATH}.tmp"
    with open(uni_tmp, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": 1,
                "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbols": pool,
            },
            fh,
            indent=2,
        )
    os.replace(uni_tmp, config.ACTIVE_UNIVERSE_PATH)

    rat_path = os.path.join("models", "universe_rationale.json")
    os.makedirs("models", exist_ok=True)
    rat_tmp = f"{rat_path}.tmp"
    with open(rat_tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    os.replace(rat_tmp, rat_path)

    print(f"[curate] wrote {len(pool)}-symbol pool -> {config.ACTIVE_UNIVERSE_PATH}")
    print(f"[curate] macro_note: {result['macro_note']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    from src.universe import _resolve_cap

    try:
        result = curate(
            target_size=args.size or _resolve_cap(None) if args.size else None,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 -- fail-closed
        print(f"[curate] FAILED: {exc}")
        print("[curate] previous active_universe.json left untouched.")
        return 1

    for sym in result["pool"]:
        why = result["rationale"].get(sym, "")
        print(f"[curate]   {sym:8} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
