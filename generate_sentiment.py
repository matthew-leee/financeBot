"""
Daily sentiment generator: the "morning helper" behind daily_sentiment.json.

Reads the active universe, builds a compact momentum context per symbol from
recent bars, asks an OpenAI-compatible LLM (default: OpenRouter / Gemini 3.7
Flash) for a strict-JSON mood report, and atomically writes the EXACT schema
the trading loop already consumes via src.sentiment.load_sentiment():

    {"SPY": {"date": "2026-08-26", "score": 7.2, "summary": "..."}, ...}

Hard rules (AGENTS.md):
  * API key from environment only (config.LLM_API_KEY_ENV -> OPENAI_API_KEY).
    Missing key -> fail closed BEFORE any network call.
  * Failure never corrupts the existing report: write is atomic (tmp+replace)
    and only attempted after a fully valid report exists in memory.
  * Rate limited like every other network path.
  * Symbols the LLM omits are simply absent -> the loop's neutral-pass
    semantics decide their fate (config.SENTIMENT_MISSING_IS_PASS).

Run manually or via deploy/financebot-sentiment.timer (weekdays 10:30 UTC).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import config


def _http_post_json(url: str, *, headers: dict, body: dict) -> dict:
    """Injected transport seam: real POST used only outside tests."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_SECONDS) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _momentum_context(symbol: str, lookback_days: int) -> dict | None:
    """Compact, numeric context for one symbol; None when data unusable."""
    import pandas as pd

    from src.data import fetch_bars
    from src.sentiment import daily_returns

    try:
        bars = fetch_bars(symbol, lookback_days=lookback_days)
    except Exception as exc:  # noqa: BLE001 -- one bad symbol must not kill run
        print(f"[sentgen] {symbol}: fetch failed ({exc})")
        return None
    if bars is None or bars.empty or len(bars) < 6:
        print(f"[sentgen] {symbol}: insufficient bars")
        return None

    close = bars["close"]
    last = float(close.iloc[-1])
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0
    daily_ret = daily_returns(bars)
    vol_20d = (
        float(daily_ret.iloc[-20:].std(ddof=1) * (252 ** 0.5))
        if len(daily_ret) >= 3
        else 0.0
    )
    if not all(_finite(x) for x in (last, ret_5d, ret_20d, vol_20d)):
        return None
    return {
        "last_close": round(last, 4),
        "ret_5d_pct": round(ret_5d * 100.0, 2),
        "ret_20d_pct": round(ret_20d * 100.0, 2),
        "ann_vol_pct": round(vol_20d * 100.0, 2),
    }


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _build_prompt(contexts: dict[str, dict]) -> tuple[str, str]:
    system = (
        "You are a cautious markets sentiment analyst for a micro-size "
        "algorithmic trader. Score each symbol's CURRENT short-term outlook "
        "(next ~5-10 sessions) from 0 (very bearish) to 10 (very bullish), "
        "weighing momentum, volatility, and general market regime. Be "
        "conservative near extremes. Respond with STRICT JSON only: an object "
        'mapping each given symbol to {"score": <number 0-10>, "summary": '
        "<=12 words}. No extra keys, no commentary."
    )
    user = json.dumps({"as_of": _today(), "contexts": contexts})
    return system, user


def _extract_json(text: str) -> dict:
    """Layered parse: strict, then sanitize pass (trailing commas etc.)."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError as first_err:
        sanitized = _sanitize_json(blob)
        return json.loads(sanitized)  # may raise with clearer context


def _sanitize_json(blob: str) -> str:
    """Repair the most common LLM JSON slips without changing semantics.

    Handles trailing commas before } or ], stray control characters inside
    strings (except \n and \t which we escape), and smart-quote substitutes.
    """
    import re

    cleaned = re.sub(r",\s*([}\]])", r"\1", blob)
    cleaned = "".join(
        ch if ch >= " " or ch in "\n\t" else " " for ch in cleaned
    )
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    return cleaned


_SALVAGE_RE = None  # compiled lazily


def _salvage_entries(blob: str) -> dict:
    """
    Last-resort extraction of well-formed {SYM: {"score": x, ...}} fragments
    from otherwise-unparseable output. Returns whatever valid pieces exist --
    partial coverage is acceptable because neutral-pass semantics govern gaps.
    """
    import re

    out: dict[str, tuple[float, str]] = {}
    pattern = re.compile(
        r'"([A-Za-z]{2,10})"\s*:\s*\{[^{}]*?"score"\s*:\s*'
        r'"?([0-9]+(?:\.[0-9]+)?)"?[^{}]*?(?:"summary"\s*:\s*"([^"]*)")?[^{}]*?\}',
        re.DOTALL,
    )
    for sym, score_raw, summary in pattern.findall(blob):
        try:
            score = max(0.0, min(10.0, float(score_raw)))
        except ValueError:
            continue
        if not _finite(score):
            continue
        out[sym.upper()] = (score, (summary or "")[:140])
    return {sym: {"score": score, "summary": summary} for sym, (score, summary) in out.items()}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _coerce_report(
    raw_scores: dict,
    *,
    contexts: dict[str, dict],
    today: str,
) -> dict[str, dict]:
    """Filter/clamp model scores into the on-disk report schema."""
    report: dict[str, dict] = {}
    for base, entry in raw_scores.items():
        base_key = str(base).upper()
        if isinstance(entry, dict):
            score_raw = entry.get("score")
            summary = str(entry.get("summary", ""))[:140]
        else:
            score_raw, summary = entry, ""
        try:
            score = max(0.0, min(10.0, float(score_raw)))
        except (TypeError, ValueError):
            continue  # omit malformed entries rather than fabricate
        if base_key not in contexts or not _finite(score):
            continue  # ignore hallucinated symbols / non-finite numbers
        report[base_key] = {"date": today, "score": round(score, 2), "summary": summary}
    return report


def generate(
    *,
    universe: list[str],
    out_path: str,
    transport=None,
) -> dict:
    """Generate + atomically write the report. Returns the written mapping."""
    api_key = os.environ.get(config.LLM_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing {config.LLM_API_KEY_ENV} in environment; refusing to "
            f"generate sentiment (fail-closed)."
        )

    contexts: dict[str, dict] = {}
    ticker_of: dict[str, str] = {}
    for symbol in universe:
        ctx = _momentum_context(symbol, lookback_days=30)
        if ctx is None:
            continue
        base = symbol.split("/")[0].upper()
        contexts[base] = ctx
        ticker_of[base] = symbol
    if not contexts:
        raise RuntimeError("No usable market context for any universe symbol.")

    if config.API_CALL_DELAY_SECONDS > 0:
        import time

        time.sleep(min(config.API_CALL_DELAY_SECONDS, 2.0))

    system, user = _build_prompt(contexts)
    post = transport or _http_post_json
    url = f"{config.LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter attribution (optional but polite).
        "X-Title": "financeBot",
    }

    today = _today()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    max_attempts = max(int(config.SENTIMENT_LLM_MAX_ATTEMPTS), 1)
    report: dict[str, dict] | None = None
    raw_content = ""
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and last_err is not None:
            # Self-correction round: show the model its own mistake.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid JSON "
                        f"({type(last_err).__name__}: {last_err}). Respond again "
                        f"with ONLY the valid JSON object mapping every given "
                        f'symbol to {{"score": <number 0-10>, "summary": '
                        f'"<=12 words"}}.'
                    ),
                }
            )
        try:
            payload = post(
                url,
                headers=headers,
                body={
                    "model": config.LLM_MODEL,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_tokens": int(config.LLM_MAX_TOKENS),
                    # Minimize invisible reasoning-token burn (the Monday
                    # truncation class). effort=low is the universally
                    # supported shape (GLM/DeepSeek/Gemini); "enabled:false"
                    # is rejected by GLM hybrids with HTTP 400.
                    "reasoning_effort": "low",
                },
            )
            raw_content = str(payload["choices"][0]["message"]["content"])
            raw_scores = _extract_json(raw_content)
            candidate = _coerce_report(raw_scores, contexts=contexts, today=today)
            if candidate:
                report = candidate
                break
            last_err = ValueError("no usable scores in parsed object")
            print(f"[sentgen] attempt {attempt}/{max_attempts}: {last_err}")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            print(f"[sentgen] attempt {attempt}/{max_attempts} failed: {exc}")
            if raw_content:
                # Diagnostic breadcrumb; never includes secrets.
                print(f"[sentgen] raw output head: {raw_content[:400]!r}")

    if report is None and raw_content:
        # Salvage: pull whatever well-formed fragments survive in bad output.
        salvaged_raw = _salvage_entries(raw_content)
        salvaged = _coerce_report(salvaged_raw, contexts=contexts, today=today)
        if salvaged:
            report = salvaged
            print(
                f"[sentgen] SALVAGED {len(report)} valid entries from malformed "
                f"output after {max_attempts} attempt(s); uncovered symbols fall "
                f"back to neutral-pass semantics."
            )

    if not report:
        raise RuntimeError(
            f"Unparseable model output after {max_attempts} attempt(s) "
            f"({last_err}); keeping previous report."
        )

    if len(report) < len(contexts):
        print(
            f"[sentgen] note: {len(contexts) - len(report)} symbol(s) omitted -> "
            f"neutral-pass semantics will govern them."
        )

    # Atomic replace: readers never observe a partial file.
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp_path, out_path)
    print(f"[sentgen] wrote {len(report)}/{len(universe)} scores -> {out_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=config.DAILY_SENTIMENT_PATH)
    args = parser.parse_args()

    from src.universe import resolve_live_universe

    universe = resolve_live_universe()
    print(f"[sentgen] universe={universe}")
    print(f"[sentgen] model={config.LLM_MODEL} @ {config.LLM_BASE_URL}")

    try:
        report = generate(universe=universe, out_path=args.out)
    except Exception as exc:  # noqa: BLE001 -- fail-closed, keep old report
        print(f"[sentgen] FAILED: {exc}")
        print("[sentgen] previous daily_sentiment.json left untouched.")
        return 1

    for sym, entry in sorted(report.items()):
        print(f"[sentgen]   {sym}: {entry['score']:.1f} - {entry['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
