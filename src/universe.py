"""
Active live target universe resolution.

The LIVE execution loop trades a small, operator-curated set of symbols. This
module resolves that set exactly once at startup from an operator-local
``active_universe.json`` (gitignored), falling back to the legacy
``config.EQUITY_SYMBOLS + config.CRYPTO_SYMBOLS`` when no active file is present
and strict mode is off.

Design rules (see the deployment prompt):

* Research, training, backtesting, and dual replay keep using their OWN
  universes. This module never touches those.
* The 47-symbol code-defined candidate set is NEVER activated automatically. It
  is only used when it appears in ``active_universe.json`` or when the operator
  explicitly opts in with ``FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE=true``.
* Strict mode (live, or ``FINANCEBOT_UNIVERSE_STRICT=true``) fails closed: a
  missing/malformed/stale/over-cap file raises :class:`UniverseError` and stops
  startup before the broker is initialized. Non-strict paper mode degrades to a
  warning and a safe fallback.
* Inverse ETFs (``config.INVERSE_SAFE_LIST``) are hedges chosen by the existing
  Active Pivot logic -- they are NOT ordinary target symbols and are never merged
  into the candidate universe here.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from typing import Iterable

import config

# ---------------------------------------------------------------------------
# Code-defined full candidate universe (NEVER auto-activated -- see module doc).
# ---------------------------------------------------------------------------
# 45 equity/ETF candidates.
EQUITY_CANDIDATES: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "TLT", "IEF", "SHY", "GLD", "SLV",
    "EFA", "EEM", "VNQ", "HYG", "LQD",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "AMD",
    "JPM", "V", "MA", "UNH", "COST", "HD", "PG", "KO", "PEP", "NFLX",
)

# 2 crypto candidates (kept separate; "/" marks them as crypto downstream).
CRYPTO_CANDIDATES: tuple[str, ...] = ("BTC/USD", "ETH/USD")

# 47 total. Inverse ETFs are intentionally excluded (they are hedges, not targets).
FULL_CANDIDATE_UNIVERSE: tuple[str, ...] = EQUITY_CANDIDATES + CRYPTO_CANDIDATES

# Accepted symbol shape. Upper-case, digits, and a few punctuation chars only.
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9./-]{0,14}$")

# Absolute bounds on the active-target cap. The cap may only ever TIGHTEN the
# hard-coded ceiling; values outside this range are configuration errors.
_CAP_MIN = 1
_CAP_MAX = 50


class UniverseError(Exception):
    """Raised when the active universe file/config is invalid or fails closed."""


def is_crypto_symbol(symbol: str) -> bool:
    """Alpaca crypto pairs contain a slash (e.g. ``BTC/USD``)."""
    return "/" in symbol


def normalize_symbols(raw_symbols: Iterable[str]) -> list[str]:
    """
    Normalize a raw symbol iterable into a validated, deduplicated list.

    * ``strip()`` + ``upper()`` each entry.
    * Every entry must be a string matching ``^[A-Z0-9][A-Z0-9./-]{0,14}$``.
    * Duplicates are removed while preserving first-seen order.

    Any invalid entry raises :class:`UniverseError` -- callers must treat a file
    with even one bad symbol as entirely invalid (no partial acceptance).
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_symbols:
        if not isinstance(raw, str):
            raise UniverseError(f"Symbol is not a string: {raw!r}")
        sym = raw.strip().upper()
        if not _SYMBOL_RE.match(sym):
            raise UniverseError(f"Invalid symbol: {raw!r}")
        if sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)
    return normalized


# ---------------------------------------------------------------------------
# Config / environment precedence helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str | None) -> str:
    """explicit arg > FINANCEBOT_UNIVERSE_FILE > config.ACTIVE_UNIVERSE_PATH."""
    if path is not None and str(path).strip():
        return str(path)
    env = os.environ.get("FINANCEBOT_UNIVERSE_FILE")
    if env is not None and env.strip():
        return env.strip()
    return config.ACTIVE_UNIVERSE_PATH


def _resolve_strict(strict: bool | None) -> bool:
    """Explicit arg wins; else live (PAPER False) or FINANCEBOT_UNIVERSE_STRICT."""
    if strict is not None:
        return bool(strict)
    if not config.PAPER:
        return True
    return config._env_flag("FINANCEBOT_UNIVERSE_STRICT", False)


def _resolve_cap(cap: int | None) -> int:
    """explicit arg > FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE > config.MAX_LIVE_UNIVERSE_SIZE."""
    if cap is not None:
        value = cap
    else:
        env = os.environ.get("FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE")
        if env is not None and env.strip():
            try:
                value = int(env.strip())
            except ValueError as exc:
                raise UniverseError(
                    f"FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE is not an integer: {env!r}"
                ) from exc
        else:
            value = config.MAX_LIVE_UNIVERSE_SIZE
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise UniverseError(f"Invalid universe cap: {value!r}") from exc
    if not (_CAP_MIN <= value <= _CAP_MAX):
        raise UniverseError(
            f"Universe cap {value} out of range {_CAP_MIN}..{_CAP_MAX}."
        )
    # The env/arg may only tighten the hard-coded ceiling.
    return min(value, config.MAX_LIVE_UNIVERSE_SIZE)


def _resolve_max_age_hours(max_age_hours: float | None) -> float:
    """explicit arg > FINANCEBOT_UNIVERSE_MAX_AGE_HOURS > config.UNIVERSE_MAX_AGE_HOURS."""
    if max_age_hours is not None:
        value = max_age_hours
    else:
        env = os.environ.get("FINANCEBOT_UNIVERSE_MAX_AGE_HOURS")
        if env is not None and env.strip():
            try:
                value = float(env.strip())
            except ValueError as exc:
                raise UniverseError(
                    f"FINANCEBOT_UNIVERSE_MAX_AGE_HOURS is not a number: {env!r}"
                ) from exc
        else:
            value = config.UNIVERSE_MAX_AGE_HOURS
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise UniverseError(f"Invalid max age hours: {value!r}") from exc
    if value <= 0:
        raise UniverseError(f"Universe max age must be positive, got {value}.")
    return value


def _parse_as_of(raw: object) -> datetime:
    """Parse an ISO-8601 UTC timestamp or YYYY-MM-DD date; naive => UTC."""
    if not isinstance(raw, str) or not raw.strip():
        raise UniverseError(f"Invalid as_of: {raw!r}")
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise UniverseError(f"Invalid as_of timestamp: {raw!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fallback_symbols() -> list[str]:
    """Legacy live symbols: preserves current runtime behavior."""
    return list(config.EQUITY_SYMBOLS) + list(config.CRYPTO_SYMBOLS)


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

def load_active_universe(
    path: str | None = None,
    *,
    strict: bool | None = None,
    cap: int | None = None,
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> list[str]:
    """
    Load, validate, and (optionally) truncate the active-universe file.

    Raises :class:`UniverseError` on a missing/malformed/invalid file, and -- in
    strict mode -- on a stale or over-cap file. In non-strict mode a stale file
    is kept with a warning and an over-cap file is deterministically truncated to
    the first ``cap`` normalized symbols.
    """
    import json

    resolved_path = _resolve_path(path)
    resolved_strict = _resolve_strict(strict)
    resolved_cap = _resolve_cap(cap)
    resolved_max_age = _resolve_max_age_hours(max_age_hours)
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    if not os.path.exists(resolved_path):
        raise UniverseError(f"Active universe file not found: {resolved_path}")

    try:
        with io.open(resolved_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        raise UniverseError(f"Cannot read active universe {resolved_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise UniverseError("Active universe must be a JSON object.")
    if payload.get("version") != 1:
        raise UniverseError("Active universe 'version' must be integer 1.")

    as_of = _parse_as_of(payload.get("as_of"))

    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise UniverseError("Active universe 'symbols' must be a non-empty list.")
    if not all(isinstance(s, str) for s in raw_symbols):
        raise UniverseError("Active universe 'symbols' must all be strings.")

    symbols = normalize_symbols(raw_symbols)
    if not symbols:
        raise UniverseError("Active universe has no valid symbols after normalization.")

    age_hours = (now_utc - as_of).total_seconds() / 3600.0
    stale = age_hours > resolved_max_age
    if stale:
        if resolved_strict:
            raise UniverseError(
                f"Active universe is stale ({age_hours:.1f}h > {resolved_max_age:.1f}h)."
            )
        print(
            f"[universe] WARNING: active universe is stale "
            f"({age_hours:.1f}h > {resolved_max_age:.1f}h); using it anyway (paper)."
        )

    if len(symbols) > resolved_cap:
        if resolved_strict:
            raise UniverseError(
                f"Active universe has {len(symbols)} symbols > cap {resolved_cap}."
            )
        print(
            f"[universe] WARNING: {len(symbols)} symbols > cap {resolved_cap}; "
            f"keeping first {resolved_cap}."
        )
        symbols = symbols[:resolved_cap]

    _log_summary(
        source=resolved_path,
        symbols=symbols,
        cap=resolved_cap,
        as_of=as_of,
        stale=stale,
    )
    return symbols


def _log_summary(*, source, symbols, cap, as_of, stale) -> None:
    print(
        "[universe] source=%s count=%d cap=%d as_of=%s stale=%s"
        % (source, len(symbols), cap, as_of.isoformat(), stale)
    )


# ---------------------------------------------------------------------------
# Startup resolution
# ---------------------------------------------------------------------------

def resolve_live_universe(*, now: datetime | None = None) -> list[str]:
    """
    Resolve the live target universe once at startup.

    Precedence and failure behavior follow the deployment prompt:

    * An explicit ``FINANCEBOT_UNIVERSE_FILE`` (or the default
      ``config.ACTIVE_UNIVERSE_PATH`` if it exists on disk) is loaded via
      :func:`load_active_universe`.
    * With no active file and no explicit request, non-strict paper mode uses the
      legacy ``config.EQUITY_SYMBOLS + config.CRYPTO_SYMBOLS``; strict mode fails
      closed unless ``FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE=true``.
    """
    strict = _resolve_strict(None)
    cap = _resolve_cap(None)
    max_age = _resolve_max_age_hours(None)
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    use_default = config._env_flag("FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE", False)

    explicit_env = os.environ.get("FINANCEBOT_UNIVERSE_FILE")
    explicit = bool(explicit_env is not None and explicit_env.strip())
    default_path = config.ACTIVE_UNIVERSE_PATH
    resolved_path = _resolve_path(None)

    file_present = os.path.exists(resolved_path)

    # Case 1: a file was explicitly requested OR the default file exists on disk.
    if explicit or file_present:
        try:
            return load_active_universe(
                resolved_path,
                strict=strict,
                cap=cap,
                max_age_hours=max_age,
                now=now_utc,
            )
        except UniverseError as exc:
            if strict:
                raise
            print(
                f"[universe] WARNING: {exc}; falling back to config "
                f"EQUITY_SYMBOLS + CRYPTO_SYMBOLS (paper)."
            )
            return _fallback_symbols()

    # Case 2: no active file and no explicit request.
    if use_default:
        # Operator explicitly opted into the code-defined candidate universe.
        symbols = normalize_symbols(FULL_CANDIDATE_UNIVERSE)
        if len(symbols) > cap:
            if strict:
                raise UniverseError(
                    f"Default candidate universe has {len(symbols)} symbols > cap {cap}."
                )
            print(
                f"[universe] WARNING: default universe {len(symbols)} > cap {cap}; "
                f"keeping first {cap}."
            )
            symbols = symbols[:cap]
        print(
            "[universe] source=default-candidate count=%d cap=%d stale=%s"
            % (len(symbols), cap, False)
        )
        return symbols

    if strict:
        raise UniverseError(
            f"No active universe file at {default_path} and strict mode is on. "
            "Provide active_universe.json or set "
            "FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE=true."
        )

    print(
        f"[universe] No active universe file at {default_path}; using config "
        "EQUITY_SYMBOLS + CRYPTO_SYMBOLS (paper default)."
    )
    return _fallback_symbols()
