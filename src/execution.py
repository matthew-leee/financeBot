"""
The live execution loop.

Deliberately THIN and DETERMINISTIC:
  fetch bars -> features -> model prob -> action -> sentiment routing -> order.

The same feature row always yields the same probability, and the same
probability always yields the same action. No hidden state, no randomness, no
external "reasoning" call in the hot decision path -- the LLM/FinRobot layer is
consumed as a pre-computed daily report (see src/sentiment.py).

Active Pivot (Dynamic Correlation Hedging):
  * Sentiment >= threshold -> buy the target asset directly.
  * Sentiment <  threshold (or missing) on a BUY signal -> DO NOT block. Pivot
    into the inverse ETF most negatively correlated to the target and buy that
    instead. The hedge trade is logged under the INVERSE ticker so FIFO
    inventory and the dashboard stay perfectly accurate.

Ordering of safety on every pass:
  1. Circuit breaker check FIRST (may sys.exit()).
  2. Sentiment routing: direct buy vs. hedge pivot.
  3. Position/size guardrails clamp anything the signal requests.
  4. Every broker call is already try/except''d + rate-limited inside Broker.
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime, timezone

import config
from src import guardrails
from src.broker import Broker
from src.data import build_features, fetch_bars
from src.fifo import FIFOInventory
from src.guardrails import (
    RiskConfigurationError,
    can_open_new_position,
    check_circuit_breaker,
    clamp_position_size,
    resolve_daily_loss_threshold,
    resolve_position_cap,
    resolve_risk_policy,
)
from src.model_io import (
    LoadedModel,
    ModelArtifactError,
    load_model,
    verify_model_artifacts,
)
from src.sentiment import is_trade_allowed, load_sentiment, select_hedge_asset
from src.trade_log import append_trade
from src.universe import UniverseError, is_crypto_symbol, resolve_live_universe


def decide_action(prob_up: float) -> str:
    """Pure function: probability -> {"buy","sell","hold"}. Fully deterministic."""
    if prob_up >= config.BUY_THRESHOLD:
        return "buy"
    if prob_up <= config.SELL_THRESHOLD:
        return "sell"
    return "hold"


def _execute_buy(broker: Broker, ticker: str, price: float) -> None:
    """Shared guarded-buy path for both direct and hedged buys."""
    if price <= 0:
        print(f"[loop] {ticker}: invalid price, skip buy.")
        return
    if broker.get_position_qty(ticker) > 0:
        print(f"[loop] {ticker}: already long, holding.")
        return
    if not can_open_new_position(len(broker.get_open_positions())):
        print(f"[loop] {ticker}: max open positions reached, skip buy.")
        return
    # Guardrail clamps notional to MAX_POSITION_SIZE -> qty.
    qty = clamp_position_size(config.MAX_POSITION_SIZE, price)
    if broker.submit_market_order(ticker, qty, "buy"):
        # Logged under the executed ticker (target OR inverse hedge).
        append_trade(ticker, "buy", price, qty)


def _pivot_to_hedge(symbol: str, broker: Broker) -> None:
    """
    Weak-sentiment BUY -> buy the most negatively correlated inverse ETF.

    Selection and pricing both go through the module-level fetch_bars so tests
    can mock the price data deterministically.
    """
    hedge = select_hedge_asset(symbol, bar_fetcher=fetch_bars)
    if hedge is None:
        print(f"[loop] {symbol}: no hedge available, declining trade.")
        return

    hedge_bars = fetch_bars(hedge, lookback_days=config.HEDGE_CORR_LOOKBACK_DAYS)
    if hedge_bars is None or hedge_bars.empty:
        print(f"[loop] {symbol}: hedge {hedge} has no price, declining trade.")
        return

    hedge_price = float(hedge_bars["close"].iloc[-1])
    print(f"[loop] {symbol}: PIVOT -> buying hedge {hedge} @ {hedge_price:.2f}.")
    _execute_buy(broker, hedge, hedge_price)


def process_symbol(
    symbol: str,
    model: LoadedModel,
    broker: Broker,
    sentiment_report: dict | None = None,
) -> None:
    """Run one deterministic decision cycle for a single symbol."""
    sentiment_report = sentiment_report if sentiment_report is not None else {}

    bars = fetch_bars(symbol, lookback_days=30)
    if bars.empty or len(bars) < 30:
        print(f"[loop] {symbol}: insufficient data, skipping.")
        return

    feats = build_features(bars).dropna()
    if feats.empty:
        print(f"[loop] {symbol}: no valid feature row, skipping.")
        return

    latest = feats.iloc[[-1]]
    prob_up = model.predict_up_proba(latest)
    action = decide_action(prob_up)
    price = float(bars["close"].iloc[-1])

    print(f"[loop] {symbol}: P(up)={prob_up:.3f} price={price:.2f} action={action}")

    if action == "buy":
        if is_trade_allowed(sentiment_report, symbol):
            # Strong sentiment -> buy the target directly.
            _execute_buy(broker, symbol, price)
        else:
            # Active Pivot: weak sentiment -> hedge instead of blocking.
            print(f"[loop] {symbol}: sentiment weak, engaging Active Pivot.")
            _pivot_to_hedge(symbol, broker)

    elif action == "sell":
        held_qty = broker.get_position_qty(symbol)
        if held_qty <= 0:
            print(f"[loop] {symbol}: flat, nothing to sell.")
            return
        # Exit the full existing position (never short beyond what we hold).
        if broker.submit_market_order(symbol, held_qty, "sell"):
            append_trade(symbol, "sell", price, held_qty)


# ===========================================================================
# HARDENED LIVE LOOP (paper VPS deployment)
# ===========================================================================
# The legacy process_symbol / _execute_buy / _pivot_to_hedge helpers above are
# preserved verbatim for their existing callers and unit tests. The hardened
# per-pass path below adds: single-resolution universe + risk policy, once-per-
# pass market clock + trusted broker snapshot, a projected pre-trade risk gate
# with an in-memory ledger, pass-local market-data caching, and telemetry.

_LEDGER_EPS = 1e-9


# ---------------------------------------------------------------------------
# Hedge pair lifecycle (pivot-origin tracking -> unwind / rotation rules)
# ---------------------------------------------------------------------------
# A "pair" records WHICH TARGET a hedge was bought for (pivot origin). With
# that provenance on disk, two cleanup rules become possible:
#
#   UNWIND   hold-H + target-T regains a clean direct-buy
#            (P>=BUY_THRESHOLD AND sentiment pass)      -> sell H, clear pair
#   ROTATION hold-T + T buy-signal + EXPLICIT bad news -> sell T
#            (the news component of T's entry thesis broke)
#
# Entries into either leg always flow through the EXISTING paths on later
# passes (direct buy / Active Pivot), so transitions are one clean reduction
# each -- no same-pass sell/rebuy fights with the position-count gates.
# Sentiment-driven transitions are dampened to once per calendar day per
# symbol (mood-card jitter); model-driven exits stay event-driven.


def _load_hedge_pairs(
    path: str | None = None,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Load ({hedge: target}, {symbol: last-transition-date}, {origins}).

    ``origins`` is the persistent set of targets that were EVER pivot-expressed;
    it survives individual hedge exits so the rotation rule stays authorized.
    Fail-safe: corrupt/missing file -> fresh empty state.
    """
    import json

    path = path or config.HEDGE_PAIRS_PATH
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh) or {}
            pairs = {str(k).upper(): str(v) for k, v in raw.get("pairs", {}).items()}
            stamps = {
                str(k).upper(): str(v) for k, v in raw.get("transitions", {}).items()
            }
            origins = {str(o).upper() for o in raw.get("origins", [])}
            return pairs, stamps, origins
    except Exception as exc:  # noqa: BLE001 -- corrupt state -> start fresh
        print(f"[pairs] unreadable {path}: {exc}; starting fresh.")
    return {}, {}, set()


def _save_hedge_pairs(
    pairs: dict[str, str],
    stamps: dict[str, str],
    origins: set[str] | None = None,
    path: str | None = None,
) -> None:
    """Atomically persist pair state; failures logged, never fatal."""
    import json
    import tempfile

    path = path or config.HEDGE_PAIRS_PATH
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pairs": pairs,
                        "transitions": stamps,
                        "origins": sorted(origins or set()),
                    },
                    fh,
                    indent=2,
                )
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception as exc:  # noqa: BLE001
        print(f"[pairs] failed to persist {path}: {exc}")


def _clear_hedge_pair(hedge: str, *, path: str | None = None) -> None:
    """Remove a consumed/stale pair entry (no-op when absent).

    Origins are intentionally KEPT: having once pivoted on a target means the
    rotation rule remains authorized for it.
    """
    pairs, stamps, origins = _load_hedge_pairs(path)
    key = str(hedge).upper()
    if key in pairs:
        del pairs[key]
        _save_hedge_pairs(pairs, stamps, origins, path)


def _target_is_clean_buy(model, target: str, pass_ctx) -> bool:
    """
    True when the PAIRED TARGET qualifies for a legitimate direct buy:
    model says BUY and today's sentiment explicitly allows it. Reuses the
    pass-local bar cache so this costs at most one extra data fetch per pass.
    """
    try:
        bars = pass_ctx.get_bars(target, lookback_days=30)
    except Exception:  # noqa: BLE001
        return False
    if bars is None or getattr(bars, "empty", True) or len(bars) < 30:
        return False
    feats = build_features(bars).dropna()
    if feats.empty:
        return False
    prob_up = model.predict_up_proba(feats.iloc[[-1]])
    if decide_action(prob_up) != "buy":
        return False
    return is_trade_allowed(pass_ctx.sentiment, target)


def _record_pivot_pair(hedge: str, target: str, *, path: str | None = None) -> None:
    pairs, stamps, origins = _load_hedge_pairs(path)
    h_key, t_key = str(hedge).upper(), str(target)
    pairs[h_key] = t_key
    origins.add(t_key.upper())
    _save_hedge_pairs(pairs, stamps, origins, path)


def _rotation_allowed(symbol: str, stamps: dict[str, str]) -> bool:
    """Once-per-calendar-day dampener for sentiment-driven transitions."""
    days = max(int(getattr(config, "PAIR_TRANSITION_DAMPENER_DAYS", 1)), 0)
    if days == 0:
        return True
    last = stamps.get(str(symbol).upper())
    if not last:
        return True
    try:
        last_date = datetime.fromisoformat(last).date()
    except ValueError:
        return True
    return last_date < datetime.now(timezone.utc).date()


def _num(value: object) -> "float | None":
    """Best-effort finite float, else None."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _norm_key(symbol: object) -> str:
    """Match symbols across forms: BTC/USD (target) vs BTCUSD (broker)."""
    return str(symbol).upper().replace("/", "")


def _order_notional(order: object) -> float:
    """Worst-case notional of a pending order (assume it fills)."""
    qty = _num(getattr(order, "qty", None))
    price = _num(getattr(order, "limit_price", None))
    if price is None:
        price = _num(getattr(order, "filled_avg_price", None))
    if qty is not None and price is not None:
        return abs(qty * price)
    notl = _num(getattr(order, "notional", None))
    if notl is not None:
        return abs(notl)
    if qty is not None:
        return abs(qty)
    return 0.0


class _ProjectedLedger:
    """
    Deterministic projected-risk ledger built from a single-pass broker snapshot.

    Pending exposure-increasing (buy) orders are assumed to fill; pending
    reducing (sell) orders are assumed NOT to fill -- worst-case exposure. The
    ledger is mutated after every SUCCESSFULLY submitted order so later symbols in
    the same pass cannot bypass the caps.
    """

    def __init__(self, snapshot, policy) -> None:
        self.policy = policy
        self.snapshot = snapshot
        self.equity = snapshot.equity if snapshot.equity_ok else None
        self.cur_notional: dict[str, float] = {}
        self.held_qty: dict[str, float] = {}
        self.symbol_worst: dict[str, float] = {}
        self.position_keys: set[str] = set()
        self.pending_new_keys: set[str] = set()
        self.committed_new_keys: set[str] = set()

        for pos in snapshot.positions:
            key = _norm_key(getattr(pos, "symbol", ""))
            if not key:
                continue
            qty = _num(getattr(pos, "qty", 0.0)) or 0.0
            market_value = _num(getattr(pos, "market_value", None))
            if market_value is None:
                avg = _num(getattr(pos, "avg_entry_price", 0.0)) or 0.0
                market_value = abs(qty) * avg
            market_value = abs(market_value)
            self.cur_notional[key] = self.cur_notional.get(key, 0.0) + market_value
            self.held_qty[key] = self.held_qty.get(key, 0.0) + qty
            self.symbol_worst[key] = self.symbol_worst.get(key, 0.0) + market_value
            if abs(qty) > _LEDGER_EPS:
                self.position_keys.add(key)

        for order in snapshot.open_orders:
            key = _norm_key(getattr(order, "symbol", ""))
            if not key:
                continue
            side = str(getattr(order, "side", "")).lower()
            if "buy" in side:
                notl = _order_notional(order)
                self.symbol_worst[key] = self.symbol_worst.get(key, 0.0) + notl
                if key not in self.position_keys:
                    self.pending_new_keys.add(key)

    def _total_worst(self) -> float:
        return sum(self.symbol_worst.values())

    def _projected_count(self) -> int:
        return len(self.position_keys | self.pending_new_keys | self.committed_new_keys)

    def is_long(self, symbol: str) -> bool:
        return self.held_qty.get(_norm_key(symbol), 0.0) > _LEDGER_EPS

    def held_quantity(self, symbol: str) -> float:
        return self.held_qty.get(_norm_key(symbol), 0.0)

    def check_buy(self, symbol: str, order_notional: float, price: "float | None"):
        """Return (ok, reason) for a projected exposure-increasing buy."""
        key = _norm_key(symbol)
        equity = self.equity
        if price is None or not math.isfinite(price) or price <= 0:
            return False, "no_trusted_price"
        if order_notional <= 0:
            return False, "zero_notional"

        cap = resolve_position_cap(self.policy, equity)
        projected_symbol = self.symbol_worst.get(key, 0.0) + order_notional
        if projected_symbol > cap + 1e-6:
            return False, "symbol_position_cap"

        opens_new = key not in (
            self.position_keys | self.pending_new_keys | self.committed_new_keys
        )
        if opens_new and self._projected_count() + 1 > self.policy.max_open_positions:
            return False, "max_open_positions"

        if self.policy.max_gross_exposure_pct is not None:
            if equity is None or not math.isfinite(equity) or equity <= 0:
                return False, "gross_unknown_equity"
            projected_gross = self._total_worst() + order_notional
            if projected_gross > equity * self.policy.max_gross_exposure_pct + 1e-6:
                return False, "gross_exposure"

        return True, "ok"

    def reserve_buy(self, symbol: str, order_notional: float) -> None:
        key = _norm_key(symbol)
        opens_new = key not in (
            self.position_keys | self.pending_new_keys | self.committed_new_keys
        )
        self.symbol_worst[key] = self.symbol_worst.get(key, 0.0) + order_notional
        self.cur_notional[key] = self.cur_notional.get(key, 0.0) + order_notional
        if opens_new:
            self.committed_new_keys.add(key)

    def apply_sell(self, symbol: str, qty: float, price: float) -> None:
        key = _norm_key(symbol)
        self.held_qty[key] = max(0.0, self.held_qty.get(key, 0.0) - qty)
        sell_notional = abs(qty * price)
        self.cur_notional[key] = max(0.0, self.cur_notional.get(key, 0.0) - sell_notional)
        self.symbol_worst[key] = max(0.0, self.symbol_worst.get(key, 0.0) - sell_notional)
        if self.held_qty.get(key, 0.0) <= _LEDGER_EPS:
            self.position_keys.discard(key)


class _PassTelemetry:
    """Per-pass counters emitted at the end of every loop pass."""

    def __init__(self) -> None:
        self.processed = 0
        self.submitted = 0
        self.skipped: dict[str, int] = {}

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


class _Pass:
    """Everything a single loop pass needs, resolved exactly once per pass."""

    def __init__(self, *, policy, snapshot, market_open, sentiment,
                 active_target_keys) -> None:
        self.policy = policy
        self.snapshot = snapshot
        self.market_open = market_open
        self.sentiment = sentiment
        self.active_target_keys = active_target_keys
        self.ledger = _ProjectedLedger(snapshot, policy)
        self.telemetry = _PassTelemetry()
        self._bar_cache: dict[str, object] = {}

    def get_bars(self, symbol: str, lookback_days: int):
        """Pass-local cache: each symbol's bars are fetched at most once per pass."""
        if symbol in self._bar_cache:
            return self._bar_cache[symbol]
        bars = fetch_bars(symbol, lookback_days=lookback_days)
        self._bar_cache[symbol] = bars
        return bars


def build_processing_universe(active_targets, snapshot, fifo_inv) -> list[str]:
    """
    active target symbols  UNION  broker-held symbols  UNION  FIFO-held symbols.

    Deduplicated by normalized key, with the active-target form preferred so a
    target's canonical symbol (e.g. BTC/USD) wins over a broker form (BTCUSD).
    Held positions/hedges remain visible for reconciliation and exit handling;
    the configured cap applies ONLY to active targets (already enforced upstream).
    """
    processing: list[str] = []
    seen: set[str] = set()
    for sym in active_targets:
        key = _norm_key(sym)
        if key not in seen:
            seen.add(key)
            processing.append(sym)
    held: list[str] = []
    for pos in getattr(snapshot, "positions", []):
        sym = getattr(pos, "symbol", None)
        if sym:
            held.append(str(sym))
    try:
        held.extend(fifo_inv.positions().keys())
    except Exception as exc:  # noqa: BLE001 -- inventory read must not crash startup
        print(f"[loop] Could not read FIFO inventory: {exc}")
    for sym in held:
        key = _norm_key(sym)
        if key not in seen:
            seen.add(key)
            processing.append(sym)
    return processing


def _hardened_buy(broker, ticker, price, pass_ctx) -> None:
    """Guarded buy honoring the trusted snapshot and projected risk gate."""
    snap = pass_ctx.snapshot
    telem = pass_ctx.telemetry

    if not (snap.positions_ok and snap.open_orders_ok):
        telem.skip("broker_truth_unknown")
        print(f"[loop] {ticker}: broker position/order truth unknown, blocking buy.")
        return
    if not snap.equity_ok:
        telem.skip("unknown_equity_blocks_buy")
        print(f"[loop] {ticker}: equity unknown, blocking exposure increase.")
        return
    if price is None or price <= 0:
        telem.skip("invalid_price")
        print(f"[loop] {ticker}: invalid price, skip buy.")
        return
    if pass_ctx.ledger.is_long(ticker):
        telem.skip("already_long")
        print(f"[loop] {ticker}: already long, holding.")
        return

    cap = resolve_position_cap(pass_ctx.policy, pass_ctx.ledger.equity)

    # Fractionability pre-check: some inverse ETFs reject fractional quantities
    # outright (Alpaca 40310000). Size in WHOLE shares for those, and skip when
    # even one share exceeds the cap. Unknown -> legacy fractional sizing.
    frac_fn = getattr(broker, "is_fractionable", None)
    fractionable = None
    if callable(frac_fn):
        try:
            fractionable = frac_fn(ticker)
        except Exception:  # noqa: BLE001 -- unknown stays unknown
            fractionable = None

    if fractionable is False:
        qty = float(int(cap // price)) if price > 0 else 0.0
        if qty < 1:
            telem.skip("non_fractionable_too_expensive")
            print(
                f"[loop] {ticker}: non-fractionable and 1 share "
                f"({price:.2f}) exceeds cap ({cap:.2f}); skipping buy."
            )
            return
        print(f"[loop] {ticker}: non-fractionable -> whole-share sizing ({qty:.0f} sh).")
    else:
        qty = clamp_position_size(cap, price, max_notional=cap)

    order_notional = qty * price
    ok, reason = pass_ctx.ledger.check_buy(ticker, order_notional, price)
    if not ok:
        telem.skip(reason)
        print(f"[loop] {ticker}: pre-trade risk gate rejected buy ({reason}).")
        return

    if broker.submit_market_order(ticker, qty, "buy"):
        pass_ctx.ledger.reserve_buy(ticker, order_notional)
        append_trade(ticker, "buy", price, qty)
        telem.submitted += 1


def _hardened_pivot(symbol, broker, pass_ctx) -> None:
    """Weak-sentiment BUY -> buy the most negatively correlated inverse hedge."""
    telem = pass_ctx.telemetry

    # Tier-aware candidate filter: a hedge that cannot currently be expressed
    # (non-fractionable with 1 share > position cap) is excluded BEFORE
    # correlation ranking, so the runner-up gets selected instead of a
    # guaranteed broker rejection. Fractionability comes from the broker's
    # cached asset lookup; unknown -> not excluded (order-time guards still apply).
    exclude: set[str] = set()
    frac_fn = getattr(broker, "is_fractionable", None)
    if callable(frac_fn):
        cap = resolve_position_cap(pass_ctx.policy, pass_ctx.ledger.equity)
        for cand in config.INVERSE_SAFE_LIST:
            try:
                if frac_fn(cand) is False:
                    # Definitively non-fractionable: expressable only if one
                    # whole share fits inside the position cap.
                    bars = pass_ctx.get_bars(cand, lookback_days=config.HEDGE_CORR_LOOKBACK_DAYS)
                    price = (
                        float(bars["close"].iloc[-1])
                        if bars is not None and not getattr(bars, "empty", True)
                        else None
                    )
                    if price is None or price > cap:
                        exclude.add(cand.upper())
            except Exception:  # noqa: BLE001 -- unknown stays unknown
                continue

    hedge = select_hedge_asset(symbol, bar_fetcher=pass_ctx.get_bars, exclude=exclude)
    if hedge is None:
        telem.skip("no_hedge")
        print(f"[loop] {symbol}: no hedge available, declining trade.")
        return

    # Closed-shop guard: an equity-ETF hedge cannot fill cleanly while the
    # equity market is closed (queued fills execute at stale reference prices).
    # Defer the whole pivot to a post-open pass. Crypto-target processing is
    # allowed around the clock, but its EQUITY hedge is not.
    if "/" not in hedge and pass_ctx.market_open is not True:
        telem.skip("hedge_market_closed")
        print(
            f"[loop] {symbol}: hedge {hedge} trades on the closed equity "
            f"market; deferring pivot until open."
        )
        return

    hedge_bars = pass_ctx.get_bars(hedge, lookback_days=config.HEDGE_CORR_LOOKBACK_DAYS)
    if hedge_bars is None or hedge_bars.empty:
        telem.skip("no_hedge_price")
        print(f"[loop] {symbol}: hedge {hedge} has no price, declining trade.")
        return
    hedge_price = float(hedge_bars["close"].iloc[-1])
    print(f"[loop] {symbol}: PIVOT -> buying hedge {hedge} @ {hedge_price:.2f}.")
    submitted_before = telem.submitted
    _hardened_buy(broker, hedge, hedge_price, pass_ctx)
    if telem.submitted > submitted_before:
        # Successful pivot: record origin so unwind/rotation lifecycle applies.
        _record_pivot_pair(hedge, symbol)


def _hardened_sell(broker, symbol, price, pass_ctx) -> None:
    """Verified full exit; long-only, never opens a short."""
    snap = pass_ctx.snapshot
    telem = pass_ctx.telemetry
    if not snap.positions_ok:
        telem.skip("positions_unknown_no_sell")
        print(f"[loop] {symbol}: position truth unknown, cannot prove reduction.")
        return
    held_qty = pass_ctx.ledger.held_quantity(symbol)
    if held_qty <= 0:
        telem.skip("flat_nothing_to_sell")
        print(f"[loop] {symbol}: flat, nothing to sell.")
        return
    sell_qty = held_qty  # clamp to verified held quantity -> never short
    if broker.submit_market_order(symbol, sell_qty, "sell"):
        append_trade(symbol, "sell", price, sell_qty)
        pass_ctx.ledger.apply_sell(symbol, sell_qty, price)
        telem.submitted += 1


def _has_pending_exit(snapshot, symbol: str) -> bool:
    """True when an open SELL order already exists for this symbol."""
    if not getattr(snapshot, "open_orders_ok", False):
        return False
    want = _norm_key(symbol)
    for order in getattr(snapshot, "open_orders", []) or []:
        if (
            _norm_key(getattr(order, "symbol", "")) == want
            and str(getattr(order, "side", "")).lower().startswith("s")
        ):
            return True
    return False


def process_symbol_hardened(symbol, model, broker, pass_ctx) -> None:
    """Hardened per-symbol decision cycle for the live loop."""
    telem = pass_ctx.telemetry
    is_crypto = is_crypto_symbol(symbol)

    # Market-hours gating: skip equities BEFORE any market-data request when the
    # equity market is closed or the clock is unknown. Crypto stays eligible.
    if not is_crypto and pass_ctx.market_open is not True:
        telem.skip("equity_market_closed")
        return

    bars = pass_ctx.get_bars(symbol, lookback_days=30)
    if bars is None or bars.empty or len(bars) < 30:
        telem.skip("insufficient_data")
        print(f"[loop] {symbol}: insufficient data, skipping.")
        return
    feats = build_features(bars).dropna()
    if feats.empty:
        telem.skip("no_features")
        print(f"[loop] {symbol}: no valid feature row, skipping.")
        return

    latest = feats.iloc[[-1]]
    prob_up = model.predict_up_proba(latest)
    action = decide_action(prob_up)
    price = float(bars["close"].iloc[-1])
    telem.processed += 1
    print(f"[loop] {symbol}: P(up)={prob_up:.3f} price={price:.2f} action={action}")

    is_target = _norm_key(symbol) in pass_ctx.active_target_keys
    pairs, stamps, origins = _load_hedge_pairs()

    paired_target = None
    if not is_target:
        # --- UNWIND: held paired hedge whose target is clean-buy-ready again --
        paired_target = pairs.get(str(symbol).upper())
        if (
            paired_target
            and getattr(pass_ctx.snapshot, "positions_ok", False)
            and pass_ctx.ledger.held_quantity(symbol) > 0
            and not _has_pending_exit(pass_ctx.snapshot, symbol)
            and _target_is_clean_buy(model, paired_target, pass_ctx)
        ):
            print(
                f"[pivot-unwind] {symbol}: paired target {paired_target} is a "
                f"clean direct-buy now; exiting hedge."
            )
            _hardened_sell(broker, symbol, price, pass_ctx)
            _clear_hedge_pair(str(symbol).upper())
            return

    if action == "buy":
        if not is_target:
            # Held-but-not-target: never open/increase; only exits are permitted.
            telem.skip("non_target_no_increase")
            print(f"[loop] {symbol}: held non-target, buys blocked (exit-only).")
            return

        # --- ROTATION (holding the paired TARGET directly) -------------------
        # Buy signal persists but today's news is explicitly bad -> the news
        # leg of the entry thesis broke; release the direct leg. Re-entry as
        # the hedge happens next pass through the normal Active Pivot path.
        already_long = pass_ctx.ledger.is_long(symbol)
        was_paired_origin = (
            any(t == str(symbol) for t in pairs.values())
            or str(symbol).upper() in origins
        )
        if (
            already_long
            and was_paired_origin
            and not is_trade_allowed(pass_ctx.sentiment, symbol)
            and _rotation_allowed(symbol, stamps)
        ):
            print(
                f"[pair-rotate] {symbol}: buy signal with explicitly bad "
                f"sentiment while long -> releasing direct leg."
            )
            _hardened_sell(broker, symbol, price, pass_ctx)
            stamps[str(symbol).upper()] = datetime.now(timezone.utc).date().isoformat()
            _save_hedge_pairs(pairs, stamps, origins)
            return

        if is_trade_allowed(pass_ctx.sentiment, symbol):
            _hardened_buy(broker, symbol, price, pass_ctx)
        else:
            print(f"[loop] {symbol}: sentiment weak, engaging Active Pivot.")
            _hardened_pivot(symbol, broker, pass_ctx)
    elif action == "sell":
        _hardened_sell(broker, symbol, price, pass_ctx)
        # A completed exit releases any stale pivot-origin record.
        if str(symbol).upper() in pairs:
            _clear_hedge_pair(str(symbol).upper())


def _enforce_circuit_breaker(broker, policy, snapshot) -> None:
    """
    Dynamic rolling-24h circuit breaker using the persisted anchor equity.

    On breach: attempt to cancel open orders (logging any failure), then exit
    with SystemExit(1) regardless of cancellation outcome. Unknown equity is not
    a breach -- it simply cannot be evaluated this pass.
    """
    if not snapshot.equity_ok or snapshot.equity is None:
        return
    equity = float(snapshot.equity)
    guardrails.record_equity_anchor(equity)
    state = guardrails._load_state()
    anchor = state.get("anchor_equity")
    if anchor is None:
        return
    threshold = resolve_daily_loss_threshold(policy, anchor)
    pnl = equity - float(anchor)
    if pnl <= threshold:
        print(
            f"[CIRCUIT BREAKER] 24h PnL {pnl:.2f} <= limit {threshold:.2f}. "
            "Shutting down."
        )
        try:
            broker.cancel_all_open_orders()
        except Exception as exc:  # noqa: BLE001 -- cancellation must not block exit
            print(f"[CIRCUIT BREAKER] cancel_all_open_orders failed: {exc}")
        sys.exit(1)


def run() -> None:
    """Main hardened loop. Runs until the circuit breaker trips or the process dies."""
    print("[loop] Starting financeBot execution loop.")

    # Fail fast BEFORE constructing a broker or risking capital.
    try:
        verify_model_artifacts()
    except ModelArtifactError as exc:
        print(f"[startup] ABORT -- {exc}")
        sys.exit(1)

    # Resolve the immutable risk policy exactly once (fail closed on bad config).
    try:
        policy = resolve_risk_policy()
    except RiskConfigurationError as exc:
        print(f"[startup] ABORT -- invalid risk configuration: {exc}")
        sys.exit(1)

    # Resolve the active target universe exactly once (fail closed in strict mode).
    try:
        active_targets = resolve_live_universe()
    except UniverseError as exc:
        print(f"[startup] ABORT -- universe resolution failed: {exc}")
        sys.exit(1)

    active_target_keys = {_norm_key(sym) for sym in active_targets}
    print(
        f"[startup] risk profile={policy.profile} active_targets={len(active_targets)}"
    )

    model = load_model()
    broker = Broker()

    while True:
        pass_start = time.time()

        # Fetch ONCE per pass: market clock, trusted broker snapshot, sentiment.
        market_open = broker.get_equity_market_open()
        snapshot = broker.get_risk_snapshot()

        # 1) Safety first: dynamic circuit breaker may cancel orders + sys.exit().
        _enforce_circuit_breaker(broker, policy, snapshot)

        # Degraded-truth handling: cannot safely submit new orders this pass.
        if not (snapshot.positions_ok and snapshot.open_orders_ok):
            print("[loop] Broker truth degraded; cancelling open orders, retrying next pass.")
            try:
                broker.cancel_all_open_orders()
            except Exception as exc:  # noqa: BLE001
                print(f"[loop] cancel_all_open_orders failed: {exc}")

        sentiment_report = load_sentiment()

        fifo_inv = FIFOInventory.load(config.INVENTORY_STATE_PATH)
        processing = build_processing_universe(active_targets, snapshot, fifo_inv)

        pass_ctx = _Pass(
            policy=policy,
            snapshot=snapshot,
            market_open=market_open,
            sentiment=sentiment_report,
            active_target_keys=active_target_keys,
        )

        for symbol in processing:
            try:
                process_symbol_hardened(symbol, model, broker, pass_ctx)
            except Exception as exc:  # noqa: BLE001 -- one symbol must not kill loop
                pass_ctx.telemetry.skip("symbol_error")
                print(f"[loop] Unhandled error on {symbol}: {exc}")

        elapsed = time.time() - pass_start
        telem = pass_ctx.telemetry
        api_error_rate = None
        if hasattr(broker, "get_api_error_rate"):
            try:
                api_error_rate = broker.get_api_error_rate()
            except Exception:  # noqa: BLE001
                api_error_rate = None
        print(
            "[telemetry] elapsed=%.2fs targets=%d processed=%d submitted=%d "
            "market_open=%s api_error_rate=%s skipped=%s"
            % (
                elapsed,
                len(active_targets),
                telem.processed,
                telem.submitted,
                market_open,
                ("%.2f" % api_error_rate) if api_error_rate is not None else "n/a",
                dict(sorted(telem.skipped.items())),
            )
        )

        # Loop sleeping with overrun handling.
        sleep_seconds = max(0.0, config.LOOP_INTERVAL_SECONDS - elapsed)
        if elapsed > config.LOOP_INTERVAL_SECONDS:
            print(
                f"[loop] WARNING: pass took {elapsed:.2f}s > interval "
                f"{config.LOOP_INTERVAL_SECONDS:.2f}s; starting next pass without extra sleep."
            )
        time.sleep(sleep_seconds)
