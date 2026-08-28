# Changelog

All notable changes to financeBot are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions are git tags when
releases start being cut. Until then each entry maps to a push to `master`.

## [Unreleased]

### Added
- Weekly universe curator (`curate_universe.py`, `financebot-curate.timer`
  Sundays 22:00 UTC): the political-economical research engine. Builds a dated
  news corpus (Google News RSS per candidate + Fed/Treasury/BLS policy feeds,
  new `src/news.py`), a FRED macro brief, and a quant screen (liquidity,
  momentum, vol, drawdown, corr-to-SPY) over all 47 candidates; one
  epistemics-framed Gemini call proposes the pool; deterministic post-
  validation enforces pool membership, liquidity floor, sector-cap breadth,
  and forced BTC/ETH. Writes `active_universe.json` + audit rationale
  artifact; fail-closed to last week's pool. `--dry-run` supported.
- `soak` risk profile: research sizing ($5/order, $10/day) with 12 concurrent
  slots -- data-gathering throughput without added blast radius.
- `FINANCEBOT_LOOP_INTERVAL_SECONDS` env knob (validated >= 5).

- Hedge pair lifecycle: pivot origins persisted to `models/hedge_pairs.json`
  (`pairs` + persistent `origins` + transition stamps). Two cleanup rules now
  close the hedge-holding loop:
  * **Unwind** -- holding a paired hedge while its target regains a clean
    direct-buy exits the hedge and clears the pair.
  * **Rotation** -- holding a pivot-expressed target when today's sentiment is
    explicitly bad releases the direct leg; re-entry as the hedge happens via
    the normal pivot path. Sentiment-driven transitions are damped to once per
    calendar day (`PAIR_TRANSITION_DAMPENER_DAYS`).
- Daily sentiment generator robustness: layered parsing (strict -> trailing
  comma/control-char sanitizer -> one self-correction re-prompt -> regex
  salvage of valid fragments), diagnostic raw-output logging in journald on
  failure, `SENTIMENT_LLM_MAX_ATTEMPTS`. A partial salvage still refreshes the
  report; full failure keeps yesterday's file.
- Decision-threshold env knobs: `FINANCEBOT_BUY_THRESHOLD` /
  `FINANCEBOT_SELL_THRESHOLD` (strategy knobs, import-time validated 0 < sell
  < buy < 1; defaults unchanged at 0.58 / 0.42).

- `growth_live` risk profile (sized for $10k-30k equity) gated by
  `verify_promotion_evidence()`: >=50 fills, positive net FIFO PnL, p95
  slippage <= 25bps, non-KILL risk state. Override via
  `FINANCEBOT_ALLOW_UNEARNED_PROMOTION` logs loudly every startup.
- Daily sentiment generator (`generate_sentiment.py`) calling an
  OpenAI-compatible LLM (default: `google/gemini-3.7-flash` via OpenRouter)
  once per morning; atomic, fail-closed writes of `daily_sentiment.json`.
  Scheduled weekdays 10:30 UTC via `financebot-sentiment.timer`.
- Cost-aware dual-engine entry gate: exposure increases require modeled edge
  >= max(`MIN_TRADE_EDGE_BPS`, `ESTIMATED_ROUND_TRIP_COST_BPS`).
- T+1 settled-funds gate on buys and cash-account turnover dampener for the
  dual executor (increases scaled/cooldown-limited; reductions never).
- Political-economical PIT ingestion: FRED/BLS/SEC EDGAR connectors,
  persisted feature store (`build_feature_store.py`), macro-aware regime
  classifier, deterministic long-only max-Sharpe core construction, and the
  dual-engine training path (`train_dual.py`).
- `deploy/financebot.service` + `deploy/financebot-sentiment.{service,timer}`.

### Changed
- Hedge selection is fractionability-aware: candidates that cannot be
  expressed at the active profile (non-fractionable, 1 share > cap -- e.g.
  SETH at $5 clips) are excluded BEFORE correlation ranking, so the runner-up
  is chosen instead of a guaranteed broker rejection. Tier-aware by
  construction: excluded names return automatically at larger caps.
- Curator liquidity metric now uses daily-resampled dollar volume (hourly
  averaging undercounted ~7x and over-dropped liquid sector ETFs).
- Missing/invalid daily sentiment scores are now NEUTRAL-PASS by default
  (`FINANCEBOT_SENTIMENT_MISSING_IS_PASS=true`): targets trade normally when
  the report omits them. Explicit low scores still pivot. Legacy behavior
  available by setting the flag false.
- Non-fractionable assets are bought in whole shares only within the position
  cap (`broker.is_fractionable()` pre-check); skipped when one share exceeds
  the cap.
- Active Pivots into equity-ETF hedges are deferred while the equity market
  is closed instead of queueing stale-price fills.

### Fixed
- systemd unit: `StartLimitIntervalSec`/`StartLimitBurst` moved to `[Unit]`
  (rate limiter was silently inert); `PYTHONUNBUFFERED=1` so journald receives
  loop telemetry live.
- Pending settled-cash commitments: prune no longer deletes the broker-id
  mirror entries needed for fill-driven release (phantom cash reservations).

## [0.1.0] - 2026-08-25

### Added
- Initial public baseline: legacy XGBoost hourly engine with sentiment-gated
  Active Pivot hedging, FIFO accounting, walk-forward validation, backtester,
  Streamlit dashboard, dual-horizon skeleton (strategist / tactical executor /
  portfolio manager / risk state machine) with full mocked test suite.
