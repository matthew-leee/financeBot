# financeBot -- Hybrid XGBoost + Active Pivot Trading Bot

A micro-live algorithmic trading bot for equities + crypto via `alpaca-py`.

The core decision engine is a locally-trained **XGBoost** model serialized to
JSON (Option A: zero API/token cost, deterministic, backtestable). A daily
**FinRobot/LLM sentiment report** shapes execution: strong sentiment buys the
target directly, while weak sentiment triggers an **Active Pivot** into an
inverse ETF chosen by **Dynamic Correlation Hedging**. Realized PnL is tracked
with a strict **FIFO** inventory queue and visualized on a decoupled, read-only
**Streamlit + Plotly** dashboard.

## Decision Flow

    train.py ── fetch history ─ features/labels ─ walk-forward validate
                             └─ fit final model ─ models/model.json (+ meta)

    run_bot.py (live loop):
        circuit breaker ─ load daily_sentiment.json ─ per symbol:
            fetch bars ─ features ─ P(up) ─ threshold ─ action
                              │
              action == BUY:
                sentiment >= MIN ?  ── yes ─> buy TARGET directly
                                    └─ no  ─> ACTIVE PIVOT:
                                              select inverse ETF with strongest
                                              negative correlation ─> buy HEDGE
                              │
              guardrails (size clamp / position cap) ─ alpaca order
                              │
              FIFO inventory update ─ append row to trades_log.csv
                              (logged under the EXECUTED ticker: target or hedge)

    backtest.py ── bars ─ align+generate historical_sentiment.csv ─ replay
                             └─ metrics (PnL/WinRate/Drawdown/Sharpe) + CSV

    dashboard.py ── read-only view of trades_log.csv + inventory_state.json
                     + daily_sentiment.json

## Active Pivot / Dynamic Correlation Hedging

Previously the sentiment gate was **fail-closed**: a score `< 5.0` blocked the
trade entirely. It is now an **active** router:

- **Strong sentiment** (`score >= SENTIMENT_MIN_SCORE`) -> buy the target asset.
- **Weak / missing sentiment** on a BUY signal -> **do not sit flat**. Pivot the
  capital into a hedge selected dynamically:
  1. Fetch trailing `HEDGE_CORR_LOOKBACK_DAYS` (default 30) of **daily returns**
     for the target and every candidate in `INVERSE_SAFE_LIST`.
  2. Compute the Pearson correlation of each candidate vs. the target on their
     overlapping dates.
  3. Select the inverse ETF with the **strongest (most negative) correlation**.
  4. Buy that hedge, and log the trade **under the inverse ticker** (e.g. `PSQ`
     instead of `AAPL`) so FIFO inventory and the dashboard stay accurate.
- **Fails safe**: if no candidate yields a computable correlation (no data, zero
  variance), `select_hedge_asset()` returns `None` and the loop declines the
  trade rather than hedging blindly.

Config:

    INVERSE_SAFE_LIST = ["SH", "PSQ", "BITI", "SARK", "SETH", "RWM", "DOG"]
    HEDGE_CORR_LOOKBACK_DAYS = 30

Key functions in `src/sentiment.py`:

- `is_trade_allowed(report, symbol)` -- does the target clear the daily gate?
- `daily_returns(bars)` -- day-resampled close returns (feed-agnostic).
- `select_hedge_asset(target, safe_list, bar_fetcher, lookback_days)` -- the
  correlation selector (injectable `bar_fetcher` for testing).

Execution routing lives in `src/execution.py` (`process_symbol`, `_execute_buy`,
`_pivot_to_hedge`).

## Dual-Horizon Engine (additive, opt-in)

A second, additive engine layers a **slow interday Strategist** and a **fast
intraday Executor** on top of the legacy loop. Every legacy contract above is
preserved. The safe/default deployment path remains legacy:

    python run_bot.py                 # FINANCEBOT_ENGINE or legacy
    python run_bot.py --engine legacy # explicit legacy engine

Dual execution now has a production boot path:

    python build_feature_store.py     # 1. populate + persist the PIT store
    python train_dual.py              # 2. train + register expected_return model
    python run_bot.py --engine dual   # 3. boots only if step 1 succeeded

`run_dual()` hydrates the persisted point-in-time store from disk and still
exits with `SystemExit(1)` when it is empty -- it refuses to trade on nothing.

### Political-economical data pipeline (new)

The strategic layer consumes long-horizon macro/filing state through the same
anti-lookahead boundary as everything else:

                      ┌────────────────────────────────────────┐
     FRED (keyless) ──►│ src/macro.py connectors                │
     BLS  (optional)──►│ vintage-safe available_at per series   │──► PointInTimeRecord
     SEC EDGAR     ───►│ rate-limited, fail-closed, mockable    │         │
                      └────────────────────────────────────────┘         ▼
                                        Alpaca daily closes ──► data/feature_store/
                                                                    records.jsonl

- `FredConnector` pulls rates (`US3M/US2Y/US5Y/US10Y` yields), CPI and fed
  funds from the keyless `fredgraph.csv` endpoint. Each series carries a
  **conservative publication lag** (`config.FRED_SERIES`) so
  `available_at = observation_date + lag`: revised macro history can never
  leak backward into features.
- `BlsConnector` posts to BLS public API v2/v1 (key optional via env,
  series map empty by default).
- `SecEdgarConnector` maps tickers -> CIK via `company_tickers.json`, then
  emits `filing_date` records for tracked forms (10-K/10-Q/8-K/...), knowable
  only after a conservative post-close lag.
- Every connector wraps its transport in strict try/except, sleeps
  `API_CALL_DELAY_SECONDS` before each call, and degrades to captured errors --
  never raises into research or live loops. HTTP + sleep are injected seams;
  tests never touch the network.

The interday snapshot now also emits `cpi_yoy`, `cpi_yoy_chg_3m`,
`unemployment`, `unemployment_chg_3m`, and `fed_funds_rate` (with automatic
missing indicators when absent), feeding the regime classifier below.

### Max-Sharpe core construction (new)

`InterdayStrategist.optimize_core_portfolio()` now builds the core book with a
deterministic **long-only max-Sharpe (tangency) construction**
(`max_sharpe_weights()`): seed `w ∝ pinv(Σ)μ` restricted to μ > 0, refined by a
fixed number of projected-gradient steps on `μᵀw / sqrt(wᵀΣw)`. Degenerate
inputs fall back to the legacy inverse-volatility heuristic automatically
(`USE_MAX_SHARPE_CORE` toggles this). Downstream safety is unchanged: vol-target
scaling, symbol/gross/net caps, turnover smoothing, and the hard per-order
`MAX_POSITION_SIZE` clamp all still apply after optimization.

### Macro-aware regime classifier (new)

`predict_regime()` keeps its legacy vol/drawdown ladder and adds two
political-economical triggers when the macro columns exist:

    crisis            drawdown <= -20%                     (legacy)
    liquidity_stress  vol_63 >= 35%                        (legacy)
    inflation_shock   cpi_yoy >= 4% AND rising (+0.2pp/3m) (new)
    growth_slowdown   drawdown <= -10% OR inverted 10y-2y  (extended)
    risk_on           otherwise

### Topology

                     ┌────────────────────────────┐
                     │ src/data.py                │
                     │ PIT store + feature engine │
                     └──────────────┬─────────────┘
                       interday / intraday snapshots
          ┌─────────────────────────┴─────────────────────────┐
    ┌─────▼──────────────┐                        ┌────────────▼───────────┐
    │ src/strategist.py  │  daily allocation ───► │ src/tactical_executor.py│
    │ Interday Strategist│  matrix + envelopes    │ Intraday Executor       │
    └─────┬──────────────┘                        └────────────┬───────────┘
          │ target weights, hedge ratios, envelopes            │ order intents
          │                                        ┌────────────▼───────────┐
          │                                        │ src/portfolio_manager.py│
          │                                        │ fills + FIFO reconcile  │
          │                                        └────────────┬───────────┘
          └───────────────────────┬────────────────────────────┘
                                  │ risk metrics
                         ┌────────▼──────────┐
                         │ src/guardrails.py │
                         │ Risk state machine│
                         └───────────────────┘

- **`src/strategist.py`** decides *structural* exposure once per day.
- **`src/tactical_executor.py`** times execution *inside* that exposure.
- **`src/portfolio_manager.py`** owns broker truth, fills, and FIFO.
- **`src/guardrails.py`** decides what actions are legal at every moment.

### Point-in-Time (PIT) data rules

`src/data.py` gains a `PointInTimeFeatureStore`, the anti-lookahead boundary:

- Every record carries `event_time` (when it happened) and `available_at` (when
  we could first know it). Every query enforces `available_at <= as_of`.
- Macro revisions are vintage-safe: the latest vintage *available as of* the
  decision time is used, never a future revision.
- Absent fields become explicit `<field>_missing` indicators; values are never
  backfilled from the future.
- News is folded into a time-decayed state via `build_news_embedding_state()`:
  `exp(-(t - available_at)/τ)`-weighted embedding direction + `log1p(mass)`
  intensity, τ = `NEWS_TAU_DAYS`.
- `build_interday_snapshot()` emits daily strategic features (multi-horizon
  returns/vol, drawdown, yield-curve level/slope/curvature, filing age, news
  intensity); `build_intraday_snapshot()` emits execution-timing features
  (1m/5m returns, realized vol, VWAP distance, spread bps, order-flow).

### Leakage-safe panel validation

`src/validation.py` adds `make_temporal_group_folds()`,
`walk_forward_validate_panel()`, and `build_forward_return_labels()`. Folds split
all symbols by **calendar time** (not row index) with an embargo that must cover
the max label horizon, killing cross-asset and label-overlap leakage. Metrics are
strategy-aware: rank IC, hit-rate, turnover, cost-adjusted PnL, net Sharpe, max
drawdown, plus explicit `leakage_checks`. Legacy `walk_forward_validate()` stays.

### Model registry

`src/model_io.ModelRegistry` is an additive, role-keyed multi-model loader
(`legacy_direction`, `expected_return`, `regime`, `covariance`, `tactical`). It
reads `models/registry/registry.json`, enforces per-artifact feature-column order
at inference, and `verify()` fails fast if a required role is missing/invalid.
Legacy `load_model()` / `verify_model_artifacts()` / `LoadedModel` are untouched.

### Allocation matrix schema

The strategist emits an `AllocationMatrix` (`as_of`, `regime`,
`covariance_version`, `feature_snapshot_id`, `diagnostics`, and per-symbol
`rows`). Each `TargetAllocation` row:

    symbol, target_weight, min_weight, max_weight, direction_bias,
    volatility_ceiling, rebalance_priority, hedge_for, hedge_ratio,
    expected_return, expected_volatility

Covariance is EWMA + shrinkage + PSD eigen-floored. Hedging is a min-variance
overlay (`h_raw = -Σ_HH⁻¹ Σ_HC w_C`) kept only if it clears
`MIN_HEDGE_EFFECTIVENESS`. Hedges no longer needed are emitted with
`target_weight = 0` so the executor exits them (orphaned-hedge fix).

### Fill-based logging & reconciliation

`src/portfolio_manager.py` is the only live writer of orders/fills/inventory.
Live PnL and FIFO lots derive from actual `BrokerFill` events — never requested
size or a stale bar close. Fills are idempotent by `fill_id`. Two new logs:

    orders_log.csv: submitted_at, client_order_id, broker_order_id, symbol, side,
                    requested_qty, order_type, limit_price, reduce_only, reason, status
    fills_log.csv:  filled_at, fill_id, broker_order_id, symbol, side, actual_qty,
                    actual_price, fees, realized_pnl, arrival_price, slippage_bps,
                    liquidity_flag

`reconcile_with_broker()` diffs broker truth vs. internal FIFO and grades breaks
`none/minor/major`; a major break escalates the risk state. `src/broker.py`
now implements the full dual-engine adapter contract: account equity/cash,
broker-truth positions, latest prices, recent fill activities, order-intent
submission with `client_order_id`, open-order counts, kill-switch cancels,
and API health/error-rate telemetry. Legacy `src/trade_log.py` remains a
shim for existing tests and the dashboard.

### Hedge pair lifecycle (unwind / rotation)

Active Pivot hedges used to lack an exit policy beyond their own sell signal,
so a hedge could linger indefinitely. Pivot origins are now persisted
(`models/hedge_pairs.json`: `pairs`, persistent `origins`, `transitions`):

- **Unwind** — holding hedge H for target T while T regains a legitimate
  direct buy (P≥`BUY_THRESHOLD` AND sentiment passes) sells H and clears the
  pair (`[pivot-unwind]`).
- **Rotation** — holding target T that was ever pivot-expressed while its own
  signal stays ≥BUY but today's card is explicitly bad (<`SENTIMENT_MIN_SCORE`)
  sells T (`[pair-rotate]`); the hedge leg is re-entered next pass by the
  normal pivot path.

Both transitions are one reduction each — entries always reuse the existing
direct/pivot paths on later passes, so no same-pass sell/rebuy races with the
position-count gates. Sentiment-driven transitions run at most once per
calendar day per symbol (mood-card jitter dampener); model-driven exits are
uncapped. Night behavior: equity-leg trades wait for RTH as usual.

The generator also hardens against malformed LLM JSON: strict parse →
sanitize (trailing commas/smart quotes) → one self-correction re-prompt →
regex salvage of valid fragments. Failed outputs are logged head-first into
journald; yesterday's file survives total failure untouched.

### Weekly universe curator (research engine)

`curate_universe.py` (Sundays 22:00 UTC, `financebot-curate.timer`) decides
WHAT is tradable -- the missing half of the macro loop:

1. Quant screen over the 47-symbol candidate pool (liquidity $-volume,
   20/60d momentum, vol, drawdown, corr-to-SPY) from 60d bars
2. FRED macro brief (curve level/slope, CPI YoY, policy rate)
3. Dated news corpus: per-symbol Google News RSS + Fed/Treasury/BLS policy
   feeds (`src/news.py`, stdlib XML, per-feed fail-closed)
4. ONE Gemini call with an explicit epistemics frame: training data declared
   stale; provided dated headlines win; rationale must cite them
5. Deterministic post-validation -- candidate-pool membership, liquidity
   floor ($50M avg daily $vol), <=6 names per sector group, BTC/USD+ETH/USD
   force-included, truncated to `FINANCEBOT_CURATOR_TARGET_SIZE` (32)

Writes `active_universe.json` (schema v1) + `models/universe_rationale.json`
(macro note + per-name cited rationale). Fail-closed: any error keeps last
week's pool trading. Run manually with `--dry-run` to inspect a proposal.

Install alongside the other schedulers:

    sudo cp deploy/financebot-curate.{service,timer} /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now financebot-curate.timer

### Fractionability & market-hours guards

Two execution-path guards close real-money gaps observed in paper soak:

- **Non-fractionable assets** (`broker.is_fractionable()`, cached per process):
  several inverse ETFs reject fractional quantities outright (Alpaca
  40310000). For such assets `_hardened_buy` sizes in WHOLE shares within the
  position cap, and skips cleanly (`non_fractionable_too_expensive`) when even
  one share exceeds it. Unknown fractionability falls back to legacy sizing.
- **Fractionability-aware hedge selection**: non-expressable candidates are
  excluded before correlation ranking (`select_hedge_asset(exclude=...)`) --
  the runner-up hedge is chosen instead of a guaranteed broker rejection.
  Exclusion is tier-aware: names return automatically at larger caps.
- **Closed-market hedge deferral** (`hedge_market_closed`): an Active Pivot to
  an equity-ETF hedge while the equity market is closed would queue a fill at
  stale reference prices. Such pivots are now deferred to the first post-open
  pass instead. Crypto-target processing itself remains 24/7.

### Daily sentiment pipeline (Gemini via OpenRouter)

`generate_sentiment.py` is the morning helper behind `daily_sentiment.json`.
Weekdays at 10:30 UTC (`deploy/financebot-sentiment.timer`) it:

1. Resolves the active universe,
2. Builds a compact momentum context per symbol (last close, 5d/20d return,
   annualized vol) from recent bars,
3. Makes ONE batched OpenAI-compatible chat call — defaults:
   `LLM_BASE_URL=https://openrouter.ai/api/v1`, `LLM_MODEL=google/gemini-3.7-flash`,
   key from the env var named by `LLM_API_KEY_ENV` (default `OPENAI_API_KEY`) —
   requesting strict JSON `{ticker: {score 0-10, summary}}`,
4. Atomically writes `daily_sentiment.json` in the exact schema the loop reads.

Fail-closed: a missing API key or any LLM/parse error leaves yesterday's report
untouched and exits 1. Symbols omitted by the model are simply absent from the
report.

**Missing-score semantics**: `SENTIMENT_MISSING_IS_PASS=true` (env
`FINANCEBOT_SENTIMENT_MISSING_IS_PASS`, default true) treats an absent score as
NEUTRAL-PASS — trade the target normally. An explicit low score always
blocks/pivots regardless; only absence is neutral, never bad news. Set the flag
false to restore legacy pivot-on-missing behavior.

Install the scheduler alongside the trading service:

    sudo cp deploy/financebot-sentiment.{service,timer} /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now financebot-sentiment.timer

### Cash-account guards (T+1 settlement reality)

Paper trading never simulates settlement, but live cash accounts cannot reuse
sell proceeds until T+1. Two guards close that gap (both env-toggleable,
conservative by default):

- **Settled-funds gate** (`src/portfolio_manager.py`) -- every BUY must fit
  inside `get_withdrawable_cash()` (preference ladder:
  `cash_withdrawable` → `non_marginable_buying_power` → `cash`) minus unfilled
  buy commitments tracked in `models/portfolio_state.json`. Unknown funds fail
  closed for buys; sells/reductions are always exempt so de-risking is never
  blocked. Blocked orders are audited in `orders_log.csv` as
  `blocked_settled_cash`.
- **Turnover dampener** (`src/tactical_executor.py`) -- exposure-INCREASING
  order deltas are scaled to `TURNOVER_DAMPING_FACTOR` (0.25) of desired size
  and cooldown-limited to one increase per symbol per
  `SYMBOL_RETRADE_COOLDOWN_SECONDS` (900). Reductions/exits pass through
  untouched, always.

### Risk state machine

`src/guardrails.RiskStateMachine` adds graduated states —
`NORMAL → FREEZE_NEW_ENTRIES → REDUCE_RISK → LIQUIDATE_ONLY → KILL_PROCESS` —
driven by a drawdown ladder, exposure caps, data staleness, and reconciliation
breaks. It escalates immediately but recovers at most one level per evaluation
after a cooldown (never LIQUIDATE_ONLY → NORMAL directly). `is_order_allowed()`
enforces per-order permissions. Legacy `check_circuit_breaker()` still calls
`sys.exit(1)` on a 24h `DAILY_LOSS_LIMIT` breach.

`src/tactical_executor.py` now builds live `RiskMetrics` from broker-truth
portfolio snapshots, persisted high-watermark equity, allocation/data freshness,
model-registry verification, broker-health hooks, open-order counts, API error
rate hooks, and reconciliation diffs before the state machine decides whether
new entries, reductions, liquidation, or process kill are allowed.

### Dual-horizon backtest

`backtest.replay_dual_horizon()` is an event-driven replay mirroring the live
flow (strategist → executor → portfolio manager → fills) so hedge exits, partial
fills, fees, and slippage are validated exactly as in production. Legacy
`simulate()` is unchanged.

## Repository Layout

| Path                       | Role                                                                 |
|----------------------------|---------------------------------------------------------------------|
| `config.py`                | Guardrails, thresholds, symbols, inverse safe list, file paths       |
| `src/data.py`              | Alpaca historical fetch + leakage-free features/labels              |
| `src/validation.py`        | Expanding-window walk-forward harness + final fit                   |
| `src/model_io.py`          | Save/load native XGBoost JSON + feature/threshold sidecar           |
| `src/guardrails.py`        | Legacy guardrails + immutable runtime risk profiles/policies         |
| `src/broker.py`            | Rate-limited alpaca-py broker adapter: orders, clock, snapshots, health |
| `src/sentiment.py`         | Daily-report loader + Active Pivot correlation hedge selection       |
| `src/fifo.py`              | FIFO inventory queue: realized PnL, avg price, positions, persistence|
| `src/trade_log.py`         | Trade CSV writer; SELL PnL computed from FIFO inventory              |
| `src/analytics.py`         | Shared PnL/win-rate/drawdown/Sharpe + open-inventory summary         |
| `src/execution.py`         | Hardened legacy live loop: universe, clock gating, snapshots, risk gate |
| `src/strategist.py`        | Interday Strategist: max-Sharpe core, covariance, hedge overlay, macro regime |
| `src/tactical_executor.py` | Intraday Executor: envelopes → order intents inside risk state |
| `src/portfolio_manager.py` | Broker truth: fills, FIFO, reconciliation, orders/fills logs |
| `src/macro.py`             | FRED/BLS/EDGAR PIT connectors + feature-store population      |
| `deploy/financebot.service`| systemd unit: restart policy, MemoryMax, env-file, hardening  |
| `src/universe.py`          | Active live universe loader, schema validation, startup resolution    |
| `train.py`                 | Legacy direction-model training entrypoint                    |
| `train_dual.py`            | Dual-engine expected_return training (panel walk-forward)     |
| `build_feature_store.py`   | Offline PIT store population (bars + FRED/BLS/EDGAR)         |
| `run_bot.py`               | Live entrypoint; `--engine` or `FINANCEBOT_ENGINE` selects engine    |
| `backtest.py`              | Hybrid backtester with `--seed` + date-aligned sentiment mock       |
| `dashboard.py`             | Read-only Streamlit + Plotly dashboard                              |
| `tests/`                   | Pytest suite (all external APIs mocked; no network)                 |
| `tests/test_run_loop.py`   | Permanent end-to-end integration guard for `run()` pivot routing     |
| `tests/test_broker.py`     | Offline broker-adapter contract tests; no Alpaca network calls       |
| `tests/test_universe.py`   | Active-universe schema, precedence, strict/live failure tests         |
| `tests/test_risk_profiles.py` | Immutable risk policy formulas and override validation             |
| `tests/test_execution_hardening.py` | Legacy VPS loop risk-gate, clock-gate, and cache tests      |
| `active_universe.example.json` | Operator template for `active_universe.json`                     |

## Data & State Files

| File                            | Written by                   | Read by                         |
|---------------------------------|------------------------------|---------------------------------|
| `models/model.json`             | `train.py`                   | `run_bot.py`, `backtest.py`     |
| `models/feature_meta.json`      | `train.py`                   | model loader (feature order)    |
| `models/bot_state.json`         | circuit breaker (24h anchor) | guardrails                      |
| `models/inventory_state.json`   | `src/trade_log.py` (FIFO)    | `dashboard.py` (Open Positions) |
| `active_universe.json`          | operator-local (gitignored)  | legacy `run_bot.py` startup     |
| `active_universe.example.json`  | repo template                | copied/trimmed by operator      |
| `daily_sentiment.json`          | FinRobot/LLM (out-of-band)   | `run_bot.py`, `dashboard.py`    |
| `trades_log.csv`                | `src/trade_log.py`           | `dashboard.py`                  |
| `historical_sentiment.csv`      | `backtest.py` (auto-aligned) | `backtest.py`                   |
| `backtest_results.csv`          | `backtest.py`                | manual review                   |
| `orders_log.csv`                | `src/portfolio_manager.py`   | audit / reconciliation          |
| `fills_log.csv`                 | `src/portfolio_manager.py`   | audit / realized PnL truth      |
| `models/portfolio_state.json`   | `src/portfolio_manager.py`   | dual engine restart recovery    |
| `models/risk_state.json`        | `src/guardrails.py`          | risk machine restart recovery   |
| `data/feature_store/records.jsonl` | `build_feature_store.py`  | `run_dual()`, `train_dual.py`   |
| `data/feature_store/news.jsonl`    | PIT store persistence     | `run_dual()` (news state)       |
| `models/registry/registry.json`    | `train_dual.py` / registry| strategist model loading        |

`trades_log.csv` schema: `timestamp,ticker,side,price,size,pnl`
(`ticker` is the EXECUTED symbol -- target or inverse hedge; `pnl` is realized
FIFO PnL on SELL rows, `0.0` on BUY rows).

## Hard Guardrails (`config.py`)

Legacy constants are intentionally unchanged and remain import-stable:

- `MAX_POSITION_SIZE = 5.00` -- legacy default notional cap per trade.
- `DAILY_LOSS_LIMIT = -10.00` -- legacy rolling 24h loss limit.
- `MAX_OPEN_POSITIONS = 3` -- legacy concurrent position cap.
- `PAPER` defaults to `True`; live still requires explicit `FINANCEBOT_PAPER=false`.
- `ENGINE` defaults to `legacy`; this deployment pass does **not** make dual VPS execution ready.

### Active live universe

`src/universe.py` resolves live legacy targets once at startup only. Research,
training, backtesting, sentiment generation, and dual replay keep using their
existing universes (`EQUITY_SYMBOLS`, `CRYPTO_SYMBOLS`, `CORE_UNIVERSE`).

- `active_universe.json` is operator-local and gitignored; copy from `active_universe.example.json`.
- Schema: `{"version": 1, "as_of": "2026-08-03T00:00:00Z", "symbols": ["SPY", "QQQ", "BTC/USD"]}`.
- `version` must be integer `1`; `as_of` accepts ISO-8601 UTC timestamps or `YYYY-MM-DD`; naive timestamps are interpreted as UTC.
- `symbols` must be a non-empty string list; each symbol is `strip()`ed, `upper()`ed, deduplicated preserving order, and matched against `^[A-Z0-9][A-Z0-9./-]{0,14}$`.
- Symbols containing `/` are classified as crypto; all others are equities.
- One invalid symbol invalidates the whole file; no partial files are silently accepted.
- The code-defined candidate set has 45 equity/ETF targets plus `BTC/USD` and `ETH/USD` (47 total), but it is activated only by file contents or `FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE=true`.
- `INVERSE_SAFE_LIST` is not merged into targets; inverse ETFs are hedges selected by Active Pivot and remain subject to risk checks.

Live universe precedence:

- Path: explicit loader argument > `FINANCEBOT_UNIVERSE_FILE` > `ACTIVE_UNIVERSE_PATH` (`active_universe.json`).
- Cap: explicit loader argument > `FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE` > `MAX_LIVE_UNIVERSE_SIZE` (`50`). Env caps may only tighten and must be `1..50`.
- Maximum age: explicit loader argument > `FINANCEBOT_UNIVERSE_MAX_AGE_HOURS` > `UNIVERSE_MAX_AGE_HOURS` (`96.0`).
- Strict mode is on when `PAPER` is false or `FINANCEBOT_UNIVERSE_STRICT=true`.

Failure behavior:

- Missing optional file in non-strict paper mode uses `EQUITY_SYMBOLS + CRYPTO_SYMBOLS`.
- Missing file in strict/live mode raises `UniverseError` unless `FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE=true`.
- Malformed/invalid/unreadable explicit files warn and fall back only in non-strict paper mode; strict/live mode stops before broker initialization.
- Stale files warn but stay active in non-strict paper mode; strict/live mode stops.
- Over-cap files truncate deterministically to the first `N` symbols only in non-strict paper mode; strict/live mode stops.

### Immutable risk profiles

`src/guardrails.py` adds an immutable `RiskPolicy` resolved once at process
startup. Environment overrides may tighten a profile only; invalid values raise
`RiskConfigurationError`, while loosening values are clamped to the code-defined
ceiling with a warning.

| Profile      | Position % | Gross % | Daily loss % | Abs position | Abs daily loss | Max positions |
|--------------|------------|---------|--------------|--------------|----------------|---------------|
| `research`   | none       | none    | none         | `$5.00`      | `$10.00`       | `3`           |
| `micro_live` | `0.02`     | `0.25`  | `0.005`      | `$100.00`    | `$50.00`       | `3`           |
| `small_live` | `0.05`     | `0.50`  | `0.01`       | `$1000.00`   | `$250.00`      | `5`           |
| `growth_live`| `0.12`     | `0.85`  | `0.03`       | `$5000.00`   | `$750.00`      | `8`           |
| `soak`       | none       | `0.05`  | none         | `$5.00`      | `$10.00`       | `50`*         |

At a $25k anchor, `growth_live` resolves to a $3,000/order cap (pct-bound),
-$750 daily loss, 8 concurrent names. **`growth_live` is evidence-gated** --
see "Promotion evidence gate" below.

\* soak: with `FINANCEBOT_SLOTS_FOLLOW_UNIVERSE=true` the effective slot
count equals the startup universe size (pool of 32 -> 32 slots); 50 is the
hard ceiling.

### Stale-conviction exit (on-call long-horizon strategist)

A held target that goes **stale** (>= 7 days, `FINANCEBOT_STALE_EXIT_DAYS`)
and **conviction-dead** (P(up) inside the 0.45-0.55 dead zone,
`FINANCEBOT_STALE_EXIT_LOW/HIGH`) triggers an on-call strategist consult --
a deep research pass, not a daily glance:

- **30-day news trajectory** for the name (dated headlines, story arc)
- **FRED macro regime** (curve, CPI, policy rate)
- **Long-horizon stats** (126d/252d momentum, vol, drawdown)
- Position facts (age, entry-vs-current, today's mood + P(up))

The strategist rules **KEEP or DISCARD** (strict JSON, must cite specific
headlines/data). KEEP holds the position today and is re-consulted daily
while still stale; DISCARD releases it via the standard exit path
(`[stale-exit]`). Verdicts are cached once per symbol per day
(`models/strategist_verdicts.json`); any LLM failure defaults to KEEP.
Targets only -- hedges remain governed by pair lifecycle + own signals.
Slots: `FINANCEBOT_SLOTS_FOLLOW_UNIVERSE=true` sizes the book to the pool.

Exact formulas:

- Position cap with trusted positive equity: `min(policy.max_position_size_abs, equity * policy.max_position_pct)`.
- Position cap when `max_position_pct is None`: `policy.max_position_size_abs`.
- Rolling 24h loss threshold: `-min(policy.daily_loss_limit_abs, anchor_equity * policy.daily_loss_pct)`.
- Daily loss when `daily_loss_pct is None`: `-policy.daily_loss_limit_abs`.
- The threshold is always negative; the stricter-limit formula is negative `min()` of positive magnitudes.
- At `$10,000` equity: research cap `$5` / loss `-$10`; micro cap `min($200, $100) = $100` / loss `-min($50, $50) = -$50`; small cap `min($500, $1000) = $500` / loss `-min($100, $250) = -$100`.

Pre-trade risk gate formulas:

- `current_notional = abs(broker position market_value)`.
- `pending_increase_notional = notional of pending buy orders for that symbol`.
- `effective_notional_before = current_notional + pending_increase_notional`.
- Buy: `projected_symbol_notional = effective_notional_before + requested_order_notional`.
- Verified sell: `projected_symbol_notional = max(0, current_notional - sell_notional) + pending_increase_notional`.
- Enforced gates: total symbol cap, max open positions, projected gross exposure, trusted positive price, long-only sell quantity.
- Pending buys count toward symbol cap, gross exposure, and open-position count.
- Exit/reduction orders remain prioritized when position truth is trusted, including when gross exposure is already high or equity is unknown.
- Target caps exclude exit-only held symbols so removing a target does not force liquidation; risk calculations still include every broker/FIFO-held position and hedge.

### Promotion evidence gate (`growth_live`)

Bigger allowance must be earned. When `FINANCEBOT_RISK_PROFILE=growth_live`,
`run_bot.py` calls `guardrails.verify_promotion_evidence()` at startup, BEFORE
any broker initialization. The gate requires the operator-local trade logs to
prove a qualifying prior-tier track record:

- **>= 50 executed trades** in `trades_log.csv` (legacy) or `fills_log.csv` (dual),
- **cumulative FIFO PnL > 0**,
- **p95 |slippage_bps| <= 25** when slippage data exists (fills log),
- current `risk_state.json` must not be `KILL_PROCESS`.

Failure prints an itemized report and exits with `SystemExit(1)`. Missing or
unreadable logs fail closed. Lower tiers are never gated.

Escape hatch: `FINANCEBOT_ALLOW_UNEARNED_PROMOTION=true` arms the profile
without evidence — and prints an explicit OVERRIDE banner at every startup so
the choice is permanently visible in logs.

Known limitation: major reconciliation breaks are not persisted as events yet;
the gate approximates track-record quality from fills/PnL/slippage plus the
current risk state. A `risk_events.csv` append-log is the noted follow-up.

### Cost-aware entry gate (dual engine)

`TacticalExecutor.process_symbol()` refuses exposure-INCREASING intents whose
allocation-row `expected_return` does not clear
`max(MIN_TRADE_EDGE_BPS=10, ESTIMATED_ROUND_TRIP_COST_BPS=20)` bps. A missing
or non-finite estimate fails closed. Reductions/exits are never gated. At
$1k+ clips this blocks most negative-EV churn — worth more than any cap raise.
Legacy engine keeps its `BUY_THRESHOLD` proxy for now.

### Circuit breaker and market hours

- The daily loss anchor is a persisted rolling 24-hour UTC anchor in `models/bot_state.json`, not a US calendar-day reset.
- On breach, the legacy loop attempts `Broker.cancel_all_open_orders()` and exits with `SystemExit(1)` regardless of cancellation outcome.
- The equity market clock is fetched once per loop pass. Closed or unknown clock state skips equity bars and equity orders before data download.
- Crypto pairs containing `/` remain eligible while the equity market is closed.
- Broker snapshot failures fail closed: unknown positions or open orders block new submissions and trigger cancel/retry; unknown equity permits only verified reductions/closures.
- Per-pass telemetry logs elapsed time, active target count, processed count, submitted count, broker error rate when available, and skipped reason counts.

### Environment variables

| Variable | Type | Default / precedence |
|----------|------|----------------------|
| `FINANCEBOT_PAPER` | bool | safe default `true`; live only with explicit `false` |
| `FINANCEBOT_ENGINE` | enum | `legacy` by default; deployment examples use `legacy` |
| `FINANCEBOT_UNIVERSE_FILE` | path | overrides `ACTIVE_UNIVERSE_PATH` |
| `FINANCEBOT_UNIVERSE_STRICT` | bool | `false`; strict also turns on automatically when paper is false |
| `FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE` | bool | `false`; explicit opt-in to 47-symbol code candidate set |
| `FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE` | int | defaults to `50`; may only tighten within `1..50` |
| `FINANCEBOT_UNIVERSE_MAX_AGE_HOURS` | float | defaults to `96.0` |
| `FINANCEBOT_RISK_PROFILE` | enum | `research`; choose `research`, `micro_live`, or `small_live` |
| `FINANCEBOT_MAX_POSITION_PCT` | fraction | optional tightening override, `0 < value <= 1` |
| `FINANCEBOT_MAX_GROSS_EXPOSURE_PCT` | fraction | optional tightening override, `0 < value <= 1` |
| `FINANCEBOT_DAILY_LOSS_LIMIT_PCT` | fraction | optional tightening override, `0 < value <= 1` |
| `FINANCEBOT_MAX_POSITION_SIZE_ABS` | positive float | optional tightening override |
| `FINANCEBOT_DAILY_LOSS_LIMIT_ABS` | positive float | optional tightening override; stored as positive magnitude |
| `FINANCEBOT_MAX_OPEN_POSITIONS` | positive int | optional tightening override |
| `FRED_API_KEY` | str | optional; unused by default (keyless fredgraph endpoint) |
| `BLS_API_KEY` | str | optional; switches BLS connector to API v2 |
| `SEC_CONTACT_EMAIL` | str | optional; SEC requires a contact User-Agent |
| `FINANCEBOT_ENFORCE_SETTLED_CASH_GATE` | bool | default `true`; blocks buys beyond settled cash |
| `FINANCEBOT_CASH_ACCOUNT_DAMPING` | bool | default `true`; scales/cooldowns exposure increases in the dual executor |
| `FINANCEBOT_ALLOW_UNEARNED_PROMOTION` | bool | default `false`; override the growth_live evidence gate (loudly logged) |
| `MIN_TRADE_EDGE_BPS` / `ESTIMATED_ROUND_TRIP_COST_BPS` | float | code constants; dual-engine entry edge floor / cost model |
| `FINANCEBOT_SENTIMENT_MISSING_IS_PASS` | bool | default `true`; absent sentiment = neutral-pass |
| `FINANCEBOT_BUY_THRESHOLD` / `FINANCEBOT_SELL_THRESHOLD` | float | decision conviction gates (defaults 0.58/0.42; validated 0<sell<buy<1) |
| `FINANCEBOT_LLM_BASE_URL` | URL | default OpenRouter (`https://openrouter.ai/api/v1`) |
| `FINANCEBOT_LLM_MODEL` | str | default `z-ai/glm-5.3-flash` (cheap flash tier, reasoning off) |
| `FINANCEBOT_LLM_MAX_TOKENS` | int | default 8000 (headroom for 32-symbol JSON) |
| `OPENAI_API_KEY` (or custom via `FINANCEBOT_LLM_API_KEY_ENV`) | str | REQUIRED for sentiment + universe curator |
| `FINANCEBOT_LOOP_INTERVAL_SECONDS` | float | execution loop cadence (default 60; >= 5 validated) |
| `FINANCEBOT_CURATOR_TARGET_SIZE` | int | weekly pool size (default 32) |
| `FINANCEBOT_CURATOR_LIQUIDITY_FLOOR_USD` | float | quant liquidity floor (default $50M) |

Also see `README_SIMPLE.md` for a plain-language tour of the whole system.

## FIFO Realized PnL

- `src/fifo.py` maintains per-ticker lot queues.
- BUY pushes a lot; SELL matches the **oldest lots first**, realizing
  `matched_qty * (sell_price - lot_price)`.
- Handles partial fills, multi-lot exits, empty-queue sells (returns `0.0`,
  never fabricates a short), and oversized sells (only realizes existing lots).
- Persisted to `models/inventory_state.json` so cost basis survives restarts.
- `src/trade_log.append_trade()` owns PnL: any caller-supplied `pnl` is ignored;
  the recorded value is always FIFO-derived. Hedge buys are keyed by the inverse
  ticker so inventory reflects what is actually held.
- Introspection for the dashboard: `open_qty()`, `average_price()`, `positions()`.

## Dashboard (decoupled, read-only)

    streamlit run dashboard.py

Sections, top to bottom:

1. **Metric cards** -- Total Trades, Win Rate, Realized PnL, Max Drawdown.
2. **Open Positions** -- live FIFO inventory from `models/inventory_state.json`:
   Ticker, **Type** (`📈 Direct Hold` vs `🛡️ Hedge`), Open Inventory, Avg Entry
   Price, Last Price, Unrealized PnL, plus Total Unrealized PnL / Direct Holds /
   Hedges metric cards. Type is derived via `classify_position()` (a position is
   a Hedge when its base asset is in `INVERSE_SAFE_LIST`). Missing price feed
   falls back to avg entry price (unrealized = 0).
3. **Realized PnL / Past Trades** -- cumulative realized-PnL equity curve, a
   **Hedge vs Direct Performance** breakdown (per-type realized PnL, trade count,
   and win rate via `realized_pnl_breakdown()`), and a recent-executions table
   (clearly separated from Open Positions).
4. **FinRobot Daily Sentiment** -- per-symbol score with PASS/BLOCK badges.

The dashboard only ever *reads* files; it never routes orders or writes state.

## Backtester

    python backtest.py             # random sentiment scores each run
    python backtest.py --seed 42   # deterministic, repeatable scores

- Downloads held-out bars once, then **auto-generates `historical_sentiment.csv`
  aligned to the actual bar dates** (static dates never intersect a trailing
  window -> the original zero-trades bug).
- `--seed` makes `generate_historical_sentiment()` deterministic via a seeded RNG.
- Prints Total PnL, Win Rate, Max Drawdown, Sharpe; saves `backtest_results.csv`.

## Setup

> **Prerequisites:** You MUST run `python train.py` at least once before
> starting the live loop (`run_bot.py`) or the backtester (`backtest.py`). Both
> fail fast at startup if `models/model.json` or `models/feature_meta.json` is
> missing, empty, or a 1-byte placeholder, printing a message telling you to
> train first. This prevents trading on a non-existent or corrupt model.

1. Python 3.11+ and install deps:

   Debian/Ubuntu VPS note: there is no bare `python` binary by default -- use
   a virtualenv, which gives you one (recommended, matches
   `deploy/financebot.service`):

       sudo apt install python3-venv python3-pip
       python3 -m venv .venv
       source .venv/bin/activate     # now `python` works inside this shell
       pip install -r requirements.txt

   Or, system-wide alias instead of a venv:

       sudo apt install python-is-python3
       pip install -r requirements.txt

   All later commands (`python train.py`, `python -m pytest -q`,
   `streamlit run dashboard.py`) assume that venv is activated.

2. Set credentials (never hard-coded):

   Windows example:

       setx APCA_API_KEY_ID "your_key_id"
       setx APCA_API_SECRET_KEY "your_secret_key"
       setx FINANCEBOT_PAPER "true"
       setx FINANCEBOT_ENGINE "legacy"

   Linux/VPS example:

       export APCA_API_KEY_ID="your_key_id"
       export APCA_API_SECRET_KEY="your_secret_key"
       export FINANCEBOT_PAPER="true"
       export FINANCEBOT_ENGINE="legacy"

   Paper trading is the safe default. Do not set `FINANCEBOT_PAPER=false` during
   this restricted paper-deployment pass.

3. Train the model locally/offline:

       python train.py

4. Prepare a small operator-local live universe:

       cp active_universe.example.json active_universe.json
       # update as_of, then trim to roughly 15-25 symbols for a 2-vCPU/4-GB VPS

5. Run components:

       python run_bot.py                  # env-selected engine, defaults legacy
       python backtest.py --seed 42       # reproducible local/offline backtest
       streamlit run dashboard.py         # read-only local monitor

   Dual-horizon engine (opt-in, after legacy is stable):

       python build_feature_store.py      # populate + persist PIT store (once)
       python train_dual.py               # train + register strategist model
       python run_bot.py --engine dual    # strategic + tactical engine

## Legacy Paper VPS Deployment Path

This patch hardens the **legacy** engine for restricted paper execution on a
small Linux VPS. The VPS should do only execution, risk enforcement, broker
reconciliation, state persistence, and logging. Research, model training,
backtesting, sentiment generation, and universe selection remain local/offline.

Recommended order:

1. **Local correctness** -- `python -m pytest -q` must pass with all Alpaca calls mocked.
2. **Prepare universe** -- copy `active_universe.example.json` to `active_universe.json`, update `as_of`, and trim to the initial 15-25 targets.
3. **Paper VPS start** -- run `FINANCEBOT_PAPER=true`, `FINANCEBOT_ENGINE=legacy`, `FINANCEBOT_RISK_PROFILE=micro_live`, and strict universe mode.
4. **Soak and inspect** -- watch loop telemetry, broker-health warnings, cancellation behavior, `trades_log.csv`, FIFO state, and circuit-breaker anchor state.
5. **Dual remains opt-in until soaked** -- `FINANCEBOT_ENGINE=dual` boots only
   after `build_feature_store.py` has populated the persisted PIT store; it still
   refuses (`SystemExit(1)`) on an empty or corrupt store. Promote it to paper
   default only after a clean dual-engine soak.

Example Linux VPS environment:

    export APCA_API_KEY_ID="..."
    export APCA_API_SECRET_KEY="..."
    export FINANCEBOT_PAPER="true"
    export FINANCEBOT_ENGINE="legacy"
    export FINANCEBOT_RISK_PROFILE="micro_live"
    export FINANCEBOT_UNIVERSE_STRICT="true"
    export FINANCEBOT_UNIVERSE_FILE="/opt/financebot/active_universe.json"
    export FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE="25"
    python run_bot.py

Or run it as a service with restart policy, memory cap, and env-file:

    sudo cp deploy/financebot.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now financebot

Expected startup/loop telemetry:

- Universe summary: source path, symbol count, cap, `as_of`, and stale status.
- Startup risk summary: risk profile and active target count.
- Pass summary: elapsed seconds, active target count, processed count, submitted order count, market-open state, broker API error rate when available, and skipped reason counts.
- Overrun warning when elapsed wall time exceeds `LOOP_INTERVAL_SECONDS`; the next pass starts without an extra full-interval sleep.

Why the 15-25 target recommendation:

- `API_CALL_DELAY_SECONDS = 1.0` deliberately rate-limits every Alpaca call.
- A 45-50 symbol universe can make the first paper deployment too slow on a 2-vCPU/4-GB VPS.
- `MAX_LIVE_UNIVERSE_SIZE` remains `50`, but start smaller and scale only after telemetry is clean.

Dual VPS execution requires a populated feature store: run
`python build_feature_store.py` (Alpaca bars + FRED + BLS + EDGAR into
`data/feature_store/records.jsonl`), then `python train_dual.py` to register the
strategist's `expected_return` model. Without those artifacts `run_dual()` fails
closed before any broker initialization.

For the local admin model, prefer a separate service boundary (for example
Ollama for simple Gemma/Qwen deployment, or vLLM on a GPU VPS). Give it
read-only access to logs/state by default and route any mutation through a
human-approved command layer.

## Determinism

- Training pins `random_state`; the live path is a pure function of the latest
  feature row and (for pivots) the trailing correlation window.
- Backtest sentiment is reproducible with `--seed`.
- FIFO PnL is exact and order-dependent by construction.

## Testing

    python -m pytest -q

- All Alpaca market-data/order calls and Streamlit/Plotly UI are mocked; the
  suite never hits the network (per AGENTS.md).
- Coverage includes: broker adapter methods, broker risk snapshots, market-clock gating, env runtime defaults,
  immutable risk profiles, guardrail typing/clamping, dynamic circuit breaker cancellation + `sys.exit()`,
  leakage-free features, active-universe schema/precedence/strict failures,
  projected pre-trade risk gates, sentiment scoring, **Active Pivot hedge selection**
  (deterministic mocked correlations picking a specific inverse ETF), FIFO edge
  cases (partial / full / empty / oversized / persistence), date-aligned
  sentiment generation, backtest merge alignment, inventory summary math,
  hedge-vs-direct realized PnL breakdown, and dashboard empty-state /
  open-positions / breakdown rendering.
- `tests/test_universe.py` covers missing/invalid/stale/over-cap active universes,
  symbol normalization, candidate count, inverse exclusion, and deterministic env precedence.
- `tests/test_risk_profiles.py` covers exact research/micro/small formulas,
  tightening/clamping/rejected overrides, unknown profiles, and dynamic loss breaches.
- `tests/test_execution_hardening.py` covers equity clock fail-closed behavior,
  crypto eligibility, unknown broker truth, repeated/pending order exposure caps,
  gross exposure exits, non-target exit-only behavior, hedge risk checks, data-cache reuse,
  and circuit-breaker open-order cancellation.
- `tests/test_run_loop.py` is a permanent **integration test** that drives the
  real `src.execution.run()` loop with a stub broker + mocked correlation
  universe (a patched `time.sleep` ends it after one pass). It strictly guards
  dynamic pivot routing plus deterministic startup universe resolution.

### Dual-horizon engine tests

No test may hit Alpaca, SEC, FRED, BLS, EDGAR, or any news provider live — the
entire dual engine is exercised with mocks, doubles, and in-memory stores:

- `tests/test_legacy_contract.py` — freezes every legacy public contract.
- `tests/test_model_registry.py` — missing-role rejection + feature-column order.
- `tests/test_pit_data.py` — `available_at > as_of` exclusion, macro-revision
  vintage, news decay/direction/intensity, missing indicators (no backfill).
- `tests/test_validation_panel.py` — deterministic folds after shuffle, no
  train-after-test / label overlap, embargo covers max label horizon.
- `tests/test_strategist.py` — negative-covariance hedge, zero-variance/singular
  covariance safety, low-effectiveness → no hedge, orphaned-hedge exit, PSD cov.
- `tests/test_portfolio_manager.py` — partial fills, duplicate `fill_id`
  idempotency, sell PnL net of fees, actual (not stale) fill price, major
  reconciliation break.
- `tests/test_risk_machine.py` — every state transition, per-order permissions,
  gradual recovery, and the preserved legacy circuit breaker.
- `tests/test_tactical_executor.py` — orphaned-hedge processing/exit, hard
  `MAX_POSITION_SIZE` re-cap, freeze blocks entries, kill cancels + exits.
- `tests/test_dual_replay.py` — full mocked strategist→executor→PM→fills replay.
- `tests/test_run_bot_hardening.py` — empty PIT feature store exits before broker initialization.
- `tests/test_macro_connectors.py` — FRED/BLS/EDGAR parsing, publication-lag
  vintage safety, rate limiting, fail-closed transport failures, and store
  population (all transports mocked; zero network).
- `tests/test_pit_persistence.py` — JSONL round-trip, vintage gating after
  reload, corrupt-file fail-closed behavior, `latest_available_at()` telemetry.
- `tests/test_max_sharpe_optimizer.py` — tangency beats heuristic on crafted
  correlated inputs, determinism, long-only + cap compliance, degenerate-mu
  fallback, and every macro regime transition.
- `tests/test_train_dual.py` — panel builder leakage safety (label_end_time >
  timestamp, labels inside stored history) on a synthetic store.
- `tests/test_cash_account_guards.py` — settled-funds gate (submit/block/
  fail-closed/fallback/pending-release/config-off), sells exempt, and every
  turnover-dampener rule (scale, cooldown, reduction passthrough, disable).
- `tests/test_promotion_gate.py` — growth_live arming blocked without logs /
  short history / negative PnL / slippage-p95 breach / KILL state; qualifying
  record passes; lower tiers never gated; override banner path.
- `tests/test_risk_profiles.py` + growth_live exact formulas at $10k/$25k
  anchors, tighten-only overrides, and gated-profile set contract.

## Notes / Next Steps

- Tune `INVERSE_SAFE_LIST` / `HEDGE_CORR_LOOKBACK_DAYS` for your hedge universe.
- Tune features in `src/data.FEATURE_COLUMNS` and hyperparameters in
  `src/validation.DEFAULT_PARAMS`; both flow through validation automatically.
- The FinRobot daily report is consumed as a pre-computed file; generating it is
  out-of-band by design to keep the LLM out of the latency-critical path.




