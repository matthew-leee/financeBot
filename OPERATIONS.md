# OPERATIONS.md — financeBot live-run runbook & handoff

> **Final goal (pi convention):** production-ready autonomous paper-trading
> platform with an evidence-gated promotion path to real capital — currently in
> the *de-risking review* phase after two months of live paper operation.

Read this file **before** touching the live bot. `AGENTS.md` holds the
engineering discipline (tests → commit → changelog → both READMEs); this file
holds what only an operator knows.

**Last verified:** 2026-10-25 (America/UTC, via server inspection)

---

## 1. Two-tree layout (the #1 gotcha)

| Tree | Role | Rule |
|---|---|---|
| `/root/financeBot` | **Development copy** (git repo, agent workspace) | Edit here, run tests here, commit/push here |
| `/opt/financebot` | **Live deployment** (owned by `financebot`, run by systemd) | **Deploy target only — never edit or commit here.** It carries a stale `.git` copy; ignore it |

Editing `/root` alone changes nothing in production. Deploy explicitly:

```bash
# From /root/financeBot after tests pass and the commit is pushed:
rsync -a --exclude '.venv' --exclude 'models/' \
  --exclude 'active_universe.json' --exclude 'daily_sentiment.json' \
  --exclude '*.csv' --exclude '__pycache__' \
  /root/financeBot/ /opt/financebot/
chown -R financebot:financebot /opt/financebot
systemctl restart financebot
journalctl -u financebot -n 15 --output=cat
```

Verify the restart log shows:
`[risk] Evidence gate …` / `OVERRIDE ACTIVE` (if growth_live), `slots follow
universe -> max_open_positions=N`, `[startup] risk profile=… active_targets=N`,
then `[telemetry]` lines.

---

## 2. Secrets & environment

- All secrets live in **`/etc/financebot.env`** (mode `600`, root-owned):
  `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` (Alpaca paper), `OPENAI_API_KEY`
  (OpenRouter key used by the sentiment/strategist/curator LLMs), plus
  `FINANCEBOT_*` runtime knobs.
- **Never `cat` / `echo` that file.** To inspect variable *names* only:
  `grep -oE '^[A-Z_]+' /etc/financebot.env`
- Shell sessions do not inherit it. For manual runs:
  `set -a && source /etc/financebot.env && set +a`
- `sudo -u financebot …` **loses the environment** (and `financebot` cannot read
  the root-only env file). For one-off live-tree scripts, run as root and then
  `chown financebot:financebot` any files you touched under `/opt/financebot`.
- Paper trading is the default: `FINANCEBOT_PAPER=true`. Live requires an
  explicit `FINANCEBOT_PAPER=false` — never set it without a decision record.

---

## 3. Services & timers

| Unit | Schedule | Purpose |
|---|---|---|
| `financebot.service` | continuous (loop every 150 s) | trading loop (legacy engine) |
| `financebot-sentiment.timer` → `.service` | weekdays ~10:30 UTC (+0–5 min jitter) | daily mood card (`daily_sentiment.json`) via OpenRouter LLM |
| `financebot-curate.timer` → `.service` | Sundays 22:00 UTC (+0–5 min jitter) | weekly universe curation (`active_universe.json` + `models/universe_rationale.json`) |

Notes:
- The live loop **hot-refreshes** its universe when `active_universe.json`
  changes (fingerprint check per pass, swapped between passes, `[universe]
  REFRESHED` line). Curator runs therefore need **no restart**.
- `Restart=on-failure` + `StartLimitIntervalSec/Burst` are in the unit's
  `[Unit]` section. A circuit-breaker `sys.exit(1)` is an *intentional* exit —
  after 5 rapid trips the unit parks in `failed` and stays down (see §6).
- The bot has also been observed receiving **graceful external restarts** of
  unknown origin (user/hermes?). State files reload cleanly; do not be
  surprised by `ActiveEnterTimestamp` changes.

---

## 4. State & data files (all gitignored under `models/` and `*.csv`)

Live tree (`/opt/financebot`):

| File | Written by | Meaning |
|---|---|---|
| `trades_log.csv` | loop (at order submission) | append-only order log; **source of truth for trade history** |
| `daily_sentiment.json` | sentiment generator | mood card (schema: `{TICKER: {date, score, summary}}`) |
| `active_universe.json` | curator / operator | tradable pool (`version:1, as_of, symbols[]`) |
| `models/model.json`, `models/feature_meta.json` | `train.py` | legacy XGBoost direction model + feature order |
| `models/bot_state.json` | guardrails | circuit-breaker 24h equity anchor |
| `models/inventory_state.json` | loop (FIFO) | lots, used by dashboard & PnL |
| `models/hedge_pairs.json` | loop | `{pairs, origins, transitions}` — hedge→target lifecycle + daily dampeners |
| `models/strategist_verdicts.json` | loop | cached KEEP/DISCARD verdicts (date + confidence + cited reason) + re-entry negotiations |
| `models/resize_state.json` | Gardener | per-symbol resize cooldown + once-per-day no-action log |
| `models/sizing_log.csv` | Gardener | per-order size decomposition notebook (stats engine) |
| `models/universe_rationale.json` | curator | macro note + per-symbol cited rationale + audit drops |
| `models/risk_state.json` | risk machine | persisted risk state / HWM (dual engine) |
| `models/portfolio_state.json` | portfolio manager | fill ids, FIFO, pending settled-cash commitments (dual engine) |

**Tests are hermetic:** `tests/conftest.py` redirects every runtime-state path
to per-test tmp dirs and forces `SIZING_MODE=flat` for legacy-semantics tests.
Never remove that fixture — without it, tests pollute real state files.

**Known accounting caveat:** `trades_log.csv` PnL is computed from *logged
reference prices*, not actual fill prices, and there are known phantom dust
lots (BTC/ETH). Notebook-implied PnL has drifted ~$250 from broker-true equity.
`fills_log.csv` / `PortfolioManager` (dual engine) is the planned fix.

---

## 5. Current runtime posture (verified 2026-10-25)

`/etc/financebot.env` (relevant lines):

```
FINANCEBOT_PAPER=true
FINANCEBOT_ENGINE=legacy
FINANCEBOT_UNIVERSE_STRICT=true
FINANCEBOT_LOOP_INTERVAL_SECONDS=150
FINANCEBOT_UNIVERSE_MAX_AGE_HOURS=200
FINANCEBOT_RISK_PROFILE=growth_live
FINANCEBOT_SLOTS_FOLLOW_UNIVERSE=true
FINANCEBOT_ALLOW_UNEARNED_PROMOTION=true
```

- `growth_live`: per-order cap `min($5,000, 12% × equity)`; daily breaker
  −$750; drawdown ladder −3%/−5%/−8%/−12%.
- **The promotion gate is currently overridden.** `verify_promotion_evidence`
  requires ≥50 fills **and positive cumulative net PnL**; the book is negative,
  so a restart without `ALLOW_UNEARNED_PROMOTION=true` will **abort at startup**
  (this already happened once — see §6). The override prints a loud banner at
  every boot, by design.
- **Do not change the risk profile, override, or sizing knobs without an
  explicit operator decision.** The audit recommendation (roll back to
  `soak`/`small_live`) is a *pending decision*, not an instruction to act.

---

## 5b. In-flight work — owned by the pi session (inventory 2026-10-25)

The working tree contains a substantial **uncommitted feature batch authored by
another agent** (pi). HEAD (`f0a5b6d`) is **309/309 tests green**; the failures
below exist only in this WIP. **The pi session owns finishing it.**

Uncommitted/untracked WIP:

| Area | Files |
|---|---|
| Accounting truth layer | `src/accounting.py`, `src/account_sync.py`, `src/account_dashboard.py`, `audit_account.py`, `tests/test_accounting.py`, `tests/test_account_sync.py`, `tests/test_account_dashboard.py` |
| Replay / research replay | `src/replay.py`, `research_replay.py`, `tests/test_replay.py`, `tests/test_research_replay.py` |
| Portfolio & hedge research | `tests/test_portfolio_sharpe.py`, `tests/test_hedge_correlation_guard.py` |
| Validation & data | `src/validation.py`, `tests/test_calendar_validation.py`, `tests/test_data.py` |
| Loop & risk plumbing | `src/execution.py`, `src/broker.py`, `src/data.py`, `src/macro.py`, `src/portfolio_manager.py`, `src/sentiment.py`, `src/trade_log.py`, `src/analytics.py`, `config.py`, `train.py`, `curate_universe.py`, `generate_sentiment.py`, `dashboard.py`, `backtest.py` |
| Docs & test hygiene | `README.md`, `README_SIMPLE.md`, `CHANGELOG.md`, `tests/conftest.py`, `tests/test_test_isolation.py`, plus several legacy test files |

**Currently failing (10) — must be green before anything is committed:**

```
tests/test_calendar_validation.py   ×5
tests/test_execution_hybrid.py      ×2
tests/test_macro_connectors.py      ×2
tests/test_data.py                  ×1
```

Rules for this WIP:

1. **Finish it first.** Get the suite green, then commit it (with its CHANGELOG +
   README updates) before starting new work. Do not start parallel features.
2. **Do not duplicate work.** This batch already builds true-price accounting
   (retiring the logged-price PnL approximation) and replay tooling — verify
   against `src/accounting.py` / `src/replay.py` before touching decision-queue
   items #3, #6, or #7 below; they may already be covered.
3. **Do not revert or rewrite it** just to get green — fix forward and keep the
   intended semantics documented in the tests.
4. `docs`-only commits from the opencode session (`f0a5b6d`, this file) are
   separate by design; leave them alone.

---

## 6. Incident log (read before restarting anything)

| When | Event | Lesson |
|---|---|---|
| Sep 3 | Permission error crash-loop (`models/` root-owned) → burst-limited | After changing ownership under `/opt`, always `chown -R financebot:financebot` |
| Sep 15 | Promoted to `growth_live` (gate passed honestly: 63 fills, positive) | Promotion is evidence-gated; the override is a deliberate exception |
| Oct 1 | LQD ejected −$170 by strategist DISCARD (rate thesis) | Strategist cadence works, but up to 7 days behind a fast regime |
| **Oct 14 07:49** | **Circuit breaker fired**: 24h PnL −$1,502 ≤ −$750 → `sys.exit(1)` | The safety belt works |
| Oct 14 07:49–51 | 5 automatic restarts each re-armed growth_live → re-tripped → **StartLimitBurst** | There is **no re-arm policy**; the unit parks in `failed` |
| Oct 14 → Oct 23 | **Bot silently down for 9 days** (revived externally) | **No down-alerting exists.** Positions sat unmanaged through the AI selloff (AMD −17% unhedged) |
| Oct 23 | Revival session: PSQ +$327.48 (best YTD trade), AMD −$235.34 (worst) | Inverse hedges pay in selloffs; a dead bot cannot defend |
| Oct 25 | Journald had rotated mid-October logs | Forensics survived only via `trades_log.csv`; **journald persistence not configured** |

---

## 7. Decision queue (in priority order)

1. **Down-alerting** — systemd `OnFailure=` hook + a dead-man's-switch check
   (no `[telemetry]` line for >30 min ⇒ alert). A 9-day outage must be
   impossible to miss.
2. **Breaker re-arm policy** — after a trip choose explicitly: stay down until
   a human clears it, or auto-step-down to `soak` sizing and resume. Today the
   override + restart policy makes it die silently.
3. **fills-log true-price accounting** — retire the logged-price PnL
   approximation. **Likely already built in the pi WIP** (`src/accounting.py`,
   `src/account_sync.py`, `audit_account.py` — see §5b): verify and wire it as
   the PnL source of truth instead of rebuilding.
4. **Journald persistence** — `Storage=persistent` + size cap.
5. **Size rollback** — audit verdict: `growth_live` + override exceeds the
   demonstrated edge. Recommended posture: `soak` ($5) or `small_live` ($1k)
   until expectancy is re-earned.
6. **Verdict-quality experiment** (do this before any strategy rewrite):
   compute forward returns of strategist KEEP vs DISCARD symbols from the
   42 cached verdicts. Positive spread ⇒ the LLM layer has signal; zero ⇒ the
   strategist is noise. The pi WIP replay tooling (`src/replay.py`,
   `research_replay.py`) may already supply the harness — check first.
7. **Sizing-notebook analysis** — 500+ `sizing_log.csv` rows: did
   conviction/vol sizing beat flat $5k, risk-adjusted? (May piggyback on the
   WIP `tests/test_portfolio_sharpe.py` work.)
8. **PSQH name-collision fix** — designed and live-verified (use the exact
   quoted fund name from Alpaca's asset `name` in the news query; bare tickers
   collide, e.g. PSQ vs PSQ Holdings). Not built yet.
9. **Dual-engine go-live path** — `build_feature_store.py` +
   `train_dual.py` exist but have **never run on this VPS**; the strategist
   currently uses deterministic fallbacks. This is the highest-upside unfinished
   workstream.

**Pre-registered kill criteria:** if net-of-cost expectancy is ≤ 0 over ≥100
additional fills at small size (6–8 weeks), declare the *price-model thesis*
dead and rebuild the signal (multiagent research engine feeding
`expected_return`), not the platform.

---

## 8. Operator rules

- **Check §5b before starting anything.** If the tree has another agent's WIP
  (it does, as of 2026-10-25), finish/commit that first; never build alongside
  it.
- **One operator at a time.** Do not let two agent sessions (opencode, pi,
  hermes) mutate the repo or the live box concurrently. Check for
  unexpected `ActiveEnterTimestamp` changes before assuming state.
- **Language:** Python 3.11+ per `AGENTS.md`. This is the documented exception
  to pi's global Rust-first default (ML/trading stack).
- Tests: `cd /root/financeBot && .venv/bin/python -m pytest -q` (~309 green,
  fully offline/mocked — any network access in tests is a bug).
- Never `systemctl restart financebot` without checking the startup log
  afterwards: a failed evidence gate exits immediately.
- `/opt/financebot/.git` is a stale artifact — treat `/opt` as deploy-only.

---

## 9. Useful commands

```bash
# status & health
systemctl status financebot --no-pager
systemctl list-timers 'financebot-*'
journalctl -u financebot -n 40 --output=cat
journalctl -u financebot -f | grep -E '\[telemetry\]|\[risk\]|\[strategist\]|\[gardener\]|\[universe\]'

# recent trades / verdicts / sizing
tail -20 /opt/financebot/trades_log.csv
cat /opt/financebot/models/strategist_verdicts.json
tail -20 /opt/financebot/models/sizing_log.csv

# manual research runs (as root with env sourced)
set -a && source /etc/financebot.env && set +a
cd /opt/financebot && .venv/bin/python curate_universe.py --dry-run
systemctl start financebot-sentiment.service   # regenerate today's mood card
systemctl start financebot-curate.service      # run weekly curation now
```
