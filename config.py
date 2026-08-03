"""
Central configuration & HARD-CODED guardrails.

These constants are the bot''s safety envelope. They are intentionally kept in
one place, in plain sight, and are treated as immutable at runtime. No signal,
model, or "AI logic" is permitted to override them (see src/guardrails.py which
enforces this).
"""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean from the environment (true/1/yes/on => True).

    Deployment knobs must be environment-driven so a VPS can flip paper/live
    without editing tracked source. The default is intentionally the SAFE value
    so a missing/typo'd env var can never silently arm live trading.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    """Read a lowercased enum-like value from the environment, else default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    return value if value in choices else default

# ---------------------------------------------------------------------------
# HARD GUARDRAILS -- do not make these dynamic. They exist to cap blast radius.
# ---------------------------------------------------------------------------

# Absolute maximum notional (USD) committed to a single trade. This is the
# single most important line in the repo: it caps the loss on any one order.
MAX_POSITION_SIZE: float = 5.00

# Daily circuit breaker. If realized+unrealized PnL over a rolling 24h window
# drops below this (negative) threshold, the bot calls sys.exit() and dies.
DAILY_LOSS_LIMIT: float = -10.00

# Hard cap on number of open positions at any time (diversification of risk).
MAX_OPEN_POSITIONS: int = 3

# ---------------------------------------------------------------------------
# RATE LIMITING -- keep us well under Alpaca''s limits and avoid IP bans.
# ---------------------------------------------------------------------------

# Minimum seconds to sleep between *any* two Alpaca API calls. The broker
# wrapper enforces this globally so no code path can spam the exchange.
API_CALL_DELAY_SECONDS: float = 1.0

# Seconds to sleep between each full pass of the execution loop.
LOOP_INTERVAL_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# TRADING UNIVERSE & DECISION THRESHOLDS
# ---------------------------------------------------------------------------

# Symbols to trade. Alpaca crypto uses the "BTC/USD" form; equities use "AAPL".
EQUITY_SYMBOLS: list[str] = ["AAPL", "MSFT"]
CRYPTO_SYMBOLS: list[str] = ["BTC/USD", "ETH/USD"]

# ---------------------------------------------------------------------------
# ACTIVE LIVE TARGET UNIVERSE (additive; research/backtest universes unchanged)
# ---------------------------------------------------------------------------
# These knobs drive src/universe.py, which resolves the *live execution* target
# list once at startup. They do NOT change EQUITY_SYMBOLS / CRYPTO_SYMBOLS /
# CORE_UNIVERSE, and research/train/backtest keep using their existing universes.
#
# active_universe.json is operator-local (gitignored). Without an active file or
# an explicit opt-in (FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE=true) the live loop
# preserves current behavior by using EQUITY_SYMBOLS + CRYPTO_SYMBOLS.
ACTIVE_UNIVERSE_PATH: str = "active_universe.json"

# An active_universe.json older than this many hours is considered stale.
UNIVERSE_MAX_AGE_HOURS: float = 96.0

# Hard ceiling on the number of *active target* symbols the live loop will trade.
# Held positions and hedges remain visible regardless of this cap.
MAX_LIVE_UNIVERSE_SIZE: int = 50

# Probability thresholds applied to the model output (class prob of "up").
# Deterministic: same prob in => same action out.
BUY_THRESHOLD: float = 0.58
SELL_THRESHOLD: float = 0.42

# ---------------------------------------------------------------------------
# DATA / MODEL PATHS
# ---------------------------------------------------------------------------

MODEL_PATH: str = "models/model.json"
FEATURE_META_PATH: str = "models/feature_meta.json"
STATE_PATH: str = "models/bot_state.json"
INVENTORY_STATE_PATH: str = "models/inventory_state.json"
DAILY_SENTIMENT_PATH: str = "daily_sentiment.json"
TRADES_LOG_PATH: str = "trades_log.csv"
HISTORICAL_SENTIMENT_PATH: str = "historical_sentiment.csv"
BACKTEST_RESULTS_PATH: str = "backtest_results.csv"

# FinRobot/LLM daily report filter. Trades are blocked below this score even if
# the XGBoost signal is strong, because this layer is a strict risk gate.
SENTIMENT_MIN_SCORE: float = 5.0

# ---------------------------------------------------------------------------
# ACTIVE PIVOT / DYNAMIC CORRELATION HEDGING
# ---------------------------------------------------------------------------
# When daily sentiment for a target asset is weak (< SENTIMENT_MIN_SCORE) and
# XGBoost still signals BUY, we do NOT block. Instead we pivot the capital into
# the inverse ETF that is most negatively correlated to the target over a
# trailing window -- an "active" hedge rather than sitting flat.
INVERSE_SAFE_LIST: list[str] = ["SH", "PSQ", "BITI", "SARK", "SETH", "RWM", "DOG"]

# Trailing window (days) used to compute target-vs-inverse return correlations.
HEDGE_CORR_LOOKBACK_DAYS: int = 30

# Bar timeframe used for both training and live features.
BAR_TIMEFRAME: str = "1Hour"

# How much history to pull when training.
TRAIN_LOOKBACK_DAYS: int = 720

# ---------------------------------------------------------------------------
# ENVIRONMENT (secrets live in Windows env vars -- NEVER hard-coded)
# ---------------------------------------------------------------------------

# Read via os.environ in src/broker.py:
#   APCA_API_KEY_ID
#   APCA_API_SECRET_KEY
# Paper endpoint by default; flip to live ONLY via environment, never by editing
# this file. Precedence: FINANCEBOT_PAPER overrides ALPACA_PAPER; if neither is
# set we default to paper=True (safe). To go live you must explicitly export
# FINANCEBOT_PAPER=false (or ALPACA_PAPER=false).
PAPER: bool = _env_flag("FINANCEBOT_PAPER", _env_flag("ALPACA_PAPER", True))

# Default engine for run_bot.py when --engine is not passed. Safe default is the
# legacy single-horizon engine until the dual engine clears paper soak testing.
ENGINE: str = _env_choice("FINANCEBOT_ENGINE", "legacy", ("legacy", "dual"))





# ===========================================================================
# DUAL-HORIZON ENGINE ADDITIONS (additive; legacy constants above unchanged)
# ===========================================================================
# The constants below wire the new interday Strategist / intraday Executor
# engine. They are purely additive: nothing above is renamed or removed, so all
# legacy import contracts keep working while the dual engine is validated.

# --- Universes --------------------------------------------------------------
# Core = everything we actively want exposure to. Hedge = inverse/defensive
# instruments the strategist may overlay. Tuples => immutable at runtime.
CORE_UNIVERSE: tuple[str, ...] = tuple(EQUITY_SYMBOLS + CRYPTO_SYMBOLS)
HEDGE_UNIVERSE: tuple[str, ...] = tuple(INVERSE_SAFE_LIST)

# --- New storage / artifact paths ------------------------------------------
FEATURE_STORE_PATH: str = "data/feature_store"
MODEL_REGISTRY_DIR: str = "models/registry"
ALLOCATION_MATRIX_PATH: str = "models/allocation_matrix.json"
ORDERS_LOG_PATH: str = "orders_log.csv"
FILLS_LOG_PATH: str = "fills_log.csv"
PORTFOLIO_STATE_PATH: str = "models/portfolio_state.json"
RISK_STATE_PATH: str = "models/risk_state.json"

# --- Freshness / lookback windows ------------------------------------------
# An allocation matrix older than this is stale and must be recomputed.
ALLOCATION_STALE_SECONDS: int = 26 * 3600
INTRADAY_LOOKBACK_MINUTES: int = 120
NEWS_TAU_DAYS: float = 7.0

# --- Covariance / hedge estimation -----------------------------------------
COVARIANCE_HALFLIFE_DAYS: int = 63
COVARIANCE_SHRINKAGE: float = 0.20
COVARIANCE_EIGEN_FLOOR: float = 1e-8
MIN_HEDGE_EFFECTIVENESS: float = 0.15
HEDGE_DECAY_PENALTY_BPS: float = 15.0

# --- Portfolio construction limits (soft, model-facing) --------------------
# These shape the strategist''s target matrix. They are NOT a substitute for the
# hard per-order MAX_POSITION_SIZE clamp, which is re-applied at order time.
MAX_GROSS_EXPOSURE: float = 1.00
MAX_NET_EXPOSURE: float = 0.60
MAX_SYMBOL_WEIGHT: float = 0.15
MAX_HEDGE_WEIGHT: float = 0.25
VOLATILITY_TARGET_ANNUAL: float = 0.12

# Mean-variance risk aversion and turnover smoothing for the core optimizer.
RISK_AVERSION: float = 8.0
TURNOVER_PENALTY_BPS: float = 10.0

# Smallest weight delta the intraday executor will bother to trade.
MINIMUM_TRADE_WEIGHT: float = 1e-4

# --- Risk state machine drawdown ladder ------------------------------------
RISK_WARN_DRAWDOWN: float = -0.03
RISK_REDUCE_DRAWDOWN: float = -0.05
RISK_LIQUIDATE_DRAWDOWN: float = -0.08
RISK_KILL_DRAWDOWN: float = -0.12
RISK_COOLDOWN_SECONDS: int = 3600
MAX_RECONCILIATION_QTY_DIFF: float = 1e-3
MAX_SPREAD_BPS: float = 25.0
MAX_DATA_STALENESS_SECONDS: int = 900
