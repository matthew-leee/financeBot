"""
Hybrid strategy backtester.

Replays held-out bars through the SAME logic the live loop uses:
  XGBoost probability -> action -> STRICT historical sentiment gate -> trade.

The sentiment gate is enforced per-bar by merging the bar''s calendar date +
ticker against historical_sentiment.csv. Trades are only opened when both the
model says "buy" AND that day''s sentiment score clears SENTIMENT_MIN_SCORE.

Because Alpaca bars are trailing/dynamic, the historical sentiment mock is
generated to align EXACTLY with the dates present in the downloaded bars (see
generate_historical_sentiment). Static hardcoded dates would never intersect the
trailing window and would silently produce zero trades.

Run:  python backtest.py
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

import config
from src.analytics import compute_metrics
from src.data import FEATURE_COLUMNS, build_features


def _base_asset(symbol: str) -> str:
    """Sentiment is keyed by base asset: 'BTC/USD' -> 'BTC', 'AAPL' -> 'AAPL'."""
    return symbol.split("/")[0]


def generate_historical_sentiment(
    bars_by_symbol: dict[str, pd.DataFrame],
    path: str | None = None,
    seed: int | None = None,
    score_min: float = 1.0,
    score_max: float = 10.0,
) -> pd.DataFrame:
    """
    Build a date-aligned mock sentiment CSV from the downloaded bars.

    For each symbol we read the UNIQUE calendar dates actually present in its
    bars and emit one randomized score (score_min..score_max) per date. This
    guarantees the sentiment dates intersect the trailing bar window, so the
    gate can actually pass/block instead of matching nothing.

    Tickers are written as base assets (matching merge_sentiment''s keying) so
    'BTC/USD' bars map to a 'BTC' sentiment row.
    """
    path = path or config.HISTORICAL_SENTIMENT_PATH
    rng = random.Random(seed)  # seedable for deterministic tests

    rows: list[dict] = []
    for symbol, bars in bars_by_symbol.items():
        if bars is None or bars.empty:
            continue
        ticker = _base_asset(symbol)
        # Unique, sorted calendar dates from this symbol''s own bars.
        unique_dates = sorted({pd.Timestamp(ts).date() for ts in bars.index})
        for day in unique_dates:
            score = round(rng.uniform(score_min, score_max), 1)
            rows.append({"date": day.isoformat(), "ticker": ticker, "score": score})

    df = pd.DataFrame(rows, columns=["date", "ticker", "score"])
    df.to_csv(path, index=False)
    print(f"[backtest] Wrote {len(df)} aligned sentiment rows -> {path}")
    return df


def load_historical_sentiment(path: str | None = None) -> pd.DataFrame:
    """Load the historical sentiment CSV, normalized to date/ticker/score."""
    path = path or config.HISTORICAL_SENTIMENT_PATH
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def merge_sentiment(
    bars: pd.DataFrame,
    sentiment: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """
    Align a per-bar sentiment score series to the bars index (no look-ahead).

    Each bar is matched to the sentiment score for its OWN calendar date and the
    given ticker. Missing days -> NaN (which the gate treats as "blocked").
    Returned series is index-aligned 1:1 with `bars`, preventing misalignment.
    """
    key = _base_asset(ticker)
    subset = sentiment[sentiment["ticker"] == key]
    lookup = dict(zip(subset["date"], subset["score"]))

    bar_dates = pd.Series(bars.index, index=bars.index)
    # Normalize timestamps -> calendar date, then map to that day''s score.
    scores = bar_dates.apply(lambda ts: lookup.get(pd.Timestamp(ts).date(), np.nan))
    scores.name = "sentiment_score"
    return scores


def simulate(
    bars: pd.DataFrame,
    model,  # anything exposing predict_up_proba(DataFrame) -> float
    sentiment: pd.DataFrame,
    ticker: str,
    min_score: float = config.SENTIMENT_MIN_SCORE,
) -> pd.DataFrame:
    """
    Bar-by-bar long-only replay. Returns a trades DataFrame with realized PnL.

    Logic per bar:
      * features -> P(up) -> action via config thresholds
      * BUY only if flat AND sentiment(date) >= min_score
      * SELL closes the open long and realizes PnL
    """
    feats = build_features(bars).dropna()
    scores = merge_sentiment(bars, sentiment, ticker)

    position_qty = 0.0
    entry_price = 0.0
    trades: list[dict] = []

    for ts, row in feats.iterrows():
        price = float(bars.loc[ts, "close"])
        feature_row = row[FEATURE_COLUMNS].to_frame().T.astype("float64")
        prob_up = model.predict_up_proba(feature_row)

        score = scores.get(ts, np.nan)
        sentiment_ok = (not pd.isna(score)) and float(score) >= min_score

        if prob_up >= config.BUY_THRESHOLD and position_qty == 0.0 and sentiment_ok:
            qty = config.MAX_POSITION_SIZE / price
            position_qty = qty
            entry_price = price
            trades.append(
                {
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "side": "buy",
                    "price": round(price, 6),
                    "size": round(qty, 6),
                    "pnl": 0.0,
                }
            )
        elif prob_up <= config.SELL_THRESHOLD and position_qty > 0.0:
            pnl = (price - entry_price) * position_qty
            trades.append(
                {
                    "timestamp": str(ts),
                    "ticker": ticker,
                    "side": "sell",
                    "price": round(price, 6),
                    "size": round(position_qty, 6),
                    "pnl": round(pnl, 6),
                }
            )
            position_qty = 0.0
            entry_price = 0.0

    return pd.DataFrame(trades, columns=["timestamp", "ticker", "side", "price", "size", "pnl"])


def parse_args() -> argparse.Namespace:
    """CLI flags for repeatable backtests."""
    parser = argparse.ArgumentParser(description="Run the hybrid XGBoost + sentiment backtest.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic mock sentiment scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    from src.data import fetch_bars
    from src.model_io import ModelArtifactError, load_model, verify_model_artifacts

    # Fail fast on missing/empty/placeholder model artifacts.
    try:
        verify_model_artifacts()
    except ModelArtifactError as exc:
        print(f"[backtest] ABORT -- {exc}")
        sys.exit(1)

    model = load_model()

    symbols = config.EQUITY_SYMBOLS + config.CRYPTO_SYMBOLS

    # 1) Download bars ONCE and reuse them for both alignment and simulation.
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bars = fetch_bars(symbol, lookback_days=config.TRAIN_LOOKBACK_DAYS)
        if bars.empty or len(bars) < 60:
            print(f"[backtest] {symbol}: insufficient data, skipping.")
            continue
        bars_by_symbol[symbol] = bars

    # 2) Regenerate the sentiment mock aligned to the ACTUAL bar dates.
    generate_historical_sentiment(bars_by_symbol, seed=args.seed)
    sentiment = load_historical_sentiment()

    # 3) Replay each symbol through the hybrid logic.
    all_trades: list[pd.DataFrame] = []
    for symbol, bars in bars_by_symbol.items():
        trades = simulate(bars, model, sentiment, symbol)
        all_trades.append(trades)

    if all_trades:
        results = pd.concat(all_trades, ignore_index=True)
    else:
        results = pd.DataFrame(columns=["timestamp", "ticker", "side", "price", "size", "pnl"])

    results.to_csv(config.BACKTEST_RESULTS_PATH, index=False)
    metrics = compute_metrics(results)

    print("==== Hybrid Backtest Results ====")
    print(f"Total Trades : {metrics['total_trades']}")
    print(f"Total PnL    : {metrics['total_pnl']:.4f}")
    print(f"Win Rate     : {metrics['win_rate'] * 100:.2f}%")
    print(f"Max Drawdown : {metrics['max_drawdown']:.4f}")
    print(f"Sharpe       : {metrics['sharpe']:.4f}")
    print(f"Saved -> {config.BACKTEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()





# ===========================================================================
# ADDITIVE: Dual-Horizon Event-Driven Replay
# ===========================================================================
# Legacy simulate() above remains the validation path for the legacy engine. The
# replay below mirrors the LIVE dual-horizon flow (strategist -> executor ->
# portfolio manager -> fills) so hedge exits, partial fills, fees, and slippage
# are validated exactly as they would occur in production.

def replay_dual_horizon(
    *,
    price_panel: pd.DataFrame,
    feature_store: object,
    strategist: object,
    tactical_executor: object,
    portfolio_manager: object,
    slippage_model: object | None = None,
) -> pd.DataFrame:
    """
    Event-driven backtest mirroring the live dual-horizon flow.

    Expects `price_panel` with columns [timestamp, symbol, close] (intraday bars
    within trading days). For each trading day the strategist emits an allocation
    at the day open; for each intraday bar the tactical executor proposes order
    intents which are filled via the injected slippage model and applied to the
    portfolio manager. Returns the fills log as a DataFrame.

    This is the validation path for the Active Pivot replacement, hedge exits,
    fills, fees, and slippage. Legacy `simulate()` remains for old tests.
    """
    from src.portfolio_manager import FILLS_LOG_COLUMNS

    if price_panel.empty:
        return pd.DataFrame(columns=FILLS_LOG_COLUMNS)

    panel = price_panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.sort_values("timestamp")
    panel["trading_day"] = panel["timestamp"].dt.date

    slippage_model = slippage_model or _ZeroSlippageModel()

    for _day, day_bars in panel.groupby("trading_day", sort=True):
        day_open = day_bars["timestamp"].min().to_pydatetime()
        allocation = strategist.generate_allocation(as_of=day_open)
        tactical_executor._allocation = allocation  # skip refresh churn in replay

        for bar_time, bar_rows in day_bars.groupby("timestamp", sort=True):
            now = bar_time.to_pydatetime()
            portfolio_manager.poll_and_apply_fills()
            portfolio_manager.reconcile_with_broker()

            symbols = tactical_executor.build_processing_universe(allocation)
            bar_prices = dict(zip(bar_rows["symbol"], bar_rows["close"]))

            for symbol in sorted(symbols):
                intent = tactical_executor.process_symbol(
                    symbol=symbol,
                    allocation=allocation,
                    risk_decision=tactical_executor.risk_machine.permissions_for_state(
                        tactical_executor.risk_machine.state
                    ),
                    now=now,
                )
                if intent is None:
                    continue
                price = bar_prices.get(symbol)
                if price is None:
                    continue
                fill = slippage_model.simulate_fill(intent=intent, bar_price=float(price), now=now)
                portfolio_manager.apply_fill(fill)

    fills_path = getattr(portfolio_manager, "fills_log_path", config.FILLS_LOG_PATH)
    if fills_path and __import__("os").path.exists(fills_path):
        return pd.read_csv(fills_path)
    return pd.DataFrame(columns=FILLS_LOG_COLUMNS)


class _ZeroSlippageModel:
    """Default fill model: fills at the bar price with zero fees/slippage."""

    def simulate_fill(self, *, intent: object, bar_price: float, now):
        import uuid

        from src.portfolio_manager import BrokerFill

        return BrokerFill(
            fill_id=str(uuid.uuid4()),
            order_id="",
            symbol=intent.symbol,
            side=intent.side,
            filled_qty=float(intent.quantity),
            filled_price=float(bar_price),
            filled_at=now,
            fees=0.0,
            liquidity_flag=None,
        )
