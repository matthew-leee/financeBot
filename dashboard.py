"""
Decoupled, READ-ONLY Streamlit dashboard.

This process only ever *reads* trades_log.csv, models/inventory_state.json, and
daily_sentiment.json. It never writes state and never routes orders, so it cannot
block or interfere with the live execution loop. Run it separately:

    streamlit run dashboard.py

Heavy UI imports (streamlit, plotly) are done lazily inside render() so the data
helpers here remain importable in tests without those packages installed.

Pages / sections (top to bottom):
  1. Headline metric cards (Total Trades, Win Rate, Total PnL, Max Drawdown)
  2. Open Positions -- live FIFO inventory + unrealized PnL (separate section)
  3. Realized PnL / Past Trades -- equity curve, hedge-vs-direct breakdown,
     and recent executions table
  4. FinRobot Daily Sentiment summary
"""

from __future__ import annotations

import config
from src.analytics import (
    compute_metrics,
    equity_curve,
    load_inventory,
    load_trades,
    open_inventory_summary,
    realized_pnl_breakdown,
)
from src.sentiment import load_sentiment


def _latest_price_lookup(ticker: str) -> float | None:
    """
    Best-effort current price for unrealized PnL.

    Tries the latest Alpaca bar; on ANY failure (no creds, network, rate limit)
    returns None so the analytics layer falls back to average entry price. The
    read-only dashboard must never crash because a price feed hiccuped.
    """
    try:
        from src.data import fetch_bars

        # Inventory is keyed by the traded symbol; crypto keeps its "BTC/USD" form.
        bars = fetch_bars(ticker, lookback_days=2)
        if bars is None or bars.empty:
            return None
        return float(bars["close"].iloc[-1])
    except Exception:  # noqa: BLE001 -- price feed is optional, never fatal
        return None


def render() -> None:
    """Render the full dashboard. Imports UI libs lazily."""
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="financeBot Dashboard", layout="wide")
    st.title("financeBot -- Hybrid Strategy Monitor")
    st.caption("Read-only viewer. Does not interact with the live execution loop.")

    trades = load_trades(config.TRADES_LOG_PATH)
    metrics = compute_metrics(trades)

    # -- Metric cards --------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", metrics["total_trades"])
    col2.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
    col3.metric("Realized PnL", f"{metrics['total_pnl']:.2f}")
    col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}")

    # -- Open Positions (live FIFO inventory) -------------------------------
    st.header("Open Positions")
    st.caption("Live holdings from models/inventory_state.json (FIFO cost basis).")
    inventory = load_inventory(config.INVENTORY_STATE_PATH)
    positions = open_inventory_summary(inventory, price_lookup=_latest_price_lookup)
    if positions.empty:
        st.info("No open positions. Inventory is flat.")
    else:
        total_unrealized = float(positions["unrealized_pnl"].sum())
        hedge_count = int((positions["position_type"] == "Hedge").sum())
        direct_count = int((positions["position_type"] == "Direct Hold").sum())

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Unrealized PnL", f"{total_unrealized:.2f}")
        mcol2.metric("Direct Holds", direct_count)
        mcol3.metric("Hedges", hedge_count)

        # Prefix the type with a badge glyph so it is scannable at a glance.
        display = positions.copy()
        badges = {"Hedge": "🛡️ Hedge", "Direct Hold": "📈 Direct Hold"}
        display["position_type"] = display["position_type"].map(
            lambda t: badges.get(t, t)
        )
        st.dataframe(
            display.rename(
                columns={
                    "ticker": "Ticker",
                    "position_type": "Type",
                    "open_qty": "Open Inventory",
                    "avg_price": "Avg Entry Price",
                    "last_price": "Last Price",
                    "unrealized_pnl": "Unrealized PnL",
                }
            ),
            use_container_width=True,
        )

    st.divider()

    # -- Realized PnL / Past Trades (clearly separated) ---------------------
    st.header("Realized PnL / Past Trades")

    st.subheader("Cumulative Realized PnL")
    curve = equity_curve(trades)
    if curve.empty:
        st.info("No trades logged yet. The equity curve will appear here.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["timestamp"],
                y=curve["cumulative_pnl"],
                mode="lines+markers",
                name="Cumulative PnL",
            )
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Cumulative Realized PnL")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hedge vs Direct Performance")
    breakdown = realized_pnl_breakdown(trades)
    if breakdown.empty:
        st.info("No realized PnL yet to break down.")
    else:
        by_type = {row["position_type"]: row for _, row in breakdown.iterrows()}
        direct = by_type.get("Direct Hold")
        hedge = by_type.get("Hedge")

        bcol1, bcol2 = st.columns(2)
        bcol1.metric(
            "📈 Direct Realized PnL",
            f"{(direct['realized_pnl'] if direct is not None else 0.0):.2f}",
            help="Realized PnL from primary target assets.",
        )
        bcol2.metric(
            "🛡️ Hedge Realized PnL",
            f"{(hedge['realized_pnl'] if hedge is not None else 0.0):.2f}",
            help="Realized PnL from Active Pivot inverse-ETF hedges.",
        )

        st.dataframe(
            breakdown.rename(
                columns={
                    "position_type": "Type",
                    "realized_pnl": "Realized PnL",
                    "trades": "Trades",
                    "wins": "Wins",
                    "win_rate": "Win Rate",
                }
            ),
            use_container_width=True,
        )

    st.subheader("Recent Executions")
    if trades.empty:
        st.info("trades_log.csv is empty.")
    else:
        st.dataframe(trades.tail(25), use_container_width=True)

    # -- Daily sentiment summary --------------------------------------------
    st.header("FinRobot Daily Sentiment")
    report = load_sentiment(config.DAILY_SENTIMENT_PATH)
    if not report:
        st.warning("No daily sentiment report found.")
    else:
        for symbol, entry in report.items():
            if isinstance(entry, dict):
                score = entry.get("score")
                summary = entry.get("summary", "")
            else:
                score, summary = entry, ""
            passed = score is not None and float(score) >= config.SENTIMENT_MIN_SCORE
            badge = "PASS" if passed else "BLOCK"
            st.write(f"**{symbol}** -- score {score} [{badge}]")
            if summary:
                st.caption(summary)


if __name__ == "__main__":
    render()


