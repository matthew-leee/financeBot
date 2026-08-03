from __future__ import annotations

import pandas as pd

import config
import dashboard
from src.analytics import compute_metrics, equity_curve, load_trades
from src.trade_log import TRADE_LOG_COLUMNS


def test_load_trades_missing_file_is_safe(tmp_path) -> None:
    missing = tmp_path / "trades_log.csv"
    df = load_trades(str(missing))
    assert list(df.columns) == TRADE_LOG_COLUMNS
    assert df.empty


def test_load_trades_header_only_file_is_safe(tmp_path) -> None:
    path = tmp_path / "trades_log.csv"
    path.write_text(",".join(TRADE_LOG_COLUMNS) + "\n", encoding="utf-8")
    df = load_trades(str(path))
    assert df.empty
    assert list(df.columns) == TRADE_LOG_COLUMNS


def test_load_trades_completely_empty_file_is_safe(tmp_path) -> None:
    path = tmp_path / "trades_log.csv"
    path.write_text("", encoding="utf-8")
    df = load_trades(str(path))
    assert df.empty


def test_metrics_on_empty_trades_are_zeroed() -> None:
    empty = load_trades("")
    metrics = compute_metrics(empty)
    assert metrics == {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
    }
    assert equity_curve(empty).empty


def test_metrics_win_rate_and_drawdown_math() -> None:
    trades = pd.DataFrame(
        {
            "timestamp": ["t1", "t2", "t3", "t4"],
            "ticker": ["SPY"] * 4,
            "side": ["sell"] * 4,
            "price": [10, 11, 9, 12],
            "size": [1, 1, 1, 1],
            "pnl": [5.0, -2.0, 3.0, -1.0],
        }
    )
    metrics = compute_metrics(trades)
    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == 0.5  # 2 wins of 4 realized
    assert metrics["total_pnl"] == 5.0
    # cumulative: 5, 3, 6, 5 -> running peak 5,5,6,6 -> drawdowns 0,-2,0,-1
    assert metrics["max_drawdown"] == -2.0


def test_dashboard_render_is_safe_with_empty_log(monkeypatch, tmp_path) -> None:
    """dashboard.render() must not crash on an empty trades_log.csv."""
    st = _install_fake_streamlit(monkeypatch)
    _install_fake_plotly(monkeypatch)

    empty_log = tmp_path / "trades_log.csv"
    empty_log.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(empty_log))
    monkeypatch.setattr(config, "DAILY_SENTIMENT_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))

    dashboard.render()

    # It reached the "no trades" info path rather than raising.
    assert any("No trades" in msg or "empty" in msg for msg in st.info_messages)


def _install_fake_streamlit(monkeypatch):
    import sys
    import types

    class FakeColumn:
        def __init__(self, sink):
            self.sink = sink

        def metric(self, label, value, **kwargs):
            self.sink.metrics.append((label, value))

    class FakeStreamlit(types.ModuleType):
        def __init__(self):
            super().__init__("streamlit")
            self.info_messages: list[str] = []
            self.metrics: list = []

        def set_page_config(self, **kwargs):
            pass

        def title(self, *a, **k):
            pass

        def caption(self, *a, **k):
            pass

        def subheader(self, *a, **k):
            pass

        def header(self, *a, **k):
            pass

        def divider(self, *a, **k):
            pass

        def write(self, *a, **k):
            pass

        def columns(self, n):
            return [FakeColumn(self) for _ in range(n)]

        def metric(self, label, value, **kwargs):
            self.metrics.append((label, value))

        def info(self, msg):
            self.info_messages.append(str(msg))

        def warning(self, msg):
            self.info_messages.append(str(msg))

        def dataframe(self, *a, **k):
            pass

        def plotly_chart(self, *a, **k):
            pass

    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


def _install_fake_plotly(monkeypatch):
    import sys
    import types

    go = types.ModuleType("plotly.graph_objects")

    class FakeFigure:
        def add_trace(self, *a, **k):
            return self

        def update_layout(self, *a, **k):
            return self

    go.Figure = FakeFigure
    go.Scatter = lambda *a, **k: None

    plotly = types.ModuleType("plotly")
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go)
    return go




def test_open_inventory_summary_computes_unrealized_pnl() -> None:
    from src.analytics import INVENTORY_COLUMNS, open_inventory_summary
    from src.fifo import FIFOInventory

    inv = FIFOInventory()
    inv.add_buy("SPY", qty=2.0, price=100.0)
    inv.add_buy("SPY", qty=2.0, price=120.0)  # avg = 110
    inv.add_buy("AAPL", qty=1.0, price=50.0)

    prices = {"SPY": 130.0, "AAPL": 40.0}
    summary = open_inventory_summary(inv, price_lookup=lambda t: prices.get(t))

    assert list(summary.columns) == INVENTORY_COLUMNS
    spy = summary[summary["ticker"] == "SPY"].iloc[0]
    assert spy["open_qty"] == 4.0
    assert spy["avg_price"] == 110.0
    assert spy["last_price"] == 130.0
    assert spy["unrealized_pnl"] == 4.0 * (130.0 - 110.0)  # 80.0

    aapl = summary[summary["ticker"] == "AAPL"].iloc[0]
    assert aapl["unrealized_pnl"] == 1.0 * (40.0 - 50.0)  # -10.0


def test_open_inventory_summary_falls_back_to_avg_price_when_no_feed() -> None:
    from src.analytics import open_inventory_summary
    from src.fifo import FIFOInventory

    inv = FIFOInventory()
    inv.add_buy("SPY", qty=3.0, price=100.0)

    # price_lookup returns None -> fallback to avg price -> unrealized 0.
    summary = open_inventory_summary(inv, price_lookup=lambda t: None)
    row = summary.iloc[0]
    assert row["last_price"] == 100.0
    assert row["unrealized_pnl"] == 0.0


def test_open_inventory_summary_empty_is_safe() -> None:
    from src.analytics import INVENTORY_COLUMNS, open_inventory_summary
    from src.fifo import FIFOInventory

    summary = open_inventory_summary(FIFOInventory(), price_lookup=lambda t: 10.0)
    assert summary.empty
    assert list(summary.columns) == INVENTORY_COLUMNS


def test_fifo_positions_snapshot_and_average_price() -> None:
    from src.fifo import FIFOInventory

    inv = FIFOInventory()
    inv.add_buy("SPY", qty=2.0, price=100.0)
    inv.add_buy("SPY", qty=2.0, price=120.0)
    inv.add_sell("SPY", qty=1.0, price=130.0)  # consumes oldest lot partially

    positions = inv.positions()
    assert positions["SPY"]["open_qty"] == 3.0
    # remaining: 1 @ 100 + 2 @ 120 -> avg = (100 + 240) / 3
    assert positions["SPY"]["avg_price"] == round((100.0 + 240.0) / 3.0, 6)


def test_dashboard_render_shows_open_positions(monkeypatch, tmp_path) -> None:
    st = _install_fake_streamlit(monkeypatch)
    _install_fake_plotly(monkeypatch)

    # Persist a real inventory save-state and point config at it.
    from src.fifo import FIFOInventory

    inv = FIFOInventory()
    inv.add_buy("SPY", qty=2.0, price=100.0)
    inv_path = tmp_path / "inventory.json"
    inv.save(str(inv_path))

    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(inv_path))
    monkeypatch.setattr(config, "DAILY_SENTIMENT_PATH", str(tmp_path / "missing.json"))
    # Deterministic price feed -> no network.
    monkeypatch.setattr(dashboard, "_latest_price_lookup", lambda ticker: 110.0)

    dashboard.render()

    # An unrealized-PnL metric was rendered for the open position.
    labels = [label for label, _ in st.metrics]
    assert "Total Unrealized PnL" in labels


def test_classify_position_hedge_vs_direct() -> None:
    from src.analytics import classify_position

    # Inverse ETFs from the expanded safe list -> Hedge.
    for etf in ("SH", "PSQ", "BITI", "SARK", "SETH", "RWM", "DOG"):
        assert classify_position(etf) == "Hedge"

    # Normal target assets -> Direct Hold (crypto uses base asset).
    assert classify_position("AAPL") == "Direct Hold"
    assert classify_position("MSFT") == "Direct Hold"
    assert classify_position("BTC/USD") == "Direct Hold"


def test_open_inventory_summary_flags_position_type() -> None:
    from src.analytics import open_inventory_summary
    from src.fifo import FIFOInventory

    inv = FIFOInventory()
    inv.add_buy("AAPL", qty=1.0, price=100.0)
    inv.add_buy("PSQ", qty=2.0, price=20.0)  # a hedge

    summary = open_inventory_summary(inv, price_lookup=lambda t: None)
    assert "position_type" in summary.columns

    types = dict(zip(summary["ticker"], summary["position_type"]))
    assert types["AAPL"] == "Direct Hold"
    assert types["PSQ"] == "Hedge"


def test_dashboard_render_flags_hedge_position(monkeypatch, tmp_path) -> None:
    st = _install_fake_streamlit(monkeypatch)
    _install_fake_plotly(monkeypatch)

    from src.fifo import FIFOInventory

    inv = FIFOInventory()
    inv.add_buy("PSQ", qty=2.0, price=20.0)  # hedge-only inventory
    inv_path = tmp_path / "inventory.json"
    inv.save(str(inv_path))

    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(tmp_path / "trades_log.csv"))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(inv_path))
    monkeypatch.setattr(config, "DAILY_SENTIMENT_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(dashboard, "_latest_price_lookup", lambda ticker: 18.0)

    dashboard.render()

    labels = [label for label, _ in st.metrics]
    assert "Hedges" in labels
    assert "Direct Holds" in labels


def test_realized_pnl_breakdown_splits_hedge_vs_direct() -> None:
    from src.analytics import realized_pnl_breakdown

    trades = pd.DataFrame(
        {
            "timestamp": ["t1", "t2", "t3", "t4", "t5"],
            "ticker": ["AAPL", "AAPL", "PSQ", "PSQ", "MSFT"],
            "side": ["buy", "sell", "buy", "sell", "sell"],
            "price": [100, 110, 20, 25, 50],
            "size": [1, 1, 1, 1, 1],
            "pnl": [0.0, 10.0, 0.0, 5.0, -3.0],
        }
    )

    breakdown = realized_pnl_breakdown(trades)
    by_type = {row["position_type"]: row for _, row in breakdown.iterrows()}

    direct = by_type["Direct Hold"]  # AAPL + MSFT
    assert direct["realized_pnl"] == 7.0   # 10 + (-3)
    assert direct["trades"] == 3
    assert direct["wins"] == 1             # only the +10 realized row
    assert direct["win_rate"] == 0.5       # 1 win of 2 realized (AAPL sell, MSFT sell)

    hedge = by_type["Hedge"]  # PSQ
    assert hedge["realized_pnl"] == 5.0
    assert hedge["trades"] == 2
    assert hedge["wins"] == 1
    assert hedge["win_rate"] == 1.0


def test_realized_pnl_breakdown_empty_is_safe() -> None:
    from src.analytics import load_trades, realized_pnl_breakdown

    breakdown = realized_pnl_breakdown(load_trades(""))
    assert breakdown.empty
    assert list(breakdown.columns) == [
        "position_type",
        "realized_pnl",
        "trades",
        "wins",
        "win_rate",
    ]


def test_dashboard_render_shows_hedge_vs_direct_breakdown(monkeypatch, tmp_path) -> None:
    st = _install_fake_streamlit(monkeypatch)
    _install_fake_plotly(monkeypatch)

    log_path = tmp_path / "trades_log.csv"
    pd.DataFrame(
        {
            "timestamp": ["t1", "t2", "t3", "t4"],
            "ticker": ["AAPL", "AAPL", "PSQ", "PSQ"],
            "side": ["buy", "sell", "buy", "sell"],
            "price": [100, 110, 20, 25],
            "size": [1, 1, 1, 1],
            "pnl": [0.0, 10.0, 0.0, 5.0],
        }
    ).to_csv(log_path, index=False)

    monkeypatch.setattr(config, "TRADES_LOG_PATH", str(log_path))
    monkeypatch.setattr(config, "INVENTORY_STATE_PATH", str(tmp_path / "inventory.json"))
    monkeypatch.setattr(config, "DAILY_SENTIMENT_PATH", str(tmp_path / "missing.json"))

    dashboard.render()

    labels = [label for label, _ in st.metrics]
    assert "📈 Direct Realized PnL" in labels
    assert "🛡️ Hedge Realized PnL" in labels

