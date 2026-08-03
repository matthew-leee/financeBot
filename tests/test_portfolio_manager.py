from __future__ import annotations

import pandas as pd

from src.fifo import FIFOInventory
from src.portfolio_manager import PortfolioManager


def _pm(tmp_path, broker):
    return PortfolioManager(
        broker=broker,
        fifo_inventory=FIFOInventory(),
        state_path=str(tmp_path / "portfolio_state.json"),
        orders_log_path=str(tmp_path / "orders_log.csv"),
        fills_log_path=str(tmp_path / "fills_log.csv"),
    )


def test_partial_fills_create_correct_fifo_lots(tmp_path, mock_broker, make_fill) -> None:
    mock_broker.set_fills(
        [
            make_fill(fill_id="f1", symbol="AAPL", side="buy", qty=0.5, price=100.0),
            make_fill(fill_id="f2", symbol="AAPL", side="buy", qty=0.5, price=102.0),
        ]
    )
    pm = _pm(tmp_path, mock_broker)
    applied = pm.poll_and_apply_fills()
    assert len(applied) == 2
    assert pm.fifo.open_qty("AAPL") == 1.0
    # Two distinct lots at their actual fill prices (weighted avg 101).
    assert abs(pm.fifo.average_price("AAPL") - 101.0) < 1e-9


def test_duplicate_fills_ignored_by_fill_id(tmp_path, mock_broker, make_fill) -> None:
    mock_broker.set_fills(
        [
            make_fill(fill_id="dup", symbol="AAPL", side="buy", qty=0.5, price=100.0),
            make_fill(fill_id="dup", symbol="AAPL", side="buy", qty=0.5, price=100.0),
        ]
    )
    pm = _pm(tmp_path, mock_broker)
    applied = pm.poll_and_apply_fills()
    assert len(applied) == 1
    assert pm.fifo.open_qty("AAPL") == 0.5

    # A second poll that replays the same fill must not double-count.
    again = pm.poll_and_apply_fills()
    assert again == []
    assert pm.fifo.open_qty("AAPL") == 0.5


def test_sell_fills_realize_fifo_pnl_net_of_fees(tmp_path, mock_broker, make_fill) -> None:
    pm = _pm(tmp_path, mock_broker)
    pm.apply_fill(make_fill(fill_id="b1", symbol="AAPL", side="buy", qty=1.0, price=100.0, fees=0.0))
    realized = pm.apply_fill(
        make_fill(fill_id="s1", symbol="AAPL", side="sell", qty=1.0, price=110.0, fees=1.0)
    )
    # (110 - 100) * 1 - 1.0 fee = 9.0
    assert abs(realized - 9.0) < 1e-9
    assert abs(pm.realized_pnl - 9.0) < 1e-9


def test_fill_uses_actual_price_not_stale_arrival(tmp_path, mock_broker, make_fill) -> None:
    pm = _pm(tmp_path, mock_broker)
    # Arrival/reference price recorded at order time.
    pm._order_ref_price["o1"] = 100.0
    pm.apply_fill(
        make_fill(fill_id="f1", order_id="o1", symbol="AAPL", side="buy", qty=1.0, price=101.0)
    )
    fills = pd.read_csv(tmp_path / "fills_log.csv")
    row = fills.iloc[0]
    assert row["actual_price"] == 101.0  # the true execution price, not 100.0
    assert row["arrival_price"] == 100.0
    assert abs(row["slippage_bps"] - 100.0) < 1e-6


def test_major_reconciliation_mismatch_flagged(tmp_path, mock_broker, make_fill) -> None:
    from tests.conftest import FakeBrokerPosition

    pm = _pm(tmp_path, mock_broker)
    pm.apply_fill(make_fill(fill_id="b1", symbol="AAPL", side="buy", qty=1.0, price=100.0))
    mock_broker.set_positions(
        [FakeBrokerPosition(symbol="AAPL", qty=5.0, market_value=500.0, avg_entry_price=100.0)]
    )
    diffs = pm.reconcile_with_broker()
    aapl = [d for d in diffs if d.symbol == "AAPL"][0]
    assert aapl.severity == "major"
    assert abs(aapl.qty_diff - 4.0) < 1e-9
    assert pm.major_break_count(diffs) == 1
