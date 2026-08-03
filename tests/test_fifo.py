from __future__ import annotations

import pandas as pd

from src.fifo import FIFOInventory
from src.trade_log import append_trade


def test_fifo_partial_fill_leaves_remainder() -> None:
    inventory = FIFOInventory()
    inventory.add_buy("SPY", qty=10.0, price=100.0)

    pnl = inventory.add_sell("SPY", qty=4.0, price=110.0)

    assert pnl == 40.0
    assert inventory.open_qty("SPY") == 6.0


def test_fifo_full_exit_consumes_multiple_lots() -> None:
    inventory = FIFOInventory()
    inventory.add_buy("SPY", qty=5.0, price=100.0)
    inventory.add_buy("SPY", qty=5.0, price=120.0)

    pnl = inventory.add_sell("SPY", qty=10.0, price=130.0)

    # FIFO: 5*(130-100) + 5*(130-120) = 150 + 50
    assert pnl == 200.0
    assert inventory.open_qty("SPY") == 0.0


def test_fifo_partial_across_two_lots() -> None:
    inventory = FIFOInventory()
    inventory.add_buy("SPY", qty=5.0, price=100.0)
    inventory.add_buy("SPY", qty=5.0, price=120.0)

    pnl = inventory.add_sell("SPY", qty=7.0, price=130.0)

    # FIFO: 5*(130-100) + 2*(130-120) = 150 + 20
    assert pnl == 170.0
    assert inventory.open_qty("SPY") == 3.0

    # The remaining 3 shares are from the second lot at 120.
    pnl2 = inventory.add_sell("SPY", qty=3.0, price=125.0)
    assert pnl2 == 15.0
    assert inventory.open_qty("SPY") == 0.0


def test_fifo_empty_queue_sell_is_safe_and_zero_pnl() -> None:
    inventory = FIFOInventory()

    pnl = inventory.add_sell("SPY", qty=10.0, price=110.0)

    assert pnl == 0.0
    assert inventory.open_qty("SPY") == 0.0


def test_fifo_oversized_sell_only_realizes_existing_inventory() -> None:
    inventory = FIFOInventory()
    inventory.add_buy("SPY", qty=2.0, price=100.0)

    pnl = inventory.add_sell("SPY", qty=5.0, price=110.0)

    # Only existing 2 shares are matched; surplus is not treated as short.
    assert pnl == 20.0
    assert inventory.open_qty("SPY") == 0.0


def test_fifo_state_persists_round_trip(tmp_path) -> None:
    path = tmp_path / "inventory.json"
    inventory = FIFOInventory()
    inventory.add_buy("SPY", qty=1.5, price=100.0)
    inventory.add_buy("SPY", qty=2.5, price=120.0)
    inventory.save(str(path))

    reloaded = FIFOInventory.load(str(path))

    assert reloaded.open_qty("SPY") == 4.0
    assert reloaded.add_sell("SPY", qty=2.0, price=130.0) == 50.0


def test_append_trade_records_fifo_realized_pnl(tmp_path) -> None:
    log_path = tmp_path / "trades_log.csv"
    inventory_path = tmp_path / "inventory.json"

    append_trade("SPY", "buy", price=100.0, size=5.0, path=str(log_path), inventory_path=str(inventory_path))
    append_trade("SPY", "buy", price=120.0, size=5.0, path=str(log_path), inventory_path=str(inventory_path))
    pnl = append_trade("SPY", "sell", price=130.0, size=7.0, path=str(log_path), inventory_path=str(inventory_path))

    assert pnl == 170.0

    logged = pd.read_csv(log_path)
    assert logged.iloc[0]["pnl"] == 0.0
    assert logged.iloc[1]["pnl"] == 0.0
    assert logged.iloc[2]["pnl"] == 170.0

    inventory = FIFOInventory.load(str(inventory_path))
    assert inventory.open_qty("SPY") == 3.0


def test_append_trade_ignores_caller_supplied_pnl_for_sell(tmp_path) -> None:
    log_path = tmp_path / "trades_log.csv"
    inventory_path = tmp_path / "inventory.json"

    append_trade("SPY", "buy", price=10.0, size=1.0, path=str(log_path), inventory_path=str(inventory_path))
    append_trade(
        "SPY",
        "sell",
        price=15.0,
        size=1.0,
        pnl=999.0,
        path=str(log_path),
        inventory_path=str(inventory_path),
    )

    logged = pd.read_csv(log_path)
    assert logged.iloc[-1]["pnl"] == 5.0
