from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from backtest import replay_dual_horizon
from src.data import FeatureSnapshot
from src.fifo import FIFOInventory
from src.guardrails import RiskLimits, RiskStateMachine
from src.portfolio_manager import BrokerFill, PortfolioManager
from src.strategist import AllocationMatrix, TargetAllocation
from src.tactical_executor import TacticalExecutor
from tests.conftest import FakeBrokerPosition, MockBroker


class _FakeIntradayStore:
    def build_intraday_snapshot(self, *, symbols, as_of, lookback_minutes):
        rows = {
            sym: {
                "last_price": 100.0 if sym == "AAA" else 50.0,
                "spread_bps": 4.0,
                "vwap_distance": 0.0,
                "order_flow_imbalance": 0.0,
                "volume_percentile": 0.5,
            }
            for sym in symbols
        }
        return FeatureSnapshot(
            frame=pd.DataFrame.from_dict(rows, orient="index"),
            as_of=as_of,
            horizon="intraday",
            lineage={},
        )


class _FakeStrategist:
    def generate_allocation(self, *, as_of):
        rows = {
            "AAA": TargetAllocation(
                symbol="AAA",
                target_weight=0.1,
                min_weight=0.0,
                max_weight=0.15,
                direction_bias="long",
                volatility_ceiling=0.12,
                rebalance_priority=0.1,
            ),
            # HDG is no longer wanted -> target zero so the executor exits it.
            "HDG": TargetAllocation(
                symbol="HDG",
                target_weight=0.0,
                min_weight=0.0,
                max_weight=0.0,
                direction_bias="flat",
                volatility_ceiling=0.12,
                rebalance_priority=1.0,
            ),
        }
        return AllocationMatrix(
            as_of=as_of,
            rows=rows,
            covariance_version="v",
            feature_snapshot_id="s",
            regime="risk_on",
        )


class _PartialFillModel:
    """Fills half the requested quantity to exercise partial-fill handling."""

    def simulate_fill(self, *, intent, bar_price, now):
        return BrokerFill(
            fill_id=str(uuid.uuid4()),
            order_id="",
            symbol=intent.symbol,
            side=intent.side,
            filled_qty=float(intent.quantity) * 0.5,
            filled_price=float(bar_price),
            filled_at=now,
            fees=0.0,
            liquidity_flag=None,
        )


def test_dual_horizon_replay_end_to_end(tmp_path) -> None:
    # Broker reports a stale HDG position so the executor sees it as held.
    broker = MockBroker(
        equity=1000.0,
        positions=[FakeBrokerPosition(symbol="HDG", qty=1.0, market_value=50.0, avg_entry_price=50.0)],
    )
    fifo = FIFOInventory()
    fifo.add_buy("HDG", 1.0, 50.0)  # internal inventory matches broker for HDG

    pm = PortfolioManager(
        broker=broker,
        fifo_inventory=fifo,
        state_path=str(tmp_path / "pstate.json"),
        orders_log_path=str(tmp_path / "orders.csv"),
        fills_log_path=str(tmp_path / "fills.csv"),
    )
    risk = RiskStateMachine(
        limits=RiskLimits.from_config(), state_path=str(tmp_path / "risk.json")
    )
    executor = TacticalExecutor(
        strategist=_FakeStrategist(),
        portfolio_manager=pm,
        feature_store=_FakeIntradayStore(),
        risk_machine=risk,
        broker=broker,
    )

    base = datetime(2026, 6, 1, 13, 30, tzinfo=timezone.utc)
    bars = []
    for i in range(3):
        ts = base.replace(minute=30 + i)
        bars.append({"timestamp": ts, "symbol": "AAA", "close": 100.0})
        bars.append({"timestamp": ts, "symbol": "HDG", "close": 50.0})
    price_panel = pd.DataFrame(bars)

    fills = replay_dual_horizon(
        price_panel=price_panel,
        feature_store=_FakeIntradayStore(),
        strategist=_FakeStrategist(),
        tactical_executor=executor,
        portfolio_manager=pm,
        slippage_model=_PartialFillModel(),
    )

    assert not fills.empty
    # AAA was accumulated toward its target via (partial) buy fills.
    assert pm.fifo.open_qty("AAA") > 0.0
    # The orphaned HDG hedge was reduced by at least one sell fill.
    hdg_sells = fills[(fills["symbol"] == "HDG") & (fills["side"] == "sell")]
    assert len(hdg_sells) >= 1
    assert pm.fifo.open_qty("HDG") < 1.0
