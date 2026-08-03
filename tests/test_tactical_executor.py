from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from src.data import FeatureSnapshot
from src.fifo import FIFOInventory
from src.guardrails import RiskLimits, RiskState, RiskStateMachine
from src.portfolio_manager import PortfolioManager
from src.strategist import AllocationMatrix, TargetAllocation
from src.tactical_executor import (
    TacticalExecutor,
    convert_weight_delta_to_order_intent,
)
from tests.conftest import FakeBrokerPosition

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


class _FakeIntradayStore:
    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    def build_intraday_snapshot(self, *, symbols, as_of, lookback_minutes):
        rows = {}
        for sym in symbols:
            rows[sym] = {
                "last_price": self._prices.get(sym, 100.0),
                "spread_bps": 5.0,
                "vwap_distance": 0.0,
                "order_flow_imbalance": 0.0,
                "volume_percentile": 0.5,
            }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        return FeatureSnapshot(frame=frame, as_of=as_of, horizon="intraday", lineage={})


class _FakeStrategist:
    def __init__(self, matrix: AllocationMatrix, model_registry=None) -> None:
        self._matrix = matrix
        self.model_registry = model_registry

    def generate_allocation(self, *, as_of):
        return self._matrix


class _FailingRegistry:
    def verify(self, required_roles):
        raise RuntimeError(f"missing roles: {required_roles}")


def _alloc(rows: dict[str, TargetAllocation]) -> AllocationMatrix:
    return AllocationMatrix(
        as_of=NOW,
        rows=rows,
        covariance_version="v",
        feature_snapshot_id="s",
        regime="risk_on",
    )


def _target(symbol, w) -> TargetAllocation:
    band = max(abs(w) * 0.25, 0.005)
    return TargetAllocation(
        symbol=symbol,
        target_weight=w,
        min_weight=w - band,
        max_weight=w + band,
        direction_bias="long" if w > 0 else "flat",
        volatility_ceiling=0.12,
        rebalance_priority=abs(w),
    )


def _executor(tmp_path, broker, matrix, prices, *, model_registry=None):
    pm = PortfolioManager(
        broker=broker,
        fifo_inventory=FIFOInventory(),
        state_path=str(tmp_path / "pstate.json"),
        orders_log_path=str(tmp_path / "orders.csv"),
        fills_log_path=str(tmp_path / "fills.csv"),
    )
    risk = RiskStateMachine(
        limits=RiskLimits.from_config(), state_path=str(tmp_path / "risk.json")
    )
    return TacticalExecutor(
        strategist=_FakeStrategist(matrix, model_registry=model_registry),
        portfolio_manager=pm,
        feature_store=_FakeIntradayStore(prices),
        risk_machine=risk,
        broker=broker,
    ), pm, risk


def test_orphaned_hedge_in_universe_and_exits(tmp_path, mock_broker) -> None:
    # Allocation controls AAA only; broker still holds a stale HDG hedge.
    matrix = _alloc({"AAA": _target("AAA", 0.1)})
    mock_broker.equity = 1000.0
    mock_broker.set_positions(
        [FakeBrokerPosition(symbol="HDG", qty=1.0, market_value=50.0, avg_entry_price=50.0)]
    )
    executor, pm, _ = _executor(tmp_path, mock_broker, matrix, {"HDG": 50.0, "AAA": 100.0})

    universe = executor.build_processing_universe(matrix)
    assert "HDG" in universe  # orphaned hedge is always seen

    decision = executor.risk_machine.permissions_for_state(RiskState.NORMAL)
    intent = executor.process_symbol(
        symbol="HDG", allocation=matrix, risk_decision=decision, now=NOW
    )
    assert intent is not None
    assert intent.side == "sell"  # exits the orphaned hedge toward zero
    assert intent.reduce_only is True


def test_process_symbol_reapplies_max_position_size(tmp_path, mock_broker) -> None:
    # A greedy 0.5 target on $1000 equity would be $500; hard cap is $5.
    matrix = _alloc({"AAA": _target("AAA", 0.5)})
    mock_broker.equity = 1000.0
    mock_broker.set_positions([])
    executor, pm, _ = _executor(tmp_path, mock_broker, matrix, {"AAA": 100.0})

    decision = executor.risk_machine.permissions_for_state(RiskState.NORMAL)
    intent = executor.process_symbol(
        symbol="AAA", allocation=matrix, risk_decision=decision, now=NOW
    )
    assert intent is not None
    notional = intent.quantity * 100.0
    assert notional <= config.MAX_POSITION_SIZE + 1e-9


def test_freeze_blocks_new_entry(tmp_path, mock_broker) -> None:
    matrix = _alloc({"AAA": _target("AAA", 0.1)})
    mock_broker.equity = 1000.0
    mock_broker.set_positions([])
    executor, pm, _ = _executor(tmp_path, mock_broker, matrix, {"AAA": 100.0})

    decision = executor.risk_machine.permissions_for_state(RiskState.FREEZE_NEW_ENTRIES)
    intent = executor.process_symbol(
        symbol="AAA", allocation=matrix, risk_decision=decision, now=NOW
    )
    assert intent is None  # opening a new position is an exposure increase -> blocked


def test_convert_weight_delta_hard_caps_notional() -> None:
    class _Snap:
        equity = 1000.0

        def weight(self, symbol):
            return 0.0

    from src.tactical_executor import TacticalSignal

    signal = TacticalSignal(
        symbol="AAA", action="buy", urgency=1.0, confidence=1.0, limit_price=None, reason="x"
    )
    intent = convert_weight_delta_to_order_intent(
        symbol="AAA", delta_weight=0.9, snapshot=_Snap(), latest_price=50.0, signal=signal
    )
    assert intent is not None
    assert intent.quantity * 50.0 <= config.MAX_POSITION_SIZE + 1e-9


def test_kill_process_raises_system_exit(tmp_path, mock_broker) -> None:
    import pytest

    matrix = _alloc({"AAA": _target("AAA", 0.1)})
    mock_broker.equity = 1000.0
    mock_broker.set_positions([])
    executor, pm, risk = _executor(tmp_path, mock_broker, matrix, {"AAA": 100.0})

    # Force the risk machine into KILL via a catastrophic metric on evaluate.
    from src.guardrails import RiskMetrics

    def _kill_eval(metrics):
        return risk.permissions_for_state(RiskState.KILL_PROCESS)

    risk.evaluate = _kill_eval  # type: ignore[assignment]
    with pytest.raises(SystemExit):
        executor.run_once(now=NOW)
    assert mock_broker.cancel_all_called == 1


def test_build_risk_metrics_uses_high_watermark_drawdown(tmp_path, mock_broker) -> None:
    matrix = _alloc({"AAA": _target("AAA", 0.1)})
    executor, pm, risk = _executor(tmp_path, mock_broker, matrix, {"AAA": 100.0})
    risk.high_watermark_equity = 1000.0
    mock_broker.equity = 940.0

    metrics = executor.build_risk_metrics(pm.snapshot(), [], now=NOW)

    assert metrics.high_watermark_equity == 1000.0
    assert metrics.drawdown == pytest.approx(-0.06)
    assert metrics.rolling_24h_pnl == 0.0


def test_build_risk_metrics_surfaces_stale_data(tmp_path, mock_broker) -> None:
    matrix = _alloc({"AAA": _target("AAA", 0.1)})
    executor, pm, _ = _executor(tmp_path, mock_broker, matrix, {"AAA": 100.0})
    executor._allocation = AllocationMatrix(
        as_of=NOW - timedelta(seconds=config.MAX_DATA_STALENESS_SECONDS + 10),
        rows=matrix.rows,
        covariance_version="v",
        feature_snapshot_id="s",
        regime="risk_on",
    )

    metrics = executor.build_risk_metrics(pm.snapshot(), [], now=NOW)

    assert metrics.data_staleness_seconds >= config.MAX_DATA_STALENESS_SECONDS


def test_build_risk_metrics_checks_model_registry_and_broker_health(tmp_path, mock_broker) -> None:
    matrix = _alloc({"AAA": _target("AAA", 0.1)})
    mock_broker.broker_available = False
    executor, pm, _ = _executor(
        tmp_path,
        mock_broker,
        matrix,
        {"AAA": 100.0},
        model_registry=_FailingRegistry(),
    )

    metrics = executor.build_risk_metrics(pm.snapshot(), [], now=NOW)

    assert metrics.model_artifacts_valid is False
    assert metrics.broker_available is False


def test_run_once_reduces_risk_from_live_drawdown_metrics(tmp_path, mock_broker) -> None:
    matrix = _alloc({"AAA": _target("AAA", 0.0)})
    mock_broker.equity = 940.0
    mock_broker.set_positions(
        [FakeBrokerPosition(symbol="AAA", qty=1.0, market_value=100.0, avg_entry_price=100.0)]
    )
    executor, _, risk = _executor(tmp_path, mock_broker, matrix, {"AAA": 100.0})
    risk.high_watermark_equity = 1000.0
    executor.portfolio_manager.fifo.add_buy("AAA", 1.0, 100.0)

    submitted = executor.run_once(now=NOW)

    assert submitted and submitted[0].side == "sell"
    assert submitted[0].reduce_only is True
    assert risk.state == RiskState.REDUCE_RISK
