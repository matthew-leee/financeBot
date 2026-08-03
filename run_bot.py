"""
Live bot entrypoint.

Run:  python run_bot.py                 # engine from FINANCEBOT_ENGINE or legacy
      python run_bot.py --engine legacy # explicit legacy engine
      python run_bot.py --engine dual   # new dual-horizon engine (strategist +
                                         # tactical executor + risk state machine)

Requires a trained model (run `python train.py` first) and the Windows env vars
APCA_API_KEY_ID / APCA_API_SECRET_KEY.

The safe default remains the legacy engine unless FINANCEBOT_ENGINE=dual or
--engine dual is provided. Paper trading remains the default unless
FINANCEBOT_PAPER=false (or ALPACA_PAPER=false) is explicitly set.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

import config
from src.execution import run as run_legacy


def run_dual() -> None:
    """Boot the dual-horizon engine: strategist -> executor -> portfolio manager."""
    # Imported lazily so the legacy path never pays for the new engine''s deps.
    from src.broker import Broker
    from src.data import PointInTimeFeatureStore
    from src.guardrails import RiskLimits, RiskStateMachine
    from src.model_io import ModelRegistry
    from src.portfolio_manager import PortfolioManager
    from src.strategist import InterdayStrategist, StrategistConfig
    from src.tactical_executor import TacticalExecutor

    feature_store = PointInTimeFeatureStore(config.FEATURE_STORE_PATH)

    # Fail closed BEFORE constructing a broker or an order loop: the dual engine
    # has no production persisted feature loader yet, so an empty feature store
    # can only emit empty allocations. Refuse to run rather than trade on nothing.
    if feature_store.is_empty():
        print(
            "[startup] ABORT -- dual execution requires a populated persisted "
            "point-in-time feature store, but it is empty. Dual VPS execution is "
            "not deployment-ready; run with FINANCEBOT_ENGINE=legacy. A future "
            "change will add persisted dual feature loading."
        )
        raise SystemExit(1)

    broker = Broker()
    model_registry = ModelRegistry(config.MODEL_REGISTRY_DIR)
    strategist = InterdayStrategist(
        config=StrategistConfig.from_config(),
        feature_store=feature_store,
        model_registry=model_registry,
    )
    portfolio_manager = PortfolioManager(broker=broker)
    risk_machine = RiskStateMachine(limits=RiskLimits.from_config())
    executor = TacticalExecutor(
        strategist=strategist,
        portfolio_manager=portfolio_manager,
        feature_store=feature_store,
        risk_machine=risk_machine,
        broker=broker,
        loop_interval_seconds=config.LOOP_INTERVAL_SECONDS,
        allocation_refresh_interval=timedelta(seconds=config.ALLOCATION_STALE_SECONDS),
    )
    executor.run_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="financeBot live entrypoint")
    parser.add_argument(
        "--engine",
        choices=("legacy", "dual"),
        default=config.ENGINE,
        help="Which trading engine to run (default: FINANCEBOT_ENGINE or legacy).",
    )
    args = parser.parse_args()

    if args.engine == "dual":
        run_dual()
    else:
        run_legacy()


if __name__ == "__main__":
    main()
