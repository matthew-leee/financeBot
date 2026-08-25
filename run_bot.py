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

    # Hydrate the persisted point-in-time state produced by
    # build_feature_store.py / research ingestion. A missing or corrupt file
    # fails closed: the store stays empty and the gate below aborts.
    loader = getattr(feature_store, "load_from_disk", None)
    loaded_records = 0
    if callable(loader):
        try:
            loaded_records = int(loader() or 0)
        except Exception as exc:  # noqa: BLE001 -- never boot on a broken store
            print(f"[startup] Feature store load failed: {exc}")
            loaded_records = 0

    # Fail closed BEFORE constructing a broker or an order loop: an empty feature
    # store can only emit empty allocations. Refuse to run rather than trade on
    # nothing.
    if feature_store.is_empty():
        print(
            "[startup] ABORT -- dual execution requires a populated persisted "
            "point-in-time feature store, but it is empty. Run "
            "`python build_feature_store.py` first (then `python train_dual.py`), "
            "or run with FINANCEBOT_ENGINE=legacy."
        )
        raise SystemExit(1)
    print(f"[startup] Dual engine: {loaded_records} PIT records loaded.")

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


def _gate_legacy_profile() -> None:
    """
    Startup gate: promotion-gated risk profiles must show earned evidence.

    Resolves the immutable policy (also validating all env overrides) and, for
    evidence-gated profiles like growth_live, verifies the operator-local
    trade logs prove a qualifying track record BEFORE any broker is built.
    Fails closed with SystemExit(1) -- identical philosophy to the dual
    engine's empty-store gate.
    """
    from src.guardrails import (
        PromotionEvidenceError,
        resolve_risk_policy,
        verify_promotion_evidence,
    )

    policy = resolve_risk_policy()
    try:
        verify_promotion_evidence(policy.profile)
    except PromotionEvidenceError as exc:
        print(f"[startup] ABORT -- {exc}")
        raise SystemExit(1)


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
        _gate_legacy_profile()
        run_legacy()


if __name__ == "__main__":
    main()
