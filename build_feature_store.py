"""
Offline research entrypoint: populate the persisted point-in-time feature store.

This is the production path that unblocks the dual-horizon engine. Run this
BEFORE `run_bot.py --engine dual` / `python train_dual.py`:

    python build_feature_store.py                 # default 400d lookback
    python build_feature_store.py --lookback-days 720

What it does:
    Alpaca daily bars (via the legacy fetch path) ─┐
    FRED rates/CPI/fed funds (keyless CSV)        ─┼─► data/feature_store/
    BLS series (only if configured)               ─ │       records.jsonl
    SEC EDGAR filings per equity ticker           ─┘

Fail-closed rules:
  * Missing Alpaca credentials abort before any partial state is written.
  * Provider failures are reported but never crash the build; the summary tells
    you exactly which sources succeeded.
"""

from __future__ import annotations

import argparse
import os
import sys

import config
from src.data import PointInTimeFeatureStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=400,
        help="How many days of history to pull from each provider.",
    )
    parser.add_argument(
        "--skip-edgar",
        action="store_true",
        help="Skip SEC EDGAR filing ingestion.",
    )
    args = parser.parse_args()

    if not os.environ.get("APCA_API_KEY_ID") or not os.environ.get("APCA_API_SECRET_KEY"):
        print(
            "[build] ABORT -- APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set in "
            "the environment. Set them first; keys are never hardcoded."
        )
        return 1

    # Lazy import so credential validation happens first.
    from src.macro import populate_feature_store

    universe = list(config.CORE_UNIVERSE) + [
        s for s in config.HEDGE_UNIVERSE if s not in config.CORE_UNIVERSE
    ]

    store = PointInTimeFeatureStore(config.FEATURE_STORE_PATH)
    print(
        f"[build] Populating feature store at {config.FEATURE_STORE_PATH} "
        f"for {len(universe)} symbols, lookback {args.lookback_days}d ..."
    )

    summary = populate_feature_store(
        store,
        symbols=universe,
        lookback_days=args.lookback_days,
        include_edgar=not args.skip_edgar,
    )

    print(f"[build] Records accepted: {summary['records']}")
    for source, count in sorted(summary["by_source"].items()):
        print(f"[build]   {source}: {count}")
    print(f"[build] Persisted to disk: {summary['persisted']}")
    if summary["errors"]:
        print(f"[build] {len(summary['errors'])} non-fatal provider issues:")
        for err in summary["errors"][:20]:
            print(f"[build]   - {err}")
        if len(summary["errors"]) > 20:
            print(f"[build]   ... and {len(summary['errors']) - 20} more")

    if store.is_empty():
        print(
            "[build] WARNING -- store is still EMPTY. The dual engine will "
            "refuse to start until at least one source succeeds."
        )
        return 1
    print("[build] Done. Next: python train_dual.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
