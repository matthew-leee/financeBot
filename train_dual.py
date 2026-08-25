"""
Training entrypoint for the DUAL-HORIZON (strategic) engine.

Trains the role models the InterdayStrategist consumes via the ModelRegistry:

    expected_return  -- per-symbol forward-return regression on interday
                        point-in-time features (macro + momentum + news state)

Pipeline:
    load persisted PIT store ──► historical interday snapshots (as_of grid)
        ──► leakage-safe panel (timestamp, symbol, features, label_end_time)
        ──► forward-return labels ──► walk-forward panel validation
        ──► fit final regressor ──► save_registry_artifact("expected_return")

Prerequisite: python build_feature_store.py must have populated the store first.

Run:  python train_dual.py [--date-step 3] [--horizon 5]
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta

import pandas as pd

import config


def _date_grid(store, *, end, lookback_days: int, step_days: int) -> list[pd.Timestamp]:
    """Business-day as_of grid derived from actual stored close history."""
    panel = store.close_panel(
        universe=list(config.CORE_UNIVERSE) + list(config.HEDGE_UNIVERSE),
        end=end,
        lookback_days=lookback_days,
    )
    if panel.empty:
        return []
    dates = pd.DatetimeIndex(sorted(panel["timestamp"].unique()))
    return list(dates[:-1][:: max(int(step_days), 1)])


def build_panel(
    store,
    *,
    lookback_days: int = 400,
    date_step_days: int = 3,
    horizon: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble the leakage-safe training panel.

    Rows are built ONLY from information available at each decision timestamp
    (build_interday_snapshot enforces available_at <= as_of). Labels are joined
    afterwards from the forward-return builder and carry label_end_time so the
    fold builder can embargo them. Returns (panel, feature_columns).
    """
    from src.data import _to_utc  # noqa: PLC2701 -- single-source UTC helper
    from src.validation import build_forward_return_labels

    end = _to_utc(pd.Timestamp.now().to_pydatetime())
    universe = list(config.CORE_UNIVERSE) + [
        s for s in config.HEDGE_UNIVERSE if s not in config.CORE_UNIVERSE
    ]

    grid = _date_grid(
        store, end=end, lookback_days=lookback_days, step_days=date_step_days
    )
    if len(grid) < 10:
        raise RuntimeError(
            "Not enough stored history to build a training panel. "
            "Run `python build_feature_store.py` first and retrain."
        )

    frames: list[pd.DataFrame] = []
    for i, ts in enumerate(grid):
        as_of = ts.to_pydatetime()
        snap = store.build_interday_snapshot(
            universe=universe,
            as_of=as_of,
            news_tau=timedelta(days=float(config.NEWS_TAU_DAYS)),
        )
        frame = snap.frame.copy()
        frame["timestamp"] = pd.Timestamp(as_of)
        frame["symbol"] = frame.index
        frames.append(frame)
        if (i + 1) % 25 == 0:
            print(f"[train_dual] snapshots {i + 1}/{len(grid)}")

    features = pd.concat(frames, ignore_index=True)
    # Merge-safe dtypes: build_forward_return_labels emits numpy-datetime
    # columns whose tz/unit can differ across pandas versions; normalize BOTH
    # sides to one identical timezone-aware dtype so the join can never fail.
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    feature_cols = [c for c in features.columns if c not in ("timestamp", "symbol")]

    prices = store.close_panel(universe=universe, end=end, lookback_days=lookback_days)
    labels = build_forward_return_labels(
        prices, horizons=(int(horizon),), benchmark_symbol=None
    )
    target_col = f"fwd_return_{int(horizon)}"
    label_end_col = f"label_end_time_{int(horizon)}"

    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    labels[label_end_col] = pd.to_datetime(labels[label_end_col], utc=True)
    # Pin both merge keys and the comparison column to one exact dtype.
    unified = labels["timestamp"].dtype
    features["timestamp"] = features["timestamp"].astype(unified)
    labels[label_end_col] = labels[label_end_col].astype(unified)

    panel = features.merge(
        labels[["symbol", "timestamp", target_col, label_end_col]],
        on=["symbol", "timestamp"],
        how="inner",
    )
    panel = panel.dropna(subset=[target_col]).reset_index(drop=True)
    if panel.empty:
        raise RuntimeError(
            "Feature/label join produced zero rows -- stored history is too "
            "short for the chosen horizon."
        )
    return panel, feature_cols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lookback-days", type=int, default=400)
    parser.add_argument("--date-step", type=int, default=3,
                        help="Days between decision timestamps (runtime knob).")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in days for the label.")
    parser.add_argument("--n-splits", type=int, default=4)
    args = parser.parse_args()

    from src.data import PointInTimeFeatureStore
    from src.model_io import save_registry_artifact
    from src.validation import walk_forward_validate_panel

    store = PointInTimeFeatureStore(config.FEATURE_STORE_PATH)
    loaded = store.load_from_disk()
    if loaded == 0:
        print(
            "[train_dual] ABORT -- persisted point-in-time store is empty. "
            "Run `python build_feature_store.py` first."
        )
        return 1
    print(f"[train_dual] Loaded {loaded} PIT records from disk.")

    panel, feature_cols = build_panel(
        store,
        lookback_days=args.lookback_days,
        date_step_days=args.date_step,
        horizon=args.horizon,
    )
    target_col = f"fwd_return_{args.horizon}"
    label_end_col = f"label_end_time_{args.horizon}"
    print(f"[train_dual] Panel: {len(panel)} rows x {len(feature_cols)} features.")

    def model_factory():
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

    report = walk_forward_validate_panel(
        panel=panel,
        feature_cols=feature_cols,
        target_col=target_col,
        timestamp_col="timestamp",
        symbol_col="symbol",
        label_end_col=label_end_col,
        model_factory=model_factory,
        n_splits=args.n_splits,
        min_train_period=timedelta(days=90),
        test_period=timedelta(days=max(args.lookback_days // 8, 20)),
        embargo=timedelta(days=args.horizon * 2),
        transaction_cost_bps=float(config.TURNOVER_PENALTY_BPS),
    )

    print("[train_dual] Leakage checks:", report.leakage_checks)
    print("[train_dual] Aggregate:", {
        k: round(v, 6) if isinstance(v, float) else v
        for k, v in report.aggregate.items()
    })

    final = model_factory()
    final.fit(panel[feature_cols], panel[target_col])

    path = save_registry_artifact(
        "expected_return",
        final,
        feature_columns=feature_cols,
        horizon="interday",
        validation={"panel": report.aggregate, "leakage": report.leakage_checks},
    )
    print(f"[train_dual] Registered expected_return artifact -> {path}")
    print("[train_dual] Done. The strategist will now use it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
