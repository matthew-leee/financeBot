from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.validation import (
    build_forward_return_labels,
    make_temporal_group_folds,
    walk_forward_validate_panel,
)


def _fold_kwargs():
    return dict(
        timestamp_col="timestamp",
        symbol_col="symbol",
        label_end_col="label_end_time",
        n_splits=4,
        min_train_period=timedelta(days=20),
        test_period=timedelta(days=8),
        embargo=timedelta(days=5),
    )


def test_shuffled_panel_produces_deterministic_folds(shuffled_panel) -> None:
    folds_a = make_temporal_group_folds(shuffled_panel, **_fold_kwargs())
    reshuffled = shuffled_panel.sample(frac=1.0, random_state=999).reset_index(drop=True)
    folds_b = make_temporal_group_folds(reshuffled, **_fold_kwargs())

    assert len(folds_a) == len(folds_b) > 0
    for fa, fb in zip(folds_a, folds_b):
        assert fa.train_start == fb.train_start
        assert fa.test_start == fb.test_start
        assert fa.test_end == fb.test_end


def test_no_train_timestamp_after_test_start(shuffled_panel) -> None:
    folds = make_temporal_group_folds(shuffled_panel, **_fold_kwargs())
    panel = shuffled_panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    for fold in folds:
        train_ts = panel.loc[fold.train_idx, "timestamp"]
        test_ts = panel.loc[fold.test_idx, "timestamp"]
        assert train_ts.max() < test_ts.min()


def test_no_label_end_time_overlap(shuffled_panel) -> None:
    folds = make_temporal_group_folds(shuffled_panel, **_fold_kwargs())
    panel = shuffled_panel.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel["label_end_time"] = pd.to_datetime(panel["label_end_time"], utc=True)
    for fold in folds:
        train_label_end = panel.loc[fold.train_idx, "label_end_time"]
        test_ts = panel.loc[fold.test_idx, "timestamp"]
        assert train_label_end.max() < test_ts.min()


def test_all_symbols_split_by_time(shuffled_panel) -> None:
    folds = make_temporal_group_folds(shuffled_panel, **_fold_kwargs())
    all_symbols = set(shuffled_panel["symbol"].unique())
    # Every symbol appears in both train and test partitions across folds.
    seen_train, seen_test = set(), set()
    for fold in folds:
        seen_train |= set(shuffled_panel.loc[fold.train_idx, "symbol"])
        seen_test |= set(shuffled_panel.loc[fold.test_idx, "symbol"])
    assert seen_train == all_symbols
    assert seen_test == all_symbols


def test_walk_forward_report_flags_leakage_checks(shuffled_panel) -> None:
    class _MeanModel:
        def fit(self, x, y):
            self._mean = float(y.mean())
            return self

        def predict(self, x):
            return np.full(len(x), self._mean)

    report = walk_forward_validate_panel(
        panel=shuffled_panel,
        feature_cols=["feat_a", "feat_b"],
        target_col="target",
        timestamp_col="timestamp",
        symbol_col="symbol",
        label_end_col="label_end_time",
        model_factory=_MeanModel,
        n_splits=4,
        min_train_period=timedelta(days=20),
        test_period=timedelta(days=8),
        embargo=timedelta(days=5),
        transaction_cost_bps=5.0,
    )
    checks = report.leakage_checks
    assert checks["no_train_after_test_start"] is True
    assert checks["no_label_overlap"] is True
    assert checks["all_symbols_split_by_time"] is True
    # Embargo (5d) covers the 5-day max label horizon.
    assert checks["embargo_covers_max_label_horizon"] is True
    assert report.aggregate["folds"] > 0


def test_embargo_must_cover_max_label_horizon(shuffled_panel) -> None:
    # A 2-day embargo cannot cover the 5-day forward label horizon.
    report = walk_forward_validate_panel(
        panel=shuffled_panel,
        feature_cols=["feat_a", "feat_b"],
        target_col="target",
        timestamp_col="timestamp",
        symbol_col="symbol",
        label_end_col="label_end_time",
        model_factory=lambda: _Zero(),
        n_splits=4,
        min_train_period=timedelta(days=20),
        test_period=timedelta(days=8),
        embargo=timedelta(days=2),
        transaction_cost_bps=5.0,
    )
    assert report.leakage_checks["embargo_covers_max_label_horizon"] is False


class _Zero:
    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.zeros(len(x))


def test_build_forward_return_labels() -> None:
    ts = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": list(ts) * 1,
            "symbol": ["AAA"] * 10,
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        }
    )
    labels = build_forward_return_labels(prices, horizons=(5,))
    # First row: 5-day forward return = 105/100 - 1 = 0.05.
    first = labels.sort_values("timestamp").iloc[0]
    assert abs(first["fwd_return_5"] - 0.05) < 1e-9
    # Last 5 rows have no future price and are dropped.
    assert labels["fwd_return_5"].notna().all()
    assert len(labels) == 5
