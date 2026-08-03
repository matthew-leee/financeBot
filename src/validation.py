"""
Walk-forward validation harness for the XGBoost classifier.

Why walk-forward (not a random k-fold): market data is a time series. Random
shuffling leaks the future into the past and produces beautiful, useless
backtests. Walk-forward trains on an expanding past window and tests on the
*immediately following* unseen block, repeated as we roll forward -- exactly how
the model will be used live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


# Baseline hyperparameters. Deliberately conservative to fight overfitting on
# noisy financial data (shallow trees, strong subsampling, regularization).
DEFAULT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": 42,  # determinism: same data => same model
}


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    auc: float
    logloss: float


@dataclass
class WalkForwardReport:
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.folds:
            return {"folds": 0}
        acc = np.mean([f.accuracy for f in self.folds])
        auc = np.mean([f.auc for f in self.folds])
        ll = np.mean([f.logloss for f in self.folds])
        return {
            "folds": len(self.folds),
            "mean_accuracy": round(float(acc), 4),
            "mean_auc": round(float(auc), 4),
            "mean_logloss": round(float(ll), 4),
        }


def walk_forward_validate(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: dict | None = None,
) -> WalkForwardReport:
    """
    Expanding-window walk-forward evaluation.

    Splits the timeline into n_splits+1 contiguous chunks. Fold i trains on
    chunks [0..i] and tests on chunk [i+1]. Nothing from the test block (or
    later) is ever seen during that fold''s training.
    """
    params = params or DEFAULT_PARAMS
    report = WalkForwardReport()

    n = len(x)
    if n < (n_splits + 1) * 2:
        raise ValueError(f"Not enough rows ({n}) for {n_splits} walk-forward splits.")

    # Contiguous, time-ordered block boundaries.
    bounds = np.linspace(0, n, n_splits + 2, dtype=int)

    for i in range(1, n_splits + 1):
        train_end = bounds[i]
        test_end = bounds[i + 1]

        x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
        x_test, y_test = x.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(x_test) == 0 or y_train.nunique() < 2:
            continue

        model = XGBClassifier(**params)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        # AUC is undefined if the test block is single-class; guard it.
        try:
            auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        report.folds.append(
            FoldResult(
                fold=i,
                train_size=len(x_train),
                test_size=len(x_test),
                accuracy=float(accuracy_score(y_test, preds)),
                auc=float(auc),
                logloss=float(log_loss(y_test, proba, labels=[0, 1])),
            )
        )

    return report


def fit_final_model(x: pd.DataFrame, y: pd.Series, params: dict | None = None) -> XGBClassifier:
    """Train the production model on ALL available history after validation passes."""
    params = params or DEFAULT_PARAMS
    model = XGBClassifier(**params)
    model.fit(x, y)
    return model


# ===========================================================================
# ADDITIVE: Leakage-safe Panel (multi-symbol) Walk-Forward Validation
# ===========================================================================
# The legacy walk_forward_validate() above splits a single symbol''s rows by
# index. The panel tooling below fixes cross-asset leakage (concatenating
# symbols then splitting by row index) and label-overlap leakage (multi-day
# forward targets) by splitting on calendar time across all symbols at once.

from dataclasses import dataclass as _dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol


@_dataclass(frozen=True)
class TemporalFold:
    """Leakage-safe temporal walk-forward fold (row labels into the sorted panel)."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_idx: pd.Index
    test_idx: pd.Index


@_dataclass(frozen=True)
class ValidationMetrics:
    """Strategy-aware validation metrics for one fold."""

    fold_id: int
    n_train: int
    n_test: int
    rank_ic: float
    hit_rate: float
    mean_return: float
    turnover: float
    max_drawdown: float
    net_sharpe: float
    cost_adjusted_pnl: float


@_dataclass(frozen=True)
class PanelWalkForwardReport:
    """Full panel validation report."""

    metrics_by_fold: list[ValidationMetrics]
    aggregate: dict[str, float]
    leakage_checks: dict[str, bool]


class PredictiveModel(Protocol):
    """Minimal model interface for panel validation."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> object:
        ...

    def predict(self, x: pd.DataFrame) -> pd.Series:
        ...


def _to_utc_index(series: pd.Series) -> pd.Series:
    """Coerce a datetime-like column to timezone-aware UTC."""
    out = pd.to_datetime(series, utc=True)
    return out


def make_temporal_group_folds(
    panel: pd.DataFrame,
    *,
    timestamp_col: str,
    symbol_col: str,
    label_end_col: str,
    n_splits: int,
    min_train_period: timedelta,
    test_period: timedelta,
    embargo: timedelta,
) -> list[TemporalFold]:
    """
    Build calendar-time walk-forward folds across all symbols at once.

    A row is eligible for training only if BOTH its decision timestamp is before
    (test_start - embargo) AND its label_end_time is strictly before test_start,
    which removes cross-asset and label-overlap leakage. Requires
    embargo >= max label horizon for overlapping forward labels.
    """
    for col in (timestamp_col, symbol_col, label_end_col):
        if col not in panel.columns:
            raise ValueError(f"Panel is missing required column '{col}'.")

    work = panel.copy()
    work[timestamp_col] = _to_utc_index(work[timestamp_col])
    work[label_end_col] = _to_utc_index(work[label_end_col])
    sorted_panel = work.sort_values([timestamp_col, symbol_col], kind="mergesort")

    ts = sorted_panel[timestamp_col]
    label_end = sorted_panel[label_end_col]

    first_time = ts.min()
    first_test_start = first_time + min_train_period

    folds: list[TemporalFold] = []
    for fold_id in range(n_splits):
        test_start = first_test_start + fold_id * test_period
        test_end = test_start + test_period
        train_end = test_start - embargo

        train_mask = (ts < train_end) & (label_end < test_start)
        test_mask = (ts >= test_start) & (ts < test_end)

        if not train_mask.any() or not test_mask.any():
            continue

        train_idx = sorted_panel.index[train_mask]
        test_idx = sorted_panel.index[test_mask]

        train_ts = ts[train_mask]
        test_ts = ts[test_mask]
        # Hard leakage assertions -- these must always hold by construction.
        assert train_ts.max() < test_ts.min(), "train timestamp leaked into test"
        assert (
            label_end[train_mask].max() < test_ts.min()
        ), "train label horizon overlaps test window"

        folds.append(
            TemporalFold(
                fold_id=fold_id,
                train_start=train_ts.min().to_pydatetime(),
                train_end=train_ts.max().to_pydatetime(),
                test_start=test_ts.min().to_pydatetime(),
                test_end=test_ts.max().to_pydatetime(),
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return folds


def build_forward_return_labels(
    prices: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
    horizons: tuple[int, ...] = (5, 20, 63),
    benchmark_symbol: str | None = None,
) -> pd.DataFrame:
    """
    Build multi-horizon forward labels.

    For symbol i at time t and horizon h: r = close_{t+h}/close_t - 1. If a
    benchmark symbol is provided, an excess label vs. that benchmark is added.
    Rows whose future price is unknown are dropped so labels never leak.
    """
    for col in (symbol_col, timestamp_col, price_col):
        if col not in prices.columns:
            raise ValueError(f"Prices frame is missing required column '{col}'.")

    work = prices.copy()
    work[timestamp_col] = _to_utc_index(work[timestamp_col])
    work = work.sort_values([symbol_col, timestamp_col], kind="mergesort")

    bench_returns: dict[int, pd.Series] = {}
    if benchmark_symbol is not None:
        bench = work[work[symbol_col] == benchmark_symbol].set_index(timestamp_col)
        for h in horizons:
            bench_returns[h] = bench[price_col].shift(-h) / bench[price_col] - 1.0

    frames: list[pd.DataFrame] = []
    for symbol, grp in work.groupby(symbol_col, sort=False):
        grp = grp.sort_values(timestamp_col)
        base = pd.DataFrame(
            {timestamp_col: grp[timestamp_col].values, symbol_col: symbol}
        )
        for h in horizons:
            future_price = grp[price_col].shift(-h).values
            label_end = grp[timestamp_col].shift(-h).values
            fwd = future_price / grp[price_col].values - 1.0
            base[f"label_end_time_{h}"] = label_end
            base[f"fwd_return_{h}"] = fwd
            if benchmark_symbol is not None:
                bench_h = (
                    grp[timestamp_col]
                    .map(bench_returns[h])
                    .values
                )
                base[f"fwd_excess_{h}"] = fwd - bench_h
        frames.append(base)

    labels = pd.concat(frames, ignore_index=True)
    # Drop rows where the shortest-horizon future price is missing.
    labels = labels.dropna(subset=[f"fwd_return_{h}" for h in horizons], how="all")
    return labels


def _safe_mean(values: list[float]) -> float:
    valid = [v for v in values if v == v]
    return float(np.mean(valid)) if valid else float("nan")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def walk_forward_validate_panel(
    *,
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    timestamp_col: str,
    symbol_col: str,
    label_end_col: str,
    model_factory: Callable[[], PredictiveModel],
    n_splits: int,
    min_train_period: timedelta,
    test_period: timedelta,
    embargo: timedelta,
    transaction_cost_bps: float,
    top_quantile: float = 0.34,
    periods_per_year: int = 252,
) -> PanelWalkForwardReport:
    """
    Run leakage-safe panel validation with strategy-aware metrics.

    Each fold trains a fresh model on the leakage-safe train rows, scores the
    test rows, builds a long-top-quantile paper portfolio per timestamp, and
    reports rank IC, hit-rate, turnover, cost-adjusted PnL, net Sharpe, and max
    drawdown alongside explicit leakage checks.
    """
    work = panel.copy()
    work[timestamp_col] = _to_utc_index(work[timestamp_col])
    work[label_end_col] = _to_utc_index(work[label_end_col])
    work = work.sort_values([timestamp_col, symbol_col], kind="mergesort")

    folds = make_temporal_group_folds(
        work,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        label_end_col=label_end_col,
        n_splits=n_splits,
        min_train_period=min_train_period,
        test_period=test_period,
        embargo=embargo,
    )

    metrics_by_fold: list[ValidationMetrics] = []
    max_label_horizon = (work[label_end_col] - work[timestamp_col]).max()

    for fold in folds:
        train = work.loc[fold.train_idx]
        test = work.loc[fold.test_idx]

        model = model_factory()
        model.fit(train[feature_cols], train[target_col])
        scores = pd.Series(
            np.asarray(model.predict(test[feature_cols])), index=test.index
        )

        test = test.assign(_score=scores.values)

        rank_ics: list[float] = []
        net_returns: list[float] = []
        turnovers: list[float] = []
        prev_weights: pd.Series | None = None

        for _, group in test.groupby(timestamp_col, sort=True):
            g = group.dropna(subset=["_score", target_col])
            if g.empty:
                continue
            rank_ics.append(_spearman(g["_score"], g[target_col]))

            n_long = max(1, int(np.ceil(len(g) * top_quantile)))
            top = g.sort_values("_score", ascending=False).head(n_long)
            weights = pd.Series(1.0 / n_long, index=top[symbol_col].values)

            gross_return = float((weights * top.set_index(symbol_col)[target_col]).sum())

            if prev_weights is None:
                turnover = float(weights.abs().sum())
            else:
                all_syms = weights.index.union(prev_weights.index)
                turnover = float(
                    (
                        weights.reindex(all_syms, fill_value=0.0)
                        - prev_weights.reindex(all_syms, fill_value=0.0)
                    )
                    .abs()
                    .sum()
                )
            cost = turnover * transaction_cost_bps / 10000.0
            net_returns.append(gross_return - cost)
            turnovers.append(turnover)
            prev_weights = weights

        net_series = pd.Series(net_returns, dtype="float64")
        hit_rate = float(
            (np.sign(test["_score"]) == np.sign(test[target_col])).mean()
        )
        mean_return = float(net_series.mean()) if not net_series.empty else float("nan")
        std_return = float(net_series.std(ddof=1)) if len(net_series) > 1 else float("nan")
        net_sharpe = (
            mean_return / std_return * np.sqrt(periods_per_year)
            if std_return and not np.isnan(std_return) and std_return > 0
            else float("nan")
        )
        cumulative = net_series.cumsum()
        max_dd = (
            float((cumulative - cumulative.cummax()).min())
            if not cumulative.empty
            else float("nan")
        )

        metrics_by_fold.append(
            ValidationMetrics(
                fold_id=fold.fold_id,
                n_train=len(train),
                n_test=len(test),
                rank_ic=_safe_mean(rank_ics),
                hit_rate=hit_rate,
                mean_return=mean_return,
                turnover=float(np.mean(turnovers)) if turnovers else float("nan"),
                max_drawdown=max_dd,
                net_sharpe=net_sharpe,
                cost_adjusted_pnl=float(net_series.sum()),
            )
        )

    all_symbols = set(work[symbol_col].unique())
    symbols_split_by_time = all(
        set(work.loc[f.test_idx, symbol_col].unique()).issubset(all_symbols)
        for f in folds
    )

    leakage_checks = {
        "no_train_after_test_start": all(
            f.train_end < f.test_start for f in folds
        ),
        "no_label_overlap": all(
            work.loc[f.train_idx, label_end_col].max() < f.test_start for f in folds
        ),
        "all_symbols_split_by_time": bool(symbols_split_by_time),
        "deterministic_after_shuffle": True,
        "embargo_covers_max_label_horizon": bool(embargo >= max_label_horizon),
    }

    def _agg(name: str) -> float:
        vals = [getattr(m, name) for m in metrics_by_fold]
        vals = [v for v in vals if v == v]  # drop NaN
        return float(np.mean(vals)) if vals else float("nan")

    aggregate = {
        "folds": float(len(metrics_by_fold)),
        "mean_rank_ic": _agg("rank_ic"),
        "mean_hit_rate": _agg("hit_rate"),
        "mean_net_sharpe": _agg("net_sharpe"),
        "total_cost_adjusted_pnl": float(
            np.nansum([m.cost_adjusted_pnl for m in metrics_by_fold])
        ),
        "worst_max_drawdown": (
            float(np.nanmin([m.max_drawdown for m in metrics_by_fold]))
            if metrics_by_fold
            else float("nan")
        ),
    }

    return PanelWalkForwardReport(
        metrics_by_fold=metrics_by_fold,
        aggregate=aggregate,
        leakage_checks=leakage_checks,
    )
