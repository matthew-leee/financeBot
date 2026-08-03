from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.data import FeatureSnapshot
from src.strategist import InterdayStrategist, StrategistConfig


def _cfg(**overrides) -> StrategistConfig:
    base = dict(
        core_universe=("AAA",),
        hedge_universe=("HDG",),
        max_gross_exposure=1.0,
        max_net_exposure=0.6,
        max_symbol_weight=0.5,
        max_hedge_weight=0.5,
        volatility_target=0.12,
        covariance_halflife_days=63,
        covariance_shrinkage=0.0,
        covariance_eigen_floor=1e-10,
        risk_aversion=8.0,
        turnover_penalty_bps=0.0,
        hedge_decay_penalty_bps=0.0,
        min_hedge_effectiveness=0.15,
    )
    base.update(overrides)
    return StrategistConfig(**base)


def _strategist(cfg) -> InterdayStrategist:
    return InterdayStrategist(config=cfg, feature_store=object(), model_registry=None)


def _cov(aaa_var, hdg_var, cov_ah):
    return pd.DataFrame(
        [[aaa_var, cov_ah], [cov_ah, hdg_var]],
        index=["AAA", "HDG"],
        columns=["AAA", "HDG"],
    )


def test_negative_covariance_inverse_etf_gives_positive_hedge() -> None:
    strat = _strategist(_cfg(hedge_decay_penalty_bps=0.0))
    cov = _cov(0.04, 0.04, -0.03)  # inverse ETF: negative covariance to core
    core = pd.Series({"AAA": 0.2})
    overlay = strat.compute_portfolio_hedge_overlay(core_weights=core, covariance=cov)
    assert overlay["HDG"] > 0.0


def test_zero_hedge_variance_returns_zero_hedge() -> None:
    strat = _strategist(_cfg())
    # Pair hedge ratio must be zero when hedge variance ~ 0.
    cov = _cov(0.04, 0.0, 0.0)
    ratio = strat.compute_pair_hedge_ratio(
        target_symbol="AAA", hedge_symbol="HDG", covariance=cov
    )
    assert ratio == 0.0


def test_singular_covariance_uses_pseudo_inverse_safely() -> None:
    strat = _strategist(_cfg())
    # Perfectly collinear -> singular covariance. pinv must keep this finite.
    cov = pd.DataFrame(
        [[0.04, 0.04], [0.04, 0.04]], index=["AAA", "HDG"], columns=["AAA", "HDG"]
    )
    core = pd.Series({"AAA": 0.2})
    overlay = strat.compute_portfolio_hedge_overlay(core_weights=core, covariance=cov)
    assert np.isfinite(overlay["HDG"])


def test_low_hedge_effectiveness_chooses_no_hedge() -> None:
    strat = _strategist(_cfg(min_hedge_effectiveness=0.5))
    # Very weak covariance -> variance reduction below the effectiveness bar.
    cov = _cov(0.04, 0.04, -0.0002)
    core = pd.Series({"AAA": 0.2})
    overlay = strat.compute_portfolio_hedge_overlay(core_weights=core, covariance=cov)
    assert overlay["HDG"] == 0.0


class _FakeFeatureStore:
    """Deterministic offline feature store double for the strategist."""

    def __init__(self, frame: pd.DataFrame, returns: pd.DataFrame) -> None:
        self._frame = frame
        self._returns = returns

    def build_interday_snapshot(self, *, universe, as_of, news_tau):
        frame = self._frame.reindex(list(universe)).fillna(0.0)
        return FeatureSnapshot(
            frame=frame, as_of=as_of, horizon="interday", lineage={"as_of": str(as_of)}
        )

    def return_panel(self, *, universe, as_of, lookback_days):
        return self._returns[[c for c in universe if c in self._returns.columns]]


def test_previously_held_hedge_exits_with_target_zero(tmp_path) -> None:
    cfg = _cfg()
    # Returns where HDG is essentially uncorrelated -> no hedge is selected now.
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.001, 0.01, size=120),
            "HDG": rng.normal(0.0, 0.01, size=120),
        },
        index=idx,
    )
    frame = pd.DataFrame({"ret_63": [0.05, 0.0]}, index=["AAA", "HDG"])
    store = _FakeFeatureStore(frame, returns)

    state_path = tmp_path / "strategist_state.json"
    # Seed prior state: HDG was an active hedge last run.
    state_path.write_text('{"weights": {"HDG": 0.1}, "active_hedges": ["HDG"]}', encoding="utf-8")

    strat = InterdayStrategist(
        config=cfg, feature_store=store, model_registry=None, state_path=str(state_path)
    )
    matrix = strat.generate_allocation(as_of=datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert "HDG" in matrix.symbols()
    assert abs(matrix.get("HDG").target_weight) < 1e-9
    assert matrix.get("HDG").direction_bias == "flat"


def test_covariance_is_psd() -> None:
    strat = _strategist(_cfg(covariance_shrinkage=0.2))
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {"AAA": rng.normal(0, 0.01, 100), "HDG": rng.normal(0, 0.01, 100)}, index=idx
    )
    cov = strat.estimate_covariance(returns, halflife_days=63, shrinkage=0.2)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigvals >= -1e-9).all()
