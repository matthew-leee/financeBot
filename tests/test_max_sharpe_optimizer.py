"""Max-Sharpe core construction + macro-aware regime classification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategist import InterdayStrategist, StrategistConfig


def _cfg(**overrides) -> StrategistConfig:
    base = dict(
        core_universe=("AAA", "BBB"),
        hedge_universe=(),
        max_gross_exposure=1.0,
        max_net_exposure=0.6,
        max_symbol_weight=0.5,
        max_hedge_weight=0.25,
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


def _strat(**overrides) -> InterdayStrategist:
    return InterdayStrategist(config=_cfg(**overrides), feature_store=object())


def _cov2(corr: float) -> pd.DataFrame:
    vols = np.array([0.20, 0.20])
    cov = corr * np.outer(vols, vols)
    np.fill_diagonal(cov, vols**2)
    return pd.DataFrame(cov, index=["AAA", "BBB"], columns=["AAA", "BBB"])


def _sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    return float(mu @ w) / float(np.sqrt(w @ sigma @ w))


# ---------------------------------------------------------------------------
# Max-Sharpe tangency construction
# ---------------------------------------------------------------------------


def test_max_sharpe_beats_heuristic_on_correlated_assets() -> None:
    strat = _strat()
    sigma = _cov2(0.75)
    mu = pd.Series({"AAA": 0.10, "BBB": 0.06})
    ms = strat.max_sharpe_weights(expected_returns=mu, covariance=sigma)

    assert ms is not None
    # With 0.75 correlation, BBB is redundant: tangency concentrates in AAA.
    assert ms["AAA"] > 0.95
    assert ms.get("BBB", 0.0) < 0.05

    mu_v = mu.to_numpy()
    sig = sigma.to_numpy()
    w_ms = ms.reindex(["AAA", "BBB"]).fillna(0.0).to_numpy()

    # Legacy heuristic (inverse-vol-scaled score) for comparison.
    vol = np.sqrt(np.diag(sig))
    raw = (mu_v / vol) / vol
    w_h = raw / raw.sum()
    assert _sharpe(w_ms, mu_v, sig) > _sharpe(w_h, mu_v, sig)


def test_max_sharpe_is_deterministic_and_long_only() -> None:
    strat = _strat()
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0008, 0.01, len(idx)),
            "BBB": rng.normal(0.0004, 0.02, len(idx)),
        },
        index=idx,
    )
    cov = strat.estimate_covariance(returns, halflife_days=63, shrinkage=0.1)
    mu = pd.Series({"AAA": 0.09, "BBB": 0.03})

    a = strat.max_sharpe_weights(expected_returns=mu, covariance=cov)
    b = strat.max_sharpe_weights(expected_returns=mu, covariance=cov)
    assert a is not None and b is not None
    pd.testing.assert_series_equal(a, b)
    assert (a >= -1e-12).all()


def test_all_nonpositive_expected_returns_degrade_to_none() -> None:
    strat = _strat()
    mu = pd.Series({"AAA": -0.01, "BBB": 0.00})
    out = strat.max_sharpe_weights(
        expected_returns=mu, covariance=_cov2(0.1)
    )
    assert out is None


def test_optimizer_respects_symbol_cap_after_vol_targeting() -> None:
    strat = _strat(max_symbol_weight=0.3)
    weights = strat.optimize_core_portfolio(
        expected_returns=pd.Series({"AAA": 0.10}),
        covariance=pd.DataFrame([[0.0001]], index=["AAA"], columns=["AAA"]),
        previous_weights=None,
    )
    assert weights.notna().all()
    assert float(weights.abs().max()) <= 0.3 + 1e-9


def test_optimize_core_portfolio_falls_back_when_disabled() -> None:
    strat = _strat(use_max_sharpe=False)
    weights = strat.optimize_core_portfolio(
        expected_returns=pd.Series({"AAA": 0.10, "BBB": 0.04}),
        covariance=_cov2(0.2),
        previous_weights=None,
    )
    assert float(weights.sum()) > 0.0
    assert float(weights["AAA"]) > float(weights["BBB"])


# ---------------------------------------------------------------------------
# Macro-aware regime classifier
# ---------------------------------------------------------------------------


def _frame(**cols) -> pd.DataFrame:
    base = {"vol_63": [0.18], "drawdown_252": [-0.04]}
    base.update(cols)
    return pd.DataFrame(base, index=["X"])


def test_inflation_shock_detected_from_cpi_state() -> None:
    strat = _strat()
    frame = _frame(cpi_yoy=[0.05], cpi_yoy_chg_3m=[0.004], curve_10y_2y=[0.02])
    assert strat.predict_regime(frame) == "inflation_shock"


def test_hot_but_cooling_cpi_is_not_a_shock() -> None:
    strat = _strat()
    frame = _frame(cpi_yoy=[0.06], cpi_yoy_chg_3m=[-0.003], curve_10y_2y=[0.02])
    assert strat.predict_regime(frame) == "risk_on"


def test_inverted_yield_curve_signals_growth_slowdown() -> None:
    strat = _strat()
    frame = _frame(cpi_yoy=[0.02], cpi_yoy_chg_3m=[-0.001], curve_10y_2y=[-0.01])
    assert strat.predict_regime(frame) == "growth_slowdown"


def test_legacy_rules_hold_without_macro_columns() -> None:
    strat = _strat()
    assert strat.predict_regime(_frame(drawdown_252=[-0.25])) == "crisis"
    assert strat.predict_regime(_frame(vol_63=[0.40])) == "liquidity_stress"
    assert strat.predict_regime(_frame(drawdown_252=[-0.12])) == "growth_slowdown"
    assert strat.predict_regime(_frame()) == "risk_on"
