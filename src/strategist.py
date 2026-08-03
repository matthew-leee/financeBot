"""
Interday Strategist -- daily structural portfolio construction.

The strategist is the *slow* brain of the dual-horizon engine. Once per day (or
on a macro/filing/news release trigger) it turns point-in-time interday features
into a target AllocationMatrix: per-symbol target weights, intraday envelopes,
and a covariance-derived hedge overlay. The intraday tactical executor may only
trade *inside* the envelopes this module emits -- it can never widen risk.

Design guarantees:
  * No broker access. This class is pure portfolio math over PIT features.
  * Covariance is EWMA + shrinkage + PSD eigen-floored, so it is always usable.
  * Hedging is covariance-based (min-variance overlay), not noisy 30-day Pearson.
  * A hedge is only kept if it clears a variance-reduction effectiveness bar.
  * Hedges that are no longer needed are emitted with target_weight 0 so the
    executor can exit them (fixes the orphaned-hedge bug).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Protocol

import numpy as np
import pandas as pd

import config

DirectionBias = Literal["long", "short", "flat", "hedge"]

_EPS = 1e-12


@dataclass(frozen=True)
class StrategistConfig:
    """Configuration for daily structural allocation."""

    core_universe: tuple[str, ...]
    hedge_universe: tuple[str, ...]
    max_gross_exposure: float
    max_net_exposure: float
    max_symbol_weight: float
    max_hedge_weight: float
    volatility_target: float
    covariance_halflife_days: int
    covariance_shrinkage: float
    covariance_eigen_floor: float
    risk_aversion: float
    turnover_penalty_bps: float
    hedge_decay_penalty_bps: float
    min_hedge_effectiveness: float
    news_tau_days: float = float(config.NEWS_TAU_DAYS)
    allow_core_shorts: bool = False

    @classmethod
    def from_config(cls) -> "StrategistConfig":
        """Build a StrategistConfig from the central config module constants."""
        return cls(
            core_universe=config.CORE_UNIVERSE,
            hedge_universe=config.HEDGE_UNIVERSE,
            max_gross_exposure=config.MAX_GROSS_EXPOSURE,
            max_net_exposure=config.MAX_NET_EXPOSURE,
            max_symbol_weight=config.MAX_SYMBOL_WEIGHT,
            max_hedge_weight=config.MAX_HEDGE_WEIGHT,
            volatility_target=config.VOLATILITY_TARGET_ANNUAL,
            covariance_halflife_days=config.COVARIANCE_HALFLIFE_DAYS,
            covariance_shrinkage=config.COVARIANCE_SHRINKAGE,
            covariance_eigen_floor=config.COVARIANCE_EIGEN_FLOOR,
            risk_aversion=config.RISK_AVERSION,
            turnover_penalty_bps=config.TURNOVER_PENALTY_BPS,
            hedge_decay_penalty_bps=config.HEDGE_DECAY_PENALTY_BPS,
            min_hedge_effectiveness=config.MIN_HEDGE_EFFECTIVENESS,
            news_tau_days=float(config.NEWS_TAU_DAYS),
        )


@dataclass(frozen=True)
class TargetAllocation:
    """One target allocation row (desired weight + intraday envelope)."""

    symbol: str
    target_weight: float
    min_weight: float
    max_weight: float
    direction_bias: DirectionBias
    volatility_ceiling: float
    rebalance_priority: float
    hedge_for: str | None = None
    hedge_ratio: float | None = None
    expected_return: float | None = None
    expected_volatility: float | None = None


@dataclass(frozen=True)
class AllocationMatrix:
    """Daily structural target allocation."""

    as_of: datetime
    rows: dict[str, TargetAllocation]
    covariance_version: str
    feature_snapshot_id: str
    regime: str
    diagnostics: dict[str, float | str] = field(default_factory=dict)

    def symbols(self) -> set[str]:
        """Return symbols explicitly controlled by this matrix."""
        return set(self.rows)

    def get(self, symbol: str) -> TargetAllocation | None:
        """Return allocation row for symbol, if present."""
        return self.rows.get(symbol)

    def to_dict(self) -> dict:
        """JSON-serializable view for persistence / inspection."""
        return {
            "as_of": self.as_of.isoformat(),
            "covariance_version": self.covariance_version,
            "feature_snapshot_id": self.feature_snapshot_id,
            "regime": self.regime,
            "diagnostics": self.diagnostics,
            "rows": {
                sym: {
                    "symbol": r.symbol,
                    "target_weight": r.target_weight,
                    "min_weight": r.min_weight,
                    "max_weight": r.max_weight,
                    "direction_bias": r.direction_bias,
                    "volatility_ceiling": r.volatility_ceiling,
                    "rebalance_priority": r.rebalance_priority,
                    "hedge_for": r.hedge_for,
                    "hedge_ratio": r.hedge_ratio,
                    "expected_return": r.expected_return,
                    "expected_volatility": r.expected_volatility,
                }
                for sym, r in self.rows.items()
            },
        }


class FeatureStoreLike(Protocol):
    def build_interday_snapshot(self, *, universe, as_of, news_tau): ...

    def return_panel(self, *, universe, as_of, lookback_days): ...


class ModelRegistryLike(Protocol):
    def predict(self, role: str, features: pd.DataFrame) -> pd.Series: ...


def _psd_floor(matrix: np.ndarray, eigen_floor: float) -> np.ndarray:
    """Symmetrize and floor eigenvalues so the matrix is positive semidefinite."""
    sym = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eigen_floor, None)
    return vecs @ np.diag(vals) @ vecs.T


class InterdayStrategist:
    """Daily portfolio construction layer (see Architecture Module C)."""

    def __init__(
        self,
        *,
        config: StrategistConfig,
        feature_store: FeatureStoreLike,
        model_registry: ModelRegistryLike | None = None,
        state_path: str = "models/strategist_state.json",
    ) -> None:
        # No broker access belongs in this class.
        self.config = config
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.state_path = state_path

    # -- top-level entrypoint ------------------------------------------------

    def generate_allocation(self, *, as_of: datetime) -> AllocationMatrix:
        """Generate the daily target allocation matrix for `as_of`."""
        cfg = self.config
        universe = tuple(dict.fromkeys(cfg.core_universe + cfg.hedge_universe))

        snapshot = self.feature_store.build_interday_snapshot(
            universe=universe,
            as_of=as_of,
            news_tau=timedelta(days=cfg.news_tau_days),
        )

        mu = self.predict_expected_returns(snapshot.frame)
        regime = self.predict_regime(snapshot.frame)

        returns = self._load_return_panel(universe, as_of)
        covariance = self.estimate_covariance(
            returns,
            halflife_days=cfg.covariance_halflife_days,
            shrinkage=cfg.covariance_shrinkage,
        )

        core_in_cov = [s for s in cfg.core_universe if s in covariance.index]
        prev_weights = self._load_previous_weights()
        core_weights = self.optimize_core_portfolio(
            expected_returns=mu.reindex(core_in_cov).fillna(0.0),
            covariance=covariance.loc[core_in_cov, core_in_cov]
            if core_in_cov
            else pd.DataFrame(),
            previous_weights=prev_weights,
        )

        hedge_weights = self.compute_portfolio_hedge_overlay(
            core_weights=core_weights, covariance=covariance
        )

        combined = core_weights.add(hedge_weights, fill_value=0.0)
        combined = self.enforce_portfolio_constraints(combined, covariance, regime)

        rows = self.build_target_rows(combined, mu, covariance, regime, hedge_weights)

        # Include zero-weight rows for previously active hedges no longer needed
        # so the executor can exit them (orphaned-hedge fix).
        prev_hedges = self._load_previous_active_hedges()
        for hedge in prev_hedges:
            if hedge not in rows or abs(rows[hedge].target_weight) <= _EPS:
                if hedge not in combined or abs(float(combined.get(hedge, 0.0))) <= _EPS:
                    rows[hedge] = TargetAllocation(
                        symbol=hedge,
                        target_weight=0.0,
                        min_weight=0.0,
                        max_weight=0.0,
                        direction_bias="flat",
                        volatility_ceiling=cfg.volatility_target,
                        rebalance_priority=1.0,
                        hedge_for=None,
                        hedge_ratio=None,
                    )

        matrix = AllocationMatrix(
            as_of=snapshot.as_of,
            rows=rows,
            covariance_version=self._covariance_version(covariance),
            feature_snapshot_id=snapshot.lineage.get("as_of", ""),
            regime=regime,
            diagnostics={
                "n_core": float(int((core_weights.abs() > _EPS).sum())),
                "n_hedge": float(int((hedge_weights.abs() > _EPS).sum())),
                "gross_exposure": float(combined.abs().sum()),
                "net_exposure": float(combined.sum()),
            },
        )

        # Persist state for turnover smoothing + hedge lifecycle next run.
        active_hedges = [
            s
            for s in cfg.hedge_universe
            if s in combined and abs(float(combined[s])) > _EPS
        ]
        self._save_state(combined, active_hedges)
        return matrix

    # -- expected returns / regime ------------------------------------------

    def predict_expected_returns(self, features: pd.DataFrame) -> pd.Series:
        """Predict horizon-consistent expected return per symbol."""
        if features.empty:
            return pd.Series(dtype="float64")

        raw: pd.Series | None = None
        if self.model_registry is not None:
            try:
                raw = pd.Series(
                    self.model_registry.predict("expected_return", features)
                ).reindex(features.index)
            except Exception:  # noqa: BLE001 -- fall back to deterministic estimate
                raw = None

        if raw is None:
            # Deterministic fallback: risk-adjusted medium-horizon momentum.
            ret_63 = features.get("ret_63", pd.Series(0.0, index=features.index))
            ret_252 = features.get("ret_252", pd.Series(0.0, index=features.index))
            raw = 0.6 * ret_63.fillna(0.0) + 0.4 * ret_252.fillna(0.0)

        raw = raw.astype("float64")
        # Clip extreme values to robust percentile bounds.
        if raw.notna().sum() >= 5:
            lo, hi = raw.quantile(0.05), raw.quantile(0.95)
            raw = raw.clip(lo, hi)

        # Liquidity / staleness penalties from missing indicators.
        stale_penalty = pd.Series(0.0, index=features.index)
        for col in features.columns:
            if col.endswith("_missing"):
                stale_penalty = stale_penalty + features[col].fillna(0.0) * 1e-4
        return (raw - stale_penalty).fillna(0.0)

    def predict_regime(self, features: pd.DataFrame) -> str:
        """Predict a coarse macro regime label from interday features."""
        if self.model_registry is not None:
            try:
                pred = self.model_registry.predict("regime", features)
                if len(pred) > 0:
                    return str(pred.iloc[0])
            except Exception:  # noqa: BLE001 -- deterministic fallback below
                pass

        if features.empty:
            return "risk_on"
        avg_vol = float(features.get("vol_63", pd.Series(dtype="float64")).mean())
        avg_dd = float(features.get("drawdown_252", pd.Series(dtype="float64")).mean())
        if np.isnan(avg_vol) and np.isnan(avg_dd):
            return "risk_on"
        if not np.isnan(avg_dd) and avg_dd <= -0.20:
            return "crisis"
        if not np.isnan(avg_vol) and avg_vol >= 0.35:
            return "liquidity_stress"
        if not np.isnan(avg_dd) and avg_dd <= -0.10:
            return "growth_slowdown"
        return "risk_on"

    # -- covariance ----------------------------------------------------------

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        *,
        halflife_days: int,
        shrinkage: float,
    ) -> pd.DataFrame:
        """Estimate an EWMA + shrinkage + PSD-floored covariance matrix."""
        if returns is None or returns.empty:
            return pd.DataFrame()
        returns = returns.sort_index().dropna(how="all").dropna(axis=1, how="all")
        cols = list(returns.columns)
        if not cols:
            return pd.DataFrame()

        x = returns.fillna(0.0).to_numpy(dtype="float64")
        n_obs = x.shape[0]
        if n_obs < 2:
            # Not enough data -> tiny diagonal so downstream math stays stable.
            floor = max(self.config.covariance_eigen_floor, 1e-8)
            return pd.DataFrame(np.eye(len(cols)) * floor, index=cols, columns=cols)

        lam = 0.5 ** (1.0 / max(halflife_days, 1))
        weights = lam ** np.arange(n_obs - 1, -1, -1)
        weights = weights / weights.sum()

        mean = np.average(x, axis=0, weights=weights)
        centered = x - mean
        ewma_cov = (centered * weights[:, None]).T @ centered

        diag_cov = np.diag(np.diag(ewma_cov))
        sigma = (1.0 - shrinkage) * ewma_cov + shrinkage * diag_cov
        sigma = _psd_floor(sigma, self.config.covariance_eigen_floor)
        return pd.DataFrame(sigma, index=cols, columns=cols)

    # -- hedge math ----------------------------------------------------------

    def compute_pair_hedge_ratio(
        self,
        *,
        target_symbol: str,
        hedge_symbol: str,
        covariance: pd.DataFrame,
    ) -> float:
        """h* = -Cov(target, hedge) / Var(hedge). Returns 0 if hedge var ~ 0."""
        if target_symbol not in covariance.index or hedge_symbol not in covariance.index:
            return 0.0
        var_hedge = float(covariance.loc[hedge_symbol, hedge_symbol])
        if var_hedge <= _EPS:
            return 0.0
        cov_th = float(covariance.loc[target_symbol, hedge_symbol])
        return -cov_th / var_hedge

    def compute_portfolio_hedge_overlay(
        self,
        *,
        core_weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """Compute a minimum-variance multi-instrument hedge overlay."""
        cfg = self.config
        zeros = pd.Series(0.0, index=list(cfg.hedge_universe), dtype="float64")
        if covariance.empty or core_weights is None or core_weights.empty:
            return zeros

        core = core_weights[core_weights.abs() > _EPS]
        core = core[[s for s in core.index if s in covariance.index]]
        hedges = [h for h in cfg.hedge_universe if h in covariance.index]
        if core.empty or not hedges:
            return zeros

        sigma_hh = covariance.loc[hedges, hedges].to_numpy(dtype="float64")
        sigma_hc = covariance.loc[hedges, core.index].to_numpy(dtype="float64")
        sigma_cc = covariance.loc[core.index, core.index].to_numpy(dtype="float64")
        w_c = core.to_numpy(dtype="float64")

        sigma_hh_inv = np.linalg.pinv(_psd_floor(sigma_hh, cfg.covariance_eigen_floor))
        h_raw = -sigma_hh_inv @ sigma_hc @ w_c

        # Soft-threshold decay/carry penalty for sparsity.
        lam_decay = cfg.hedge_decay_penalty_bps / 10000.0
        h = np.zeros_like(h_raw)
        for j in range(len(h_raw)):
            var_jj = max(sigma_hh[j, j], _EPS)
            penalty = lam_decay * 1.0 / (2.0 * var_jj)
            magnitude = max(0.0, abs(h_raw[j]) - penalty)
            h[j] = np.sign(h_raw[j]) * magnitude
            h[j] = float(np.clip(h[j], -cfg.max_hedge_weight, cfg.max_hedge_weight))

        pre_var = float(w_c @ sigma_cc @ w_c)
        if pre_var <= _EPS:
            return zeros
        variance_reduction = float(w_c @ sigma_hc.T @ sigma_hh_inv @ sigma_hc @ w_c)
        effectiveness = variance_reduction / pre_var
        if effectiveness < cfg.min_hedge_effectiveness:
            return zeros

        overlay = pd.Series(h, index=hedges, dtype="float64")
        return overlay.reindex(cfg.hedge_universe).fillna(0.0)

    # -- core optimization ---------------------------------------------------

    def optimize_core_portfolio(
        self,
        *,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        previous_weights: pd.Series | None,
    ) -> pd.Series:
        """Build a constrained, volatility-targeted core allocation."""
        cfg = self.config
        if expected_returns is None or expected_returns.empty or covariance.empty:
            return pd.Series(dtype="float64")

        symbols = [s for s in expected_returns.index if s in covariance.index]
        if not symbols:
            return pd.Series(dtype="float64")

        mu = expected_returns.reindex(symbols).fillna(0.0)
        sigma = covariance.loc[symbols, symbols]
        vol = np.sqrt(np.clip(np.diag(sigma.to_numpy()), _EPS, None))
        vol = pd.Series(vol, index=symbols)

        score = mu / vol.clip(lower=_EPS)
        if not cfg.allow_core_shorts:
            score = score.clip(lower=0.0)
            keep = score > 0.0
        else:
            keep = score.abs() > 0.0

        raw = pd.Series(0.0, index=symbols, dtype="float64")
        raw[keep] = score[keep] / vol[keep].clip(lower=_EPS)
        gross = raw.abs().sum()
        if gross <= _EPS:
            return pd.Series(0.0, index=symbols, dtype="float64")
        weights = raw / gross

        # Scale to the annualized volatility target.
        port_vol = float(np.sqrt(max(weights.values @ sigma.to_numpy() @ weights.values, _EPS)))
        if port_vol > _EPS:
            weights = weights * (cfg.volatility_target / port_vol)

        # Per-symbol caps, then rescale to gross/net exposure caps.
        weights = weights.clip(-cfg.max_symbol_weight, cfg.max_symbol_weight)
        weights = self._apply_exposure_caps(weights)

        # Turnover smoothing versus previous weights.
        if previous_weights is not None and not previous_weights.empty:
            prev = previous_weights.reindex(symbols).fillna(0.0)
            blend = 1.0 - min(max(cfg.turnover_penalty_bps / 100.0, 0.0), 0.9)
            weights = blend * weights + (1.0 - blend) * prev
            weights = weights.clip(-cfg.max_symbol_weight, cfg.max_symbol_weight)
            weights = self._apply_exposure_caps(weights)

        return weights

    def enforce_portfolio_constraints(
        self, combined: pd.Series, covariance: pd.DataFrame, regime: str
    ) -> pd.Series:
        """Apply per-symbol, hedge, gross, and net caps (regime-aware)."""
        cfg = self.config
        if combined.empty:
            return combined
        weights = combined.copy().astype("float64")

        for sym in weights.index:
            cap = (
                cfg.max_hedge_weight
                if sym in cfg.hedge_universe
                else cfg.max_symbol_weight
            )
            weights[sym] = float(np.clip(weights[sym], -cap, cap))

        # Regime tightens the gross budget in stressed states.
        regime_factor = {
            "risk_on": 1.0,
            "growth_slowdown": 0.8,
            "inflation_shock": 0.7,
            "liquidity_stress": 0.6,
            "crisis": 0.4,
        }.get(regime, 1.0)
        gross_cap = cfg.max_gross_exposure * regime_factor
        weights = self._apply_exposure_caps(weights, gross_cap=gross_cap)
        return weights

    def _apply_exposure_caps(
        self, weights: pd.Series, gross_cap: float | None = None
    ) -> pd.Series:
        cfg = self.config
        gross_cap = cfg.max_gross_exposure if gross_cap is None else gross_cap
        gross = float(weights.abs().sum())
        if gross > gross_cap and gross > _EPS:
            weights = weights * (gross_cap / gross)
        net = float(weights.sum())
        if abs(net) > cfg.max_net_exposure and abs(net) > _EPS:
            # Shift uniformly to pull net back within the cap.
            adjustment = (abs(net) - cfg.max_net_exposure) * np.sign(net)
            weights = weights - adjustment / len(weights)
        return weights

    # -- row assembly --------------------------------------------------------

    def build_target_rows(
        self,
        combined: pd.Series,
        mu: pd.Series,
        covariance: pd.DataFrame,
        regime: str,
        hedge_weights: pd.Series,
    ) -> dict[str, TargetAllocation]:
        cfg = self.config
        rows: dict[str, TargetAllocation] = {}
        for sym in combined.index:
            w = float(combined[sym])
            is_hedge = sym in cfg.hedge_universe
            vol = (
                float(np.sqrt(max(covariance.loc[sym, sym], 0.0)))
                if sym in covariance.index
                else float("nan")
            )
            if abs(w) <= _EPS:
                bias: DirectionBias = "flat"
            elif is_hedge:
                bias = "hedge"
            elif w > 0:
                bias = "long"
            else:
                bias = "short"

            band = max(abs(w) * 0.25, 0.005)
            hedge_for = None
            hedge_ratio = None
            if is_hedge and abs(w) > _EPS:
                # Attribute the hedge to the largest core position it offsets.
                core_syms = [
                    s for s in cfg.core_universe if s in combined.index
                ]
                if core_syms:
                    hedge_for = max(core_syms, key=lambda s: abs(float(combined[s])))
                    hedge_ratio = self.compute_pair_hedge_ratio(
                        target_symbol=hedge_for,
                        hedge_symbol=sym,
                        covariance=covariance,
                    )

            rows[sym] = TargetAllocation(
                symbol=sym,
                target_weight=w,
                min_weight=w - band,
                max_weight=w + band,
                direction_bias=bias,
                volatility_ceiling=cfg.volatility_target,
                rebalance_priority=abs(w),
                hedge_for=hedge_for,
                hedge_ratio=hedge_ratio,
                expected_return=float(mu.get(sym, float("nan"))),
                expected_volatility=vol,
            )
        return rows

    # -- state / helpers -----------------------------------------------------

    def _load_return_panel(self, universe, as_of: datetime) -> pd.DataFrame:
        try:
            return self.feature_store.return_panel(
                universe=universe, as_of=as_of, lookback_days=400
            )
        except AttributeError:
            return pd.DataFrame()

    def _covariance_version(self, covariance: pd.DataFrame) -> str:
        if covariance.empty:
            return "cov-empty"
        return f"cov-{len(covariance.index)}x{len(covariance.columns)}"

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:  # noqa: BLE001 -- corrupt state -> start fresh
                return {}
        return {}

    def _load_previous_weights(self) -> pd.Series:
        state = self._load_state()
        weights = state.get("weights", {})
        if not weights:
            return pd.Series(dtype="float64")
        return pd.Series(weights, dtype="float64")

    def _load_previous_active_hedges(self) -> list[str]:
        return list(self._load_state().get("active_hedges", []))

    def _save_state(self, combined: pd.Series, active_hedges: list[str]) -> None:
        state = {
            "weights": {k: float(v) for k, v in combined.items()},
            "active_hedges": active_hedges,
        }
        try:
            directory = os.path.dirname(self.state_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:  # noqa: BLE001 -- persistence must not crash run
            print(f"[strategist] Failed to persist state: {exc}")
