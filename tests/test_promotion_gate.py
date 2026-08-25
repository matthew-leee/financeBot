"""Promotion evidence gate: growth_live must be EARNED, not declared."""

from __future__ import annotations

import csv
import json

import pytest

from src.guardrails import (
    ALLOW_UNEARNED_PROMOTION_ENV,
    PromotionEvidenceError,
    verify_promotion_evidence,
)

_ENV = (ALLOW_UNEARNED_PROMOTION_ENV,)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _write_trades(path, rows) -> str:
    """rows: list of (pnl, slippage_or_None). Legacy trades_log schema."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "ticker", "side", "price", "size", "pnl"]
        )
        writer.writeheader()
        for i, (pnl, _slip) in enumerate(rows):
            writer.writerow(
                {
                    "timestamp": f"2026-01-{(i % 28) + 1:02d}T12:00:00Z",
                    "ticker": "AAPL" if i % 2 else "BTC/USD",
                    "side": "sell" if pnl != 0 else "buy",
                    "price": 100.0,
                    "size": 0.01,
                    "pnl": pnl,
                }
            )
    return str(path)


def test_lower_tier_profiles_are_never_gated(tmp_path):
    # No logs exist at all -- lower tiers must sail through.
    for profile in ("research", "micro_live", "small_live"):
        verify_promotion_evidence(
            profile,
            trades_log_path=str(tmp_path / "missing.csv"),
            fills_log_path=str(tmp_path / "missing2.csv"),
            risk_state_path=str(tmp_path / "no_state.json"),
        )


def test_growth_live_fails_closed_without_any_logs(tmp_path):
    with pytest.raises(PromotionEvidenceError) as exc:
        verify_promotion_evidence(
            "growth_live",
            trades_log_path=str(tmp_path / "missing.csv"),
            fills_log_path=str(tmp_path / "missing2.csv"),
            risk_state_path=str(tmp_path / "no_state.json"),
        )
    assert "no trade history" in str(exc.value)


def test_growth_live_blocked_on_insufficient_trades(tmp_path):
    log = _write_trades(tmp_path / "trades_log.csv", [(5.0, None)] * 10)
    with pytest.raises(PromotionEvidenceError) as exc:
        verify_promotion_evidence("growth_live", trades_log_path=log)
    assert "only 10 executed trades" in str(exc.value)
    assert "(need >= 50)" in str(exc.value)


def test_growth_live_blocked_on_negative_pnl(tmp_path):
    log = _write_trades(tmp_path / "trades_log.csv", [(-1.0, None)] * 60)
    with pytest.raises(PromotionEvidenceError) as exc:
        verify_promotion_evidence("growth_live", trades_log_path=log)
    assert "cumulative net PnL" in str(exc.value)


def test_growth_live_passes_with_qualifying_record(tmp_path, capsys):
    rows = [(2.0, None)] * 55
    log = _write_trades(tmp_path / "trades_log.csv", rows)
    verify_promotion_evidence(
        "growth_live",
        trades_log_path=log,
        risk_state_path=str(tmp_path / "no_state.json"),
    )
    out = capsys.readouterr().out
    assert "Evidence gate PASSED" in out
    assert "55 trades" in out


def test_growth_live_blocked_when_risk_state_is_kill(tmp_path):
    log = _write_trades(tmp_path / "trades_log.csv", [(2.0, None)] * 55)
    state = tmp_path / "risk_state.json"
    state.write_text(json.dumps({"state": "KILL_PROCESS"}), encoding="utf-8")
    with pytest.raises(PromotionEvidenceError) as exc:
        verify_promotion_evidence("growth_live", trades_log_path=log, risk_state_path=str(state))
    assert "KILL_PROCESS" in str(exc.value)


def test_slippage_p95_budget_enforced_from_fills_log(tmp_path):
    # 60 fills, positive PnL, but 6 of them (>5%) slipped 40bps -> p95 >= 40.
    fills = tmp_path / "fills_log.csv"
    with open(fills, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filled_at", "fill_id", "broker_order_id", "symbol", "side",
                "actual_qty", "actual_price", "fees", "realized_pnl",
                "arrival_price", "slippage_bps", "liquidity_flag",
            ],
        )
        writer.writeheader()
        for i in range(60):
            slip = 40.0 if i < 6 else 3.0
            writer.writerow(
                {
                    "filled_at": "2026-02-01T00:00:00Z",
                    "fill_id": f"f{i}",
                    "broker_order_id": f"o{i}",
                    "symbol": "AAPL",
                    "side": "buy",
                    "actual_qty": 0.1,
                    "actual_price": 100.0,
                    "fees": 0.0,
                    "realized_pnl": 0.05 if slip == 3.0 else -0.02,
                    "arrival_price": 100.0,
                    "slippage_bps": slip,
                    "liquidity_flag": "",
                }
            )
    total_pnl = sum([(-0.02)] * 6 + [0.05] * 54)
    assert total_pnl > 0  # precondition: only slippage may fail the gate
    with pytest.raises(PromotionEvidenceError) as exc:
        verify_promotion_evidence(
            "growth_live",
            trades_log_path=str(tmp_path / "missing.csv"),
            fills_log_path=str(fills),
            max_slippage_bps_p95=25.0,
        )
    assert "p95 slippage" in str(exc.value)


def test_operator_override_allows_but_shouts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ALLOW_UNEARNED_PROMOTION_ENV, "true")
    verify_promotion_evidence(
        "growth_live",
        trades_log_path=str(tmp_path / "missing.csv"),
        risk_state_path=str(tmp_path / "no_state.json"),
    )
    out = capsys.readouterr().out
    assert "OVERRIDE ACTIVE" in out
