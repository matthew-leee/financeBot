from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pytest

import config
from src import universe
from src.universe import (
    FULL_CANDIDATE_UNIVERSE,
    UniverseError,
    load_active_universe,
    normalize_symbols,
    resolve_live_universe,
)

_NOW = datetime(2026, 8, 3, tzinfo=timezone.utc)

_UNIVERSE_ENV = (
    "FINANCEBOT_UNIVERSE_FILE",
    "FINANCEBOT_UNIVERSE_STRICT",
    "FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE",
    "FINANCEBOT_UNIVERSE_MAX_AGE_HOURS",
    "FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _UNIVERSE_ENV:
        monkeypatch.delenv(name, raising=False)
    # Default to safe paper mode unless a test flips it.
    monkeypatch.setattr(config, "PAPER", True)


def _write(path, *, symbols, as_of="2026-08-03T00:00:00Z", version=1):
    payload = {"version": version, "as_of": as_of, "symbols": symbols}
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return str(path)


# --- normalization ---------------------------------------------------------

def test_whitespace_and_lowercase_are_normalized():
    assert normalize_symbols([" spy ", "qqQ"]) == ["SPY", "QQQ"]


def test_duplicates_removed_preserving_order():
    assert normalize_symbols(["SPY", "QQQ", "SPY", "spy", "IWM"]) == ["SPY", "QQQ", "IWM"]


def test_btc_and_eth_survive_normalization():
    assert normalize_symbols(["btc/usd", "ETH/USD"]) == ["BTC/USD", "ETH/USD"]


def test_invalid_symbol_raises():
    with pytest.raises(UniverseError):
        normalize_symbols(["SPY", "BAD SYMBOL!"])


# --- candidate set ---------------------------------------------------------

def test_full_candidate_universe_has_47_unique_symbols():
    assert len(FULL_CANDIDATE_UNIVERSE) == 47
    assert len(set(FULL_CANDIDATE_UNIVERSE)) == 47


def test_inverse_safe_list_not_merged_into_targets():
    for inverse in config.INVERSE_SAFE_LIST:
        assert inverse not in FULL_CANDIDATE_UNIVERSE


# --- file loading ----------------------------------------------------------

def test_valid_file_loads_symbols_in_order(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY", "QQQ", "BTC/USD"])
    assert load_active_universe(path, strict=False, now=_NOW) == ["SPY", "QQQ", "BTC/USD"]


def test_invalid_symbol_invalidates_file(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY", "no good"])
    with pytest.raises(UniverseError):
        load_active_universe(path, strict=False, now=_NOW)


def test_bad_version_invalidates_file(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY"], version=2)
    with pytest.raises(UniverseError):
        load_active_universe(path, strict=False, now=_NOW)


def test_missing_explicit_file_strict_raises(tmp_path):
    missing = str(tmp_path / "missing.json")
    with pytest.raises(UniverseError):
        load_active_universe(missing, strict=True, now=_NOW)


def test_stale_paper_file_warns_but_stays_active(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY", "QQQ"], as_of="2000-01-01T00:00:00Z")
    assert load_active_universe(path, strict=False, now=_NOW) == ["SPY", "QQQ"]


def test_stale_strict_file_raises(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY"], as_of="2000-01-01T00:00:00Z")
    with pytest.raises(UniverseError):
        load_active_universe(path, strict=True, now=_NOW)


def test_over_cap_paper_file_truncates_deterministically(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY", "QQQ", "IWM", "DIA"])
    assert load_active_universe(path, strict=False, cap=2, now=_NOW) == ["SPY", "QQQ"]


def test_over_cap_strict_file_raises(tmp_path):
    path = _write(tmp_path / "u.json", symbols=["SPY", "QQQ", "IWM", "DIA"])
    with pytest.raises(UniverseError):
        load_active_universe(path, strict=True, cap=2, now=_NOW)


def test_cap_out_of_range_is_config_error(tmp_path, monkeypatch):
    path = _write(tmp_path / "u.json", symbols=["SPY"])
    monkeypatch.setenv("FINANCEBOT_MAX_LIVE_UNIVERSE_SIZE", "99")
    with pytest.raises(UniverseError):
        load_active_universe(path, strict=False, now=_NOW)


# --- resolve_live_universe -------------------------------------------------

def test_missing_optional_file_nonstrict_uses_config_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", str(tmp_path / "none.json"))
    monkeypatch.setattr(config, "EQUITY_SYMBOLS", ["AAPL", "MSFT"])
    monkeypatch.setattr(config, "CRYPTO_SYMBOLS", ["BTC/USD", "ETH/USD"])
    assert resolve_live_universe(now=_NOW) == ["AAPL", "MSFT", "BTC/USD", "ETH/USD"]


def test_no_file_strict_raises_without_optin(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", str(tmp_path / "none.json"))
    monkeypatch.setattr(config, "PAPER", False)  # live => strict
    with pytest.raises(UniverseError):
        resolve_live_universe(now=_NOW)


def test_no_file_strict_optin_uses_default_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", str(tmp_path / "none.json"))
    monkeypatch.setattr(config, "PAPER", False)
    monkeypatch.setenv("FINANCEBOT_USE_DEFAULT_LIVE_UNIVERSE", "true")
    resolved = resolve_live_universe(now=_NOW)
    assert len(resolved) == 47


def test_missing_optional_file_strict_but_present_default_path(tmp_path, monkeypatch):
    path = _write(tmp_path / "active.json", symbols=["SPY", "QQQ"])
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", path)
    monkeypatch.setattr(config, "PAPER", False)  # strict, but file exists + fresh
    assert resolve_live_universe(now=_NOW) == ["SPY", "QQQ"]


def test_env_precedence_is_deterministic(tmp_path, monkeypatch):
    env_file = _write(tmp_path / "env.json", symbols=["SPY"])
    default_file = _write(tmp_path / "default.json", symbols=["QQQ"])
    explicit_file = _write(tmp_path / "explicit.json", symbols=["IWM"])
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", default_file)
    monkeypatch.setenv("FINANCEBOT_UNIVERSE_FILE", env_file)
    # Explicit arg beats env; env beats config default.
    assert load_active_universe(explicit_file, strict=False, now=_NOW) == ["IWM"]
    assert resolve_live_universe(now=_NOW) == ["SPY"]


def test_malformed_file_nonstrict_falls_back(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    with io.open(bad, "w", encoding="utf-8") as fh:
        fh.write("{ not valid json")
    monkeypatch.setattr(config, "ACTIVE_UNIVERSE_PATH", str(bad))
    monkeypatch.setattr(config, "EQUITY_SYMBOLS", ["AAPL"])
    monkeypatch.setattr(config, "CRYPTO_SYMBOLS", ["BTC/USD"])
    assert resolve_live_universe(now=_NOW) == ["AAPL", "BTC/USD"]
