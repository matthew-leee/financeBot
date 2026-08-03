"""Phase 0: freeze the legacy public contract.

Every symbol listed in Architecture.txt section 1 (Non-Negotiable Legacy
Compatibility) must remain importable. If any of these break, the dual-horizon
migration has regressed a legacy contract and existing tooling will break.
"""

from __future__ import annotations

import inspect


def test_config_legacy_constants_present() -> None:
    import config

    for name in (
        "MAX_POSITION_SIZE",
        "DAILY_LOSS_LIMIT",
        "MAX_OPEN_POSITIONS",
        "BUY_THRESHOLD",
        "SELL_THRESHOLD",
        "SENTIMENT_MIN_SCORE",
        "INVERSE_SAFE_LIST",
        "HEDGE_CORR_LOOKBACK_DAYS",
        "EQUITY_SYMBOLS",
        "CRYPTO_SYMBOLS",
    ):
        assert hasattr(config, name), f"config.{name} missing"


def test_data_legacy_contract() -> None:
    from src import data

    assert isinstance(data.FEATURE_COLUMNS, list)
    for fn in ("fetch_bars", "build_features", "make_dataset"):
        assert callable(getattr(data, fn))


def test_sentiment_legacy_contract() -> None:
    from src import sentiment

    for fn in (
        "load_sentiment",
        "get_score",
        "is_trade_allowed",
        "daily_returns",
        "select_hedge_asset",
    ):
        assert callable(getattr(sentiment, fn))


def test_execution_legacy_contract() -> None:
    from src import execution

    for fn in ("run", "process_symbol", "decide_action", "_execute_buy", "_pivot_to_hedge"):
        assert callable(getattr(execution, fn))


def test_fifo_and_trade_log_contract() -> None:
    from src.fifo import FIFOInventory
    from src.trade_log import TRADE_LOG_COLUMNS, append_trade

    assert inspect.isclass(FIFOInventory)
    assert callable(append_trade)
    assert TRADE_LOG_COLUMNS[0] == "timestamp"


def test_guardrails_legacy_contract() -> None:
    from src import guardrails

    for fn in (
        "clamp_position_size",
        "can_open_new_position",
        "record_equity_anchor",
        "check_circuit_breaker",
    ):
        assert callable(getattr(guardrails, fn))


def test_model_io_legacy_contract() -> None:
    from src import model_io

    assert issubclass(model_io.ModelArtifactError, Exception)
    for fn in ("verify_model_artifacts", "load_model", "save_model"):
        assert callable(getattr(model_io, fn))
    assert inspect.isclass(model_io.LoadedModel)


def test_backtest_legacy_contract() -> None:
    import backtest

    for fn in (
        "simulate",
        "merge_sentiment",
        "generate_historical_sentiment",
        "load_historical_sentiment",
    ):
        assert callable(getattr(backtest, fn))
