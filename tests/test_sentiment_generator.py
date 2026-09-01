"""Sentiment generator robustness: sanitize / self-correct / salvage layers."""

from __future__ import annotations

import json

import pytest

import config
from generate_sentiment import (
    _extract_json,
    _salvage_entries,
    _sanitize_json,
    generate,
)

CONTEXTS = {"SPY": {}, "QQQ": {}, "AAPL": {}}
GOOD = json.dumps(
    {
        "SPY": {"score": 5.2, "summary": "flat"},
        "QQQ": {"score": 6.4, "summary": "up"},
        "AAPL": {"score": 4.1, "summary": "down"},
    }
)


class FakeResponse(dict):
    """Shape-compatible stand-in for a chat-completions payload."""


def _payload(content: str) -> FakeResponse:
    return FakeResponse(choices=[{"message": {"content": content}}])


def test_sanitize_repairs_trailing_commas_exact_failure_shape():
    # Reproduces the live failure: Gemini output with trailing commas ->
    # "Expecting ',' delimiter" under strict json.loads.
    bad = """{
  "SPY": {"score": 5.2, "summary": "flat",},
  "QQQ": {"score": 5.2,"summary": "x",},
}"""
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad)
    out = _extract_json(bad)
    assert set(out) == {"SPY", "QQQ"}


def test_self_correction_retry_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "dummy-key-for-tests")
    monkeypatch.setattr(
        "generate_sentiment._momentum_context",
        lambda sym, lookback_days: {"last_close": 100.0, "ret_5d_pct": 0.5,
                                    "ret_20d_pct": 1.0, "ann_vol_pct": 12.0},
    )
    calls = []

    def transport(url, *, headers, body):
        calls.append(body["messages"])
        if len(calls) == 1:
            return _payload("{ oops not json")
        return _payload(GOOD)

    out = str(tmp_path / "ds.json")
    report = generate(universe=list(CONTEXTS), out_path=out, transport=transport)
    assert len(calls) == 2
    assert report["SPY"]["score"] == 5.2
    # Self-correction message appended on the retry round.
    assert "invalid JSON" in calls[1][-1]["content"]
    on_disk = json.loads(open(out).read())
    assert on_disk == report


def test_salvage_yields_partial_report_from_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "dummy-key-for-tests")
    monkeypatch.setattr(
        "generate_sentiment._momentum_context",
        lambda sym, lookback_days: {"last_close": 100.0, "ret_5d_pct": 0.5,
                                    "ret_20d_pct": 1.0, "ann_vol_pct": 12.0},
    )
    # Two independent JSON objects + prose: slicing yields "Extra data"
    # (uncorrectable), so all attempts must fail before salvage kicks in.
    always_bad = (
        'noise {"AAPL": {"score": 7.0, "summary": "up"}} middle '
        '{"QQQ": {"score": 3.5, "summary": "down"}} end'
    )
    transport_calls = []

    def transport(url, *, headers, body):
        transport_calls.append(1)
        return _payload(always_bad)

    out = str(tmp_path / "ds.json")
    report = generate(universe=list(CONTEXTS), out_path=out, transport=transport)
    assert len(transport_calls) == int(config.SENTIMENT_LLM_MAX_ATTEMPTS)
    # Salvage accepted the one recoverable entry; others -> neutral-pass.
    assert set(report) == {"AAPL", "QQQ"}  # both fragments salvaged
    assert report["AAPL"]["score"] == 7.0
    assert report["QQQ"]["score"] == 3.5
    assert json.loads(open(out).read()) == report


def test_total_garbage_keeps_previous_file_and_fails(tmp_path):
    prior = {"SPY": {"date": "2026-08-01", "score": 9.9, "summary": "old"}}
    out = tmp_path / "ds.json"
    out.write_text(json.dumps(prior), encoding="utf-8")

    def transport(url, *, headers, body):
        return _payload("### no json at all ###")

    with pytest.raises(RuntimeError):
        generate(universe=list(CONTEXTS), out_path=str(out), transport=transport)

    assert json.loads(out.read_text()) == prior  # fail-closed preserved


def test_missing_key_fails_before_network(tmp_path):
    import os

    saved = os.environ.pop(config.LLM_API_KEY_ENV, None)
    try:
        def transport(url, **kw):  # pragma: no cover - must never be reached
            raise AssertionError("network call attempted without key")

        with pytest.raises(RuntimeError):
            generate(universe=["SPY"], out_path=str(tmp_path / "x.json"), transport=transport)
    finally:
        if saved is not None:
            os.environ[config.LLM_API_KEY_ENV] = saved


def test_request_body_has_headroom_and_reasoning_off(tmp_path, monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "dummy-key-for-tests")
    monkeypatch.setattr(
        "generate_sentiment._momentum_context",
        lambda sym, lookback_days: {"last_close": 1.0, "ret_5d_pct": 0.0,
                                    "ret_20d_pct": 0.0, "ann_vol_pct": 10.0},
    )
    captured: dict = {}

    def transport(url, *, headers, body):
        captured.update(body)
        return _payload(GOOD)

    generate(universe=list(CONTEXTS), out_path=str(tmp_path / "ds.json"), transport=transport)
    assert captured["max_tokens"] >= 8000
    assert captured["reasoning_effort"] == "low"
    assert "glm-5.3-flash" in captured["model"]
