from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import portfolio_manager.pipeline_adapter as adapter_module
from portfolio_manager.pipeline_adapter import (
    CALIBRATION_FORMAT,
    load_pipeline_signal,
)
from web.signal_ledger import SignalLedger

BAR_TS = 1_784_790_000


def _canonical_hash(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _calibration() -> dict:
    history = [
        {
            "bar_ts": BAR_TS - (252 - index) * 86_400,
            "raw_score": index / 10,
        }
        for index in range(252)
    ]
    body = {
        "version": CALIBRATION_FORMAT,
        "window_bars": 500,
        "history": history,
    }
    return {**body, "history_sha256": _canonical_hash(body)}


def _payload() -> dict:
    fingerprint = "a" * 64
    return {
        "format": "alphamaster_signal_simulation_v3",
        "run_id": "run_20260723T235959Z_12345678",
        "strategy_fingerprint": fingerprint,
        "market_source": "akshare_sina_hfq_ohlcv",
        "market_data_sha256": "b" * 64,
        "signal_input_sha256": "c" * 64,
        "symbol": "000001",
        "timeframe": "1d",
        "last_bar_ts": BAR_TS,
        "raw_signal": {
            "state": "ok",
            "position_raw": 0.4,
            "position": 0.4,
            "factor_value_raw": 0.4236123456789,
            "factor_value": 0.423612,
            "strength_raw": 0.4,
            "strength": 0.4,
        },
        "calibration": _calibration(),
        "lifecycle_event": {
            "symbol": "000001",
            "bar_ts": BAR_TS,
            "strategy_fingerprint": fingerprint,
            "action": "BUY",
        },
    }


def _expectation(**changes) -> adapter_module._PipelineSignalExpectation:
    values = {
        "run_id": "run_20260723T235959Z_12345678",
        "symbol": "000001",
        "strategy_fingerprint": "a" * 64,
        "market_data_sha256": "b" * 64,
        "signal_input_sha256": "c" * 64,
        "last_bar_ts": BAR_TS,
    }
    values.update(changes)
    return adapter_module._PipelineSignalExpectation(**values)


def test_pipeline_payload_maps_to_model_signal() -> None:
    signal = adapter_module._model_signal_from_pipeline_payload(
        _payload(),
        expected=_expectation(),
    )

    assert signal.run_id == "run_20260723T235959Z_12345678"
    assert signal.symbol == "000001"
    assert signal.bar_ts == BAR_TS
    assert signal.session_date == "2026-07-23"
    assert signal.timeframe == "1d"
    assert signal.market_source == "akshare_sina_hfq_ohlcv"
    assert signal.raw_score == pytest.approx(0.4236123456789)
    assert signal.requested_exposure == pytest.approx(0.4)
    assert len(signal.history_scores) == 252
    assert signal.model_version == "a" * 64
    assert signal.data_version == "b" * 64
    assert signal.model_exit is False


def test_negative_position_becomes_long_only_zero_exposure() -> None:
    payload = _payload()
    payload["raw_signal"]["position_raw"] = -0.8
    payload["raw_signal"]["position"] = -0.8
    payload["raw_signal"]["strength_raw"] = 0.8
    payload["raw_signal"]["strength"] = 0.8

    signal = adapter_module._model_signal_from_pipeline_payload(
        payload,
        expected=_expectation(),
    )

    assert signal.requested_exposure == 0.0


def test_stop_loss_forces_exit_even_if_model_position_is_positive() -> None:
    payload = _payload()
    payload["lifecycle_event"]["action"] = "STOP_LOSS"

    signal = adapter_module._model_signal_from_pipeline_payload(
        payload,
        expected=_expectation(),
    )

    assert signal.model_exit is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "v2", "simulation_v3"),
        ("market_data_sha256", "c" * 64, "预期"),
        ("signal_input_sha256", "d" * 64, "预期"),
        ("run_id", "run_20260723T235959Z_deadbeef", "预期"),
        ("market_source", "tongdaxin", "预期"),
        ("timeframe", "1h", "预期"),
    ],
)
def test_identity_mismatch_fails_closed(field, value, message) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(),
        )


def test_self_reported_hash_cannot_override_expected_identity() -> None:
    payload = _payload()
    payload["market_data_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="预期"):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(market_data_sha256="b" * 64),
        )


def test_calibration_content_and_hash_are_bound() -> None:
    payload = _payload()
    payload["calibration"]["history"][0]["raw_score"] = 999.0

    with pytest.raises(ValueError, match="history_sha256"):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(),
        )


@pytest.mark.parametrize(
    ("window_bars", "history_count", "message"),
    [
        (200, 252, "500"),
        (500, 20, "252"),
    ],
)
def test_calibration_contract_is_exact(
    window_bars,
    history_count,
    message,
) -> None:
    payload = _payload()
    history = payload["calibration"]["history"][-history_count:]
    body = {
        "version": CALIBRATION_FORMAT,
        "window_bars": window_bars,
        "history": history,
    }
    payload["calibration"] = {
        **body,
        "history_sha256": _canonical_hash(body),
    }

    with pytest.raises(ValueError, match=message):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(),
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("last_bar_ts",), BAR_TS + 0.5, "整数"),
        (("raw_signal", "position_raw"), "0.4", "数值"),
        (("raw_signal", "position_raw"), 3.0, r"\[-1, 1\]"),
        (("raw_signal", "strength_raw"), True, "数值"),
    ],
)
def test_implicit_numeric_coercion_is_rejected(path, value, message) -> None:
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(),
        )


def test_daily_timestamp_must_be_shanghai_session_close() -> None:
    payload = _payload()
    payload["last_bar_ts"] = BAR_TS - 3_600
    payload["lifecycle_event"]["bar_ts"] = BAR_TS - 3_600

    with pytest.raises(ValueError, match="15:00"):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(last_bar_ts=BAR_TS - 3_600),
        )


def test_bar_timestamp_cannot_change_without_frozen_market_identity() -> None:
    payload = _payload()
    saturday_ts = BAR_TS + 2 * 86_400
    payload["last_bar_ts"] = saturday_ts
    payload["lifecycle_event"]["bar_ts"] = saturday_ts

    with pytest.raises(ValueError, match="冻结行情"):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(),
        )


def test_missing_calibration_fails_closed() -> None:
    payload = _payload()
    del payload["calibration"]

    with pytest.raises(ValueError, match="calibration"):
        adapter_module._model_signal_from_pipeline_payload(
            payload,
            expected=_expectation(),
        )


def _write_verified_run(tmp_path: Path) -> Path:
    run_id = "run_20260723T235959Z_12345678"
    symbol = "000001"
    run_dir = tmp_path / "local_runs" / run_id
    post_dir = run_dir / "postprocess"
    report_dir = post_dir / "backtest_output"
    report_dir.mkdir(parents=True)

    training_data_sha256 = "e" * 64
    manifest = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": "D1",
        "local_source": "ashare_akshare_sina_hfq",
        "data_sha256": training_data_sha256,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    strategy = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": "D1",
        "data_sha256": training_data_sha256,
        "vocab_version": "v-test",
        "formula": [1, 2],
    }
    strategy_path = post_dir / "published_strategy.json"
    strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
    fingerprint = _canonical_hash(
        {
            "formula": [1, 2],
            "vocab_version": "v-test",
            "symbol": symbol,
            "timeframe": "D1",
        }
    )

    market_contract = {
        "source": "akshare_sina_hfq_ohlcv",
        "symbol": symbol,
        "timeframe": "D1",
        "bars": [
            [
                BAR_TS - (751 - index) * 86_400,
                10.0,
                11.0,
                9.0,
                10.5,
                1_000 + index,
            ]
            for index in range(752)
        ],
    }
    market_data_sha256 = _canonical_hash(market_contract)
    base_payload = _payload()
    base_payload["strategy_fingerprint"] = fingerprint
    base_payload["market_data_sha256"] = market_data_sha256
    input_body = {
        "format": "alphamaster_signal_input_v1",
        "run_id": run_id,
        "strategy_fingerprint": fingerprint,
        "training_timeframe": "D1",
        "market_source": "akshare_sina_hfq_ohlcv",
        "market_contract": market_contract,
        "market_data_sha256": market_data_sha256,
        "market_data_evidence": {
            "history_response_sha256": "a" * 64,
            "factor_response_sha256": "b" * 64,
        },
        "symbol": symbol,
        "timeframe": "1d",
        "bars_used": 752,
        "last_bar_ts": BAR_TS,
        "last_close": 10.5,
        "raw_signal": base_payload["raw_signal"],
        "calibration": base_payload["calibration"],
    }
    input_sha256 = _canonical_hash(input_body)
    signal_input_path = post_dir / "signal_input.json"
    signal_input_path.write_text(
        json.dumps(
            {
                **input_body,
                "input_sha256": input_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ledger = SignalLedger(post_dir / "signal_simulation.sqlite3")
    watch_id = (
        f"pipeline:{run_id}:akshare_sina_hfq_ohlcv:{symbol}:1d:{fingerprint[:12]}"
    )
    event, created = ledger.process_bar(
        watch_id=watch_id,
        source="akshare_sina_hfq_ohlcv",
        symbol=symbol,
        timeframe="1d",
        strategy_name=strategy_path.stem,
        strategy_fingerprint=fingerprint,
        bar_ts=BAR_TS,
        price=10.5,
        raw_position=base_payload["raw_signal"]["position_raw"],
        factor_value=base_payload["raw_signal"]["factor_value_raw"],
        strength=base_payload["raw_signal"]["strength_raw"],
        rebalance_delta=0.10,
        minimum_exposure=0.05,
        stop_loss_pct=-0.02,
        take_profit_pct=0.04,
        take_profit_remaining_ratio=0.50,
    )
    assert created is True
    if event.should_push:
        ledger.record_delivery(
            event.event_id,
            "SKIPPED",
            "流水线虚拟信号验证：未发送飞书",
        )
    lifecycle_event = ledger.get_event(event.event_id)
    assert lifecycle_event is not None
    signal_payload = {
        **base_payload,
        "lifecycle_event": lifecycle_event,
        "strategy_file": str(strategy_path),
        "signal_input_path": str(signal_input_path.relative_to(tmp_path)).replace(
            "\\", "/"
        ),
        "signal_input_sha256": input_sha256,
    }
    signal_path = post_dir / "signal_simulation.json"
    signal_path.write_text(
        json.dumps(signal_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    report = {
        "symbol": symbol,
        "timeframe": "D1",
        "data_sha256": training_data_sha256,
    }
    report_path = report_dir / "multi_factor_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    state = {
        "format": "alphamaster_a_share_pipeline_v2",
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": "D1",
        "status": "READY",
        "attempts": 1,
        "stages": {
            "training": {"status": "READY"},
            "backtest": {
                "status": "READY",
                "report_path": str(report_path.relative_to(tmp_path)).replace(
                    "\\",
                    "/",
                ),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "data_sha256": training_data_sha256,
            },
            "signal": {
                "status": "READY",
                "output_path": str(signal_path.relative_to(tmp_path)).replace(
                    "\\",
                    "/",
                ),
                "output_sha256": hashlib.sha256(signal_path.read_bytes()).hexdigest(),
                "signal_input_sha256": input_sha256,
                "strategy_fingerprint": fingerprint,
                "market_data_sha256": market_data_sha256,
            },
        },
    }
    (run_dir / "pipeline_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    return run_dir


def test_verified_loader_builds_identity_from_full_pipeline_chain(tmp_path) -> None:
    signal = load_pipeline_signal(
        _write_verified_run(tmp_path),
        project_root=tmp_path,
    )

    assert signal.run_id == "run_20260723T235959Z_12345678"
    assert signal.symbol == "000001"
    assert signal.raw_score == pytest.approx(0.4236123456789)
    assert signal.model_version != "a" * 64


def test_verified_loader_rejects_payload_and_state_rewritten_together(
    tmp_path,
) -> None:
    run_dir = _write_verified_run(tmp_path)
    signal_path = run_dir / "postprocess" / "signal_simulation.json"
    payload = json.loads(signal_path.read_text(encoding="utf-8"))
    payload["raw_signal"]["factor_value_raw"] = 999_999.0
    signal_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = run_dir / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"]["signal"]["output_sha256"] = hashlib.sha256(
        signal_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="冻结输入"):
        load_pipeline_signal(run_dir, project_root=tmp_path)


def test_verified_loader_rejects_forged_lifecycle_action_and_state_hash(
    tmp_path,
) -> None:
    run_dir = _write_verified_run(tmp_path)
    signal_path = run_dir / "postprocess" / "signal_simulation.json"
    payload = json.loads(signal_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_event"]["action"] == "BUY"
    payload["lifecycle_event"]["action"] = "STOP_LOSS"
    signal_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = run_dir / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"]["signal"]["output_sha256"] = hashlib.sha256(
        signal_path.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="SQLite"):
        load_pipeline_signal(run_dir, project_root=tmp_path)
