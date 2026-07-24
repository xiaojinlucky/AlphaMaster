from __future__ import annotations

import json

import pytest

import web.a_share_pipeline as pipeline_module
from web.a_share_pipeline import ASharePipelineManager
from web.data_sources.sina_hfq_daily import (
    DailyBar,
    SinaHfqSnapshot,
)


def test_d1_pipeline_uses_training_source_hfq_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    bars = tuple(
        DailyBar(
            close_ts=1_784_790_000 - (751 - index) * 86_400,
            open=500.0 + index / 100,
            high=505.0 + index / 100,
            low=498.0 + index / 100,
            close=502.11 + index / 100,
            volume=19_123_849 + index,
        )
        for index in range(752)
    )
    snapshot = SinaHfqSnapshot(
        symbol="000333",
        bars=bars,
        history_response_sha256="a" * 64,
        factor_response_sha256="b" * 64,
    )

    class FakeSource:
        def fetch(self, symbol: str, *, n: int, drop_forming: bool):
            assert symbol == "000333"
            assert n == 752
            assert drop_forming is True
            return snapshot

    monkeypatch.setattr(
        pipeline_module,
        "SinaHfqDailySource",
        FakeSource,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_load_strategy_meta",
        lambda _path: {"formula": [1, 2], "fingerprint": "c" * 64},
    )

    def fake_evaluate(_formula, _raw, timestamps, *, window_bars, history_count):
        assert len(timestamps) == 752
        assert window_bars == 500
        assert history_count == 252
        result = {
            "state": "ok",
            "direction": "FLAT",
            "strength_raw": 0.04996,
            "strength": 0.05,
            "position_raw": 0.04996,
            "position": 0.05,
            "factor_value_raw": 0.4236123456789,
            "factor_value": 0.4236,
            "threshold": 0.05,
            "bars_used": 500,
            "message": "",
        }
        calibration = {
            "version": "alphamaster_rolling_factor_calibration_v1",
            "window_bars": 500,
            "history": [
                {"bar_ts": timestamps[index], "raw_score": float(index)}
                for index in range(20)
            ],
            "history_sha256": "d" * 64,
        }
        return result, calibration

    monkeypatch.setattr(
        pipeline_module,
        "evaluate_with_rolling_calibration",
        fake_evaluate,
    )
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    manager = ASharePipelineManager(local_runs_root=tmp_path)

    payload = manager._run_signal(
        run_id="run_20260723T235959Z_12345678",
        symbol="000333",
        timeframe="D1",
        strategy_file=tmp_path / "best_000333.json",
    )

    assert payload["format"] == "alphamaster_signal_simulation_v3"
    assert payload["market_source"] == "akshare_sina_hfq_ohlcv"
    assert payload["timeframe"] == "1d"
    assert payload["last_bar_ts"] == snapshot.last_bar_ts
    assert payload["market_data_sha256"] == snapshot.market_data_sha256
    assert payload["market_data_evidence"] == {
        "history_response_sha256": "a" * 64,
        "factor_response_sha256": "b" * 64,
    }
    assert payload["lifecycle_event"]["source"] == "akshare_sina_hfq_ohlcv"
    assert payload["lifecycle_event"]["action"] == "HOLD"
    assert payload["lifecycle_event"]["requested_exposure"] == 0.0
    assert payload["lifecycle_event"]["raw_position"] == pytest.approx(0.04996)
    assert payload["signal_input_sha256"]

    output = (
        tmp_path
        / "run_20260723T235959Z_12345678"
        / "postprocess"
        / "signal_simulation.json"
    )
    assert json.loads(output.read_text(encoding="utf-8"))["market_source"] == (
        "akshare_sina_hfq_ohlcv"
    )


def test_retry_reuses_frozen_signal_input_instead_of_new_market_day(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    test_d1_pipeline_uses_training_source_hfq_snapshot(monkeypatch, tmp_path)
    manager = ASharePipelineManager(local_runs_root=tmp_path)
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_module,
        "_load_strategy_meta",
        lambda _path: {"formula": [1, 2], "fingerprint": "c" * 64},
    )

    class MustNotFetch:
        def fetch(self, *_args, **_kwargs):
            raise AssertionError("重试不得重新选择最新 K 线")

    monkeypatch.setattr(pipeline_module, "SinaHfqDailySource", MustNotFetch)
    payload = manager._run_signal(
        run_id="run_20260723T235959Z_12345678",
        symbol="000333",
        timeframe="D1",
        strategy_file=tmp_path / "best_000333.json",
    )

    assert payload["last_bar_ts"] == 1_784_790_000
