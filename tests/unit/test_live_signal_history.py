from __future__ import annotations

import math

import pytest
import torch

import strategy_manager.live_signal as live_signal


class _FakeFeatureEngineer:
    @staticmethod
    def compute_features(_raw):
        return torch.zeros((1, 1, live_signal.MIN_BARS), dtype=torch.float32)


class _FakeVM:
    @staticmethod
    def execute(_formula, _features):
        return torch.linspace(
            -1.0,
            1.0000004768371582,
            live_signal.MIN_BARS,
        ).reshape(1, -1)


class _ThresholdVM:
    @staticmethod
    def execute(_formula, _features):
        values = torch.zeros(
            (1, live_signal.MIN_BARS),
            dtype=torch.float64,
        )
        values[0, -1] = math.atanh(0.04996)
        return values


def test_evaluate_signal_preserves_full_precision_for_portfolio_calibration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        live_signal,
        "MT5FeatureEngineer",
        _FakeFeatureEngineer,
    )
    monkeypatch.setattr(live_signal, "_VM", _FakeVM())

    result = live_signal.evaluate_signal(
        [1],
        {"close": torch.ones((1, live_signal.MIN_BARS))},
    )

    assert result["state"] == "ok"
    assert result["factor_value"] == 1.0
    assert result["factor_value_raw"] == 1.0000004768371582
    assert result["position_raw"] == pytest.approx(math.tanh(1.0000004768371582))
    assert result["strength_raw"] == pytest.approx(abs(result["position_raw"]))
    assert "calibration_history" not in result


def test_rounding_cannot_turn_subthreshold_position_into_entry(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        live_signal,
        "MT5FeatureEngineer",
        _FakeFeatureEngineer,
    )
    monkeypatch.setattr(live_signal, "_VM", _ThresholdVM())

    result = live_signal.evaluate_signal(
        [1],
        {"close": torch.ones((1, live_signal.MIN_BARS))},
    )

    assert result["direction"] == live_signal.DIR_FLAT
    assert result["position_raw"] == pytest.approx(0.04996)
    assert result["position"] == 0.05
