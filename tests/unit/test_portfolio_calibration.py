from __future__ import annotations

import hashlib
import json

import pytest
import torch

import portfolio_manager.calibration as calibration_module
from portfolio_manager.calibration import evaluate_with_rolling_calibration


def _raw(total: int) -> dict[str, torch.Tensor]:
    values = torch.arange(total, dtype=torch.float64).reshape(1, -1)
    return {
        "open": values,
        "high": values + 1,
        "low": values - 1,
        "close": values,
        "volume": values + 100,
        "time": values + 1_000,
    }


def _hash(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_each_history_point_uses_its_own_fixed_window(monkeypatch) -> None:
    observed_windows: list[tuple[float, float]] = []

    def fake_evaluate(_formula, raw):
        close = raw["close"]
        observed_windows.append((float(close[0, 0]), float(close[0, -1])))
        raw_score = float(close[0, -1]) + 0.123456789
        return {
            "state": "ok",
            "factor_value_raw": raw_score,
            "factor_value": round(raw_score, 6),
            "position_raw": 0.5,
            "position": 0.5,
            "strength_raw": 0.5,
            "strength": 0.5,
        }

    monkeypatch.setattr(calibration_module, "evaluate_signal", fake_evaluate)
    total = 220
    timestamps = tuple(range(10_000, 10_000 + total))

    current, calibration = evaluate_with_rolling_calibration(
        [1],
        _raw(total),
        timestamps,
        window_bars=200,
        history_count=20,
    )

    assert len(observed_windows) == 21
    assert observed_windows[0] == (0.0, 199.0)
    assert observed_windows[-2] == (19.0, 218.0)
    assert observed_windows[-1] == (20.0, 219.0)
    assert calibration["history"][0] == {
        "bar_ts": timestamps[199],
        "raw_score": pytest.approx(199.123456789),
    }
    assert calibration["history"][-1]["bar_ts"] == timestamps[218]
    assert current["factor_value_raw"] == pytest.approx(219.123456789)
    body = {
        "version": calibration["version"],
        "window_bars": calibration["window_bars"],
        "history": calibration["history"],
    }
    assert calibration["history_sha256"] == _hash(body)


def test_rolling_calibration_rejects_short_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="至少需要"):
        evaluate_with_rolling_calibration(
            [1],
            _raw(219),
            tuple(range(1, 220)),
            window_bars=200,
            history_count=20,
        )

    timestamps = list(range(1, 221))
    timestamps[-1] = timestamps[-2]
    with pytest.raises(ValueError, match="严格递增"):
        evaluate_with_rolling_calibration(
            [1],
            _raw(220),
            timestamps,
            window_bars=200,
            history_count=20,
        )
