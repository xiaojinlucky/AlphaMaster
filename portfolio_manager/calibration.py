"""按每个历史时点自己的固定窗口重放因子，生成可审计校准历史。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from strategy_manager.live_signal import MIN_BARS, evaluate_signal

CALIBRATION_FORMAT = "alphamaster_rolling_factor_calibration_v1"


def _positive_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} 必须至少为 {minimum}")
    return number


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def evaluate_with_rolling_calibration(
    formula: list[int],
    raw_dict: Mapping[str, Any],
    bar_timestamps: Sequence[int],
    *,
    window_bars: int = 500,
    history_count: int = 252,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """用固定长度历史窗口逐日重放，当前值与历史值保持同一精度。"""
    window = _positive_integer("window_bars", window_bars, minimum=MIN_BARS)
    count = _positive_integer("history_count", history_count, minimum=20)
    timestamps = tuple(
        _positive_integer(f"bar_timestamps[{index}]", value)
        for index, value in enumerate(bar_timestamps)
    )
    if any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("bar_timestamps 必须严格递增")

    required = window + count
    if len(timestamps) < required:
        raise ValueError(f"滚动校准至少需要 {required} 根已收盘 K 线")
    total = len(timestamps)
    normalized_raw: dict[str, Any] = {}
    for name, tensor in raw_dict.items():
        if (
            not hasattr(tensor, "ndim")
            or tensor.ndim != 2
            or int(tensor.shape[0]) != 1
            or int(tensor.shape[1]) != total
        ):
            raise ValueError(f"raw_dict[{name}] 必须是 [1, {total}] 张量")
        normalized_raw[str(name)] = tensor

    def evaluate_window(end_exclusive: int) -> dict[str, Any]:
        start = end_exclusive - window
        sliced = {
            name: tensor[:, start:end_exclusive]
            for name, tensor in normalized_raw.items()
        }
        result = evaluate_signal(formula, sliced)
        if result.get("state") != "ok":
            raise RuntimeError(
                "滚动校准因子计算失败: " + str(result.get("message") or result)
            )
        raw_score = result.get("factor_value_raw")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
        ):
            raise RuntimeError("滚动校准没有产生有限的原始因子值")
        return result

    history: list[dict[str, object]] = []
    first_end = total - count
    for end_exclusive in range(first_end, total):
        result = evaluate_window(end_exclusive)
        history.append(
            {
                "bar_ts": timestamps[end_exclusive - 1],
                "raw_score": float(result["factor_value_raw"]),
            }
        )

    current = evaluate_window(total)
    calibration_body = {
        "version": CALIBRATION_FORMAT,
        "window_bars": window,
        "history": history,
    }
    calibration = {
        **calibration_body,
        "history_sha256": _canonical_sha256(calibration_body),
    }
    return current, calibration


__all__ = ["CALIBRATION_FORMAT", "evaluate_with_rolling_calibration"]
