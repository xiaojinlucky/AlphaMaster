"""执行 AlphaMaster 的下一开盘时钟固定样本。"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_fixture, verify_repo_paths_clean, write_result

from data_pipeline.data_manager import MT5DataManager
from model_core.engine import AlphaEngine


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = (
        left_centered.square().mean().sqrt()
        * right_centered.square().mean().sqrt()
    )
    if float(denominator) < 1e-12:
        raise RuntimeError("固定 IC 探针不能使用常量序列")
    return float(
        (left_centered * right_centered).mean() / denominator
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture, fixture_sha256 = load_fixture(args.fixture)
    verify_repo_paths_clean(
        REPO_ROOT,
        [
            "data_pipeline/data_manager.py",
            "model_core/backtest.py",
            "model_core/engine.py",
        ],
    )
    clock = fixture["clock"]
    open_prices = [float(value) for value in clock["open_prices"]]
    expected = [float(value) for value in clock["expected_target_returns"]]
    if len(open_prices) < 6 or len(expected) != len(open_prices):
        raise RuntimeError("固定时钟样本至少需要 6 个开盘价")
    valid_target_count = int(clock["valid_target_count"])
    if valid_target_count != len(open_prices) - 2:
        raise RuntimeError("有效收益数量必须等于开盘价数量减 2")

    actual_tensor = MT5DataManager._compute_target_ret(
        torch.tensor([open_prices], dtype=torch.float32)
    )
    actual = [float(value) for value in actual_tensor[0].tolist()]
    tolerance = 1e-6
    formula_matches = all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
        for left, right in zip(actual, expected, strict=True)
    )
    signal_step = int(clock["signal_step"])
    execution_open_step = int(clock["execution_open_step"])
    return_end_open_step = int(clock["return_end_open_step"])
    explicit_return = math.log(
        open_prices[return_end_open_step] / open_prices[execution_open_step]
    )
    next_open_interval_matches = math.isclose(
        actual[signal_step],
        explicit_return,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    if not formula_matches or not next_open_interval_matches:
        raise RuntimeError(
            "AlphaMaster 固定样本未满足 t 日信号对应下一根开盘收益区间"
        )

    diagnostic_factor = actual_tensor.clone()
    reported_ic, _ = AlphaEngine._compute_ic(
        diagnostic_factor,
        actual_tensor,
    )
    pnl_clock_ic = _pearson(
        diagnostic_factor[0, :valid_target_count],
        actual_tensor[0, :valid_target_count],
    )
    one_extra_bar_ic_valid_only = _pearson(
        diagnostic_factor[0, : valid_target_count - 1],
        actual_tensor[0, 1:valid_target_count],
    )
    one_extra_bar_ic_with_padding = _pearson(
        diagnostic_factor[0, :-1],
        actual_tensor[0, 1:],
    )
    reported_ic_value = float(reported_ic)
    ic_matches_pnl_clock = math.isclose(
        reported_ic_value,
        pnl_clock_ic,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    ic_extra_shift_reproduced = math.isclose(
        reported_ic_value,
        one_extra_bar_ic_with_padding,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    padding_pollution_reproduced = not math.isclose(
        reported_ic_value,
        one_extra_bar_ic_valid_only,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    if (
        ic_matches_pnl_clock
        or not ic_extra_shift_reproduced
        or not padding_pollution_reproduced
    ):
        raise RuntimeError("AlphaMaster IC 额外错位反例没有按预期复现")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    write_result(
        args.output,
        {
            "engine": "alphamaster",
            "engine_version": head,
            "scope": "target_return_clock_runtime",
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "clock_contract": (
                "target_ret[t] = log(open[t+2] / open[t+1])"
            ),
            "execution_mapping_evidence": (
                "not_executed_by_this_target_return_probe"
            ),
            "actual_target_returns": actual,
            "expected_target_returns": expected,
            "formula_matches": formula_matches,
            "next_open_interval_matches": next_open_interval_matches,
            "ic_probe": {
                "reported_ic": reported_ic_value,
                "pnl_clock_ic": pnl_clock_ic,
                "one_extra_bar_ic_valid_only": (
                    one_extra_bar_ic_valid_only
                ),
                "one_extra_bar_ic_with_padding": (
                    one_extra_bar_ic_with_padding
                ),
                "ic_matches_pnl_clock": ic_matches_pnl_clock,
                "ic_extra_shift_reproduced": (
                    ic_extra_shift_reproduced
                ),
                "padding_pollution_reproduced": (
                    padding_pollution_reproduced
                ),
                "valid_target_count": valid_target_count,
            },
            "runtime_method_executed": True,
        },
    )


if __name__ == "__main__":
    main()
