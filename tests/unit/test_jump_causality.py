from __future__ import annotations

import math

import pytest
import torch

from model_core.formula_contract import (
    STACKVM_JUMP_EPS,
    STACKVM_JUMP_THRESHOLD,
    STACKVM_JUMP_WINDOW,
)
from model_core.ops import _op_jump
from model_core.vocab import FORMULA_VOCAB
from model_core.vm import StackVM


def _jump_oracle(x: torch.Tensor) -> torch.Tensor:
    work = torch.nan_to_num(
        x.to(dtype=torch.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    columns: list[torch.Tensor] = []
    for t in range(work.shape[1]):
        window = work[:, max(0, t - STACKVM_JUMP_WINDOW + 1) : t + 1]
        if window.shape[1] < 2:
            z = torch.zeros_like(work[:, t])
        else:
            shifted = window - window[:, :1]
            mean = shifted.mean(dim=1)
            std = shifted.std(dim=1, correction=1)
            z = torch.where(
                std > STACKVM_JUMP_EPS,
                (shifted[:, -1] - mean) / (std + STACKVM_JUMP_EPS),
                torch.zeros_like(std),
            )
        columns.append(torch.tanh(z - STACKVM_JUMP_THRESHOLD))
    if not columns:
        return x.clone()
    return torch.stack(columns, dim=1).to(dtype=x.dtype)


def _jump_token() -> int:
    index = FORMULA_VOCAB.operator_names.index("JUMP")
    return FORMULA_VOCAB.operator_offset + index


def test_jump_matches_current_inclusive_rolling_oracle() -> None:
    x = torch.arange(1, 206, dtype=torch.float64).reshape(1, -1)

    actual = _op_jump(x)
    expected = _jump_oracle(x)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_jump_prefix_is_independent_of_later_sequence_length() -> None:
    base = torch.linspace(-2.0, 3.0, 260, dtype=torch.float64)
    x = (base + 0.3 * torch.sin(base * 7.0)).reshape(1, -1)
    full = _op_jump(x)

    for end in (1, 2, 3, 199, 200, 201, 260):
        prefix = _op_jump(x[:, :end])
        torch.testing.assert_close(full[:, :end], prefix, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "future",
    [
        torch.tensor([0.25, -3.5, 7.75, 1.0]),
        torch.tensor([1e150, 1e150, 1e150, 1e150]),
        torch.tensor([-1e150, -1e150, -1e150, -1e150]),
        torch.tensor([1e150, -1e150, 1e150, -1e150]),
    ],
    ids=("finite-random", "extreme-positive", "extreme-negative", "alternating"),
)
def test_jump_future_perturbation_cannot_change_history(
    future: torch.Tensor,
) -> None:
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )
    changed = x.clone()
    changed[:, 4:] = future.to(dtype=torch.float64)

    torch.testing.assert_close(
        _op_jump(x)[:, :4],
        _op_jump(changed)[:, :4],
        rtol=1e-10,
        atol=1e-10,
    )


def test_stackvm_jump_formula_is_prefix_invariant() -> None:
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        dtype=torch.float64,
    )
    changed = x.clone()
    changed[:, 3:] = torch.tensor(
        [[100.0, -100.0, 50.0]],
        dtype=torch.float64,
    )
    features = torch.zeros(
        (1, FORMULA_VOCAB.feature_count, x.shape[1]),
        dtype=x.dtype,
    )
    changed_features = features.clone()
    features[:, 0, :] = x
    changed_features[:, 0, :] = changed
    formula = [0, _jump_token()]
    vm = StackVM()

    original = vm.execute(formula, features)
    perturbed = vm.execute(formula, changed_features)

    assert original is not None
    assert perturbed is not None
    torch.testing.assert_close(
        original[:, :3],
        perturbed[:, :3],
        rtol=1e-10,
        atol=1e-10,
    )


def test_jump_handles_empty_non_finite_and_constant_inputs() -> None:
    empty = torch.empty((2, 0), dtype=torch.float32)
    assert _op_jump(empty).shape == empty.shape

    values = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [float("nan"), float("inf"), -float("inf"), 2.0],
        ],
        dtype=torch.float32,
    )
    result = _op_jump(values)

    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert result.device == values.device
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, _jump_oracle(values))


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([0.0] * 16, torch.float32),
        ([0.0] * 8 + [10.0] + [0.0] * 7, torch.float32),
        ([0.0] * 8 + [-10.0] + [0.0] * 7, torch.float64),
        ([float(index) for index in range(32)], torch.float64),
        (
            [1e12 + math.sin(index) for index in range(260)],
            torch.float64,
        ),
        ([1e15 + float(index) for index in range(260)], torch.float64),
    ],
    ids=(
        "zero-variance-float32",
        "single-positive-jump-float32",
        "single-negative-jump-float64",
        "continuous-jump-float64",
        "large-base-small-wave-float64",
        "large-base-linear-float64",
    ),
)
def test_jump_special_inputs_match_direct_window_oracle(
    values: list[float],
    dtype: torch.dtype,
) -> None:
    x = torch.tensor([values], dtype=dtype)

    actual = _op_jump(x)
    expected = _jump_oracle(x)

    assert torch.isfinite(actual).all()
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_jump_201st_value_excludes_first_observation() -> None:
    tail = torch.linspace(-3.0, 5.0, 200, dtype=torch.float64)
    first_a = torch.cat((torch.tensor([1e6], dtype=torch.float64), tail))
    first_b = torch.cat((torch.tensor([-1e6], dtype=torch.float64), tail))

    actual_a = _op_jump(first_a.reshape(1, -1))
    actual_b = _op_jump(first_b.reshape(1, -1))

    torch.testing.assert_close(
        actual_a[:, 200],
        actual_b[:, 200],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前 PyTorch 无 CUDA")
def test_jump_cuda_matches_cpu() -> None:
    cpu = torch.linspace(-5.0, 7.0, 260, dtype=torch.float64).reshape(1, -1)

    expected = _op_jump(cpu)
    actual = _op_jump(cpu.cuda()).cpu()

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
