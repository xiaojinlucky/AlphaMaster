from __future__ import annotations

import torch

from model_core.ops import _op_jump


def _expanding_jump_oracle(x: torch.Tensor) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for t in range(x.shape[1]):
        prefix = x[:, : t + 1]
        mean = prefix.mean(dim=1)
        var = (prefix * prefix).mean(dim=1) - mean * mean
        std = var.clamp(min=1e-12).sqrt() + 1e-6
        z = (x[:, t] - mean) / std
        columns.append(torch.tanh(z - 1.5))
    if not columns:
        return x.clone()
    return torch.stack(columns, dim=1)


def test_jump_future_perturbation_cannot_change_past() -> None:
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=torch.float64,
    )
    changed = x.clone()
    changed[:, 10:] = torch.tensor([[200.0, -300.0]], dtype=torch.float64)

    torch.testing.assert_close(
        _op_jump(x)[:, :10],
        _op_jump(changed)[:, :10],
        rtol=0.0,
        atol=0.0,
    )


def test_jump_prefix_independence() -> None:
    x = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]],
        dtype=torch.float64,
    )
    full = _op_jump(x)
    prefix = _op_jump(x[:, :5])

    torch.testing.assert_close(full[:, :5], prefix, rtol=0.0, atol=0.0)


def test_jump_tanh_threshold() -> None:
    x = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    actual = _op_jump(x)
    mean = x[:, :2].mean(dim=1)
    var = (x[:, :2] * x[:, :2]).mean(dim=1) - mean * mean
    z = (x[:, 1] - mean) / (var.clamp(min=1e-12).sqrt() + 1e-6)

    torch.testing.assert_close(actual[:, 1], torch.tanh(z - 1.5))


def test_jump_matches_expanding_oracle() -> None:
    x = torch.tensor(
        [[-3.0, 0.5, 2.0, -1.5, 4.0, 3.0, -2.0]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        _op_jump(x),
        _expanding_jump_oracle(x),
        rtol=1e-12,
        atol=1e-12,
    )
