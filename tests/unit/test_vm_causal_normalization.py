from __future__ import annotations

import torch

from model_core.vm import StackVM
def test_future_perturbation_cannot_change_past_factor() -> None:
    original = torch.tensor(
        [[-2.0, -0.5, 0.25, 1.5, 2.0, 3.0, 4.0, 5.0]],
        dtype=torch.float32,
    )
    perturbed = original.clone()
    perturbed[:, 4:] = torch.tensor([[200.0, -300.0, 500.0, -800.0]])

    factor_original = StackVM._normalize_output(original)
    factor_perturbed = StackVM._normalize_output(perturbed)
    assert torch.equal(factor_original[:, :4], factor_perturbed[:, :4])


def test_full_sequence_prefix_equals_independent_prefix_calculation() -> None:
    values = torch.tensor(
        [[0.2, -0.1, 0.4, 1.2, -0.7, 0.3, 2.1, -1.4, 0.8]],
        dtype=torch.float32,
    )
    prefix_bars = 6

    full = StackVM._normalize_output(values)
    prefix = StackVM._normalize_output(values[:, :prefix_bars])
    torch.testing.assert_close(full[:, :prefix_bars], prefix, rtol=0.0, atol=0.0)


def test_constant_factor_returns_original() -> None:
    values = torch.full((1, 8), 7.0, dtype=torch.float32)

    assert torch.equal(StackVM._normalize_output(values), values)


def test_clip_range() -> None:
    values = torch.cat(
        [torch.zeros(1, 20, dtype=torch.float32), torch.tensor([[10000.0]])],
        dim=1,
    )

    actual = StackVM._normalize_output(values)

    assert torch.all(actual >= -3.0)
    assert torch.all(actual <= 3.0)
