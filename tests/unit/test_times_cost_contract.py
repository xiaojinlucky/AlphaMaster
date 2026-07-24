"""独立训练和样本外窗口的首笔建仓成本合同。"""
from __future__ import annotations

import numpy as np
import torch

from times import _numpy_turnover_from_flat, _torch_turnover_from_flat


def test_torch_turnover_charges_first_entry_from_flat() -> None:
    position = torch.tensor([1.0, 1.0, -1.0, 0.0])

    turnover = _torch_turnover_from_flat(position)

    assert torch.equal(turnover, torch.tensor([1.0, 0.0, 2.0, 1.0]))


def test_numpy_turnover_charges_first_entry_from_flat() -> None:
    position = np.array([1.0, 1.0, -1.0, 0.0])

    turnover = _numpy_turnover_from_flat(position)

    np.testing.assert_array_equal(turnover, np.array([1.0, 0.0, 2.0, 1.0]))
