"""walk-forward 隔离带必须真实存在，禁止缩短或训练验证重叠。"""

from __future__ import annotations

import pytest

from model_core.engine import _build_walk_forward_folds


def test_expected_folds_preserve_requested_gap() -> None:
    folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)

    assert folds == [
        {
            "train_start": 0,
            "train_end": 84,
            "val_start": 104,
            "val_end": 188,
            "gap": 20,
        },
        {
            "train_start": 0,
            "train_end": 188,
            "val_start": 208,
            "val_end": 292,
            "gap": 20,
        },
        {
            "train_start": 0,
            "train_end": 292,
            "val_start": 312,
            "val_end": 396,
            "gap": 20,
        },
        {
            "train_start": 0,
            "train_end": 396,
            "val_start": 416,
            "val_end": 500,
            "gap": 20,
        },
    ]


def test_insufficient_data_reduces_fold_count_not_gap() -> None:
    folds = _build_walk_forward_folds(T=50, n_folds=5, gap=20)

    assert folds == [
        {
            "train_start": 0,
            "train_end": 3,
            "val_start": 23,
            "val_end": 26,
            "gap": 20,
        },
        {
            "train_start": 0,
            "train_end": 26,
            "val_start": 46,
            "val_end": 50,
            "gap": 20,
        },
    ]


@pytest.mark.parametrize("gap", [0, 1])
def test_gap_below_target_horizon_is_rejected(gap: int) -> None:
    with pytest.raises(ValueError, match="不得小于 2"):
        _build_walk_forward_folds(T=500, n_folds=5, gap=gap)


def test_too_little_data_is_rejected_without_overlap_fallback() -> None:
    with pytest.raises(ValueError, match="无法.*独立训练/验证"):
        _build_walk_forward_folds(T=8, n_folds=5, gap=20)


@pytest.mark.parametrize(
    ("T", "n_folds", "gap"),
    [
        (0, 5, 20),
        (True, 5, 20),
        (500, 1, 20),
        (500, True, 20),
        (500, 5, True),
        (500, 5, -1),
    ],
)
def test_invalid_arguments_are_rejected(T: int, n_folds: int, gap: int) -> None:
    with pytest.raises(ValueError):
        _build_walk_forward_folds(T=T, n_folds=n_folds, gap=gap)


def test_every_fold_has_strict_non_overlapping_boundaries() -> None:
    folds = _build_walk_forward_folds(T=997, n_folds=8, gap=37)

    for fold in folds:
        assert set(fold) == {
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "gap",
        }
        assert fold["train_start"] == 0
        assert fold["gap"] == 37
        assert fold["val_start"] - fold["train_end"] == 37
        assert 0 < fold["train_end"] < fold["val_start"]
        assert fold["val_start"] < fold["val_end"] <= 997
