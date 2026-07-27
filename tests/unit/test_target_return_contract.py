"""目标收益时钟、尾部裁剪和评分合同的 P0 回归。"""
from __future__ import annotations

import pytest
import torch

from model_core.backtest import MT5Backtest
from model_core.engine import AlphaEngine
from model_core.evaluator import (
    EffectivenessEvaluator,
    ScoreResult,
    _align_causal,
    ablate,
    prune,
)
from model_core.target_contract import (
    TARGET_RETURN_HORIZON,
    align_target_return_window,
    valid_target_length,
)


def test_target_contract_pairs_same_index_and_drops_exact_tail() -> None:
    candidate = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    target = candidate.clone()
    target[:, -TARGET_RETURN_HORIZON:] = 999.0

    candidate_valid, target_valid = align_target_return_window(candidate, target)

    assert valid_target_length(12) == 10
    assert candidate_valid.shape == target_valid.shape == (1, 10)
    assert torch.equal(candidate_valid, candidate[:, :10])
    assert torch.equal(target_valid, target[:, :10])


@pytest.mark.parametrize("horizon", [0, -1, 1.5, "2", True])
def test_target_contract_rejects_implicit_or_invalid_horizon(horizon: object) -> None:
    values = torch.zeros(1, 10)
    with pytest.raises(ValueError):
        align_target_return_window(values, values, horizon)  # type: ignore[arg-type]


def test_target_contract_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="形状必须完全一致"):
        align_target_return_window(torch.zeros(1, 10), torch.zeros(2, 10))


def test_engine_ic_uses_same_clock_and_ignores_padded_tail() -> None:
    torch.manual_seed(7)
    valid = torch.randn(3, 30)
    factor = torch.cat((valid, torch.tensor([[50.0, -50.0]]).repeat(3, 1)), dim=1)
    target = torch.cat((valid.clone(), torch.zeros(3, 2)), dim=1)

    ic_mean, _ = AlphaEngine._compute_ic(factor, target)

    assert ic_mean.item() == pytest.approx(1.0, abs=1e-6)


def test_backtest_ic_uses_same_clock_on_already_valid_window() -> None:
    torch.manual_seed(11)
    target = torch.randn(2, 30)
    factors = target.clone()

    stability = MT5Backtest()._ts_ic_stability(factors, target)

    assert stability == pytest.approx(3.0)


def test_backtest_score_and_cost_ignore_unrealizable_tail_positions() -> None:
    torch.manual_seed(17)
    factors = torch.randn(2, 40)
    target = torch.randn(2, 40) * 0.01
    changed_factors = factors.clone()
    changed_target = target.clone()
    changed_factors[:, -2:] = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    changed_target[:, -2:] = torch.tensor([[99.0, -99.0], [-99.0, 99.0]])

    bt = MT5Backtest()
    original_score, original_oos = bt.evaluate(factors, {}, target)
    changed_score, changed_oos = bt.evaluate(changed_factors, {}, changed_target)

    assert torch.equal(original_score, changed_score)
    assert original_oos == changed_oos


def test_walk_forward_fold_cannot_reach_padded_target_tail() -> None:
    factors = torch.randn(1, 12)
    target = torch.randn(1, 12)
    bt = MT5Backtest()

    with pytest.raises(ValueError, match="有效窗口"):
        bt.evaluate_fold(factors, target, 0, 4, 6, 12)

    train_score, val_score = bt.evaluate_fold(factors, target, 0, 4, 6, 10)
    assert torch.isfinite(train_score)
    assert torch.isfinite(val_score)


def test_walk_forward_validation_window_charges_first_position_from_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = torch.full((1, 22), 10.0)
    target = torch.zeros_like(factors)
    captured: list[torch.Tensor] = []
    bt = MT5Backtest(cost_rate=0.01)

    def capture_objective(
        _factors: torch.Tensor,
        _target: torch.Tensor,
        pnl: torch.Tensor,
        _position: torch.Tensor,
        *,
        eval_bars: int,
    ) -> torch.Tensor:
        assert eval_bars == pnl.shape[1]
        captured.append(pnl.clone())
        return torch.tensor(1.0)

    monkeypatch.setattr(bt, "_multi_objective", capture_objective)
    monkeypatch.setattr(bt, "_turnover_penalty", lambda _turnover: torch.tensor(0.0))
    monkeypatch.setattr(bt, "_sortino", lambda _pnl: torch.tensor(0.0))

    bt.evaluate_fold(factors, target, 0, 8, 10, 20)

    assert len(captured) == 2
    assert captured[1][0, 0].item() == pytest.approx(-0.01, abs=1e-5)
    assert torch.count_nonzero(captured[1][0, 1:]) == 0


def test_non_walk_forward_validation_window_charges_first_position_from_flat() -> None:
    factors = torch.full((1, 22), 10.0)
    target = torch.zeros_like(factors)
    bt = MT5Backtest(cost_rate=0.01)

    _, mean_val = bt.evaluate(factors, {}, target)

    assert mean_val == pytest.approx(-0.0025, abs=1e-5)


def test_effectiveness_evaluator_defaults_to_current_target_horizon() -> None:
    evaluator = EffectivenessEvaluator()
    candidate = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    target = candidate.clone()

    candidate_valid, target_valid = _align_causal(
        candidate,
        target,
        evaluator.target_horizon,
    )

    assert evaluator.target_horizon == TARGET_RETURN_HORIZON
    assert candidate_valid.shape == target_valid.shape == (1, 6)


def test_ablation_aligns_once_before_splitting_windows() -> None:
    target = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    candidate = target.clone()
    target[:, -2:] = torch.tensor([[999.0, -999.0]])

    result = ablate(
        "good",
        {"good": candidate},
        target,
        n_windows=5,
    )

    assert result.error is None
    assert result.marginal_contribution == pytest.approx(1.0, abs=1e-6)


def test_prune_correlation_ignores_unrealizable_tail() -> None:
    scores = [
        ScoreResult("leader", "test", 1, 1, 1, 1, 1.0, False),
        ScoreResult("follower", "test", 0.5, 0.5, 0.5, 0.5, 0.5, False),
    ]
    leader = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0]])
    follower = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -100.0, 100.0]])

    rows = prune(
        scores,
        {"leader": leader, "follower": follower},
        corr_threshold=0.99,
    )

    by_name = {row.candidate: row for row in rows}
    assert by_name["follower"].retention_status == "pruned"
    assert by_name["follower"].pruned_in_favor_of == "leader"
