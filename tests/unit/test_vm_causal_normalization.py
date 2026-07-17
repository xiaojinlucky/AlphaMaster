from __future__ import annotations

import math

import numpy as np
import torch

from backtest_viz.engine import BacktestEngine
from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.live_signal import evaluate_signal
from strategy_manager.signal import compute_target_positions_stateless


def _one_feature(values: torch.Tensor) -> torch.Tensor:
    """把 [N,T] 原始序列包装成 StackVM 所需的 [N,F,T]。"""
    return values.unsqueeze(1)


def _market_data(bars: int = 240) -> dict[str, torch.Tensor]:
    """构造含前缀冲击和后续状态变化的确定性单品种 OHLCV。"""
    steps = torch.arange(bars, dtype=torch.float64)
    log_ret = 0.0015 * torch.sin(steps / 7.0)
    log_ret[219] = 0.08
    log_ret[220:] = torch.linspace(-0.025, 0.035, bars - 220)
    close = (100.0 * torch.exp(torch.cumsum(log_ret, dim=0))).float().unsqueeze(0)
    open_ = torch.cat([close[:, :1], close[:, :-1]], dim=1)
    spread = close * 0.002
    return {
        "open": open_,
        "high": torch.maximum(open_, close) + spread,
        "low": torch.minimum(open_, close) - spread,
        "close": close,
        "volume": (1000.0 + steps.float() * 3.0).unsqueeze(0),
        "time": torch.arange(bars, dtype=torch.int64).unsqueeze(0),
    }


def test_future_perturbation_cannot_change_past_factor_or_position() -> None:
    vm = StackVM()
    original = torch.tensor(
        [[-2.0, -0.5, 0.25, 1.5, 2.0, 3.0, 4.0, 5.0]],
        dtype=torch.float32,
    )
    perturbed = original.clone()
    perturbed[:, 4:] = torch.tensor([[200.0, -300.0, 500.0, -800.0]])

    factor_original = vm.execute([0], _one_feature(original))
    factor_perturbed = vm.execute([0], _one_feature(perturbed))

    assert factor_original is not None
    assert factor_perturbed is not None
    assert torch.equal(factor_original[:, :4], factor_perturbed[:, :4])

    position_original = compute_target_positions_stateless(factor_original)
    position_perturbed = compute_target_positions_stateless(factor_perturbed)
    assert torch.equal(position_original[:, :4], position_perturbed[:, :4])


def test_full_sequence_prefix_equals_independent_prefix_calculation() -> None:
    vm = StackVM()
    values = torch.tensor(
        [[0.2, -0.1, 0.4, 1.2, -0.7, 0.3, 2.1, -1.4, 0.8]],
        dtype=torch.float32,
    )
    prefix_bars = 6

    full = vm.execute([0], _one_feature(values))
    prefix = vm.execute([0], _one_feature(values[:, :prefix_bars]))

    assert full is not None
    assert prefix is not None
    torch.testing.assert_close(full[:, :prefix_bars], prefix, rtol=0.0, atol=0.0)


def test_warmup_uses_only_available_observations() -> None:
    vm = StackVM()
    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    actual = vm.execute([0], _one_feature(values))

    assert actual is not None
    expected = torch.tensor(
        [[0.0, 1.0, math.sqrt(1.5)]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rolling_window_excludes_observations_older_than_200_bars() -> None:
    vm = StackVM()
    values = torch.arange(1.0, 202.0, dtype=torch.float32).unsqueeze(0)

    actual = vm.execute([0], _one_feature(values))

    assert actual is not None
    final_window = values[:, -200:].double()
    expected_last = (
        (final_window[:, -1] - final_window.mean(dim=1))
        / final_window.std(dim=1, unbiased=False)
    )
    torch.testing.assert_close(
        actual[:, -1].double(),
        expected_last,
        rtol=1e-6,
        atol=1e-7,
    )


def test_constant_and_extreme_inputs_are_safe() -> None:
    vm = StackVM()

    constant = vm.execute([0], _one_feature(torch.full((1, 220), 7.0)))
    assert constant is not None
    assert torch.equal(constant, torch.zeros_like(constant))

    extreme_values = torch.tensor(
        [[1e20, -1e20, 5e19, -5e19, float("nan"), float("inf"), 0.0]],
        dtype=torch.float32,
    )
    extreme = vm.execute([0], _one_feature(extreme_values))
    assert extreme is not None
    assert extreme.dtype == extreme_values.dtype
    assert torch.isfinite(extreme).all()
    assert float(extreme.abs().max()) <= 3.0

    empty = vm.execute([0], torch.empty((1, 1, 0), dtype=torch.float32))
    assert empty is not None
    assert empty.shape == (1, 0)


def test_cross_sectional_output_is_causal_and_falls_back_when_dispersion_is_zero() -> None:
    vm = StackVM()
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    actual = vm.execute([0], _one_feature(values))

    assert actual is not None
    expected = torch.tensor(
        [
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    extended = torch.cat(
        [
            values,
            torch.tensor(
                [[1000.0, -2000.0], [-3000.0, 4000.0]],
                dtype=torch.float32,
            ),
        ],
        dim=1,
    )
    extended_actual = vm.execute([0], _one_feature(extended))
    assert extended_actual is not None
    torch.testing.assert_close(
        extended_actual[:, : values.shape[1]],
        actual,
        rtol=0.0,
        atol=0.0,
    )


def test_training_backtest_and_live_match_at_the_same_historical_prefix() -> None:
    formula = [0]
    full_raw = _market_data()
    prefix_bars = 220
    prefix_raw = {key: value[:, :prefix_bars] for key, value in full_raw.items()}

    full_features = MT5FeatureEngineer.compute_features(full_raw)
    prefix_features = MT5FeatureEngineer.compute_features(prefix_raw)

    training_full = StackVM().execute(formula, full_features)
    training_prefix = StackVM().execute(formula, prefix_features)
    assert training_full is not None
    assert training_prefix is not None

    backtest = BacktestEngine(formula=formula, cost_rate=0.0).run(
        full_raw,
        full_features,
        ["XAUUSD"],
    )[0]
    live = evaluate_signal(formula, prefix_raw)
    assert live["state"] == "ok"

    full_factor_at_prefix = float(training_full[0, prefix_bars - 1])
    prefix_factor_last = float(training_prefix[0, -1])
    backtest_factor_at_prefix = float(backtest.factor[prefix_bars - 1])

    assert math.isclose(
        full_factor_at_prefix,
        prefix_factor_last,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        backtest_factor_at_prefix,
        prefix_factor_last,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(live["factor_value"]),
        prefix_factor_last,
        rel_tol=0.0,
        abs_tol=5e-7,
    )

    training_position = float(
        compute_target_positions_stateless(training_prefix)[0, -1]
    )
    assert abs(training_position) > 0.05
    assert math.isclose(
        float(backtest.position[prefix_bars - 1]),
        training_position,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(live["position"]),
        training_position,
        rel_tol=0.0,
        abs_tol=5e-5,
    )

    assert np.isfinite(backtest.factor).all()
