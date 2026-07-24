"""
tests/unit/test_runner.py — MT5StrategyRunner 单元测试

涵盖：
  - 策略文件不存在时 → sys.exit(1) 被调用（Req 10.1）
  - shutdown() 调用 mt5.shutdown()（Req 10.7）
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import strategy_manager.runner as runner_module
from strategy_manager.runner import MT5StrategyRunner


# ─────────────────────────────────────────────────────────────────
# 测试 1：策略文件不存在 → sys.exit(1)（Req 10.1）
# ─────────────────────────────────────────────────────────────────

def test_missing_strategy_file_causes_sys_exit():
    """当 best_mt5_strategy.json 不存在时，__init__() 必须调用 sys.exit(1)。

    通过 patch os.path.exists 返回 False 来模拟文件不存在的情况。
    pytest.raises(SystemExit) 捕获 sys.exit() 抛出的 SystemExit 异常，
    并验证退出码为 1。
    """
    with patch("strategy_manager.runner.os.path.exists", return_value=False):
        with pytest.raises(SystemExit) as exc_info:
            MT5StrategyRunner()

    assert exc_info.value.code == 1, (
        f"期望退出码为 1，实际为 {exc_info.value.code}"
    )


# ─────────────────────────────────────────────────────────────────
# 测试 2：shutdown() 调用 mt5.shutdown()（Req 10.7）
# ─────────────────────────────────────────────────────────────────

def test_shutdown_calls_mt5_shutdown():
    """shutdown() 必须调用 mt5.shutdown() 以释放 MT5 连接。

    使用 __new__ 绕过 __init__（避免文件加载和子模块初始化），
    直接设置必要属性后调用 shutdown()，验证 mt5.shutdown() 被调用。
    """
    # 用 __new__ 创建实例，跳过 __init__
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)

    # 设置 shutdown() 所依赖的属性
    runner.formula = [1, 2, 3]
    runner._fetcher = None  # shutdown() 中会检查 _fetcher

    mock_mt5 = MagicMock()

    with patch.object(runner_module, "mt5", mock_mt5):
        runner.shutdown()

    mock_mt5.shutdown.assert_called_once(), "shutdown() 必须恰好调用一次 mt5.shutdown()"


def test_bar_time_failure_never_authorizes_rebalance():
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._last_bar_time = torch.tensor([100], dtype=torch.int64)

    class BrokenManager:
        @property
        def bar_time(self):
            raise RuntimeError("broken clock")

    runner._data_manager = BrokenManager()

    with patch.object(runner_module.Config, "SYMBOLS", ["XAUUSD"]):
        assert runner._has_new_closed_bar() is False
    assert torch.equal(
        runner._last_bar_time,
        torch.tensor([100], dtype=torch.int64),
    )


def test_bar_time_shape_mismatch_never_authorizes_rebalance():
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._last_bar_time = torch.tensor([100], dtype=torch.int64)
    runner._data_manager = SimpleNamespace(
        bar_time=torch.tensor([100, 200], dtype=torch.int64)
    )

    with patch.object(runner_module.Config, "SYMBOLS", ["XAUUSD"]):
        assert runner._has_new_closed_bar() is False


def test_bar_time_regression_never_authorizes_rebalance():
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._last_bar_time = torch.tensor([200], dtype=torch.int64)
    runner._data_manager = SimpleNamespace(
        bar_time=torch.tensor([100], dtype=torch.int64)
    )

    with patch.object(runner_module.Config, "SYMBOLS", ["XAUUSD"]):
        assert runner._has_new_closed_bar() is False
    assert runner._last_bar_time.item() == 200


def test_bar_time_strict_forward_move_authorizes_once():
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._last_bar_time = torch.tensor([100], dtype=torch.int64)
    runner._data_manager = SimpleNamespace(
        bar_time=torch.tensor([200], dtype=torch.int64)
    )

    with patch.object(runner_module.Config, "SYMBOLS", ["XAUUSD"]):
        assert runner._has_new_closed_bar() is True
        assert runner._has_new_closed_bar() is False


def test_startup_bar_time_failure_never_calls_initial_reconcile():
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner.portfolio = MagicMock()

    class BrokenManager:
        @property
        def bar_time(self):
            raise RuntimeError("broken startup clock")

    runner._data_manager = BrokenManager()
    runner._compute_targets = MagicMock()
    runner._reconcile_positions = MagicMock()

    with patch.object(runner_module.Config, "SYMBOLS", ["XAUUSD"]):
        assert runner._run_initial_reconcile() is False

    runner._compute_targets.assert_not_called()
    runner._reconcile_positions.assert_not_called()
