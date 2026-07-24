"""风险监控与真实 Runner 必须共用同一套策略和多公式信号。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import monitor_live_risk
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.runner import (
    _DryRunTraderProxy,
    apply_runner_position_state,
    compute_latest_formula_target,
    formula_set_sha256,
    load_symbol_formulas,
    verify_formula_set_hashes,
)
from strategy_manager.portfolio import Position


def _write_strategy(path: Path, **changes) -> None:
    payload = {
        "vocab_version": FORMULA_VOCAB.version,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "formula": [0],
        "best_score": 1.0,
        **changes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runner_loader_accepts_current_positive_score_contract(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies"
    _write_strategy(strategies / "best_XAUUSD.json")

    loaded = load_symbol_formulas(
        ["XAUUSD"],
        strategies_dir=strategies,
    )

    assert loaded == {"XAUUSD": [[0]]}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scoring_contract_version", None),
        ("scoring_contract_version", "previous_contract"),
        ("vocab_version", None),
        ("vocab_version", "vprevious0000"),
        ("best_score", None),
        ("best_score", 0.0),
        ("best_score", -1.0),
    ],
)
def test_runner_loader_fails_when_only_strategy_is_not_production_valid(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    strategies = tmp_path / "strategies"
    _write_strategy(strategies / "best_XAUUSD.json", **{field: value})

    with pytest.raises(FileNotFoundError):
        load_symbol_formulas(
            ["XAUUSD"],
            strategies_dir=strategies,
        )


def test_monitor_uses_runner_multi_formula_average(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_strategy(
        tmp_path / "strategies" / "best_XAUUSD.json",
        formula=[0],
    )
    _write_strategy(
        tmp_path / "strategies" / "best_metals_comm.json",
        formula=[1],
    )

    class FakeVm:
        def execute(self, formula, _feature):
            latest = 1.0 if formula == [0] else -1.0
            return torch.tensor([[0.0, latest]])

    monkeypatch.setattr(monitor_live_risk, "StackVM", FakeVm)
    monkeypatch.setattr(
        monitor_live_risk.MT5FeatureEngineer,
        "compute_features",
        lambda _raw: torch.zeros((1, 2, 2)),
    )
    manager = SimpleNamespace(symbols=["XAUUSD"], raw_dict={})

    targets = monitor_live_risk._current_targets(manager)

    assert targets["XAUUSD"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    "formula",
    [
        [-1],
        [1.9],
        ["1"],
        [True],
        [FORMULA_VOCAB.size],
        [FORMULA_VOCAB.operator_offset],
        [0, 1],
    ],
    ids=(
        "negative",
        "float",
        "string",
        "bool",
        "out_of_range",
        "operator_underflow",
        "extra_stack",
    ),
)
def test_runner_loader_rejects_malformed_formula_tokens(
    tmp_path: Path,
    formula,
) -> None:
    strategies = tmp_path / "strategies"
    _write_strategy(strategies / "best_XAUUSD.json", formula=formula)

    with pytest.raises(FileNotFoundError):
        load_symbol_formulas(
            ["XAUUSD"],
            strategies_dir=strategies,
        )


def test_runner_rejects_partial_symbol_formula_set(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies"
    _write_strategy(strategies / "best_XAUUSD.json", formula=[0])
    _write_strategy(
        strategies / "best_US30.cash.json",
        formula=[1],
        best_score=0.0,
    )

    with pytest.raises(ValueError, match="拒绝部分策略集合启动"):
        load_symbol_formulas(
            ["XAUUSD", "US30.cash"],
            strategies_dir=strategies,
        )


def test_multi_formula_signal_rejects_one_failed_formula() -> None:
    class FakeVm:
        def execute(self, formula, _feature):
            if formula == [1]:
                return None
            return torch.zeros((1, 2))

    with pytest.raises(RuntimeError, match=r"formula\[1\].*执行失败"):
        compute_latest_formula_target(
            FakeVm(),
            [[0], [1]],
            torch.zeros((1, 2, 2)),
            symbol="XAUUSD",
            min_exposure=0.05,
        )


def test_multi_formula_signal_rejects_all_failed_formulas() -> None:
    class FakeVm:
        def execute(self, _formula, _feature):
            return None

    with pytest.raises(RuntimeError, match=r"formula\[0\].*执行失败"):
        compute_latest_formula_target(
            FakeVm(),
            [[0]],
            torch.zeros((1, 2, 2)),
            symbol="XAUUSD",
            min_exposure=0.05,
        )


def test_multi_formula_signal_rejects_non_finite_output() -> None:
    class FakeVm:
        def execute(self, _formula, _feature):
            return torch.tensor([[0.0, float("nan")]])

    with pytest.raises(RuntimeError, match="非有限值"):
        compute_latest_formula_target(
            FakeVm(),
            [[0]],
            torch.zeros((1, 2, 2)),
            symbol="XAUUSD",
            min_exposure=0.05,
        )


def test_final_runner_formula_hash_check_rejects_post_scan_replacement() -> None:
    with pytest.raises(ValueError, match="公式集合与扫描报告不一致"):
        verify_formula_set_hashes(
            {"XAUUSD": [[1]]},
            {"XAUUSD": formula_set_sha256([[0]])},
        )


@pytest.mark.parametrize(
    "method",
    ["buy", "sell", "open_short", "close_position", "close_all_positions"],
)
def test_dry_run_proxy_never_calls_order_writer(method: str) -> None:
    class FakeTrader:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __getattr__(self, name: str):
            def call(*_args, **_kwargs):
                self.calls.append(name)
                return True

            return call

    inner = FakeTrader()
    proxy = _DryRunTraderProxy(inner)

    assert getattr(proxy, method)("XAUUSD") is False
    assert inner.calls == []


def test_dry_run_proxy_blocks_future_unknown_callable_by_default() -> None:
    class FakeTrader:
        def __init__(self) -> None:
            self.called = False

        def place_order(self, *_args, **_kwargs) -> bool:
            self.called = True
            return True

    inner = FakeTrader()
    proxy = _DryRunTraderProxy(inner)

    assert proxy.place_order("XAUUSD") is False
    assert inner.called is False


def test_runner_does_not_fallback_when_all_symbol_strategies_are_missing(
    tmp_path: Path,
) -> None:
    fallback = tmp_path / "best_mt5_strategy.json"
    _write_strategy(fallback, formula=[0])

    with pytest.raises(FileNotFoundError, match="均无有效策略文件"):
        load_symbol_formulas(
            ["AAA", "BBB"],
            strategies_dir=tmp_path / "strategies",
        )


def test_strategy_symbol_cannot_escape_strategy_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="非法交易品种名|不得包含"):
        load_symbol_formulas(
            [r"foo\..\..\evil"],
            strategies_dir=tmp_path / "strategies",
        )


def test_runner_position_state_holds_size_when_direction_is_unchanged() -> None:
    targets = torch.tensor([[0.06, 0.90, 0.40, 0.0, -0.20, -0.80]])

    executed = apply_runner_position_state(
        targets,
        min_exposure=0.05,
        fixed_exposure=False,
    )

    assert executed.tolist()[0] == pytest.approx(
        [0.06, 0.06, 0.06, 0.0, -0.20, -0.20]
    )


def test_runner_fixed_lot_state_ignores_signal_size_after_entry() -> None:
    targets = torch.tensor([[0.06, 0.90, 0.0, -0.20]])

    executed = apply_runner_position_state(
        targets,
        min_exposure=0.05,
        fixed_exposure=True,
    )

    assert executed.tolist()[0] == pytest.approx([1.0, 1.0, 0.0, -1.0])


def test_monitor_uses_recorded_entry_exposure_not_current_same_side_signal() -> None:
    recorded = Position(
        symbol="US30.cash",
        ticket=1,
        entry_price=100.0,
        entry_time=1.0,
        lot_size=0.5,
        direction="BUY",
        highest_price=100.0,
        lowest_price=100.0,
        is_partial_closed=False,
        target_exposure=0.9,
    )

    assert monitor_live_risk._monitor_exposure(
        signal=0.1,
        live_side="BUY",
        recorded=recorded,
    ) == pytest.approx(0.9)


def test_monitor_rejects_live_position_without_recorded_entry_exposure() -> None:
    recorded = Position(
        symbol="US30.cash",
        ticket=1,
        entry_price=100.0,
        entry_time=1.0,
        lot_size=0.5,
        direction="BUY",
        highest_price=100.0,
        lowest_price=100.0,
        is_partial_closed=False,
    )

    assert (
        monitor_live_risk._monitor_exposure(
            signal=0.1,
            live_side="BUY",
            recorded=recorded,
        )
        is None
    )


def test_runner_blocks_unknown_live_position_before_any_order() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from strategy_manager.runner import MT5StrategyRunner

    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner.dry_run = False
    runner._data_manager = SimpleNamespace(symbols=["XAUUSD"])
    runner.trader = MagicMock()
    runner.trader.get_positions.return_value = [
        SimpleNamespace(type=0, volume=0.01)
    ]
    runner.portfolio = MagicMock()
    runner.portfolio.positions = {
        "XAUUSD": Position(
            symbol="XAUUSD",
            ticket=1,
            entry_price=100.0,
            entry_time=1.0,
            lot_size=0.01,
            direction="BUY",
            highest_price=100.0,
            lowest_price=100.0,
            is_partial_closed=False,
        )
    }

    from unittest.mock import patch

    with (
        patch("strategy_manager.runner.Config.SYMBOLS", ["XAUUSD"]),
        pytest.raises(RuntimeError, match="缺少可信 target_exposure"),
    ):
        runner._reconcile_positions(torch.tensor([0.5]))

    runner.trader.buy.assert_not_called()
    runner.trader.open_short.assert_not_called()
    runner.trader.close_position.assert_not_called()
    runner.trader.close_all_positions.assert_not_called()


def test_runner_rejects_partial_target_vector_before_any_order() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from strategy_manager.runner import MT5StrategyRunner

    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner.dry_run = False
    runner._data_manager = SimpleNamespace(
        symbols=["XAUUSD", "US30.cash"]
    )
    runner.trader = MagicMock()
    runner.portfolio = MagicMock()

    with (
        patch(
            "strategy_manager.runner.Config.SYMBOLS",
            ["XAUUSD", "US30.cash"],
        ),
        pytest.raises(RuntimeError, match="完整覆盖全部交易品种"),
    ):
        runner._reconcile_positions(torch.tensor([0.5]))

    runner.trader.get_positions.assert_not_called()


def test_runner_blocks_position_read_failure_before_any_order() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from strategy_manager.runner import MT5StrategyRunner

    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner.dry_run = False
    runner._data_manager = SimpleNamespace(symbols=["XAUUSD"])
    runner.trader = MagicMock()
    runner.trader.get_positions.return_value = None
    runner.portfolio = MagicMock()

    with (
        patch("strategy_manager.runner.Config.SYMBOLS", ["XAUUSD"]),
        pytest.raises(RuntimeError, match="无法确认实盘持仓"),
    ):
        runner._reconcile_positions(torch.tensor([0.5]))

    runner.trader.buy.assert_not_called()
    runner.trader.open_short.assert_not_called()
    runner.trader.close_position.assert_not_called()
    runner.trader.close_all_positions.assert_not_called()
