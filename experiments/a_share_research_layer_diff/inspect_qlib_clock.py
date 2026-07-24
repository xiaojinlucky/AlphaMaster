"""检查 Qlib 固定版本与源码快照的信号时钟合同。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

from common import load_fixture, verify_git_source, write_result

EXPECTED_PACKAGE_VERSION = "0.9.7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_method(
    tree: ast.Module,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise RuntimeError(f"未找到 {class_name}.{method_name}")


def _tuple_names(target: ast.expr) -> list[str]:
    if not isinstance(target, (ast.Tuple, ast.List)):
        return []
    return [
        item.id
        for item in target.elts
        if isinstance(item, ast.Name)
    ]


def _clock_contract(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = _find_method(
        tree,
        "TopkDropoutStrategy",
        "generate_trade_decision",
    )
    shifted_assignment = False
    shifted_names: list[str] = []
    signal_uses_shifted_range = False

    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "get_step_time"
            ):
                continue
            shift_values = [
                keyword.value.value
                for keyword in node.value.keywords
                if keyword.arg == "shift"
                and isinstance(keyword.value, ast.Constant)
            ]
            if shift_values == [1] and len(node.targets) == 1:
                shifted_names = _tuple_names(node.targets[0])
                shifted_assignment = shifted_names == [
                    "pred_start_time",
                    "pred_end_time",
                ]

        if isinstance(node, ast.Call):
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "get_signal"
            ):
                continue
            keyword_names = {
                keyword.arg: keyword.value.id
                for keyword in node.keywords
                if keyword.arg
                and isinstance(keyword.value, ast.Name)
            }
            signal_uses_shifted_range = keyword_names == {
                "start_time": "pred_start_time",
                "end_time": "pred_end_time",
            }

    return {
        "source_sha256": _sha256(path),
        "method_ast_sha256": hashlib.sha256(
            ast.dump(method, include_attributes=False).encode("utf-8")
        ).hexdigest(),
        "shifted_assignment": shifted_assignment,
        "shifted_names": shifted_names,
        "signal_uses_shifted_range": signal_uses_shifted_range,
        "contract_matches": (
            shifted_assignment and signal_uses_shifted_range
        ),
    }


def _shift_semantics(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = _find_method(
        tree,
        "TradeCalendarManager",
        "get_step_time",
    )
    calendar_index_subtracts_shift = False
    for node in ast.walk(method):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "calendar_index"
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Sub)
            and isinstance(node.value.right, ast.Name)
            and node.value.right.id == "shift"
        ):
            continue
        calendar_index_subtracts_shift = True

    return {
        "source_sha256": _sha256(path),
        "method_ast_sha256": hashlib.sha256(
            ast.dump(method, include_attributes=False).encode("utf-8")
        ).hexdigest(),
        "calendar_index_subtracts_shift": (
            calendar_index_subtracts_shift
        ),
        "positive_shift_is_earlier": calendar_index_subtracts_shift,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture, fixture_sha256 = load_fixture(args.fixture)
    version = importlib.metadata.version("pyqlib")
    if version != EXPECTED_PACKAGE_VERSION:
        raise RuntimeError(
            f"Qlib 版本必须是 {EXPECTED_PACKAGE_VERSION}，实际为 {version}"
        )
    verified_source_commit = verify_git_source(
        args.source_root,
        args.source_commit,
    )

    distribution = importlib.metadata.distribution("pyqlib")
    installed_path = Path(
        distribution.locate_file(
            "qlib/contrib/strategy/signal_strategy.py"
        )
    ).resolve()
    snapshot_path = (
        args.source_root
        / "qlib"
        / "contrib"
        / "strategy"
        / "signal_strategy.py"
    ).resolve()
    installed_calendar_path = Path(
        distribution.locate_file("qlib/backtest/utils.py")
    ).resolve()
    snapshot_calendar_path = (
        args.source_root / "qlib" / "backtest" / "utils.py"
    ).resolve()
    required_paths = (
        installed_path,
        snapshot_path,
        installed_calendar_path,
        snapshot_calendar_path,
    )
    if any(not path.is_file() for path in required_paths):
        raise RuntimeError("Qlib 策略源码路径不存在")

    installed = _clock_contract(installed_path)
    snapshot = _clock_contract(snapshot_path)
    installed_shift = _shift_semantics(installed_calendar_path)
    snapshot_shift = _shift_semantics(snapshot_calendar_path)
    if not (
        installed["contract_matches"]
        and snapshot["contract_matches"]
        and installed_shift["positive_shift_is_earlier"]
        and snapshot_shift["positive_shift_is_earlier"]
    ):
        raise RuntimeError("Qlib 策略源码不再满足上一时点信号合同")

    write_result(
        args.output,
        {
            "engine": "qlib",
            "engine_version": version,
            "source_commit": verified_source_commit,
            "scope": "strategy_source_contract",
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "clock_contract": (
                "trade_step[t] reads the earlier signal range because "
                "get_step_time subtracts positive shift from calendar index"
            ),
            "installed_release": installed,
            "source_snapshot": snapshot,
            "installed_calendar_shift": installed_shift,
            "source_snapshot_calendar_shift": snapshot_shift,
            "method_ast_equivalent": (
                installed["method_ast_sha256"]
                == snapshot["method_ast_sha256"]
            ),
            "calendar_method_ast_equivalent": (
                installed_shift["method_ast_sha256"]
                == snapshot_shift["method_ast_sha256"]
            ),
            "runtime_method_executed": False,
            "runtime_status": (
                "not_executed_by_this_source_contract_probe"
            ),
        },
    )


if __name__ == "__main__":
    main()
