"""执行 vn.py 原始 AlphaDataset 的历史成分区间筛选。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from common import load_fixture, verify_git_source, write_result


def _install_package_shell(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _install_alphalens_stubs() -> None:
    alphalens = types.ModuleType("alphalens")
    alphalens.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("alphalens.utils")
    tears = types.ModuleType("alphalens.tears")

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("本实验不调用 Alphalens 可视化接口")

    utils.get_clean_factor_and_forward_returns = unavailable
    tears.create_full_tear_sheet = unavailable
    sys.modules["alphalens"] = alphalens
    sys.modules["alphalens.utils"] = utils
    sys.modules["alphalens.tears"] = tears


def _source_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise RuntimeError("vn.py 源码未声明 __version__")


def _load_alpha_dataset(source_root: Path) -> type:
    package_root = source_root / "vnpy"
    alpha_root = package_root / "alpha"
    dataset_root = alpha_root / "dataset"
    for path in (package_root, alpha_root, dataset_root):
        if not path.is_dir():
            raise RuntimeError(f"vn.py 源码目录不存在: {path}")

    _install_package_shell("vnpy", package_root)
    _install_package_shell("vnpy.alpha", alpha_root)
    _install_package_shell("vnpy.alpha.dataset", dataset_root)
    _install_alphalens_stubs()
    module = importlib.import_module("vnpy.alpha.dataset.template")
    return module.AlphaDataset


def _row_key(row: dict[str, Any]) -> list[str]:
    return [
        row["datetime"].isoformat(),
        str(row["vt_symbol"]),
    ]


def _new_dataset(
    dataset_type: type,
    raw: pl.DataFrame,
    feature: pl.DataFrame,
) -> Any:
    dataset = dataset_type(
        raw,
        train_period=("2024-01-01", "2024-01-02"),
        valid_period=("2024-01-03", "2024-01-03"),
        test_period=("2024-01-04", "2024-01-04"),
    )
    dataset.add_feature("fixed_feature", result=feature)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture, fixture_sha256 = load_fixture(args.fixture)
    constituents = fixture["constituents"]
    source_root = args.source_root.resolve()
    verified_source_commit = verify_git_source(
        source_root,
        args.source_commit,
    )
    template_path = (
        source_root / "vnpy" / "alpha" / "dataset" / "template.py"
    )
    version = _source_version(source_root / "vnpy" / "__init__.py")
    AlphaDataset = _load_alpha_dataset(source_root)

    sessions = [
        datetime.fromisoformat(value)
        for value in constituents["sessions"]
    ]
    symbols = [str(value) for value in constituents["symbols"]]
    rows = [
        {
            "datetime": session,
            "vt_symbol": symbol,
            "close": float(index + 1),
        }
        for index, session in enumerate(sessions)
        for symbol in symbols
    ]
    raw = pl.DataFrame(rows)
    feature = raw.select(["datetime", "vt_symbol"]).with_columns(
        pl.int_range(0, raw.height, eager=True)
        .cast(pl.Float64)
        .alias("data")
    )
    filters = {
        symbol: [
            (
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
            )
            for start, end in ranges
        ]
        for symbol, ranges in constituents["filters"].items()
    }

    dataset = _new_dataset(AlphaDataset, raw, feature)
    dataset.prepare_data(filters=filters, max_workers=1)
    actual_rows = [_row_key(row) for row in dataset.raw_df.to_dicts()]
    expected_rows = constituents["expected_rows"]
    inclusive_filter_matches = actual_rows == expected_rows
    if not inclusive_filter_matches:
        raise RuntimeError(
            f"vn.py 历史成分筛选不匹配: {actual_rows!r}"
        )

    overlap_filters = {
        symbol: [
            (
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
            )
            for start, end in ranges
        ]
        for symbol, ranges in constituents["overlap_probe"][
            "filters"
        ].items()
    }
    overlap_dataset = _new_dataset(AlphaDataset, raw, feature)
    overlap_dataset.prepare_data(
        filters=overlap_filters,
        max_workers=1,
    )
    overlap_rows = [
        _row_key(row) for row in overlap_dataset.raw_df.to_dicts()
    ]
    duplicated_row = constituents["overlap_probe"]["duplicated_row"]
    overlap_duplicate_count = overlap_rows.count(duplicated_row)
    overlap_duplicates_reproduced = overlap_duplicate_count == 2

    empty_dict_dataset = _new_dataset(AlphaDataset, raw, feature)
    empty_dict_dataset.prepare_data(filters={}, max_workers=1)
    empty_dict_rows = [
        _row_key(row)
        for row in empty_dict_dataset.raw_df.to_dicts()
    ]
    raw_rows = [_row_key(row) for row in raw.to_dicts()]
    empty_dict_leaks_all_rows = (
        empty_dict_rows == raw_rows
    )

    empty_ranges = {
        symbol: [
            (
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
            )
            for start, end in ranges
        ]
        for symbol, ranges in constituents[
            "empty_ranges_probe"
        ].items()
    }
    empty_ranges_dataset = _new_dataset(AlphaDataset, raw, feature)
    empty_ranges_error = ""
    empty_ranges_message = ""
    try:
        empty_ranges_dataset.prepare_data(
            filters=empty_ranges,
            max_workers=1,
        )
    except ValueError as exc:
        empty_ranges_error = type(exc).__name__
        empty_ranges_message = str(exc)
    empty_ranges_failure_reproduced = (
        empty_ranges_error == "ValueError"
        and empty_ranges_message == "cannot concat empty list"
    )
    if not (
        overlap_duplicates_reproduced
        and empty_dict_leaks_all_rows
        and empty_ranges_failure_reproduced
    ):
        raise RuntimeError("vn.py 历史区间边界反例没有按预期复现")

    write_result(
        args.output,
        {
            "engine": "vnpy",
            "engine_version": version,
            "source_commit": verified_source_commit,
            "source_sha256": hashlib.sha256(
                template_path.read_bytes()
            ).hexdigest(),
            "scope": "alpha_dataset_historical_constituent_filter",
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "inclusive_filter_matches": inclusive_filter_matches,
            "overlap_probe": {
                "duplicate_count": overlap_duplicate_count,
                "duplicates_reproduced": (
                    overlap_duplicates_reproduced
                ),
            },
            "empty_filter_probe": {
                "empty_dict_leaks_all_rows": (
                    empty_dict_leaks_all_rows
                ),
                "empty_ranges_error": empty_ranges_error,
                "empty_ranges_message": empty_ranges_message,
                "empty_ranges_failure_reproduced": (
                    empty_ranges_failure_reproduced
                ),
            },
            "runtime_method_executed": True,
            "unrelated_import_boundary": (
                "Alphalens report functions were stubbed because "
                "prepare_data does not call them"
            ),
        },
    )


if __name__ == "__main__":
    main()
