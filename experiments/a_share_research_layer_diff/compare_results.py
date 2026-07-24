"""比较研究层三个固定实验的合同结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_fixture, write_result


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--alphamaster", required=True, type=Path)
    parser.add_argument("--qlib", required=True, type=Path)
    parser.add_argument("--vnpy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture, fixture_sha256 = load_fixture(args.fixture)
    alphamaster = _load(args.alphamaster)
    qlib = _load(args.qlib)
    vnpy = _load(args.vnpy)
    results = (alphamaster, qlib, vnpy)
    if any(
        result.get("fixture_sha256") != fixture_sha256
        for result in results
    ):
        raise RuntimeError("输入结果不是由同一份 fixture 生成")
    expected_identity = {
        "alphamaster": ("alphamaster", fixture["case_id"]),
        "qlib": ("qlib", fixture["case_id"]),
        "vnpy": ("vnpy", fixture["case_id"]),
    }
    for name, result in (
        ("alphamaster", alphamaster),
        ("qlib", qlib),
        ("vnpy", vnpy),
    ):
        expected_engine, expected_case = expected_identity[name]
        if (
            result.get("engine") != expected_engine
            or result.get("case_id") != expected_case
        ):
            raise RuntimeError(f"{name} 引擎或样本身份不匹配")
    if (
        qlib.get("engine_version") != "0.9.7"
        or qlib.get("source_commit")
        != "79633dd9506ea689e5400dea0197717b5b3d74b7"
        or vnpy.get("engine_version") != "4.4.0"
        or vnpy.get("source_commit")
        != "1b78494979deb4c4996f6b864f234d9839f2f239"
    ):
        raise RuntimeError("Qlib 或 vn.py 版本身份不匹配")

    checks = {
        "alphamaster_target_return_clock_runtime": bool(
            alphamaster["runtime_method_executed"]
            and alphamaster["formula_matches"]
            and alphamaster["next_open_interval_matches"]
        ),
        "alphamaster_ic_extra_shift_reproduced": bool(
            alphamaster["ic_probe"]["ic_extra_shift_reproduced"]
            and not alphamaster["ic_probe"]["ic_matches_pnl_clock"]
            and alphamaster["ic_probe"][
                "padding_pollution_reproduced"
            ]
        ),
        "qlib_installed_release_source_contract": bool(
            qlib["installed_release"]["contract_matches"]
            and qlib["installed_calendar_shift"][
                "positive_shift_is_earlier"
            ]
            and qlib["method_ast_equivalent"]
            and qlib["calendar_method_ast_equivalent"]
        ),
        "qlib_snapshot_source_contract": bool(
            qlib["source_snapshot"]["contract_matches"]
            and qlib["source_snapshot_calendar_shift"][
                "positive_shift_is_earlier"
            ]
        ),
        "vnpy_historical_constituents_runtime": bool(
            vnpy["runtime_method_executed"]
            and vnpy["inclusive_filter_matches"]
        ),
        "vnpy_overlap_and_empty_filter_risks_reproduced": bool(
            vnpy["overlap_probe"]["duplicates_reproduced"]
            and vnpy["empty_filter_probe"]["empty_dict_leaks_all_rows"]
            and vnpy["empty_filter_probe"][
                "empty_ranges_failure_reproduced"
            ]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"研究层差分实验失败: {checks}")

    write_result(
        args.output,
        {
            "contract_version": fixture["contract_version"],
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "status": (
                "PASS_WITH_RUNTIME_BOUNDARY_AND_ADOPTION_FINDINGS"
            ),
            "checks": checks,
            "evidence_boundary": {
                "alphamaster": "runtime_method",
                "qlib": "installed_and_snapshot_source_contract_only",
                "vnpy": "runtime_method_with_unrelated_report_import_stub",
            },
            "findings": [
                {
                    "severity": "P0",
                    "component": "alphamaster_ic_clock",
                    "finding": (
                        "target-return construction uses t->t+1/t+2 "
                        "open interval, but IC adds one bar and includes "
                        "the padded tail"
                    ),
                },
                {
                    "severity": "P0_ADOPTION_BLOCKER",
                    "component": "vnpy_constituent_filter_contract",
                    "finding": (
                        "overlapping ranges duplicate rows; empty dict "
                        "disables filtering; all-empty ranges raise"
                    ),
                }
            ],
        },
    )


if __name__ == "__main__":
    main()
