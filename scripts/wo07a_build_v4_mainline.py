"""按原始 WO-AM-07A 主线构建 26 只股票的独立 qfq v4 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wo07a_mainline_core import (
    build_qfq,
    decode_factor,
    decode_history,
    make_d1,
    ohlc_relation_violation_dates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = Path(
    r"G:\QuantData\free-stockdb\am_exports"
    r"\20260726_csi300_historical_am_inputs_v3"
)
OUTPUT_ROOT = Path(
    r"G:\QuantData\free-stockdb\am_exports"
    r"\20260727_csi300_qfq_repair_v4"
)
CAPTURE_ROOT = (
    PROJECT_ROOT
    / "scratch"
    / "goal_am_custody"
    / "wo07a_adjudication_capture"
    / "wo07a-20260729T005459992563Z-10ff978b"
)
V3_MANIFEST_SHA256 = (
    "e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c"
)
REPLAY_START = "2006-10-31"
SOURCE_AS_OF = "2026-07-24"
MEMBERSHIP_DATES = (
    "2026-01-30",
    "2026-02-27",
    "2026-03-31",
    "2026-04-30",
    "2026-05-29",
    "2026-06-30",
)
MINIMUM_BARS_CODES = {"001280", "001391", "600930"}
EXPECTED_REPAIR_CODES = {
    "000001",
    "000408",
    "000538",
    "000651",
    "000657",
    "000661",
    "000876",
    "002074",
    "002466",
    "600039",
    "600089",
    "600219",
    "600515",
    "600660",
    "600674",
    "600690",
    "600741",
    "600760",
    "600803",
    "600886",
    "600887",
    "601607",
    "688047",
    "688506",
    "688521",
    "688981",
}


class MainlineBuildError(RuntimeError):
    pass


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def derive_scope(v3_root: Path) -> tuple[list[str], list[str]]:
    history = pd.read_parquet(
        v3_root / "source" / "evidence" / "trusted_history.parquet"
    )
    coverage = pd.read_parquet(v3_root / "coverage_matrix.parquet")
    status_by_code = coverage.set_index("code")["status"].to_dict()
    recent = history[history["date"].astype(str).isin(MEMBERSHIP_DATES)]
    non_available = sorted(
        {
            str(code)
            for code in recent["code"]
            if status_by_code.get(str(code)) != "available"
        }
    )
    repair = sorted(set(non_available) - MINIMUM_BARS_CODES)
    if len(non_available) != 29 or set(repair) != EXPECTED_REPAIR_CODES:
        raise MainlineBuildError(
            f"07A 范围漂移：non_available={len(non_available)} repair={repair}"
        )
    return non_available, repair


def preflight(
    v3_root: Path,
    output_root: Path,
    capture_root: Path,
    *,
    require_absent_output: bool,
) -> tuple[list[str], list[dict[str, object]]]:
    if (
        sha256_bytes((v3_root / "manifest.json").read_bytes())
        != V3_MANIFEST_SHA256
    ):
        raise MainlineBuildError("v3 manifest 哈希漂移")
    if not capture_root.is_dir():
        raise MainlineBuildError(f"冻结 capture 不存在：{capture_root}")
    if require_absent_output and output_root.exists():
        raise MainlineBuildError(f"目标目录已存在，拒绝覆盖：{output_root}")

    _, repair_codes = derive_scope(v3_root)
    records: list[dict[str, object]] = []
    for code in repair_codes:
        source = capture_root / "source_snapshots" / code
        history_blob = (source / "sina_history.js").read_bytes()
        factor_blob = (source / "sina_qfq_factor.js").read_bytes()
        history = decode_history(history_blob, code)
        factors = decode_factor(factor_blob, code)
        history = history[
            history["date"].between(REPLAY_START, SOURCE_AS_OF)
        ].reset_index(drop=True)
        qfq, switches, large_switches, violations = build_qfq(history, factors)
        ohlc_violations = ohlc_relation_violation_dates(qfq)
        d1 = make_d1(qfq)
        if violations or ohlc_violations:
            raise MainlineBuildError(
                f"{code} 预检失败：continuity={violations} ohlc={ohlc_violations}"
            )
        records.append(
            {
                "code": code,
                "history_blob": history_blob,
                "factor_blob": factor_blob,
                "d1": d1,
                "rows": len(d1),
                "first_date": str(history["date"].iloc[0].date()),
                "last_date": str(history["date"].iloc[-1].date()),
                "factor_switches": list(switches),
                "large_factor_switches": list(large_switches),
            }
        )
    return repair_codes, records


def build(
    *,
    v3_root: Path,
    output_root: Path,
    capture_root: Path,
    check_only: bool,
) -> None:
    repair_codes, records = preflight(
        v3_root,
        output_root,
        capture_root,
        require_absent_output=not check_only,
    )
    print(
        f"PRECHECK PASS repair_codes={len(repair_codes)} "
        "continuity_violations=0 ohlc_violations=0"
    )
    if check_only:
        return

    output_root.mkdir(parents=False, exist_ok=False)
    (output_root / "D1").mkdir()
    (output_root / "audits").mkdir()
    (output_root / "source").mkdir()
    inventory: list[dict[str, object]] = []

    for record in records:
        code = str(record["code"])
        d1_path = output_root / "D1" / f"{code}_D1.parquet"
        record["d1"].to_parquet(d1_path, index=False)

        source_dir = output_root / "source" / code
        source_dir.mkdir()
        history_path = source_dir / "sina_history.js"
        factor_path = source_dir / "sina_qfq_factor.js"
        history_path.write_bytes(record["history_blob"])
        factor_path.write_bytes(record["factor_blob"])

        audit = {
            "format": "wo_am07a_repair_audit_v4_mainline",
            "code": code,
            "source_id": "sina_frozen_capture_qfq_20260729",
            "source_capture_id": capture_root.name,
            "replay_start": REPLAY_START,
            "source_as_of": SOURCE_AS_OF,
            "rows": record["rows"],
            "first_date": record["first_date"],
            "last_date": record["last_date"],
            "factor_switch_count": len(record["factor_switches"]),
            "large_factor_switch_count": len(record["large_factor_switches"]),
            "continuity_violations": 0,
            "ohlc_violations": 0,
            "inputs": {
                "sina_history_sha256": sha256_bytes(record["history_blob"]),
                "sina_qfq_factor_sha256": sha256_bytes(record["factor_blob"]),
            },
            "output": {
                "relative_path": f"D1/{code}_D1.parquet",
                "sha256": sha256_bytes(d1_path.read_bytes()),
                "bytes": d1_path.stat().st_size,
            },
        }
        audit_path = output_root / "audits" / f"{code}.json"
        audit_path.write_bytes(canonical_json(audit))

        for path in (d1_path, audit_path, history_path, factor_path):
            inventory.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_bytes(path.read_bytes()),
                }
            )

    manifest = {
        "format": "free_stockdb_csi300_qfq_repair_v4_mainline",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "sina_frozen_capture_qfq_20260729",
        "source_capture_id": capture_root.name,
        "source_capture_manifest_sha256": sha256_bytes(
            (capture_root / "capture_manifest.json").read_bytes()
        ),
        "parent_v3_manifest_sha256": V3_MANIFEST_SHA256,
        "research_semantics": {
            "retrospective_only": True,
            "point_in_time_eligible": False,
            "sealed_evaluation_eligible": False,
        },
        "scope": {
            "membership_dates": list(MEMBERSHIP_DATES),
            "non_available_union_count": 29,
            "minimum_bars_adjudicated_codes": sorted(MINIMUM_BARS_CODES),
            "repair_count": len(repair_codes),
            "repair_codes": repair_codes,
        },
        "data_contract": {
            "adjustment": "qfq",
            "timeframe": "D1",
            "bar_timestamp_semantics": "bar_close",
            "session_close_time": "15:00 Asia/Shanghai",
            "replay_start": REPLAY_START,
            "source_as_of": SOURCE_AS_OF,
            "columns": ["time", "open", "high", "low", "close", "tick_volume"],
        },
        "acceptance": {
            "repair_codes": len(repair_codes),
            "continuity_violations": 0,
            "ohlc_violations": 0,
        },
        "inventory": sorted(inventory, key=lambda item: item["relative_path"]),
    }
    (output_root / "manifest.json").write_bytes(canonical_json(manifest))
    print(
        f"BUILD PASS output={output_root} repair_codes={len(repair_codes)} "
        f"files={len(inventory) + 1} continuity_violations=0"
    )
    print(f"V3_MANIFEST_SHA256={V3_MANIFEST_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 WO-AM-07A 26 只 qfq v4")
    parser.add_argument("--v3", type=Path, default=V3_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--capture", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    build(
        v3_root=args.v3.resolve(),
        output_root=args.output.resolve(),
        capture_root=args.capture.resolve(),
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()
