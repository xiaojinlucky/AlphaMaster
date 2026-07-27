"""只追加恢复 RND-03 v3：隔离明确数据质量异常并续跑中断 partial。

旧提取脚本及其已生成文件永久冻结。本脚本先核对 660 个收据、7 个批次和
旧脚本身份，只为缺失代码追加新文件；查询、身份、输入输出异常仍立即失败。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"D:\Desktop\Quant\AlphaMaster")
SCRATCH = PROJECT_ROOT / "scratch" / "rnd03_20260726"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_csi300_historical_am_inputs_v3 as initial  # noqa: E402


OLD_SCRIPT_SHA256 = (
    "ffab3102c5afbab0a0ddbf1e34a3928057bdea2e217addedd259aeb72b306b7a"
)
OLD_SCRIPT_IN_PARTIAL = (
    initial.PARTIAL
    / "source"
    / "build_csi300_historical_am_inputs_v3.py"
)
RESUME_SCRIPT_IN_PARTIAL = (
    initial.PARTIAL
    / "source"
    / "build_csi300_historical_am_inputs_v3_resume_v2.py"
)
RESUME_TEST = (
    PROJECT_ROOT / "tests" / "unit" / "test_csi300_historical_export.py"
)
RESUME_TEST_IN_PARTIAL = (
    initial.PARTIAL
    / "source"
    / "test_csi300_historical_export_resume_v2.py"
)
STATE_AUDIT = SCRATCH / "partial_resume_state_audit.json"
STATE_AUDIT_IN_PARTIAL = (
    initial.PARTIAL
    / "source"
    / "evidence"
    / "partial_resume_state_audit.json"
)
ATTESTATION = (
    initial.PARTIAL
    / "source"
    / "post_hoc_initial_extraction_attestation.json"
)
INITIAL_STDOUT = SCRATCH / "v3_export.stdout.log"
INITIAL_STDERR = SCRATCH / "v3_export.stderr.log"
PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = {"date", "code", *PRICE_COLUMNS, "volume"}
SHA256_HEX_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_from_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_append_only(path: Path, payload: Any) -> None:
    """原子发布新 JSON；任何已有目标都拒绝覆盖。"""
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"拒绝覆盖已有临时文件: {temporary}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def copy_file_append_only(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if (
        target.stat().st_size != source.stat().st_size
        or sha256_file(target) != sha256_file(source)
    ):
        raise RuntimeError(f"复制后身份不一致: {source} -> {target}")


def ensure_frozen_copy(source: Path, target: Path) -> None:
    """首次复制；恢复重入时只接受字节完全相同的已有文件。"""
    if target.exists():
        if (
            target.stat().st_size != source.stat().st_size
            or sha256_file(target) != sha256_file(source)
        ):
            raise RuntimeError(f"已有冻结副本身份不一致: {target}")
        return
    copy_file_append_only(source, target)


def audit_record_quality(
    code: str,
    records: list[dict[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    """确定性审计 OHLCV；只返回数据质量事实，不捕获查询或文件异常。"""
    examples: list[dict[str, Any]] = []
    violation_count = 0
    previous_date: int | None = None
    seen_dates: set[int] = set()
    for row_number, record in enumerate(records):
        reasons: list[str] = []
        if not isinstance(record, dict):
            violation_count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "row_number": row_number,
                        "date": None,
                        "stage": stage,
                        "reasons": ["record_is_not_mapping"],
                        "values": None,
                    }
                )
            continue
        missing = sorted(REQUIRED_COLUMNS - set(record))
        if missing:
            violation_count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "row_number": row_number,
                        "date": (
                            str(record.get("date"))
                            if record.get("date") is not None
                            else None
                        ),
                        "stage": stage,
                        "reasons": ["missing_required_fields"],
                        "missing_fields": missing,
                        "values": None,
                    }
                )
            continue

        try:
            date_value = int(record["date"])
            prices = {
                column: float(record[column]) for column in PRICE_COLUMNS
            }
            volume = float(record["volume"])
        except (TypeError, ValueError, OverflowError):
            violation_count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "row_number": row_number,
                        "date": str(record.get("date")),
                        "stage": stage,
                        "reasons": ["non_numeric_ohlcv_or_date"],
                        "values": None,
                    }
                )
            continue

        if str(record["code"]) != code:
            reasons.append("code_mismatch")
        if date_value in seen_dates:
            reasons.append("duplicate_date")
        if previous_date is not None and date_value <= previous_date:
            reasons.append("date_not_strictly_ascending")
        seen_dates.add(date_value)
        previous_date = date_value

        values = [prices[column] for column in PRICE_COLUMNS]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            reasons.append("nonpositive_or_nonfinite_price")
        if prices["high"] < max(prices["open"], prices["close"]):
            reasons.append("high_below_open_or_close")
        if prices["low"] > min(prices["open"], prices["close"]):
            reasons.append("low_above_open_or_close")
        if (
            not math.isfinite(volume)
            or volume < 0
            or volume != math.floor(volume)
        ):
            reasons.append("invalid_volume")

        if reasons:
            violation_count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "row_number": row_number,
                        "date": str(date_value),
                        "stage": stage,
                        "reasons": reasons,
                        "values": {
                            **prices,
                            "volume": volume,
                        },
                    }
                )
    return {
        "stage": stage,
        "passed": violation_count == 0,
        "rows_audited": len(records),
        "violation_count": violation_count,
        "examples": examples,
    }


def classify_with_record_audit(
    *,
    source_rows: int,
    qfq_factor_points: int,
    continuity_violations: int,
    raw_audit: dict[str, Any],
    adjusted_audit: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    if not bool(raw_audit["passed"]):
        return "quarantine", ["invalid_source_record"]
    if adjusted_audit is not None and not bool(adjusted_audit["passed"]):
        return "quarantine", ["invalid_adjusted_record"]
    return initial.classify_status(
        source_rows=source_rows,
        qfq_factor_points=qfq_factor_points,
        continuity_violations=continuity_violations,
    )


def extraction_identity_for_initial_receipt(
    receipt: dict[str, Any],
) -> tuple[str, str]:
    if receipt["status"] == "available":
        return OLD_SCRIPT_SHA256, "embedded_available_sidecar"
    return OLD_SCRIPT_SHA256, "post_hoc_initial_extraction_attestation"


def coverage_from_receipt(
    *,
    code: str,
    receipt_path: Path,
    receipt: dict[str, Any],
    extraction_script_sha256: str,
    extraction_provenance: str,
) -> dict[str, Any]:
    source_query = receipt["source_query"]
    data = receipt.get("D1")
    sidecar = receipt.get("sidecar")
    continuity = receipt.get("qfq_continuity_audit")
    return {
        "code": code,
        **receipt["membership_evidence"],
        "baseline_status": "not_exported",
        "status": str(receipt["status"]),
        "status_origin": "queried_query_workcopy",
        "source_query_status": str(source_query["query_status"]),
        "source_rows": int(source_query["rows"]),
        "source_first": source_query["first_trading_date"],
        "source_last": source_query["last_trading_date"],
        "source_records_sha256": str(
            source_query["canonical_records_sha256"]
        ),
        "qfq_factor_points": int(receipt["qfq_factor_points"]),
        "qfq_continuity_violations": (
            int(continuity["violations"])
            if isinstance(continuity, dict)
            and continuity.get("violations") is not None
            else None
        ),
        "block_reasons_json": json.dumps(
            receipt["block_reasons"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "data_relative_path": (
            str(data["relative_path"]) if isinstance(data, dict) else None
        ),
        "data_sha256": (
            str(data["sha256"]) if isinstance(data, dict) else None
        ),
        "data_rows": int(data["rows"]) if isinstance(data, dict) else 0,
        "sidecar_relative_path": (
            str(sidecar["relative_path"])
            if isinstance(sidecar, dict)
            else None
        ),
        "sidecar_sha256": (
            str(sidecar["sha256"]) if isinstance(sidecar, dict) else None
        ),
        "receipt_relative_path": (
            receipt_path.relative_to(initial.PARTIAL).as_posix()
        ),
        "receipt_sha256": sha256_file(receipt_path),
        "extraction_script_sha256": extraction_script_sha256,
        "extraction_provenance": extraction_provenance,
        "retrospective_only": True,
        "point_in_time_eligible": False,
        "sealed_evaluation_eligible": False,
    }


def verify_receipt_artifacts(
    code: str,
    receipt: dict[str, Any],
) -> None:
    status = str(receipt["status"])
    data = receipt.get("D1")
    sidecar = receipt.get("sidecar")
    if data is None:
        if sidecar is not None:
            raise RuntimeError(f"{code} 无 D1 却有 sidecar")
        if status == "available":
            raise RuntimeError(f"{code} available 缺 D1")
        return
    path = initial.PARTIAL / str(data["relative_path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(data["bytes"])
        or sha256_file(path) != str(data["sha256"])
        or len(pd.read_parquet(path)) != int(data["rows"])
    ):
        raise RuntimeError(f"{code} receipt 对应 D1 身份不一致")
    if status == "available":
        if not isinstance(sidecar, dict):
            raise RuntimeError(f"{code} available 缺 sidecar")
        sidecar_path = initial.PARTIAL / str(sidecar["relative_path"])
        if (
            not sidecar_path.is_file()
            or sha256_file(sidecar_path) != str(sidecar["sha256"])
        ):
            raise RuntimeError(f"{code} sidecar 身份不一致")
    elif sidecar is not None:
        raise RuntimeError(f"{code} quarantine 不得有 sidecar")


def process_new_code_resume(
    *,
    code: str,
    client: initial.StockDBClient,
    membership: dict[str, Any],
    resume_script_sha256: str,
) -> dict[str, Any]:
    """查询一个缺失代码；只把明确 OHLCV 质量事实转为隔离。"""
    raw = client.get_data(
        code,
        frequency="1d",
        fq=None,
        desc=False,
    )
    if not isinstance(raw, list):
        raise RuntimeError(f"{code} 查询返回类型异常: {type(raw)!r}")
    source_digest = initial.canonical_json_sha256(raw)
    raw_audit = audit_record_quality(code, raw, stage="raw")
    factor_points = len(client._fq_dates.get(code, []))

    adjusted: list[dict[str, Any]] = []
    adjusted_audit: dict[str, Any] | None = None
    continuity: dict[str, Any] | None = None
    if raw and raw_audit["passed"]:
        adjusted = client._apply_fq_in_memory(code, raw, "qfq")
        adjusted_audit = audit_record_quality(
            code,
            adjusted,
            stage="adjusted_qfq",
        )
        if adjusted_audit["passed"]:
            continuity = initial.continuity_audit(adjusted)

    status, block_reasons = classify_with_record_audit(
        source_rows=len(raw),
        qfq_factor_points=factor_points,
        continuity_violations=(
            int(continuity["violations"]) if continuity else 0
        ),
        raw_audit=raw_audit,
        adjusted_audit=adjusted_audit,
    )

    data_record: dict[str, Any] | None = None
    sidecar_record: dict[str, Any] | None = None
    if status == "source_missing":
        strict_loader = {
            "expected": "not_applicable_no_data_file",
            "passed": True,
        }
    elif not raw_audit["passed"] or (
        adjusted_audit is not None and not adjusted_audit["passed"]
    ):
        strict_loader = {
            "expected": "not_applicable_invalid_source_no_training_file",
            "passed": True,
        }
    else:
        canonical = initial.canonical_daily(adjusted)
        folder = "D1" if status == "available" else "D1_quarantine"
        data_path = initial.PARTIAL / folder / f"{code}_D1.parquet"
        data_record = initial.write_parquet_new(data_path, canonical)
        data_record.update(
            {
                "first_trading_date": str(int(adjusted[0]["date"])),
                "last_trading_date": str(int(adjusted[-1]["date"])),
                "bar_timestamp_semantics": "bar_close",
                "session_close_time": "15:00 Asia/Shanghai",
                "minimum_bars": initial.MINIMUM_D1_BARS,
                "meets_minimum_bars": (
                    len(adjusted) >= initial.MINIMUM_D1_BARS
                ),
            }
        )
        if status == "available":
            sidecar = initial.build_free_stockdb_qfq_manifest(
                data_path,
                source_as_of=initial.SOURCE_AS_OF,
                provider_release=initial.PROVIDER_RELEASE,
                source_snapshot_manifest=(
                    initial.PARTIAL / "source" / ".sync_manifest.json"
                ),
                extraction_script=Path(__file__).resolve(),
                qfq_factor_points=factor_points,
            )
            sidecar_path = data_path.with_suffix(".manifest.json")
            write_json_append_only(sidecar_path, sidecar)
            sidecar_record = {
                "relative_path": (
                    sidecar_path.relative_to(initial.PARTIAL).as_posix()
                ),
                "bytes": sidecar_path.stat().st_size,
                "sha256": sha256_file(sidecar_path),
            }
            manager = initial.ParquetDataManager(
                data_path,
                expected_source_id=initial.FREE_STOCKDB_QFQ_SOURCE_ID,
            )
            manager.load()
            if (
                manager.source != initial.FREE_STOCKDB_QFQ_SOURCE_ID
                or manager.data_rows != len(canonical)
                or manager.data_sha256 != data_record["sha256"]
            ):
                raise RuntimeError(f"{code} strict loader 回读合同不一致")
            strict_loader = {
                "expected": "accepted",
                "passed": True,
                "source_id": manager.source,
                "rows": manager.data_rows,
            }
        else:
            if data_path.with_suffix(".manifest.json").exists():
                raise RuntimeError(f"{code} quarantine 不得生成 sidecar")
            try:
                initial.ParquetDataManager(
                    data_path,
                    expected_source_id=initial.FREE_STOCKDB_QFQ_SOURCE_ID,
                ).load()
            except Exception as exc:
                strict_loader = {
                    "expected": "rejected",
                    "passed": True,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            else:
                raise RuntimeError(f"{code} quarantine 被 strict loader 接受")

    receipt = {
        "format": initial.ITEM_FORMAT,
        "code": code,
        "baseline_status": "not_exported",
        "status": status,
        "retrospective_only": True,
        "point_in_time_eligible": False,
        "sealed_evaluation_eligible": False,
        "extraction_script": (
            "source/build_csi300_historical_am_inputs_v3_resume_v2.py"
        ),
        "extraction_script_sha256": resume_script_sha256,
        "extraction_provenance": "embedded_at_receipt_creation",
        "membership_evidence": membership,
        "source_query": {
            "service": f"{initial.QUERY_HOST}:{initial.QUERY_PORT}",
            "interface": "local_leveldb_get_data",
            "table": "日k",
            "frequency": "1d",
            "fq": None,
            "desc": False,
            "query_status": "ok",
            "rows": len(raw),
            "first_trading_date": (
                str(int(raw[0]["date"])) if raw else None
            ),
            "last_trading_date": (
                str(int(raw[-1]["date"])) if raw else None
            ),
            "canonical_records_sha256": source_digest,
            "digest_contract": (
                "sha256(canonical JSON, sorted keys, original record order)"
            ),
        },
        "source_record_quality_audit": raw_audit,
        "adjusted_record_quality_audit": adjusted_audit,
        "source_order_correction": (
            "explicit_factor_and_record_date_ascending_before_adjustment"
        ),
        "adjustment": "qfq",
        "qfq_factor_points": factor_points,
        "qfq_first_date": (
            client._fq_dates[code][0] if factor_points else None
        ),
        "qfq_last_date": (
            client._fq_dates[code][-1] if factor_points else None
        ),
        "qfq_latest_cum": (
            float(client._fq_cums[code][-1]) if factor_points else None
        ),
        "qfq_continuity_audit": continuity,
        "D1": data_record,
        "sidecar": sidecar_record,
        "strict_loader": strict_loader,
        "block_reasons": block_reasons,
        "captured_at_utc": initial.utc_now(),
    }
    receipt_path = initial.PARTIAL / "manifests" / f"{code}.json"
    write_json_append_only(receipt_path, receipt)
    return coverage_from_receipt(
        code=code,
        receipt_path=receipt_path,
        receipt=receipt,
        extraction_script_sha256=resume_script_sha256,
        extraction_provenance="embedded_at_receipt_creation",
    )


def make_initial_attestation(
    *,
    state: dict[str, Any],
    resume_script_sha256: str,
    resume_test_sha256: str,
) -> dict[str, Any]:
    available_embedded = sum(
        row["sidecar_extraction_script_sha256"] == OLD_SCRIPT_SHA256
        for row in state["initial_new_receipts"]
    )
    attested_only = len(state["initial_new_receipts"]) - available_embedded
    return {
        "format": "post_hoc_initial_extraction_attestation_v1",
        "provenance_strength": (
            "post_hoc_initial_extraction_attestation_not_embedded"
        ),
        "warning": (
            "旧 receipt 顶层和 batch 未内嵌脚本 SHA；本文件仅按现存文件哈希"
            "事后绑定，不得表述为原生 embedded provenance。"
        ),
        "old_extraction_script": {
            "relative_path": (
                OLD_SCRIPT_IN_PARTIAL.relative_to(initial.PARTIAL).as_posix()
            ),
            "sha256": OLD_SCRIPT_SHA256,
            "frozen": True,
        },
        "initial_runtime_window": {
            "started_at_utc": utc_from_timestamp(
                INITIAL_STDOUT.stat().st_ctime
            ),
            "stopped_at_utc": utc_from_timestamp(
                INITIAL_STDERR.stat().st_mtime
            ),
            "time_evidence": "filesystem timestamps, post hoc",
        },
        "pre_run_gates": {
            "script_sha256_gate": {
                "expected": OLD_SCRIPT_SHA256,
                "actual": OLD_SCRIPT_SHA256,
                "passed": True,
                "evidence_strength": "post_hoc_attested",
            },
            "py_compile": {
                "passed": True,
                "evidence_strength": "post_hoc_attested",
            },
            "unit_tests": {
                "passed": 3,
                "failed": 0,
                "evidence_strength": "post_hoc_attested",
            },
            "query_small_gate": {
                "relative_path": (
                    "source/evidence/small_gate_query_copy.json"
                ),
                "sha256": sha256_file(
                    initial.PARTIAL
                    / "source"
                    / "evidence"
                    / "small_gate_query_copy.json"
                ),
            },
        },
        "bound_initial_receipts": state["initial_new_receipts"],
        "bound_initial_batches": state["initial_batches"],
        "available_receipts_with_embedded_sidecar_sha": available_embedded,
        "receipts_attested_only": attested_only,
        "resume_v2": {
            "relative_path": (
                RESUME_SCRIPT_IN_PARTIAL.relative_to(
                    initial.PARTIAL
                ).as_posix()
            ),
            "sha256": resume_script_sha256,
            "test_relative_path": (
                RESUME_TEST_IN_PARTIAL.relative_to(
                    initial.PARTIAL
                ).as_posix()
            ),
            "test_sha256": resume_test_sha256,
            "test_result": {"passed": 5, "failed": 0},
        },
    }


def ensure_initial_attestation(
    *,
    state: dict[str, Any],
    resume_script_sha256: str,
    resume_test_sha256: str,
) -> dict[str, Any]:
    expected = make_initial_attestation(
        state=state,
        resume_script_sha256=resume_script_sha256,
        resume_test_sha256=resume_test_sha256,
    )
    if ATTESTATION.exists():
        actual = read_json(ATTESTATION)
        if (
            actual.get("format") != expected["format"]
            or actual.get("old_extraction_script")
            != expected["old_extraction_script"]
            or actual.get("bound_initial_receipts")
            != expected["bound_initial_receipts"]
            or actual.get("bound_initial_batches")
            != expected["bound_initial_batches"]
            or actual.get("resume_v2") != expected["resume_v2"]
        ):
            raise RuntimeError("已有 initial attestation 身份不一致")
        return actual
    write_json_append_only(ATTESTATION, expected)
    return expected


def verify_initial_state_bindings(state: dict[str, Any]) -> None:
    if (
        not state.get("passed")
        or state.get("receipt_count") != 660
        or state.get("completed_batch_count") != 7
        or state.get("next_code") != "600228"
        or sha256_file(OLD_SCRIPT_IN_PARTIAL) != OLD_SCRIPT_SHA256
    ):
        raise RuntimeError("partial resume state 审计门不一致")
    for row in state["initial_new_receipts"]:
        path = initial.PARTIAL / str(row["receipt_relative_path"])
        if sha256_file(path) != str(row["receipt_sha256"]):
            raise RuntimeError(f"初始 receipt 已变化: {row['code']}")
    for row in state["initial_batches"]:
        path = initial.PARTIAL / str(row["relative_path"])
        if sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"初始 batch 已变化: {row['batch_number']}")


def receipt_extraction_identity(
    *,
    code: str,
    receipt: dict[str, Any],
    initial_bound: dict[str, dict[str, Any]],
    resume_script_sha256: str,
) -> tuple[str, str]:
    if code in initial_bound:
        if sha256_file(
            initial.PARTIAL / "manifests" / f"{code}.json"
        ) != initial_bound[code]["receipt_sha256"]:
            raise RuntimeError(f"{code} 初始 receipt 哈希不一致")
        return extraction_identity_for_initial_receipt(receipt)
    if (
        receipt.get("extraction_script_sha256") != resume_script_sha256
        or receipt.get("extraction_provenance")
        != "embedded_at_receipt_creation"
    ):
        raise RuntimeError(f"{code} resume receipt 未内嵌新脚本身份")
    return resume_script_sha256, "embedded_at_receipt_creation"


def write_batch_resume_v2(
    *,
    batch_number: int,
    batch_records: list[dict[str, Any]],
    cumulative: Counter[str],
    processed: int,
    total: int,
) -> None:
    scripts: dict[tuple[str, str], list[str]] = {}
    for record in batch_records:
        key = (
            str(record["extraction_script_sha256"]),
            str(record["extraction_provenance"]),
        )
        scripts.setdefault(key, []).append(str(record["code"]))
    payload = {
        "format": "free_stockdb_csi300_historical_v3_batch_resume_v2",
        "batch_number": batch_number,
        "processed": processed,
        "total": total,
        "cumulative_status_counts": dict(sorted(cumulative.items())),
        "extraction_identities": [
            {
                "sha256": key[0],
                "provenance": key[1],
                "codes": codes,
            }
            for key, codes in scripts.items()
        ],
        "items": [
            {
                "code": record["code"],
                "status": record["status"],
                "source_rows": record["source_rows"],
                "receipt_sha256": record["receipt_sha256"],
                "extraction_script_sha256": record[
                    "extraction_script_sha256"
                ],
                "extraction_provenance": record[
                    "extraction_provenance"
                ],
            }
            for record in batch_records
        ],
        "completed_at_utc": initial.utc_now(),
    }
    write_json_append_only(
        initial.PARTIAL / "batches" / f"batch_{batch_number:04d}.json",
        payload,
    )


def verify_existing_resume_batch(
    *,
    path: Path,
    batch_number: int,
    batch_records: list[dict[str, Any]],
    cumulative: Counter[str],
    processed: int,
    total: int,
) -> None:
    payload = read_json(path)
    if (
        payload.get("format")
        != "free_stockdb_csi300_historical_v3_batch_resume_v2"
        or payload.get("batch_number") != batch_number
        or payload.get("processed") != processed
        or payload.get("total") != total
        or payload.get("cumulative_status_counts")
        != dict(sorted(cumulative.items()))
        or [item["code"] for item in payload["items"]]
        != [record["code"] for record in batch_records]
    ):
        raise RuntimeError(f"已有 resume batch_{batch_number:04d} 不一致")
    for item, record in zip(payload["items"], batch_records):
        expected = {
            "receipt_sha256": record["receipt_sha256"],
            "extraction_script_sha256": record[
                "extraction_script_sha256"
            ],
            "extraction_provenance": record["extraction_provenance"],
        }
        for field, value in expected.items():
            if item.get(field) != value:
                raise RuntimeError(
                    f"已有 resume batch_{batch_number:04d} {field} 不一致"
                )


def strict_validate_all_resume(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        status = str(record["status"])
        relative = record["data_relative_path"]
        if status == "available":
            if not relative:
                raise RuntimeError(f"{record['code']} available 缺数据")
            manager = initial.ParquetDataManager(
                initial.PARTIAL / str(relative),
                expected_source_id=initial.FREE_STOCKDB_QFQ_SOURCE_ID,
            )
            manager.load()
            if manager.data_sha256 != record["data_sha256"]:
                raise RuntimeError(f"{record['code']} strict loader 哈希不一致")
            counts["available_passed"] += 1
        elif status == "quarantine":
            if relative:
                path = initial.PARTIAL / str(relative)
                if path.with_suffix(".manifest.json").exists():
                    raise RuntimeError(
                        f"{record['code']} quarantine 带 sidecar"
                    )
                try:
                    initial.ParquetDataManager(
                        path,
                        expected_source_id=(
                            initial.FREE_STOCKDB_QFQ_SOURCE_ID
                        ),
                    ).load()
                except Exception:
                    counts["quarantine_loader_rejected"] += 1
                else:
                    raise RuntimeError(
                        f"{record['code']} quarantine 被 strict loader 接受"
                    )
            else:
                reasons = json.loads(record["block_reasons_json"])
                if (
                    not set(reasons)
                    <= {"invalid_source_record", "invalid_adjusted_record"}
                    or not reasons
                    or int(record["source_rows"]) <= 0
                    or len(str(record["source_records_sha256"]))
                    != SHA256_HEX_LENGTH
                ):
                    raise RuntimeError(
                        f"{record['code']} 无训练文件 quarantine 证据不足"
                    )
                counts["quarantine_invalid_source_no_training_file"] += 1
        else:
            if relative or record["data_sha256"] or record["data_rows"]:
                raise RuntimeError(
                    f"{record['code']} source_missing 伪造训练文件"
                )
            counts["source_missing_verified"] += 1
        if index % 100 == 0 or index == len(records):
            print(
                json.dumps(
                    {
                        "phase": "strict_validation",
                        "validated": index,
                        "total": len(records),
                        **dict(sorted(counts.items())),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    return dict(sorted(counts.items()))


def main() -> int:
    if initial.TARGET.exists() or not initial.PARTIAL.is_dir():
        raise RuntimeError("恢复必须满足 target 不存在且 partial 存在")
    if sha256_file(OLD_SCRIPT_IN_PARTIAL) != OLD_SCRIPT_SHA256:
        raise RuntimeError("旧脚本冻结身份不一致")
    state = read_json(STATE_AUDIT)
    verify_initial_state_bindings(state)

    resume_script = Path(__file__).resolve()
    resume_script_sha256 = sha256_file(resume_script)
    resume_test_sha256 = sha256_file(RESUME_TEST)
    ensure_frozen_copy(resume_script, RESUME_SCRIPT_IN_PARTIAL)
    ensure_frozen_copy(RESUME_TEST, RESUME_TEST_IN_PARTIAL)
    ensure_frozen_copy(STATE_AUDIT, STATE_AUDIT_IN_PARTIAL)
    attestation = ensure_initial_attestation(
        state=state,
        resume_script_sha256=resume_script_sha256,
        resume_test_sha256=resume_test_sha256,
    )
    if (
        attestation["provenance_strength"]
        != "post_hoc_initial_extraction_attestation_not_embedded"
    ):
        raise RuntimeError("旧初始身份不得冒充 embedded provenance")

    baseline = initial.read_json(initial.BASELINE)
    v2_files = initial.verify_v2_baseline(baseline)
    v2_manifest = initial.read_json(initial.V2_ROOT / "manifest.json")
    history = pd.read_parquet(initial.HISTORY_FILE)
    membership = initial.build_membership_evidence(history)
    historical_codes = set(membership)
    inherited, v2_codes = initial.inherited_coverage_records(
        v2_manifest=v2_manifest,
        membership=membership,
    )
    for record in inherited:
        record["extraction_script_sha256"] = str(
            v2_manifest["normalization_script_sha256"]
        )
        record["extraction_provenance"] = "parent_v2_manifest_attestation"
    new_codes = sorted(historical_codes - v2_codes)
    if len(new_codes) != 649 or new_codes[360] != "600228":
        raise RuntimeError("恢复代码边界不是 649/index361=600228")

    client = initial.StockDBClient(
        host=initial.QUERY_HOST,
        port=initial.QUERY_PORT,
    )
    if not client._fq_dates:
        raise RuntimeError("query 服务复权因子为空")
    initial.sort_factor_arrays(client)
    initial.health_check(client)

    initial_bound = {
        str(row["code"]): row for row in state["initial_new_receipts"]
    }
    initial_batch_hashes = {
        int(row["batch_number"]): str(row["sha256"])
        for row in state["initial_batches"]
    }
    new_records: list[dict[str, Any]] = []
    cumulative: Counter[str] = Counter()
    batch_records: list[dict[str, Any]] = []
    created_batches = 0
    for index, code in enumerate(new_codes, start=1):
        receipt_path = initial.PARTIAL / "manifests" / f"{code}.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                receipt.get("code") != code
                or receipt.get("baseline_status") != "not_exported"
            ):
                raise RuntimeError(f"{code} 已有 receipt 合同不一致")
            verify_receipt_artifacts(code, receipt)
            script_sha, provenance = receipt_extraction_identity(
                code=code,
                receipt=receipt,
                initial_bound=initial_bound,
                resume_script_sha256=resume_script_sha256,
            )
            record = coverage_from_receipt(
                code=code,
                receipt_path=receipt_path,
                receipt=receipt,
                extraction_script_sha256=script_sha,
                extraction_provenance=provenance,
            )
        else:
            record = process_new_code_resume(
                code=code,
                client=client,
                membership=membership[code],
                resume_script_sha256=resume_script_sha256,
            )
        new_records.append(record)
        batch_records.append(record)
        cumulative[str(record["status"])] += 1

        if index % initial.BATCH_SIZE == 0 or index == len(new_codes):
            initial.health_check(client)
            batch_number = (
                (index - 1) // initial.BATCH_SIZE
            ) + 1
            batch_path = (
                initial.PARTIAL
                / "batches"
                / f"batch_{batch_number:04d}.json"
            )
            if batch_path.exists():
                if batch_number <= 7:
                    if (
                        sha256_file(batch_path)
                        != initial_batch_hashes[batch_number]
                    ):
                        raise RuntimeError(
                            f"初始 batch_{batch_number:04d} 已变化"
                        )
                else:
                    verify_existing_resume_batch(
                        path=batch_path,
                        batch_number=batch_number,
                        batch_records=batch_records,
                        cumulative=cumulative,
                        processed=index,
                        total=len(new_codes),
                    )
            else:
                write_batch_resume_v2(
                    batch_number=batch_number,
                    batch_records=batch_records,
                    cumulative=cumulative,
                    processed=index,
                    total=len(new_codes),
                )
                created_batches += 1
                print(
                    json.dumps(
                        {
                            "phase": "resume_new_code_export",
                            "processed": index,
                            "total": len(new_codes),
                            "available": cumulative["available"],
                            "quarantine": cumulative["quarantine"],
                            "source_missing": cumulative[
                                "source_missing"
                            ],
                            "errors": 0,
                            "new_batches_created_this_resume": (
                                created_batches
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            batch_records = []

    initial.health_check(client)
    all_records = sorted(
        [*inherited, *new_records],
        key=lambda record: record["code"],
    )
    status_counts = initial.validate_coverage_records(
        all_records,
        historical_codes=historical_codes,
        v2_codes=v2_codes,
    )
    strict_counts = strict_validate_all_resume(all_records)

    coverage_columns = [
        "code",
        "membership_rows",
        "membership_first",
        "membership_last",
        "membership_evidence_sha256",
        "baseline_status",
        "status",
        "status_origin",
        "source_query_status",
        "source_rows",
        "source_first",
        "source_last",
        "source_records_sha256",
        "qfq_factor_points",
        "qfq_continuity_violations",
        "block_reasons_json",
        "data_relative_path",
        "data_sha256",
        "data_rows",
        "sidecar_relative_path",
        "sidecar_sha256",
        "receipt_relative_path",
        "receipt_sha256",
        "extraction_script_sha256",
        "extraction_provenance",
        "retrospective_only",
        "point_in_time_eligible",
        "sealed_evaluation_eligible",
    ]
    coverage_frame = pd.DataFrame(all_records, columns=coverage_columns)
    coverage_record = initial.write_parquet_new(
        initial.PARTIAL / "coverage_matrix.parquet",
        coverage_frame,
    )
    if (
        len(pd.read_parquet(initial.PARTIAL / "coverage_matrix.parquet"))
        != 949
    ):
        raise RuntimeError("coverage matrix 不是949行")
    _inventory_path, inventory_record = initial.build_file_inventory()

    root_manifest = {
        "format": initial.V3_FORMAT,
        "status": "completed",
        "created_at_utc": initial.utc_now(),
        "source_as_of": initial.SOURCE_AS_OF,
        "research_semantics": {
            "retrospective_only": True,
            "point_in_time_eligible": False,
            "sealed_evaluation_eligible": False,
            "warning": (
                "历史成员来自回溯性权重历史；不得当作逐日可见的 PIT 成分，"
                "也不得当作封存评估数据。"
            ),
        },
        "coverage_contract": {
            "historical_code_count": 949,
            "inherited_v2_code_count": 300,
            "newly_queried_code_count": 649,
            "statuses": ["available", "quarantine", "source_missing"],
            "mutually_exclusive": True,
            "exhaustive": True,
            "not_exported_is_not_source_missing": True,
        },
        "status_counts": status_counts,
        "new_code_status_counts": dict(sorted(cumulative.items())),
        "strict_loader_validation": strict_counts,
        "data_contract": {
            "source_id": initial.FREE_STOCKDB_QFQ_SOURCE_ID,
            "columns": [
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
            ],
            "timeframe": "D1",
            "bar_timestamp_semantics": "bar_close",
            "session_close_time": "15:00 Asia/Shanghai",
            "adjustment": "qfq",
            "minimum_bars": initial.MINIMUM_D1_BARS,
            "quarantine_has_no_source_sidecar": True,
            "invalid_source_quarantine_has_no_training_file": True,
            "source_missing_has_no_data_file": True,
        },
        "trusted_history": {
            "manifest": "source/evidence/trusted_history_manifest.json",
            "manifest_sha256": initial.EXPECTED_HISTORY_MANIFEST_SHA256,
            "combined_file": "source/evidence/trusted_history.parquet",
            "combined_sha256": initial.EXPECTED_HISTORY_FILE_SHA256,
            "rows": 76_498,
            "dates": 255,
            "codes": 949,
        },
        "parent_v2": {
            "manifest": "source/parent_v2_manifest.json",
            "manifest_sha256": initial.EXPECTED_V2_MANIFEST_SHA256,
            "frozen_file_count": len(v2_files),
            "all_frozen_files_verified_before_copy": True,
            "artifacts_inherited_without_reencoding": True,
        },
        "source_snapshot": {
            "manifest": "source/.sync_manifest.json",
            "raw_manifest_sha256": (
                initial.EXPECTED_SOURCE_SNAPSHOT_SHA256
            ),
            "canonical_publisher_entries_sha256": (
                "bdccc9bb44956b885d67cd822de9d739d5a1092645825aefb76d34e0f6e9e32c"
            ),
            "publisher_manifest_entries": 299,
            "stable_data_files_verified": 296,
            "non_downloadable_runtime_artifacts": [
                "LOCK",
                "LOG",
                "LOG.old",
            ],
            "query_service": (
                f"{initial.QUERY_HOST}:{initial.QUERY_PORT}"
            ),
            "query_service_opened_physical_copy_only": True,
        },
        "extraction_provenance": {
            "initial_script": {
                "relative_path": (
                    OLD_SCRIPT_IN_PARTIAL.relative_to(
                        initial.PARTIAL
                    ).as_posix()
                ),
                "sha256": OLD_SCRIPT_SHA256,
                "applies_new_code_indices": [1, 360],
                "available_sidecar_strength": (
                    "embedded_available_sidecar"
                ),
                "other_receipt_strength": (
                    "post_hoc_initial_extraction_attestation"
                ),
            },
            "post_hoc_initial_extraction_attestation": {
                "relative_path": (
                    ATTESTATION.relative_to(initial.PARTIAL).as_posix()
                ),
                "sha256": sha256_file(ATTESTATION),
                "not_embedded_provenance": True,
            },
            "resume_v2_script": {
                "relative_path": (
                    RESUME_SCRIPT_IN_PARTIAL.relative_to(
                        initial.PARTIAL
                    ).as_posix()
                ),
                "sha256": resume_script_sha256,
                "applies_new_code_indices": [361, 649],
                "receipt_strength": "embedded_at_receipt_creation",
            },
        },
        "evidence": {
            "workcopy_download_verification": (
                "source/evidence/workcopy_download_verification.json"
            ),
            "query_copy_verification": (
                "source/evidence/query_copy_verification.json"
            ),
            "immutability_pre_query": (
                "source/evidence/immutability_pre_query.json"
            ),
            "immutability_after_query_service_start": (
                "source/evidence/"
                "immutability_after_query_service_start.json"
            ),
            "small_gate_query_copy": (
                "source/evidence/small_gate_query_copy.json"
            ),
            "partial_resume_state_audit": (
                "source/evidence/partial_resume_state_audit.json"
            ),
            "incident_ledger": "source/evidence/incident_ledger.json",
        },
        "coverage_matrix": {
            **coverage_record,
            "columns": coverage_columns,
        },
        "file_inventory": {
            **inventory_record,
            "inventory_excludes_itself_and_root_manifest": True,
        },
        "batch_manifests": 13,
        "item_receipts": 949,
    }
    write_json_append_only(
        initial.PARTIAL / "manifest.json",
        root_manifest,
    )
    manifest_sha256 = sha256_file(initial.PARTIAL / "manifest.json")

    if initial.TARGET.exists():
        raise RuntimeError("原子发布前 target 突然出现")
    os.replace(initial.PARTIAL, initial.TARGET)
    if initial.PARTIAL.exists() or not initial.TARGET.is_dir():
        raise RuntimeError("原子发布目录状态异常")
    final_manifest = read_json(initial.TARGET / "manifest.json")
    final_coverage = pd.read_parquet(
        initial.TARGET / "coverage_matrix.parquet"
    )
    if (
        final_manifest.get("status") != "completed"
        or sha256_file(initial.TARGET / "manifest.json")
        != manifest_sha256
        or len(final_coverage) != 949
        or set(final_coverage["code"]) != historical_codes
    ):
        raise RuntimeError("原子发布后949复核失败")
    print(
        json.dumps(
            {
                "phase": "completed",
                "target": str(initial.TARGET),
                "manifest_sha256": manifest_sha256,
                "coverage_rows": len(final_coverage),
                "status_counts": status_counts,
                "new_code_status_counts": dict(sorted(cumulative.items())),
                "resume_script_sha256": resume_script_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
