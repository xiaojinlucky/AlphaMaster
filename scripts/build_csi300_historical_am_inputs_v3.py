"""把可信沪深300历史成员扩展为可审计的 AlphaMaster D1 输入。

这是一次回溯性覆盖审计，不是时点（PIT）成分数据，也不是封存评估数据。
脚本只查询独立的 FreeStockDB query 工作副本，继承 v2 的 300 个冻结产物，
再处理此前未导出的 649 个历史代码，最终原子发布 949 代码覆盖矩阵。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\Desktop\Quant\AlphaMaster")
SCRATCH = PROJECT_ROOT / "scratch" / "rnd03_20260726"
FREE_STOCKDB_ROOT = Path(r"G:\QuantData\free-stockdb")
V2_ROOT = (
    FREE_STOCKDB_ROOT
    / "am_exports"
    / "20260724_csi300_am_inputs_v2"
)
TARGET = (
    FREE_STOCKDB_ROOT
    / "am_exports"
    / "20260726_csi300_historical_am_inputs_v3"
)
PARTIAL = TARGET.with_name(f"{TARGET.name}.partial")
HISTORY_ROOT = (
    FREE_STOCKDB_ROOT
    / "online_api"
    / "snapshots"
    / "20260630_csi300_weight_history_v1"
)
HISTORY_FILE = HISTORY_ROOT / "csi300_weight_history.parquet"
HISTORY_MANIFEST = HISTORY_ROOT / "manifest.json"
BASELINE = SCRATCH / "baseline.json"
QUERY_SDK = (
    FREE_STOCKDB_ROOT
    / "rnd03_workcopies"
    / "20260726_csi300_historical_v3_leveldb_copy"
    / "query_stockdb"
    / "pybao"
)
QUERY_SOURCE_SNAPSHOT = V2_ROOT / "source" / ".sync_manifest.json"
QUERY_HOST = "127.0.0.1"
QUERY_PORT = 17910
QUERY_HEALTH_CODE = "601872"
QUERY_HEALTH_ROWS = 4645
SOURCE_AS_OF = "2026-07-24"
PROVIDER_RELEASE = "v0.2.1-more-power"
V3_FORMAT = "free_stockdb_csi300_historical_am_inputs_v3"
ITEM_FORMAT = "free_stockdb_csi300_historical_am_inputs_v3_item"
MINIMUM_D1_BARS = 484
PRICE_COLUMNS = ("open", "high", "low", "close")
BATCH_SIZE = 50
EXPECTED_HISTORY_ROWS = 76_498
EXPECTED_HISTORY_DATES = 255
EXPECTED_HISTORY_CODES = 949
EXPECTED_V2_CODES = 300
EXPECTED_NEW_CODES = 649
EXPECTED_V2_MANIFEST_SHA256 = (
    "bdc3d3ae775267bc3cfdb3f5682f090bb010769d7765c47abe9a8d3807715be5"
)
EXPECTED_HISTORY_MANIFEST_SHA256 = (
    "1a7f10aa803ab2e3656c449c4cb6dec5b6da0f77df5d76b886dac6a4504ffcc3"
)
EXPECTED_HISTORY_FILE_SHA256 = (
    "68e68f13755c45e7e2981db0ca6bc0536e8363929ff4880bdb1afb678092dd8e"
)
EXPECTED_SOURCE_SNAPSHOT_SHA256 = (
    "a668c49f4e581de43b9fa8f1767d92c88eff13a0c4092b0daf318142695e5240"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(QUERY_SDK))

from data_pipeline.a_share_data import (  # noqa: E402
    validate_canonical_training_frame,
)
from data_pipeline.dataset_contracts import (  # noqa: E402
    FREE_STOCKDB_QFQ_SOURCE_ID,
)
from data_pipeline.free_stockdb_data import (  # noqa: E402
    build_free_stockdb_qfq_manifest,
)
from data_pipeline.parquet_manager import ParquetDataManager  # noqa: E402
from stock_sdk import StockDBClient  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: Any) -> None:
    """只发布新文件；发现同名文件时立即失败。"""
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_parquet_new(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """原子写入一个新 Parquet，并复读验证内容。"""
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在: {temporary}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    persisted = pd.read_parquet(temporary)
    if not persisted.equals(frame):
        raise RuntimeError(f"{path.name} 写入后复读不一致")
    os.replace(temporary, path)
    return {
        "relative_path": path.relative_to(PARTIAL).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(frame),
    }


def copy_file_new(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if (
        target.stat().st_size != source.stat().st_size
        or sha256_file(target) != sha256_file(source)
    ):
        raise RuntimeError(f"复制后校验失败: {source} -> {target}")


def sort_factor_arrays(client: StockDBClient) -> None:
    """发行 SDK 误把 LevelDB 顺序当升序；复权前必须显式排序。"""
    for code in list(client._fq_dates):
        pairs = sorted(
            zip(client._fq_dates[code], client._fq_cums[code]),
            key=lambda pair: pair[0],
        )
        if len({date for date, _cum in pairs}) != len(pairs):
            raise RuntimeError(f"{code} 复权因子日期重复")
        client._fq_dates[code] = [date for date, _cum in pairs]
        client._fq_cums[code] = [float(cum) for _date, cum in pairs]


def validate_records(
    code: str,
    records: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    required = {"date", "code", *PRICE_COLUMNS, "volume"}
    dates: list[int] = []
    for row_number, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"{code} {label} 第{row_number}行不是对象")
        missing = required - set(record)
        if missing:
            raise RuntimeError(
                f"{code} {label} 第{row_number}行缺字段: {sorted(missing)}"
            )
        if str(record["code"]) != code:
            raise RuntimeError(f"{code} {label} 混入 {record['code']}")
        dates.append(int(record["date"]))
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise RuntimeError(f"{code} {label} 时间非严格升序")
    if not records:
        return
    prices = np.asarray(
        [[record[column] for column in PRICE_COLUMNS] for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise RuntimeError(f"{code} {label} 含非法价格")
    if np.any(prices[:, 1] < np.maximum(prices[:, 0], prices[:, 3])):
        raise RuntimeError(f"{code} {label} high 非法")
    if np.any(prices[:, 2] > np.minimum(prices[:, 0], prices[:, 3])):
        raise RuntimeError(f"{code} {label} low 非法")
    volumes = np.asarray([record["volume"] for record in records], dtype=np.float64)
    if (
        not np.isfinite(volumes).all()
        or np.any(volumes < 0)
        or np.any(volumes != np.floor(volumes))
    ):
        raise RuntimeError(f"{code} {label} volume 非法")


def continuity_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """用记录内 pct_chg 检查复权后相邻收盘收益的数量级断裂。"""
    violations: list[dict[str, Any]] = []
    max_difference = 0.0
    for previous, current in zip(records, records[1:]):
        previous_close = float(previous["close"])
        pct_chg = current.get("pct_chg")
        if previous_close <= 0 or pct_chg is None:
            continue
        calculated = (
            float(current["close"]) / previous_close - 1.0
        ) * 100.0
        difference = abs(calculated - float(pct_chg))
        max_difference = max(max_difference, difference)
        if difference > 10.0:
            violations.append(
                {
                    "date": str(int(current["date"])),
                    "calculated_close_return_pct": round(calculated, 6),
                    "record_pct_chg": float(pct_chg),
                    "difference_percentage_points": round(difference, 6),
                }
            )
    return {
        "threshold_percentage_points": 10.0,
        "violations": len(violations),
        "max_difference_percentage_points": round(max_difference, 6),
        "examples": violations[:5],
    }


def classify_status(
    *,
    source_rows: int,
    qfq_factor_points: int,
    continuity_violations: int,
    minimum_bars: int = MINIMUM_D1_BARS,
) -> tuple[str, list[str]]:
    """给非异常查询结果分配唯一状态。"""
    if source_rows == 0:
        return "source_missing", ["source_missing"]
    block_reasons: list[str] = []
    if source_rows < minimum_bars:
        block_reasons.append("minimum_bars")
    if qfq_factor_points == 0:
        block_reasons.append("missing_qfq_factor_points")
    if continuity_violations:
        block_reasons.append("qfq_continuity")
    return ("quarantine", block_reasons) if block_reasons else ("available", [])


def canonical_daily(records: list[dict[str, Any]]) -> pd.DataFrame:
    dates = pd.to_datetime(
        pd.Series([str(int(record["date"])) for record in records]),
        format="%Y%m%d",
        errors="raise",
    )
    local_close = (
        dates + pd.Timedelta(hours=15)
    ).dt.tz_localize("Asia/Shanghai")
    unix_seconds = (
        local_close.dt.tz_convert("UTC")
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy(dtype=np.int64)
        // 1_000_000_000
    )
    frame = pd.DataFrame(
        {
            "time": unix_seconds,
            "open": np.asarray(
                [row["open"] for row in records],
                dtype=np.float32,
            ),
            "high": np.asarray(
                [row["high"] for row in records],
                dtype=np.float32,
            ),
            "low": np.asarray(
                [row["low"] for row in records],
                dtype=np.float32,
            ),
            "close": np.asarray(
                [row["close"] for row in records],
                dtype=np.float32,
            ),
            "tick_volume": np.asarray(
                [row["volume"] for row in records],
                dtype=np.int64,
            ),
        }
    )
    validate_canonical_training_frame(frame)
    return frame


def _membership_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = frame.sort_values(
        ["date", "display_name", "weight"],
        kind="stable",
    )
    return [
        {
            "code": str(row.code),
            "date": pd.Timestamp(row.date).date().isoformat(),
            "weight": float(row.weight),
            "display_name": str(row.display_name),
        }
        for row in ordered.itertuples(index=False)
    ]


def build_membership_evidence(
    history: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    required = {"code", "date", "weight", "display_name"}
    if set(history.columns) != required:
        raise RuntimeError(
            f"可信历史列合同不一致: {list(history.columns)}"
        )
    normalized = history.copy()
    normalized["code"] = normalized["code"].astype(str).str[:6]
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="raise",
    )
    if normalized.duplicated(["code", "date"]).any():
        raise RuntimeError("可信历史存在重复 code/date")
    evidence: dict[str, dict[str, Any]] = {}
    for code, group in normalized.groupby("code", sort=True):
        rows = _membership_rows(group)
        evidence[code] = {
            "membership_rows": len(rows),
            "membership_first": rows[0]["date"],
            "membership_last": rows[-1]["date"],
            "membership_evidence_sha256": canonical_json_sha256(rows),
        }
    return evidence


def validate_coverage_records(
    records: Iterable[dict[str, Any]],
    *,
    historical_codes: set[str],
    v2_codes: set[str],
) -> dict[str, int]:
    materialized = list(records)
    codes = [str(record["code"]) for record in materialized]
    if len(codes) != len(set(codes)):
        raise RuntimeError("覆盖矩阵代码不唯一")
    if set(codes) != historical_codes:
        missing = sorted(historical_codes - set(codes))
        extra = sorted(set(codes) - historical_codes)
        raise RuntimeError(
            f"覆盖矩阵不是可信历史精确全集: missing={missing}, extra={extra}"
        )
    allowed = {"available", "quarantine", "source_missing"}
    statuses = [str(record["status"]) for record in materialized]
    if not set(statuses) <= allowed:
        raise RuntimeError(f"覆盖矩阵含非法状态: {sorted(set(statuses) - allowed)}")
    for record in materialized:
        code = str(record["code"])
        expected_baseline = (
            record["status"] if code in v2_codes else "not_exported"
        )
        if code in v2_codes:
            expected_baseline = str(record["baseline_status"])
            if expected_baseline not in {"available", "quarantine"}:
                raise RuntimeError(f"{code} v2 基线状态非法")
        elif record["baseline_status"] != "not_exported":
            raise RuntimeError(f"{code} 未导出基线被错误改写")
        if not bool(record["retrospective_only"]):
            raise RuntimeError(f"{code} 缺少 retrospective_only 标记")
        if bool(record["point_in_time_eligible"]):
            raise RuntimeError(f"{code} 不得声明 PIT 可用")
        if bool(record["sealed_evaluation_eligible"]):
            raise RuntimeError(f"{code} 不得声明 sealed evaluation 可用")
    return dict(sorted(Counter(statuses).items()))


def health_check(client: StockDBClient) -> None:
    records = client.get_data(
        QUERY_HEALTH_CODE,
        frequency="1d",
        fq=None,
        desc=False,
    )
    if not isinstance(records, list) or len(records) != QUERY_HEALTH_ROWS:
        raise RuntimeError(
            "query 服务健康检查失败；拒绝把连接故障归类为 source_missing"
        )
    validate_records(QUERY_HEALTH_CODE, records, label="健康检查D1")


def verify_v2_baseline(
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    if sha256_file(V2_ROOT / "manifest.json") != EXPECTED_V2_MANIFEST_SHA256:
        raise RuntimeError("v2 根 manifest 相对冻结基线已变化")
    records = baseline["v2_files"]
    if len(records) != 877:
        raise RuntimeError("v2 冻结文件清单不是 877 项")
    for row in records:
        path = V2_ROOT / row["relative_path"]
        stat = path.stat()
        if (
            stat.st_size != int(row["bytes"])
            or stat.st_mtime_ns != int(row["mtime_ns"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"v2 冻结文件已变化: {row['relative_path']}")
    return records


def inherit_v2_files(records: list[dict[str, Any]]) -> None:
    for row in records:
        relative_path = str(row["relative_path"])
        source = V2_ROOT / relative_path
        target = (
            PARTIAL / "source" / "parent_v2_manifest.json"
            if relative_path == "manifest.json"
            else PARTIAL / relative_path
        )
        copy_file_new(source, target)


def copy_evidence() -> None:
    evidence_files = [
        "baseline.json",
        "incident_ledger.json",
        "source_drift_after_service.json",
        "workcopy_download_verification.json",
        "query_copy_verification.json",
        "immutability_pre_query.json",
        "immutability_after_query_service_start.json",
        "small_gate_query_copy.json",
    ]
    for name in evidence_files:
        copy_file_new(
            SCRATCH / name,
            PARTIAL / "source" / "evidence" / name,
        )
    copy_file_new(
        SCRATCH
        / "remote_manifest_probe"
        / "protocol_capture"
        / "response_0001.bin",
        PARTIAL / "source" / "remote_sync_manifest.json",
    )
    copy_file_new(
        HISTORY_MANIFEST,
        PARTIAL / "source" / "evidence" / "trusted_history_manifest.json",
    )
    copy_file_new(
        HISTORY_FILE,
        PARTIAL / "source" / "evidence" / "trusted_history.parquet",
    )
    copy_file_new(
        Path(__file__).resolve(),
        PARTIAL / "source" / Path(__file__).name,
    )


def inherited_coverage_records(
    *,
    v2_manifest: dict[str, Any],
    membership: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    items = v2_manifest["items"]
    if len(items) != EXPECTED_V2_CODES:
        raise RuntimeError("v2 items 不是 300 项")
    records: list[dict[str, Any]] = []
    codes: set[str] = set()
    for item in items:
        code = str(item["code"])
        if code in codes:
            raise RuntimeError(f"v2 代码重复: {code}")
        codes.add(code)
        if code not in membership:
            raise RuntimeError(f"v2 代码不在可信历史: {code}")
        status = "available" if bool(item["am_ready"]) else "quarantine"
        data = item["D1"]
        sidecar = item["sidecar"]
        receipt = PARTIAL / "manifests" / f"{code}.json"
        if sha256_file(receipt) != sha256_file(
            V2_ROOT / "manifests" / f"{code}.json"
        ):
            raise RuntimeError(f"{code} v2 receipt 继承不一致")
        records.append(
            {
                "code": code,
                **membership[code],
                "baseline_status": status,
                "status": status,
                "status_origin": "inherited_v2",
                "source_query_status": "not_requeried_inherited_v2",
                "source_rows": int(data["rows"]),
                "source_first": str(data["first_trading_date"]),
                "source_last": str(data["last_trading_date"]),
                "source_records_sha256": None,
                "qfq_factor_points": int(item["qfq_factor_points"]),
                "qfq_continuity_violations": int(
                    item["qfq_continuity_audit"]["violations"]
                ),
                "block_reasons_json": json.dumps(
                    item["block_reasons"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "data_relative_path": str(data["relative_path"]),
                "data_sha256": str(data["sha256"]),
                "data_rows": int(data["rows"]),
                "sidecar_relative_path": (
                    str(sidecar["relative_path"]) if sidecar else None
                ),
                "sidecar_sha256": (
                    str(sidecar["sha256"]) if sidecar else None
                ),
                "receipt_relative_path": (
                    receipt.relative_to(PARTIAL).as_posix()
                ),
                "receipt_sha256": sha256_file(receipt),
                "retrospective_only": True,
                "point_in_time_eligible": False,
                "sealed_evaluation_eligible": False,
            }
        )
    return records, codes


def process_new_code(
    *,
    code: str,
    client: StockDBClient,
    membership: dict[str, Any],
) -> dict[str, Any]:
    raw = client.get_data(
        code,
        frequency="1d",
        fq=None,
        desc=False,
    )
    if not isinstance(raw, list):
        raise RuntimeError(f"{code} 查询返回类型异常: {type(raw)!r}")
    validate_records(code, raw, label="D1原始")
    raw_hash = canonical_json_sha256(raw)
    factor_points = len(client._fq_dates.get(code, []))

    adjusted: list[dict[str, Any]] = []
    continuity = {
        "threshold_percentage_points": 10.0,
        "violations": 0,
        "max_difference_percentage_points": 0.0,
        "examples": [],
    }
    if raw:
        adjusted = client._apply_fq_in_memory(code, raw, "qfq")
        validate_records(code, adjusted, label="D1前复权")
        continuity = continuity_audit(adjusted)
    status, block_reasons = classify_status(
        source_rows=len(raw),
        qfq_factor_points=factor_points,
        continuity_violations=int(continuity["violations"]),
    )

    data_record: dict[str, Any] | None = None
    sidecar_record: dict[str, Any] | None = None
    strict_loader: dict[str, Any]
    if status == "source_missing":
        strict_loader = {
            "expected": "not_applicable_no_data_file",
            "passed": True,
        }
    else:
        canonical = canonical_daily(adjusted)
        folder = "D1" if status == "available" else "D1_quarantine"
        data_path = PARTIAL / folder / f"{code}_D1.parquet"
        data_record = write_parquet_new(data_path, canonical)
        data_record.update(
            {
                "first_trading_date": str(int(adjusted[0]["date"])),
                "last_trading_date": str(int(adjusted[-1]["date"])),
                "bar_timestamp_semantics": "bar_close",
                "session_close_time": "15:00 Asia/Shanghai",
                "minimum_bars": MINIMUM_D1_BARS,
                "meets_minimum_bars": len(adjusted) >= MINIMUM_D1_BARS,
            }
        )
        if status == "available":
            sidecar = build_free_stockdb_qfq_manifest(
                data_path,
                source_as_of=SOURCE_AS_OF,
                provider_release=PROVIDER_RELEASE,
                source_snapshot_manifest=(
                    PARTIAL / "source" / ".sync_manifest.json"
                ),
                extraction_script=Path(__file__).resolve(),
                qfq_factor_points=factor_points,
            )
            sidecar_path = data_path.with_suffix(".manifest.json")
            write_json_new(sidecar_path, sidecar)
            sidecar_record = {
                "relative_path": sidecar_path.relative_to(PARTIAL).as_posix(),
                "bytes": sidecar_path.stat().st_size,
                "sha256": sha256_file(sidecar_path),
            }
            manager = ParquetDataManager(
                data_path,
                expected_source_id=FREE_STOCKDB_QFQ_SOURCE_ID,
            )
            manager.load()
            if (
                manager.source != FREE_STOCKDB_QFQ_SOURCE_ID
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
            sidecar_path = data_path.with_suffix(".manifest.json")
            if sidecar_path.exists():
                raise RuntimeError(f"{code} quarantine 不得生成 sidecar")
            try:
                ParquetDataManager(
                    data_path,
                    expected_source_id=FREE_STOCKDB_QFQ_SOURCE_ID,
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
        "format": ITEM_FORMAT,
        "code": code,
        "baseline_status": "not_exported",
        "status": status,
        "retrospective_only": True,
        "point_in_time_eligible": False,
        "sealed_evaluation_eligible": False,
        "membership_evidence": membership,
        "source_query": {
            "service": f"{QUERY_HOST}:{QUERY_PORT}",
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
            "canonical_records_sha256": raw_hash,
        },
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
        "captured_at_utc": utc_now(),
    }
    receipt_path = PARTIAL / "manifests" / f"{code}.json"
    write_json_new(receipt_path, receipt)
    return {
        "code": code,
        **membership,
        "baseline_status": "not_exported",
        "status": status,
        "status_origin": "queried_query_workcopy",
        "source_query_status": "ok",
        "source_rows": len(raw),
        "source_first": (
            str(int(raw[0]["date"])) if raw else None
        ),
        "source_last": (
            str(int(raw[-1]["date"])) if raw else None
        ),
        "source_records_sha256": raw_hash,
        "qfq_factor_points": factor_points,
        "qfq_continuity_violations": int(continuity["violations"]),
        "block_reasons_json": json.dumps(
            block_reasons,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "data_relative_path": (
            str(data_record["relative_path"]) if data_record else None
        ),
        "data_sha256": (
            str(data_record["sha256"]) if data_record else None
        ),
        "data_rows": (
            int(data_record["rows"]) if data_record else 0
        ),
        "sidecar_relative_path": (
            str(sidecar_record["relative_path"])
            if sidecar_record
            else None
        ),
        "sidecar_sha256": (
            str(sidecar_record["sha256"]) if sidecar_record else None
        ),
        "receipt_relative_path": (
            receipt_path.relative_to(PARTIAL).as_posix()
        ),
        "receipt_sha256": sha256_file(receipt_path),
        "retrospective_only": True,
        "point_in_time_eligible": False,
        "sealed_evaluation_eligible": False,
    }


def write_batch(
    *,
    batch_number: int,
    batch_records: list[dict[str, Any]],
    cumulative: Counter[str],
    processed: int,
    total: int,
) -> None:
    write_json_new(
        PARTIAL / "batches" / f"batch_{batch_number:04d}.json",
        {
            "format": "free_stockdb_csi300_historical_v3_batch_v1",
            "batch_number": batch_number,
            "processed": processed,
            "total": total,
            "cumulative_status_counts": dict(sorted(cumulative.items())),
            "items": [
                {
                    "code": record["code"],
                    "status": record["status"],
                    "source_rows": record["source_rows"],
                    "receipt_sha256": record["receipt_sha256"],
                }
                for record in batch_records
            ],
            "completed_at_utc": utc_now(),
        },
    )


def strict_validate_all(records: list[dict[str, Any]]) -> dict[str, int]:
    passed_available = 0
    rejected_quarantine = 0
    verified_missing = 0
    for index, record in enumerate(records, start=1):
        status = str(record["status"])
        relative = record["data_relative_path"]
        if status == "available":
            if not relative:
                raise RuntimeError(f"{record['code']} available 缺数据路径")
            manager = ParquetDataManager(
                PARTIAL / str(relative),
                expected_source_id=FREE_STOCKDB_QFQ_SOURCE_ID,
            )
            manager.load()
            if manager.data_sha256 != record["data_sha256"]:
                raise RuntimeError(f"{record['code']} 最终 strict loader 哈希不一致")
            passed_available += 1
        elif status == "quarantine":
            if not relative:
                raise RuntimeError(f"{record['code']} quarantine 缺数据路径")
            path = PARTIAL / str(relative)
            if path.with_suffix(".manifest.json").exists():
                raise RuntimeError(f"{record['code']} quarantine 带 sidecar")
            try:
                ParquetDataManager(
                    path,
                    expected_source_id=FREE_STOCKDB_QFQ_SOURCE_ID,
                ).load()
            except Exception:
                rejected_quarantine += 1
            else:
                raise RuntimeError(f"{record['code']} quarantine 被接受")
        else:
            if relative or record["data_sha256"] or record["data_rows"]:
                raise RuntimeError(f"{record['code']} source_missing 伪造数据")
            verified_missing += 1
        if index % 100 == 0 or index == len(records):
            print(
                json.dumps(
                    {
                        "phase": "strict_validation",
                        "validated": index,
                        "total": len(records),
                        "available_passed": passed_available,
                        "quarantine_rejected": rejected_quarantine,
                        "source_missing_verified": verified_missing,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {
        "available_passed": passed_available,
        "quarantine_rejected": rejected_quarantine,
        "source_missing_verified": verified_missing,
    }


def build_file_inventory() -> tuple[Path, dict[str, Any]]:
    excluded = {"manifest.json", "file_inventory.parquet"}
    rows: list[dict[str, Any]] = []
    for path in sorted(PARTIAL.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PARTIAL).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["relative_path", "bytes", "sha256"],
    )
    path = PARTIAL / "file_inventory.parquet"
    record = write_parquet_new(path, frame)
    return path, record


def main() -> int:
    if TARGET.exists() or PARTIAL.exists():
        raise RuntimeError("v3 目标或唯一 partial 已存在，拒绝覆盖")
    if sha256_file(HISTORY_MANIFEST) != EXPECTED_HISTORY_MANIFEST_SHA256:
        raise RuntimeError("可信历史 manifest 哈希不一致")
    if sha256_file(HISTORY_FILE) != EXPECTED_HISTORY_FILE_SHA256:
        raise RuntimeError("可信历史合并文件哈希不一致")
    if sha256_file(QUERY_SOURCE_SNAPSHOT) != EXPECTED_SOURCE_SNAPSHOT_SHA256:
        raise RuntimeError("冻结 source snapshot 哈希不一致")

    baseline = read_json(BASELINE)
    v2_files = verify_v2_baseline(baseline)
    v2_manifest = read_json(V2_ROOT / "manifest.json")
    if v2_manifest.get("status") != "completed":
        raise RuntimeError("v2 根 manifest 未完成")
    history_manifest = read_json(HISTORY_MANIFEST)
    if history_manifest.get("status") != "completed":
        raise RuntimeError("可信历史 manifest 未完成")
    history = pd.read_parquet(HISTORY_FILE)
    if (
        len(history) != EXPECTED_HISTORY_ROWS
        or history["date"].nunique() != EXPECTED_HISTORY_DATES
        or history["code"].astype(str).str[:6].nunique() != EXPECTED_HISTORY_CODES
    ):
        raise RuntimeError("可信历史 76498/255/949 门未通过")
    membership = build_membership_evidence(history)
    historical_codes = set(membership)

    client = StockDBClient(host=QUERY_HOST, port=QUERY_PORT)
    if not client._fq_dates:
        raise RuntimeError("query 服务复权因子为空")
    sort_factor_arrays(client)
    health_check(client)

    PARTIAL.mkdir(parents=False)
    for folder in (
        "D1",
        "D1_quarantine",
        "manifests",
        "batches",
        "source",
    ):
        (PARTIAL / folder).mkdir()
    inherit_v2_files(v2_files)
    copy_evidence()
    if sha256_file(PARTIAL / "source" / ".sync_manifest.json") != (
        EXPECTED_SOURCE_SNAPSHOT_SHA256
    ):
        raise RuntimeError("partial 中冻结 source snapshot 复制不一致")

    inherited, v2_codes = inherited_coverage_records(
        v2_manifest=v2_manifest,
        membership=membership,
    )
    if len(v2_codes) != EXPECTED_V2_CODES:
        raise RuntimeError("v2 代码集合不是 300")
    new_codes = sorted(historical_codes - v2_codes)
    if len(new_codes) != EXPECTED_NEW_CODES:
        raise RuntimeError("历史减 v2 不是精确 649")

    new_records: list[dict[str, Any]] = []
    cumulative: Counter[str] = Counter()
    batch_records: list[dict[str, Any]] = []
    batch_number = 0
    for index, code in enumerate(new_codes, start=1):
        record = process_new_code(
            code=code,
            client=client,
            membership=membership[code],
        )
        new_records.append(record)
        batch_records.append(record)
        cumulative[record["status"]] += 1
        if index % BATCH_SIZE == 0 or index == len(new_codes):
            health_check(client)
            batch_number += 1
            write_batch(
                batch_number=batch_number,
                batch_records=batch_records,
                cumulative=cumulative,
                processed=index,
                total=len(new_codes),
            )
            print(
                json.dumps(
                    {
                        "phase": "new_code_export",
                        "processed": index,
                        "total": len(new_codes),
                        "available": cumulative["available"],
                        "quarantine": cumulative["quarantine"],
                        "source_missing": cumulative["source_missing"],
                        "errors": 0,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            batch_records = []

    health_check(client)
    all_records = sorted(
        [*inherited, *new_records],
        key=lambda record: record["code"],
    )
    status_counts = validate_coverage_records(
        all_records,
        historical_codes=historical_codes,
        v2_codes=v2_codes,
    )
    strict_counts = strict_validate_all(all_records)

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
        "retrospective_only",
        "point_in_time_eligible",
        "sealed_evaluation_eligible",
    ]
    coverage_frame = pd.DataFrame(all_records, columns=coverage_columns)
    coverage_path = PARTIAL / "coverage_matrix.parquet"
    coverage_record = write_parquet_new(coverage_path, coverage_frame)
    reloaded_coverage = pd.read_parquet(coverage_path)
    if (
        len(reloaded_coverage) != EXPECTED_HISTORY_CODES
        or set(reloaded_coverage["code"]) != historical_codes
    ):
        raise RuntimeError("coverage_matrix 发布后不是 949 精确全集")

    _inventory_path, inventory_record = build_file_inventory()
    root_manifest = {
        "format": V3_FORMAT,
        "status": "completed",
        "created_at_utc": utc_now(),
        "source_as_of": SOURCE_AS_OF,
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
            "historical_code_count": EXPECTED_HISTORY_CODES,
            "inherited_v2_code_count": EXPECTED_V2_CODES,
            "newly_queried_code_count": EXPECTED_NEW_CODES,
            "statuses": ["available", "quarantine", "source_missing"],
            "mutually_exclusive": True,
            "exhaustive": True,
            "not_exported_is_not_source_missing": True,
        },
        "status_counts": status_counts,
        "new_code_status_counts": dict(sorted(cumulative.items())),
        "strict_loader_validation": strict_counts,
        "data_contract": {
            "source_id": FREE_STOCKDB_QFQ_SOURCE_ID,
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
            "minimum_bars": MINIMUM_D1_BARS,
            "quarantine_has_no_source_sidecar": True,
            "source_missing_has_no_data_file": True,
        },
        "trusted_history": {
            "manifest": "source/evidence/trusted_history_manifest.json",
            "manifest_sha256": EXPECTED_HISTORY_MANIFEST_SHA256,
            "combined_file": "source/evidence/trusted_history.parquet",
            "combined_sha256": EXPECTED_HISTORY_FILE_SHA256,
            "rows": EXPECTED_HISTORY_ROWS,
            "dates": EXPECTED_HISTORY_DATES,
            "codes": EXPECTED_HISTORY_CODES,
        },
        "parent_v2": {
            "manifest": "source/parent_v2_manifest.json",
            "manifest_sha256": EXPECTED_V2_MANIFEST_SHA256,
            "frozen_file_count": len(v2_files),
            "all_frozen_files_verified_before_copy": True,
            "artifacts_inherited_without_reencoding": True,
        },
        "source_snapshot": {
            "manifest": "source/.sync_manifest.json",
            "raw_manifest_sha256": EXPECTED_SOURCE_SNAPSHOT_SHA256,
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
            "query_service": f"{QUERY_HOST}:{QUERY_PORT}",
            "query_service_opened_physical_copy_only": True,
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
            "incident_ledger": "source/evidence/incident_ledger.json",
        },
        "extraction_script": (
            "source/build_csi300_historical_am_inputs_v3.py"
        ),
        "extraction_script_sha256": sha256_file(Path(__file__).resolve()),
        "coverage_matrix": {
            **coverage_record,
            "columns": coverage_columns,
        },
        "file_inventory": {
            **inventory_record,
            "inventory_excludes_itself_and_root_manifest": True,
        },
        "batch_manifests": batch_number,
        "item_receipts": EXPECTED_HISTORY_CODES,
    }
    write_json_new(PARTIAL / "manifest.json", root_manifest)
    manifest_sha256 = sha256_file(PARTIAL / "manifest.json")

    if TARGET.exists():
        raise RuntimeError("原子发布前 v3 目标突然出现")
    os.replace(PARTIAL, TARGET)
    if not TARGET.is_dir() or PARTIAL.exists():
        raise RuntimeError("v3 原子发布后目录状态异常")
    final_manifest = read_json(TARGET / "manifest.json")
    final_coverage = pd.read_parquet(TARGET / "coverage_matrix.parquet")
    if (
        final_manifest.get("status") != "completed"
        or sha256_file(TARGET / "manifest.json") != manifest_sha256
        or len(final_coverage) != EXPECTED_HISTORY_CODES
        or set(final_coverage["code"]) != historical_codes
    ):
        raise RuntimeError("v3 原子发布后复核失败")
    print(
        json.dumps(
            {
                "phase": "completed",
                "target": str(TARGET),
                "manifest_sha256": manifest_sha256,
                "coverage_rows": len(final_coverage),
                "status_counts": status_counts,
                "new_code_status_counts": dict(sorted(cumulative.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
