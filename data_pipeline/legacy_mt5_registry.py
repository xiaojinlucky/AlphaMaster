"""为用户明确确认来源的旧 MT5 Parquet 建立可审计 sidecar。

历史注册只能证明“用户确认这组精确文件字节来自旧 MT5 数据”，不能把它
升级成由 AlphaMaster 官方导出器同步生成的 verified 数据。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.dataset_contracts import (
    MT5_LEGACY_FORMAT,
    MT5_LEGACY_SOURCE,
    MT5_LEGACY_SOURCE_ID,
)
from data_pipeline.parquet_manager import inspect_parquet_file

PLAN_FORMAT = "alphamaster_mt5_legacy_registration_plan_v1"
REPORT_FORMAT = "alphamaster_mt5_legacy_registration_report_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MT5_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


class LegacyRegistrationError(RuntimeError):
    """旧 MT5 数据计划或发布不满足安全合同。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: dict[str, Any], *, excluded: str) -> str:
    body = {key: value for key, value in payload.items() if key != excluded}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        with staging.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _relative_file(root: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise LegacyRegistrationError(f"文件逃逸批量根目录: {path}") from exc
    if path.is_symlink():
        raise LegacyRegistrationError(f"拒绝符号链接文件: {relative}")
    return relative.as_posix()


def _file_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "data_sha256": _sha256_file(path),
    }


def _source_report_evidence(
    source_report: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[dict[str, Any]]:
    if source_report is None:
        return []
    candidates = (
        [source_report]
        if isinstance(source_report, (str, Path))
        else list(source_report)
    )
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        report = Path(candidate).resolve(strict=True)
        if not report.is_file():
            raise LegacyRegistrationError(f"来源报告不是普通文件: {report}")
        key = str(report).casefold()
        if key in seen:
            raise LegacyRegistrationError(f"来源报告重复: {report}")
        seen.add(key)
        evidence.append({
            "filename": report.name,
            "sha256": _sha256_file(report),
        })
    return evidence


def _plan_entry(
    root: Path,
    path: Path,
    *,
    casefold_seen: dict[str, str],
) -> dict[str, Any]:
    try:
        relative = _relative_file(root, path)
        folded = relative.casefold()
        if folded in casefold_seen:
            raise LegacyRegistrationError(
                f"Windows 大小写路径冲突: {casefold_seen[folded]} / {relative}"
            )
        casefold_seen[folded] = relative
        before = _file_snapshot(path)
        info = inspect_parquet_file(path)
        after = _file_snapshot(path)
        if before != after:
            raise LegacyRegistrationError("文件在检查期间发生变化")
        if info.get("columns") != _MT5_COLUMNS:
            raise LegacyRegistrationError(
                "旧 MT5 注册只接受精确列顺序 "
                "time/open/high/low/close/tick_volume"
            )

        registration = str(info.get("registration") or "")
        if registration == "registered":
            status = "already_registered"
            eligible = False
            reason = f"已有有效 manifest，来源={info.get('source')}"
        elif info.get("source") == "local_file":
            status = "eligible"
            eligible = True
            reason = ""
        else:
            status = "rejected"
            eligible = False
            reason = f"不支持的未注册来源状态: {info.get('source')}"

        return {
            "relative_path": relative,
            "status": status,
            "eligible": eligible,
            "reason": reason,
            "size": before["size"],
            "mtime_ns": before["mtime_ns"],
            "data_sha256": before["data_sha256"],
            "symbol": info["symbol"],
            "timeframe": info["timeframe"],
            "data_rows": info["bars"],
            "data_start": info["data_start"],
            "data_end": info["data_end"],
            "columns": info["columns"],
            "periods_per_year": info["periods_per_year"],
            "minimum_bars": info["minimum_bars"],
        }
    except Exception as exc:
        try:
            relative = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            relative = str(path).replace("\\", "/")
        return {
            "relative_path": relative,
            "status": "rejected",
            "eligible": False,
            "reason": str(exc),
        }


def _finalize_plan(
    *,
    root: Path,
    recursive: bool,
    feed_id: str,
    source_reports: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "created_at": _utc_now(),
        "input_root": str(root),
        "recursive": bool(recursive),
        "source": MT5_LEGACY_SOURCE,
        "source_format": MT5_LEGACY_FORMAT,
        "source_id": MT5_LEGACY_SOURCE_ID,
        "provenance_level": "legacy_user_attested",
        "attestation_scope": "exact_file_bytes",
        "feed_id": feed_id,
        "source_reports": source_reports,
        "files": entries,
        "summary": {
            "total": len(entries),
            "eligible": sum(row.get("status") == "eligible" for row in entries),
            "already_registered": sum(
                row.get("status") == "already_registered" for row in entries
            ),
            "rejected": sum(row.get("status") == "rejected" for row in entries),
        },
    }
    plan["plan_sha256"] = _payload_sha256(plan, excluded="plan_sha256")
    return plan


def build_registration_plan(
    input_dir: str | Path,
    *,
    recursive: bool = False,
    source_report: (
        str | Path | list[str | Path] | tuple[str | Path, ...] | None
    ) = None,
    feed_id: str = "",
) -> dict[str, Any]:
    """只读扫描旧数据目录，返回含 plan SHA 的注册计划。"""
    root = Path(input_dir).resolve(strict=True)
    if not root.is_dir():
        raise LegacyRegistrationError(f"输入路径不是目录: {root}")
    clean_feed = str(feed_id or "").strip()
    if len(clean_feed) > 80 or any(ord(char) < 32 for char in clean_feed):
        raise LegacyRegistrationError("feed_id 必须是不超过 80 字符的非敏感单行标签")

    evidence = _source_report_evidence(source_report)
    pattern = "**/*.parquet" if recursive else "*.parquet"
    files = sorted(root.glob(pattern), key=lambda path: str(path).casefold())
    casefold_seen: dict[str, str] = {}
    entries: list[dict[str, Any]] = []

    for path in files:
        entries.append(
            _plan_entry(root, path, casefold_seen=casefold_seen)
        )
    return _finalize_plan(
        root=root,
        recursive=recursive,
        feed_id=clean_feed,
        source_reports=evidence,
        entries=entries,
    )


def build_single_file_registration_plan(
    data_file: str | Path,
    *,
    feed_id: str = "",
) -> dict[str, Any]:
    """为前端当前选择的单个文件生成轻量、只读计划。"""
    path = Path(data_file)
    root = path.parent.resolve(strict=True)
    clean_feed = str(feed_id or "").strip()
    if len(clean_feed) > 80 or any(ord(char) < 32 for char in clean_feed):
        raise LegacyRegistrationError("feed_id 必须是不超过 80 字符的非敏感单行标签")
    entry = _plan_entry(root, path, casefold_seen={})
    return _finalize_plan(
        root=root,
        recursive=False,
        feed_id=clean_feed,
        source_reports=[],
        entries=[entry],
    )


def write_registration_plan(path: str | Path, plan: dict[str, Any]) -> Path:
    target = Path(path)
    expected = _payload_sha256(plan, excluded="plan_sha256")
    if plan.get("plan_sha256") != expected:
        raise LegacyRegistrationError("注册计划 SHA-256 不匹配")
    _atomic_write_json(target, plan)
    return target.resolve()


def _manifest_from_plan(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    digest = row["data_sha256"]
    return {
        "format": MT5_LEGACY_FORMAT,
        "source": MT5_LEGACY_SOURCE,
        "source_family": "MetaTrader5",
        "provenance_level": "legacy_user_attested",
        "attestation_scope": "exact_file_bytes",
        "registration_method": "legacy_sidecar_registration_v1",
        "registered_at": _utc_now(),
        "feed_id": plan.get("feed_id") or "",
        "source_reports": plan.get("source_reports") or [],
        "registration_plan_sha256": plan["plan_sha256"],
        "symbol": row["symbol"],
        "timeframe": row["timeframe"],
        "data_filename": Path(row["relative_path"]).name,
        "data_sha256": digest,
        "dataset_id": f"sha256:{digest}",
        "data_rows": row["data_rows"],
        "data_start": row["data_start"],
        "data_end": row["data_end"],
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "bar_timestamp_semantics": "source_bar_open",
        "columns": row["columns"],
        "periods_per_year": row["periods_per_year"],
        "minimum_bars": row["minimum_bars"],
    }


def _publish_manifest_without_overwrite(
    manifest_path: Path,
    payload: dict[str, Any],
) -> str:
    if manifest_path.exists():
        raise LegacyRegistrationError(f"manifest 已存在，拒绝覆盖: {manifest_path.name}")
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    lock_token = uuid.uuid4().hex
    staging = manifest_path.with_name(f".{manifest_path.name}.{lock_token}.partial")
    published_sha = ""
    try:
        with lock_path.open("x", encoding="ascii") as lock:
            lock.write(lock_token)
            lock.flush()
            os.fsync(lock.fileno())
        body = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        with staging.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(staging.read_text(encoding="utf-8"))
        published_sha = _sha256_file(staging)
        os.link(staging, manifest_path)
        return published_sha
    finally:
        staging.unlink(missing_ok=True)
        try:
            if lock_path.read_text(encoding="ascii") == lock_token:
                lock_path.unlink()
        except OSError:
            pass


def apply_registration_plan(
    plan: dict[str, Any],
    *,
    expected_plan_sha256: str,
    source_acknowledgement: str,
) -> dict[str, Any]:
    """重新验证计划中每个文件，并为未漂移的 eligible 文件发布 sidecar。"""
    if plan.get("format") != PLAN_FORMAT:
        raise LegacyRegistrationError("注册计划格式不受支持")
    actual_plan_sha = _payload_sha256(plan, excluded="plan_sha256")
    if (
        plan.get("plan_sha256") != actual_plan_sha
        or expected_plan_sha256 != actual_plan_sha
        or _SHA256_RE.fullmatch(expected_plan_sha256) is None
    ):
        raise LegacyRegistrationError("注册计划 SHA-256 不匹配")
    if source_acknowledgement != "MetaTrader5":
        raise LegacyRegistrationError(
            "必须明确确认这些精确文件字节来自旧 MetaTrader5 数据"
        )

    root = Path(str(plan.get("input_root") or "")).resolve(strict=True)
    started_at = _utc_now()
    results: list[dict[str, Any]] = []
    for row in plan.get("files") or []:
        relative = str(row.get("relative_path") or "")
        result = {"relative_path": relative, "result": "skipped", "message": ""}
        if row.get("status") != "eligible":
            result["result"] = str(row.get("status") or "skipped")
            result["message"] = str(row.get("reason") or "")
            results.append(result)
            continue
        try:
            unresolved = root / Path(relative)
            if unresolved.is_symlink():
                raise LegacyRegistrationError("拒绝符号链接文件")
            candidate = unresolved.resolve(strict=True)
            _relative_file(root, candidate)
            before = _file_snapshot(candidate)
            expected_snapshot = {
                "size": row.get("size"),
                "mtime_ns": row.get("mtime_ns"),
                "data_sha256": row.get("data_sha256"),
            }
            if before != expected_snapshot:
                raise LegacyRegistrationError("文件身份相对计划发生漂移")
            info = inspect_parquet_file(candidate)
            after = _file_snapshot(candidate)
            if before != after or info["data_sha256"] != before["data_sha256"]:
                raise LegacyRegistrationError("文件在 apply 复验期间发生变化")
            for field in (
                "symbol",
                "timeframe",
                "bars",
                "data_start",
                "data_end",
                "columns",
                "periods_per_year",
                "minimum_bars",
            ):
                planned_field = "data_rows" if field == "bars" else field
                if info[field] != row[planned_field]:
                    raise LegacyRegistrationError(f"文件字段相对计划发生漂移: {field}")

            manifest_path = candidate.with_suffix(".manifest.json")
            payload = _manifest_from_plan(plan, row)
            manifest_sha = _publish_manifest_without_overwrite(manifest_path, payload)
            try:
                verified = inspect_parquet_file(candidate)
                if (
                    verified.get("source") != MT5_LEGACY_SOURCE_ID
                    or verified.get("dataset_id") != f"sha256:{before['data_sha256']}"
                ):
                    raise LegacyRegistrationError("发布后的正式 loader 复验失败")
            except Exception:
                if (
                    manifest_path.is_file()
                    and _sha256_file(manifest_path) == manifest_sha
                ):
                    manifest_path.unlink()
                raise
            result.update(
                {
                    "result": "registered",
                    "manifest": manifest_path.name,
                    "manifest_sha256": manifest_sha,
                    "data_sha256": before["data_sha256"],
                }
            )
        except Exception as exc:
            result["result"] = "failed"
            result["message"] = str(exc)
        results.append(result)

    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "plan_sha256": actual_plan_sha,
        "input_root": str(root),
        "source": MT5_LEGACY_SOURCE,
        "source_id": MT5_LEGACY_SOURCE_ID,
        "results": results,
        "summary": {
            "total": len(results),
            "registered": sum(row["result"] == "registered" for row in results),
            "already_registered": sum(
                row["result"] == "already_registered" for row in results
            ),
            "rejected": sum(row["result"] == "rejected" for row in results),
            "failed": sum(row["result"] == "failed" for row in results),
        },
    }
    report["report_sha256"] = _payload_sha256(report, excluded="report_sha256")
    return report


def load_registration_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyRegistrationError("注册计划无法读取") from exc
    if not isinstance(payload, dict):
        raise LegacyRegistrationError("注册计划必须是 JSON 对象")
    return payload


def write_registration_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    expected = _payload_sha256(report, excluded="report_sha256")
    if report.get("report_sha256") != expected:
        raise LegacyRegistrationError("注册报告 SHA-256 不匹配")
    _atomic_write_json(target, report)
    return target.resolve()
