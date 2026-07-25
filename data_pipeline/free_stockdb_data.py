"""free-stockdb 前复权 A 股日线的来源合同。"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from data_pipeline.a_share_data import (
    ASHARE_SPECS_BY_TIMEFRAME,
    CANONICAL_COLUMNS,
    sha256_file,
    validate_canonical_training_frame,
)
from data_pipeline.dataset_contracts import (
    FREE_STOCKDB_QFQ_FORMAT,
    FREE_STOCKDB_QFQ_SOURCE_ID,
    FREE_STOCKDB_SOURCE,
)


FREE_STOCKDB_PROVIDER = "hello245m/free-stockdb"
FREE_STOCKDB_PROVIDER_INTERFACE = "local_leveldb_get_data"
FREE_STOCKDB_ADJUSTMENT = "qfq"
FREE_STOCKDB_ADJUSTMENT_ENGINE = "release_sdk_in_memory_factor_adjustment"
FREE_STOCKDB_ORDER_CORRECTION = "explicit_date_ascending_before_adjustment"
FREE_STOCKDB_BAR_COMPLETION = "completed_trading_days_only"
FREE_STOCKDB_SNAPSHOT_MANIFEST = ".sync_manifest.json"
FREE_STOCKDB_VOLUME_SEMANTICS = "source_daily_share_volume"
FREE_STOCKDB_VOLUME_UNIT = "shares"
FREE_STOCKDB_VOLUME_ADJUSTMENT = "unadjusted_source_volume"
_CANONICAL_D1_RE = re.compile(r"^(?P<symbol>[0-9]{6})_D1\.parquet$", re.IGNORECASE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class FreeStockDBDataError(ValueError):
    """free-stockdb 数据或来源合同不一致。"""


def _utc_iso(unix_seconds: int) -> str:
    return (
        datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_daily_close_times(frame: pd.DataFrame) -> None:
    local = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_convert(_SHANGHAI)
    if not ((local.dt.hour == 15) & (local.dt.minute == 0) & (local.dt.second == 0)).all():
        raise FreeStockDBDataError("free-stockdb D1 time 必须是上海交易日 15:00 收盘时刻")


def _parse_source_as_of(value: Any) -> date:
    if not isinstance(value, str):
        raise FreeStockDBDataError("source_as_of 必须是 YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise FreeStockDBDataError("source_as_of 必须是真实的 YYYY-MM-DD 日期") from exc
    if parsed.isoformat() != value:
        raise FreeStockDBDataError("source_as_of 必须是规范 YYYY-MM-DD")
    return parsed


def _validate_temporal_provenance(
    frame: pd.DataFrame,
    *,
    source_as_of: Any,
    source_snapshot_generated_at: Any,
) -> None:
    as_of = _parse_source_as_of(source_as_of)
    if (
        isinstance(source_snapshot_generated_at, bool)
        or not isinstance(source_snapshot_generated_at, int)
        or source_snapshot_generated_at <= 0
    ):
        raise FreeStockDBDataError("source_snapshot_generated_at 必须是正整数")
    local = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_convert(_SHANGHAI)
    if local.dt.date.max() > as_of:
        raise FreeStockDBDataError("free-stockdb 数据晚于 source_as_of")
    snapshot_utc = datetime.fromtimestamp(
        source_snapshot_generated_at,
        tz=timezone.utc,
    )
    if int(frame["time"].max()) > int(snapshot_utc.timestamp()):
        raise FreeStockDBDataError("free-stockdb 数据晚于源快照生成时刻")
    if as_of > snapshot_utc.astimezone(_SHANGHAI).date():
        raise FreeStockDBDataError("source_as_of 晚于源快照生成日期")


def _read_source_snapshot(path: str | Path) -> tuple[str, int]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FreeStockDBDataError("源 .sync_manifest.json 不存在")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreeStockDBDataError("源 .sync_manifest.json 不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise FreeStockDBDataError("源 .sync_manifest.json 顶层必须是对象")
    generated_at = payload.get("generated_at")
    if (
        isinstance(generated_at, bool)
        or not isinstance(generated_at, int)
        or generated_at <= 0
    ):
        raise FreeStockDBDataError("源 .sync_manifest.json 的 generated_at 非法")
    return sha256_file(manifest_path), generated_at


def build_free_stockdb_qfq_manifest(
    data_file: str | Path,
    *,
    source_as_of: str,
    provider_release: str,
    source_snapshot_manifest: str | Path,
    extraction_script: str | Path,
    qfq_factor_points: int,
) -> dict[str, Any]:
    """为已冻结的规范 D1 Parquet 构建可复核来源合同。"""
    path = Path(data_file)
    match = _CANONICAL_D1_RE.fullmatch(path.name)
    if match is None:
        raise FreeStockDBDataError("free-stockdb 训练文件必须是 6位代码_D1.parquet")
    frame = pd.read_parquet(path)
    validate_canonical_training_frame(frame)
    _validate_daily_close_times(frame)
    spec = ASHARE_SPECS_BY_TIMEFRAME["D1"]
    if len(frame) < spec.minimum_bars:
        raise FreeStockDBDataError(
            f"D1 数据不足: {len(frame)} bars（A股至少需要 {spec.minimum_bars}）"
        )
    if not isinstance(provider_release, str) or not provider_release.strip():
        raise FreeStockDBDataError("provider_release 不能为空")
    source_snapshot_sha256, source_snapshot_generated_at = _read_source_snapshot(
        source_snapshot_manifest
    )
    extraction_script_path = Path(extraction_script)
    if not extraction_script_path.is_file():
        raise FreeStockDBDataError("提取脚本不存在")
    extraction_script_sha256 = sha256_file(extraction_script_path)
    _validate_temporal_provenance(
        frame,
        source_as_of=source_as_of,
        source_snapshot_generated_at=source_snapshot_generated_at,
    )
    if (
        isinstance(qfq_factor_points, bool)
        or not isinstance(qfq_factor_points, int)
        or qfq_factor_points < 0
    ):
        raise FreeStockDBDataError("qfq_factor_points 必须是非负整数")

    digest = sha256_file(path)
    rows = len(frame)
    return {
        "format": FREE_STOCKDB_QFQ_FORMAT,
        "source": FREE_STOCKDB_SOURCE,
        "source_id": FREE_STOCKDB_QFQ_SOURCE_ID,
        "market": "CN_A_SHARE",
        "symbol": match.group("symbol"),
        "timeframe": "D1",
        "periods_per_year": spec.periods_per_year,
        "minimum_bars": spec.minimum_bars,
        "data_filename": path.name,
        "data_sha256": digest,
        "dataset_id": f"sha256:{digest}",
        "data_rows": rows,
        "data_start": _utc_iso(int(frame["time"].iloc[0])),
        "data_end": _utc_iso(int(frame["time"].iloc[-1])),
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "bar_timestamp_semantics": "bar_close",
        "columns": list(CANONICAL_COLUMNS),
        "source_timezone": "Asia/Shanghai",
        "session_close_times": ["15:00"],
        "bar_completion": FREE_STOCKDB_BAR_COMPLETION,
        "provider": FREE_STOCKDB_PROVIDER,
        "provider_release": provider_release,
        "provider_interface": FREE_STOCKDB_PROVIDER_INTERFACE,
        "adjustment": FREE_STOCKDB_ADJUSTMENT,
        "adjustment_engine": FREE_STOCKDB_ADJUSTMENT_ENGINE,
        "source_order_correction": FREE_STOCKDB_ORDER_CORRECTION,
        "tick_volume_semantics": FREE_STOCKDB_VOLUME_SEMANTICS,
        "tick_volume_unit": FREE_STOCKDB_VOLUME_UNIT,
        "tick_volume_adjustment": FREE_STOCKDB_VOLUME_ADJUSTMENT,
        "source_as_of": source_as_of,
        "source_snapshot_manifest": FREE_STOCKDB_SNAPSHOT_MANIFEST,
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_snapshot_generated_at": source_snapshot_generated_at,
        "extraction_script_sha256": extraction_script_sha256,
        "qfq_factor_points": qfq_factor_points,
        "coverage_in_trading_years": round(rows / spec.periods_per_year, 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_free_stockdb_qfq_manifest(
    data_file: str | Path,
    frame: pd.DataFrame,
) -> dict[str, Any] | None:
    """若 sidecar 声明 FreeStockDB，则完整复核并返回。"""
    path = Path(data_file)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreeStockDBDataError("free-stockdb manifest 不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise FreeStockDBDataError("free-stockdb manifest 顶层必须是对象")
    if payload.get("source") != FREE_STOCKDB_SOURCE:
        return None

    match = _CANONICAL_D1_RE.fullmatch(path.name)
    if match is None:
        raise FreeStockDBDataError("free-stockdb 训练文件必须是 6位代码_D1.parquet")
    validate_canonical_training_frame(frame)
    _validate_daily_close_times(frame)
    spec = ASHARE_SPECS_BY_TIMEFRAME["D1"]
    digest = sha256_file(path)
    expected = {
        "format": FREE_STOCKDB_QFQ_FORMAT,
        "source": FREE_STOCKDB_SOURCE,
        "source_id": FREE_STOCKDB_QFQ_SOURCE_ID,
        "market": "CN_A_SHARE",
        "symbol": match.group("symbol"),
        "timeframe": "D1",
        "periods_per_year": spec.periods_per_year,
        "minimum_bars": spec.minimum_bars,
        "data_filename": path.name,
        "data_sha256": digest,
        "dataset_id": f"sha256:{digest}",
        "data_rows": len(frame),
        "data_start": _utc_iso(int(frame["time"].iloc[0])),
        "data_end": _utc_iso(int(frame["time"].iloc[-1])),
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "bar_timestamp_semantics": "bar_close",
        "columns": list(CANONICAL_COLUMNS),
        "source_timezone": "Asia/Shanghai",
        "session_close_times": ["15:00"],
        "bar_completion": FREE_STOCKDB_BAR_COMPLETION,
        "provider": FREE_STOCKDB_PROVIDER,
        "provider_interface": FREE_STOCKDB_PROVIDER_INTERFACE,
        "adjustment": FREE_STOCKDB_ADJUSTMENT,
        "adjustment_engine": FREE_STOCKDB_ADJUSTMENT_ENGINE,
        "source_order_correction": FREE_STOCKDB_ORDER_CORRECTION,
        "tick_volume_semantics": FREE_STOCKDB_VOLUME_SEMANTICS,
        "tick_volume_unit": FREE_STOCKDB_VOLUME_UNIT,
        "tick_volume_adjustment": FREE_STOCKDB_VOLUME_ADJUSTMENT,
        "source_snapshot_manifest": FREE_STOCKDB_SNAPSHOT_MANIFEST,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise FreeStockDBDataError(
                f"free-stockdb manifest 的 {field} 与数据合同不匹配"
            )
    if len(frame) < spec.minimum_bars:
        raise FreeStockDBDataError(
            f"D1 数据不足: {len(frame)} bars（A股至少需要 {spec.minimum_bars}）"
        )
    if not isinstance(payload.get("provider_release"), str) or not payload["provider_release"]:
        raise FreeStockDBDataError("free-stockdb manifest 缺少 provider_release")
    for field in ("source_snapshot_sha256", "extraction_script_sha256"):
        if _SHA256_RE.fullmatch(str(payload.get(field) or "")) is None:
            raise FreeStockDBDataError(f"free-stockdb manifest 的 {field} 非法")
    generated_at = payload.get("source_snapshot_generated_at")
    _validate_temporal_provenance(
        frame,
        source_as_of=payload.get("source_as_of"),
        source_snapshot_generated_at=generated_at,
    )
    factor_points = payload.get("qfq_factor_points")
    if (
        isinstance(factor_points, bool)
        or not isinstance(factor_points, int)
        or factor_points < 0
    ):
        raise FreeStockDBDataError("free-stockdb manifest 的 qfq_factor_points 非法")
    return payload
