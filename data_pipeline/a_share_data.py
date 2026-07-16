"""A 股旧 Parquet 到 AlphaMaster 严格训练数据的转换与合同校验。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype


ASHARE_SOURCE = "AShareLocal"
ASHARE_SOURCE_ID = "ashare_local"
ASHARE_DATASET_FORMAT = "alphamaster_ashare_local_dataset_v1"
CANONICAL_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class ASharePeriodSpec:
    legacy_period: str
    timeframe: str
    periods_per_year: int
    minimum_bars: int
    close_times: tuple[time, ...]


def _clock_range(start_hour: int, start_minute: int, count: int, step: int) -> tuple[time, ...]:
    start = datetime(2000, 1, 1, start_hour, start_minute)
    return tuple((start + timedelta(minutes=index * step)).time() for index in range(count))


ASHARE_PERIOD_SPECS: dict[str, ASharePeriodSpec] = {
    "5min": ASharePeriodSpec(
        legacy_period="5min",
        timeframe="M5",
        periods_per_year=11_616,
        minimum_bars=23_232,
        close_times=(*_clock_range(9, 35, 24, 5), *_clock_range(13, 5, 24, 5)),
    ),
    "15min": ASharePeriodSpec(
        legacy_period="15min",
        timeframe="M15",
        periods_per_year=3_872,
        minimum_bars=7_744,
        close_times=(*_clock_range(9, 45, 8, 15), *_clock_range(13, 15, 8, 15)),
    ),
    "60min": ASharePeriodSpec(
        legacy_period="60min",
        timeframe="H1",
        periods_per_year=968,
        minimum_bars=1_936,
        close_times=(time(10, 30), time(11, 30), time(14, 0), time(15, 0)),
    ),
    "daily": ASharePeriodSpec(
        legacy_period="daily",
        timeframe="D1",
        periods_per_year=242,
        minimum_bars=484,
        close_times=(time(15, 0),),
    ),
}
ASHARE_SPECS_BY_TIMEFRAME = {spec.timeframe: spec for spec in ASHARE_PERIOD_SPECS.values()}

_LEGACY_FILENAME_RE = re.compile(
    r"^(?P<symbol>[0-9]{6})_(?P<period>5min|15min|60min|daily)\.parquet$",
    re.IGNORECASE,
)
_CANONICAL_FILENAME_RE = re.compile(
    r"^(?P<symbol>[0-9]{6})_(?P<timeframe>M5|M15|H1|D1)\.parquet$",
    re.IGNORECASE,
)


class AShareDataError(ValueError):
    """A 股数据违反可逆转换或训练合同。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_iso(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def validate_canonical_training_frame(frame: pd.DataFrame) -> None:
    """严格验证规范 OHLCV；不排序、不去重、不补值。"""
    columns = tuple(str(column) for column in frame.columns)
    if columns != CANONICAL_COLUMNS:
        raise AShareDataError(
            f"训练 Parquet 列必须严格为 {list(CANONICAL_COLUMNS)}，当前为 {list(columns)}"
        )
    if frame.empty:
        raise AShareDataError("训练 Parquet 不能为空")
    if is_bool_dtype(frame["time"].dtype) or frame["time"].dtype != np.dtype("int64"):
        raise AShareDataError("time 必须是 unix_seconds int64 整数列")
    if (
        is_bool_dtype(frame["tick_volume"].dtype)
        or frame["tick_volume"].dtype != np.dtype("int64")
    ):
        raise AShareDataError("tick_volume 必须是非负整数列")
    for column in _PRICE_COLUMNS:
        if is_bool_dtype(frame[column].dtype) or frame[column].dtype != np.dtype("float32"):
            raise AShareDataError(f"{column} 必须是 float32 数值列")

    timestamps = frame["time"].to_numpy(dtype=np.int64, copy=False)
    if np.any(timestamps <= 0):
        raise AShareDataError("time 必须是正的 unix_seconds")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise AShareDataError("time 必须严格递增且不得重复；拒绝静默排序或去重")

    prices = frame[list(_PRICE_COLUMNS)].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(prices).all():
        raise AShareDataError("OHLC 含 NaN 或无穷值")
    if np.any(prices <= 0):
        raise AShareDataError("OHLC 必须全部大于 0")
    open_values, high_values, low_values, close_values = (prices[:, index] for index in range(4))
    if np.any(high_values < np.maximum(open_values, close_values)) or np.any(
        low_values > np.minimum(open_values, close_values)
    ):
        raise AShareDataError("OHLC 价格关系非法")

    volumes = frame["tick_volume"].to_numpy(dtype=np.int64, copy=False)
    if np.any(volumes < 0):
        raise AShareDataError("tick_volume 不得为负")


def _legacy_identity(path: Path) -> tuple[str, ASharePeriodSpec]:
    if any(part.lower() == "indices" for part in path.parts):
        raise AShareDataError("indices 数据整类拒绝：旧指数文件不满足可信 OHLCV 合同")
    if path.parent.name.lower() != "stocks":
        raise AShareDataError("只接受 stocks 目录下的 A 股个股文件")
    match = _LEGACY_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise AShareDataError("旧文件名必须为 6位代码_(5min|15min|60min|daily).parquet")
    return match.group("symbol"), ASHARE_PERIOD_SPECS[match.group("period").lower()]


def _wall_clock_date(raw_bucket: int, row_number: int) -> date:
    try:
        return datetime.fromtimestamp(raw_bucket * 1000, tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError) as exc:
        raise AShareDataError(f"第 {row_number} 行旧 time 超出可解析范围") from exc


def _reconstruct_utc_timestamps(raw_times: np.ndarray, spec: ASharePeriodSpec) -> np.ndarray:
    """按原始行序和固定收盘时刻表唯一重建 UTC 秒。"""
    dates = [_wall_clock_date(int(value), index + 1) for index, value in enumerate(raw_times)]
    if any(current < previous for previous, current in zip(dates, dates[1:])):
        raise AShareDataError("旧 time 对应交易日倒序；拒绝自动排序")

    reconstructed: list[int] = []
    cursor = 0
    expected_per_day = len(spec.close_times)
    while cursor < len(raw_times):
        trading_day = dates[cursor]
        if trading_day.weekday() >= 5:
            raise AShareDataError(f"{trading_day.isoformat()} 是周末，不能作为 A 股交易日")
        end = cursor + 1
        while end < len(raw_times) and dates[end] == trading_day:
            end += 1
        actual_per_day = end - cursor
        if actual_per_day != expected_per_day:
            raise AShareDataError(
                f"交易日 {trading_day.isoformat()} 的 {spec.timeframe} 行数为 {actual_per_day}，"
                f"严格要求 {expected_per_day}；拒绝补行或丢行"
            )

        for offset, close_time in enumerate(spec.close_times):
            row_index = cursor + offset
            pseudo_wall_clock = datetime.combine(trading_day, close_time, tzinfo=timezone.utc)
            pseudo_unix_seconds = int(pseudo_wall_clock.timestamp())
            expected_bucket = pseudo_unix_seconds // 1000
            actual_bucket = int(raw_times[row_index])
            if actual_bucket != expected_bucket:
                raise AShareDataError(
                    f"第 {row_index + 1} 行旧 time={actual_bucket} 无法反验 "
                    f"{trading_day.isoformat()} {close_time.strftime('%H:%M')} 的 1000秒桶 "
                    f"{expected_bucket}"
                )
            actual_close = datetime.combine(
                trading_day,
                close_time,
                tzinfo=_SHANGHAI_TIMEZONE,
            )
            reconstructed.append(int(actual_close.timestamp()))
        cursor = end

    output = np.asarray(reconstructed, dtype=np.int64)
    if len(output) != len(raw_times):
        raise AShareDataError("重建后行数与原始行数不一致")
    if len(output) > 1 and np.any(np.diff(output) <= 0):
        raise AShareDataError("重建后的 UTC time 不严格递增")
    return output


def _read_legacy_frame(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise AShareDataError("旧 A 股 Parquet 无法读取") from exc
    columns = tuple(str(column) for column in frame.columns)
    if columns != CANONICAL_COLUMNS:
        raise AShareDataError(
            f"旧 A 股列必须严格为 {list(CANONICAL_COLUMNS)}，当前为 {list(columns)}"
        )
    if frame.empty:
        raise AShareDataError("旧 A 股 Parquet 不能为空")
    if is_bool_dtype(frame["time"].dtype) or not is_integer_dtype(frame["time"].dtype):
        raise AShareDataError("旧 time 必须是整数 1000秒桶")
    if frame["time"].isna().any():
        raise AShareDataError("旧 time 含空值")

    int64_limits = np.iinfo(np.int64)
    for column in ("time", "tick_volume"):
        series = frame[column]
        if is_bool_dtype(series.dtype) or not is_integer_dtype(series.dtype):
            raise AShareDataError(f"旧 {column} 必须是非 bool 整数列")
        if series.isna().any():
            raise AShareDataError(f"旧 {column} 含空值")
        minimum = int(series.min())
        maximum = int(series.max())
        if minimum < int64_limits.min or maximum > int64_limits.max:
            raise AShareDataError(f"旧 {column} 超出 int64 可无损表示范围")
        converted = series.to_numpy(dtype=np.int64)
        if not np.array_equal(series.to_numpy(dtype=object), converted.astype(object)):
            raise AShareDataError(f"旧 {column} 转为 int64 后数值发生变化")

    for column in _PRICE_COLUMNS:
        series = frame[column]
        if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
            raise AShareDataError(f"旧 {column} 必须是非 bool 数值列")
        if series.isna().any():
            raise AShareDataError(f"旧 {column} 含空值")
        source_values = series.to_numpy(dtype=np.float64)
        converted = series.to_numpy(dtype=np.float32)
        if not np.isfinite(source_values).all() or not np.isfinite(converted).all():
            raise AShareDataError(f"旧 {column} 含 NaN、无穷值或 float32 溢出")
        if not np.array_equal(source_values, converted.astype(np.float64)):
            raise AShareDataError(f"旧 {column} 转为 float32 后数值发生变化")

    prices = frame[list(_PRICE_COLUMNS)].to_numpy(dtype=np.float64, copy=False)
    if np.any(prices <= 0):
        raise AShareDataError("旧 OHLC 必须全部大于 0")
    open_values, high_values, low_values, close_values = (prices[:, index] for index in range(4))
    if np.any(high_values < np.maximum(open_values, close_values)) or np.any(
        low_values > np.minimum(open_values, close_values)
    ):
        raise AShareDataError("旧 OHLC 价格关系非法")
    if np.any(frame["tick_volume"].to_numpy(dtype=np.int64, copy=False) < 0):
        raise AShareDataError("旧 tick_volume 不得为负")
    return frame


def _canonical_frame(legacy: pd.DataFrame, spec: ASharePeriodSpec) -> pd.DataFrame:
    raw_times = legacy["time"].to_numpy(dtype=np.int64, copy=False)
    utc_times = _reconstruct_utc_timestamps(raw_times, spec)
    output = pd.DataFrame(
        {
            "time": utc_times,
            "open": legacy["open"].to_numpy(dtype=np.float32),
            "high": legacy["high"].to_numpy(dtype=np.float32),
            "low": legacy["low"].to_numpy(dtype=np.float32),
            "close": legacy["close"].to_numpy(dtype=np.float32),
            "tick_volume": legacy["tick_volume"].to_numpy(dtype=np.int64),
        },
        columns=list(CANONICAL_COLUMNS),
    )
    validate_canonical_training_frame(output)
    if len(output) < spec.minimum_bars:
        raise AShareDataError(
            f"{spec.timeframe} 数据不足: {len(output)} bars（A股至少需要 "
            f"{spec.minimum_bars}，约 2 个交易年）"
        )
    return output


def _manifest_for(
    *,
    source_path: Path,
    output_path: Path,
    symbol: str,
    spec: ASharePeriodSpec,
    frame: pd.DataFrame,
    source_sha256: str,
) -> dict[str, Any]:
    digest = sha256_file(output_path)
    rows = len(frame)
    return {
        "format": ASHARE_DATASET_FORMAT,
        "source": ASHARE_SOURCE,
        "market": "CN_A_SHARE",
        "symbol": symbol,
        "timeframe": spec.timeframe,
        "periods_per_year": spec.periods_per_year,
        "minimum_bars": spec.minimum_bars,
        "data_filename": output_path.name,
        "data_sha256": digest,
        "dataset_id": f"sha256:{digest}",
        "data_rows": rows,
        "data_start": _utc_iso(int(frame["time"].iloc[0])),
        "data_end": _utc_iso(int(frame["time"].iloc[-1])),
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "bar_timestamp_semantics": "bar_close",
        "columns": list(CANONICAL_COLUMNS),
        "source_filename": source_path.name,
        "source_sha256": source_sha256,
        "source_time_encoding": "floor(china_local_wall_clock_unix_seconds/1000)",
        "source_timezone": "Asia/Shanghai",
        "session_close_times": [value.strftime("%H:%M") for value in spec.close_times],
        "coverage_in_trading_years": round(rows / spec.periods_per_year, 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def convert_legacy_a_share_file(input_file: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """转换一个旧个股文件；原文件只读，目标存在时拒绝覆盖。"""
    source_path = Path(input_file)
    if not source_path.is_file():
        raise AShareDataError(f"旧 A 股文件不存在: {source_path}")
    symbol, spec = _legacy_identity(source_path)
    destination_dir = Path(output_dir)
    raw_root = source_path.parent.parent.resolve()
    destination_resolved = destination_dir.resolve()
    if destination_resolved == raw_root or raw_root in destination_resolved.parents:
        raise AShareDataError("输出目录不得位于旧 A 股原始 parquet 目录内")
    output_path = destination_dir / f"{symbol}_{spec.timeframe}.parquet"
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise AShareDataError(f"目标已存在，拒绝覆盖: {output_path.name}")
    if output_path.resolve() == source_path.resolve():
        raise AShareDataError("输出路径不得覆盖原始 A 股文件")

    source_sha256 = sha256_file(source_path)
    legacy = _read_legacy_frame(source_path)
    canonical = _canonical_frame(legacy, spec)
    if sha256_file(source_path) != source_sha256:
        raise AShareDataError("原始 A 股文件在转换期间发生变化，拒绝发布")
    destination_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(8)
    data_temp = output_path.with_name(f".{output_path.name}.{token}.tmp")
    manifest_temp = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    output_published = False
    manifest_published = False
    try:
        canonical.to_parquet(data_temp, index=False)
        persisted = pd.read_parquet(data_temp)
        validate_canonical_training_frame(persisted)
        if not persisted.equals(canonical):
            raise AShareDataError("规范 Parquet 写入后复读不一致")
        manifest = _manifest_for(
            source_path=source_path,
            output_path=data_temp,
            symbol=symbol,
            spec=spec,
            frame=persisted,
            source_sha256=source_sha256,
        )
        manifest["data_filename"] = output_path.name
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for source, target, label in (
            (data_temp, output_path, "data"),
            (manifest_temp, manifest_path, "manifest"),
        ):
            try:
                # 同目录硬链接既拒绝覆盖并发创建的目标，也避开 Windows 上
                # os.replace 覆盖预占位文件时偶发的 WinError 5。
                os.link(source, target)
            except FileExistsError as exc:
                raise AShareDataError(
                    f"目标在转换期间已被其他进程创建，拒绝覆盖: {target.name}"
                ) from exc
            else:
                if label == "data":
                    output_published = True
                else:
                    manifest_published = True
    except Exception:
        # 只回滚本事务发布的硬链接，避免留下半成品或删除并发结果。
        if output_published:
            output_path.unlink(missing_ok=True)
        if manifest_published:
            manifest_path.unlink(missing_ok=True)
        raise
    finally:
        data_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)

    return {**manifest, "data_file": str(output_path.resolve()), "manifest_file": str(manifest_path.resolve())}


def load_a_share_manifest(data_file: str | Path, frame: pd.DataFrame) -> dict[str, Any] | None:
    """若 sidecar 声明 AShareLocal，则完整复核并返回；其他来源返回 None。"""
    path = Path(data_file)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AShareDataError("数据 manifest 不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise AShareDataError("数据 manifest 顶层必须是对象")
    if payload.get("source") != ASHARE_SOURCE:
        return None

    match = _CANONICAL_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise AShareDataError("AShareLocal 文件名必须为 6位代码_(M5|M15|H1|D1).parquet")
    symbol = match.group("symbol")
    timeframe = match.group("timeframe").upper()
    spec = ASHARE_SPECS_BY_TIMEFRAME[timeframe]
    validate_canonical_training_frame(frame)
    digest = sha256_file(path)
    expected = {
        "format": ASHARE_DATASET_FORMAT,
        "source": ASHARE_SOURCE,
        "market": "CN_A_SHARE",
        "symbol": symbol,
        "timeframe": timeframe,
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
        "source_time_encoding": "floor(china_local_wall_clock_unix_seconds/1000)",
        "session_close_times": [value.strftime("%H:%M") for value in spec.close_times],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise AShareDataError(f"AShareLocal manifest 的 {field} 与数据合同不匹配")
    if not isinstance(payload.get("source_filename"), str) or not payload["source_filename"]:
        raise AShareDataError("AShareLocal manifest 缺少 source_filename")
    if not isinstance(payload.get("source_sha256"), str) or re.fullmatch(
        r"[0-9a-f]{64}", payload["source_sha256"]
    ) is None:
        raise AShareDataError("AShareLocal manifest 的 source_sha256 非法")
    if len(frame) < spec.minimum_bars:
        raise AShareDataError(
            f"{timeframe} 数据不足: {len(frame)} bars（A股至少需要 {spec.minimum_bars}）"
        )
    return payload
