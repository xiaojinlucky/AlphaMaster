"""从 AKShare 获取可冻结、可复核的 A 股前复权日线训练数据。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import re
import secrets
from datetime import date, datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_pipeline.a_share_data import (
    AShareDataError,
    ASHARE_SPECS_BY_TIMEFRAME,
    CANONICAL_COLUMNS,
    sha256_file,
    validate_canonical_training_frame,
)
from data_pipeline.dataset_contracts import (
    AKSHARE_HFQ_FORMAT,
    AKSHARE_HFQ_SOURCE_ID,
    AKSHARE_SOURCE,
)


AKSHARE_INTERFACE = "stock_zh_a_daily"
AKSHARE_ADJUSTMENT = "hfq"
AKSHARE_DOC_URL = "https://akshare.akfamily.xyz/data/stock/stock.html"
AKSHARE_PROVIDER_UPSTREAM = (
    "Sina Finance A-share historical daily data and adjustment factors"
)
AKSHARE_SLICE_FORMAT = "alphamaster_ashare_dataset_slice_v1"
AKSHARE_SLICE_TRAINING = "training"
AKSHARE_SLICE_SEALED_EVALUATION = "sealed_oos_evaluation"
AKSHARE_MIN_EVALUATION_WARMUP_BARS = 200
AKSHARE_SOURCE_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "outstanding_share",
    "turnover",
)
_CANONICAL_FILENAME_RE = re.compile(r"^(?P<symbol>[0-9]{6})_D1\.parquet$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[.+-].*)?$")
_SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
_D1_SPEC = ASHARE_SPECS_BY_TIMEFRAME["D1"]

Fetcher = Callable[..., pd.DataFrame]


class AKShareDataError(AShareDataError):
    """AKShare 数据或来源合同不满足训练要求。"""


def _validate_symbol(value: str) -> str:
    symbol = str(value)
    if re.fullmatch(r"[0-9]{6}", symbol) is None:
        raise AKShareDataError("A 股代码必须是 6 位数字")
    return symbol


def akshare_sina_provider_symbol(symbol: str) -> str:
    """把规范 6 位代码转换为新浪接口实际接收的市场前缀代码。"""
    canonical = _validate_symbol(symbol)
    if canonical.startswith("6"):
        return f"sh{canonical}"
    if canonical.startswith(("0", "3")):
        return f"sz{canonical}"
    raise AKShareDataError(
        f"新浪 A 股日线接口不支持该股票代码前缀: {canonical}"
    )


def _parse_yyyymmdd(value: str, label: str) -> date:
    if re.fullmatch(r"[0-9]{8}", str(value)) is None:
        raise AKShareDataError(f"{label} 必须是 YYYYMMDD")
    try:
        return datetime.strptime(str(value), "%Y%m%d").date()
    except ValueError as exc:
        raise AKShareDataError(f"{label} 不是合法日期") from exc


def _utc_iso(unix_seconds: int) -> str:
    return (
        datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise AKShareDataError(f"{label} 必须是 UTC 时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AKShareDataError(f"{label} 必须是 UTC 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AKShareDataError(f"{label} 必须是 UTC 时间")
    return parsed


def _provider_payload_sha256(frame: pd.DataFrame) -> str:
    """冻结供应商返回值；哈希只作来源证据，不代替正式 Parquet 哈希。"""
    serialized = frame.to_json(
        orient="split",
        date_format="iso",
        force_ascii=False,
        double_precision=15,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _sina_fetch_worker(connection: Connection, request: dict[str, str]) -> None:
    """在独立进程中调用 AKShare，允许父进程硬性终止卡死请求。"""
    try:
        import akshare as ak

        frame = ak.stock_zh_a_daily(**request)
        connection.send({"ok": True, "frame": frame})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        connection.close()


def _fetch_sina_hfq_with_timeout(
    *,
    provider_symbol: str,
    start_date: str,
    end_date: str,
    timeout: float,
) -> pd.DataFrame:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    request = {
        "symbol": provider_symbol,
        "start_date": start_date,
        "end_date": end_date,
        "adjust": AKSHARE_ADJUSTMENT,
    }
    process = context.Process(
        target=_sina_fetch_worker,
        args=(sender, request),
        daemon=True,
    )
    process.start()
    sender.close()
    payload: Any = None
    try:
        if not receiver.poll(timeout):
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError(
                f"AKShare 新浪日线请求超过 {timeout:g} 秒"
            )
        try:
            payload = receiver.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"AKShare 新浪下载子进程提前退出，exitcode={process.exitcode}"
            ) from exc
    finally:
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        error_type = (
            payload.get("error_type")
            if isinstance(payload, dict)
            else "InvalidWorkerResponse"
        )
        error = payload.get("error") if isinstance(payload, dict) else repr(payload)
        raise RuntimeError(f"{error_type}: {error}")
    frame = payload.get("frame")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("AKShare 新浪下载子进程没有返回 DataFrame")
    return frame


def canonicalize_akshare_hfq_daily(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """把 AKShare 前复权日线严格转换为 AlphaMaster 规范列。"""
    symbol = _validate_symbol(symbol)
    if not isinstance(raw, pd.DataFrame):
        raise AKShareDataError("AKShare 返回值必须是 DataFrame")
    columns = tuple(str(column) for column in raw.columns)
    if columns != AKSHARE_SOURCE_COLUMNS:
        raise AKShareDataError(
            "AKShare 返回列合同变化，拒绝猜测映射: "
            f"期望 {list(AKSHARE_SOURCE_COLUMNS)}，实际 {list(columns)}"
        )
    if raw.empty:
        raise AKShareDataError("AKShare 没有返回日线数据")

    try:
        trading_dates = pd.to_datetime(
            raw["date"],
            format="%Y-%m-%d",
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise AKShareDataError("AKShare 日期列无法按交易日解析") from exc
    if trading_dates.isna().any():
        raise AKShareDataError("AKShare 日期列含空值")
    if not trading_dates.is_monotonic_increasing or trading_dates.duplicated().any():
        raise AKShareDataError("AKShare 日期必须严格递增且不得重复")
    if bool((trading_dates.dt.weekday >= 5).any()):
        raise AKShareDataError("AKShare 日线包含周末日期")

    price_columns = ("open", "high", "low", "close")
    numeric: dict[str, pd.Series] = {}
    try:
        for column in price_columns:
            numeric[column] = pd.to_numeric(raw[column], errors="raise")
        volume = pd.to_numeric(raw["volume"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise AKShareDataError("AKShare OHLCV 含非数值") from exc
    if any(series.isna().any() for series in numeric.values()) or volume.isna().any():
        raise AKShareDataError("AKShare OHLCV 含空值")
    volume_values = volume.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(volume_values).all() or np.any(volume_values < 0):
        raise AKShareDataError("AKShare 成交量必须是非负有限值")
    if not np.equal(volume_values, np.floor(volume_values)).all():
        raise AKShareDataError("AKShare 成交量必须是整数股")
    if np.any(volume_values > np.iinfo(np.int64).max):
        raise AKShareDataError("AKShare 成交量超出 int64 范围")

    local_close = trading_dates.dt.tz_localize(_SHANGHAI_TIMEZONE) + pd.Timedelta(
        hours=15
    )
    # pandas 2/3 的内部 datetime 分辨率可能是 ns 或 us，不能用固定除数。
    unix_seconds = local_close.dt.tz_convert(timezone.utc).map(
        lambda value: int(value.timestamp())
    ).to_numpy(dtype=np.int64)
    canonical = pd.DataFrame(
        {
            "time": unix_seconds,
            "open": numeric["open"].to_numpy(dtype=np.float32),
            "high": numeric["high"].to_numpy(dtype=np.float32),
            "low": numeric["low"].to_numpy(dtype=np.float32),
            "close": numeric["close"].to_numpy(dtype=np.float32),
            "tick_volume": volume_values.astype(np.int64),
        },
        columns=list(CANONICAL_COLUMNS),
    )
    validate_canonical_training_frame(canonical)
    if len(canonical) < _D1_SPEC.minimum_bars:
        raise AKShareDataError(
            f"D1 数据不足: {len(canonical)} bars（至少需要 "
            f"{_D1_SPEC.minimum_bars}，约两个交易年）"
        )
    return canonical


def _manifest_for(
    *,
    output_path: Path,
    symbol: str,
    frame: pd.DataFrame,
    provider_version: str,
    provider_symbol: str,
    request_start_date: str,
    request_end_date: str,
    source_response_sha256: str,
) -> dict[str, Any]:
    digest = sha256_file(output_path)
    rows = len(frame)
    snapshot_date = datetime.fromtimestamp(
        int(frame["time"].iloc[-1]),
        tz=timezone.utc,
    ).astimezone(_SHANGHAI_TIMEZONE).date()
    return {
        "format": AKSHARE_HFQ_FORMAT,
        "source": AKSHARE_SOURCE,
        "source_id": AKSHARE_HFQ_SOURCE_ID,
        "market": "CN_A_SHARE",
        "symbol": symbol,
        "timeframe": "D1",
        "periods_per_year": _D1_SPEC.periods_per_year,
        "minimum_bars": _D1_SPEC.minimum_bars,
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
        "source_timezone": "Asia/Shanghai",
        "session_close_times": ["15:00"],
        "provider": "AKShare",
        "provider_version": provider_version,
        "provider_interface": AKSHARE_INTERFACE,
        "provider_documentation": AKSHARE_DOC_URL,
        "provider_upstream": AKSHARE_PROVIDER_UPSTREAM,
        "request": {
            "canonical_symbol": symbol,
            "symbol": provider_symbol,
            "start_date": request_start_date,
            "end_date": request_end_date,
            "adjust": AKSHARE_ADJUSTMENT,
        },
        "adjustment": AKSHARE_ADJUSTMENT,
        "adjustment_history_semantics": (
            "cumulative_historical_factor_not_latest_price_normalized"
        ),
        "bar_completion": "completed_trading_days_only",
        "tick_volume_semantics": "新浪成交量原值，单位为股，未换算为手",
        "source_columns": list(AKSHARE_SOURCE_COLUMNS),
        "source_response_sha256": source_response_sha256,
        "data_snapshot_date": snapshot_date.strftime("%Y%m%d"),
        "coverage_in_trading_years": round(
            rows / _D1_SPEC.periods_per_year,
            6,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def download_akshare_hfq_daily(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    timeout: float = 20.0,
    fetcher: Fetcher | None = None,
    provider_version: str | None = None,
) -> dict[str, Any]:
    """下载一个标的并发布不可覆盖的 Parquet + manifest。"""
    symbol = _validate_symbol(symbol)
    start = _parse_yyyymmdd(start_date, "start_date")
    end = _parse_yyyymmdd(end_date, "end_date")
    if start >= end:
        raise AKShareDataError("start_date 必须早于 end_date")
    if end >= datetime.now(_SHANGHAI_TIMEZONE).date():
        raise AKShareDataError("end_date 必须早于上海当前日期，避免未收盘日线")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise AKShareDataError("timeout 必须是正数")
    provider_symbol = akshare_sina_provider_symbol(symbol)

    destination_dir = Path(output_dir).resolve()
    output_path = destination_dir / f"{symbol}_D1.parquet"
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise AKShareDataError(f"目标已存在，拒绝静默覆盖: {output_path.name}")

    if fetcher is None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise AKShareDataError("缺少 akshare，无法下载 A 股前复权日线") from exc
        provider_version = importlib.metadata.version("akshare")
    elif provider_version is None:
        provider_version = "test-double"
    if provider_version != "test-double" and _VERSION_RE.fullmatch(provider_version) is None:
        raise AKShareDataError("AKShare 版本号非法")

    try:
        provider_request = {
            "symbol": provider_symbol,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "adjust": AKSHARE_ADJUSTMENT,
        }
        if fetcher is None:
            raw = _fetch_sina_hfq_with_timeout(
                provider_symbol=provider_symbol,
                start_date=provider_request["start_date"],
                end_date=provider_request["end_date"],
                timeout=float(timeout),
            )
        else:
            raw = fetcher(**provider_request)
    except Exception as exc:
        raise AKShareDataError(
            f"AKShare 下载失败，未生成任何数据: {type(exc).__name__}: {exc}"
        ) from exc
    source_response_sha256 = _provider_payload_sha256(raw)
    canonical = canonicalize_akshare_hfq_daily(raw, symbol=symbol)
    actual_start = datetime.fromtimestamp(
        int(canonical["time"].iloc[0]),
        tz=timezone.utc,
    ).astimezone(_SHANGHAI_TIMEZONE).date()
    actual_end = datetime.fromtimestamp(
        int(canonical["time"].iloc[-1]),
        tz=timezone.utc,
    ).astimezone(_SHANGHAI_TIMEZONE).date()
    if actual_start < start or actual_end > end:
        raise AKShareDataError("AKShare 返回日期超出请求范围")

    destination_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(8)
    data_temp = output_path.with_name(f".{output_path.name}.{token}.tmp")
    manifest_temp = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    data_published = False
    manifest_published = False
    try:
        canonical.to_parquet(data_temp, index=False)
        persisted = pd.read_parquet(data_temp)
        validate_canonical_training_frame(persisted)
        if not persisted.equals(canonical):
            raise AKShareDataError("规范 Parquet 写入后复读不一致")
        manifest = _manifest_for(
            output_path=data_temp,
            symbol=symbol,
            frame=persisted,
            provider_version=str(provider_version),
            provider_symbol=provider_symbol,
            request_start_date=start.strftime("%Y%m%d"),
            request_end_date=end.strftime("%Y%m%d"),
            source_response_sha256=source_response_sha256,
        )
        manifest["data_filename"] = output_path.name
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for source_path, target_path, label in (
            (data_temp, output_path, "data"),
            (manifest_temp, manifest_path, "manifest"),
        ):
            try:
                os.link(source_path, target_path)
            except FileExistsError as exc:
                raise AKShareDataError(
                    f"目标在下载期间被其他进程创建，拒绝覆盖: {target_path.name}"
                ) from exc
            else:
                if label == "data":
                    data_published = True
                else:
                    manifest_published = True
    except Exception:
        if data_published:
            output_path.unlink(missing_ok=True)
        if manifest_published:
            manifest_path.unlink(missing_ok=True)
        raise
    finally:
        data_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)

    return {
        **manifest,
        "data_file": str(output_path),
        "manifest_file": str(manifest_path),
    }


def publish_akshare_hfq_slice(
    *,
    parent_data_file: str | Path,
    output_dir: str | Path,
    start_index: int,
    end_index: int,
    purpose: str,
    universe_contract_sha256: str,
    score_start_index: int | None = None,
) -> dict[str, Any]:
    """从已冻结父数据发布物理隔离的训练或封存评估切片。"""
    parent_path = Path(parent_data_file).resolve()
    parent_manifest_path = parent_path.with_suffix(".manifest.json")
    try:
        parent_frame = pd.read_parquet(parent_path)
    except Exception as exc:
        raise AKShareDataError(
            f"无法读取父级 AKShare 数据: {type(exc).__name__}: {exc}"
        ) from exc
    parent_manifest = load_akshare_hfq_manifest(parent_path, parent_frame)
    if parent_manifest is None:
        raise AKShareDataError("父级数据不是 AKShare 新浪后复权合同")
    if parent_manifest.get("derivation") is not None:
        raise AKShareDataError("只允许从供应商原始冻结数据创建一级切片")
    if (
        not isinstance(universe_contract_sha256, str)
        or _SHA256_RE.fullmatch(universe_contract_sha256) is None
    ):
        raise AKShareDataError("universe_contract_sha256 非法")
    for value, label in (
        (start_index, "start_index"),
        (end_index, "end_index"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise AKShareDataError(f"{label} 必须是整数")
    if not 0 <= start_index < end_index <= len(parent_frame):
        raise AKShareDataError("切片索引范围非法")
    if purpose not in {
        AKSHARE_SLICE_TRAINING,
        AKSHARE_SLICE_SEALED_EVALUATION,
    }:
        raise AKShareDataError("切片 purpose 不受支持")

    if purpose == AKSHARE_SLICE_TRAINING:
        if score_start_index is not None:
            raise AKShareDataError("训练切片不得声明 score_start_index")
        if start_index != 0 or end_index >= len(parent_frame):
            raise AKShareDataError("训练切片必须从父数据起点开始并在终点前结束")
        warmup_bars = 0
        score_start = None
    else:
        if (
            isinstance(score_start_index, bool)
            or not isinstance(score_start_index, int)
            or not start_index < score_start_index < end_index
        ):
            raise AKShareDataError("封存评估切片的 score_start_index 非法")
        if end_index != len(parent_frame):
            raise AKShareDataError("封存评估切片必须覆盖父数据终点")
        warmup_bars = score_start_index - start_index
        if warmup_bars < AKSHARE_MIN_EVALUATION_WARMUP_BARS:
            raise AKShareDataError(
                "封存评估切片至少需要 200 根只读预热 K 线"
            )
        score_start = _utc_iso(int(parent_frame["time"].iloc[score_start_index]))

    frame = parent_frame.iloc[start_index:end_index].reset_index(drop=True)
    validate_canonical_training_frame(frame)
    if len(frame) < _D1_SPEC.minimum_bars:
        raise AKShareDataError(
            f"切片数据不足: {len(frame)} bars（至少需要 {_D1_SPEC.minimum_bars}）"
        )

    destination_dir = Path(output_dir).resolve()
    output_path = destination_dir / parent_path.name
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise AKShareDataError(f"切片目标已存在，拒绝覆盖: {output_path.name}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    token = secrets.token_hex(8)
    data_temp = output_path.with_name(f".{output_path.name}.{token}.tmp")
    manifest_temp = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    data_published = False
    manifest_published = False
    try:
        frame.to_parquet(data_temp, index=False)
        persisted = pd.read_parquet(data_temp)
        validate_canonical_training_frame(persisted)
        if not persisted.equals(frame):
            raise AKShareDataError("切片 Parquet 写入后复读不一致")
        digest = sha256_file(data_temp)
        data_start = _utc_iso(int(persisted["time"].iloc[0]))
        data_end = _utc_iso(int(persisted["time"].iloc[-1]))
        manifest = dict(parent_manifest)
        manifest.update(
            {
                "data_filename": output_path.name,
                "data_sha256": digest,
                "dataset_id": f"sha256:{digest}",
                "data_rows": len(persisted),
                "data_start": data_start,
                "data_end": data_end,
                "data_snapshot_date": datetime.fromtimestamp(
                    int(persisted["time"].iloc[-1]),
                    tz=timezone.utc,
                ).astimezone(_SHANGHAI_TIMEZONE).strftime("%Y%m%d"),
                "coverage_in_trading_years": round(
                    len(persisted) / _D1_SPEC.periods_per_year,
                    6,
                ),
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "derivation": {
                    "format": AKSHARE_SLICE_FORMAT,
                    "purpose": purpose,
                    "parent_data_sha256": parent_manifest["data_sha256"],
                    "parent_dataset_id": parent_manifest["dataset_id"],
                    "parent_manifest_sha256": sha256_file(parent_manifest_path),
                    "parent_data_rows": parent_manifest["data_rows"],
                    "parent_data_start": parent_manifest["data_start"],
                    "parent_data_end": parent_manifest["data_end"],
                    "slice_start": data_start,
                    "slice_end": data_end,
                    "score_start": score_start,
                    "warmup_bars": warmup_bars,
                    "universe_contract_sha256": universe_contract_sha256,
                },
            }
        )
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for source_path, target_path, label in (
            (data_temp, output_path, "data"),
            (manifest_temp, manifest_path, "manifest"),
        ):
            try:
                os.link(source_path, target_path)
            except FileExistsError as exc:
                raise AKShareDataError(
                    f"切片目标在发布期间被创建，拒绝覆盖: {target_path.name}"
                ) from exc
            if label == "data":
                data_published = True
            else:
                manifest_published = True
    except Exception:
        if data_published:
            output_path.unlink(missing_ok=True)
        if manifest_published:
            manifest_path.unlink(missing_ok=True)
        raise
    finally:
        data_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)

    loaded = load_akshare_hfq_manifest(output_path, pd.read_parquet(output_path))
    if loaded is None:
        raise AKShareDataError("发布后的切片未被 AKShare 合同识别")
    return {
        **loaded,
        "data_file": str(output_path),
        "manifest_file": str(manifest_path),
    }


def _validate_slice_derivation(
    payload: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    derivation = payload.get("derivation")
    if derivation is None:
        return
    expected_keys = {
        "format",
        "purpose",
        "parent_data_sha256",
        "parent_dataset_id",
        "parent_manifest_sha256",
        "parent_data_rows",
        "parent_data_start",
        "parent_data_end",
        "slice_start",
        "slice_end",
        "score_start",
        "warmup_bars",
        "universe_contract_sha256",
    }
    if not isinstance(derivation, dict) or set(derivation) != expected_keys:
        raise AKShareDataError("AKShare hfq derivation 字段合同不匹配")
    if derivation["format"] != AKSHARE_SLICE_FORMAT:
        raise AKShareDataError("AKShare hfq derivation format 不匹配")
    purpose = derivation["purpose"]
    if purpose not in {
        AKSHARE_SLICE_TRAINING,
        AKSHARE_SLICE_SEALED_EVALUATION,
    }:
        raise AKShareDataError("AKShare hfq derivation purpose 不受支持")
    for field in (
        "parent_data_sha256",
        "parent_manifest_sha256",
        "universe_contract_sha256",
    ):
        value = derivation[field]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise AKShareDataError(f"AKShare hfq derivation.{field} 非法")
    if derivation["parent_dataset_id"] != (
        f"sha256:{derivation['parent_data_sha256']}"
    ):
        raise AKShareDataError("AKShare hfq derivation 父数据身份不一致")
    if derivation["parent_data_sha256"] == payload["data_sha256"]:
        raise AKShareDataError("AKShare hfq 切片不能与父数据使用相同 hash")
    parent_rows = derivation["parent_data_rows"]
    if (
        isinstance(parent_rows, bool)
        or not isinstance(parent_rows, int)
        or parent_rows <= len(frame)
    ):
        raise AKShareDataError("AKShare hfq derivation.parent_data_rows 非法")
    parent_start = _parse_utc(
        derivation["parent_data_start"],
        "derivation.parent_data_start",
    )
    parent_end = _parse_utc(
        derivation["parent_data_end"],
        "derivation.parent_data_end",
    )
    slice_start = _parse_utc(
        derivation["slice_start"],
        "derivation.slice_start",
    )
    slice_end = _parse_utc(
        derivation["slice_end"],
        "derivation.slice_end",
    )
    if not parent_start <= slice_start < slice_end <= parent_end:
        raise AKShareDataError("AKShare hfq derivation 切片范围超出父数据")
    if (
        derivation["slice_start"] != payload["data_start"]
        or derivation["slice_end"] != payload["data_end"]
    ):
        raise AKShareDataError("AKShare hfq derivation 切片范围与数据不一致")
    warmup = derivation["warmup_bars"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise AKShareDataError("AKShare hfq derivation.warmup_bars 非法")

    if purpose == AKSHARE_SLICE_TRAINING:
        if (
            derivation["score_start"] is not None
            or warmup != 0
            or slice_start != parent_start
            or slice_end >= parent_end
        ):
            raise AKShareDataError("AKShare hfq 训练切片语义不匹配")
        return

    score_start = _parse_utc(
        derivation["score_start"],
        "derivation.score_start",
    )
    if (
        warmup < AKSHARE_MIN_EVALUATION_WARMUP_BARS
        or not slice_start < score_start <= slice_end
        or slice_end != parent_end
    ):
        raise AKShareDataError("AKShare hfq 封存评估切片语义不匹配")
    score_seconds = int(score_start.timestamp())
    times = frame["time"].to_numpy(dtype=np.int64, copy=False)
    score_index = int(np.searchsorted(times, score_seconds, side="left"))
    if (
        score_index >= len(times)
        or int(times[score_index]) != score_seconds
        or score_index != warmup
    ):
        raise AKShareDataError("AKShare hfq 封存评估预热边界不匹配")


def load_akshare_hfq_manifest(
    data_file: str | Path,
    frame: pd.DataFrame,
) -> dict[str, Any] | None:
    """若 sidecar 声明 AKShare 新浪后复权，则完整复核并返回。"""
    path = Path(data_file)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AKShareDataError("数据 manifest 不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise AKShareDataError("数据 manifest 顶层必须是对象")
    if payload.get("source") != AKSHARE_SOURCE:
        return None

    match = _CANONICAL_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise AKShareDataError("AKShare A股文件名必须为 6位代码_D1.parquet")
    symbol = match.group("symbol")
    validate_canonical_training_frame(frame)
    digest = sha256_file(path)
    expected = {
        "format": AKSHARE_HFQ_FORMAT,
        "source": AKSHARE_SOURCE,
        "source_id": AKSHARE_HFQ_SOURCE_ID,
        "market": "CN_A_SHARE",
        "symbol": symbol,
        "timeframe": "D1",
        "periods_per_year": _D1_SPEC.periods_per_year,
        "minimum_bars": _D1_SPEC.minimum_bars,
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
        "provider": "AKShare",
        "provider_interface": AKSHARE_INTERFACE,
        "provider_documentation": AKSHARE_DOC_URL,
        "provider_upstream": AKSHARE_PROVIDER_UPSTREAM,
        "adjustment": AKSHARE_ADJUSTMENT,
        "adjustment_history_semantics": (
            "cumulative_historical_factor_not_latest_price_normalized"
        ),
        "bar_completion": "completed_trading_days_only",
        "tick_volume_semantics": "新浪成交量原值，单位为股，未换算为手",
        "source_columns": list(AKSHARE_SOURCE_COLUMNS),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise AKShareDataError(
                f"AKShare hfq manifest 的 {field} 与数据合同不匹配"
            )
    version = payload.get("provider_version")
    if version != "test-double" and (
        not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None
    ):
        raise AKShareDataError("AKShare hfq manifest 的 provider_version 非法")
    source_hash = payload.get("source_response_sha256")
    if not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None:
        raise AKShareDataError("AKShare hfq manifest 的来源响应哈希非法")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise AKShareDataError("AKShare hfq manifest 缺少 request")
    expected_request = {
        "canonical_symbol": symbol,
        "symbol": akshare_sina_provider_symbol(symbol),
        "start_date": request.get("start_date"),
        "end_date": request.get("end_date"),
        "adjust": AKSHARE_ADJUSTMENT,
    }
    if request != expected_request:
        raise AKShareDataError("AKShare hfq manifest 的 request 不匹配")
    request_start = _parse_yyyymmdd(request.get("start_date"), "request.start_date")
    request_end = _parse_yyyymmdd(request.get("end_date"), "request.end_date")
    if request_start >= request_end:
        raise AKShareDataError("AKShare hfq manifest 的请求日期范围非法")
    data_start = _parse_utc(payload.get("data_start"), "data_start").astimezone(
        _SHANGHAI_TIMEZONE
    ).date()
    data_end = _parse_utc(payload.get("data_end"), "data_end").astimezone(
        _SHANGHAI_TIMEZONE
    ).date()
    if data_start < request_start or data_end > request_end:
        raise AKShareDataError("AKShare hfq 数据范围超出请求范围")
    if payload.get("data_snapshot_date") != data_end.strftime("%Y%m%d"):
        raise AKShareDataError("AKShare hfq manifest 的 data_snapshot_date 不匹配")
    _parse_utc(payload.get("created_at"), "created_at")
    _validate_slice_derivation(payload, frame)
    if len(frame) < _D1_SPEC.minimum_bars:
        raise AKShareDataError(
            f"D1 数据不足: {len(frame)} bars（至少需要 {_D1_SPEC.minimum_bars}）"
        )
    return payload
