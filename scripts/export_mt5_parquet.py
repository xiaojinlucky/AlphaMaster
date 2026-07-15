"""从已登录的本机 MT5 导出已收盘 K 线和可校验的数据 manifest。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMEFRAME_ATTRIBUTES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}
REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]
DEFAULT_BARS = 50_000
_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')


class ExportError(RuntimeError):
    """MT5 数据或导出合同不满足要求。"""


def _positive_int(value: str) -> int:
    """解析严格正整数，拒绝符号、小数和非 ASCII 数字。"""
    if not value or not value.isascii() or not value.isdigit():
        raise argparse.ArgumentTypeError("必须是严格正整数")
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _validate_symbol(symbol: str) -> str:
    """保留 MT5 原始品种名，仅拒绝无法安全成为 Windows 文件名的值。"""
    if not symbol or symbol != symbol.strip() or symbol in {".", ".."}:
        raise ExportError("MT5 symbol 不能为空、包含首尾空白或为相对路径标记")
    if symbol.endswith((" ", ".")):
        raise ExportError("MT5 symbol 不能以空格或点结尾")
    if any(ord(char) < 32 or char in _WINDOWS_FORBIDDEN for char in symbol):
        raise ExportError("MT5 symbol 含有 Windows 文件名不允许的字符")
    return symbol


def _load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise ExportError("未安装官方 MetaTrader5 Python 包") from exc
    return mt5


def _timeframe_constant(mt5: Any, timeframe: str) -> int:
    attr = TIMEFRAME_ATTRIBUTES.get(timeframe)
    if attr is None:
        allowed = ", ".join(TIMEFRAME_ATTRIBUTES)
        raise ExportError(f"不支持的 timeframe: {timeframe}；仅允许 {allowed}")
    if not hasattr(mt5, attr):
        raise ExportError(f"当前 MetaTrader5 包缺少常量 {attr}")
    return int(getattr(mt5, attr))


def _build_dataframe(rates: Any) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ExportError(f"MT5 K 线缺少字段: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    try:
        raw_time = pd.to_numeric(df["time"], errors="raise").to_numpy(dtype=np.float64)
        numeric = df[["open", "high", "low", "close", "tick_volume"]].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ExportError(f"MT5 K 线包含非数值字段: {exc}") from exc

    int64 = np.iinfo(np.int64)
    if (
        not np.isfinite(raw_time).all()
        or (raw_time < 0).any()
        or (raw_time != np.floor(raw_time)).any()
        or (raw_time > int64.max).any()
    ):
        raise ExportError("time 必须是非负、有限的 int64 Unix 秒")

    numeric_values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ExportError("OHLCV 包含 NaN 或无穷值")
    volume = numeric["tick_volume"].to_numpy(dtype=np.float64)
    if (volume < 0).any() or (volume != np.floor(volume)).any():
        raise ExportError("tick_volume 必须是非负整数")

    df["time"] = raw_time.astype(np.int64)
    for column in ("open", "high", "low", "close"):
        df[column] = numeric[column].astype(np.float64)
    df["tick_volume"] = volume.astype(np.int64)

    # MT5 理论上已按时间返回；仍按数据合同显式去重、升序，重复时间保留最后一条。
    df = (
        df.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time", kind="mergesort")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ExportError("MT5 未返回可导出的已收盘 K 线")
    return df


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_iso_from_unix(value: int) -> str:
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError) as exc:
        raise ExportError(f"Unix 时间超出可表示范围: {value}") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_json_staging(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_mt5_parquet(
    *,
    symbol: str,
    timeframe: str,
    bars: int,
    output_dir: str | Path,
    mt5_module: Any | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """导出一个准确 MT5 品种/周期；不读取或保存账户凭据。"""
    symbol = _validate_symbol(symbol)
    if not isinstance(bars, int) or isinstance(bars, bool) or bars <= 0:
        raise ExportError("bars 必须是严格正整数")

    mt5 = mt5_module if mt5_module is not None else _load_mt5()
    timeframe_value = _timeframe_constant(mt5, timeframe)
    initialized = False
    try:
        if not mt5.initialize():
            raise ExportError(f"MT5 初始化失败: {mt5.last_error()}")
        initialized = True
        if mt5.account_info() is None:
            raise ExportError("MT5 终端未登录交易账户")

        info = mt5.symbol_info(symbol)
        if info is None or getattr(info, "name", None) != symbol:
            raise ExportError(f"MT5 中不存在精确品种名: {symbol}")
        if not bool(getattr(info, "visible", False)):
            if not mt5.symbol_select(symbol, True):
                raise ExportError(f"无法在 Market Watch 中启用品种: {symbol}")
            info = mt5.symbol_info(symbol)
            if info is None or getattr(info, "name", None) != symbol:
                raise ExportError(f"启用后无法再次确认精确品种名: {symbol}")

        # MT5 position=0 是正在形成的当前 bar；从 1 开始只请求已收盘 bar。
        rates = mt5.copy_rates_from_pos(symbol, timeframe_value, 1, bars)
        if rates is None or len(rates) == 0:
            raise ExportError(
                f"MT5 未返回 {symbol} {timeframe} 的已收盘 K 线: {mt5.last_error()}"
            )
        df = _build_dataframe(rates)
    finally:
        if initialized:
            mt5.shutdown()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{timeframe}"
    parquet_path = destination / f"{stem}.parquet"
    manifest_path = destination / f"{stem}.manifest.json"
    token = uuid.uuid4().hex
    parquet_staging = destination / f".{parquet_path.name}.{token}.partial"
    manifest_staging = destination / f".{manifest_path.name}.{token}.partial"

    try:
        df.to_parquet(parquet_staging, index=False)
        data_sha256 = _sha256(parquet_staging)
        exported_at = _utc_now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        manifest: dict[str, Any] = {
            "format": "alphamaster_mt5_dataset_v1",
            "source": "MetaTrader5",
            "symbol": symbol,
            "timeframe": timeframe,
            "data_filename": parquet_path.name,
            "data_sha256": data_sha256,
            "data_rows": int(len(df)),
            "data_start": _utc_iso_from_unix(int(df["time"].iloc[0])),
            "data_end": _utc_iso_from_unix(int(df["time"].iloc[-1])),
            "data_timezone": "UTC",
            "time_unit": "unix_seconds",
            "exported_at": exported_at,
            "columns": list(REQUIRED_COLUMNS),
        }
        _write_json_staging(manifest_staging, manifest)

        # 两个文件均先在同目录完整写入，再以原子替换发布；manifest 最后发布。
        os.replace(parquet_staging, parquet_path)
        os.replace(manifest_staging, manifest_path)
    finally:
        parquet_staging.unlink(missing_ok=True)
        manifest_staging.unlink(missing_ok=True)

    return parquet_path, manifest_path, manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从本机 MT5 导出已收盘 K 线 Parquet")
    parser.add_argument("--symbol", required=True, help="MT5 Market Watch 中的准确品种名")
    parser.add_argument(
        "--timeframe",
        required=True,
        choices=tuple(TIMEFRAME_ATTRIBUTES),
        help="准确周期名，不进行大小写或别名转换",
    )
    parser.add_argument(
        "--bars",
        type=_positive_int,
        default=DEFAULT_BARS,
        help=f"最多导出的已收盘 K 线数（默认: {DEFAULT_BARS}）",
    )
    parser.add_argument("--output-dir", required=True, help="Parquet 与 manifest 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        parquet_path, manifest_path, manifest = export_mt5_parquet(
            symbol=args.symbol,
            timeframe=args.timeframe,
            bars=args.bars,
            output_dir=args.output_dir,
        )
    except (ExportError, ImportError, OSError, ValueError) as exc:
        print(f"导出失败: {exc}", file=sys.stderr)
        return 1

    print(
        f"导出完成: {parquet_path} | rows={manifest['data_rows']} | "
        f"sha256={manifest['data_sha256']}"
    )
    print(f"数据 manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
