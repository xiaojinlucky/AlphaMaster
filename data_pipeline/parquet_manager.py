"""Load training data from a single Parquet K-line file."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from config import Config
from data_pipeline.a_share_data import (
    AShareDataError,
    ASHARE_SOURCE,
    ASHARE_SOURCE_ID,
    ASHARE_SPECS_BY_TIMEFRAME,
    load_a_share_manifest,
    sha256_file,
    validate_canonical_training_frame,
)
from data_pipeline.a_share_akshare import (
    AKSHARE_SLICE_SEALED_EVALUATION,
    load_akshare_hfq_manifest,
)
from data_pipeline.dataset_contracts import (
    AKSHARE_HFQ_SOURCE_ID,
    AKSHARE_SOURCE,
    DATA_SOURCE_IDS,
    FREE_STOCKDB_QFQ_SOURCE_ID,
    FREE_STOCKDB_SOURCE,
    GENERIC_SOURCE_CONTRACTS,
    MT5_LEGACY_SOURCE_ID,
    OKX_LEGACY_SOURCE_ID,
    OKX_SOURCE_ID,
    infer_periods_per_year,
    resolve_okx_source_id,
)
from data_pipeline.free_stockdb_data import (
    FreeStockDBDataError,
    load_free_stockdb_qfq_manifest,
)
from data_pipeline.data_manager import MT5DataManager
from model_core.features import MT5FeatureEngineer

_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
_PARQUET_RE = re.compile(
    r"^(.+)_([^.]+)\.parquet$",
    re.IGNORECASE,
)
_TF_ALIASES: dict[str, str] = {
    # Bare 1m means one minute. Month aliases must be explicit (1mo/month/...)
    # so that third-party minute exports cannot be silently interpreted as months.
    "m1": "M1",
    "1m": "M1",
    "1min": "M1",
    "min1": "M1",
    "m5": "M5",
    "5m": "M5",
    "5min": "M5",
    "min5": "M5",
    "m15": "M15",
    "15m": "M15",
    "15min": "M15",
    "min15": "M15",
    "m30": "M30",
    "30m": "M30",
    "30min": "M30",
    "min30": "M30",
    "h1": "H1",
    "1h": "H1",
    "60m": "H1",
    "60min": "H1",
    "min60": "H1",
    "60": "H1",
    "h4": "H4",
    "4h": "H4",
    "240m": "H4",
    "240min": "H4",
    "min240": "H4",
    "240": "H4",
    "d1": "D1",
    "1d": "D1",
    "day": "D1",
    "daily": "D1",
    "1440m": "D1",
    "1440min": "D1",
    "w1": "W1",
    "1w": "W1",
    "week": "W1",
    "weekly": "W1",
    "mn1": "MN1",
    "1mo": "MN1",
    "1mon": "MN1",
    "month": "MN1",
    "monthly": "MN1",
}
_CANONICAL_ASHARE_RE = re.compile(r"^[0-9]{6}_(M5|M15|H1|D1)\.parquet$", re.IGNORECASE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _utc_iso(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _read_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("数据 manifest 不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("数据 manifest 顶层必须是对象")
    return payload


def _validate_generic_manifest(
    *,
    path: Path,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    digest: str,
    inferred_periods_per_year: int,
) -> tuple[str, str]:
    source_id = GENERIC_SOURCE_CONTRACTS.get(
        (payload.get("source"), payload.get("format"))
    )
    if source_id is None:
        raise ValueError("数据 manifest 的 source/format 不受支持")
    if source_id == MT5_LEGACY_SOURCE_ID:
        if payload.get("source_family") != "MetaTrader5":
            raise ValueError("旧 MT5 manifest 的 source_family 必须是 MetaTrader5")
        if payload.get("provenance_level") != "legacy_user_attested":
            raise ValueError("旧 MT5 manifest 必须声明 legacy_user_attested")
        if payload.get("attestation_scope") != "exact_file_bytes":
            raise ValueError("旧 MT5 manifest 必须声明 exact_file_bytes")
        if payload.get("registration_method") != "legacy_sidecar_registration_v1":
            raise ValueError("旧 MT5 manifest 的 registration_method 不受支持")
        plan_sha = payload.get("registration_plan_sha256")
        if not isinstance(plan_sha, str) or _SHA256_RE.fullmatch(plan_sha) is None:
            raise ValueError("旧 MT5 manifest 的 registration_plan_sha256 非法")
        if payload.get("bar_timestamp_semantics") != "source_bar_open":
            raise ValueError("旧 MT5 manifest 必须声明 source_bar_open")
    if source_id == OKX_SOURCE_ID:
        source_id = resolve_okx_source_id(
            payload,
            symbol=symbol,
            timeframe=timeframe,
        )
    if (
        payload.get("periods_per_year", inferred_periods_per_year)
        != inferred_periods_per_year
    ):
        raise ValueError("MT5/OKX manifest 的 periods_per_year 与数据范围不匹配")
    if payload.get("minimum_bars", Config.MIN_BARS) != Config.MIN_BARS:
        raise ValueError(
            f"MT5/OKX manifest 的 minimum_bars 必须是 {Config.MIN_BARS}"
        )
    expected = {
        "symbol": symbol,
        "timeframe": timeframe,
        "data_filename": path.name,
        "data_sha256": digest,
        "data_rows": len(frame),
        "data_start": _utc_iso(int(frame["time"].iloc[0])),
        "data_end": _utc_iso(int(frame["time"].iloc[-1])),
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "columns": [str(column) for column in frame.columns],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"数据 manifest 的 {field} 与 Parquet 不一致")
    dataset_id = f"sha256:{digest}"
    if "dataset_id" in payload and payload.get("dataset_id") != dataset_id:
        raise ValueError("数据 manifest 的 dataset_id 与 data_sha256 不一致")
    return source_id, dataset_id


def _validate_generic_training_frame(df: pd.DataFrame) -> str:
    """保留非 A 股 volume 别名兼容，但拒绝排序、去重和坏 OHLCV。"""
    volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    required = ["time", "open", "high", "low", "close", volume_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Parquet 缺少列: {missing}")
    if df.empty:
        raise ValueError("Parquet 不能为空")
    if is_bool_dtype(df["time"].dtype) or not is_integer_dtype(df["time"].dtype):
        raise ValueError("time 必须是 unix_seconds 整数列")
    timestamps = df["time"].to_numpy(dtype=np.int64, copy=False)
    if np.any(timestamps <= 0):
        raise ValueError("time 必须是正的 unix_seconds")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("time 必须严格递增且不得重复；拒绝静默排序或去重")
    for column in ["open", "high", "low", "close", volume_col]:
        if is_bool_dtype(df[column].dtype) or not is_numeric_dtype(df[column].dtype):
            raise ValueError(f"{column} 必须是数值列")
    prices = df[["open", "high", "low", "close"]].to_numpy(dtype=np.float64, copy=False)
    volumes = df[volume_col].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(prices).all() or not np.isfinite(volumes).all():
        raise ValueError("OHLCV 含 NaN 或无穷值")
    if np.any(prices <= 0) or np.any(volumes < 0):
        raise ValueError("OHLC 必须大于 0，volume 不得为负")
    open_values, high_values, low_values, close_values = (prices[:, index] for index in range(4))
    if np.any(high_values < np.maximum(open_values, close_values)) or np.any(
        low_values > np.minimum(open_values, close_values)
    ):
        raise ValueError("OHLC 价格关系非法")
    return volume_col


def _resolve_training_contract(
    path: Path,
    frame: pd.DataFrame,
    timeframe: str,
    *,
    expected_source_id: str | None = None,
    expected_periods_per_year: int | None = None,
    expected_minimum_bars: int | None = None,
) -> tuple[str, int, int, str, str, str]:
    if expected_source_id is not None and expected_source_id not in DATA_SOURCE_IDS:
        raise ValueError("expected_source_id 不受支持")
    symbol, parsed_timeframe = parse_parquet_filename(path)
    if parsed_timeframe != timeframe:
        raise ValueError("内部 timeframe 解析不一致")
    payload = _read_manifest(path)
    digest = sha256_file(path)
    canonical_a_share_name = _CANONICAL_ASHARE_RE.fullmatch(path.name) is not None
    a_share_manifest = None
    a_share_source_id: str | None = None
    if payload is not None and payload.get("source") == ASHARE_SOURCE:
        a_share_manifest = load_a_share_manifest(path, frame)
        a_share_source_id = ASHARE_SOURCE_ID
    elif payload is not None and payload.get("source") == AKSHARE_SOURCE:
        a_share_manifest = load_akshare_hfq_manifest(path, frame)
        a_share_source_id = AKSHARE_HFQ_SOURCE_ID
    elif payload is not None and payload.get("source") == FREE_STOCKDB_SOURCE:
        try:
            a_share_manifest = load_free_stockdb_qfq_manifest(path, frame)
        except FreeStockDBDataError as exc:
            raise ValueError(str(exc)) from exc
        a_share_source_id = FREE_STOCKDB_QFQ_SOURCE_ID
    elif canonical_a_share_name and payload is not None:
        raise ValueError("六位 A 股规范文件必须使用受支持且有效的 A 股 manifest")
    elif (
        canonical_a_share_name
        and expected_source_id == FREE_STOCKDB_QFQ_SOURCE_ID
    ):
        raise ValueError("free-stockdb A 股数据必须携带来源 manifest")
    elif canonical_a_share_name and expected_source_id not in {
        ASHARE_SOURCE_ID,
        AKSHARE_HFQ_SOURCE_ID,
    }:
        raise ValueError("六位 A 股规范文件缺少有效的 A 股来源合同")

    if a_share_manifest is not None:
        volume_col = "tick_volume"
        periods_per_year = int(a_share_manifest["periods_per_year"])
        minimum_bars = int(a_share_manifest["minimum_bars"])
        if a_share_source_id is None:
            raise ValueError("A 股来源身份缺失")
        source_id = a_share_source_id
        data_sha256 = str(a_share_manifest["data_sha256"])
        dataset_id = str(a_share_manifest["dataset_id"])
    elif expected_source_id in {ASHARE_SOURCE_ID, AKSHARE_HFQ_SOURCE_ID}:
        if not canonical_a_share_name:
            raise ValueError(
                "manifestless A 股文件名必须是 6位代码_(M5|M15|H1|D1).parquet"
            )
        spec = ASHARE_SPECS_BY_TIMEFRAME.get(timeframe)
        if spec is None:
            raise ValueError("A 股训练数据只支持 M5/M15/H1/D1")
        if expected_source_id == AKSHARE_HFQ_SOURCE_ID and timeframe != "D1":
            raise ValueError(f"{AKSHARE_HFQ_SOURCE_ID} 只支持 D1")
        if expected_minimum_bars is None:
            raise ValueError("manifestless A 股数据必须显式传入 minimum_bars")
        validate_canonical_training_frame(frame)
        volume_col = "tick_volume"
        periods_per_year = spec.periods_per_year
        minimum_bars = spec.minimum_bars
        source_id = expected_source_id
        data_sha256 = digest
        dataset_id = f"sha256:{digest}"
    else:
        volume_col = _validate_generic_training_frame(frame)
        periods_per_year = infer_periods_per_year(
            rows=len(frame),
            start_unix=int(frame["time"].iloc[0]),
            end_unix=int(frame["time"].iloc[-1]),
        )
        minimum_bars = Config.MIN_BARS
        if payload is not None:
            source_id, dataset_id = _validate_generic_manifest(
                path=path,
                frame=frame,
                symbol=symbol,
                timeframe=timeframe,
                payload=payload,
                digest=digest,
                inferred_periods_per_year=periods_per_year,
            )
        else:
            source_id = expected_source_id or "local_file"
            dataset_id = f"sha256:{digest}"
        data_sha256 = digest

    if expected_source_id is not None and expected_source_id != source_id:
        raise ValueError(
            f"命令行 data_source 与数据 manifest 不一致: {expected_source_id} != {source_id}"
        )
    if (
        expected_periods_per_year is not None
        and expected_periods_per_year != periods_per_year
    ):
        raise ValueError(
            "命令行 periods_per_year 与数据 manifest 不一致: "
            f"{expected_periods_per_year} != {periods_per_year}"
        )
    if expected_minimum_bars is not None and expected_minimum_bars != minimum_bars:
        raise ValueError(
            "命令行 minimum_bars 与数据合同不一致: "
            f"{expected_minimum_bars} != {minimum_bars}"
        )
    return volume_col, periods_per_year, minimum_bars, source_id, data_sha256, dataset_id


def normalize_timeframe_token(token: str) -> str | None:
    """把文件名中的周期别名归一为下游使用的标准 token。"""
    raw = (token or "").strip()
    if not raw:
        return None
    key = raw.lower().replace("-", "").replace("_", "")
    if key in _TF_ALIASES:
        return _TF_ALIASES[key]
    upper = raw.upper()
    if upper in _TIMEFRAMES:
        return upper
    return None


def parse_parquet_filename(path: str | Path) -> tuple[str, str]:
    """Parse ``{symbol}_{timeframe}.parquet`` e.g. ``AAPL_H1.parquet``."""
    name = Path(path).name
    m = _PARQUET_RE.match(name)
    if not m:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet；当前: {name}"
        )
    timeframe = normalize_timeframe_token(m.group(2))
    if timeframe is None:
        raise ValueError(f"文件名周期不受支持: {m.group(2)}；当前: {name}")
    return m.group(1), timeframe


def inspect_parquet_file(
    path: str | Path,
    *,
    expected_source_id: str | None = None,
    expected_periods_per_year: int | None = None,
    expected_minimum_bars: int | None = None,
) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    symbol, timeframe = parse_parquet_filename(p)
    try:
        df = pd.read_parquet(p)
        (
            volume_col,
            periods_per_year,
            minimum_bars,
            source_id,
            data_sha256,
            dataset_id,
        ) = _resolve_training_contract(
            p,
            df,
            timeframe,
            expected_source_id=expected_source_id,
            expected_periods_per_year=expected_periods_per_year,
            expected_minimum_bars=expected_minimum_bars,
        )
    except AShareDataError as exc:
        raise ValueError(str(exc)) from exc
    bars = len(df)
    if bars < minimum_bars:
        raise ValueError(
            f"数据不足: {bars} bars（至少需要 {minimum_bars}）"
        )

    years = round(bars / periods_per_year, 2)
    manifest = _read_manifest(p)
    derivation = (
        manifest.get("derivation")
        if isinstance(manifest, dict)
        and isinstance(manifest.get("derivation"), dict)
        else None
    )
    dataset_purpose = (
        str(derivation.get("purpose"))
        if isinstance(derivation, dict)
        else None
    )
    sealed_evaluation = (
        source_id == AKSHARE_HFQ_SOURCE_ID
        and dataset_purpose == AKSHARE_SLICE_SEALED_EVALUATION
    )
    return {
        "data_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "data_start": _utc_iso(int(df["time"].iloc[0])),
        "data_end": _utc_iso(int(df["time"].iloc[-1])),
        "columns": [str(column) for column in df.columns],
        "volume_column": volume_col,
        "years_h1": years if timeframe == "H1" else None,
        "years": years,
        "periods_per_year": periods_per_year,
        "minimum_bars": minimum_bars,
        "source": source_id,
        "data_sha256": data_sha256,
        "dataset_id": dataset_id,
        "dataset_purpose": dataset_purpose,
        "manifest_path": str(p.with_suffix(".manifest.json").resolve()),
        "registration": "registered" if manifest is not None else "bare_legacy",
        "capabilities": {
            "local_training": True,
            "remote_training": (
                source_id != "local_file" and not sealed_evaluation
            ),
            "backtest": True,
            "legacy_registration": (
                source_id == "local_file" and volume_col == "tick_volume"
            ),
        },
        "valid": True,
        "message": "",
    }


class ParquetDataManager:
    """Single-symbol data manager backed by one Parquet file."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        expected_source_id: str | None = None,
        expected_periods_per_year: int | None = None,
        expected_minimum_bars: int | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(self.file_path)
        self.expected_source_id = expected_source_id
        self.expected_periods_per_year = expected_periods_per_year
        self.expected_minimum_bars = expected_minimum_bars
        self.periods_per_year = 0
        self.minimum_bars = Config.MIN_BARS
        self.source = "local_file"
        self.data_sha256 = ""
        self.dataset_id = ""
        self.data_rows = 0
        self.data_start = ""
        self.data_end = ""
        self.columns: list[str] = []
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None

    def load(self) -> None:
        df = pd.read_parquet(self.file_path)
        try:
            (
                volume_col,
                self.periods_per_year,
                self.minimum_bars,
                self.source,
                self.data_sha256,
                self.dataset_id,
            ) = _resolve_training_contract(
                self.file_path,
                df,
                self.timeframe,
                expected_source_id=self.expected_source_id,
                expected_periods_per_year=self.expected_periods_per_year,
                expected_minimum_bars=self.expected_minimum_bars,
            )
        except AShareDataError as exc:
            raise ValueError(str(exc)) from exc
        if len(df) < self.minimum_bars:
            raise ValueError(
                f"数据不足: {len(df)} bars（至少需要 {self.minimum_bars}）"
            )
        self.data_rows = int(len(df))
        self.data_start = _utc_iso(int(df["time"].iloc[0]))
        self.data_end = _utc_iso(int(df["time"].iloc[-1]))
        self.columns = [str(column) for column in df.columns]

        sub = df[["time", "open", "high", "low", "close", volume_col]].copy()
        sub = sub.rename(columns={volume_col: "volume"})

        rows = {field: sub[field].values for field in ["open", "high", "low", "close", "volume"]}
        import numpy as np

        raw: dict[str, torch.Tensor] = {
            field: torch.tensor(np.array([rows[field]]), dtype=torch.float32)
            for field in ["open", "high", "low", "close", "volume"]
        }
        raw["time"] = torch.tensor(
            np.array([sub["time"].values.astype("int64")]),
            dtype=torch.int64,
        )

        self._raw_dict = raw
        self._target_ret = MT5DataManager._compute_target_ret(raw["open"])
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，年化周期={self.periods_per_year}，"
            f"文件 {self.file_path.name}"
        )

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if self._raw_dict is None:
            raise RuntimeError("Call load() first")
        return self._raw_dict

    @property
    def feat_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        if self._target_ret is None:
            raise RuntimeError("Call load() first")
        return self._target_ret

    @property
    def bar_time(self) -> torch.Tensor:
        raw = self.raw_dict
        if "time" in raw:
            return raw["time"][:, -1].long()
        return torch.zeros(1, dtype=torch.int64)
