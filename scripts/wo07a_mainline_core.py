"""WO-AM-07A 主线构建专用的最小行情解码与 qfq 计算核心。

本模块只服务 26 只股票的已批准主线构建，不包含 99 点裁决、公司公告、
StockDB 或联网取证逻辑。独立审计器不得导入本模块。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
from typing import Any

import numpy as np
import pandas as pd


D1_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")


class MainlineCoreError(RuntimeError):
    pass


def _market_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _parse_js_assignment(raw: bytes, label: str) -> tuple[str, str]:
    text = raw.decode("utf-8-sig")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("/*", "//", "*"))
    ]
    if not lines or "=" not in lines[0]:
        raise MainlineCoreError(f"{label} 首个有效行不是 JavaScript 赋值")
    lhs, rhs = lines[0].split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if rhs.endswith(";"):
        rhs = rhs[:-1].strip()
    if not lhs or not rhs:
        raise MainlineCoreError(f"{label} JavaScript 赋值为空")
    return lhs, rhs


def _strict_decimal_json_object(blob: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复键 {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"非法 JSON 数值 {value}")

    try:
        value = json.loads(
            blob.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_float=Decimal,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MainlineCoreError(f"{label} 不是严格 UTF-8 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise MainlineCoreError(f"{label} JSON 根节点必须是对象")
    return value


def decode_history(raw: bytes, code: str) -> pd.DataFrame:
    try:
        from akshare.stock.cons import hk_js_decode
        from py_mini_racer import MiniRacer
    except ImportError as exc:
        raise MainlineCoreError("AKShare 的新浪安全解码依赖不可用") from exc

    lhs, rhs = _parse_js_assignment(raw, f"{code} 新浪历史")
    if lhs != f"var KLC_K2_{_market_symbol(code)}":
        raise MainlineCoreError(f"{code} 新浪历史变量名不符：{lhs!r}")
    try:
        encoded = json.loads(rhs)
    except json.JSONDecodeError as exc:
        raise MainlineCoreError(f"{code} 新浪历史编码字符串不是合法 JSON") from exc
    if not isinstance(encoded, str):
        raise MainlineCoreError(f"{code} 新浪历史赋值右侧不是字符串")

    context = MiniRacer()
    context.eval(hk_js_decode)
    records = context.call("d", encoded)
    if not isinstance(records, list) or not records:
        raise MainlineCoreError("新浪历史 JS 解码后没有记录")
    frame = pd.DataFrame(records)
    required = ("date", "open", "high", "low", "close", "volume")
    if not set(required).issubset(frame.columns):
        raise MainlineCoreError(
            f"新浪历史缺列：{sorted(set(required) - set(frame.columns))}"
        )
    frame = frame.loc[:, list(required)].copy()
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="raise", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise MainlineCoreError("新浪历史含重复交易日")
    values = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(values).all().all():
        raise MainlineCoreError("新浪历史含非有限数值")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise MainlineCoreError("新浪历史含非正价格")
    if (frame["volume"] < 0).any():
        raise MainlineCoreError("新浪历史含负成交量")
    return frame


def decode_factor(raw: bytes, code: str) -> pd.DataFrame:
    lhs, rhs = _parse_js_assignment(raw, f"{code} 新浪 qfq-factor")
    if lhs != f"var {_market_symbol(code)}qfq":
        raise MainlineCoreError(f"{code} 新浪 qfq-factor 变量名不符：{lhs!r}")
    payload = _strict_decimal_json_object(
        rhs.encode("utf-8"),
        f"{code} 新浪 qfq-factor",
    )
    records = payload.get("data")
    total = payload.get("total")
    if not isinstance(records, list) or not records:
        raise MainlineCoreError("新浪 qfq-factor 没有 data 列表")
    if total is not None and int(total) != len(records):
        raise MainlineCoreError(
            f"新浪 qfq-factor total={total!r} 与 data 行数 {len(records)} 不符"
        )

    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"d", "f"}:
            raise MainlineCoreError("新浪 qfq-factor 记录字段必须精确为 d/f")
        day_value = record.get("d")
        factor_value = record.get("f")
        if (
            day_value is None
            or factor_value is None
            or isinstance(factor_value, (bool, np.bool_))
        ):
            raise MainlineCoreError("新浪 qfq-factor 记录缺少 d/f")
        try:
            factor_decimal = Decimal(str(factor_value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise MainlineCoreError("新浪 qfq-factor 因子不是十进制数") from exc
        if not factor_decimal.is_finite() or factor_decimal <= 0:
            raise MainlineCoreError("新浪 qfq-factor 必须全部为有限正数")
        factor_float = float(factor_decimal)
        if not math.isfinite(factor_float) or factor_float <= 0:
            raise MainlineCoreError("新浪 qfq-factor 无法安全映射到 float64")
        normalized.append(
            {
                "date": pd.Timestamp(day_value).normalize(),
                "factor": factor_float,
                "factor_decimal": format(factor_decimal, "f"),
            }
        )
    frame = pd.DataFrame(normalized)
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise MainlineCoreError("新浪 qfq-factor 含重复日期")
    return frame


def build_qfq(
    history: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if tuple(factors.columns) != ("date", "factor", "factor_decimal"):
        raise MainlineCoreError("新浪 qfq-factor 精确/float64 双轨列合同不符")
    factor_series = factors.set_index("date")["factor"]
    factor_decimal_series = factors.set_index("date")["factor_decimal"]
    union_index = history["date"].drop_duplicates().sort_values()
    expanded_index = union_index.to_list() + factor_series.index.to_list()
    index = pd.DatetimeIndex(expanded_index).unique().sort_values()
    aligned = factor_series.reindex(index).ffill().reindex(
        pd.DatetimeIndex(union_index)
    )
    aligned_decimal = factor_decimal_series.reindex(index).ffill().reindex(
        pd.DatetimeIndex(union_index)
    )
    if aligned.isna().any() or aligned_decimal.isna().any():
        raise MainlineCoreError("最早交易日没有可前向填充的正 qfq 因子")

    work = history.copy()
    work["factor"] = aligned.to_numpy(dtype="float64")
    work["factor_decimal"] = aligned_decimal.to_numpy(dtype=object)
    for column in ("open", "high", "low", "close"):
        work[column] = work[column].astype("float64") / work["factor"]
    factor_changed = (
        work["factor_decimal"].ne(work["factor_decimal"].shift(1))
        & work.index.to_series().gt(0)
    )
    raw_return = history["close"].astype("float64").pct_change()
    qfq_return = work["close"].pct_change()
    difference = qfq_return.sub(raw_return).abs()
    unexplained = (
        difference.gt(0.10)
        & ~factor_changed
        & raw_return.notna()
        & qfq_return.notna()
    )
    large_switch = (
        difference.gt(0.10)
        & factor_changed
        & raw_return.notna()
        & qfq_return.notna()
    )
    return (
        work,
        tuple(work.loc[factor_changed, "date"].dt.strftime("%Y-%m-%d")),
        tuple(work.loc[large_switch, "date"].dt.strftime("%Y-%m-%d")),
        tuple(work.loc[unexplained, "date"].dt.strftime("%Y-%m-%d")),
    )


def ohlc_relation_violation_dates(frame: pd.DataFrame) -> tuple[str, ...]:
    bad = (
        frame["high"].lt(frame[["open", "low", "close"]].max(axis=1))
        | frame["low"].gt(frame[["open", "high", "close"]].min(axis=1))
    )
    return tuple(frame.loc[bad, "date"].dt.strftime("%Y-%m-%d"))


def make_d1(qfq: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(qfq["date"])
    localized = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    epoch_seconds = localized.as_unit("ns").asi8 // 1_000_000_000
    volume = qfq["volume"].to_numpy(dtype="float64")
    if (
        not np.isfinite(volume).all()
        or (volume < 0).any()
        or not np.equal(volume, np.floor(volume)).all()
    ):
        raise MainlineCoreError("新浪成交量无法无损转换为股数 int64")
    result = pd.DataFrame(
        {
            "time": epoch_seconds.astype("int64"),
            "open": qfq["open"].to_numpy(dtype="float32"),
            "high": qfq["high"].to_numpy(dtype="float32"),
            "low": qfq["low"].to_numpy(dtype="float32"),
            "close": qfq["close"].to_numpy(dtype="float32"),
            "tick_volume": volume.astype("int64"),
        }
    )
    if tuple(result.columns) != D1_COLUMNS:
        raise MainlineCoreError("D1 列合同失败")
    expected_dtypes = {
        "time": "int64",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "tick_volume": "int64",
    }
    if {
        column: str(result[column].dtype) for column in result.columns
    } != expected_dtypes:
        raise MainlineCoreError("D1 dtype 合同失败")
    if (
        not result["time"].is_monotonic_increasing
        or result["time"].duplicated().any()
    ):
        raise MainlineCoreError("D1 time 必须严格递增且唯一")
    prices = result[["open", "high", "low", "close"]]
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise MainlineCoreError("D1 OHLC 必须为有限正数")
    if (result["high"] < result[["open", "low", "close"]].max(axis=1)).any():
        raise MainlineCoreError("D1 high 合同失败")
    if (result["low"] > result[["open", "high", "close"]].min(axis=1)).any():
        raise MainlineCoreError("D1 low 合同失败")
    if (result["tick_volume"] < 0).any():
        raise MainlineCoreError("D1 tick_volume 不得为负")
    return result
