"""与大 A 训练同源的新浪后复权日线行情。

AKShare ``stock_zh_a_daily`` 会额外请求流通股本来计算换手率。AlphaMaster
实时信号只需要 OHLCV；流通股本端点失效不应阻断同源 K 线和复权因子。
本模块仍复用 AKShare 的新浪历史数据解码器，只跳过无关的流通股本请求。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timezone
from numbers import Real
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

SOURCE_ID = "akshare_sina_hfq_ohlcv"
TIMEFRAME = "D1"
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SESSION_CLOSE = time(15, 0)
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_HISTORY_COLUMNS = ("date", *_PRICE_COLUMNS, "volume")


class SinaHfqDailyError(RuntimeError):
    """新浪后复权日线不满足严格输入合同时抛出。"""


class SinaHfqTransportError(SinaHfqDailyError):
    """网络、超时或 HTTP 传输失败；短时间重试可能恢复。"""


@dataclass(frozen=True)
class DailyBar:
    """一根已完成的 A 股日线；时间戳使用上海 15:00 收盘时刻。"""

    close_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class SinaHfqSnapshot:
    """不可变的同源行情快照及其来源证据。"""

    symbol: str
    bars: tuple[DailyBar, ...]
    history_response_sha256: str
    factor_response_sha256: str

    @property
    def last_bar_ts(self) -> int:
        if not self.bars:
            raise SinaHfqDailyError("新浪后复权快照为空")
        return self.bars[-1].close_ts

    @property
    def last_close(self) -> float:
        if not self.bars:
            raise SinaHfqDailyError("新浪后复权快照为空")
        return self.bars[-1].close

    @property
    def market_data_sha256(self) -> str:
        payload = {
            "source": SOURCE_ID,
            "symbol": self.symbol,
            "timeframe": TIMEFRAME,
            "bars": [
                [
                    bar.close_ts,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                ]
                for bar in self.bars
            ],
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


ResponseFetcher = Callable[[str, float], bytes]
HistoryDecoder = Callable[[str, str], Sequence[dict]]


def _provider_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
        raise SinaHfqDailyError("A 股代码必须是 6 位数字")
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    raise SinaHfqDailyError(f"新浪 A 股日线不支持代码 {symbol}")


def _default_fetch(url: str, timeout: float) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return bytes(response.content)


def _decode_history_response(
    text: str,
    provider_symbol: str,
) -> Sequence[dict]:
    """严格拆出新浪压缩载荷，再复用 AKShare 自带解码器。"""
    marker = "="
    if marker not in text:
        raise SinaHfqDailyError("新浪历史 K 线响应缺少赋值标记")
    prefix, raw_value = text.split(marker, 1)
    if prefix.strip() != f"var KLC_K2_{provider_symbol}":
        raise SinaHfqDailyError("新浪历史 K 线响应前缀变化")
    raw_value = raw_value.strip()
    encoded_literal, separator, trailing = raw_value.partition(";")
    if not separator:
        raise SinaHfqDailyError("新浪历史 K 线响应缺少结束符")
    if trailing.strip() and not trailing.lstrip().startswith("/*"):
        raise SinaHfqDailyError("新浪历史 K 线响应含未知尾部")
    try:
        encoded = json.loads(encoded_literal)
    except json.JSONDecodeError as exc:
        raise SinaHfqDailyError("新浪历史 K 线压缩载荷不是合法字符串") from exc
    if not isinstance(encoded, str) or not encoded:
        raise SinaHfqDailyError("新浪历史 K 线压缩载荷为空")

    try:
        from akshare.stock import stock_zh_a_sina as sina

        javascript = sina.py_mini_racer.MiniRacer()
        javascript.eval(sina.hk_js_decode)
        rows = javascript.call("d", encoded)
    except Exception as exc:
        raise SinaHfqDailyError("AKShare 无法解码新浪历史 K 线") from exc
    if not isinstance(rows, list):
        raise SinaHfqDailyError("新浪历史 K 线解码结果不是列表")
    return rows


def _parse_factor_response(text: str, provider_symbol: str) -> pd.DataFrame:
    first_line = text.splitlines()[0] if text.splitlines() else ""
    expected_prefix = f"var {provider_symbol}hfq="
    if not first_line.startswith(expected_prefix):
        raise SinaHfqDailyError("新浪后复权因子响应前缀变化")
    try:
        payload = json.loads(first_line[len(expected_prefix) :])
    except json.JSONDecodeError as exc:
        raise SinaHfqDailyError("新浪后复权因子不是合法 JSON") from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise SinaHfqDailyError("新浪没有返回后复权因子")
    if payload.get("total") != len(rows):
        raise SinaHfqDailyError("新浪后复权因子数量合同不一致")

    factors = pd.DataFrame(rows)
    if tuple(str(column) for column in factors.columns) != ("d", "f"):
        raise SinaHfqDailyError("新浪后复权因子列合同变化")
    try:
        dates = pd.to_datetime(
            factors["d"],
            format="%Y-%m-%d",
            errors="raise",
        ).astype("datetime64[ns]")
        values = pd.to_numeric(factors["f"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SinaHfqDailyError("新浪后复权因子含非法日期或数值") from exc
    if dates.duplicated().any():
        raise SinaHfqDailyError("新浪后复权因子日期重复")
    numeric = values.to_numpy(dtype=float, copy=False)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in numeric):
        raise SinaHfqDailyError("新浪后复权因子必须为正的有限数")
    return (
        pd.DataFrame({"date": dates, "factor": numeric})
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def _parse_history_rows(rows: Sequence[dict]) -> pd.DataFrame:
    if not rows:
        raise SinaHfqDailyError("新浪没有返回历史 K 线")
    frame = pd.DataFrame(rows)
    missing = [column for column in _HISTORY_COLUMNS if column not in frame.columns]
    if missing:
        raise SinaHfqDailyError(f"新浪历史 K 线缺少字段: {missing}")
    try:
        parsed_dates = pd.to_datetime(frame["date"], errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise SinaHfqDailyError("新浪历史 K 线日期非法") from exc
    dates = pd.to_datetime(parsed_dates.dt.date).astype("datetime64[ns]")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise SinaHfqDailyError("新浪历史 K 线日期必须严格递增且不得重复")
    if any(date.weekday() >= 5 for date in dates.dt.date):
        raise SinaHfqDailyError("新浪历史 K 线包含周末日期")

    numeric: dict[str, pd.Series] = {}
    try:
        for column in (*_PRICE_COLUMNS, "volume"):
            numeric[column] = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SinaHfqDailyError("新浪历史 K 线 OHLCV 含非法数值") from exc
    values = pd.DataFrame(numeric)
    if values.isna().any().any():
        raise SinaHfqDailyError("新浪历史 K 线 OHLCV 含空值")
    for column in _PRICE_COLUMNS:
        column_values = values[column].to_numpy(dtype=float, copy=False)
        if not all(
            math.isfinite(float(value)) and float(value) > 0 for value in column_values
        ):
            raise SinaHfqDailyError("新浪历史 K 线价格必须为正的有限数")
    volumes = values["volume"].to_numpy(dtype=float, copy=False)
    if not all(
        math.isfinite(float(value)) and float(value) >= 0 and float(value).is_integer()
        for value in volumes
    ):
        raise SinaHfqDailyError("新浪历史 K 线成交量必须为非负整数")

    output = values.copy()
    output.insert(0, "date", dates)
    return output


def _apply_hfq(history: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    adjusted = pd.merge_asof(
        history.sort_values("date", kind="stable"),
        factors.sort_values("date", kind="stable"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    if adjusted["factor"].isna().any():
        raise SinaHfqDailyError("新浪后复权因子没有覆盖全部历史 K 线")
    for column in _PRICE_COLUMNS:
        adjusted[column] = (adjusted[column] * adjusted["factor"]).round(2)

    prices = adjusted[list(_PRICE_COLUMNS)].to_numpy(dtype=float, copy=False)
    if any(
        high < max(open_price, close) or low > min(open_price, close)
        for open_price, high, low, close in prices
    ):
        raise SinaHfqDailyError("新浪后复权 OHLC 价格关系非法")
    return adjusted


def _completed_rows(
    frame: pd.DataFrame,
    *,
    now: datetime,
    drop_forming: bool,
) -> pd.DataFrame:
    current = now
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise SinaHfqDailyError("now 必须带时区")
    current = current.astimezone(_SHANGHAI)
    dates = frame["date"].dt.date
    if bool((dates > current.date()).any()):
        raise SinaHfqDailyError("新浪返回了未来交易日")
    if (
        drop_forming
        and current.time() < _SESSION_CLOSE
        and not frame.empty
        and dates.iloc[-1] == current.date()
    ):
        return frame.iloc[:-1].copy()
    return frame


class SinaHfqDailySource:
    """只读取新浪历史 K 线与后复权因子，不请求换手率等无关字段。"""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        fetcher: ResponseFetcher | None = None,
        history_decoder: HistoryDecoder | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, Real):
            raise ValueError("timeout 必须为正的有限数")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("timeout 必须为正的有限数")
        self.timeout = float(timeout)
        self.fetcher = fetcher or _default_fetch
        self.history_decoder = history_decoder or _decode_history_response
        self.now = now or (lambda: datetime.now(timezone.utc))

    def fetch(
        self,
        symbol: str,
        *,
        n: int = 500,
        drop_forming: bool = True,
    ) -> SinaHfqSnapshot:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n 必须为正整数")
        if not isinstance(drop_forming, bool):
            raise ValueError("drop_forming 必须是布尔值")
        count = n
        provider_symbol = _provider_symbol(symbol)
        try:
            from akshare.stock import stock_zh_a_sina as sina
        except ImportError as exc:
            raise SinaHfqDailyError("缺少 AKShare，无法读取新浪日线") from exc

        history_url = sina.zh_sina_a_stock_hist_url.format(provider_symbol)
        factor_url = sina.zh_sina_a_stock_hfq_url.format(provider_symbol)
        try:
            history_bytes = self.fetcher(history_url, self.timeout)
            factor_bytes = self.fetcher(factor_url, self.timeout)
        except Exception as exc:
            raise SinaHfqTransportError(f"新浪后复权日线请求失败: {exc}") from exc
        if not isinstance(history_bytes, bytes) or not isinstance(
            factor_bytes,
            bytes,
        ):
            raise SinaHfqDailyError("新浪响应必须是原始字节")
        if not history_bytes or len(history_bytes) > 20 * 1024 * 1024:
            raise SinaHfqDailyError("新浪历史 K 线响应为空或超过 20 MiB")
        if not factor_bytes or len(factor_bytes) > 2 * 1024 * 1024:
            raise SinaHfqDailyError("新浪后复权因子响应为空或超过 2 MiB")
        try:
            history_text = bytes(history_bytes).decode("utf-8")
            factor_text = bytes(factor_bytes).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SinaHfqDailyError("新浪后复权响应不是 UTF-8") from exc

        history = _parse_history_rows(
            self.history_decoder(history_text, provider_symbol)
        )
        factors = _parse_factor_response(factor_text, provider_symbol)
        adjusted = _completed_rows(
            _apply_hfq(history, factors),
            now=self.now(),
            drop_forming=drop_forming,
        )
        if adjusted.empty:
            raise SinaHfqDailyError("新浪没有已完成的后复权日线")
        adjusted = adjusted.iloc[-count:]

        bars: list[DailyBar] = []
        for row in adjusted.itertuples(index=False):
            close_at = datetime.combine(
                row.date.date(),
                _SESSION_CLOSE,
                tzinfo=_SHANGHAI,
            )
            bars.append(
                DailyBar(
                    close_ts=int(close_at.timestamp()),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=int(row.volume),
                )
            )
        return SinaHfqSnapshot(
            symbol=str(symbol),
            bars=tuple(bars),
            history_response_sha256=hashlib.sha256(history_bytes).hexdigest(),
            factor_response_sha256=hashlib.sha256(factor_bytes).hexdigest(),
        )


def snapshot_to_raw_dict(snapshot: SinaHfqSnapshot) -> dict[str, Any]:
    """按训练数据的收盘时间语义转换为因子引擎输入。"""
    import torch

    if not snapshot.bars:
        raise SinaHfqDailyError("新浪后复权快照为空")

    def tensor(values: list[float]):
        return torch.tensor([values], dtype=torch.float32)

    return {
        "open": tensor([bar.open for bar in snapshot.bars]),
        "high": tensor([bar.high for bar in snapshot.bars]),
        "low": tensor([bar.low for bar in snapshot.bars]),
        "close": tensor([bar.close for bar in snapshot.bars]),
        "volume": tensor([float(bar.volume) for bar in snapshot.bars]),
        "time": tensor([float(bar.close_ts) for bar in snapshot.bars]),
    }


__all__ = [
    "SOURCE_ID",
    "DailyBar",
    "SinaHfqDailyError",
    "SinaHfqTransportError",
    "SinaHfqDailySource",
    "SinaHfqSnapshot",
    "snapshot_to_raw_dict",
]
