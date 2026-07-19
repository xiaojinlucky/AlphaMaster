"""通达信数据源（pytdx，免费行情服务器，A 股 / 指数）。"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 通达信行情服务器（多个备选）
_SERVERS = [
    ("115.238.90.165", 7709),
    ("180.153.18.170", 7709),
    ("119.147.212.81", 7709),
    ("14.17.75.71", 7709),
    ("59.173.18.77", 7709),
]

# 项目周期 -> pytdx category
_CAT = {
    "1m": 8,
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "1h": 3,
    "1d": 9,
    "1w": 5,
    "1M": 6,
}

_PRESETS = ["600519", "000001", "300750", "601318", "000858", "sh000001", "sz399006"]
_CST = timezone(timedelta(hours=8))


def _parse_market(code: str) -> tuple[int, str]:
    """返回 (market, pure_code)。1=上海, 0=深圳。"""
    c = code.strip().upper()
    if c.startswith("SH"):
        return 1, c[2:]
    if c.startswith("SZ"):
        return 0, c[2:]
    if c[:1] in ("6", "5", "9") or c.startswith("11") or c.startswith("13"):
        return 1, c
    return 0, c


def _is_index(market: int, code: str) -> bool:
    return (market == 1 and code.startswith("000")) or (
        market == 0 and code.startswith("399")
    )


def _parse_dt(s: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(str(s), fmt).replace(tzinfo=_CST).timestamp())
        except ValueError:
            continue
    return 0


def _looks_corrupted(raw) -> bool:
    """日期年份异常占比至少三成时，拒绝把乱码当作行情。"""
    if not raw:
        return False

    corrupted = 0
    total = 0
    for row in raw:
        total += 1
        value = row.get("datetime", "") if isinstance(row, dict) else ""
        try:
            year = int(str(value)[:4])
        except (TypeError, ValueError):
            corrupted += 1
            continue
        if not 1990 <= year <= 2035:
            corrupted += 1
    return total > 0 and corrupted / total >= 0.3


class TongdaxinSource(DataSource):
    kind = "tongdaxin"
    label = "通达信"

    def __init__(self) -> None:
        self._api = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import pytdx  # noqa: F401
        except ImportError:
            return (False, "未安装 pytdx：pip install pytdx")
        return (True, "免费行情服务器 · A 股 / 指数")

    def supported_timeframes(self) -> list[str]:
        return list(_CAT.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        if self._api is not None:
            return
        try:
            from pytdx.hq import TdxHq_API
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 pytdx") from exc
        api = TdxHq_API()
        for host, port in _SERVERS:
            try:
                if api.connect(host, port):
                    self._api = api
                    return
            except Exception:
                continue
        raise DataSourceUnavailable("通达信所有行情服务器连接失败")

    def disconnect(self) -> None:
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
        self._api = None

    def _fetch_raw(
        self,
        cat: int,
        market: int,
        code: str,
        want: int,
        is_index: bool,
    ):
        if is_index:
            return self._api.get_index_bars(cat, market, code, 0, want)
        return self._api.get_security_bars(cat, market, code, 0, want)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _CAT:
            raise DataSourceUnavailable(f"通达信不支持周期 {timeframe}")
        market, code = _parse_market(symbol)
        is_index = _is_index(market, code)
        cat = _CAT[timeframe]
        want = min(max(n + 2, 20), 800)  # 单次上限 800

        with self._lock:
            self.connect()
            try:
                raw = self._fetch_raw(cat, market, code, want, is_index)
            except Exception as exc:
                # 连接可能失效，重连一次
                self._api = None
                self.connect()
                raw = self._fetch_raw(cat, market, code, want, is_index)

        if not raw:
            raise DataSourceUnavailable(
                f"通达信无数据：{symbol}；指数需带 sh/sz 前缀（如 sh000001）"
            )
        if _looks_corrupted(raw):
            raise DataSourceUnavailable(
                "通达信返回异常日期；指数请使用 sh/sz 前缀（如 sh000001），"
                "股票请确认代码"
            )

        bars: list[Bar] = []
        for r in raw:
            bars.append(
                Bar(
                    ts=_parse_dt(r.get("datetime", "")),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    # 单位：手（1手=100股），与 OKX/MT5 量纲不同，跨源不可比。
                    volume=float(r.get("vol", 0.0) or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts)  # 保证升序
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()
        return bars[-n:]
