from __future__ import annotations

import pytest

from web.data_sources.base import DataSourceUnavailable
from web.data_sources.tongdaxin_source import (
    TongdaxinSource,
    _parse_dt,
)


class FakeTdxApi:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[str] = []

    def get_security_bars(self, *_args):
        self.calls.append("security")
        return list(self.rows)

    def get_index_bars(self, *_args):
        self.calls.append("index")
        return list(self.rows)


def _row(moment: str, close: float = 10.0) -> dict:
    return {
        "datetime": moment,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "vol": 100,
    }


def _source(rows: list[dict]) -> tuple[TongdaxinSource, FakeTdxApi]:
    source = TongdaxinSource()
    api = FakeTdxApi(rows)
    source._api = api
    return source, api


def test_intraday_close_time_is_normalized_to_bar_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, api = _source(
        [
            _row("2026-07-23 09:45", 10.0),
            _row("2026-07-23 10:00", 10.1),
        ]
    )
    now = _parse_dt("2026-07-23 09:50")
    monkeypatch.setattr("web.data_sources.tongdaxin_source.time.time", lambda: now)

    bars = source.fetch_bars("600519", "15m", 10, drop_forming=True)

    assert api.calls == ["security"]
    assert len(bars) == 1
    assert bars[0].ts == _parse_dt("2026-07-23 09:30")


def test_index_uses_index_endpoint() -> None:
    source, api = _source([_row("2026-07-23 15:00")])

    bars = source.fetch_bars("sh000001", "15m", 10, drop_forming=False)

    assert api.calls == ["index"]
    assert len(bars) == 1


def test_weekly_and_monthly_are_not_exposed_without_closed_bar_calendar() -> None:
    source, _ = _source([_row("2026-07-23 15:00")])

    assert "1w" not in source.supported_timeframes()
    assert "1M" not in source.supported_timeframes()
    with pytest.raises(DataSourceUnavailable, match="不支持周期"):
        source.fetch_bars("600519", "1w", 10)


@pytest.mark.parametrize(
    "rows",
    [
        [_row("bad-time")],
        [_row("2026-07-23 15:00"), _row("2026-07-23 15:00")],
    ],
)
def test_invalid_or_duplicate_bar_time_fails_closed(rows: list[dict]) -> None:
    source, _ = _source(rows)

    with pytest.raises(DataSourceUnavailable, match="异常日期|K 线时间"):
        source.fetch_bars("600519", "15m", 10, drop_forming=False)
