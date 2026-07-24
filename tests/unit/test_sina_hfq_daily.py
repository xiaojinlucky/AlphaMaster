from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from web.data_sources.sina_hfq_daily import (
    SOURCE_ID,
    SinaHfqDailyError,
    SinaHfqDailySource,
    SinaHfqTransportError,
    snapshot_to_raw_dict,
)

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _rows() -> list[dict]:
    return [
        {
            "date": "2026-07-22T00:00:00.000Z",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100,
        },
        {
            "date": "2026-07-23T00:00:00.000Z",
            "open": 10.5,
            "high": 12.0,
            "low": 10.0,
            "close": 11.0,
            "volume": 200,
        },
        {
            "date": "2026-07-24T00:00:00.000Z",
            "open": 11.0,
            "high": 12.0,
            "low": 10.5,
            "close": 11.5,
            "volume": 300,
        },
    ]


def _source(
    *,
    rows: list[dict] | None = None,
    factor: bytes | None = None,
    now: datetime | None = None,
) -> SinaHfqDailySource:
    responses = {
        "history": b"history-response",
        "factor": factor
        or (
            b'var sz000333hfq={"total":2,"data":['
            b'{"d":"2026-07-24","f":"3.0"},'
            b'{"d":"1900-01-01","f":"2.0"}]}\n'
        ),
    }

    def fetcher(url: str, _timeout: float) -> bytes:
        return responses["factor" if url.endswith("hfq.js") else "history"]

    return SinaHfqDailySource(
        fetcher=fetcher,
        history_decoder=lambda _text, _symbol: list(rows or _rows()),
        now=lambda: now or datetime(2026, 7, 24, 14, 0, tzinfo=_SHANGHAI),
    )


def test_same_source_hfq_is_applied_and_forming_day_is_dropped() -> None:
    snapshot = _source().fetch("000333", n=10, drop_forming=True)

    assert len(snapshot.bars) == 2
    assert snapshot.bars[0].open == pytest.approx(20.0)
    assert snapshot.bars[1].close == pytest.approx(22.0)
    assert snapshot.bars[1].volume == 200
    assert snapshot.bars[1].close_ts == int(
        datetime(2026, 7, 23, 15, 0, tzinfo=_SHANGHAI).timestamp()
    )
    assert snapshot.market_data_sha256 == snapshot.market_data_sha256
    assert len(snapshot.market_data_sha256) == 64


def test_completed_current_day_uses_new_factor() -> None:
    source = _source(now=datetime(2026, 7, 24, 15, 1, tzinfo=_SHANGHAI))

    snapshot = source.fetch("000333", n=1, drop_forming=True)

    assert len(snapshot.bars) == 1
    assert snapshot.bars[0].close == pytest.approx(34.5)
    assert snapshot.bars[0].volume == 300


def test_raw_dict_preserves_training_close_timestamp_semantics() -> None:
    snapshot = _source().fetch("000333", n=2)

    raw = snapshot_to_raw_dict(snapshot)

    assert raw["close"].shape == (1, 2)
    assert raw["close"][0, -1].item() == pytest.approx(22.0)
    assert raw["time"][0, -1].item() == pytest.approx(snapshot.last_bar_ts)


def test_malformed_factor_response_fails_closed() -> None:
    source = _source(factor=b"unexpected")

    with pytest.raises(SinaHfqDailyError, match="因子响应前缀"):
        source.fetch("000333")


def test_duplicate_history_date_fails_closed() -> None:
    rows = _rows()
    rows.append(dict(rows[-1]))
    source = _source(rows=rows)

    with pytest.raises(SinaHfqDailyError, match="严格递增"):
        source.fetch("000333")


def test_symbol_and_source_identity_are_explicit() -> None:
    snapshot = _source().fetch("000333")

    assert SOURCE_ID == "akshare_sina_hfq_ohlcv"
    assert snapshot.symbol == "000333"
    with pytest.raises(SinaHfqDailyError, match="6 位数字"):
        _source().fetch("sz000333")


def test_weekend_bar_is_rejected() -> None:
    rows = _rows()
    rows.append(
        {
            "date": "2026-07-25T00:00:00.000Z",
            "open": 11.0,
            "high": 12.0,
            "low": 10.5,
            "close": 11.5,
            "volume": 300,
        }
    )

    with pytest.raises(SinaHfqDailyError, match="周末"):
        _source(
            rows=rows,
            now=datetime(2026, 7, 27, 15, 1, tzinfo=_SHANGHAI),
        ).fetch("000333")


def test_transport_error_is_distinct_from_contract_error() -> None:
    def failed_fetch(_url: str, _timeout: float) -> bytes:
        raise TimeoutError("timeout")

    source = SinaHfqDailySource(
        fetcher=failed_fetch,
        history_decoder=lambda _text, _symbol: _rows(),
    )

    with pytest.raises(SinaHfqTransportError, match="请求失败"):
        source.fetch("000333")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout": "15"}, "timeout"),
        ({"timeout": True}, "timeout"),
    ],
)
def test_constructor_rejects_implicit_timeout_coercion(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SinaHfqDailySource(**kwargs)


def test_fetch_rejects_non_boolean_drop_forming() -> None:
    with pytest.raises(ValueError, match="drop_forming"):
        _source().fetch("000333", drop_forming="true")
