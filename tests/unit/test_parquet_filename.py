from __future__ import annotations

import pytest

from data_pipeline.parquet_manager import (
    normalize_timeframe_token,
    parse_parquet_filename,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("AAPL_H1.parquet", ("AAPL", "H1")),
        ("XAUUSD_H1.parquet", ("XAUUSD", "H1")),
        ("US30.cash_H1.parquet", ("US30.cash", "H1")),
        ("002008_60min.parquet", ("002008", "H1")),
        ("002008_60m.parquet", ("002008", "H1")),
        ("BTCUSDT_1h.parquet", ("BTCUSDT", "H1")),
        ("BTCUSDT_1m.parquet", ("BTCUSDT", "M1")),
        ("BTCUSDT_1mo.parquet", ("BTCUSDT", "MN1")),
        ("600519_5min.parquet", ("600519", "M5")),
        ("600519_15m.parquet", ("600519", "M15")),
        ("ETHUSDT_4h.parquet", ("ETHUSDT", "H4")),
        ("000001_1d.parquet", ("000001", "D1")),
        ("foo_D1.parquet", ("foo", "D1")),
        ("bar_m30.parquet", ("bar", "M30")),
    ],
)
def test_parse_parquet_filename_normalizes_timeframe(
    filename: str,
    expected: tuple[str, str],
) -> None:
    assert parse_parquet_filename(filename) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("60min", "H1"),
        ("H1", "H1"),
        ("1M", "M1"),
        ("1mo", "MN1"),
        ("MN1", "MN1"),
        ("nope", None),
    ],
)
def test_normalize_timeframe_token(token: str, expected: str | None) -> None:
    assert normalize_timeframe_token(token) == expected


def test_parse_rejects_unknown_tf() -> None:
    with pytest.raises(ValueError, match="周期"):
        parse_parquet_filename("002008_xyz.parquet")
