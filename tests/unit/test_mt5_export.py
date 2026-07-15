"""MT5 Parquet 导出合同测试，不连接真实终端。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import export_mt5_parquet as exporter


class FakeMT5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769
    TIMEFRAME_MN1 = 49153

    def __init__(self, rates: list[dict], *, exact_name: str = "XAUUSD") -> None:
        self.rates = rates
        self.exact_name = exact_name
        self.copy_call: tuple[str, int, int, int] | None = None
        self.shutdown_called = False

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def account_info(self):
        return object()

    def symbol_info(self, _symbol: str):
        return SimpleNamespace(name=self.exact_name, visible=True)

    def symbol_select(self, _symbol: str, _selected: bool) -> bool:
        return True

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        self.copy_call = (symbol, timeframe, start_pos, count)
        return self.rates

    def last_error(self):
        return (0, "ok")


def _rates() -> list[dict]:
    return [
        {"time": 300, "open": 3, "high": 4, "low": 2, "close": 3.5, "tick_volume": 30},
        {"time": 100, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "tick_volume": 10},
        {"time": 200, "open": 2, "high": 3, "low": 1, "close": 2.2, "tick_volume": 20},
        {"time": 200, "open": 2, "high": 3, "low": 1, "close": 2.5, "tick_volume": 21},
    ]


def test_export_uses_only_closed_bars_and_writes_verified_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mt5 = FakeMT5(_rates())
    captured: dict[str, pd.DataFrame] = {}

    def fake_to_parquet(self: pd.DataFrame, path: Path, *, index: bool) -> None:
        assert index is False
        captured["frame"] = self.copy()
        Path(path).write_bytes(b"PAR1-test-payload")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(
        exporter,
        "_utc_now",
        lambda: datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc),
    )

    parquet_path, manifest_path, manifest = exporter.export_mt5_parquet(
        symbol="XAUUSD",
        timeframe="H1",
        bars=4,
        output_dir=tmp_path,
        mt5_module=fake_mt5,
    )

    assert fake_mt5.copy_call == ("XAUUSD", fake_mt5.TIMEFRAME_H1, 1, 4)
    assert fake_mt5.shutdown_called is True
    frame = captured["frame"]
    assert frame["time"].tolist() == [100, 200, 300]
    assert str(frame["time"].dtype) == "int64"
    assert frame.loc[frame["time"] == 200, "close"].item() == 2.5

    expected_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["symbol"] == "XAUUSD"
    assert manifest["timeframe"] == "H1"
    assert manifest["data_rows"] == 3
    assert manifest["data_timezone"] == "UTC"
    assert manifest["time_unit"] == "unix_seconds"
    assert manifest["data_start"] == "1970-01-01T00:01:40Z"
    assert manifest["data_end"] == "1970-01-01T00:05:00Z"
    assert manifest["exported_at"] == "2026-07-14T08:00:00Z"
    assert manifest["data_sha256"] == expected_hash
    assert not list(tmp_path.glob("*.partial"))


def test_export_rejects_non_exact_symbol_name(tmp_path: Path) -> None:
    fake_mt5 = FakeMT5(_rates(), exact_name="XAUUSDm")
    with pytest.raises(exporter.ExportError, match="精确品种名"):
        exporter.export_mt5_parquet(
            symbol="XAUUSD",
            timeframe="H1",
            bars=4,
            output_dir=tmp_path,
            mt5_module=fake_mt5,
        )
    assert fake_mt5.shutdown_called is True


def test_export_rejects_timeframe_alias_without_conversion(tmp_path: Path) -> None:
    with pytest.raises(exporter.ExportError, match="不支持的 timeframe"):
        exporter.export_mt5_parquet(
            symbol="XAUUSD",
            timeframe="1h",
            bars=4,
            output_dir=tmp_path,
            mt5_module=FakeMT5(_rates()),
        )


def test_failed_parquet_write_does_not_replace_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "XAUUSD_H1.parquet"
    manifest_path = tmp_path / "XAUUSD_H1.manifest.json"
    parquet_path.write_bytes(b"old-parquet")
    manifest_path.write_text("old-manifest", encoding="utf-8")

    def fail_to_parquet(self: pd.DataFrame, path: Path, *, index: bool) -> None:
        Path(path).write_bytes(b"partial-new-data")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(OSError, match="disk full"):
        exporter.export_mt5_parquet(
            symbol="XAUUSD",
            timeframe="H1",
            bars=4,
            output_dir=tmp_path,
            mt5_module=FakeMT5(_rates()),
        )

    assert parquet_path.read_bytes() == b"old-parquet"
    assert manifest_path.read_text(encoding="utf-8") == "old-manifest"
    assert not list(tmp_path.glob("*.partial"))
