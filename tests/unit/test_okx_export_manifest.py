from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import download_okx_klines as okx
from data_pipeline.parquet_manager import inspect_parquet_file


def _frame(rows: int = 3000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1_700_000_000 + index * 300 for index in range(rows)],
            "open": [100.0 + index / 1000 for index in range(rows)],
            "high": [101.0 + index / 1000 for index in range(rows)],
            "low": [99.0 + index / 1000 for index in range(rows)],
            "close": [100.5 + index / 1000 for index in range(rows)],
            "tick_volume": [100 + index for index in range(rows)],
        }
    )


def _candle(ts_ms: int, confirm: str) -> list[str]:
    return [
        str(ts_ms),
        "100",
        "101",
        "99",
        "100.5",
        "123",
        "0",
        "0",
        confirm,
    ]


def _utc_iso(value: int) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_legacy_okx_pair(path: Path) -> dict:
    frame = _frame()
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "format": "alphamaster_okx_dataset_v1",
        "source": "OKX",
        "provenance_status": "legacy_archive_attestation",
        "source_instrument": "BTC-USDT-SWAP",
        "source_endpoint": "/api/v5/market/history-candles",
        "source_bar": "5m",
        "volume_semantics": "OKX contract volume mapped to tick_volume",
        "provenance": "user_provided_archive:OKX_K线数据.zip",
        "symbol": "BTCUSDT",
        "timeframe": "M5",
        "data_filename": path.name,
        "data_sha256": digest,
        "dataset_id": f"sha256:{digest}",
        "data_rows": len(frame),
        "data_start": _utc_iso(int(frame["time"].iloc[0])),
        "data_end": _utc_iso(int(frame["time"].iloc[-1])),
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "closed_bars_only": True,
        "derived_from": {
            "archive_member": f"OKX_K线数据/{path.name}",
            "data_sha256": "a" * 64,
        },
        "transform": {
            "dropped_trailing_unclosed_bars": 1,
            "cutoff_reference": "source_file_mtime",
        },
        "columns": list(frame.columns),
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def test_download_history_keeps_only_confirmed_candles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        okx,
        "okx_get",
        lambda _path, _params: [
            _candle(1_700_000_300_000, "0"),
            _candle(1_700_000_000_000, "1"),
        ],
    )

    frame = okx.download_history("BTC-USDT-SWAP", "5m")

    assert frame["time"].tolist() == [1_700_000_000]


def test_download_history_rejects_response_with_only_uncompleted_candles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        okx,
        "okx_get",
        lambda _path, _params: [_candle(1_700_000_300_000, "0")],
    )

    with pytest.raises(RuntimeError, match="没有已完成 K 线"):
        okx.download_history("BTC-USDT-SWAP", "5m")


def test_download_history_rejects_malformed_candle_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        okx,
        "okx_get",
        lambda _path, _params: [_candle(1_700_000_000_000, "1")[:-1]],
    )

    with pytest.raises(RuntimeError, match="严格为 9 个字段"):
        okx.download_history("BTC-USDT-SWAP", "5m")


def test_okx_download_writes_remote_training_sidecar(tmp_path: Path) -> None:
    data_path = tmp_path / "BTCUSDT_M5.parquet"

    manifest = okx.save_parquet(_frame(), data_path)

    manifest_path = data_path.with_suffix(".manifest.json")
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert manifest == persisted
    assert manifest["source"] == "OKX"
    assert manifest["data_sha256"] == digest
    assert manifest["dataset_id"] == f"sha256:{digest}"
    assert manifest["bar_completion"] == "confirmed_only"
    assert manifest["periods_per_year"] > 100_000
    info = inspect_parquet_file(data_path)
    assert info["source"] == "okx"
    assert info["capabilities"]["remote_training"] is True
    assert not list(tmp_path.glob("*.partial"))


def test_legacy_okx_archive_keeps_distinct_attested_identity(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "BTCUSDT_M5.parquet"
    manifest = _write_legacy_okx_pair(data_path)

    info = inspect_parquet_file(
        data_path,
        expected_source_id="okx_legacy_attested",
    )

    assert info["source"] == "okx_legacy_attested"
    assert info["dataset_id"] == manifest["dataset_id"]
    assert info["capabilities"]["remote_training"] is True


def test_legacy_okx_archive_rejects_missing_closed_bar_attestation(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "BTCUSDT_M5.parquet"
    manifest = _write_legacy_okx_pair(data_path)
    manifest["closed_bars_only"] = False
    data_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="closed_bars_only"):
        inspect_parquet_file(data_path)


def test_resume_requires_valid_sidecar_not_only_parquet(tmp_path: Path) -> None:
    data_path = tmp_path / "BTCUSDT_M5.parquet"
    _frame().to_parquet(data_path, index=False)
    assert okx._verified_okx_pair_exists(data_path) is False

    okx.save_parquet(_frame(), data_path)
    assert okx._verified_okx_pair_exists(data_path) is True

    data_path.write_bytes(data_path.read_bytes() + b"tampered")
    assert okx._verified_okx_pair_exists(data_path) is False
