from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.a_share_data import sha256_file
from data_pipeline.dataset_contracts import FREE_STOCKDB_QFQ_SOURCE_ID
from data_pipeline.free_stockdb_data import (
    FreeStockDBDataError,
    build_free_stockdb_qfq_manifest,
)
from data_pipeline.parquet_manager import inspect_parquet_file
from web.slurm_training_manager import SlurmTrainingManager


def _write_daily(path: Path, *, rows: int = 500, hour: int = 15) -> None:
    dates = pd.bdate_range("2022-01-04", periods=rows, tz="Asia/Shanghai")
    closes = dates + pd.Timedelta(hours=hour)
    prices = 100.0 + np.arange(rows, dtype=np.float32) * 0.01
    frame = pd.DataFrame(
        {
            "time": (
                closes.tz_convert("UTC")
                .astype("datetime64[ns, UTC]")
                .astype("int64")
                // 1_000_000_000
            ).astype(np.int64),
            "open": prices,
            "high": prices + np.float32(0.2),
            "low": prices - np.float32(0.2),
            "close": prices + np.float32(0.1),
            "tick_volume": np.arange(rows, dtype=np.int64) + 10_000,
        }
    )
    frame.to_parquet(path, index=False)


def _publish_manifest(
    path: Path,
    *,
    source_as_of: str = "2026-07-24",
) -> dict:
    source_snapshot = path.parent / ".sync_manifest.json"
    source_snapshot.write_text(
        json.dumps({"generated_at": 1_784_957_450}),
        encoding="utf-8",
    )
    extraction_script = path.parent / "build_dataset.py"
    extraction_script.write_text("print('extract')\n", encoding="utf-8")
    manifest = build_free_stockdb_qfq_manifest(
        path,
        source_as_of=source_as_of,
        provider_release="v0.2.1-more-power",
        source_snapshot_manifest=source_snapshot,
        extraction_script=extraction_script,
        qfq_factor_points=3,
    )
    assert manifest["source_snapshot_sha256"] == sha256_file(source_snapshot)
    assert manifest["extraction_script_sha256"] == sha256_file(extraction_script)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def test_free_stockdb_daily_contract_loads_locally_and_for_slurm(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_daily(data_file)
    manifest = _publish_manifest(data_file)

    info = inspect_parquet_file(data_file)
    assert info["source"] == FREE_STOCKDB_QFQ_SOURCE_ID
    assert info["periods_per_year"] == 242
    assert info["minimum_bars"] == 484
    assert info["data_sha256"] == manifest["data_sha256"]
    assert manifest["tick_volume_unit"] == "shares"
    assert manifest["tick_volume_adjustment"] == "unadjusted_source_volume"
    assert datetime.fromisoformat(
        info["data_start"].replace("Z", "+00:00")
    ).astimezone(timezone.utc).hour == 7

    manager = object.__new__(SlurmTrainingManager)
    loaded = manager._load_data_manifest(data_file)
    assert loaded["_source_id"] == FREE_STOCKDB_QFQ_SOURCE_ID


def test_free_stockdb_midnight_daily_marker_is_rejected(tmp_path: Path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_daily(data_file, hour=0)
    with pytest.raises(FreeStockDBDataError, match="15:00"):
        _publish_manifest(data_file)


def test_free_stockdb_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_daily(data_file)
    manifest = _publish_manifest(data_file)
    manifest["source_snapshot_sha256"] = "invalid"
    data_file.with_suffix(".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_snapshot_sha256"):
        inspect_parquet_file(data_file)


def test_free_stockdb_short_daily_history_is_not_trainable(tmp_path: Path) -> None:
    data_file = tmp_path / "001280_D1.parquet"
    _write_daily(data_file, rows=155)
    with pytest.raises(FreeStockDBDataError, match="数据不足"):
        _publish_manifest(data_file)


def test_free_stockdb_source_as_of_must_be_real_date(tmp_path: Path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_daily(data_file)
    with pytest.raises(FreeStockDBDataError, match="真实"):
        _publish_manifest(data_file, source_as_of="2026-99-99")


def test_free_stockdb_rejects_bars_after_source_as_of(tmp_path: Path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_daily(data_file)
    with pytest.raises(FreeStockDBDataError, match="晚于 source_as_of"):
        _publish_manifest(data_file, source_as_of="2022-01-04")
