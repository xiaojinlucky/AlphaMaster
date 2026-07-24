from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import Config
from data_pipeline.a_share_data import (
    ASHARE_PERIOD_SPECS,
    AShareDataError,
    convert_legacy_a_share_file,
    sha256_file,
)
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file


def _legacy_h1(
    path: Path,
    *,
    days: int = 484,
    start: datetime = datetime(2024, 1, 2),
) -> Path:
    spec = ASHARE_PERIOD_SPECS["60min"]
    rows: list[dict[str, int | float]] = []
    trading_days = []
    candidate = start.date()
    while len(trading_days) < days:
        if candidate.weekday() < 5:
            trading_days.append(candidate)
        candidate += timedelta(days=1)
    for day_index, trading_day in enumerate(trading_days):
        for bar_index, close_time in enumerate(spec.close_times):
            wall_clock = datetime.combine(trading_day, close_time, tzinfo=timezone.utc)
            base = 10.0 + day_index * 0.001 + bar_index * 0.01
            rows.append(
                {
                    "time": int(wall_clock.timestamp()) // 1000,
                    "open": base,
                    "high": base + 0.2,
                    "low": base - 0.2,
                    "close": base + 0.05,
                    "tick_volume": 1000 + day_index + bar_index,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).astype(
        {
            "time": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "tick_volume": "int64",
        }
    ).to_parquet(path, index=False)
    return path


def _rewrite_column(path: Path, column: str, values: pd.Series) -> None:
    frame = pd.read_parquet(path)
    frame[column] = values
    frame.to_parquet(path, index=False)


def _generic_h1(path: Path) -> Path:
    rows = Config.MIN_BARS
    timestamps = np.arange(1_700_000_000, 1_700_000_000 + rows * 3600, 3600, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "open": np.full(rows, 10.0, dtype=np.float32),
            "high": np.full(rows, 11.0, dtype=np.float32),
            "low": np.full(rows, 9.0, dtype=np.float32),
            "close": np.full(rows, 10.5, dtype=np.float32),
            "tick_volume": np.full(rows, 100, dtype=np.int64),
        }
    )
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture()
def converted_h1(tmp_path: Path) -> tuple[Path, dict]:
    source = _legacy_h1(tmp_path / "raw" / "stocks" / "000001_60min.parquet")
    result = convert_legacy_a_share_file(source, tmp_path / "converted")
    return Path(result["data_file"]), result


def test_strict_h1_conversion_reconstructs_utc_and_uses_market_threshold(
    converted_h1: tuple[Path, dict],
) -> None:
    output, result = converted_h1
    frame = pd.read_parquet(output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert len(frame) == 1936 < Config.MIN_BARS
    assert frame["time"].is_unique
    assert frame["time"].is_monotonic_increasing
    assert datetime.fromtimestamp(int(frame["time"].iloc[0]), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    ) == "2024-01-02 02:30"
    assert manifest["source"] == "AShareLocal"
    assert manifest["timeframe"] == "H1"
    assert manifest["periods_per_year"] == 968
    assert manifest["minimum_bars"] == 1936
    assert manifest["data_rows"] == len(frame)
    assert manifest["source_filename"] == "000001_60min.parquet"
    assert result["dataset_id"] == manifest["dataset_id"]

    info = inspect_parquet_file(output)
    assert info["minimum_bars"] == 1936
    assert info["periods_per_year"] == 968
    manager = ParquetDataManager(output, expected_periods_per_year=968)
    manager.load()
    assert manager.raw_dict["open"].shape == (1, 1936)
    assert manager.periods_per_year == 968


def test_historical_shanghai_dst_is_applied(tmp_path: Path) -> None:
    source = _legacy_h1(
        tmp_path / "raw" / "stocks" / "000001_60min.parquet",
        start=datetime(1991, 6, 3),
    )
    result = convert_legacy_a_share_file(source, tmp_path / "converted")
    frame = pd.read_parquet(result["data_file"])
    assert datetime.fromtimestamp(int(frame["time"].iloc[3]), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    ) == "1991-06-03 06:00"


def test_weekend_is_rejected_before_minimum_bar_check(tmp_path: Path) -> None:
    source = _legacy_h1(
        tmp_path / "raw" / "stocks" / "000001_60min.parquet",
        days=1,
        start=datetime(2024, 1, 6),
    )
    # helper skips weekends, so rewrite the encoded wall-clock date to Saturday.
    frame = pd.read_parquet(source)
    saturday = datetime(2024, 1, 6)
    spec = ASHARE_PERIOD_SPECS["60min"]
    frame["time"] = [
        int(datetime.combine(saturday.date(), close, tzinfo=timezone.utc).timestamp()) // 1000
        for close in spec.close_times
    ]
    frame.to_parquet(source, index=False)
    with pytest.raises(AShareDataError, match="周末"):
        convert_legacy_a_share_file(source, tmp_path / "converted")


def test_indices_are_rejected_as_a_class(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "indices" / "idx_000001_60min.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not-read")
    with pytest.raises(AShareDataError, match="indices"):
        convert_legacy_a_share_file(source, tmp_path / "converted")


def test_output_cannot_be_written_back_into_raw_tree(tmp_path: Path) -> None:
    source = _legacy_h1(
        tmp_path / "raw" / "stocks" / "000001_60min.parquet",
        days=1,
    )
    with pytest.raises(AShareDataError, match="原始 parquet 目录"):
        convert_legacy_a_share_file(source, tmp_path / "raw" / "converted")


def test_partial_trading_day_is_rejected_without_output(tmp_path: Path) -> None:
    source = _legacy_h1(tmp_path / "raw" / "stocks" / "000001_60min.parquet")
    frame = pd.read_parquet(source).iloc[:-1]
    frame.to_parquet(source, index=False)
    output_dir = tmp_path / "converted"
    with pytest.raises(AShareDataError, match="行数为 3"):
        convert_legacy_a_share_file(source, output_dir)
    assert not (output_dir / "000001_H1.parquet").exists()


def test_legacy_bucket_must_round_trip_exactly(tmp_path: Path) -> None:
    source = _legacy_h1(tmp_path / "raw" / "stocks" / "000001_60min.parquet")
    frame = pd.read_parquet(source)
    frame.loc[0, "time"] += 1
    frame.to_parquet(source, index=False)
    with pytest.raises(AShareDataError, match="无法反验"):
        convert_legacy_a_share_file(source, tmp_path / "converted")


def test_less_than_two_trading_years_is_rejected(tmp_path: Path) -> None:
    source = _legacy_h1(
        tmp_path / "raw" / "stocks" / "000001_60min.parquet",
        days=483,
    )
    with pytest.raises(AShareDataError, match="至少需要 1936"):
        convert_legacy_a_share_file(source, tmp_path / "converted")


@pytest.mark.parametrize(
    ("column", "mutator", "message"),
    [
        ("open", lambda series: series.astype(str), "open 必须是非 bool 数值列"),
        ("tick_volume", lambda series: series.astype("float64"), "非 bool 整数列"),
        ("tick_volume", lambda series: series.astype(bool), "非 bool 整数列"),
    ],
)
def test_legacy_source_types_are_strictly_rejected(
    tmp_path: Path,
    column: str,
    mutator,
    message: str,
) -> None:
    source = _legacy_h1(
        tmp_path / "raw" / "stocks" / "000001_60min.parquet",
        days=1,
    )
    frame = pd.read_parquet(source)
    _rewrite_column(source, column, mutator(frame[column]))
    with pytest.raises(AShareDataError, match=message):
        convert_legacy_a_share_file(source, tmp_path / "converted")


def test_lossy_float64_price_is_rejected(tmp_path: Path) -> None:
    source = _legacy_h1(
        tmp_path / "raw" / "stocks" / "000001_60min.parquet",
        days=1,
    )
    frame = pd.read_parquet(source)
    frame["open"] = frame["open"].astype("float64")
    frame.loc[0, "open"] = 10.0000001
    frame.to_parquet(source, index=False)
    with pytest.raises(AShareDataError, match="转为 float32 后数值发生变化"):
        convert_legacy_a_share_file(source, tmp_path / "converted")


def test_second_publish_failure_rolls_back_both_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _legacy_h1(tmp_path / "raw" / "stocks" / "000001_60min.parquet")
    output_dir = tmp_path / "converted"
    real_link = os.link
    calls = 0

    def fail_second_link(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated manifest publish failure")
        return real_link(source_path, destination_path)

    monkeypatch.setattr(os, "link", fail_second_link)
    with pytest.raises(OSError, match="manifest publish"):
        convert_legacy_a_share_file(source, output_dir)

    assert not (output_dir / "000001_H1.parquet").exists()
    assert not (output_dir / "000001_H1.manifest.json").exists()


def test_concurrent_target_creation_is_not_overwritten_or_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _legacy_h1(tmp_path / "raw" / "stocks" / "000001_60min.parquet")
    output_dir = tmp_path / "converted"
    output = output_dir / "000001_H1.parquet"
    real_to_parquet = pd.DataFrame.to_parquet

    def write_temp_then_create_competing_target(frame, path, *args, **kwargs):
        result = real_to_parquet(frame, path, *args, **kwargs)
        output.write_bytes(b"concurrent-owner")
        return result

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_temp_then_create_competing_target)
    with pytest.raises(AShareDataError, match="其他进程创建"):
        convert_legacy_a_share_file(source, output_dir)

    assert output.read_bytes() == b"concurrent-owner"
    assert not output.with_suffix(".manifest.json").exists()


def test_loader_rejects_duplicate_canonical_time_instead_of_deduplicating(
    tmp_path: Path,
) -> None:
    rows = Config.MIN_BARS
    timestamps = np.arange(1_700_000_000, 1_700_000_000 + rows * 3600, 3600, dtype=np.int64)
    timestamps[100] = timestamps[99]
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "open": np.full(rows, 10.0, dtype=np.float32),
            "high": np.full(rows, 11.0, dtype=np.float32),
            "low": np.full(rows, 9.0, dtype=np.float32),
            "close": np.full(rows, 10.5, dtype=np.float32),
            "tick_volume": np.full(rows, 100, dtype=np.int64),
        }
    )
    path = tmp_path / "TEST_H1.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="不得重复"):
        ParquetDataManager(path).load()


def test_tampered_periods_per_year_is_rejected(converted_h1: tuple[Path, dict]) -> None:
    output, _result = converted_h1
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["periods_per_year"] = 6240
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="periods_per_year"):
        inspect_parquet_file(output)


def test_six_digit_canonical_name_without_manifest_is_not_treated_as_generic(
    converted_h1: tuple[Path, dict],
) -> None:
    output, _result = converted_h1
    output.with_suffix(".manifest.json").unlink()
    with pytest.raises(ValueError, match="缺少有效的 A 股来源合同"):
        inspect_parquet_file(output)


def test_six_digit_canonical_name_rejects_non_ashare_sidecar(
    converted_h1: tuple[Path, dict],
) -> None:
    output, _result = converted_h1
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = "OKX"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="必须使用受支持且有效的 A 股 manifest",
    ):
        inspect_parquet_file(output)


def test_generic_file_without_sidecar_gets_hash_identity(tmp_path: Path) -> None:
    path = _generic_h1(tmp_path / "TEST_H1.parquet")
    digest = sha256_file(path)
    info = inspect_parquet_file(path)
    assert info["source"] == "local_file"
    assert info["data_sha256"] == digest
    assert info["dataset_id"] == f"sha256:{digest}"


def test_supported_generic_manifest_is_verified(tmp_path: Path) -> None:
    path = _generic_h1(tmp_path / "TEST_H1.parquet")
    frame = pd.read_parquet(path)
    digest = sha256_file(path)
    manifest = {
        "format": "alphamaster_mt5_dataset_v1",
        "source": "MetaTrader5",
        "symbol": "TEST",
        "timeframe": "H1",
        "data_filename": path.name,
        "data_sha256": digest,
        "data_rows": len(frame),
        "data_start": datetime.fromtimestamp(
            int(frame["time"].iloc[0]), tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "data_end": datetime.fromtimestamp(
            int(frame["time"].iloc[-1]), tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "data_timezone": "UTC",
        "time_unit": "unix_seconds",
        "columns": list(frame.columns),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    info = inspect_parquet_file(path)
    assert info["source"] == "mt5"
    assert info["data_sha256"] == digest
    assert info["dataset_id"] == f"sha256:{digest}"

    manifest["periods_per_year"] = 968
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="periods_per_year"):
        inspect_parquet_file(path)
    manifest.pop("periods_per_year")

    manifest["minimum_bars"] = 1
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum_bars"):
        inspect_parquet_file(path)
    manifest.pop("minimum_bars")

    manifest["data_sha256"] = "0" * 64
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data_sha256"):
        inspect_parquet_file(path)


def test_verified_worker_contract_loads_ashare_without_sidecar(
    converted_h1: tuple[Path, dict],
) -> None:
    output, _result = converted_h1
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.rename(manifest_path.with_suffix(".manifest.saved"))

    info = inspect_parquet_file(
        output,
        expected_source_id="ashare_local",
        expected_periods_per_year=968,
        expected_minimum_bars=1936,
    )
    assert info["periods_per_year"] == 968
    manager = ParquetDataManager(
        output,
        expected_source_id="ashare_local",
        expected_periods_per_year=968,
        expected_minimum_bars=1936,
    )
    manager.load()
    assert manager.source == "ashare_local"
    assert manager.minimum_bars == 1936
