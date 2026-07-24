from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.a_share_akshare import (
    AKSHARE_SOURCE_COLUMNS,
    AKShareDataError,
    akshare_sina_provider_symbol,
    canonicalize_akshare_hfq_daily,
    download_akshare_hfq_daily,
    load_akshare_hfq_manifest,
)
from data_pipeline.dataset_contracts import AKSHARE_HFQ_SOURCE_ID
from data_pipeline.parquet_manager import ParquetDataManager


def _raw_daily(symbol: str = "600519", rows: int = 520) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=rows)
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.05
    return pd.DataFrame(
        {
            "date": dates.date,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "volume": np.arange(rows, dtype=np.int64) + 10_000,
            "amount": (np.arange(rows, dtype=np.float64) + 1) * 1_000_000,
            "outstanding_share": np.full(rows, 1_000_000_000),
            "turnover": np.full(rows, 0.001),
        },
        columns=list(AKSHARE_SOURCE_COLUMNS),
    )


def _fake_fetcher(frame: pd.DataFrame):
    calls: list[dict] = []

    def fetcher(**kwargs):
        calls.append(dict(kwargs))
        return frame.copy()

    return fetcher, calls


def test_download_hfq_daily_publishes_frozen_contract(tmp_path: Path) -> None:
    raw = _raw_daily()
    fetcher, calls = _fake_fetcher(raw)

    result = download_akshare_hfq_daily(
        symbol="600519",
        start_date="20200101",
        end_date="20240101",
        output_dir=tmp_path,
        fetcher=fetcher,
        provider_version="1.18.64",
    )

    assert calls == [
        {
            "symbol": "sh600519",
            "start_date": "20200101",
            "end_date": "20240101",
            "adjust": "hfq",
        }
    ]
    data_file = Path(result["data_file"])
    frame = pd.read_parquet(data_file)
    manifest = json.loads(
        data_file.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert list(frame.columns) == [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
    ]
    assert frame.dtypes.astype(str).to_dict() == {
        "time": "int64",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "tick_volume": "int64",
    }
    first_close = datetime.fromtimestamp(
        int(frame["time"].iloc[0]),
        tz=timezone.utc,
    )
    assert (first_close.hour, first_close.minute) == (7, 0)
    assert manifest["source_id"] == AKSHARE_HFQ_SOURCE_ID
    assert manifest["provider_version"] == "1.18.64"
    assert manifest["provider_interface"] == "stock_zh_a_daily"
    assert manifest["adjustment"] == "hfq"
    assert manifest["request"]["canonical_symbol"] == "600519"
    assert manifest["request"]["symbol"] == "sh600519"
    assert manifest["request"]["adjust"] == "hfq"
    assert manifest["data_sha256"] == result["data_sha256"]
    assert len(manifest["source_response_sha256"]) == 64

    loaded = load_akshare_hfq_manifest(data_file, frame)
    assert loaded is not None
    manager = ParquetDataManager(
        data_file,
        expected_source_id=AKSHARE_HFQ_SOURCE_ID,
        expected_periods_per_year=242,
        expected_minimum_bars=484,
    )
    manager.load()
    assert manager.source == AKSHARE_HFQ_SOURCE_ID
    assert manager.periods_per_year == 242


def test_provider_column_drift_is_rejected() -> None:
    raw = _raw_daily().rename(columns={"volume": "trade_volume"})
    with pytest.raises(AKShareDataError, match="返回列合同变化"):
        canonicalize_akshare_hfq_daily(raw, symbol="600519")


def test_provider_symbol_and_non_monotonic_dates_are_strict() -> None:
    assert akshare_sina_provider_symbol("600519") == "sh600519"
    assert akshare_sina_provider_symbol("000333") == "sz000333"
    assert akshare_sina_provider_symbol("300750") == "sz300750"
    with pytest.raises(AKShareDataError, match="不支持"):
        akshare_sina_provider_symbol("400001")

    reversed_dates = _raw_daily()
    reversed_dates.loc[[3, 4], "date"] = reversed_dates.loc[
        [4, 3], "date"
    ].to_numpy()
    with pytest.raises(AKShareDataError, match="严格递增"):
        canonicalize_akshare_hfq_daily(reversed_dates, symbol="600519")


def test_fetch_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    def fail(**_kwargs):
        raise TimeoutError("upstream timeout")

    with pytest.raises(AKShareDataError, match="下载失败"):
        download_akshare_hfq_daily(
            symbol="600519",
            start_date="20200101",
            end_date="20240101",
            output_dir=tmp_path,
            fetcher=fail,
            provider_version="1.18.64",
        )
    assert list(tmp_path.iterdir()) == []


def test_existing_snapshot_is_never_overwritten(tmp_path: Path) -> None:
    fetcher, _calls = _fake_fetcher(_raw_daily())
    kwargs = {
        "symbol": "600519",
        "start_date": "20200101",
        "end_date": "20240101",
        "output_dir": tmp_path,
        "fetcher": fetcher,
        "provider_version": "1.18.64",
    }
    first = download_akshare_hfq_daily(**kwargs)
    original = Path(first["data_file"]).read_bytes()
    with pytest.raises(AKShareDataError, match="拒绝静默覆盖"):
        download_akshare_hfq_daily(**kwargs)
    assert Path(first["data_file"]).read_bytes() == original


def test_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    fetcher, _calls = _fake_fetcher(_raw_daily())
    result = download_akshare_hfq_daily(
        symbol="600519",
        start_date="20200101",
        end_date="20240101",
        output_dir=tmp_path,
        fetcher=fetcher,
        provider_version="1.18.64",
    )
    data_file = Path(result["data_file"])
    manifest_path = data_file.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adjustment"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AKShareDataError, match="adjustment"):
        load_akshare_hfq_manifest(data_file, pd.read_parquet(data_file))
