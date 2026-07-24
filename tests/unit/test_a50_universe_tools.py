from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.a_share_akshare import (
    AKSHARE_SOURCE_COLUMNS,
    download_akshare_hfq_daily,
)
from scripts import freeze_csi_a50_universe as freeze_tool
from scripts.download_a_share_training_data import (
    BatchDownloadError,
    download_universe_data,
)


def _official_frame(rows: int = 50) -> pd.DataFrame:
    records = []
    for offset in range(rows):
        symbol = f"{offset + 1:06d}"
        records.append(
            {
                "日期Date": "20260723",
                "指数代码 Index Code": "930050",
                "指数名称 Index Name": "中证A50",
                "指数英文名称Index Name(Eng)": "CSI A50",
                "成份券代码Constituent Code": symbol,
                "成份券名称Constituent Name": f"股票{symbol}",
                "成份券英文名称Constituent Name(Eng)": f"Stock {symbol}",
                "交易所Exchange": "深圳证券交易所",
                "交易所英文名称Exchange(Eng)": "Shenzhen Stock Exchange",
            }
        )
    return pd.DataFrame(records, columns=list(freeze_tool.EXPECTED_COLUMNS))


def _freeze_with_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    *,
    output_name: str = "universe.json",
) -> tuple[Path, dict]:
    source = tmp_path / "930050cons.xls"
    source.write_bytes(b"official-test-xls")
    monkeypatch.setattr(
        freeze_tool.pd,
        "read_excel",
        lambda *_args, **_kwargs: frame.copy(),
    )
    output = tmp_path / output_name
    payload = freeze_tool.freeze_csi_a50_universe(source, output)
    return output, payload


def _raw_daily(symbol: str, rows: int = 520) -> pd.DataFrame:
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


def _publish_snapshot(
    *,
    symbol: str,
    output_dir: Path,
    start_date: str = "20200101",
    end_date: str = "20240101",
) -> dict:
    def fetcher(**_kwargs):
        return _raw_daily(symbol)

    return download_akshare_hfq_daily(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        fetcher=fetcher,
        provider_version="1.18.64",
    )


def test_freeze_publishes_exact_fifty_and_recomputable_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, payload = _freeze_with_frame(tmp_path, monkeypatch, _official_frame())

    loaded = freeze_tool.load_frozen_universe(output)
    assert loaded == payload
    assert payload["constituent_count"] == 50
    assert len(payload["constituents"]) == 50
    assert payload["source_url"] == freeze_tool.DEFAULT_SOURCE_URL
    assert len(payload["source_xls_sha256"]) == 64
    assert len(payload["contract_sha256"]) == 64


def test_xls_column_drift_and_non_fifty_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = _official_frame().rename(columns={"交易所Exchange": "市场Exchange"})
    source = tmp_path / "930050cons.xls"
    source.write_bytes(b"official-test-xls")
    monkeypatch.setattr(
        freeze_tool.pd,
        "read_excel",
        lambda *_args, **_kwargs: drifted.copy(),
    )
    with pytest.raises(freeze_tool.UniverseContractError, match="列合同发生变化"):
        freeze_tool.freeze_csi_a50_universe(source, tmp_path / "drift.json")

    monkeypatch.setattr(
        freeze_tool.pd,
        "read_excel",
        lambda *_args, **_kwargs: _official_frame(49),
    )
    with pytest.raises(freeze_tool.UniverseContractError, match="恰好有 50 行"):
        freeze_tool.freeze_csi_a50_universe(source, tmp_path / "forty-nine.json")


def test_freeze_and_summary_never_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _payload = _freeze_with_frame(tmp_path, monkeypatch, _official_frame())
    original = output.read_bytes()
    with pytest.raises(freeze_tool.UniverseContractError, match="拒绝覆盖"):
        freeze_tool.freeze_csi_a50_universe(
            tmp_path / "930050cons.xls",
            output,
        )
    assert output.read_bytes() == original

    summary = tmp_path / "summary.json"
    summary.write_text('{"keep": true}\n', encoding="utf-8")
    with pytest.raises(BatchDownloadError, match="summary 已存在"):
        download_universe_data(
            universe_json=output,
            output_dir=tmp_path / "data",
            start_date="20200101",
            end_date="20240101",
            summary_json=summary,
        )
    assert json.loads(summary.read_text(encoding="utf-8")) == {"keep": True}


def test_frozen_json_contract_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _payload = _freeze_with_frame(tmp_path, monkeypatch, _official_frame())
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(
        freeze_tool.UniverseContractError,
        match="顶层字段合同发生变化",
    ):
        freeze_tool.load_frozen_universe(drifted)


def test_valid_existing_snapshot_is_verified_and_skipped_before_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, _payload = _freeze_with_frame(
        tmp_path,
        monkeypatch,
        _official_frame(),
    )
    data_dir = tmp_path / "data"
    _publish_snapshot(symbol="000001", output_dir=data_dir)
    calls: list[str] = []

    def fail_second(**kwargs):
        calls.append(kwargs["symbol"])
        raise TimeoutError("no retry")

    summary_path = tmp_path / "summary.json"
    with pytest.raises(BatchDownloadError, match="000002 处理失败"):
        download_universe_data(
            universe_json=universe,
            output_dir=data_dir,
            start_date="20200101",
            end_date="20240101",
            summary_json=summary_path,
            download_func=fail_second,
            sleep_func=lambda _seconds: None,
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert calls == ["000002"]
    assert summary["status"] == "failed"
    assert summary["processed_count"] == 2
    assert summary["skipped_verified_count"] == 1
    assert summary["download_attempt_count"] == 1
    assert [item["status"] for item in summary["items"]] == [
        "skipped_verified",
        "failed",
    ]


def test_tampered_resume_snapshot_stops_without_redownload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, _payload = _freeze_with_frame(
        tmp_path,
        monkeypatch,
        _official_frame(),
    )
    data_dir = tmp_path / "data"
    result = _publish_snapshot(symbol="000001", output_dir=data_dir)
    manifest_path = Path(result["manifest_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[str] = []

    def should_not_download(**kwargs):
        calls.append(kwargs["symbol"])
        raise AssertionError("不得重下")

    summary_path = tmp_path / "summary.json"
    with pytest.raises(BatchDownloadError, match="000001 处理失败"):
        download_universe_data(
            universe_json=universe,
            output_dir=data_dir,
            start_date="20200101",
            end_date="20240101",
            summary_json=summary_path,
            download_func=should_not_download,
            sleep_func=lambda _seconds: None,
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert calls == []
    assert summary["processed_count"] == 1
    assert summary["download_attempt_count"] == 0
    assert summary["items"][0]["status"] == "failed"


def test_actual_downloads_are_serial_throttled_and_stop_on_first_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe, _payload = _freeze_with_frame(
        tmp_path,
        monkeypatch,
        _official_frame(),
    )
    data_dir = tmp_path / "data"
    calls: list[str] = []
    sleeps: list[float] = []

    def downloader(**kwargs):
        symbol = kwargs["symbol"]
        calls.append(symbol)
        if symbol == "000003":
            raise TimeoutError("stop now")

        def fetcher(**_fetch_kwargs):
            return _raw_daily(symbol)

        return download_akshare_hfq_daily(
            **kwargs,
            fetcher=fetcher,
            provider_version="1.18.64",
        )

    with pytest.raises(BatchDownloadError, match="000003 处理失败"):
        download_universe_data(
            universe_json=universe,
            output_dir=data_dir,
            start_date="20200101",
            end_date="20240101",
            summary_json=tmp_path / "summary.json",
            throttle_seconds=1.25,
            download_func=downloader,
            sleep_func=sleeps.append,
        )

    assert calls == ["000001", "000002", "000003"]
    assert sleeps == [1.25, 1.25]
