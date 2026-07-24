from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.a_share_akshare import (
    AKSHARE_SLICE_SEALED_EVALUATION,
    AKSHARE_SLICE_TRAINING,
    AKSHARE_SOURCE_COLUMNS,
    AKShareDataError,
    download_akshare_hfq_daily,
    load_akshare_hfq_manifest,
    publish_akshare_hfq_slice,
)
from data_pipeline.dataset_contracts import AKSHARE_HFQ_SOURCE_ID
from data_pipeline.parquet_manager import (
    ParquetDataManager,
    inspect_parquet_file,
)
from web.slurm_training_manager import SlurmTrainingManager


def _raw_daily(rows: int = 1100) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=rows)
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.03
    return pd.DataFrame(
        {
            "date": dates.date,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "volume": np.arange(rows, dtype=np.int64) + 1_000_000,
            "amount": (np.arange(rows, dtype=np.float64) + 1) * 1_000_000,
            "outstanding_share": np.full(rows, 1_000_000_000),
            "turnover": np.full(rows, 0.001),
        },
        columns=list(AKSHARE_SOURCE_COLUMNS),
    )


def _parent(tmp_path: Path) -> Path:
    result = download_akshare_hfq_daily(
        symbol="600519",
        start_date="20190101",
        end_date="20250101",
        output_dir=tmp_path / "parent",
        fetcher=lambda **_kwargs: _raw_daily(),
        provider_version="1.18.64",
    )
    return Path(result["data_file"])


def test_training_and_sealed_slices_are_physical_and_auditable(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path)
    universe_hash = "b" * 64

    training = publish_akshare_hfq_slice(
        parent_data_file=parent,
        output_dir=tmp_path / "training",
        start_index=0,
        end_index=800,
        purpose=AKSHARE_SLICE_TRAINING,
        universe_contract_sha256=universe_hash,
    )
    evaluation = publish_akshare_hfq_slice(
        parent_data_file=parent,
        output_dir=tmp_path / "evaluation",
        start_index=550,
        end_index=1100,
        score_start_index=800,
        purpose=AKSHARE_SLICE_SEALED_EVALUATION,
        universe_contract_sha256=universe_hash,
    )

    assert training["data_sha256"] != evaluation["data_sha256"]
    assert training["derivation"]["parent_data_sha256"] == (
        evaluation["derivation"]["parent_data_sha256"]
    )
    assert training["derivation"]["purpose"] == AKSHARE_SLICE_TRAINING
    assert evaluation["derivation"]["purpose"] == (
        AKSHARE_SLICE_SEALED_EVALUATION
    )
    assert evaluation["derivation"]["warmup_bars"] == 250
    assert evaluation["derivation"]["score_start"] > training["data_end"]

    for result in (training, evaluation):
        path = Path(result["data_file"])
        frame = pd.read_parquet(path)
        assert load_akshare_hfq_manifest(path, frame) is not None
        manager = ParquetDataManager(
            path,
            expected_source_id=AKSHARE_HFQ_SOURCE_ID,
            expected_periods_per_year=242,
            expected_minimum_bars=484,
        )
        manager.load()
        assert manager.data_sha256 == result["data_sha256"]

    training_info = inspect_parquet_file(training["data_file"])
    evaluation_info = inspect_parquet_file(evaluation["data_file"])
    assert training_info["dataset_purpose"] == AKSHARE_SLICE_TRAINING
    assert training_info["capabilities"]["remote_training"] is True
    assert (
        evaluation_info["dataset_purpose"]
        == AKSHARE_SLICE_SEALED_EVALUATION
    )
    assert evaluation_info["capabilities"]["remote_training"] is False

    slurm = SlurmTrainingManager(
        client=object(),
        local_runs_root=tmp_path / "runs",
    )
    with pytest.raises(RuntimeError, match="封存评估数据禁止"):
        slurm._load_data_manifest(Path(evaluation["data_file"]))


def test_slice_target_is_never_overwritten(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    kwargs = {
        "parent_data_file": parent,
        "output_dir": tmp_path / "training",
        "start_index": 0,
        "end_index": 800,
        "purpose": AKSHARE_SLICE_TRAINING,
        "universe_contract_sha256": "c" * 64,
    }
    first = publish_akshare_hfq_slice(**kwargs)
    original = Path(first["data_file"]).read_bytes()

    with pytest.raises(AKShareDataError, match="拒绝覆盖"):
        publish_akshare_hfq_slice(**kwargs)
    assert Path(first["data_file"]).read_bytes() == original


def test_sealed_slice_rejects_too_short_warmup(tmp_path: Path) -> None:
    parent = _parent(tmp_path)

    with pytest.raises(AKShareDataError, match="至少需要 200"):
        publish_akshare_hfq_slice(
            parent_data_file=parent,
            output_dir=tmp_path / "evaluation",
            start_index=601,
            end_index=1100,
            score_start_index=800,
            purpose=AKSHARE_SLICE_SEALED_EVALUATION,
            universe_contract_sha256="d" * 64,
        )


def test_tampered_slice_derivation_is_rejected(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    result = publish_akshare_hfq_slice(
        parent_data_file=parent,
        output_dir=tmp_path / "evaluation",
        start_index=550,
        end_index=1100,
        score_start_index=800,
        purpose=AKSHARE_SLICE_SEALED_EVALUATION,
        universe_contract_sha256="e" * 64,
    )
    data_path = Path(result["data_file"])
    manifest_path = data_path.with_suffix(".manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["derivation"]["warmup_bars"] = 249
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(AKShareDataError, match="预热边界"):
        load_akshare_hfq_manifest(data_path, pd.read_parquet(data_path))
