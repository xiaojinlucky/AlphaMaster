from __future__ import annotations

import json
import hashlib
import shutil
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
    preflight_dataset_access,
)
import data_pipeline.dataset_purpose as dataset_purpose
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

    for result, access_mode in (
        (training, "training"),
        (evaluation, "sealed_oos_evaluation"),
    ):
        path = Path(result["data_file"])
        frame = pd.read_parquet(path)
        assert load_akshare_hfq_manifest(path, frame) is not None
        manager = ParquetDataManager(
            path,
            access_mode=access_mode,
            expected_source_id=AKSHARE_HFQ_SOURCE_ID,
            expected_periods_per_year=242,
            expected_minimum_bars=484,
        )
        manager.load()
        assert manager.data_sha256 == result["data_sha256"]

    training_info = inspect_parquet_file(
        training["data_file"],
        access_mode="training",
    )
    evaluation_info = inspect_parquet_file(
        evaluation["data_file"],
        access_mode="sealed_oos_evaluation",
    )
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


def test_sealed_slice_is_rejected_before_any_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _parent(tmp_path)
    evaluation = publish_akshare_hfq_slice(
        parent_data_file=parent,
        output_dir=tmp_path / "evaluation",
        start_index=550,
        end_index=1100,
        score_start_index=800,
        purpose=AKSHARE_SLICE_SEALED_EVALUATION,
        universe_contract_sha256="f" * 64,
    )
    data_path = Path(evaluation["data_file"])
    reads = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("封存数据不得进入 pd.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    for access_mode in ("standard", "training"):
        with pytest.raises(ValueError, match="封存评估数据"):
            inspect_parquet_file(data_path, access_mode=access_mode)
        manager = ParquetDataManager(data_path, access_mode=access_mode)
        with pytest.raises(ValueError, match="封存评估数据"):
            manager.load()
    assert reads == 0


def test_legacy_split_hash_blocks_renamed_sealed_bytes_without_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"time": [1], "close": [1.0]})
    training_path = tmp_path / "training.parquet"
    sealed_path = tmp_path / "sealed.parquet"
    frame.to_parquet(training_path, index=False)
    frame.assign(close=[2.0]).to_parquet(sealed_path, index=False)
    training_hash = hashlib.sha256(training_path.read_bytes()).hexdigest()
    sealed_hash = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    training_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": training_hash,
                "derivation": {"purpose": "training"},
            }
        ),
        encoding="utf-8",
    )
    items = [
        {
            "symbol": f"{index:06d}",
            "training": {
                "data_sha256": (
                    training_hash if index == 1 else f"{index:064x}"
                )
            },
            "sealed_evaluation": {
                "data_sha256": (
                    sealed_hash
                    if index == 1
                    else f"{index + 100:064x}"
                )
            },
        }
        for index in range(1, 51)
    ]
    body = {
        "format": "alphamaster_a50_sealed_split_v1",
        "symbol_count": 50,
        "items": items,
    }
    contract_hash = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    split_path = tmp_path / "trusted-split.json"
    split_path.write_text(
        json.dumps({**body, "contract_sha256": contract_hash}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dataset_purpose,
        "TRUSTED_SEALED_SPLIT_CONTRACTS",
        (split_path,),
    )

    assert (
        preflight_dataset_access(training_path, access_mode="training")
        == "training"
    )

    renamed = tmp_path / "innocent_training.parquet"
    shutil.copyfile(sealed_path, renamed)
    reads = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("用途门禁前不得调用 pd.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    with pytest.raises(ValueError, match="封存评估数据"):
        preflight_dataset_access(renamed, access_mode="training")
    assert reads == 0


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
