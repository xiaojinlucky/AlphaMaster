from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.a_share_akshare import (
    AKSHARE_SOURCE_COLUMNS,
    download_akshare_hfq_daily,
    load_akshare_hfq_manifest,
)
from scripts import freeze_csi_a50_universe as freeze_tool
from scripts.build_a50_sealed_split import (
    SPLIT_KEYS,
    SealedSplitError,
    _canonical_hash,
    build_a50_sealed_split,
    load_a50_sealed_split,
)
from scripts.enqueue_a50_training_batch import (
    build_a50_batch_plan,
    enqueue_a50_batch,
)
from web.training_queue import TrainingQueue


def _universe(path: Path) -> dict:
    rows = []
    for index in range(50):
        symbol = f"{index + 1:06d}"
        rows.append(
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
    payload = freeze_tool._build_payload(
        pd.DataFrame(rows, columns=list(freeze_tool.EXPECTED_COLUMNS)),
        "a" * 64,
    )
    freeze_tool.write_json_exclusive(path, payload)
    return payload


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


def test_builds_exact_fifty_physical_train_and_sealed_slices(
    tmp_path: Path,
) -> None:
    universe_path = tmp_path / "universe.json"
    universe = _universe(universe_path)
    parent_dir = tmp_path / "parent"
    raw = _raw_daily()
    for constituent in universe["constituents"]:
        download_akshare_hfq_daily(
            symbol=constituent["symbol"],
            start_date="20190101",
            end_date="20250101",
            output_dir=parent_dir,
            fetcher=lambda **_kwargs: raw.copy(),
            provider_version="1.18.64",
        )

    requested_start = raw["date"].iloc[800].strftime("%Y%m%d")
    training_dir = tmp_path / "training"
    evaluation_dir = tmp_path / "evaluation"
    contract_path = tmp_path / "split.json"
    contract = build_a50_sealed_split(
        universe_json=universe_path,
        parent_dir=parent_dir,
        training_dir=training_dir,
        evaluation_dir=evaluation_dir,
        requested_test_start=requested_start,
        output_contract=contract_path,
        warmup_bars=252,
    )

    assert contract["symbol_count"] == 50
    assert contract["common_test_bar_count"] == 300
    assert contract["minimum_symbol_test_bar_count"] == 300
    assert contract["maximum_symbol_test_bar_count"] == 300
    assert len(contract["items"]) == 50
    assert len(contract["contract_sha256"]) == 64
    assert len(list(training_dir.glob("*_D1.parquet"))) == 50
    assert len(list(evaluation_dir.glob("*_D1.parquet"))) == 50
    assert all(item["training"]["data_rows"] == 800 for item in contract["items"])
    assert all(
        item["sealed_evaluation"]["warmup_bars"] == 252
        for item in contract["items"]
    )
    loaded = load_a50_sealed_split(contract_path)
    assert loaded["contract_sha256"] == contract["contract_sha256"]
    plan = build_a50_batch_plan(contract_path)
    assert len(plan.items) == 50
    assert len({item.planned_run_id for item in plan.items}) == 50
    assert all(item.train_steps == 200 for item in plan.items)
    assert all(item.cpus_per_task == 12 for item in plan.items)
    queue_path = tmp_path / "queue.sqlite3"
    first_batch = enqueue_a50_batch(plan, queue_db=queue_path)
    repeated_batch = enqueue_a50_batch(plan, queue_db=queue_path)
    assert first_batch.created is True
    assert repeated_batch.created is False
    queue = TrainingQueue(queue_path)
    queued_items = queue.list_items(first_batch.batch.batch_id)
    assert len(queued_items) == 50
    assert all(item.status == "QUEUED" for item in queued_items)

    sample = evaluation_dir / "000001_D1.parquet"
    manifest = load_akshare_hfq_manifest(sample, pd.read_parquet(sample))
    assert manifest is not None
    assert manifest["derivation"]["score_start"] == contract["test_start"]

    with pytest.raises(SealedSplitError, match="拒绝覆盖"):
        build_a50_sealed_split(
            universe_json=universe_path,
            parent_dir=parent_dir,
            training_dir=training_dir,
            evaluation_dir=evaluation_dir,
            requested_test_start=requested_start,
            output_contract=contract_path,
            warmup_bars=252,
        )

    swapped = dict(contract)
    swapped["training_data_dir"] = contract["evaluation_data_dir"]
    swapped["evaluation_data_dir"] = contract["training_data_dir"]
    swapped["contract_sha256"] = _canonical_hash(
        {key: swapped[key] for key in SPLIT_KEYS[:-1]}
    )
    swapped_path = tmp_path / "swapped.json"
    swapped_path.write_text(
        json.dumps(swapped, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(SealedSplitError, match="物理数据|切片派生身份"):
        load_a50_sealed_split(swapped_path)
