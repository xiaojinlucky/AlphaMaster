from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import evaluation.sealed_oos_campaign as sealed_oos_campaign
from evaluation.sealed_oos_campaign import (
    ContractValidationError,
    RevealAlreadyStartedError,
    ResultAlreadyExistsError,
    SealedOOSItem,
    authorize_sealed_oos_report,
    create_sealed_oos_campaign as _create_sealed_oos_campaign,
    evaluate_sealed_oos_campaign,
    load_sealed_oos_campaign,
)
from model_core.target_contract import SCORING_CONTRACT_VERSION


TEST_START = "2025-01-02T07:00:00Z"
TEST_END = "2025-12-31T07:00:00Z"
COMMISSION_PCT = 0.02
SLIPPAGE_PCT = 0.01
COST_RATE = 0.0003
SPLIT_CONTRACT_SHA256 = "c" * 64
UNIVERSE_CONTRACT_SHA256 = "d" * 64
RUNTIME_GIT_COMMIT = "e" * 40


def create_sealed_oos_campaign(*args, **kwargs):
    kwargs.setdefault("split_contract_sha256", SPLIT_CONTRACT_SHA256)
    kwargs.setdefault("split_contract_path", "artifacts/split.json")
    kwargs.setdefault("universe_contract_sha256", UNIVERSE_CONTRACT_SHA256)
    kwargs.setdefault(
        "universe_contract_path", "artifacts/universe.json"
    )
    return _create_sealed_oos_campaign(*args, **kwargs)


@pytest.fixture(autouse=True)
def _isolated_reveal_registry(tmp_path: Path, monkeypatch):
    registry = tmp_path / "global-reveal-registry"
    monkeypatch.setattr(
        sealed_oos_campaign,
        "REVEAL_REGISTRY_DIR",
        registry,
    )
    return registry


def _hash(prefix: str, index: int) -> str:
    return hashlib.sha256(f"{prefix}-{index}".encode("ascii")).hexdigest()


def _items() -> list[SealedOOSItem]:
    return [
        SealedOOSItem(
            symbol=f"{index:06d}",
            data_sha256=_hash("data", index),
            data_manifest_path=(
                f"artifacts/{index:06d}/sealed.manifest.json"
            ),
            data_manifest_sha256=_hash("manifest", index),
            strategy_sha256=_hash("strategy", index),
            published_strategy_path=(
                f"artifacts/{index:06d}/published_strategy.json"
            ),
            published_strategy_sha256=_hash("strategy", index),
            training_run_id=(
                f"run_20250101T000000Z_{index:08x}"
            ),
            training_result_manifest_path=(
                f"artifacts/{index:06d}/result_manifest.json"
            ),
            training_result_manifest_sha256=_hash("result", index),
            runtime_git_commit=RUNTIME_GIT_COMMIT,
            scoring_contract_version=SCORING_CONTRACT_VERSION,
            test_start=TEST_START,
            test_end=TEST_END,
            report_path=f"reports/{index:06d}.json",
        )
        for index in range(1, 51)
    ]


def _contract_artifact(payload: dict[str, object]) -> dict[str, object]:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "contract_sha256": digest}


def _materialize_provenance(
    root: Path,
    items: list[SealedOOSItem],
) -> tuple[list[SealedOOSItem], dict[str, object], dict[str, object]]:
    universe = _contract_artifact(
        {
            "format": "test_universe_v1",
            "constituents": [
                {"symbol": item.symbol} for item in items
            ],
        }
    )
    universe_path = root / "artifacts" / "universe.json"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.write_text(
        json.dumps(universe, ensure_ascii=False),
        encoding="utf-8",
    )
    materialized: list[SealedOOSItem] = []
    for item in items:
        data_manifest_path = root / str(item.data_manifest_path)
        strategy_path = root / str(item.published_strategy_path)
        result_path = root / str(item.training_result_manifest_path)
        data_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data_manifest = {
            "symbol": item.symbol,
            "data_sha256": item.data_sha256,
            "derivation": {
                "purpose": "sealed_oos_evaluation",
                "universe_contract_sha256": universe[
                    "contract_sha256"
                ],
            },
        }
        data_manifest_path.write_text(
            json.dumps(data_manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        strategy = {
            "symbol": item.symbol,
            "run_id": item.training_run_id,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
        }
        strategy_path.write_text(
            json.dumps(strategy, ensure_ascii=False),
            encoding="utf-8",
        )
        strategy_hash = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
        checkpoint_path = (
            f"checkpoints/D1/{item.data_sha256}/"
            f"{item.training_run_id}/ckpt_{item.symbol}_step_0001.pt"
        )
        checkpoint_hash = hashlib.sha256(
            f"checkpoint:{item.symbol}".encode("ascii")
        ).hexdigest()
        result = {
            "status": "COMPLETED",
            "run_id": item.training_run_id,
            "symbol": item.symbol,
            "git_commit": item.runtime_git_commit,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "checkpoint_files": [checkpoint_path],
            "artifact_sha256": {
                f"strategies/best_{item.symbol}.json": strategy_hash,
                checkpoint_path: checkpoint_hash,
            },
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False),
            encoding="utf-8",
        )
        materialized.append(
            replace(
                item,
                data_manifest_sha256=hashlib.sha256(
                    data_manifest_path.read_bytes()
                ).hexdigest(),
                strategy_sha256=strategy_hash,
                published_strategy_sha256=strategy_hash,
                training_result_manifest_sha256=hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
            )
        )
    split = _contract_artifact(
        {
            "format": "test_split_v1",
            "universe_contract_sha256": universe["contract_sha256"],
            "items": [
                {
                    "symbol": item.symbol,
                    "sealed_evaluation": {
                        "data_sha256": item.data_sha256,
                        "data_manifest_sha256": item.data_manifest_sha256,
                        "score_start": item.test_start,
                        "data_end": item.test_end,
                    },
                }
                for item in materialized
            ],
        }
    )
    (root / "artifacts" / "split.json").write_text(
        json.dumps(split, ensure_ascii=False),
        encoding="utf-8",
    )
    return materialized, split, universe


def _report(
    item: SealedOOSItem,
    contract: dict[str, object],
    sharpe: float = 1.5,
) -> dict[str, object]:
    return {
        "format": "alphamaster_sealed_oos_report_v3",
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "sealed_dataset_sha256": contract["sealed_dataset_sha256"],
        "split_contract_sha256": contract["split_contract_sha256"],
        "universe_contract_sha256": contract[
            "universe_contract_sha256"
        ],
        "symbol": item.symbol,
        "data_sha256": item.data_sha256,
        "data_manifest_sha256": item.data_manifest_sha256,
        "strategy_sha256": item.strategy_sha256,
        "published_strategy_sha256": item.published_strategy_sha256,
        "training_run_id": item.training_run_id,
        "training_result_manifest_sha256": (
            item.training_result_manifest_sha256
        ),
        "runtime_git_commit": item.runtime_git_commit,
        "scoring_contract_version": item.scoring_contract_version,
        "evaluation_mode": "sealed_oos",
        "test_start": item.test_start,
        "test_end": item.test_end,
        "commission_pct": COMMISSION_PCT,
        "slippage_pct": SLIPPAGE_PCT,
        "cost_rate": COST_RATE,
        "sharpe": sharpe,
    }


def _write_reports(
    contract_path: Path,
    items: list[SealedOOSItem],
    sharpe: float = 1.5,
    sharpe_overrides: dict[str, float] | None = None,
) -> None:
    contract = load_sealed_oos_campaign(contract_path)
    root = contract_path.parent
    for item in items:
        session = sealed_oos_campaign._begin_controlled_sealed_oos_run(
            contract_path,
            symbol=item.symbol,
            data_sha256=item.data_sha256,
            strategy_sha256=item.strategy_sha256,
            report_path=root / str(item.report_path),
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )
        sealed_oos_campaign._complete_controlled_sealed_oos_run(
            contract_path,
            symbol=item.symbol,
            controlled_session=session,
            report_payload=_report(
                item,
                contract,
                (sharpe_overrides or {}).get(item.symbol, sharpe),
            ),
        )


def _create_campaign(
    tmp_path: Path,
    items: list[SealedOOSItem] | None = None,
) -> tuple[Path, list[SealedOOSItem]]:
    selected, split, universe = _materialize_provenance(
        tmp_path,
        items or _items(),
    )
    contract_path = tmp_path / "campaign.json"
    create_sealed_oos_campaign(
        contract_path,
        campaign_id="csi-a50-seal-2025",
        items=selected,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        split_contract_sha256=split["contract_sha256"],
        universe_contract_sha256=universe["contract_sha256"],
    )
    return contract_path, selected


def _reveal_lock_path(contract: dict[str, object]) -> Path:
    return sealed_oos_campaign.REVEAL_REGISTRY_DIR / (
        f"{contract['sealed_dataset_sha256']}.reveal.lock"
    )


def test_public_split_authorization_cannot_mint_zero_read_report_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开 API 必须在任何 Parquet 读取前拒绝拆分授权/铸据链。"""
    contract_path, items = _create_campaign(tmp_path)
    parquet_reads = 0

    def counted_read_parquet(*args, **kwargs):
        nonlocal parquet_reads
        parquet_reads += 1
        raise AssertionError("拆分授权入口不得读取 Parquet")

    monkeypatch.setattr(pd, "read_parquet", counted_read_parquet)
    item = items[0]
    with pytest.raises(ContractValidationError, match="受控 runner"):
        authorize_sealed_oos_report(
            contract_path,
            symbol=item.symbol,
            data_sha256=item.data_sha256,
            strategy_sha256=item.strategy_sha256,
            report_path=tmp_path / str(item.report_path),
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )
    with pytest.raises(ContractValidationError, match="受控 runner"):
        sealed_oos_campaign.register_authorized_sealed_report(
            contract_path,
            symbol=item.symbol,
            authorization_token="0" * 64,
        )

    assert parquet_reads == 0
    assert not (tmp_path / str(item.report_path)).exists()
    assert not sealed_oos_campaign._report_receipt_path(
        load_sealed_oos_campaign(contract_path),
        item.symbol,
    ).exists()


def test_create_contract_freezes_exactly_50_sorted_unique_items(tmp_path) -> None:
    items = list(reversed(_items()))
    contract_path, _ = _create_campaign(tmp_path, items)

    contract = load_sealed_oos_campaign(contract_path)

    assert contract["symbol_count"] == 50
    assert [row["symbol"] for row in contract["items"]] == [
        f"{index:06d}" for index in range(1, 51)
    ]
    assert len(contract["contract_sha256"]) == 64
    assert len(contract["sealed_dataset_sha256"]) == 64
    assert contract["costs"] == {
        "commission_pct": COMMISSION_PCT,
        "slippage_pct": SLIPPAGE_PCT,
        "cost_rate": COST_RATE,
    }
    assert contract["result_path"] == "campaign.result.json"
    with pytest.raises(ResultAlreadyExistsError):
        create_sealed_oos_campaign(
            contract_path,
            campaign_id="csi-a50-seal-2025",
            items=items,
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )


@pytest.mark.parametrize(
    "items, message",
    [
        (_items()[:-1], "恰好包含 50"),
        (_items() + [_items()[0]], "恰好包含 50"),
        (_items()[:-1] + [_items()[0]], "必须唯一"),
    ],
)
def test_contract_rejects_wrong_count_or_duplicate_symbols(
    tmp_path, items, message
) -> None:
    with pytest.raises(ContractValidationError, match=message):
        create_sealed_oos_campaign(
            tmp_path / "campaign.json",
            campaign_id="invalid",
            items=items,
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )


def test_contract_rejects_invalid_identity_and_window(tmp_path) -> None:
    items = _items()
    items[0] = replace(
        items[0],
        symbol="60051",
        data_sha256="A" * 64,
        test_start=TEST_END,
        test_end=TEST_START,
    )
    with pytest.raises(ContractValidationError, match="6 位数字"):
        create_sealed_oos_campaign(
            tmp_path / "campaign.json",
            campaign_id="invalid",
            items=items,
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )


def test_contract_rejects_zero_total_cost(tmp_path) -> None:
    with pytest.raises(ContractValidationError, match="严格大于 0"):
        create_sealed_oos_campaign(
            tmp_path / "campaign.json",
            campaign_id="zero-cost",
            items=_items(),
            commission_pct=0.0,
            slippage_pct=0.0,
        )


def test_contract_creation_rejects_inconsistent_test_windows(tmp_path) -> None:
    items = _items()
    items[17] = replace(
        items[17],
        test_start="2025-01-03T07:00:00Z",
    )

    with pytest.raises(ContractValidationError, match="完全一致"):
        create_sealed_oos_campaign(
            tmp_path / "campaign.json",
            campaign_id="inconsistent-window",
            items=items,
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )


def test_contract_loading_rejects_inconsistent_test_windows(tmp_path) -> None:
    contract_path, items = _create_campaign(tmp_path)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["items"][17]["test_start"] = "2025-01-03T07:00:00Z"
    body = {
        key: value
        for key, value in payload.items()
        if key != "contract_sha256"
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractValidationError, match="完全一致"):
        load_sealed_oos_campaign(contract_path)


def test_contract_hash_detects_tampering(tmp_path) -> None:
    contract_path, _ = _create_campaign(tmp_path)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["items"][0]["data_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ContractValidationError,
        match="sealed_dataset_sha256|contract_sha256",
    ):
        load_sealed_oos_campaign(contract_path)


def test_all_50_strictly_above_one_pass_and_result_is_machine_readable(
    tmp_path,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(
        contract_path,
        items,
        sharpe=1.25,
        sharpe_overrides={items[0].symbol: 1.01},
    )

    result = evaluate_sealed_oos_campaign(contract_path)
    persisted = json.loads(
        (tmp_path / "campaign.result.json").read_text(encoding="utf-8")
    )

    assert result == persisted
    assert result["status"] == "PASS"
    assert result["pass_count"] == 50
    assert result["minimum_sharpe"] == 1.01
    assert len(result["results"]) == 50
    assert all(row["status"] == "PASS" for row in result["results"])
    assert all(len(row["report_sha256"]) == 64 for row in result["results"])
    assert result["contract_sha256"] == load_sealed_oos_campaign(contract_path)[
        "contract_sha256"
    ]
    contract = load_sealed_oos_campaign(contract_path)
    lock = json.loads(_reveal_lock_path(contract).read_text(encoding="utf-8"))
    assert lock == {
        "format": "alphamaster_sealed_oos_reveal_lock_v3",
        "campaign_id": "csi-a50-seal-2025",
        "contract_sha256": result["contract_sha256"],
        "sealed_dataset_sha256": contract["sealed_dataset_sha256"],
        "status": "REVEAL_STARTED",
    }


def test_sharpe_equal_to_one_fails_entire_batch_and_seal_cannot_be_reused(
    tmp_path,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(
        contract_path,
        items,
        sharpe=1.5,
        sharpe_overrides={items[7].symbol: 1.0},
    )
    failed_path = tmp_path / str(items[7].report_path)

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 49
    assert result["minimum_sharpe"] == 1.0
    assert result["results"][7]["failure_codes"] == ["sharpe_not_above_1"]

    failed_path.write_text(
        json.dumps(
            _report(
                items[7],
                load_sealed_oos_campaign(contract_path),
                sharpe=9.0,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResultAlreadyExistsError, match="禁止覆盖|禁止.*重测"):
        evaluate_sealed_oos_campaign(contract_path)


def test_existing_matching_reveal_lock_allows_same_campaign_to_continue(
    tmp_path,
    monkeypatch,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    contract = load_sealed_oos_campaign(contract_path)
    lock_path = _reveal_lock_path(contract)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "format": "alphamaster_sealed_oos_reveal_lock_v3",
                "campaign_id": contract["campaign_id"],
                "contract_sha256": contract["contract_sha256"],
                "sealed_dataset_sha256": contract["sealed_dataset_sha256"],
                "status": "REVEAL_STARTED",
            }
        ),
        encoding="utf-8",
    )
    _write_reports(contract_path, items)
    result = evaluate_sealed_oos_campaign(contract_path)
    assert result["status"] == "PASS"


def test_concurrent_same_contract_has_one_atomic_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    original_link = sealed_oos_campaign.os.link
    successes = 0
    guard = threading.Lock()

    def counted_link(source, target):
        nonlocal successes
        result = original_link(source, target)
        with guard:
            successes += 1
        return result

    monkeypatch.setattr(sealed_oos_campaign.os, "link", counted_link)

    def authorize(index: int) -> dict[str, object]:
        item = items[index]
        return sealed_oos_campaign._begin_controlled_sealed_oos_run(
            contract_path,
            symbol=item.symbol,
            data_sha256=item.data_sha256,
            strategy_sha256=item.strategy_sha256,
            report_path=tmp_path / str(item.report_path),
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        authorizations = list(pool.map(authorize, range(24)))

    assert successes == 1
    assert {row["contract_sha256"] for row in authorizations} == {
        load_sealed_oos_campaign(contract_path)["contract_sha256"]
    }


def test_handwritten_self_consistent_reports_without_receipts_are_rejected(
    tmp_path: Path,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    contract = load_sealed_oos_campaign(contract_path)
    for item in items:
        path = tmp_path / str(item.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_report(item, contract), ensure_ascii=False),
            encoding="utf-8",
        )

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 0
    assert all(
        "authorization_receipt_missing" in row["failure_codes"]
        for row in result["results"]
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "universe",
        "split",
        "data_manifest",
        "training_result_manifest",
        "published_strategy",
    ],
)
def test_provenance_artifact_tampering_blocks_report_registration(
    tmp_path: Path,
    artifact: str,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    contract = load_sealed_oos_campaign(contract_path)
    item = items[0]
    paths = {
        "universe": tmp_path / str(contract["universe_contract_path"]),
        "split": tmp_path / str(contract["split_contract_path"]),
        "data_manifest": tmp_path / str(item.data_manifest_path),
        "training_result_manifest": (
            tmp_path / str(item.training_result_manifest_path)
        ),
        "published_strategy": (
            tmp_path / str(item.published_strategy_path)
        ),
    }
    path = paths[artifact]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tampered"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractValidationError, match="身份|冻结"):
        sealed_oos_campaign._begin_controlled_sealed_oos_run(
            contract_path,
            symbol=item.symbol,
            data_sha256=item.data_sha256,
            strategy_sha256=item.strategy_sha256,
            report_path=tmp_path / str(item.report_path),
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
        )


def test_crash_after_atomic_claim_can_only_resume_same_frozen_campaign(
    tmp_path,
    monkeypatch,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(contract_path, items)
    original_reader = sealed_oos_campaign._read_and_evaluate_report

    def crash_on_first_report(**_):
        raise RuntimeError("simulated evaluator crash")

    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        crash_on_first_report,
    )
    with pytest.raises(RuntimeError, match="simulated evaluator crash"):
        evaluate_sealed_oos_campaign(contract_path)

    contract = load_sealed_oos_campaign(contract_path)
    lock_path = _reveal_lock_path(contract)
    assert lock_path.exists()
    assert not (tmp_path / "campaign.result.json").exists()

    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        original_reader,
    )
    result = evaluate_sealed_oos_campaign(contract_path)
    assert result["status"] == "PASS"


def test_same_sealed_dataset_cannot_be_revealed_via_new_contract_or_strategy(
    tmp_path,
    monkeypatch,
) -> None:
    first_contract_path, first_items = _create_campaign(tmp_path / "first")
    second_items = [
        replace(
            item,
            strategy_sha256=_hash("replacement-strategy", index),
            published_strategy_sha256=_hash(
                "replacement-strategy", index
            ),
            report_path=f"replacement-reports/{item.symbol}.json",
        )
        for index, item in enumerate(first_items, start=1)
    ]
    second_contract_path = tmp_path / "second" / "different-name.json"
    create_sealed_oos_campaign(
        second_contract_path,
        campaign_id="replacement-campaign",
        items=second_items,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        result_path="different-result.json",
    )
    first_contract = load_sealed_oos_campaign(first_contract_path)
    second_contract = load_sealed_oos_campaign(second_contract_path)
    assert first_contract["contract_sha256"] != second_contract["contract_sha256"]
    assert (
        first_contract["sealed_dataset_sha256"]
        == second_contract["sealed_dataset_sha256"]
    )

    _write_reports(first_contract_path, first_items, sharpe=1.5)
    evaluate_sealed_oos_campaign(first_contract_path)
    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        lambda **_: pytest.fail("全局锁必须在读取替换策略报告前拒绝"),
    )

    with pytest.raises(
        RevealAlreadyStartedError,
        match="绑定其他合同|禁止换合同",
    ):
        evaluate_sealed_oos_campaign(second_contract_path)


def test_same_sealed_bytes_cannot_be_revealed_via_new_symbols_or_window(
    tmp_path,
    monkeypatch,
) -> None:
    first_contract_path, first_items = _create_campaign(tmp_path / "first")
    reversed_hashes = [
        item.data_sha256 for item in reversed(first_items)
    ]
    second_items = [
        replace(
            first_items[index - 1],
            symbol=f"{100000 + index:06d}",
            data_sha256=reversed_hashes[index - 1],
            strategy_sha256=_hash("replacement-strategy", index),
            published_strategy_sha256=_hash(
                "replacement-strategy", index
            ),
            test_start="2025-02-03T07:00:00Z",
            test_end="2026-02-02T07:00:00Z",
            report_path=f"replacement-reports/{100000 + index:06d}.json",
        )
        for index in range(1, 51)
    ]
    second_contract_path = tmp_path / "second" / "different-window.json"
    create_sealed_oos_campaign(
        second_contract_path,
        campaign_id="replacement-labels-and-window",
        items=second_items,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        result_path="different-result.json",
    )
    first_contract = load_sealed_oos_campaign(first_contract_path)
    second_contract = load_sealed_oos_campaign(second_contract_path)
    assert first_contract["contract_sha256"] != second_contract["contract_sha256"]
    assert (
        first_contract["sealed_dataset_sha256"]
        == second_contract["sealed_dataset_sha256"]
    )

    _write_reports(first_contract_path, first_items, sharpe=1.5)
    evaluate_sealed_oos_campaign(first_contract_path)
    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        lambda **_: pytest.fail(
            "同一批密封文件不能通过改标签或评分窗口再次读取"
        ),
    )

    with pytest.raises(
        RevealAlreadyStartedError,
        match="绑定其他合同|禁止换合同",
    ):
        evaluate_sealed_oos_campaign(second_contract_path)


def test_missing_and_nonfinite_reports_fail_without_short_circuiting(tmp_path) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(contract_path, items, sharpe=1.5)
    (tmp_path / str(items[0].report_path)).unlink()
    (tmp_path / str(items[1].report_path)).write_text(
        json.dumps(
            _report(
                items[1],
                load_sealed_oos_campaign(contract_path),
                sharpe=1.5,
            )
        ).replace("1.5", "NaN"),
        encoding="utf-8",
    )

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 48
    assert result["minimum_sharpe"] is None
    assert len(result["results"]) == 50
    assert result["results"][0]["failure_codes"] == ["report_missing"]
    assert result["results"][1]["failure_codes"] == [
        "authorization_receipt_mismatch",
        "report_invalid_json",
    ]
    assert result["results"][-1]["status"] == "PASS"


@pytest.mark.parametrize(
    "field, bad_value, failure_code",
    [
        ("format", "alphamaster_sealed_oos_report_v1", "format_mismatch"),
        ("symbol", "600519", "symbol_mismatch"),
        ("data_sha256", "0" * 64, "data_sha256_mismatch"),
        (
            "data_manifest_sha256",
            "0" * 64,
            "data_manifest_sha256_mismatch",
        ),
        ("strategy_sha256", "0" * 64, "strategy_sha256_mismatch"),
        (
            "published_strategy_sha256",
            "0" * 64,
            "published_strategy_sha256_mismatch",
        ),
        ("training_run_id", "run_bad", "training_run_id_mismatch"),
        (
            "training_result_manifest_sha256",
            "0" * 64,
            "training_result_manifest_sha256_mismatch",
        ),
        (
            "runtime_git_commit",
            "0" * 40,
            "runtime_git_commit_mismatch",
        ),
        (
            "scoring_contract_version",
            "previous",
            "scoring_contract_version_mismatch",
        ),
        ("campaign_id", "other", "campaign_id_mismatch"),
        ("contract_sha256", "0" * 64, "contract_sha256_mismatch"),
        (
            "sealed_dataset_sha256",
            "0" * 64,
            "sealed_dataset_sha256_mismatch",
        ),
        (
            "split_contract_sha256",
            "0" * 64,
            "split_contract_sha256_mismatch",
        ),
        (
            "universe_contract_sha256",
            "0" * 64,
            "universe_contract_sha256_mismatch",
        ),
        ("evaluation_mode", "replay", "evaluation_mode_mismatch"),
        ("test_start", "2025-01-03T07:00:00Z", "test_start_mismatch"),
        ("test_end", "2026-01-01T07:00:00Z", "test_end_mismatch"),
        ("commission_pct", 0.01, "commission_pct_mismatch"),
        ("slippage_pct", 0.02, "slippage_pct_mismatch"),
        ("cost_rate", 0.0, "cost_rate_mismatch"),
    ],
)
def test_any_report_identity_mismatch_fails_entire_batch(
    tmp_path, field, bad_value, failure_code
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(contract_path, items, sharpe=1.5)
    path = tmp_path / str(items[10].report_path)
    report = _report(
        items[10],
        load_sealed_oos_campaign(contract_path),
        sharpe=1.5,
    )
    report[field] = bad_value
    path.write_text(json.dumps(report), encoding="utf-8")

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 49
    assert failure_code in result["results"][10]["failure_codes"]


@pytest.mark.parametrize("sharpe", [float("inf"), float("-inf")])
def test_infinite_sharpe_fails(tmp_path, sharpe) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(contract_path, items, sharpe=1.5)
    path = tmp_path / str(items[4].report_path)
    path.write_text(
        json.dumps(
            _report(
                items[4],
                load_sealed_oos_campaign(contract_path),
                sharpe=sharpe,
            )
        ),
        encoding="utf-8",
    )

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 49
    assert result["results"][4]["failure_codes"] == [
        "authorization_receipt_mismatch",
        "report_invalid_json",
    ]
