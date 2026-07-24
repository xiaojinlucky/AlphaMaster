from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import evaluation.sealed_oos_campaign as sealed_oos_campaign
from evaluation.sealed_oos_campaign import (
    ContractValidationError,
    RevealAlreadyStartedError,
    ResultAlreadyExistsError,
    SealedOOSItem,
    create_sealed_oos_campaign,
    evaluate_sealed_oos_campaign,
    load_sealed_oos_campaign,
)


TEST_START = "2025-01-02T07:00:00Z"
TEST_END = "2025-12-31T07:00:00Z"
COMMISSION_PCT = 0.02
SLIPPAGE_PCT = 0.01
COST_RATE = 0.0003


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
            strategy_sha256=_hash("strategy", index),
            test_start=TEST_START,
            test_end=TEST_END,
            report_path=f"reports/{index:06d}.json",
        )
        for index in range(1, 51)
    ]


def _report(item: SealedOOSItem, sharpe: float = 1.5) -> dict[str, object]:
    return {
        "format": "alphamaster_sealed_oos_report_v2",
        "symbol": item.symbol,
        "data_sha256": item.data_sha256,
        "strategy_sha256": item.strategy_sha256,
        "evaluation_mode": "sealed_oos",
        "test_start": item.test_start,
        "test_end": item.test_end,
        "commission_pct": COMMISSION_PCT,
        "slippage_pct": SLIPPAGE_PCT,
        "cost_rate": COST_RATE,
        "sharpe": sharpe,
    }


def _write_reports(root: Path, items: list[SealedOOSItem], sharpe: float = 1.5) -> None:
    for item in items:
        path = root / str(item.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_report(item, sharpe), ensure_ascii=False),
            encoding="utf-8",
        )


def _create_campaign(
    tmp_path: Path,
    items: list[SealedOOSItem] | None = None,
) -> tuple[Path, list[SealedOOSItem]]:
    selected = items or _items()
    contract_path = tmp_path / "campaign.json"
    create_sealed_oos_campaign(
        contract_path,
        campaign_id="csi-a50-seal-2025",
        items=selected,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
    )
    return contract_path, selected


def _reveal_lock_path(contract: dict[str, object]) -> Path:
    return sealed_oos_campaign.REVEAL_REGISTRY_DIR / (
        f"{contract['sealed_dataset_sha256']}.reveal.lock"
    )


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
    items[0] = SealedOOSItem(
        symbol="60051",
        data_sha256="A" * 64,
        strategy_sha256=items[0].strategy_sha256,
        test_start=TEST_END,
        test_end=TEST_START,
        report_path=items[0].report_path,
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
    items[17] = SealedOOSItem(
        symbol=items[17].symbol,
        data_sha256=items[17].data_sha256,
        strategy_sha256=items[17].strategy_sha256,
        test_start="2025-01-03T07:00:00Z",
        test_end=items[17].test_end,
        report_path=items[17].report_path,
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
    contract_path, _ = _create_campaign(tmp_path)
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
    _write_reports(tmp_path, items, sharpe=1.25)
    first_report = tmp_path / str(items[0].report_path)
    first_report.write_text(
        json.dumps(_report(items[0], sharpe=1.01)),
        encoding="utf-8",
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
        "format": "alphamaster_sealed_oos_reveal_lock_v2",
        "campaign_id": "csi-a50-seal-2025",
        "contract_sha256": result["contract_sha256"],
        "sealed_dataset_sha256": contract["sealed_dataset_sha256"],
        "status": "REVEAL_STARTED",
    }


def test_sharpe_equal_to_one_fails_entire_batch_and_seal_cannot_be_reused(
    tmp_path,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(tmp_path, items, sharpe=1.5)
    failed_path = tmp_path / str(items[7].report_path)
    failed_path.write_text(
        json.dumps(_report(items[7], sharpe=1.0)),
        encoding="utf-8",
    )

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 49
    assert result["minimum_sharpe"] == 1.0
    assert result["results"][7]["failure_codes"] == ["sharpe_not_above_1"]

    failed_path.write_text(
        json.dumps(_report(items[7], sharpe=9.0)),
        encoding="utf-8",
    )
    with pytest.raises(ResultAlreadyExistsError, match="禁止覆盖|禁止.*重测"):
        evaluate_sealed_oos_campaign(contract_path)


def test_existing_reveal_lock_without_result_rejects_before_report_read(
    tmp_path,
    monkeypatch,
) -> None:
    contract_path, _ = _create_campaign(tmp_path)
    contract = load_sealed_oos_campaign(contract_path)
    lock_path = _reveal_lock_path(contract)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "format": "alphamaster_sealed_oos_reveal_lock_v2",
                "campaign_id": contract["campaign_id"],
                "contract_sha256": contract["contract_sha256"],
                "sealed_dataset_sha256": contract["sealed_dataset_sha256"],
                "status": "REVEAL_STARTED",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        lambda **_: pytest.fail("已有揭盲锁时不应读取任何报告"),
    )

    with pytest.raises(RevealAlreadyStartedError, match="禁止再次读取"):
        evaluate_sealed_oos_campaign(contract_path)
    assert not (tmp_path / "campaign.result.json").exists()


def test_crash_after_atomic_claim_permanently_consumes_same_seal(
    tmp_path,
    monkeypatch,
) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(tmp_path, items)
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
    with pytest.raises(RevealAlreadyStartedError, match="禁止再次读取"):
        evaluate_sealed_oos_campaign(contract_path)


def test_same_sealed_dataset_cannot_be_revealed_via_new_contract_or_strategy(
    tmp_path,
    monkeypatch,
) -> None:
    first_contract_path, first_items = _create_campaign(tmp_path / "first")
    second_items = [
        SealedOOSItem(
            symbol=item.symbol,
            data_sha256=item.data_sha256,
            strategy_sha256=_hash("replacement-strategy", index),
            test_start=item.test_start,
            test_end=item.test_end,
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

    _write_reports(first_contract_path.parent, first_items, sharpe=1.5)
    evaluate_sealed_oos_campaign(first_contract_path)
    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        lambda **_: pytest.fail("全局锁必须在读取替换策略报告前拒绝"),
    )

    with pytest.raises(RevealAlreadyStartedError, match="禁止再次读取"):
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
        SealedOOSItem(
            symbol=f"{100000 + index:06d}",
            data_sha256=reversed_hashes[index - 1],
            strategy_sha256=_hash("replacement-strategy", index),
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

    _write_reports(first_contract_path.parent, first_items, sharpe=1.5)
    evaluate_sealed_oos_campaign(first_contract_path)
    monkeypatch.setattr(
        sealed_oos_campaign,
        "_read_and_evaluate_report",
        lambda **_: pytest.fail(
            "同一批密封文件不能通过改标签或评分窗口再次读取"
        ),
    )

    with pytest.raises(RevealAlreadyStartedError, match="禁止再次读取"):
        evaluate_sealed_oos_campaign(second_contract_path)


def test_missing_and_nonfinite_reports_fail_without_short_circuiting(tmp_path) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(tmp_path, items, sharpe=1.5)
    (tmp_path / str(items[0].report_path)).unlink()
    (tmp_path / str(items[1].report_path)).write_text(
        json.dumps(_report(items[1], sharpe=1.5)).replace("1.5", "NaN"),
        encoding="utf-8",
    )

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 48
    assert result["minimum_sharpe"] is None
    assert len(result["results"]) == 50
    assert result["results"][0]["failure_codes"] == ["report_missing"]
    assert result["results"][1]["failure_codes"] == ["report_invalid_json"]
    assert result["results"][-1]["status"] == "PASS"


@pytest.mark.parametrize(
    "field, bad_value, failure_code",
    [
        ("format", "alphamaster_sealed_oos_report_v1", "format_mismatch"),
        ("symbol", "600519", "symbol_mismatch"),
        ("data_sha256", "0" * 64, "data_sha256_mismatch"),
        ("strategy_sha256", "0" * 64, "strategy_sha256_mismatch"),
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
    _write_reports(tmp_path, items, sharpe=1.5)
    path = tmp_path / str(items[10].report_path)
    report = _report(items[10], sharpe=1.5)
    report[field] = bad_value
    path.write_text(json.dumps(report), encoding="utf-8")

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 49
    assert failure_code in result["results"][10]["failure_codes"]


@pytest.mark.parametrize("sharpe", [float("inf"), float("-inf")])
def test_infinite_sharpe_fails(tmp_path, sharpe) -> None:
    contract_path, items = _create_campaign(tmp_path)
    _write_reports(tmp_path, items, sharpe=1.5)
    path = tmp_path / str(items[4].report_path)
    path.write_text(
        json.dumps(_report(items[4], sharpe=sharpe)),
        encoding="utf-8",
    )

    result = evaluate_sealed_oos_campaign(contract_path)

    assert result["status"] == "FAIL"
    assert result["pass_count"] == 49
    assert result["results"][4]["failure_codes"] == ["report_invalid_json"]
