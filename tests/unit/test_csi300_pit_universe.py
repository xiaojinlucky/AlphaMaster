from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import portfolio_manager.controller as controller_module
import portfolio_manager.universe as universe_module
from portfolio_manager.controller import ModelSignalSnapshot, PortfolioPolicy
from portfolio_manager.execution import (
    AShareFeeSchedule,
    ExecutionQuote,
    VirtualAccount,
    account_snapshot_sha256,
    execute_portfolio_decision,
)
from portfolio_manager.ledger import PortfolioDecisionLedger
from portfolio_manager.universe import (
    HistoricalUniverseContract,
    UNIVERSE_CONTRACT_TYPE_HISTORICAL,
    UNIVERSE_CONTRACT_TYPE_TRUSTED_STATIC,
    UNIVERSE_QUERY_MODE_RECONSTRUCTED,
    UNIVERSE_QUERY_MODE_STATIC,
    UNIVERSE_QUERY_MODE_STRICT,
    UniverseAvailabilityError,
    UniverseContract,
    load_csi300_historical_universe_contract,
    load_csi_a50_universe_contract,
)

_FORMAT = "free_stockdb_csi300_weight_history_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_month(
    root: Path,
    *,
    requested_date: str,
    effective_date: str,
    symbols: tuple[str, ...],
    strict_available_at: str | None = None,
) -> dict:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    stem = requested_date.replace("-", "")
    data_path = root / "data" / f"{stem}.parquet"
    receipt_path = root / "manifests" / f"{stem}.json"
    weight = 100.0 / len(symbols)
    frame = pd.DataFrame(
        {
            "code": list(symbols),
            "date": [effective_date] * len(symbols),
            "weight": [weight] * len(symbols),
            "display_name": [f"股票{symbol}" for symbol in symbols],
        }
    )
    frame.to_parquet(data_path, index=False)
    receipt = {
        "format": _FORMAT,
        "requested_date": requested_date,
        "actual_weight_date": effective_date,
        "rows": len(frame),
        "weight_sum": float(frame["weight"].sum()),
        "data_file": data_path.name,
        "data_bytes": data_path.stat().st_size,
        "data_sha256": _sha256(data_path),
        "captured_at_utc": "2026-07-25T04:00:00Z",
    }
    if strict_available_at is not None:
        receipt.update(
            {
                "observed_at": "2020-01-30T08:00:00Z",
                "receipt_at": "2020-01-30T08:01:00Z",
                "strict_available_at": strict_available_at,
            }
        )
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def _publish_history(root: Path, items: list[dict]) -> None:
    if not items:
        raise ValueError("测试历史不能为空")
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    combined_path = root / "csi300_weight_history.parquet"
    combined = pd.concat(
        [
            pd.read_parquet(root / "data" / str(item["data_file"]))
            for item in items
        ],
        ignore_index=True,
    )
    combined.to_parquet(combined_path, index=False)
    requested_dates = [str(item["requested_date"]) for item in items]
    effective_dates = [str(item["actual_weight_date"]) for item in items]
    manifest = {
        "format": _FORMAT,
        "status": "completed",
        "endpoint": "test.invalid:1",
        "sdk_file": "test/stockdb.pyd",
        "sdk_sha256": "a" * 64,
        "index": "000300.XSHG",
        "request_count": len(items),
        "successful_api_calls": len(items),
        "first_requested_date": min(requested_dates),
        "last_requested_date": max(requested_dates),
        "first_actual_weight_date": min(effective_dates),
        "last_actual_weight_date": max(effective_dates),
        "unique_actual_weight_dates": len(set(effective_dates)),
        "total_rows": len(combined),
        "combined_file": combined_path.name,
        "combined_bytes": combined_path.stat().st_size,
        "combined_sha256": _sha256(combined_path),
        "items": items,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _register_test_root(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    policy: str = "test_csi300_history_roots",
) -> str:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = (
        "format",
        "status",
        "endpoint",
        "sdk_sha256",
        "index",
        "request_count",
        "successful_api_calls",
        "first_requested_date",
        "last_requested_date",
        "first_actual_weight_date",
        "last_actual_weight_date",
        "unique_actual_weight_dates",
        "total_rows",
        "combined_file",
        "combined_bytes",
        "combined_sha256",
    )
    policies = deepcopy(universe_module._CSI300_HISTORY_TRUSTED_ROOTS)
    roots = policies.setdefault(policy, {})
    roots[_sha256(manifest_path)] = {
        key: manifest[key] for key in expected_keys
    }
    monkeypatch.setattr(
        universe_module,
        "_CSI300_HISTORY_TRUSTED_ROOTS",
        policies,
    )
    return policy


def test_reconstructed_query_is_explicitly_not_pit_or_sealed_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002", "000003"),
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)

    selected = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-02-15",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    assert isinstance(selected, HistoricalUniverseContract)
    assert selected.symbols == ("000001", "000002", "000003")
    assert selected.source_effective_date == "2020-01-31"
    assert selected.observed_at == "2026-07-25T04:00:00Z"
    assert selected.receipt_at == "2026-07-25T04:00:00Z"
    assert selected.strict_available_at is None
    assert selected.reconstructed is True
    assert selected.point_in_time_safe is False
    assert selected.sealed_oos_eligible is False


def test_historical_contract_serialization_round_trips_canonical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)
    selected = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-02-15",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )
    payload = selected.to_dict()

    assert payload["contract_type"] == "historical"
    assert (
        payload["selection_format"]
        == "alphamaster_historical_universe_selection_v1"
    )
    rebuilt = HistoricalUniverseContract(
        universe_id=payload["universe_id"],
        snapshot_date=payload["snapshot_date"],
        constituent_count=payload["constituent_count"],
        universe_sha256=payload["universe_sha256"],
        symbols=tuple(payload["symbols"]),
        contract_type=payload["contract_type"],
        query_mode=payload["query_mode"],
        point_in_time_safe=payload["point_in_time_safe"],
        sealed_oos_eligible=payload["sealed_oos_eligible"],
        provenance_identity=payload["provenance_identity"],
        contract_sha256=payload["contract_sha256"],
        as_of_date=payload["as_of_date"],
        source_trust_policy=payload["source_trust_policy"],
        source_effective_date=payload["source_effective_date"],
        source_effective_until_exclusive=payload[
            "source_effective_until_exclusive"
        ],
        observed_at=payload["observed_at"],
        receipt_at=payload["receipt_at"],
        strict_available_at=payload["strict_available_at"],
        reconstructed=payload["reconstructed"],
        source_data_sha256=payload["source_data_sha256"],
        source_receipt_sha256=payload["source_receipt_sha256"],
        constituents=selected.constituents,
        source_history_root=payload["source_history_root"],
        query_at=payload["query_at"],
    )
    rebuilt.validate_contract_identity()
    assert rebuilt.to_dict() == payload


def test_strict_query_rejects_receipt_without_historical_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)

    with pytest.raises(
        UniverseAvailabilityError,
        match="没有 strict_available_at",
    ):
        load_csi300_historical_universe_contract(
            root,
            as_of_date="2020-01-31",
            mode=UNIVERSE_QUERY_MODE_STRICT,
            query_at="2020-01-31T15:00:00+08:00",
            trust_policy=trust_policy,
        )


def test_strict_query_uses_only_explicit_strict_available_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
        strict_available_at="2020-01-30T08:02:00Z",
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)

    with pytest.raises(
        UniverseAvailabilityError,
        match="尚无严格可知证据",
    ):
        load_csi300_historical_universe_contract(
            root,
            as_of_date="2020-01-31",
            mode=UNIVERSE_QUERY_MODE_STRICT,
            query_at="2020-01-30T08:01:59Z",
            trust_policy=trust_policy,
        )

    selected = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-01-31",
        mode=UNIVERSE_QUERY_MODE_STRICT,
        query_at="2020-01-30T08:02:00Z",
        trust_policy=trust_policy,
    )
    assert selected.reconstructed is False
    assert selected.point_in_time_safe is True
    assert selected.sealed_oos_eligible is True
    assert selected.query_at == "2020-01-30T08:02:00Z"
    selected.validate_contract_identity()


def test_appending_future_month_does_not_change_historical_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)
    before = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-02-15",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    february = _publish_month(
        root,
        requested_date="2020-02-29",
        effective_date="2020-02-28",
        symbols=("000002", "000003"),
    )
    march = _publish_month(
        root,
        requested_date="2020-03-31",
        effective_date="2020-03-31",
        symbols=("000003", "000004"),
    )
    _publish_history(root, [january, february, march])
    trust_policy = _register_test_root(monkeypatch, root)
    after = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-02-15",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    assert after.to_dict() == before.to_dict()


def test_coordinated_manifest_receipt_and_parquet_tamper_rejects_untrusted_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    original = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
    )
    _publish_history(root, [original])
    trust_policy = _register_test_root(monkeypatch, root)
    load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-01-31",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    forged = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000003", "000004"),
    )
    _publish_history(root, [forged])

    with pytest.raises(
        UniverseAvailabilityError,
        match="根 manifest SHA-256 不在代码受信锚",
    ):
        load_csi300_historical_universe_contract(
            root,
            as_of_date="2020-01-31",
            mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
            trust_policy=trust_policy,
        )


def test_historical_contract_cannot_be_stripped_to_plain_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)
    selected = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-01-31",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    assert isinstance(selected, UniverseContract)
    assert not hasattr(selected, "universe")
    assert selected.to_dict()["mode"] == UNIVERSE_QUERY_MODE_RECONSTRUCTED
    assert selected.to_dict()["point_in_time_safe"] is False
    assert selected.to_dict()["sealed_oos_eligible"] is False
    with pytest.raises(ValueError, match="不可剥离"):
        UniverseContract(
            universe_id="renamed-csi300-history",
            snapshot_date=selected.snapshot_date,
            constituent_count=selected.constituent_count,
            universe_sha256=selected.universe_sha256,
            symbols=selected.symbols,
            contract_type=selected.contract_type,
            query_mode=selected.query_mode,
            point_in_time_safe=selected.point_in_time_safe,
            sealed_oos_eligible=selected.sealed_oos_eligible,
            provenance_identity=selected.provenance_identity,
            contract_sha256=selected.contract_sha256,
        )
    with pytest.raises(ValueError, match="canonical 合同内容不一致"):
        UniverseContract(
            universe_id="renamed-csi300-history",
            snapshot_date=selected.snapshot_date,
            constituent_count=selected.constituent_count,
            universe_sha256=selected.universe_sha256,
            symbols=selected.symbols,
            contract_sha256=selected.contract_sha256,
        )


def test_controller_and_execution_ledger_preserve_historical_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)
    historical = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-02-14",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )
    bar_ts = int(
        datetime(
            2020,
            2,
            14,
            15,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ).timestamp()
    )
    signals = [
        ModelSignalSnapshot(
            run_id=(
                "run_20200214T070000Z_"
                + hashlib.sha256(symbol.encode()).hexdigest()[:8]
            ),
            symbol=symbol,
            bar_ts=bar_ts,
            session_date="2020-02-14",
            timeframe="1d",
            market_source="ashare_test",
            raw_score=2.0 - index,
            requested_exposure=0.5,
            confidence=0.8,
            model_version=hashlib.sha256(f"model:{symbol}".encode()).hexdigest(),
            data_version=hashlib.sha256(f"data:{symbol}".encode()).hexdigest(),
            calibration_version="test_calibration_v1",
            calibration_history_sha256=hashlib.sha256(
                f"history:{symbol}".encode()
            ).hexdigest(),
            history_scores=(0.0, 1.0, 2.0),
        )
        for index, symbol in enumerate(historical.symbols)
    ]
    account = VirtualAccount(cash=100_000.0)
    decision = controller_module._build_portfolio_decision(
        signals,
        universe=historical,
        current_weights={},
        account_snapshot_sha256=account_snapshot_sha256(account, ()),
        policy=PortfolioPolicy(
            top_k=1,
            dropout_rank=2,
            minimum_history=3,
        ),
    )

    serialized = decision.to_dict()["universe"]
    assert serialized["mode"] == UNIVERSE_QUERY_MODE_RECONSTRUCTED
    assert serialized["point_in_time_safe"] is False
    assert serialized["sealed_oos_eligible"] is False
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    stored_decision, created = ledger.record_decision(decision)
    assert created is True
    assert stored_decision["universe"] == serialized
    execution_quotes = tuple(
        ExecutionQuote(
            symbol=symbol,
            session_date="2020-02-17",
            price=10.0 + index,
            status="OPEN",
        )
        for index, symbol in enumerate(historical.symbols)
    )
    fees = AShareFeeSchedule(
        commission_rate=0.0003,
        minimum_commission=5.0,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.001,
    )
    result = execute_portfolio_decision(
        decision,
        execution_session="2020-02-17",
        account=account,
        decision_quotes=(),
        quotes=execution_quotes,
        fee_schedule=fees,
    )
    stored_execution, created = ledger.record_execution(
        result,
        decision_quotes=(),
        execution_quotes=execution_quotes,
        fee_schedule=fees,
        bootstrap_account=True,
    )
    assert created is True
    assert stored_execution["result"] == result.to_dict()

    restarted = PortfolioDecisionLedger(db)
    restored_decision = restarted.get_decision(decision.decision_id)
    assert restored_decision is not None
    assert restored_decision["universe"]["query_mode"] == (
        UNIVERSE_QUERY_MODE_RECONSTRUCTED
    )
    assert restored_decision["universe"]["point_in_time_safe"] is False
    assert restored_decision["universe"]["sealed_oos_eligible"] is False
    assert restarted.get_execution(result.execution_id) == stored_execution

    renamed_plain = UniverseContract(
        universe_id="renamed-csi300-history",
        snapshot_date=historical.snapshot_date,
        constituent_count=historical.constituent_count,
        universe_sha256=historical.universe_sha256,
        symbols=historical.symbols,
    )
    washed_decision = controller_module._build_portfolio_decision(
        signals,
        universe=renamed_plain,
        current_weights={},
        account_snapshot_sha256="a" * 64,
        policy=PortfolioPolicy(
            top_k=1,
            dropout_rank=2,
            minimum_history=3,
        ),
    )
    washed_universe = washed_decision.to_dict()["universe"]
    assert washed_universe["contract_type"] == "untrusted"
    assert washed_universe["query_mode"] == "untrusted"
    assert washed_universe["point_in_time_safe"] is False
    assert washed_universe["sealed_oos_eligible"] is False
    with pytest.raises(RuntimeError, match="untrusted universe"):
        PortfolioDecisionLedger(
            tmp_path / "washed.sqlite3"
        ).record_decision(washed_decision)


def test_historical_contract_cannot_upgrade_reconstructed_gates_with_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
        strict_available_at="2020-01-30T08:02:00Z",
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)
    reconstructed = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-01-31",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    with pytest.raises(ValueError):
        replace(
            reconstructed,
            query_mode=UNIVERSE_QUERY_MODE_STRICT,
            point_in_time_safe=True,
            sealed_oos_eligible=True,
            reconstructed=False,
            contract_sha256=None,
        )

    forged = replace(reconstructed)
    object.__setattr__(forged, "query_mode", UNIVERSE_QUERY_MODE_STRICT)
    object.__setattr__(forged, "point_in_time_safe", True)
    object.__setattr__(forged, "sealed_oos_eligible", True)
    object.__setattr__(forged, "reconstructed", False)
    object.__setattr__(forged, "contract_sha256", None)
    forged._validate_contract_sha256()
    with pytest.raises(UniverseAvailabilityError, match="query_at"):
        forged.validate_contract_identity()
    with pytest.raises(UniverseAvailabilityError, match="query_at"):
        controller_module._build_portfolio_decision(
            [],
            universe=forged,
            current_weights={},
            account_snapshot_sha256="a" * 64,
            policy=PortfolioPolicy(top_k=1, dropout_rank=2),
        )


def test_historical_contract_cannot_self_report_strict_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    january = _publish_month(
        root,
        requested_date="2020-01-31",
        effective_date="2020-01-31",
        symbols=("000001", "000002"),
        strict_available_at="2020-01-30T08:02:00Z",
    )
    _publish_history(root, [january])
    trust_policy = _register_test_root(monkeypatch, root)
    reconstructed = load_csi300_historical_universe_contract(
        root,
        as_of_date="2020-01-31",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )

    with pytest.raises(ValueError):
        HistoricalUniverseContract(
            universe_id=reconstructed.universe_id.replace(
                UNIVERSE_QUERY_MODE_RECONSTRUCTED,
                UNIVERSE_QUERY_MODE_STRICT,
            ),
            snapshot_date=reconstructed.snapshot_date,
            constituent_count=reconstructed.constituent_count,
            universe_sha256=reconstructed.universe_sha256,
            symbols=reconstructed.symbols,
            contract_type=reconstructed.contract_type,
            query_mode=UNIVERSE_QUERY_MODE_STRICT,
            point_in_time_safe=True,
            sealed_oos_eligible=True,
            provenance_identity=reconstructed.provenance_identity,
            contract_sha256=None,
            as_of_date=reconstructed.as_of_date,
            source_trust_policy=reconstructed.source_trust_policy,
            source_effective_date=reconstructed.source_effective_date,
            source_effective_until_exclusive=(
                reconstructed.source_effective_until_exclusive
            ),
            observed_at=reconstructed.observed_at,
            receipt_at=reconstructed.receipt_at,
            strict_available_at=reconstructed.strict_available_at,
            reconstructed=False,
            source_data_sha256=reconstructed.source_data_sha256,
            source_receipt_sha256=reconstructed.source_receipt_sha256,
            constituents=reconstructed.constituents,
        )


def test_re_signed_plain_contract_cannot_upgrade_to_historical() -> None:
    forged = UniverseContract(
        universe_id="renamed-csi300-history",
        snapshot_date="20200131",
        constituent_count=2,
        universe_sha256="a" * 64,
        symbols=("000001", "000002"),
    )
    object.__setattr__(
        forged,
        "contract_type",
        UNIVERSE_CONTRACT_TYPE_HISTORICAL,
    )
    object.__setattr__(forged, "query_mode", UNIVERSE_QUERY_MODE_STRICT)
    object.__setattr__(forged, "point_in_time_safe", True)
    object.__setattr__(forged, "sealed_oos_eligible", True)
    object.__setattr__(forged, "provenance_identity", "b" * 64)
    object.__setattr__(forged, "contract_sha256", None)
    forged._validate_contract_sha256()

    with pytest.raises(ValueError, match="历史股票池必须"):
        forged.validate_contract_identity()
    with pytest.raises(ValueError, match="历史股票池必须"):
        controller_module._build_portfolio_decision(
            [],
            universe=forged,
            current_weights={},
            account_snapshot_sha256="a" * 64,
            policy=PortfolioPolicy(top_k=1, dropout_rank=2),
        )


def test_re_signed_plain_contract_cannot_forge_trusted_static() -> None:
    a50_hash = (
        "987387945fba0cb778b648860bc7579a3cf49e9c3b788596464f714e968bb896"
    )
    forged = UniverseContract(
        universe_id="renamed-static",
        snapshot_date="20260723",
        constituent_count=2,
        universe_sha256=a50_hash,
        symbols=("000001", "000002"),
    )
    object.__setattr__(
        forged,
        "contract_type",
        UNIVERSE_CONTRACT_TYPE_TRUSTED_STATIC,
    )
    object.__setattr__(forged, "query_mode", UNIVERSE_QUERY_MODE_STATIC)
    object.__setattr__(forged, "point_in_time_safe", True)
    object.__setattr__(forged, "sealed_oos_eligible", True)
    object.__setattr__(forged, "provenance_identity", a50_hash)
    object.__setattr__(forged, "contract_sha256", None)
    forged._validate_contract_sha256()

    with pytest.raises(ValueError, match="trusted_static"):
        forged.validate_contract_identity()
    with pytest.raises(ValueError, match="trusted_static"):
        controller_module._build_portfolio_decision(
            [],
            universe=forged,
            current_weights={},
            account_snapshot_sha256="a" * 64,
            policy=PortfolioPolicy(top_k=1, dropout_rank=2),
        )


def test_incomplete_snapshot_blocks_its_whole_validity_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "history"
    december_symbols = tuple(f"{value:06d}" for value in range(1, 299))
    january_symbols = tuple(f"{value:06d}" for value in range(1, 301))
    december = _publish_month(
        root,
        requested_date="2009-12-31",
        effective_date="2009-12-31",
        symbols=december_symbols,
    )
    january = _publish_month(
        root,
        requested_date="2010-01-31",
        effective_date="2010-01-29",
        symbols=january_symbols,
    )
    _publish_history(root, [december, january])
    trust_policy = _register_test_root(monkeypatch, root)
    availability_policies = deepcopy(
        getattr(
            universe_module,
            "_CSI300_HISTORY_AVAILABILITY_POLICIES",
            {},
        )
    )
    availability_policies.setdefault(trust_policy, {})[
        "incomplete_snapshots"
    ] = {
        "2009-12-31": {
            "source_data_sha256": december["data_sha256"],
            "source_receipt_sha256": _sha256(
                root / "manifests" / "20091231.json"
            ),
            "constituent_count": 298,
            "valid_until_exclusive": "2010-01-29",
            "reason": "受信快照只有 298 只成分",
        }
    }
    monkeypatch.setattr(
        universe_module,
        "_CSI300_HISTORY_AVAILABILITY_POLICIES",
        availability_policies,
        raising=False,
    )

    for as_of_date in ("2009-12-31", "2010-01-15"):
        with pytest.raises(
            UniverseAvailabilityError,
            match="不完整",
        ):
            load_csi300_historical_universe_contract(
                root,
                as_of_date=as_of_date,
                mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
                trust_policy=trust_policy,
            )

    recovered = load_csi300_historical_universe_contract(
        root,
        as_of_date="2010-01-29",
        mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
        trust_policy=trust_policy,
    )
    assert recovered.constituent_count == 300
    assert recovered.source_effective_date == "2010-01-29"

    february = _publish_month(
        root,
        requested_date="2010-02-28",
        effective_date="2010-02-26",
        symbols=january_symbols,
    )
    _publish_history(root, [december, january, february])
    trust_policy = _register_test_root(monkeypatch, root)
    with pytest.raises(UniverseAvailabilityError, match="不完整"):
        load_csi300_historical_universe_contract(
            root,
            as_of_date="2010-01-15",
            mode=UNIVERSE_QUERY_MODE_RECONSTRUCTED,
            trust_policy=trust_policy,
        )


def test_a50_trusted_contract_hash_is_unchanged() -> None:
    contract = load_csi_a50_universe_contract()

    assert contract.constituent_count == 50
    assert contract.universe_sha256 == (
        "987387945fba0cb778b648860bc7579a3cf49e9c3b788596464f714e968bb896"
    )
