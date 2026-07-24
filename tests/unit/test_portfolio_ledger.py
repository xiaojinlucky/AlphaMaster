from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import portfolio_manager.controller as controller_module
from portfolio_manager.controller import (
    ModelSignalSnapshot,
    PortfolioPolicy,
)
from portfolio_manager.ledger import PortfolioDecisionLedger
from portfolio_manager.universe import UniverseContract

SYMBOLS = ("000001", "000002", "000003")
UNIVERSE = UniverseContract(
    universe_id="test-universe:20260723",
    snapshot_date="20260723",
    constituent_count=3,
    universe_sha256="e" * 64,
    symbols=SYMBOLS,
)
POLICY = PortfolioPolicy(
    top_k=2,
    dropout_rank=3,
    minimum_history=3,
)
BASE_TS = 1_784_790_000


def _decision(bar_ts: int, *, model_suffix: str = ""):
    session_date = (
        datetime.fromtimestamp(
            bar_ts,
            tz=ZoneInfo("Asia/Shanghai"),
        )
        .date()
        .isoformat()
    )
    signals = [
        ModelSignalSnapshot(
            run_id=(
                "run_20260723T235959Z_"
                + hashlib.sha256(symbol.encode()).hexdigest()[:8]
            ),
            symbol=symbol,
            bar_ts=bar_ts,
            session_date=session_date,
            timeframe="1d",
            market_source="akshare_sina_hfq_ohlcv",
            raw_score=5.0 - index,
            requested_exposure=0.5,
            confidence=0.8,
            model_version=hashlib.sha256(
                f"model-{symbol}{model_suffix}".encode()
            ).hexdigest(),
            data_version=hashlib.sha256(f"data-{symbol}".encode()).hexdigest(),
            calibration_version="alphamaster_rolling_factor_calibration_v1",
            calibration_history_sha256=hashlib.sha256(
                f"history-{symbol}".encode()
            ).hexdigest(),
            history_scores=(0.0, 1.0, 2.0),
        )
        for index, symbol in enumerate(SYMBOLS)
    ]
    return controller_module._build_portfolio_decision(
        signals,
        universe=UNIVERSE,
        current_weights={},
        account_snapshot_sha256="a" * 64,
        policy=POLICY,
    )


def test_ledger_is_idempotent_and_restores_target_state(tmp_path) -> None:
    db = tmp_path / "portfolio.sqlite3"
    decision = _decision(BASE_TS)
    ledger = PortfolioDecisionLedger(db)

    stored, created = ledger.record_decision(decision)
    assert created is True
    assert stored == decision.to_dict()

    duplicate, created = ledger.record_decision(decision)
    assert created is False
    assert duplicate == decision.to_dict()

    restarted = PortfolioDecisionLedger(db)
    assert restarted.get_decision(decision.decision_id) == decision.to_dict()
    state = restarted.get_target_state()
    assert state is not None
    assert state["decision_id"] == decision.decision_id
    assert state["bar_ts"] == BASE_TS
    assert state["target_weights"] == dict(decision.target_weights)
    assert state["cash_weight"] == pytest.approx(decision.cash_weight)


def test_same_bar_conflict_and_time_reversal_fail_closed(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    first = _decision(BASE_TS + 86_400)
    ledger.record_decision(first)

    conflicting = _decision(BASE_TS + 86_400, model_suffix="-changed")
    assert conflicting.decision_id != first.decision_id
    with pytest.raises(ValueError, match="同一根 K 线"):
        ledger.record_decision(conflicting)

    with pytest.raises(ValueError, match="时间未前进"):
        ledger.record_decision(_decision(BASE_TS))


def test_same_decision_id_cannot_hide_changed_payload(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision = _decision(BASE_TS)
    ledger.record_decision(decision)

    forged = replace(decision, cash_weight=0.25)
    with pytest.raises(RuntimeError, match="不同组合决策内容"):
        ledger.record_decision(forged)


def test_decision_history_is_newest_first_and_bounded(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    first = _decision(BASE_TS)
    second = _decision(BASE_TS + 86_400)
    ledger.record_decision(first)
    ledger.record_decision(second)

    rows = ledger.list_decisions(limit=1)
    assert [row["decision_id"] for row in rows] == [second.decision_id]


def test_corrupt_target_state_cannot_hide_latest_decision(tmp_path) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    ledger.record_decision(_decision(BASE_TS + 86_400))
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE target_portfolio_state
            SET bar_ts = 0, target_weights_json = '{}', cash_weight = 1
            """
        )

    with pytest.raises(RuntimeError, match="最新目标状态"):
        ledger.get_target_state()
    with pytest.raises(ValueError, match="时间未前进"):
        ledger.record_decision(_decision(BASE_TS))


@pytest.mark.parametrize(
    ("payload_mutation", "message"),
    [
        ("not-json", "payload 不是合法 JSON"),
        ({"bar_ts": None}, "bar_ts 必须是整数"),
    ],
)
def test_corrupt_decision_payload_fails_with_runtime_error(
    tmp_path,
    payload_mutation,
    message,
) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    decision = _decision(BASE_TS)
    ledger.record_decision(decision)
    payload_json = (
        payload_mutation
        if isinstance(payload_mutation, str)
        else json.dumps(
            {
                **decision.to_dict(),
                **payload_mutation,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE portfolio_decisions
            SET payload_json = ?
            WHERE decision_id = ?
            """,
            (payload_json, decision.decision_id),
        )

    with pytest.raises(RuntimeError, match=message):
        ledger.get_target_state()


def test_tampered_ranking_cannot_reuse_original_decision_id(tmp_path) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    decision = _decision(BASE_TS)
    ledger.record_decision(decision)
    payload = decision.to_dict()
    payload["ranking"][0]["raw_score"] += 999.0
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE portfolio_decisions
            SET payload_json = ?
            WHERE decision_id = ?
            """,
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                decision.decision_id,
            ),
        )

    with pytest.raises(RuntimeError, match="payload 与 decision_id"):
        ledger.get_decision(decision.decision_id)


def test_recomputed_payload_id_cannot_escape_database_primary_key(tmp_path) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    decision = _decision(BASE_TS)
    ledger.record_decision(decision)
    payload = decision.to_dict()
    payload["ranking"][0]["raw_score"] += 999.0
    identity = dict(payload)
    identity.pop("decision_id")
    payload["decision_id"] = controller_module._canonical_decision_id(identity)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE portfolio_decisions
            SET payload_json = ?
            WHERE decision_id = ?
            """,
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                decision.decision_id,
            ),
        )

    with pytest.raises(RuntimeError, match="账本 decision_id"):
        ledger.get_decision(decision.decision_id)


@pytest.mark.parametrize("field", ["bar_ts", "target_weight"])
def test_decision_readback_rejects_numeric_strings(tmp_path, field) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    decision = _decision(BASE_TS)
    ledger.record_decision(decision)
    payload = decision.to_dict()
    if field == "bar_ts":
        payload["bar_ts"] = str(payload["bar_ts"])
    else:
        symbol = next(iter(payload["target_weights"]))
        payload["target_weights"][symbol] = str(payload["target_weights"][symbol])
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE portfolio_decisions
            SET payload_json = ?
            WHERE decision_id = ?
            """,
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                decision.decision_id,
            ),
        )

    with pytest.raises(RuntimeError, match="必须是"):
        ledger.get_decision(decision.decision_id)


@pytest.mark.parametrize("limit", [True, 1.5, "2"])
def test_list_limit_rejects_implicit_integer_coercion(tmp_path, limit) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    with pytest.raises(ValueError, match="limit"):
        ledger.list_decisions(limit=limit)
