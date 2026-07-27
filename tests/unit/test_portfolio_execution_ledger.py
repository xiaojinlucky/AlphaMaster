from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import portfolio_manager.controller as controller_module
from portfolio_manager.controller import ModelSignalSnapshot, PortfolioPolicy
from portfolio_manager.execution import (
    AShareFeeSchedule,
    ExecutionQuote,
    PortfolioExecutionResult,
    VirtualAccount,
    account_snapshot_sha256,
    execute_portfolio_decision,
)
from portfolio_manager.ledger import PortfolioDecisionLedger
from portfolio_manager.universe import load_csi_a50_universe_contract

UNIVERSE = load_csi_a50_universe_contract()
SYMBOLS = UNIVERSE.symbols
POLICY = PortfolioPolicy(
    top_k=2,
    dropout_rank=3,
    minimum_history=3,
)
FEES = AShareFeeSchedule(
    commission_rate=0.0003,
    minimum_commission=5.0,
    stamp_duty_rate=0.0005,
    transfer_fee_rate=0.00001,
    slippage_rate=0.001,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _bar_ts(session_date: str) -> int:
    return int(
        datetime.fromisoformat(f"{session_date}T15:00:00")
        .replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        .timestamp()
    )


def _prices(offset: float = 0.0) -> dict[str, float]:
    return {
        symbol: 10.0 + offset + index
        for index, symbol in enumerate(SYMBOLS)
    }


def _quotes(
    session_date: str,
    *,
    prices: dict[str, float],
) -> tuple[ExecutionQuote, ...]:
    return tuple(
        ExecutionQuote(
            symbol=symbol,
            session_date=session_date,
            price=prices[symbol],
            status="OPEN",
        )
        for symbol in SYMBOLS
    )


def _decision_quotes(
    account: VirtualAccount,
    session_date: str,
    *,
    prices: dict[str, float],
) -> tuple[ExecutionQuote, ...]:
    held = {lot.symbol for lot in account.lots}
    return tuple(
        quote
        for quote in _quotes(session_date, prices=prices)
        if quote.symbol in held
    )


def _current_weights(
    account: VirtualAccount,
    *,
    prices: dict[str, float],
) -> dict[str, float]:
    shares: dict[str, int] = {}
    for lot in account.lots:
        shares[lot.symbol] = shares.get(lot.symbol, 0) + lot.shares
    nav = account.cash + sum(
        shares[symbol] * prices[symbol] for symbol in shares
    )
    return {
        symbol: shares[symbol] * prices[symbol] / nav
        for symbol in sorted(shares)
    }


def _decision(
    account: VirtualAccount,
    session_date: str = "2026-07-23",
    *,
    decision_prices: dict[str, float] | None = None,
    suffix: str = "a",
):
    prices = decision_prices or _prices()
    bar_ts = _bar_ts(session_date)
    compact_date = session_date.replace("-", "")
    signals = tuple(
        ModelSignalSnapshot(
            run_id=(
                f"run_{compact_date}T235959Z_"
                + _sha(f"{suffix}:{symbol}")[:8]
            ),
            symbol=symbol,
            bar_ts=bar_ts,
            session_date=session_date,
            timeframe="1d",
            market_source="akshare_sina_hfq_ohlcv",
            raw_score=float(len(SYMBOLS) - index),
            requested_exposure=0.8,
            confidence=0.9,
            model_version=_sha(f"model:{suffix}:{symbol}"),
            data_version=_sha(f"data:{symbol}"),
            calibration_version="test-v1",
            calibration_history_sha256=_sha(f"history:{symbol}"),
            history_scores=(0.0, 1.0, 2.0),
        )
        for index, symbol in enumerate(SYMBOLS)
    )
    decision_quotes = _decision_quotes(
        account,
        session_date,
        prices=prices,
    )
    return controller_module._build_portfolio_decision(
        signals,
        universe=UNIVERSE,
        current_weights=_current_weights(account, prices=prices),
        account_snapshot_sha256=account_snapshot_sha256(
            account,
            decision_quotes,
        ),
        policy=POLICY,
    )


def _execution_case(
    *,
    account: VirtualAccount | None = None,
    decision_session: str = "2026-07-23",
    execution_session: str = "2026-07-24",
    decision_prices: dict[str, float] | None = None,
    execution_prices: dict[str, float] | None = None,
    suffix: str = "a",
):
    account_before = account or VirtualAccount(cash=100_000.0)
    bound_decision_prices = decision_prices or _prices()
    bound_execution_prices = execution_prices or _prices(0.25)
    decision = _decision(
        account_before,
        decision_session,
        decision_prices=bound_decision_prices,
        suffix=suffix,
    )
    decision_quotes = _decision_quotes(
        account_before,
        decision_session,
        prices=bound_decision_prices,
    )
    execution_quotes = _quotes(
        execution_session,
        prices=bound_execution_prices,
    )
    result = execute_portfolio_decision(
        decision,
        execution_session=execution_session,
        account=account_before,
        decision_quotes=decision_quotes,
        quotes=execution_quotes,
        fee_schedule=FEES,
    )
    return decision, decision_quotes, execution_quotes, result


def _record_case(
    ledger: PortfolioDecisionLedger,
    result: PortfolioExecutionResult,
    *,
    decision_quotes: tuple[ExecutionQuote, ...],
    execution_quotes: tuple[ExecutionQuote, ...],
    bootstrap_account: bool,
):
    return ledger.record_execution(
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        fee_schedule=FEES,
        bootstrap_account=bootstrap_account,
    )


def test_record_execution_reruns_and_is_idempotent_after_restart(tmp_path) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)

    stored, created = _record_case(
        ledger,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )
    assert created is True
    assert stored["result"] == result.to_dict()
    assert stored["execution_id"] == result.execution_id
    assert len(stored["orders"]) == len(result.orders)
    assert len(stored["fills"]) == sum(
        order.filled_shares > 0 for order in result.orders
    )

    restarted = PortfolioDecisionLedger(db)
    duplicate, created = _record_case(
        restarted,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )
    assert created is False
    assert duplicate == stored
    assert restarted.get_execution(result.execution_id) == stored
    assert restarted.get_account_state()["account"] == (
        result.account_after.to_dict()
    )


def test_existing_execution_rejects_changed_bootstrap_identity(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)
    _record_case(
        ledger,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )

    with pytest.raises(RuntimeError, match="canonical identity"):
        _record_case(
            ledger,
            result,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            bootstrap_account=False,
        )


def test_unrecorded_decision_and_self_reported_result_fail_before_write(
    tmp_path,
) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    _, decision_quotes, execution_quotes, result = _execution_case()

    with pytest.raises(KeyError, match="组合决策"):
        _record_case(
            ledger,
            result,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            bootstrap_account=True,
        )

    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)
    forged = replace(result, nav_before=result.nav_before + 1.0)
    with pytest.raises(RuntimeError, match="重算"):
        _record_case(
            ledger,
            forged,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            bootstrap_account=True,
        )
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM portfolio_executions"
        ).fetchone()[0] == 0


def test_first_execution_requires_explicit_bootstrap(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)

    with pytest.raises(RuntimeError, match="bootstrap"):
        _record_case(
            ledger,
            result,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            bootstrap_account=False,
        )


def test_same_decision_cannot_persist_a_second_execution(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)
    _record_case(
        ledger,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )

    later_quotes = _quotes("2026-07-25", prices=_prices(0.5))
    later_result = execute_portfolio_decision(
        decision,
        execution_session="2026-07-25",
        account=result.account_before,
        decision_quotes=decision_quotes,
        quotes=later_quotes,
        fee_schedule=FEES,
    )
    with pytest.raises(RuntimeError, match="已经存在执行结果"):
        _record_case(
            ledger,
            later_result,
            decision_quotes=decision_quotes,
            execution_quotes=later_quotes,
            bootstrap_account=False,
        )


def test_mid_transaction_failure_rolls_back_all_execution_rows(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)
    with sqlite3.connect(ledger.path) as conn:
        conn.executescript(
            """
            CREATE TRIGGER force_order_failure
            BEFORE INSERT ON portfolio_execution_orders
            BEGIN
                SELECT RAISE(ABORT, 'forced order failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced order failure"):
        _record_case(
            ledger,
            result,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            bootstrap_account=True,
        )

    with sqlite3.connect(ledger.path) as conn:
        for table in (
            "portfolio_executions",
            "portfolio_execution_orders",
            "portfolio_execution_fills",
            "portfolio_account_state",
        ):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
        conn.execute("DROP TRIGGER force_order_failure")

    _, created = _record_case(
        ledger,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )
    assert created is True


@pytest.mark.parametrize(
    "tamper",
    [
        "input_numeric_string",
        "result_payload",
        "account_payload",
        "delete_order",
        "delete_fill",
        "order_primary_key",
    ],
)
def test_execution_readback_rejects_tamper(tmp_path, tamper: str) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)
    _record_case(
        ledger,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )

    with sqlite3.connect(ledger.path) as conn:
        if tamper == "input_numeric_string":
            raw = conn.execute(
                """
                SELECT input_payload_json
                FROM portfolio_executions
                WHERE execution_id = ?
                """,
                (result.execution_id,),
            ).fetchone()[0]
            payload = json.loads(raw)
            payload["fee_schedule"]["commission_rate"] = "0.0003"
            tampered = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            conn.execute(
                """
                UPDATE portfolio_executions
                SET input_payload_json = ?, input_sha256 = ?
                WHERE execution_id = ?
                """,
                (
                    tampered,
                    _sha(tampered),
                    result.execution_id,
                ),
            )
        elif tamper == "result_payload":
            raw = conn.execute(
                """
                SELECT result_payload_json
                FROM portfolio_executions
                WHERE execution_id = ?
                """,
                (result.execution_id,),
            ).fetchone()[0]
            payload = json.loads(raw)
            payload["nav_before"] += 1.0
            tampered = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            conn.execute(
                """
                UPDATE portfolio_executions
                SET result_payload_json = ?, result_sha256 = ?
                WHERE execution_id = ?
                """,
                (
                    tampered,
                    _sha(tampered),
                    result.execution_id,
                ),
            )
        elif tamper == "account_payload":
            tampered = json.dumps(
                result.account_before.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            conn.execute(
                """
                UPDATE portfolio_account_state
                SET account_payload_json = ?, account_sha256 = ?
                WHERE singleton_id = 1
                """,
                (
                    tampered,
                    _sha(tampered),
                ),
            )
        elif tamper == "delete_order":
            conn.execute(
                """
                DELETE FROM portfolio_execution_orders
                WHERE order_index = 1
                """
            )
        elif tamper == "delete_fill":
            conn.execute(
                """
                DELETE FROM portfolio_execution_fills
                WHERE fill_index = 1
                """
            )
        else:
            row = conn.execute(
                """
                SELECT order_id
                FROM portfolio_execution_orders
                ORDER BY order_index
                LIMIT 1
                """
            ).fetchone()
            conn.execute(
                """
                UPDATE portfolio_execution_orders
                SET order_id = ?
                WHERE order_id = ?
                """,
                (row[0] + "-TAMPERED", row[0]),
            )

    with pytest.raises(RuntimeError):
        ledger.get_execution(result.execution_id)


@pytest.mark.parametrize(
    ("tamper", "readback"),
    [
        ("execution", "execution"),
        ("account", "account"),
        ("both", "both"),
    ],
)
def test_audit_timestamp_tamper_blocks_readback_and_next_write(
    tmp_path,
    tamper: str,
    readback: str,
) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    first, first_dq, first_eq, first_result = _execution_case()
    ledger.record_decision(first)
    _record_case(
        ledger,
        first_result,
        decision_quotes=first_dq,
        execution_quotes=first_eq,
        bootstrap_account=True,
    )
    second, second_dq, second_eq, second_result = _execution_case(
        account=first_result.account_after,
        decision_session="2026-07-24",
        execution_session="2026-07-25",
        decision_prices=_prices(0.25),
        execution_prices=_prices(0.5),
        suffix="b",
    )
    ledger.record_decision(second)

    with sqlite3.connect(ledger.path) as conn:
        if tamper in {"execution", "both"}:
            conn.execute("UPDATE portfolio_executions SET created_at = 0")
        if tamper in {"account", "both"}:
            conn.execute("UPDATE portfolio_account_state SET updated_at = 0")

    if readback in {"execution", "both"}:
        with pytest.raises(RuntimeError, match="审计身份"):
            ledger.get_execution(first_result.execution_id)
    if readback in {"account", "both"}:
        with pytest.raises(RuntimeError, match="审计身份"):
            ledger.get_account_state()
    with pytest.raises(RuntimeError, match="审计身份"):
        _record_case(
            ledger,
            second_result,
            decision_quotes=second_dq,
            execution_quotes=second_eq,
            bootstrap_account=False,
        )
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM portfolio_executions"
        ).fetchone()[0] == 1


def test_later_execution_consumes_current_account_and_survives_restart(
    tmp_path,
) -> None:
    db = tmp_path / "portfolio.sqlite3"
    ledger = PortfolioDecisionLedger(db)
    first, first_dq, first_eq, first_result = _execution_case()
    ledger.record_decision(first)
    _record_case(
        ledger,
        first_result,
        decision_quotes=first_dq,
        execution_quotes=first_eq,
        bootstrap_account=True,
    )

    second, second_dq, second_eq, second_result = _execution_case(
        account=first_result.account_after,
        decision_session="2026-07-24",
        execution_session="2026-07-25",
        decision_prices=_prices(0.25),
        execution_prices=_prices(0.5),
        suffix="b",
    )
    ledger.record_decision(second)
    stored, created = _record_case(
        ledger,
        second_result,
        decision_quotes=second_dq,
        execution_quotes=second_eq,
        bootstrap_account=False,
    )
    assert created is True
    assert stored["input"]["account_before"] == (
        first_result.account_after.to_dict()
    )

    restarted = PortfolioDecisionLedger(db)
    assert restarted.get_execution(second_result.execution_id) == stored
    assert restarted.get_account_state()["account"] == (
        second_result.account_after.to_dict()
    )


def test_two_ledger_instances_converge_on_one_execution(tmp_path) -> None:
    db = tmp_path / "portfolio.sqlite3"
    first_ledger = PortfolioDecisionLedger(db)
    second_ledger = PortfolioDecisionLedger(db)
    decision, decision_quotes, execution_quotes, result = _execution_case()
    first_ledger.record_decision(decision)

    def write(ledger: PortfolioDecisionLedger):
        return _record_case(
            ledger,
            result,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            bootstrap_account=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(write, (first_ledger, second_ledger)))

    assert sorted(created for _, created in outcomes) == [False, True]
    assert outcomes[0][0] == outcomes[1][0]
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM portfolio_executions"
        ).fetchone()[0] == 1


def test_account_state_rollback_and_wrong_next_account_fail_closed(
    tmp_path,
) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    decision, decision_quotes, execution_quotes, result = _execution_case()
    ledger.record_decision(decision)
    _record_case(
        ledger,
        result,
        decision_quotes=decision_quotes,
        execution_quotes=execution_quotes,
        bootstrap_account=True,
    )

    with sqlite3.connect(ledger.path) as conn:
        conn.execute(
            """
            UPDATE portfolio_account_state
            SET execution_session = '2026-07-23'
            WHERE singleton_id = 1
            """
        )
    with pytest.raises(RuntimeError, match="账户"):
        ledger.get_account_state()

    with sqlite3.connect(ledger.path) as conn:
        conn.execute(
            """
            UPDATE portfolio_account_state
            SET execution_session = ?
            WHERE singleton_id = 1
            """,
            (result.execution_session,),
        )
    wrong_account = VirtualAccount(cash=result.account_after.cash + 1.0)
    second, second_decision_quotes, second_execution_quotes, second_result = (
        _execution_case(
            account=wrong_account,
            decision_session="2026-07-24",
            execution_session="2026-07-25",
            decision_prices=_prices(0.25),
            execution_prices=_prices(0.5),
            suffix="b",
        )
    )
    ledger.record_decision(second)
    with pytest.raises(RuntimeError, match="当前持久账户"):
        _record_case(
            ledger,
            second_result,
            decision_quotes=second_decision_quotes,
            execution_quotes=second_execution_quotes,
            bootstrap_account=False,
        )


def test_execution_session_must_advance_across_persisted_account(tmp_path) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    first, first_dq, first_eq, first_result = _execution_case(
        execution_session="2026-07-25",
    )
    ledger.record_decision(first)
    _record_case(
        ledger,
        first_result,
        decision_quotes=first_dq,
        execution_quotes=first_eq,
        bootstrap_account=True,
    )

    second, second_dq, second_eq, second_result = _execution_case(
        account=first_result.account_after,
        decision_session="2026-07-24",
        execution_session="2026-07-25",
        decision_prices=_prices(0.25),
        execution_prices=_prices(0.5),
        suffix="b",
    )
    ledger.record_decision(second)
    with pytest.raises(
        RuntimeError,
        match="组合决策日期早于前次执行日期",
    ):
        _record_case(
            ledger,
            second_result,
            decision_quotes=second_dq,
            execution_quotes=second_eq,
            bootstrap_account=False,
        )


def test_execution_chain_rejects_reversed_decision_clock_before_write(
    tmp_path,
) -> None:
    ledger = PortfolioDecisionLedger(tmp_path / "portfolio.sqlite3")
    account = VirtualAccount(cash=100_000.0)
    newer, newer_dq, newer_eq, newer_result = _execution_case(
        account=account,
        decision_session="2026-07-24",
        execution_session="2026-07-25",
        decision_prices=_prices(),
        execution_prices=_prices(0.25),
        suffix="newer",
    )
    older, older_dq, older_eq, older_result = _execution_case(
        account=newer_result.account_after,
        decision_session="2026-07-23",
        execution_session="2026-07-26",
        decision_prices=_prices(0.25),
        execution_prices=_prices(0.5),
        suffix="older",
    )
    ledger.record_decision(older)
    ledger.record_decision(newer)
    _record_case(
        ledger,
        newer_result,
        decision_quotes=newer_dq,
        execution_quotes=newer_eq,
        bootstrap_account=True,
    )

    with pytest.raises(RuntimeError, match="组合决策日期未严格前进"):
        _record_case(
            ledger,
            older_result,
            decision_quotes=older_dq,
            execution_quotes=older_eq,
            bootstrap_account=False,
        )
    expected_counts = {
        "portfolio_executions": 1,
        "portfolio_execution_orders": len(newer_result.orders),
        "portfolio_execution_fills": sum(
            order.filled_shares > 0 for order in newer_result.orders
        ),
        "portfolio_account_state": 1,
    }
    with sqlite3.connect(ledger.path) as conn:
        for table, expected_count in expected_counts.items():
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == expected_count
