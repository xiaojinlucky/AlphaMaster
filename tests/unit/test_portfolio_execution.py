from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal

import pytest

import portfolio_manager.controller as controller_module
from portfolio_manager.controller import (
    ModelSignalSnapshot,
    PortfolioPolicy,
)
from portfolio_manager.execution import (
    AShareFeeSchedule,
    ExecutionQuote,
    PositionLot,
    VirtualAccount,
    account_snapshot_sha256,
    execute_portfolio_decision,
)
from portfolio_manager.universe import UniverseContract

SYMBOLS = ("000001", "000002", "000003")
UNIVERSE = UniverseContract(
    universe_id="test-execution:20260723",
    snapshot_date="20260723",
    constituent_count=len(SYMBOLS),
    universe_sha256="e" * 64,
    symbols=SYMBOLS,
)
POLICY = PortfolioPolicy(
    top_k=2,
    dropout_rank=2,
    minimum_history=5,
    minimum_model_exposure=0.05,
    minimum_confidence=0.20,
    minimum_calibrated_score=0.50,
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


def _signal(symbol: str, score: float) -> ModelSignalSnapshot:
    return ModelSignalSnapshot(
        run_id=f"run_20260723T235959Z_{_sha(symbol)[:8]}",
        symbol=symbol,
        bar_ts=1_784_790_000,
        session_date="2026-07-23",
        timeframe="1d",
        market_source="akshare_sina_hfq_ohlcv",
        raw_score=score,
        requested_exposure=0.8,
        confidence=0.9,
        model_version=_sha(f"model:{symbol}"),
        data_version=_sha(f"data:{symbol}"),
        calibration_version="test-v1",
        calibration_history_sha256=_sha(f"history:{symbol}"),
        history_scores=(0, 1, 2, 3, 4),
    )


def _decision(
    *,
    scores: tuple[float, float, float] = (5.0, 4.0, -1.0),
    current_weights: dict[str, float] | None = None,
    account: VirtualAccount | None = None,
    policy: PortfolioPolicy = POLICY,
):
    bound_account = account or VirtualAccount(cash=100_000.0)
    return controller_module._build_portfolio_decision(
        [_signal(symbol, score) for symbol, score in zip(SYMBOLS, scores)],
        universe=UNIVERSE,
        current_weights=current_weights or {},
        account_snapshot_sha256=account_snapshot_sha256(
            bound_account,
            _decision_quotes(bound_account),
        ),
        policy=policy,
    )


def _quotes(
    *,
    status_1: str = "OPEN",
    status_2: str = "OPEN",
    status_3: str = "OPEN",
):
    return (
        ExecutionQuote("000001", "2026-07-24", 10.0, status_1),
        ExecutionQuote("000002", "2026-07-24", 20.0, status_2),
        ExecutionQuote("000003", "2026-07-24", 5.0, status_3),
    )


def _decision_quotes(account: VirtualAccount):
    held = {lot.symbol for lot in account.lots}
    return tuple(
        ExecutionQuote(
            quote.symbol,
            "2026-07-23",
            quote.price,
            quote.status,
            quote.lot_size,
        )
        for quote in _quotes()
        if quote.symbol in held
    )


def _current_weights(account: VirtualAccount) -> dict[str, float]:
    prices = {quote.symbol: quote.price for quote in _decision_quotes(account)}
    shares: dict[str, int] = {}
    for lot in account.lots:
        shares[lot.symbol] = shares.get(lot.symbol, 0) + lot.shares
    nav = account.cash + sum(shares[symbol] * prices[symbol] for symbol in shares)
    return {
        symbol: shares[symbol] * prices[symbol] / nav
        for symbol in sorted(shares)
    }


def test_initial_buy_uses_board_lots_costs_and_cash_limit() -> None:
    account = VirtualAccount(cash=100_000.0)
    result = execute_portfolio_decision(
        _decision(account=account),
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )

    assert result.execution_id.startswith("AM-EXEC-")
    assert [order.side for order in result.orders] == ["BUY", "BUY"]
    assert all(order.filled_shares % 100 == 0 for order in result.orders)
    assert result.orders[0].fill_price == pytest.approx(10.01)
    assert result.orders[0].fees > 0
    assert result.account_after.cash >= 0
    assert sum(lot.shares for lot in result.account_after.lots) > 0


def test_sell_runs_before_buy_and_releases_cash() -> None:
    account = VirtualAccount(
        cash=100.0,
        lots=(
            PositionLot("000003", "2026-07-22", 10_000, 4.0),
        ),
    )
    result = execute_portfolio_decision(
        _decision(current_weights=_current_weights(account), account=account),
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )

    assert result.orders[0].side == "SELL"
    assert result.orders[0].symbol == "000003"
    assert result.orders[0].status == "FILLED"
    assert any(order.side == "BUY" and order.filled_shares > 0 for order in result.orders)


def test_t_plus_one_blocks_same_day_sale_without_replacement() -> None:
    account = VirtualAccount(
        cash=0.0,
        lots=(
            PositionLot("000003", "2026-07-24", 1_000, 5.0),
        ),
    )
    result = execute_portfolio_decision(
        _decision(current_weights=_current_weights(account), account=account),
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )

    sell = result.orders[0]
    assert sell.side == "SELL"
    assert sell.status == "REJECTED"
    assert sell.reason == "T_PLUS_ONE"
    assert sell.filled_shares == 0
    assert all(order.filled_shares == 0 for order in result.orders[1:])
    assert sum(lot.shares for lot in result.account_after.lots) == 1_000


@pytest.mark.parametrize(
    ("side", "status", "reason"),
    [
        ("BUY", "LIMIT_UP_LOCKED", "LIMIT_UP_LOCKED"),
        ("SELL", "LIMIT_DOWN_LOCKED", "LIMIT_DOWN_LOCKED"),
        ("BUY", "SUSPENDED", "SUSPENDED"),
        ("SELL", "SUSPENDED", "SUSPENDED"),
    ],
)
def test_explicit_market_status_blocks_only_the_affected_order(
    side: str,
    status: str,
    reason: str,
) -> None:
    if side == "BUY":
        account = VirtualAccount(cash=100_000.0)
        quotes = _quotes(status_1=status)
        decision = _decision(account=account)
        symbol = "000001"
    else:
        account = VirtualAccount(
            cash=0.0,
            lots=(PositionLot("000003", "2026-07-22", 1_000, 5.0),),
        )
        quotes = _quotes(status_3=status)
        decision = _decision(
            current_weights=_current_weights(account),
            account=account,
        )
        symbol = "000003"

    result = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=quotes,
        fee_schedule=FEES,
    )
    blocked = next(
        order
        for order in result.orders
        if order.side == side and order.symbol == symbol
    )

    assert blocked.status == "REJECTED"
    assert blocked.reason == reason
    assert blocked.filled_shares == 0


def test_minimum_commission_can_make_buy_partial() -> None:
    account = VirtualAccount(cash=1_010.0)
    result = execute_portfolio_decision(
        _decision(
            account=account,
            policy=replace(POLICY, top_k=1, dropout_rank=1),
        ),
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=replace(FEES, minimum_commission=20.0),
    )

    first = result.orders[0]
    assert first.side == "BUY"
    assert first.status == "REJECTED"
    assert first.reason == "INSUFFICIENT_CASH"
    assert result.account_after.cash >= 0


def test_execution_is_deterministic_across_quote_order() -> None:
    account = VirtualAccount(cash=100_000.0)
    decision = _decision(account=account)

    forward = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )
    reverse = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=account,
        decision_quotes=reversed(_decision_quotes(account)),
        quotes=reversed(_quotes()),
        fee_schedule=FEES,
    )

    assert forward.to_dict() == reverse.to_dict()


def test_invalid_execution_inputs_fail_closed() -> None:
    account = VirtualAccount(cash=100_000.0)
    decision = _decision(account=account)
    with pytest.raises(ValueError, match="晚于"):
        execute_portfolio_decision(
            decision,
            execution_session="2026-07-23",
            account=account,
            decision_quotes=_decision_quotes(account),
            quotes=_quotes(),
            fee_schedule=FEES,
        )
    with pytest.raises(ValueError, match="重复股票"):
        execute_portfolio_decision(
            decision,
            execution_session="2026-07-24",
            account=account,
            decision_quotes=_decision_quotes(account),
            quotes=(*_quotes(), _quotes()[0]),
            fee_schedule=FEES,
        )
    with pytest.raises(ValueError, match="绑定身份"):
        changed_account = VirtualAccount(
            cash=100_000.0,
            lots=(PositionLot("000001", "2026-07-22", 101, 10.0),),
        )
        execute_portfolio_decision(
            decision,
            execution_session="2026-07-24",
            account=changed_account,
            decision_quotes=_decision_quotes(changed_account),
            quotes=_quotes(),
            fee_schedule=FEES,
        )
    with pytest.raises(ValueError, match="有效日历日期"):
        execute_portfolio_decision(
            decision,
            execution_session="2026-02-30",
            account=account,
            decision_quotes=_decision_quotes(account),
            quotes=(),
            fee_schedule=FEES,
        )


def test_account_holdings_must_match_decision_current_symbols() -> None:
    account = VirtualAccount(
        cash=90_000.0,
        lots=(PositionLot("000003", "2026-07-22", 1_000, 5.0),),
    )
    with pytest.raises(ValueError, match="绑定身份"):
        execute_portfolio_decision(
            _decision(),
            execution_session="2026-07-24",
            account=account,
            decision_quotes=_decision_quotes(account),
            quotes=_quotes(),
            fee_schedule=FEES,
        )


@pytest.mark.parametrize(
    "tampered",
    [
        lambda decision: replace(decision, decision_id="AM-PORT-" + "0" * 24),
        lambda decision: replace(decision, cash_weight=0.999),
        lambda decision: replace(decision, entered_symbols=()),
        lambda decision: replace(decision, ranking=tuple(reversed(decision.ranking))),
    ],
)
def test_tampered_portfolio_decision_is_rejected(tampered) -> None:
    account = VirtualAccount(cash=100_000.0)
    with pytest.raises(ValueError, match="身份或内容校验失败"):
        execute_portfolio_decision(
            tampered(_decision(account=account)),
            execution_session="2026-07-24",
            account=account,
            decision_quotes=_decision_quotes(account),
            quotes=_quotes(),
            fee_schedule=FEES,
        )


def test_equivalent_numeric_types_share_one_execution_identity() -> None:
    account = VirtualAccount(cash=100_000)
    decision = _decision(account=account)
    integer_result = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=(
            ExecutionQuote("000001", "2026-07-24", 10, "OPEN"),
            ExecutionQuote("000002", "2026-07-24", 20, "OPEN"),
            ExecutionQuote("000003", "2026-07-24", 5, "OPEN"),
        ),
        fee_schedule=FEES,
    )
    decimal_result = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=VirtualAccount(cash=Decimal("100000.0")),
        decision_quotes=_decision_quotes(account),
        quotes=(
            ExecutionQuote("000001", "2026-07-24", Decimal("10.0"), "OPEN"),
            ExecutionQuote("000002", "2026-07-24", Decimal("20.00"), "OPEN"),
            ExecutionQuote("000003", "2026-07-24", Decimal("5.000"), "OPEN"),
        ),
        fee_schedule=AShareFeeSchedule(
            commission_rate=Decimal("0.00030"),
            minimum_commission=Decimal("5.00"),
            stamp_duty_rate=Decimal("0.000500"),
            transfer_fee_rate=Decimal("0.000010"),
            slippage_rate=Decimal("0.0010"),
        ),
    )

    assert integer_result.execution_id == decimal_result.execution_id
    assert integer_result.to_dict() == decimal_result.to_dict()


def test_existing_odd_lot_can_be_sold_but_buys_remain_board_lots() -> None:
    account = VirtualAccount(
        cash=0.0,
        lots=(PositionLot("000003", "2026-07-22", 101, 5.0),),
    )
    result = execute_portfolio_decision(
        _decision(current_weights=_current_weights(account), account=account),
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )

    sell = next(order for order in result.orders if order.side == "SELL")
    buys = [order for order in result.orders if order.side == "BUY"]
    assert sell.requested_shares == sell.filled_shares == 101
    assert all(order.filled_shares % 100 == 0 for order in buys)


def test_slippage_cannot_create_zero_price_sell() -> None:
    account = VirtualAccount(
        cash=0.0,
        lots=(PositionLot("000003", "2026-07-22", 1_000, 5.0),),
    )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        execute_portfolio_decision(
            _decision(current_weights=_current_weights(account), account=account),
            execution_session="2026-07-24",
            account=account,
            decision_quotes=_decision_quotes(account),
            quotes=_quotes(),
            fee_schedule=replace(FEES, slippage_rate=1.0),
        )


def test_cash_and_reference_value_conservation() -> None:
    account = VirtualAccount(
        cash=100.0,
        lots=(PositionLot("000003", "2026-07-22", 10_000, 4.0),),
    )
    result = execute_portfolio_decision(
        _decision(current_weights=_current_weights(account), account=account),
        execution_session="2026-07-24",
        account=account,
        decision_quotes=_decision_quotes(account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )
    cash_change = sum(order.cash_change for order in result.orders)
    assert result.account_after.cash == pytest.approx(account.cash + cash_change)

    reference_prices = {quote.symbol: quote.price for quote in _quotes()}
    after_shares: dict[str, int] = {}
    for lot in result.account_after.lots:
        after_shares[lot.symbol] = after_shares.get(lot.symbol, 0) + lot.shares
    nav_after_at_reference = result.account_after.cash + sum(
        shares * reference_prices[symbol]
        for symbol, shares in after_shares.items()
    )
    costs = sum(order.fees for order in result.orders)
    slippage_loss = sum(
        abs(order.reference_price - (order.fill_price or order.reference_price))
        * order.filled_shares
        for order in result.orders
    )
    assert result.nav_before - nav_after_at_reference == pytest.approx(
        costs + slippage_loss
    )


def test_equivalent_lot_order_has_one_identity_and_result() -> None:
    lots = (
        PositionLot("000003", "2026-07-22", 100, 4.0),
        PositionLot("000003", "2026-07-22", 100, 6.0),
    )
    forward_account = VirtualAccount(cash=0.0, lots=lots)
    reverse_account = VirtualAccount(cash=0.0, lots=tuple(reversed(lots)))
    policy = replace(
        POLICY,
        top_k=1,
        dropout_rank=1,
        gross_exposure=0.5,
    )
    decision = _decision(
        scores=(-1.0, -2.0, 5.0),
        current_weights=_current_weights(forward_account),
        account=forward_account,
        policy=policy,
    )

    forward = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=forward_account,
        decision_quotes=_decision_quotes(forward_account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )
    reverse = execute_portfolio_decision(
        decision,
        execution_session="2026-07-24",
        account=reverse_account,
        decision_quotes=_decision_quotes(reverse_account),
        quotes=_quotes(),
        fee_schedule=FEES,
    )

    assert account_snapshot_sha256(
        forward_account,
        _decision_quotes(forward_account),
    ) == account_snapshot_sha256(
        reverse_account,
        _decision_quotes(reverse_account),
    )
    assert forward.to_dict() == reverse.to_dict()
