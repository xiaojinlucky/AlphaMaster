"""普通 A 股组合目标的确定性虚拟执行约束层。

本模块不连接券商，也不猜测历史 ST、涨跌停比例或停牌状态。调用方必须
为执行日显式提供行情状态；执行器只负责 T+1、整手、现金和费用约束。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Iterable, Mapping

from portfolio_manager.controller import PortfolioDecision

_SYMBOL_RE = re.compile(r"^\d{6}$")
_SESSION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_QUOTE_STATUSES = frozenset(
    {
        "OPEN",
        "SUSPENDED",
        "LIMIT_UP_LOCKED",
        "LIMIT_DOWN_LOCKED",
    }
)
_ZERO = Decimal("0")
_ONE = Decimal("1")
_WEIGHT_TOLERANCE = Decimal("0.000000001")
_EXECUTION_CONTRACT_VERSION = "a-share-virtual-execution-v1"


def _decimal(name: str, value: object, *, minimum: Decimal = _ZERO) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} 必须是数值")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数") from exc
    if not number.is_finite() or number < minimum:
        raise ValueError(f"{name} 必须是不小于 {minimum} 的有限数")
    return number


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是正整数")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return number


def _symbol(value: object) -> str:
    if not isinstance(value, str) or _SYMBOL_RE.fullmatch(value) is None:
        raise ValueError(f"股票代码必须是 6 位数字文本: {value!r}")
    return value


def _session_date(name: str, value: object) -> str:
    if not isinstance(value, str) or _SESSION_DATE_RE.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} 不是有效日历日期") from exc
    return value


def _as_float(value: Decimal) -> float:
    return round(float(value), 12)


def _canonicalize(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (Integral, Real, Decimal)):
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("canonical_number 必须是有限数") from exc
        if not number.is_finite():
            raise ValueError("canonical_number 必须是有限数")
        if number == _ZERO:
            return "0"
        return format(number.normalize(), "f")
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    raise ValueError(f"不能规范化的执行身份字段: {type(value).__name__}")


@dataclass(frozen=True)
class AShareFeeSchedule:
    """费用参数由调用方显式提供，避免把可能变化的费率写死在执行器中。"""

    commission_rate: float
    minimum_commission: float
    stamp_duty_rate: float
    transfer_fee_rate: float
    slippage_rate: float

    def validated(self) -> "AShareFeeSchedule":
        for name in (
            "commission_rate",
            "stamp_duty_rate",
            "transfer_fee_rate",
            "slippage_rate",
        ):
            value = _decimal(name, getattr(self, name))
            if name == "slippage_rate" and value >= _ONE:
                raise ValueError("slippage_rate 必须位于 [0, 1)")
            if value > _ONE:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        _decimal("minimum_commission", self.minimum_commission)
        return self

    def to_dict(self) -> dict[str, float]:
        self.validated()
        return asdict(self)


@dataclass(frozen=True)
class PositionLot:
    symbol: str
    acquired_session: str
    shares: int
    unit_cost: float

    def validated(self, *, lot_size: int | None = None) -> "PositionLot":
        _symbol(self.symbol)
        _session_date("acquired_session", self.acquired_session)
        _positive_integer("shares", self.shares)
        _decimal("unit_cost", self.unit_cost, minimum=Decimal("0.000000000001"))
        return self

    def to_dict(self) -> dict[str, object]:
        self.validated()
        return asdict(self)


def _lot_sort_key(lot: PositionLot) -> tuple[str, str, Decimal, int]:
    lot.validated()
    return (
        lot.symbol,
        lot.acquired_session,
        _decimal("unit_cost", lot.unit_cost),
        lot.shares,
    )


@dataclass(frozen=True)
class VirtualAccount:
    cash: float
    lots: tuple[PositionLot, ...] = ()

    def to_dict(self) -> dict[str, object]:
        _decimal("cash", self.cash)
        return {
            "cash": self.cash,
            "lots": [
                lot.to_dict() for lot in sorted(self.lots, key=_lot_sort_key)
            ],
        }


@dataclass(frozen=True)
class ExecutionQuote:
    symbol: str
    session_date: str
    price: float
    status: str
    lot_size: int = 100

    def validated(self) -> "ExecutionQuote":
        _symbol(self.symbol)
        _session_date("session_date", self.session_date)
        _decimal("price", self.price, minimum=Decimal("0.000000000001"))
        if self.status not in _QUOTE_STATUSES:
            raise ValueError(
                f"{self.symbol}.status 必须是 {sorted(_QUOTE_STATUSES)} 之一"
            )
        _positive_integer("lot_size", self.lot_size)
        return self

    def to_dict(self) -> dict[str, object]:
        self.validated()
        return asdict(self)


@dataclass(frozen=True)
class VirtualOrder:
    order_id: str
    symbol: str
    side: str
    requested_shares: int
    filled_shares: int
    status: str
    reason: str
    reference_price: float
    fill_price: float | None
    gross_amount: float
    fees: float
    cash_change: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioExecutionResult:
    execution_id: str
    decision_id: str
    decision_session: str
    execution_session: str
    nav_before: float
    account_before: VirtualAccount
    account_after: VirtualAccount
    orders: tuple[VirtualOrder, ...]
    target_shares: tuple[tuple[str, int], ...]
    resulting_weights: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "decision_id": self.decision_id,
            "decision_session": self.decision_session,
            "execution_session": self.execution_session,
            "nav_before": self.nav_before,
            "account_before": self.account_before.to_dict(),
            "account_after": self.account_after.to_dict(),
            "orders": [order.to_dict() for order in self.orders],
            "target_shares": dict(self.target_shares),
            "resulting_weights": dict(self.resulting_weights),
        }


def _fee_amount(
    gross: Decimal,
    *,
    side: str,
    schedule: AShareFeeSchedule,
) -> Decimal:
    if gross <= _ZERO:
        return _ZERO
    commission = max(
        gross * Decimal(str(schedule.commission_rate)),
        Decimal(str(schedule.minimum_commission)),
    )
    transfer = gross * Decimal(str(schedule.transfer_fee_rate))
    stamp = (
        gross * Decimal(str(schedule.stamp_duty_rate))
        if side == "SELL"
        else _ZERO
    )
    return commission + transfer + stamp


def _canonical_execution_id(
    *,
    decision: PortfolioDecision,
    execution_session: str,
    account: VirtualAccount,
    quotes: Mapping[str, ExecutionQuote],
    schedule: AShareFeeSchedule,
) -> str:
    payload = {
        "contract_version": _EXECUTION_CONTRACT_VERSION,
        "decision": decision.to_dict(),
        "execution_session": execution_session,
        "account": account.to_dict(),
        "quotes": {
            symbol: quotes[symbol].to_dict() for symbol in sorted(quotes)
        },
        "fee_schedule": schedule.to_dict(),
    }
    raw = json.dumps(
        _canonicalize(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "AM-EXEC-" + hashlib.sha256(raw).hexdigest()[:24].upper()


def account_snapshot_sha256(
    account: VirtualAccount,
    valuation_quotes: Iterable[ExecutionQuote],
) -> str:
    """绑定现金、逐批持仓以及决策日估值行情的确定性身份。"""
    if not isinstance(account, VirtualAccount):
        raise ValueError("account 必须是 VirtualAccount")
    account.to_dict()
    by_symbol: dict[str, ExecutionQuote] = {}
    for quote in valuation_quotes:
        if not isinstance(quote, ExecutionQuote):
            raise ValueError("valuation_quotes 必须由 ExecutionQuote 组成")
        quote.validated()
        if quote.symbol in by_symbol:
            raise ValueError(f"valuation_quotes 出现重复股票: {quote.symbol}")
        by_symbol[quote.symbol] = quote
    held = {lot.symbol for lot in account.lots}
    if set(by_symbol) != held:
        raise ValueError("决策日估值行情必须与持仓股票集合完全一致")
    payload = {
        "contract_version": _EXECUTION_CONTRACT_VERSION,
        "account": {
            "cash": account.cash,
            "lots": account.to_dict()["lots"],
        },
        "valuation_quotes": {
            symbol: by_symbol[symbol].to_dict() for symbol in sorted(by_symbol)
        },
    }
    raw = json.dumps(
        _canonicalize(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _shares_by_symbol(lots: Iterable[PositionLot]) -> dict[str, int]:
    shares: dict[str, int] = {}
    for lot in lots:
        shares[lot.symbol] = shares.get(lot.symbol, 0) + lot.shares
    return shares


def _remove_sellable_lots(
    lots: list[PositionLot],
    *,
    symbol: str,
    session_date: str,
    shares: int,
) -> None:
    remaining = shares
    for index, lot in sorted(
        enumerate(lots),
        key=lambda item: (item[1].acquired_session, item[0]),
    ):
        if (
            remaining <= 0
            or lot.symbol != symbol
            or lot.acquired_session >= session_date
            or lot.shares <= 0
        ):
            continue
        removed = min(remaining, lot.shares)
        remaining -= removed
        left = lot.shares - removed
        lots[index] = (
            PositionLot(
                symbol=lot.symbol,
                acquired_session=lot.acquired_session,
                shares=left,
                unit_cost=lot.unit_cost,
            )
            if left
            else PositionLot(
                symbol=lot.symbol,
                acquired_session=lot.acquired_session,
                shares=0,
                unit_cost=lot.unit_cost,
            )
        )
    if remaining:
        raise RuntimeError("内部错误：卖出数量超过可卖持仓")
    lots[:] = [lot for lot in lots if lot.shares > 0]


def _max_affordable_shares(
    *,
    desired_shares: int,
    lot_size: int,
    fill_price: Decimal,
    cash: Decimal,
    schedule: AShareFeeSchedule,
) -> int:
    maximum_lots = desired_shares // lot_size
    low = 0
    high = maximum_lots
    while low < high:
        middle = (low + high + 1) // 2
        gross = fill_price * (middle * lot_size)
        required = gross + _fee_amount(
            gross,
            side="BUY",
            schedule=schedule,
        )
        if required <= cash:
            low = middle
        else:
            high = middle - 1
    return low * lot_size


def execute_portfolio_decision(
    decision: PortfolioDecision,
    *,
    execution_session: str,
    account: VirtualAccount,
    decision_quotes: Iterable[ExecutionQuote],
    quotes: Iterable[ExecutionQuote],
    fee_schedule: AShareFeeSchedule,
) -> PortfolioExecutionResult:
    """把目标权重转换为受 A 股约束的虚拟成交结果。"""

    if not isinstance(decision, PortfolioDecision):
        raise ValueError("decision 必须是 PortfolioDecision")
    from portfolio_manager.ledger import _validate_decision_payload

    try:
        _validate_decision_payload(decision.to_dict())
    except RuntimeError as exc:
        raise ValueError(f"组合决策身份或内容校验失败: {exc}") from exc
    execution_date = _session_date("execution_session", execution_session)
    decision_date = _session_date("decision.session_date", decision.session_date)
    if execution_date <= decision_date:
        raise ValueError("执行交易日必须晚于产生组合信号的交易日")
    fee_schedule.validated()
    cash = _decimal("account.cash", account.cash)
    universe_list = tuple(_symbol(symbol) for symbol in decision.universe.symbols)
    universe_symbols = frozenset(universe_list)
    if len(universe_symbols) != len(universe_list):
        raise ValueError("冻结股票池含重复股票")
    selected_symbols = tuple(_symbol(symbol) for symbol in decision.selected_symbols)
    if len(set(selected_symbols)) != len(selected_symbols):
        raise ValueError("selected_symbols 含重复股票")
    if not set(selected_symbols).issubset(universe_symbols):
        raise ValueError("selected_symbols 包含冻结股票池外标的")

    raw_target_weights = tuple(decision.target_weights)
    target_symbols = tuple(_symbol(symbol) for symbol, _ in raw_target_weights)
    if len(set(target_symbols)) != len(target_symbols):
        raise ValueError("target_weights 含重复股票")
    if set(target_symbols) != set(selected_symbols):
        raise ValueError("target_weights 必须与 selected_symbols 完全一致")
    target_total = sum(
        (
            _decimal(f"target_weights[{symbol}]", weight)
            for symbol, weight in raw_target_weights
        ),
        start=_ZERO,
    )
    if target_total > _ONE:
        raise ValueError("target_weights 合计不能超过 1")

    by_symbol: dict[str, ExecutionQuote] = {}
    for quote in quotes:
        if not isinstance(quote, ExecutionQuote):
            raise ValueError("quotes 必须由 ExecutionQuote 组成")
        quote.validated()
        if quote.symbol in by_symbol:
            raise ValueError(f"quotes 出现重复股票: {quote.symbol}")
        if quote.symbol not in universe_symbols:
            raise ValueError(f"quotes 包含冻结股票池外标的: {quote.symbol}")
        if quote.session_date != execution_date:
            raise ValueError(f"{quote.symbol} 行情不属于执行交易日")
        by_symbol[quote.symbol] = quote

    lots = sorted(account.lots, key=_lot_sort_key)
    held_symbols: set[str] = set()
    for lot in lots:
        symbol = _symbol(lot.symbol)
        if symbol not in universe_symbols:
            raise ValueError(f"虚拟持仓不属于冻结股票池: {symbol}")
        if symbol not in by_symbol:
            raise ValueError(f"缺少持仓股票的执行行情: {symbol}")
        lot.validated()
        if lot.acquired_session > execution_date:
            raise ValueError(f"{symbol} 持仓批次来自未来交易日")
        held_symbols.add(symbol)

    decision_quote_map: dict[str, ExecutionQuote] = {}
    for quote in decision_quotes:
        if not isinstance(quote, ExecutionQuote):
            raise ValueError("decision_quotes 必须由 ExecutionQuote 组成")
        quote.validated()
        if quote.symbol in decision_quote_map:
            raise ValueError(f"decision_quotes 出现重复股票: {quote.symbol}")
        if quote.session_date != decision_date:
            raise ValueError(f"{quote.symbol} 决策估值行情不属于决策交易日")
        decision_quote_map[quote.symbol] = quote
    if account_snapshot_sha256(account, decision_quote_map.values()) != (
        decision.account_snapshot_sha256
    ):
        raise ValueError("真实账户快照与组合决策绑定身份不一致")

    declared_current = {
        _symbol(symbol): _decimal(f"current_weights[{symbol}]", weight)
        for symbol, weight in decision.current_weights
        if _decimal(f"current_weights[{symbol}]", weight) > _ZERO
    }
    declared_current_symbols = set(declared_current)
    if held_symbols != declared_current_symbols:
        raise ValueError("虚拟持仓股票集合与组合决策的 current_weights 不一致")
    decision_nav = cash + sum(
        Decimal(_shares_by_symbol(lots)[symbol])
        * _decimal(f"{symbol}.decision_price", decision_quote_map[symbol].price)
        for symbol in held_symbols
    )
    if decision_nav <= _ZERO:
        raise ValueError("决策日账户净值必须大于 0")
    for symbol in held_symbols:
        actual_weight = (
            Decimal(_shares_by_symbol(lots)[symbol])
            * _decimal(f"{symbol}.decision_price", decision_quote_map[symbol].price)
            / decision_nav
        )
        if abs(actual_weight - declared_current[symbol]) > _WEIGHT_TOLERANCE:
            raise ValueError(f"{symbol} 真实账户权重与组合决策不一致")

    target_weights = dict(raw_target_weights)
    required_quotes = held_symbols | set(target_weights)
    missing_quotes = sorted(required_quotes - set(by_symbol))
    if missing_quotes:
        raise ValueError(f"缺少目标或持仓股票的执行行情: {missing_quotes}")

    current_shares = _shares_by_symbol(lots)
    nav_before = cash + sum(
        Decimal(current_shares.get(symbol, 0))
        * _decimal(f"{symbol}.price", by_symbol[symbol].price)
        for symbol in current_shares
    )
    if nav_before <= _ZERO:
        raise ValueError("执行前组合净值必须大于 0")

    target_shares: dict[str, int] = {}
    for symbol, weight_value in sorted(target_weights.items()):
        weight = _decimal(f"target_weights[{symbol}]", weight_value)
        if weight > _ONE:
            raise ValueError(f"target_weights[{symbol}] 必须位于 [0, 1]")
        quote = by_symbol[symbol]
        price = _decimal(f"{symbol}.price", quote.price)
        lots_count = int(
            (nav_before * weight / price)
            // Decimal(quote.lot_size)
        )
        target_shares[symbol] = lots_count * quote.lot_size

    execution_id = _canonical_execution_id(
        decision=decision,
        execution_session=execution_date,
        account=account,
        quotes=by_symbol,
        schedule=fee_schedule,
    )
    orders: list[VirtualOrder] = []
    order_index = 0

    # 先卖后买，现金释放后才参与买入计算；失败卖单不自动递补其他股票。
    for symbol in sorted(set(current_shares) | set(target_shares)):
        current = _shares_by_symbol(lots).get(symbol, 0)
        target = target_shares.get(symbol, 0)
        requested = max(0, current - target)
        if requested <= 0:
            continue
        quote = by_symbol[symbol]
        order_index += 1
        order_id = f"{execution_id}-{order_index:03d}"
        reason = ""
        if quote.status == "SUSPENDED":
            filled = 0
            reason = "SUSPENDED"
        elif quote.status == "LIMIT_DOWN_LOCKED":
            filled = 0
            reason = "LIMIT_DOWN_LOCKED"
        else:
            sellable = sum(
                lot.shares
                for lot in lots
                if (
                    lot.symbol == symbol
                    and lot.acquired_session < execution_date
                )
            )
            filled = min(requested, sellable)
            if filled < requested:
                reason = "T_PLUS_ONE"

        reference = _decimal(f"{symbol}.price", quote.price)
        if filled:
            fill_price = reference * (
                _ONE - Decimal(str(fee_schedule.slippage_rate))
            )
            gross = fill_price * filled
            fees = _fee_amount(
                gross,
                side="SELL",
                schedule=fee_schedule,
            )
            proceeds = gross - fees
            if proceeds < _ZERO:
                raise ValueError("卖出费用不能超过成交金额")
            cash += proceeds
            _remove_sellable_lots(
                lots,
                symbol=symbol,
                session_date=execution_date,
                shares=filled,
            )
        else:
            fill_price = None
            gross = fees = proceeds = _ZERO
        status = (
            "FILLED"
            if filled == requested
            else ("PARTIAL" if filled > 0 else "REJECTED")
        )
        orders.append(
            VirtualOrder(
                order_id=order_id,
                symbol=symbol,
                side="SELL",
                requested_shares=requested,
                filled_shares=filled,
                status=status,
                reason=reason,
                reference_price=_as_float(reference),
                fill_price=(
                    _as_float(fill_price) if fill_price is not None else None
                ),
                gross_amount=_as_float(gross),
                fees=_as_float(fees),
                cash_change=_as_float(proceeds),
            )
        )

    for symbol in selected_symbols:
        current = _shares_by_symbol(lots).get(symbol, 0)
        target = target_shares.get(symbol, 0)
        requested = max(0, target - current)
        if requested <= 0:
            continue
        quote = by_symbol[symbol]
        order_index += 1
        order_id = f"{execution_id}-{order_index:03d}"
        reason = ""
        reference = _decimal(f"{symbol}.price", quote.price)
        fill_price = reference * (
            _ONE + Decimal(str(fee_schedule.slippage_rate))
        )
        if quote.status == "SUSPENDED":
            filled = 0
            reason = "SUSPENDED"
        elif quote.status == "LIMIT_UP_LOCKED":
            filled = 0
            reason = "LIMIT_UP_LOCKED"
        else:
            filled = _max_affordable_shares(
                desired_shares=requested,
                lot_size=quote.lot_size,
                fill_price=fill_price,
                cash=cash,
                schedule=fee_schedule,
            )
            if filled < requested:
                reason = "INSUFFICIENT_CASH"

        if filled:
            gross = fill_price * filled
            fees = _fee_amount(
                gross,
                side="BUY",
                schedule=fee_schedule,
            )
            outflow = gross + fees
            if outflow > cash:
                raise RuntimeError("内部错误：买入成交超过可用现金")
            cash -= outflow
            lots.append(
                PositionLot(
                    symbol=symbol,
                    acquired_session=execution_date,
                    shares=filled,
                    unit_cost=_as_float(outflow / filled),
                )
            )
        else:
            gross = fees = outflow = _ZERO
        status = (
            "FILLED"
            if filled == requested
            else ("PARTIAL" if filled > 0 else "REJECTED")
        )
        orders.append(
            VirtualOrder(
                order_id=order_id,
                symbol=symbol,
                side="BUY",
                requested_shares=requested,
                filled_shares=filled,
                status=status,
                reason=reason,
                reference_price=_as_float(reference),
                fill_price=_as_float(fill_price) if filled else None,
                gross_amount=_as_float(gross),
                fees=_as_float(fees),
                cash_change=-_as_float(outflow),
            )
        )

    normalized_lots = tuple(
        sorted(
            lots,
            key=_lot_sort_key,
        )
    )
    account_after = VirtualAccount(
        cash=_as_float(cash),
        lots=normalized_lots,
    )
    resulting_shares = _shares_by_symbol(normalized_lots)
    nav_after = cash + sum(
        Decimal(shares)
        * _decimal(f"{symbol}.price", by_symbol[symbol].price)
        for symbol, shares in resulting_shares.items()
    )
    resulting_weights = tuple(
        (
            symbol,
            _as_float(
                Decimal(shares)
                * _decimal(f"{symbol}.price", by_symbol[symbol].price)
                / nav_after
            ),
        )
        for symbol, shares in sorted(resulting_shares.items())
        if shares > 0
    )
    if not math.isfinite(float(nav_after)) or nav_after <= _ZERO:
        raise RuntimeError("执行后组合净值非法")

    return PortfolioExecutionResult(
        execution_id=execution_id,
        decision_id=decision.decision_id,
        decision_session=decision.session_date,
        execution_session=execution_date,
        nav_before=_as_float(nav_before),
        account_before=account,
        account_after=account_after,
        orders=tuple(orders),
        target_shares=tuple(sorted(target_shares.items())),
        resulting_weights=resulting_weights,
    )


__all__ = [
    "AShareFeeSchedule",
    "ExecutionQuote",
    "PortfolioExecutionResult",
    "PositionLot",
    "VirtualAccount",
    "VirtualOrder",
    "account_snapshot_sha256",
    "execute_portfolio_decision",
]
