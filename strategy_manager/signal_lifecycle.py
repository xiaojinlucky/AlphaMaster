"""普通 A 股只做多信号的确定性生命周期。

本模块只做决策，不连接行情源、数据库、飞书或券商。输入当前虚拟仓位、
最新目标仓位和已收盘 K 线价格，输出唯一动作与更新后的虚拟仓位。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal


HOLD = "HOLD"
BUY = "BUY"
ADD = "ADD"
REDUCE = "REDUCE"
EXIT = "EXIT"
TAKE_PROFIT = "TAKE_PROFIT"
STOP_LOSS = "STOP_LOSS"

PUSH_ACTIONS = frozenset({BUY, ADD, REDUCE, EXIT, TAKE_PROFIT, STOP_LOSS})

ACTION_CN = {
    HOLD: "继续持有 / 观望",
    BUY: "买入",
    ADD: "加仓",
    REDUCE: "减仓",
    EXIT: "离场",
    TAKE_PROFIT: "止盈",
    STOP_LOSS: "止损",
}


@dataclass(frozen=True)
class LongOnlyState:
    """单个监控项的虚拟长仓状态。"""

    exposure: float = 0.0
    entry_price: float | None = None
    take_profit_done: bool = False
    last_bar_ts: int | None = None
    last_price: float | None = None


@dataclass(frozen=True)
class LongOnlyDecision:
    """一根已收盘 K 线对应的唯一信号决策。"""

    action: str
    reason: str
    previous_exposure: float
    requested_exposure: float
    resulting_exposure: float
    entry_price: float | None
    stop_price: float | None
    take_profit_price: float | None
    take_profit_done: bool


def _checked_fraction(name: str, value: float, *, allow_zero: bool = True) -> float:
    number = float(value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not math.isfinite(number) or not lower_ok or number > 1.0:
        bound = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} 必须位于 {bound}")
    return number


def _risk_prices(
    entry_price: float | None,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> tuple[float | None, float | None]:
    if entry_price is None:
        return None, None
    return (
        round(entry_price * (1.0 + stop_loss_pct), 6),
        round(entry_price * (1.0 + take_profit_pct), 6),
    )


def _price_reached(price: float, entry_price: float, change_pct: float, *, lower: bool) -> bool:
    """用十进制语义比较价格边界，避免 100 × 1.04 一类浮点临界值漏判。"""
    actual = Decimal(str(price))
    threshold = Decimal(str(entry_price)) * (
        Decimal("1") + Decimal(str(change_pct))
    )
    return actual <= threshold if lower else actual >= threshold


def decide_long_only(
    state: LongOnlyState,
    *,
    raw_position: float,
    price: float,
    minimum_exposure: float,
    rebalance_delta: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    take_profit_remaining_ratio: float,
) -> LongOnlyDecision:
    """把连续因子仓位转换为普通 A 股账户可执行的只做多动作。

    负仓位不会转成做空：空仓时继续观望，已有长仓时目标仓位变为 0。
    止损优先级最高；信号离场高于止盈；止盈首次触发后只减一次仓。
    """
    raw = float(raw_position)
    last_price = float(price)
    if not math.isfinite(raw):
        raise ValueError("raw_position 必须是有限数")
    if not math.isfinite(last_price) or last_price <= 0:
        raise ValueError("price 必须是大于 0 的有限数")

    current = _checked_fraction("state.exposure", state.exposure)
    minimum = _checked_fraction("minimum_exposure", minimum_exposure)
    delta_gate = _checked_fraction("rebalance_delta", rebalance_delta, allow_zero=False)
    remaining_ratio = _checked_fraction(
        "take_profit_remaining_ratio",
        take_profit_remaining_ratio,
    )
    stop_pct = float(stop_loss_pct)
    take_pct = float(take_profit_pct)
    if not math.isfinite(stop_pct) or not -1.0 < stop_pct < 0.0:
        raise ValueError("stop_loss_pct 必须位于 (-1, 0)")
    if not math.isfinite(take_pct) or take_pct <= 0.0:
        raise ValueError("take_profit_pct 必须大于 0")

    requested = min(1.0, max(0.0, raw))
    if requested < minimum:
        requested = 0.0
    entry = float(state.entry_price) if state.entry_price is not None else None
    if current > 0.0 and (entry is None or not math.isfinite(entry) or entry <= 0.0):
        raise ValueError("已有虚拟仓位时 entry_price 必须大于 0")

    if current > 0.0 and _price_reached(
        last_price,
        entry,
        stop_pct,
        lower=True,
    ):
        return LongOnlyDecision(
            action=STOP_LOSS,
            reason=f"收盘价触及虚拟止损线（{stop_pct:.2%}）",
            previous_exposure=current,
            requested_exposure=requested,
            resulting_exposure=0.0,
            entry_price=None,
            stop_price=None,
            take_profit_price=None,
            take_profit_done=False,
        )

    if current > 0.0 and requested == 0.0:
        return LongOnlyDecision(
            action=EXIT,
            reason="目标长仓降为 0；负因子按普通 A 股账户解释为离场而非做空",
            previous_exposure=current,
            requested_exposure=requested,
            resulting_exposure=0.0,
            entry_price=None,
            stop_price=None,
            take_profit_price=None,
            take_profit_done=False,
        )

    if (
        current > 0.0
        and not state.take_profit_done
        and _price_reached(
            last_price,
            entry,
            take_pct,
            lower=False,
        )
    ):
        resulting = round(current * remaining_ratio, 6)
        if resulting <= 0.0:
            resulting = 0.0
            next_entry = None
            take_done = False
        else:
            next_entry = entry
            take_done = True
        stop_price, take_price = _risk_prices(next_entry, stop_pct, take_pct)
        return LongOnlyDecision(
            action=TAKE_PROFIT,
            reason=f"收盘价首次触及虚拟止盈线（+{take_pct:.2%}）",
            previous_exposure=current,
            requested_exposure=requested,
            resulting_exposure=resulting,
            entry_price=next_entry,
            stop_price=stop_price,
            take_profit_price=take_price,
            take_profit_done=take_done,
        )

    if current == 0.0:
        if requested == 0.0:
            return LongOnlyDecision(
                action=HOLD,
                reason="当前空仓且目标长仓为 0",
                previous_exposure=0.0,
                requested_exposure=0.0,
                resulting_exposure=0.0,
                entry_price=None,
                stop_price=None,
                take_profit_price=None,
                take_profit_done=False,
            )
        stop_price, take_price = _risk_prices(last_price, stop_pct, take_pct)
        return LongOnlyDecision(
            action=BUY,
            reason="目标长仓由 0 升至可执行区间",
            previous_exposure=0.0,
            requested_exposure=requested,
            resulting_exposure=requested,
            entry_price=last_price,
            stop_price=stop_price,
            take_profit_price=take_price,
            take_profit_done=False,
        )

    exposure_change = requested - current
    if exposure_change >= delta_gate:
        resulting = requested
        added = resulting - current
        next_entry = ((entry * current) + (last_price * added)) / resulting
        stop_price, take_price = _risk_prices(next_entry, stop_pct, take_pct)
        return LongOnlyDecision(
            action=ADD,
            reason=f"目标长仓提高至少 {delta_gate:.0%}",
            previous_exposure=current,
            requested_exposure=requested,
            resulting_exposure=resulting,
            entry_price=round(next_entry, 6),
            stop_price=stop_price,
            take_profit_price=take_price,
            take_profit_done=state.take_profit_done,
        )

    if exposure_change <= -delta_gate:
        resulting = requested
        stop_price, take_price = _risk_prices(entry, stop_pct, take_pct)
        return LongOnlyDecision(
            action=REDUCE,
            reason=f"目标长仓降低至少 {delta_gate:.0%}",
            previous_exposure=current,
            requested_exposure=requested,
            resulting_exposure=resulting,
            entry_price=entry,
            stop_price=stop_price,
            take_profit_price=take_price,
            take_profit_done=state.take_profit_done,
        )

    stop_price, take_price = _risk_prices(entry, stop_pct, take_pct)
    return LongOnlyDecision(
        action=HOLD,
        reason=f"目标长仓变化未达到 {delta_gate:.0%} 调仓门槛",
        previous_exposure=current,
        requested_exposure=requested,
        resulting_exposure=current,
        entry_price=entry,
        stop_price=stop_price,
        take_profit_price=take_price,
        take_profit_done=state.take_profit_done,
    )
