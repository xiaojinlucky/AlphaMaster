from __future__ import annotations

import pytest

from strategy_manager.signal_lifecycle import (
    ADD,
    BUY,
    EXIT,
    HOLD,
    REDUCE,
    STOP_LOSS,
    TAKE_PROFIT,
    LongOnlyState,
    decide_long_only,
)
from web.signal_ledger import SignalLedger


DEFAULTS = {
    "minimum_exposure": 0.05,
    "rebalance_delta": 0.10,
    "stop_loss_pct": -0.02,
    "take_profit_pct": 0.04,
    "take_profit_remaining_ratio": 0.50,
}


def decide(state: LongOnlyState, *, raw_position: float, price: float):
    return decide_long_only(
        state,
        raw_position=raw_position,
        price=price,
        **DEFAULTS,
    )


def test_long_only_negative_signal_never_opens_short() -> None:
    decision = decide(LongOnlyState(), raw_position=-0.80, price=100.0)

    assert decision.action == HOLD
    assert decision.requested_exposure == 0.0
    assert decision.resulting_exposure == 0.0


def test_signal_inside_no_trade_band_does_not_open_position() -> None:
    decision = decide(LongOnlyState(), raw_position=0.049, price=100.0)

    assert decision.action == HOLD
    assert decision.requested_exposure == 0.0
    assert decision.resulting_exposure == 0.0


def test_long_only_buy_add_reduce_and_exit() -> None:
    bought = decide(LongOnlyState(), raw_position=0.40, price=100.0)
    assert bought.action == BUY
    assert bought.resulting_exposure == pytest.approx(0.40)
    assert bought.entry_price == pytest.approx(100.0)

    added = decide(
        LongOnlyState(
            exposure=bought.resulting_exposure,
            entry_price=bought.entry_price,
        ),
        raw_position=0.65,
        price=102.0,
    )
    assert added.action == ADD
    assert added.resulting_exposure == pytest.approx(0.65)
    assert added.entry_price == pytest.approx(100.769231, abs=1e-6)

    reduced = decide(
        LongOnlyState(
            exposure=added.resulting_exposure,
            entry_price=added.entry_price,
        ),
        raw_position=0.40,
        price=101.0,
    )
    assert reduced.action == REDUCE
    assert reduced.resulting_exposure == pytest.approx(0.40)
    assert reduced.entry_price == pytest.approx(added.entry_price)

    exited = decide(
        LongOnlyState(
            exposure=reduced.resulting_exposure,
            entry_price=reduced.entry_price,
        ),
        raw_position=-0.20,
        price=101.0,
    )
    assert exited.action == EXIT
    assert exited.resulting_exposure == 0.0
    assert exited.entry_price is None


def test_small_target_change_does_not_move_virtual_position() -> None:
    decision = decide(
        LongOnlyState(exposure=0.50, entry_price=100.0),
        raw_position=0.56,
        price=101.0,
    )

    assert decision.action == HOLD
    assert decision.requested_exposure == pytest.approx(0.56)
    assert decision.resulting_exposure == pytest.approx(0.50)


def test_stop_loss_has_priority_over_stronger_model_signal() -> None:
    decision = decide(
        LongOnlyState(exposure=0.60, entry_price=100.0),
        raw_position=0.90,
        price=97.9,
    )

    assert decision.action == STOP_LOSS
    assert decision.resulting_exposure == 0.0


def test_take_profit_only_reduces_once_per_position_lifecycle() -> None:
    first = decide(
        LongOnlyState(exposure=0.80, entry_price=100.0),
        raw_position=0.80,
        price=104.0,
    )
    assert first.action == TAKE_PROFIT
    assert first.resulting_exposure == pytest.approx(0.40)
    assert first.take_profit_done is True

    second = decide(
        LongOnlyState(
            exposure=first.resulting_exposure,
            entry_price=first.entry_price,
            take_profit_done=first.take_profit_done,
        ),
        raw_position=0.40,
        price=106.0,
    )
    assert second.action == HOLD
    assert second.resulting_exposure == pytest.approx(0.40)


def _ledger_kwargs(**overrides):
    values = {
        "watch_id": "tongdaxin:600519:15m:best:abc",
        "source": "tongdaxin",
        "symbol": "600519",
        "timeframe": "15m",
        "strategy_name": "best_600519",
        "strategy_fingerprint": "abc123",
        "bar_ts": 1000,
        "price": 100.0,
        "raw_position": 0.40,
        "factor_value": 0.423649,
        "strength": 0.40,
        **DEFAULTS,
    }
    values.update(overrides)
    return values


def test_ledger_is_idempotent_across_restart_and_tracks_delivery(tmp_path) -> None:
    db = tmp_path / "signals.sqlite3"
    ledger = SignalLedger(db)

    first, created = ledger.process_bar(**_ledger_kwargs())
    assert created is True
    assert first.action == BUY
    assert first.delivery_status == "PENDING"

    duplicate, created = ledger.process_bar(**_ledger_kwargs())
    assert created is False
    assert duplicate.event_id == first.event_id

    restarted = SignalLedger(db)
    duplicate_after_restart, created = restarted.process_bar(**_ledger_kwargs())
    assert created is False
    assert duplicate_after_restart.event_id == first.event_id

    restarted.record_delivery(first.event_id, "DELIVERED", "ok")
    delivered = restarted.get_event(first.event_id)
    assert delivered is not None
    assert delivered["delivery_status"] == "DELIVERED"
    assert delivered["delivery_attempts"] == 1

    position = restarted.get_position(first.watch_id)
    assert position is not None
    assert position["exposure"] == pytest.approx(0.40)


def test_ledger_preserves_action_sequence_and_rejects_time_reversal(tmp_path) -> None:
    ledger = SignalLedger(tmp_path / "signals.sqlite3")

    bought, _ = ledger.process_bar(**_ledger_kwargs())
    added, created = ledger.process_bar(
        **_ledger_kwargs(
            bar_ts=2000,
            price=101.0,
            raw_position=0.70,
            strength=0.70,
        )
    )
    assert created is True
    assert added.action == ADD

    exited, _ = ledger.process_bar(
        **_ledger_kwargs(
            bar_ts=3000,
            price=102.0,
            raw_position=-0.20,
            strength=0.20,
        )
    )
    assert exited.action == EXIT

    events = ledger.list_events(limit=10)
    assert [row["action"] for row in events] == [EXIT, ADD, BUY]
    assert events[-1]["event_id"] == bought.event_id

    with pytest.raises(ValueError, match="K 线时间未前进"):
        ledger.process_bar(**_ledger_kwargs(bar_ts=2500))
