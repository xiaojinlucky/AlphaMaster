from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import portfolio_manager.controller as controller_module
import portfolio_manager.universe as universe_module
from portfolio_manager.controller import (
    ModelSignalSnapshot,
    PortfolioPolicy,
    build_csi_a50_portfolio_decision,
)
from portfolio_manager.universe import (
    UniverseContract,
    load_csi_a50_universe_contract,
)

SYMBOLS = ("000001", "000002", "000003", "000004", "000005")
UNIVERSE = UniverseContract(
    universe_id="test-universe:20260723",
    snapshot_date="20260723",
    constituent_count=len(SYMBOLS),
    universe_sha256="e" * 64,
    symbols=SYMBOLS,
)
POLICY = PortfolioPolicy(
    top_k=2,
    dropout_rank=3,
    minimum_history=5,
    minimum_model_exposure=0.05,
    minimum_confidence=0.20,
    minimum_calibrated_score=0.50,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signal(
    symbol: str,
    *,
    raw_score: float,
    history_scores: tuple[float, ...] = (0, 1, 2, 3, 4),
    bar_ts: int = 1_784_790_000,
    requested_exposure: float = 0.5,
    confidence: float = 0.8,
    model_exit: bool = False,
) -> ModelSignalSnapshot:
    return ModelSignalSnapshot(
        run_id=f"run_20260723T235959Z_{_sha(symbol)[:8]}",
        symbol=symbol,
        bar_ts=bar_ts,
        session_date="2026-07-23",
        timeframe="1d",
        market_source="akshare_sina_hfq_ohlcv",
        raw_score=raw_score,
        requested_exposure=requested_exposure,
        confidence=confidence,
        model_version=_sha(f"model:{symbol}"),
        data_version=_sha(f"data:{symbol}"),
        calibration_version="alphamaster_rolling_factor_calibration_v1",
        calibration_history_sha256=_sha(f"history:{symbol}"),
        history_scores=history_scores,
        model_exit=model_exit,
    )


def _signals(scores: dict[str, float]) -> list[ModelSignalSnapshot]:
    return [_signal(symbol, raw_score=scores[symbol]) for symbol in SYMBOLS]


def _decision(
    signals,
    *,
    current_weights=None,
    policy=POLICY,
    universe=UNIVERSE,
    previous_decision_ts=None,
):
    return controller_module._build_portfolio_decision(
        signals,
        universe=universe,
        current_weights=current_weights or {},
        account_snapshot_sha256="a" * 64,
        policy=policy,
        previous_decision_ts=previous_decision_ts,
    )


def test_initial_top_k_uses_self_history_percentile_and_equal_slots() -> None:
    decision = _decision(
        _signals(
            {
                "000001": 5.0,
                "000002": 4.0,
                "000003": 3.0,
                "000004": 1.0,
                "000005": -1.0,
            }
        )
    )

    assert decision.selected_symbols == ("000001", "000002")
    assert dict(decision.target_weights) == {
        "000001": pytest.approx(0.5),
        "000002": pytest.approx(0.5),
    }
    assert decision.to_dict()["universe"] == UNIVERSE.to_dict()
    assert decision.to_dict()["policy"]["top_k"] == 2
    assert decision.entered_symbols == ("000001", "000002")
    assert decision.retained_symbols == ()
    assert decision.exited_symbols == ()
    assert decision.cash_weight == pytest.approx(0.0)


def test_dropout_buffer_retains_rank_three_and_replaces_worse_holding() -> None:
    decision = _decision(
        _signals(
            {
                "000001": 5.0,
                "000002": 4.0,
                "000003": 3.0,
                "000004": 1.0,
                "000005": -1.0,
            }
        ),
        current_weights={"000003": 0.5, "000004": 0.5},
    )

    assert decision.selected_symbols == ("000003", "000001")
    assert decision.retained_symbols == ("000003",)
    assert decision.entered_symbols == ("000001",)
    assert decision.exited_symbols == ("000004",)
    assert dict(decision.current_weights) == {"000003": 0.5, "000004": 0.5}


def test_model_exit_overrides_dropout_retention() -> None:
    signals = _signals(
        {
            "000001": 5.0,
            "000002": 4.0,
            "000003": 3.0,
            "000004": 1.0,
            "000005": -1.0,
        }
    )
    signals[0] = _signal("000001", raw_score=5.0, model_exit=True)

    decision = _decision(signals, current_weights={"000001": 0.5})

    assert decision.selected_symbols == ("000002", "000003")
    assert decision.exited_symbols == ("000001",)
    rejected = next(row for row in decision.ranking if row.symbol == "000001")
    assert rejected.eligible is False
    assert rejected.rejection_reason == "单股模型主动离场"


def test_unfilled_slots_remain_cash_instead_of_concentrating() -> None:
    signals = [
        _signal(
            symbol,
            raw_score=5.0 if symbol == "000001" else -1.0,
        )
        for symbol in SYMBOLS
    ]

    decision = _decision(signals)

    assert decision.selected_symbols == ("000001",)
    assert dict(decision.target_weights) == {"000001": pytest.approx(0.5)}
    assert decision.cash_weight == pytest.approx(0.5)


def test_decision_identity_is_independent_of_input_order() -> None:
    signals = _signals(
        {
            "000001": 5.0,
            "000002": 4.0,
            "000003": 3.0,
            "000004": 1.0,
            "000005": -1.0,
        }
    )

    forward = _decision(signals, current_weights={"000003": 0.5})
    reverse = _decision(
        reversed(signals),
        current_weights={"000003": 0.5},
    )

    assert forward.decision_id == reverse.decision_id
    assert forward.to_dict() == reverse.to_dict()


def test_incomplete_or_misaligned_snapshot_fails_closed() -> None:
    complete = _signals({symbol: 5.0 for symbol in SYMBOLS})

    with pytest.raises(ValueError, match="完整覆盖"):
        _decision(complete[:-1])

    complete[-1] = replace(
        _signal("000005", raw_score=5.0),
        bar_ts=1_784_876_400,
        session_date="2026-07-24",
    )
    with pytest.raises(ValueError, match="同一根"):
        _decision(complete)


def test_real_a50_contract_rejects_four_ready_symbols() -> None:
    universe = load_csi_a50_universe_contract()
    four_symbols = universe.symbols[:4]
    four_signals = [_signal(symbol, raw_score=5.0) for symbol in four_symbols]

    with pytest.raises(ValueError, match="完整覆盖"):
        build_csi_a50_portfolio_decision(
            four_signals,
            current_weights={},
            account_snapshot_sha256="a" * 64,
            policy=POLICY,
        )


def test_verified_a50_entry_accepts_exact_fifty_signal_contract() -> None:
    universe = load_csi_a50_universe_contract()
    signals = [
        _signal(symbol, raw_score=5.0 - index / 100)
        for index, symbol in enumerate(universe.symbols)
    ]
    policy = replace(POLICY, top_k=5, dropout_rank=10)

    decision = build_csi_a50_portfolio_decision(
        signals,
        current_weights={},
        account_snapshot_sha256="a" * 64,
        policy=policy,
    )

    assert decision.universe.constituent_count == 50
    assert decision.universe.universe_sha256 == universe.universe_sha256
    assert len(decision.ranking) == 50
    assert len(decision.selected_symbols) == 5


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"timeframe": "1h"}, "只接受 1d"),
        ({"market_source": "tongdaxin"}, "akshare_sina_hfq_ohlcv"),
    ],
)
def test_verified_a50_entry_rejects_consistently_wrong_signal_contract(
    changes,
    message,
) -> None:
    universe = load_csi_a50_universe_contract()
    signals = [
        replace(_signal(symbol, raw_score=5.0), **changes)
        for symbol in universe.symbols
    ]

    with pytest.raises(ValueError, match=message):
        build_csi_a50_portfolio_decision(
            signals,
            current_weights={},
            account_snapshot_sha256="a" * 64,
            policy=replace(POLICY, top_k=5, dropout_rank=10),
        )


def test_a50_loader_rejects_rewritten_constituent_with_recomputed_hash(
    monkeypatch,
    tmp_path,
) -> None:
    payload = json.loads(
        universe_module._CSI_A50_UNIVERSE_PATH.read_text(encoding="utf-8")
    )
    payload["constituents"][0]["symbol"] = "999999"
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    forged_path = tmp_path / "forged_a50.json"
    forged_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        universe_module,
        "_CSI_A50_UNIVERSE_PATH",
        forged_path,
    )

    with pytest.raises(ValueError, match="内置可信"):
        load_csi_a50_universe_contract()


def test_time_source_and_timeframe_must_align() -> None:
    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    with pytest.raises(ValueError, match="时间未前进"):
        _decision(signals, previous_decision_ts=1_784_790_000)

    signals[-1] = replace(signals[-1], timeframe="1h")
    with pytest.raises(ValueError, match="同一周期"):
        _decision(signals)

    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    signals[-1] = replace(signals[-1], market_source="tongdaxin")
    with pytest.raises(ValueError, match="同一行情来源"):
        _decision(signals)


def test_current_weights_and_calibration_numbers_are_strict() -> None:
    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    with pytest.raises(ValueError, match="不能超过 1"):
        _decision(
            signals,
            current_weights={"000001": 0.6, "000002": 0.5},
        )

    signals[0] = replace(signals[0], history_scores=(1.0, 2.0))
    with pytest.raises(ValueError, match="历史校准样本不足"):
        _decision(signals)

    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    signals[0] = replace(signals[0], raw_score=float("nan"))
    with pytest.raises(ValueError, match="raw_score"):
        _decision(signals)


@pytest.mark.parametrize(
    "policy",
    [
        replace(POLICY, top_k=1.9),
        replace(POLICY, dropout_rank=True),
        replace(POLICY, gross_exposure="1.0"),
    ],
)
def test_policy_rejects_implicit_numeric_coercion(policy) -> None:
    with pytest.raises(ValueError):
        _decision(_signals({symbol: 5.0 for symbol in SYMBOLS}), policy=policy)


def test_signal_rejects_implicit_type_coercion() -> None:
    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    signals[0] = replace(signals[0], bar_ts=1_000.9)
    with pytest.raises(ValueError, match="bar_ts"):
        _decision(signals)

    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    signals[0] = replace(signals[0], model_exit="false")
    with pytest.raises(ValueError, match="model_exit"):
        _decision(signals)


def test_equal_percentile_uses_symbol_not_raw_confidence_as_tie_break() -> None:
    signals = _signals({symbol: 2.0 for symbol in SYMBOLS})
    signals[0] = replace(signals[0], confidence=0.2)
    signals[1] = replace(signals[1], confidence=0.9)

    decision = _decision(signals)

    assert decision.selected_symbols == ("000001", "000002")


def test_decision_id_covers_raw_score_and_full_ranking_payload() -> None:
    first = _decision(_signals({symbol: 5.0 for symbol in SYMBOLS}))
    changed = _signals({symbol: 5.0 for symbol in SYMBOLS})
    changed[0] = replace(changed[0], raw_score=6.0)
    second = _decision(changed)

    assert first.decision_id != second.decision_id


def test_duplicate_symbols_and_invalid_identity_fail_closed() -> None:
    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    signals[-1] = _signal("000001", raw_score=5.0)
    with pytest.raises(ValueError, match="重复股票"):
        _decision(signals)

    signals = _signals({symbol: 5.0 for symbol in SYMBOLS})
    signals[0] = replace(signals[0], model_version=" ")
    with pytest.raises(ValueError, match="model_version"):
        _decision(signals)
