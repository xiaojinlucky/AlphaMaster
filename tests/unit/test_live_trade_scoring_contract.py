"""实盘品种筛选只能消费当前评分合同的扫描报告。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import live_trade
import monitor_live_risk
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import FORMULA_VOCAB
from scan_all_factors import factor_scan_execution_contract
from strategy_manager.runner import formula_set_sha256


def _write_strategy(path: Path, formula: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "vocab_version": FORMULA_VOCAB.version,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "formula": formula,
                "best_score": 1.0,
            }
        ),
        encoding="utf-8",
    )


def _write_scan(
    tmp_path: Path,
    contract: str | None,
    *,
    rows: list[dict] | None = None,
) -> None:
    output = tmp_path / "backtest_output"
    output.mkdir(exist_ok=True)
    source_rows = (
        rows
        if rows is not None
        else [
            {
                "symbol": "US100.cash",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0]]),
            },
            {
                "symbol": "XAGUSD",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0]]),
            },
        ]
    )
    normalized_rows = []
    for source_row in source_rows:
        row = dict(source_row)
        symbol = row.get("symbol")
        if isinstance(symbol, str):
            row.setdefault(
                "execution_contract",
                factor_scan_execution_contract(symbol),
            )
        normalized_rows.append(row)
    payload = {"valid": normalized_rows}
    if contract is not None:
        payload["scoring_contract_version"] = contract
    (output / "factor_scan.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_live_trade_accepts_current_scan_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    _write_strategy(tmp_path / "strategies" / "best_US100.cash.json", [0])
    _write_scan(tmp_path, SCORING_CONTRACT_VERSION)

    symbols = live_trade._load_valid_from_scan()

    assert symbols is not None
    assert "US100.cash" in symbols
    assert "XAGUSD" not in symbols


def test_monitor_and_trade_resolve_same_scan_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    _write_strategy(tmp_path / "strategies" / "best_COCOA.c.json", [0])
    _write_scan(
        tmp_path,
        SCORING_CONTRACT_VERSION,
        rows=[
            {
                "symbol": "COCOA.c",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0]]),
            }
        ],
    )

    assert monitor_live_risk._symbols() == live_trade.resolve_live_symbols()
    assert monitor_live_risk._symbols() == ["COCOA.c"]


def test_monitor_and_trade_resolve_same_explicit_override() -> None:
    override = ["US30.cash"]

    assert monitor_live_risk._symbols(override) == live_trade.resolve_live_symbols(override)
    assert live_trade.resolve_live_symbols(override) == ["US30.cash"]


@pytest.mark.parametrize("contract", [None, "previous_contract"], ids=("missing", "previous"))
def test_live_trade_rejects_previous_scan_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: str | None,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    _write_scan(tmp_path, contract)

    with pytest.raises(RuntimeError, match="评分合同不兼容"):
        live_trade._load_valid_from_scan()


def test_live_trade_rejects_missing_scan_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))

    with pytest.raises(RuntimeError, match="扫描报告不存在"):
        live_trade.resolve_live_symbols()


def test_live_trade_rejects_empty_current_scan_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    _write_scan(tmp_path, SCORING_CONTRACT_VERSION, rows=[])

    with pytest.raises(RuntimeError, match="没有合格品种"):
        live_trade.resolve_live_symbols()


def test_live_trade_rejects_scan_with_only_excluded_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    _write_scan(
        tmp_path,
        SCORING_CONTRACT_VERSION,
        rows=[
            {
                "symbol": "XAGUSD",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0]]),
            }
        ],
    )

    with pytest.raises(RuntimeError, match="过滤后没有合格品种"):
        live_trade.resolve_live_symbols()


def test_live_trade_rejects_formula_replaced_after_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    strategy = tmp_path / "strategies" / "best_US30.cash.json"
    _write_strategy(strategy, [0])
    _write_scan(
        tmp_path,
        SCORING_CONTRACT_VERSION,
        rows=[
            {
                "symbol": "US30.cash",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0]]),
            }
        ],
    )
    _write_strategy(strategy, [1])

    with pytest.raises(RuntimeError, match="公式集合与扫描报告不一致"):
        live_trade.resolve_live_symbols()


def test_live_trade_binds_runner_multi_formula_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    strategies = tmp_path / "strategies"
    _write_strategy(strategies / "best_XAUUSD.json", [0])
    _write_strategy(strategies / "best_metals_comm.json", [1])
    _write_scan(
        tmp_path,
        SCORING_CONTRACT_VERSION,
        rows=[
            {
                "symbol": "XAUUSD",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0], [1]]),
            }
        ],
    )

    plan = live_trade.resolve_live_strategy_plan()

    assert list(plan.symbols) == ["XAUUSD"]
    assert plan.expected_formula_set_sha256 == {
        "XAUUSD": formula_set_sha256([[0], [1]])
    }


def test_live_trade_cli_does_not_consume_flags_as_symbols() -> None:
    args = live_trade._parse_args(
        ["--symbols", "US30.cash", "--dry-run"]
    )

    assert args.symbols == ["US30.cash"]
    assert args.dry_run is True


def test_live_trade_rejects_removed_fake_single_mode() -> None:
    with pytest.raises(SystemExit):
        live_trade._parse_args(["--single"])


def test_live_trade_passes_scan_hashes_to_final_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"US30.cash": formula_set_sha256([[0]])}
    min_exposure = float(live_trade.Config.MIN_TRADE_EXPOSURE)
    plan = live_trade.LiveStrategyPlan(
        ("US30.cash",),
        expected,
        min_exposure,
    )
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self) -> None:
            captured["ran"] = True

        def shutdown(self) -> None:
            captured["shutdown"] = True

    monkeypatch.setattr(
        live_trade,
        "resolve_live_strategy_plan",
        lambda _symbols: plan,
    )
    monkeypatch.setattr(live_trade, "MT5StrategyRunner", FakeRunner)

    live_trade.main([])

    assert captured["expected_formula_set_sha256"] == expected
    assert captured["min_trade_exposure"] == min_exposure
    assert captured["dry_run"] is False
    assert captured["ran"] is True
    assert captured["shutdown"] is True


def test_live_trade_rejects_threshold_changed_after_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_trade, "_dir", str(tmp_path))
    strategy = tmp_path / "strategies" / "best_US30.cash.json"
    _write_strategy(strategy, [0])
    _write_scan(
        tmp_path,
        SCORING_CONTRACT_VERSION,
        rows=[
            {
                "symbol": "US30.cash",
                "valid": True,
                "formula_set_sha256": formula_set_sha256([[0]]),
            }
        ],
    )
    monkeypatch.setattr(live_trade.Config, "MIN_TRADE_EXPOSURE", 0.9)

    with pytest.raises(RuntimeError, match="执行合同与当前配置不一致"):
        live_trade.resolve_live_strategy_plan()
