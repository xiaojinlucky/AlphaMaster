"""独立研究脚本不得把旧训练分数混入当前评分合同报告。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import backtest_current
import rigorous_backtest_audit
import verify_all_strategies
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import FORMULA_VOCAB


def _strategy(version: str | None) -> dict:
    payload = {
        "vocab_version": FORMULA_VOCAB.version,
        "formula": [1, 2, 3],
        "best_score": 9.99,
    }
    if version is not None:
        payload["scoring_contract_version"] = version
    return payload


@pytest.mark.parametrize("version", [None, "previous_contract"], ids=("missing", "previous"))
def test_backtest_current_rejects_previous_scoring_contract(
    tmp_path: Path,
    version: str | None,
) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(_strategy(version)), encoding="utf-8")

    assert backtest_current.load_formula(path) is None


@pytest.mark.parametrize("version", [None, "previous_contract"], ids=("missing", "previous"))
def test_rigorous_audit_rejects_previous_scoring_contract(
    tmp_path: Path,
    version: str | None,
) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(_strategy(version)), encoding="utf-8")

    loaded = rigorous_backtest_audit.load_candidate(str(path))

    assert loaded is not None
    assert "scoring contract mismatch" in loaded["error"]


@pytest.mark.parametrize("version", [None, "previous_contract"], ids=("missing", "previous"))
def test_verify_all_rejects_previous_scoring_contract(
    tmp_path: Path,
    version: str | None,
) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(_strategy(version)), encoding="utf-8")

    loaded = verify_all_strategies.load_strategy(str(path))

    assert "scoring contract mismatch" in loaded["error"]


def test_research_loaders_accept_current_scoring_contract(tmp_path: Path) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps(_strategy(SCORING_CONTRACT_VERSION)),
        encoding="utf-8",
    )

    assert backtest_current.load_formula(path) is not None
    assert "error" not in rigorous_backtest_audit.load_candidate(str(path))
    assert "error" not in verify_all_strategies.load_strategy(str(path))


@pytest.mark.parametrize(
    "filename",
    [
        "run_backtest.py",
        "backtest_current.py",
        "backtest_all_groups.py",
        "backtest_detailed.py",
        "backtest_index_best.py",
        "deep_backtest_metals.py",
        "generate_factor_equity_curves.py",
        "scan_all_factors.py",
        "rigorous_backtest_audit.py",
        "verify_all_strategies.py",
        "backtest_xauusd.py",
    ],
)
def test_current_backtest_reports_declare_scoring_contract(filename: str) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / filename).read_text(encoding="utf-8")

    assert '"scoring_contract_version"' in source
    assert "SCORING_CONTRACT_VERSION" in source
