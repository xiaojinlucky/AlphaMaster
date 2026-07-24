"""因子扫描必须绑定 Runner 真正执行的完整公式集合。"""
from __future__ import annotations

import json
from pathlib import Path

from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import FORMULA_VOCAB
from scan_all_factors import discover_formula_sets


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


def test_scan_discovers_same_multi_formula_set_as_runner(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies"
    _write_strategy(strategies / "best_XAUUSD.json", [0])
    _write_strategy(strategies / "best_metals_comm.json", [1])

    assert discover_formula_sets(strategies) == {"XAUUSD": [[0], [1]]}
