from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import web.progress as progress_module
from model_core.formula_contract import (
    FORMULA_EXECUTION_CONTRACT,
    STACKVM_JUMP_EPS,
    STACKVM_JUMP_THRESHOLD,
    STACKVM_JUMP_WINDOW,
    STACKVM_OUTPUT_NORM_CLIP,
    STACKVM_OUTPUT_NORM_WINDOW,
)
from model_core.vocab import (
    FORMULA_VOCAB,
    VOCAB_VERSION,
    VocabVersionMismatchError,
)
from run_backtest import load_strategy
from web.realtime_manager import _load_strategy_meta


def _token_only_version() -> str:
    joined = "\n".join(FORMULA_VOCAB.token_names)
    return "v" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _strategy(version: str) -> dict[str, object]:
    return {
        "vocab_version": version,
        "symbol": "XAUUSD",
        "formula": [0],
        "best_score": 1.0,
    }


def test_compatibility_version_includes_fixed_execution_contract() -> None:
    legacy_version = _token_only_version()

    assert VOCAB_VERSION != legacy_version
    assert f"w{STACKVM_OUTPUT_NORM_WINDOW}" in FORMULA_EXECUTION_CONTRACT
    assert f"clip-{int(STACKVM_OUTPUT_NORM_CLIP)}" in FORMULA_EXECUTION_CONTRACT
    assert "jump-v1" in FORMULA_EXECUTION_CONTRACT
    assert (
        f"zscore-w{STACKVM_JUMP_WINDOW}-current-inclusive"
        in FORMULA_EXECUTION_CONTRACT
    )
    assert f"eps-{STACKVM_JUMP_EPS!r}" in FORMULA_EXECUTION_CONTRACT
    assert "nonfinite-to-zero" in FORMULA_EXECUTION_CONTRACT
    assert f"threshold-{STACKVM_JUMP_THRESHOLD}" in FORMULA_EXECUTION_CONTRACT
    with pytest.raises(VocabVersionMismatchError):
        FORMULA_VOCAB.verify(legacy_version)
    FORMULA_VOCAB.verify(VOCAB_VERSION)


def test_backtest_and_realtime_reject_previous_execution_semantics(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(_strategy(_token_only_version())),
        encoding="utf-8",
    )

    with pytest.raises(VocabVersionMismatchError):
        load_strategy(legacy)
    with pytest.raises(VocabVersionMismatchError):
        _load_strategy_meta(str(legacy))

    current = tmp_path / "current.json"
    current.write_text(
        json.dumps(_strategy(VOCAB_VERSION)),
        encoding="utf-8",
    )
    assert load_strategy(current) == _strategy(VOCAB_VERSION)
    assert _load_strategy_meta(str(current))["vocab_version"] == VOCAB_VERSION


def test_web_progress_cannot_relabel_old_strategy_or_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAINING_BACKEND", raising=False)
    monkeypatch.setattr(progress_module, "STRATEGIES_DIR", tmp_path)

    strategy_path = tmp_path / "best_XAUUSD.json"
    strategy_path.write_text(
        json.dumps(_strategy(_token_only_version())),
        encoding="utf-8",
    )
    assert progress_module._load_strategy("XAUUSD") is None

    strategy_path.write_text(
        json.dumps(_strategy(VOCAB_VERSION)),
        encoding="utf-8",
    )
    assert progress_module._load_strategy("XAUUSD") == _strategy(VOCAB_VERSION)

    checkpoint = tmp_path / "ckpt_XAUUSD_step_0010.pt"
    torch.save(
        {
            "step": 10,
            "vocab_version": _token_only_version(),
            "training_history": {},
        },
        checkpoint,
    )
    progress_module.invalidate_checkpoint_cache()
    with pytest.raises(VocabVersionMismatchError):
        progress_module._load_checkpoint_meta(checkpoint)

    torch.save(
        {
            "step": 10,
            "vocab_version": VOCAB_VERSION,
            "training_history": {},
        },
        checkpoint,
    )
    progress_module.invalidate_checkpoint_cache()
    assert progress_module._load_checkpoint_meta(checkpoint)["step"] == 10
