"""Strategy data_file fallback from training settings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_core.target_contract import (
    SCORING_CONTRACT_VERSION,
    ScoringContractMismatchError,
)
from model_core.vocab import FORMULA_VOCAB
from web import strategy_file as sf
from web.settings import save_settings


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAINING_BACKEND", "local")
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    settings_path = tmp_path / "web_settings.json"
    import web.progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sf, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sf, "checkpoint_glob", lambda _symbol: [])
    import web.settings as settings_mod

    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_mod, "STRATEGIES_DIR", strategies)
    monkeypatch.chdir(tmp_path)

    def fake_inspect(path: str) -> dict:
        return {"symbol": "XAUUSD", "timeframe": "H1", "bars": 100, "valid": True}

    monkeypatch.setattr(sf, "inspect_parquet_file", fake_inspect)
    return tmp_path


def test_inspect_strategy_fills_data_file_from_settings(project: Path) -> None:
    parquet = project / "XAUUSD_H1.parquet"
    parquet.write_bytes(b"PAR1")
    save_settings({"last_data_file": str(parquet.resolve())})

    strat_path = project / "strategies" / "best_XAUUSD.json"
    strat_path.write_text(
        json.dumps(
            {
                "symbol": "XAUUSD",
                "formula": [1, 2, 3],
                "best_score": 1.5,
                "vocab_version": FORMULA_VOCAB.version,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
            }
        ),
        encoding="utf-8",
    )

    info = sf.inspect_strategy_file(str(strat_path))
    assert info["data_file"] == str(parquet.resolve())
    assert info["data_file_exists"] is True
    assert info["timeframe"] == "H1"


def test_sync_best_writes_data_file(project: Path) -> None:
    parquet = project / "XAUUSD_H1.parquet"
    parquet.write_bytes(b"PAR1")
    save_settings({"last_data_file": str(parquet.resolve())})

    strat_path = project / "strategies" / "best_XAUUSD.json"
    strat_path.write_text(
        json.dumps(
            {
                "symbol": "XAUUSD",
                "formula": [1, 2, 3],
                "best_score": 1.5,
                "formula_decoded": "A → B",
                "vocab_version": FORMULA_VOCAB.version,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
            }
        ),
        encoding="utf-8",
    )

    info = sf.sync_best_strategy_for_symbol("XAUUSD")
    assert info is not None
    assert info["data_file"] == str(parquet.resolve())

    saved = json.loads(strat_path.read_text(encoding="utf-8"))
    assert saved["data_file"] == str(parquet.resolve())


def _complete_strategy() -> dict:
    digest = "a" * 64
    return {
        "vocab_version": FORMULA_VOCAB.version,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "formula": [1, 2, 3],
        "best_score": 1.5,
        "local_source": "mt5",
        "periods_per_year": 6240,
        "minimum_bars": 3000,
        "dataset_id": f"sha256:{digest}",
        "data_sha256": digest,
        "data_rows": 3000,
        "data_start": "2023-01-01T00:00:00Z",
        "data_end": "2024-01-01T00:00:00Z",
        "columns": [
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
        ],
    }


def test_complete_strategy_identity_is_registered_for_backtest(project: Path) -> None:
    path = project / "strategies" / "best_XAUUSD.json"
    path.write_text(json.dumps(_complete_strategy()), encoding="utf-8")

    info = sf.inspect_strategy_file(str(path))

    assert info["identity_registration"] == "registered"


@pytest.mark.parametrize("version", [None, "previous_contract"], ids=("missing", "previous"))
def test_inspect_marks_previous_scoring_contract_as_diagnostic_only(
    project: Path,
    version: str | None,
) -> None:
    path = project / "strategies" / "best_XAUUSD.json"
    strategy = _complete_strategy()
    if version is None:
        strategy.pop("scoring_contract_version")
    else:
        strategy["scoring_contract_version"] = version
    path.write_text(json.dumps(strategy), encoding="utf-8")

    info = sf.inspect_strategy_file(str(path))

    assert info["score_compatible"] is False
    assert info["valid"] is False
    assert "不能正式回测" in info["message"]


@pytest.mark.parametrize("field", ["columns", "data_start", "data_end", "data_rows"])
def test_incomplete_oos_identity_stays_legacy_unknown(
    project: Path,
    field: str,
) -> None:
    strategy = _complete_strategy()
    strategy.pop(field)
    path = project / "strategies" / "best_XAUUSD.json"
    path.write_text(json.dumps(strategy), encoding="utf-8")

    info = sf.inspect_strategy_file(str(path))

    assert info["identity_registration"] == "legacy_unknown"


@pytest.mark.parametrize("version", [None, "vprevious0000"], ids=("missing", "previous"))
def test_inspect_rejects_previous_formula_execution_version(
    project: Path,
    version: str | None,
) -> None:
    path = project / "strategies" / "best_XAUUSD.json"
    strategy = _complete_strategy()
    if version is None:
        strategy.pop("vocab_version")
    else:
        strategy["vocab_version"] = version
    path.write_text(json.dumps(strategy), encoding="utf-8")

    with pytest.raises(ValueError, match="词表版本不匹配"):
        sf.inspect_strategy_file(str(path))


@pytest.mark.parametrize("version", [None, "vprevious0000"], ids=("missing", "previous"))
def test_local_strategy_list_excludes_previous_formula_execution_version(
    project: Path,
    version: str | None,
) -> None:
    import web.progress as progress_mod

    old_path = project / "strategies" / "best_XAUUSD.json"
    old_strategy = _complete_strategy()
    if version is None:
        old_strategy.pop("vocab_version")
    else:
        old_strategy["vocab_version"] = version
    old_path.write_text(json.dumps(old_strategy), encoding="utf-8")

    current_path = project / "strategies" / "best_EURUSD.json"
    current_strategy = _complete_strategy()
    current_strategy["symbol"] = "EURUSD"
    current_path.write_text(json.dumps(current_strategy), encoding="utf-8")

    rows = progress_mod.list_strategies()

    assert [row["file"] for row in rows] == ["best_EURUSD.json"]


@pytest.mark.parametrize("version", [None, "vprevious0000"], ids=("missing", "previous"))
def test_backtest_manager_rejects_previous_version_before_starting_process(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
) -> None:
    import web.backtest_manager as backtest_mod

    strategy_path = project / "strategies" / "best_XAUUSD.json"
    strategy = _complete_strategy()
    if version is None:
        strategy.pop("vocab_version")
    else:
        strategy["vocab_version"] = version
    strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
    data_path = project / "XAUUSD_H1.parquet"
    data_path.write_bytes(b"PAR1")

    def unexpected_popen(*_args, **_kwargs):
        pytest.fail("旧公式执行版本不得创建回测子进程")

    monkeypatch.setattr(backtest_mod.subprocess, "Popen", unexpected_popen)
    manager = backtest_mod.BacktestManager()

    with pytest.raises(ValueError, match="词表版本不匹配"):
        manager.start(
            strategy_file=str(strategy_path),
            data_file=str(data_path),
        )

    assert manager.status()["job"] is None


@pytest.mark.parametrize("version", [None, "previous_contract"], ids=("missing", "previous"))
def test_backtest_manager_rejects_previous_scoring_contract_before_process(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
) -> None:
    import web.backtest_manager as backtest_mod

    strategy_path = project / "strategies" / "best_XAUUSD.json"
    strategy = _complete_strategy()
    if version is None:
        strategy.pop("scoring_contract_version")
    else:
        strategy["scoring_contract_version"] = version
    strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
    data_path = project / "XAUUSD_H1.parquet"
    data_path.write_bytes(b"PAR1")

    def unexpected_popen(*_args, **_kwargs):
        pytest.fail("旧评分合同不得创建回测子进程")

    monkeypatch.setattr(backtest_mod.subprocess, "Popen", unexpected_popen)
    manager = backtest_mod.BacktestManager()

    with pytest.raises(ValueError, match="不能正式回测"):
        manager.start(
            strategy_file=str(strategy_path),
            data_file=str(data_path),
        )

    assert manager.status()["job"] is None


def test_sync_skips_previous_contract_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_path = project / "old.pt"
    current_path = project / "current.pt"
    monkeypatch.setattr(sf, "checkpoint_glob", lambda _symbol: [old_path, current_path])

    def fake_meta(path: Path) -> dict:
        if path == old_path:
            raise ScoringContractMismatchError("旧评分合同")
        return {
            "best_score": 2.0,
            "best_formula": [1, 2, 3],
            "step": 7,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
        }

    monkeypatch.setattr(sf, "_load_checkpoint_meta", fake_meta)

    info = sf.sync_best_strategy_for_symbol("XAUUSD")

    assert info is not None
    assert info["best_score"] == pytest.approx(2.0)


def test_sync_does_not_fall_back_to_only_previous_contract_strategy(
    project: Path,
) -> None:
    path = project / "strategies" / "best_XAUUSD.json"
    strategy = _complete_strategy()
    strategy["scoring_contract_version"] = "previous_contract"
    path.write_text(json.dumps(strategy), encoding="utf-8")

    assert sf.sync_best_strategy_for_symbol("XAUUSD") is None
