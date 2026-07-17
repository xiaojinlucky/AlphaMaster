from __future__ import annotations

import hashlib
import json
import inspect
import os
from pathlib import Path

import pytest
import torch

import model_core.engine as engine_module
import train_file
from model_core.engine import (
    AlphaEngine,
    CHECKPOINT_IDENTITY_FIELDS,
    CheckpointIdentityError,
)
from model_core.vocab import FORMULA_VOCAB, VocabVersionMismatchError


class _TinyState:
    def __init__(self, value: int) -> None:
        self.value = value
        self.loaded: list[dict] = []

    def state_dict(self) -> dict:
        return {"value": torch.tensor([self.value])}

    def load_state_dict(self, state: dict) -> None:
        self.loaded.append(state)


def _identity(digest: str = "a" * 64) -> dict:
    return {
        "symbol": "BTCUSDT",
        "timeframe": "H1",
        "dataset_id": f"sha256:{digest}",
        "data_sha256": digest,
        "local_source": "okx",
        "periods_per_year": 6240,
        "minimum_bars": 3000,
    }


def _legacy_token_only_vocab_version() -> str:
    joined = "\n".join(FORMULA_VOCAB.token_names)
    return "v" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _engine(identity: dict | None = None) -> AlphaEngine:
    identity = identity or _identity()
    engine = object.__new__(AlphaEngine)
    engine.target_symbol = identity["symbol"]
    for field in CHECKPOINT_IDENTITY_FIELDS:
        if field != "symbol":
            setattr(engine, field, identity[field])
    engine.model = _TinyState(1)
    engine.opt = _TinyState(2)
    engine.best_score = 3.5
    engine.best_formula = [1, 2]
    engine._best_snapshot = None
    engine.factor_pool = []
    engine._factor_pool_counter = 0
    engine._elite_pool = []
    engine._elite_counter = 0
    engine._restart_count = 0
    engine.training_history = {"step": [20], "_low_entropy_streak": 4}
    return engine


def test_live_strategy_save_is_atomic_and_preserves_full_training_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = _engine()
    source.data_file = str((tmp_path / "BTCUSDT_H1.parquet").resolve())
    source.mode = "parquet_file"
    source.train_steps = 9000
    source.data_rows = 50_000
    source.data_start = "2020-01-01T00:00:00Z"
    source.data_end = "2026-01-01T00:00:00Z"
    source.data_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
    ]

    source._save_strategy_live()

    strategy = json.loads(
        (tmp_path / "strategies" / "best_BTCUSDT.json").read_text(
            encoding="utf-8"
        )
    )
    assert strategy["timeframe"] == "H1"
    assert strategy["dataset_id"] == f"sha256:{'a' * 64}"
    assert strategy["data_sha256"] == "a" * 64
    assert strategy["local_source"] == "okx"
    assert strategy["data_rows"] == 50_000
    assert strategy["data_start"] == "2020-01-01T00:00:00Z"
    assert strategy["data_end"] == "2026-01-01T00:00:00Z"
    assert strategy["columns"] == source.data_columns
    assert not list((tmp_path / "strategies").glob("*.partial"))


def test_training_loop_has_no_minimal_strategy_overwrite_path() -> None:
    source = inspect.getsource(AlphaEngine.train)
    assert "json.dump(strategy_data" not in source
    assert source.count("self._save_strategy_live()") >= 3


def test_checkpoint_round_trip_saves_full_identity_and_isolated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", root)
    source = _engine()

    checkpoint = Path(source.save_checkpoint(20))

    assert checkpoint.parent == root / "H1" / ("a" * 64)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert {field: payload[field] for field in CHECKPOINT_IDENTITY_FIELDS} == _identity()
    assert "_low_entropy_streak" not in payload["training_history"]

    target = _engine()
    assert target.load_checkpoint(str(checkpoint)) == 20
    assert len(target.model.loaded) == 1
    assert len(target.opt.loaded) == 1


def test_checkpoint_save_and_load_pass_open_binary_handles_to_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    real_save = torch.save
    real_load = torch.load
    observed: dict[str, object] = {}

    def save_spy(payload: dict, destination: object) -> None:
        observed["save"] = destination
        assert callable(getattr(destination, "write", None))
        real_save(payload, destination)

    def load_spy(source: object, **kwargs: object) -> dict:
        observed["load"] = source
        assert callable(getattr(source, "read", None))
        return real_load(source, **kwargs)

    monkeypatch.setattr(engine_module.torch, "save", save_spy)
    monkeypatch.setattr(engine_module.torch, "load", load_spy)
    source = _engine()
    source.save_checkpoint(20, str(checkpoint))
    target = _engine()

    assert target.load_checkpoint(str(checkpoint)) == 20
    assert observed.keys() == {"save", "load"}


def test_windows_checkpoint_path_uses_extended_prefix(tmp_path: Path) -> None:
    path = engine_module._checkpoint_filesystem_path(tmp_path / "checkpoint.pt")
    if engine_module.sys.platform == "win32":
        assert str(path).startswith("\\\\?\\")
    else:
        assert path == tmp_path / "checkpoint.pt"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("symbol", "ETHUSDT"),
        ("timeframe", "M15"),
        ("local_source", "mt5"),
        ("periods_per_year", 968),
        ("minimum_bars", 1936),
    ],
)
def test_checkpoint_identity_mismatch_is_rejected_before_state_apply(
    tmp_path: Path, field: str, replacement: object
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _engine().save_checkpoint(20, str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload[field] = replacement
    torch.save(payload, checkpoint)
    target = _engine()

    with pytest.raises(CheckpointIdentityError, match=field):
        target.load_checkpoint(str(checkpoint))

    assert target.model.loaded == []
    assert target.opt.loaded == []


def test_checkpoint_dataset_hash_mismatch_is_rejected_before_state_apply(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _engine().save_checkpoint(20, str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["data_sha256"] = "b" * 64
    payload["dataset_id"] = "sha256:" + "b" * 64
    torch.save(payload, checkpoint)
    target = _engine()

    with pytest.raises(CheckpointIdentityError, match="dataset_id"):
        target.load_checkpoint(str(checkpoint))

    assert target.model.loaded == []
    assert target.opt.loaded == []


def test_checkpoint_from_previous_execution_contract_is_rejected_before_state_apply(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _engine().save_checkpoint(20, str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["vocab_version"] = _legacy_token_only_vocab_version()
    torch.save(payload, checkpoint)
    target = _engine()

    with pytest.raises(VocabVersionMismatchError):
        target.load_checkpoint(str(checkpoint))

    assert target.model.loaded == []
    assert target.opt.loaded == []


def test_checkpoint_rejects_bool_for_integer_identity_before_state_apply(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _engine().save_checkpoint(20, str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["periods_per_year"] = True
    torch.save(payload, checkpoint)
    target = _engine()

    with pytest.raises(CheckpointIdentityError, match="periods_per_year"):
        target.load_checkpoint(str(checkpoint))

    assert target.model.loaded == []
    assert target.opt.loaded == []


@pytest.mark.parametrize(
    "dataset_id",
    ["sha256:../escape", "sha256:" + "A" * 64, "md5:" + "a" * 64],
)
def test_checkpoint_directory_rejects_noncanonical_dataset_id(
    dataset_id: str,
) -> None:
    with pytest.raises(CheckpointIdentityError):
        engine_module.checkpoint_identity_directory("H1", dataset_id)


def test_legacy_checkpoint_without_identity_is_explicitly_rejected(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    _engine().save_checkpoint(20, str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for field in CHECKPOINT_IDENTITY_FIELDS:
        payload.pop(field)
    torch.save(payload, checkpoint)
    target = _engine()

    with pytest.raises(CheckpointIdentityError, match="旧版产物.*拒绝续训"):
        target.load_checkpoint(str(checkpoint))

    assert target.model.loaded == []
    assert target.opt.loaded == []


def test_latest_run_wins_over_older_higher_step_without_deleting_old_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_root = tmp_path / "H1" / ("a" * 64)
    old_run = identity_root / "run_00000000000000000001"
    new_run = identity_root / "run_00000000000000000002"
    old_run.mkdir(parents=True)
    old_checkpoint = old_run / "ckpt_BTCUSDT_step_9000.pt"
    old_checkpoint.write_bytes(b"old")
    new_run.mkdir()
    new_checkpoint = new_run / "ckpt_BTCUSDT_step_0020.pt"
    new_checkpoint.write_bytes(b"new")
    os.utime(old_run, ns=(2, 2))
    os.utime(new_run, ns=(1, 1))
    monkeypatch.setattr(train_file.time, "time_ns", lambda: 3)

    selected = train_file._latest_identity_checkpoint(identity_root, "BTCUSDT")
    next_run = train_file._new_checkpoint_run_directory(identity_root)

    assert selected == new_checkpoint
    assert next_run == identity_root / "run_00000000000000000003"
    assert old_checkpoint.read_bytes() == b"old"
    assert new_checkpoint.read_bytes() == b"new"


def test_mismatched_best_strategy_is_not_seeded_for_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    mismatched = {**_identity(), "timeframe": "M15", "formula": [9], "best_score": 99.0}
    (strategies / "best_BTCUSDT.json").write_text(
        json.dumps(mismatched), encoding="utf-8"
    )
    engine = _engine()
    engine.best_formula = None
    engine.best_score = -float("inf")

    train_file._seed_best_from_strategy(engine, "BTCUSDT")

    assert engine.best_formula is None
    assert engine.best_score == -float("inf")


def test_matching_best_strategy_can_seed_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    matching = {
        **_identity(),
        "vocab_version": FORMULA_VOCAB.version,
        "formula": [9],
        "best_score": 99.0,
    }
    (strategies / "best_BTCUSDT.json").write_text(
        json.dumps(matching), encoding="utf-8"
    )
    engine = _engine()
    engine.best_formula = None
    engine.best_score = -float("inf")

    train_file._seed_best_from_strategy(engine, "BTCUSDT")

    assert engine.best_formula == [9]
    assert engine.best_score == 99.0


def test_previous_execution_contract_strategy_cannot_seed_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    legacy = {
        **_identity(),
        "vocab_version": _legacy_token_only_vocab_version(),
        "formula": [9],
        "best_score": 99.0,
    }
    (strategies / "best_BTCUSDT.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    engine = _engine()
    engine.best_formula = None
    engine.best_score = -float("inf")

    train_file._seed_best_from_strategy(engine, "BTCUSDT")

    assert engine.best_formula is None
    assert engine.best_score == -float("inf")


def test_previous_execution_contract_score_cannot_block_current_strategy_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    legacy = {
        **_identity(),
        "vocab_version": _legacy_token_only_vocab_version(),
        "formula": [9],
        "best_score": 99.0,
    }
    strategy_path = strategies / "best_BTCUSDT.json"
    strategy_path.write_text(json.dumps(legacy), encoding="utf-8")
    engine = _engine()

    train_file._save_strategy(
        engine,
        "BTCUSDT",
        "H1",
        str(tmp_path / "BTCUSDT_H1.parquet"),
        6240,
        3000,
        "okx",
        f"sha256:{'a' * 64}",
        "a" * 64,
    )

    saved = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert saved["vocab_version"] == FORMULA_VOCAB.version
    assert saved["formula"] == engine.best_formula
    assert saved["best_score"] == engine.best_score
