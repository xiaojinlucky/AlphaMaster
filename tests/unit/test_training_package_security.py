from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pytest
import torch

from model_core.alphagpt import AlphaGPT
from model_core.vocab import FORMULA_VOCAB, VocabVersionMismatchError
import web.training_package as package_module


SYMBOL = "XAUUSD"
STEP = 10
STRATEGY = "strategies/best_XAUUSD.json"
HISTORY = "training_history_XAUUSD.json"
DATA_HASH = "a" * 64
SLURM_RUN_ID = "run_20260715T031117Z_2ad7721d"
CHECKPOINT_RUN_ID = "run_12345678901234567890"
CHECKPOINT = (
    f"checkpoints/H1/{DATA_HASH}/{CHECKPOINT_RUN_ID}/ckpt_XAUUSD_step_0010.pt"
)
FLAT_CHECKPOINT = "checkpoints/ckpt_XAUUSD_step_0010.pt"


@lru_cache(maxsize=1)
def _valid_resume_states() -> tuple[dict[str, Any], dict[str, Any]]:
    model = AlphaGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model.state_dict(), optimizer.state_dict()


def _checkpoint_bytes(*, symbol: str = SYMBOL, step: int = STEP) -> bytes:
    model_state, optimizer_state = _valid_resume_states()
    buffer = io.BytesIO()
    torch.save(
        {
            "vocab_version": FORMULA_VOCAB.version,
            "symbol": symbol,
            "timeframe": "H1",
            "dataset_id": f"sha256:{DATA_HASH}",
            "data_sha256": DATA_HASH,
            "local_source": "mt5",
            "periods_per_year": 6240,
            "minimum_bars": 3000,
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "best_score": 0.5,
            "best_formula": [0],
            "training_history": {},
        },
        buffer,
    )
    return buffer.getvalue()


def _legacy_token_only_vocab_version() -> str:
    joined = "\n".join(FORMULA_VOCAB.token_names)
    return "v" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _strategy_bytes(
    *,
    data_hash: str = DATA_HASH,
    run_id: str | None = None,
    timeframe: str | None = "H1",
) -> bytes:
    payload: dict[str, Any] = {
        "vocab_version": FORMULA_VOCAB.version,
        "symbol": SYMBOL,
        "formula": [0],
        "best_score": 0.5,
        "dataset_id": f"sha256:{data_hash}",
        "data_sha256": data_hash,
        "local_source": "mt5",
        "periods_per_year": 6240,
        "minimum_bars": 3000,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if timeframe is not None:
        payload["timeframe"] = timeframe
    return json.dumps(payload).encode("utf-8")


def _history_bytes() -> bytes:
    return json.dumps({"step": list(range(STEP))}).encode("utf-8")


def _manifest(
    payloads: dict[str, bytes],
    *,
    package_format: str = package_module._PACKAGE_FORMAT,
    bind_data: bool = True,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": package_format,
        "symbol": SYMBOL,
        "step": STEP,
        "checkpoint": CHECKPOINT,
        "files": list(payloads),
        "artifacts": [
            {
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in payloads.items()
        ],
    }
    if bind_data:
        manifest.update(
            {
                "timeframe": "H1",
                "checkpoint_run_id": CHECKPOINT_RUN_ID,
                "dataset_id": f"sha256:{DATA_HASH}",
                "data_sha256": DATA_HASH,
                "local_source": "mt5",
                "periods_per_year": 6240,
                "minimum_bars": 3000,
            }
        )
    return manifest


def _zip_bytes(
    entries: list[tuple[str | zipfile.ZipInfo, bytes]],
    manifest: dict[str, Any] | bytes,
) -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries:
                archive.writestr(name, content)
            manifest_bytes = (
                manifest
                if isinstance(manifest, bytes)
                else json.dumps(manifest).encode("utf-8")
            )
            archive.writestr("manifest.json", manifest_bytes)
    return buffer.getvalue()


def _secure_package(
    *,
    checkpoint: bytes | None = None,
    strategy: bytes | None = None,
    history: bytes | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> bytes:
    payloads = {
        CHECKPOINT: checkpoint if checkpoint is not None else _checkpoint_bytes(),
        STRATEGY: strategy if strategy is not None else _strategy_bytes(),
        HISTORY: history if history is not None else _history_bytes(),
    }
    manifest = _manifest(payloads)
    if mutate_manifest:
        mutate_manifest(manifest)
    return _zip_bytes(list(payloads.items()), manifest)


def _nested_secure_package(*, path_hash: str = DATA_HASH) -> bytes:
    checkpoint_relative = (
        f"checkpoints/H1/{path_hash}/{CHECKPOINT_RUN_ID}/ckpt_XAUUSD_step_0010.pt"
    )
    payloads = {
        checkpoint_relative: _checkpoint_bytes(),
        STRATEGY: _strategy_bytes(run_id=SLURM_RUN_ID, timeframe="H1"),
        HISTORY: _history_bytes(),
    }
    manifest = _manifest(payloads)
    manifest.update(
        {
            "checkpoint": checkpoint_relative,
            "run_id": SLURM_RUN_ID,
            "checkpoint_run_id": CHECKPOINT_RUN_ID,
            "timeframe": "H1",
        }
    )
    return _zip_bytes(list(payloads.items()), manifest)


@pytest.fixture()
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    (project / "scratch").mkdir(parents=True)
    monkeypatch.delenv("TRAINING_BACKEND", raising=False)
    monkeypatch.setattr(package_module, "PROJECT_ROOT", project)
    return project


def _seed_existing(project: Path) -> dict[Path, bytes]:
    artifacts = {
        project / "checkpoints" / "ckpt_XAUUSD_step_0001.pt": b"old-checkpoint",
        project / STRATEGY: b"old-strategy",
        project / HISTORY: b"old-history",
        project / "checkpoints" / "ckpt_BTCUSDT_step_0001.pt": b"other-symbol",
    }
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return artifacts


def _assert_existing_unchanged(snapshot: dict[Path, bytes]) -> None:
    for path, content in snapshot.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize(
    "bad_name",
    [
        "/tmp/escape.pt",
        "C:/escape.pt",
        "../escape.pt",
        "checkpoints/../escape.pt",
        "checkpoints\\escape.pt",
    ],
)
def test_rejects_unsafe_member_paths_without_touching_existing_artifacts(
    isolated_project: Path,
    bad_name: str,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = _checkpoint_bytes()
    strategy = _strategy_bytes()
    history = _history_bytes()
    payloads = {CHECKPOINT: checkpoint, STRATEGY: strategy, HISTORY: history}
    content = _zip_bytes(
        [(CHECKPOINT, checkpoint), (STRATEGY, strategy), (bad_name, history)],
        _manifest(payloads),
    )

    with pytest.raises(ValueError):
        package_module.import_training_package(content, "attack.zip", SYMBOL)

    _assert_existing_unchanged(snapshot)
    assert not list((isolated_project / "scratch").glob("training-import-*"))


@pytest.mark.parametrize(
    ("create_system", "external_attr"),
    [
        (3, (stat.S_IFDIR | 0o755) << 16),
        (0, 0x10),
    ],
)
def test_rejects_directory_member_without_touching_existing_artifacts(
    isolated_project: Path,
    create_system: int,
    external_attr: int,
) -> None:
    snapshot = _seed_existing(isolated_project)
    directory = zipfile.ZipInfo("checkpoints")
    directory.create_system = create_system
    directory.external_attr = external_attr
    payloads = {CHECKPOINT: _checkpoint_bytes(), STRATEGY: _strategy_bytes()}
    content = _zip_bytes(
        [(CHECKPOINT, payloads[CHECKPOINT]), (STRATEGY, payloads[STRATEGY]), (directory, b"")],
        _manifest(payloads),
    )

    with pytest.raises(ValueError, match="目录"):
        package_module.import_training_package(content, "directory.zip", SYMBOL)

    _assert_existing_unchanged(snapshot)


def test_rejects_symlink_member_without_touching_existing_artifacts(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    symlink = zipfile.ZipInfo(STRATEGY)
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    checkpoint = _checkpoint_bytes()
    strategy = _strategy_bytes()
    payloads = {CHECKPOINT: checkpoint, STRATEGY: strategy}
    content = _zip_bytes(
        [(CHECKPOINT, checkpoint), (symlink, strategy)],
        _manifest(payloads),
    )

    with pytest.raises(ValueError, match="链接"):
        package_module.import_training_package(content, "symlink.zip", SYMBOL)

    _assert_existing_unchanged(snapshot)


def test_rejects_duplicate_member_without_touching_existing_artifacts(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = _checkpoint_bytes()
    payloads = {CHECKPOINT: checkpoint}
    content = _zip_bytes(
        [(CHECKPOINT, checkpoint), (CHECKPOINT, checkpoint)],
        _manifest(payloads, bind_data=False),
    )

    with pytest.raises(ValueError, match="重复"):
        package_module.import_training_package(content, "duplicate.zip", SYMBOL)

    _assert_existing_unchanged(snapshot)


def test_rejects_member_outside_strict_allowlist(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    payloads = {
        CHECKPOINT: _checkpoint_bytes(),
        STRATEGY: _strategy_bytes(),
        "strategies/evil.json": b"{}",
    }
    content = _zip_bytes(list(payloads.items()), _manifest(payloads))

    with pytest.raises(ValueError, match="allowlist"):
        package_module.import_training_package(content, "extra.zip", SYMBOL)

    _assert_existing_unchanged(snapshot)


@pytest.mark.parametrize(
    "package_format",
    [package_module._LEGACY_PACKAGE_FORMAT, package_module._PACKAGE_FORMAT],
)
def test_rejects_legacy_or_hashless_package_fail_closed(
    isolated_project: Path,
    package_format: str,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = _checkpoint_bytes()
    manifest = _manifest(
        {CHECKPOINT: checkpoint},
        package_format=package_format,
        bind_data=False,
    )
    manifest.pop("artifacts")
    content = _zip_bytes([(CHECKPOINT, checkpoint)], manifest)

    with pytest.raises(ValueError):
        package_module.import_training_package(content, "hashless.zip", SYMBOL)

    _assert_existing_unchanged(snapshot)


def test_rejects_direct_pt_without_touching_existing_artifacts(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)

    with pytest.raises(ValueError, match="v2 ZIP"):
        package_module.import_training_package(
            _checkpoint_bytes(),
            "ckpt_XAUUSD_step_0010.pt",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_rejects_v2_package_with_flat_checkpoint(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    payloads = {
        FLAT_CHECKPOINT: _checkpoint_bytes(),
        STRATEGY: _strategy_bytes(),
        HISTORY: _history_bytes(),
    }
    manifest = _manifest(payloads)
    manifest["checkpoint"] = FLAT_CHECKPOINT
    manifest.pop("checkpoint_run_id")

    with pytest.raises(ValueError, match="v2 训练包只接受"):
        package_module.import_training_package(
            _zip_bytes(list(payloads.items()), manifest),
            "flat.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_rejects_declared_size_or_hash_mismatch_before_publish(
    isolated_project: Path,
    field: str,
) -> None:
    snapshot = _seed_existing(isolated_project)

    def corrupt(manifest: dict[str, Any]) -> None:
        row = manifest["artifacts"][0]
        row[field] = row[field] + 1 if field == "size" else "0" * 64

    with pytest.raises(ValueError):
        package_module.import_training_package(
            _secure_package(mutate_manifest=corrupt),
            "integrity.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_rejects_checkpoint_identity_mismatch_before_publish(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)

    with pytest.raises(ValueError, match="品种"):
        package_module.import_training_package(
            _secure_package(checkpoint=_checkpoint_bytes(symbol="BTCUSDT")),
            "wrong-checkpoint.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_rejects_checkpoint_from_previous_execution_contract_before_publish(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = torch.load(
        io.BytesIO(_checkpoint_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["vocab_version"] = _legacy_token_only_vocab_version()
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)

    with pytest.raises(VocabVersionMismatchError):
        package_module.import_training_package(
            _secure_package(checkpoint=buffer.getvalue()),
            "old-execution-contract.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


@pytest.mark.parametrize("missing_field", ["model_state_dict", "optimizer_state_dict"])
def test_rejects_checkpoint_missing_resume_state_before_publish(
    isolated_project: Path,
    missing_field: str,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = torch.load(
        io.BytesIO(_checkpoint_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint.pop(missing_field)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)

    with pytest.raises(ValueError, match=missing_field):
        package_module.import_training_package(
            _secure_package(checkpoint=buffer.getvalue()),
            "missing-state.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("elite_pool", "not-a-list", "factor_pool/elite_pool"),
        ("best_score", "not-a-number", "best_score"),
    ],
)
def test_rejects_checkpoint_invalid_runtime_state_before_publish(
    isolated_project: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = torch.load(
        io.BytesIO(_checkpoint_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint[field] = value
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)

    with pytest.raises(ValueError, match=message):
        package_module.import_training_package(
            _secure_package(checkpoint=buffer.getvalue()),
            "bad-runtime-state.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_rejects_checkpoint_model_shape_mismatch_before_publish(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    checkpoint = torch.load(
        io.BytesIO(_checkpoint_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    first_key = next(iter(checkpoint["model_state_dict"]))
    checkpoint["model_state_dict"][first_key] = torch.zeros(1)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)

    with pytest.raises(ValueError, match="形状或 dtype"):
        package_module.import_training_package(
            _secure_package(checkpoint=buffer.getvalue()),
            "bad-model-shape.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


@pytest.mark.skipif(os.name != "nt", reason="Windows 扩展路径合同")
def test_checkpoint_validation_uses_file_handle_for_long_windows_path(
    tmp_path: Path,
) -> None:
    base = tmp_path / "long-checkpoint"
    directory = base
    while len(str(directory)) < 280:
        directory /= "identity-segment"
    checkpoint = directory / "ckpt_XAUUSD_step_0010.pt"
    extended_directory = package_module._checkpoint_filesystem_path(directory)
    extended_checkpoint = package_module._checkpoint_filesystem_path(checkpoint)
    os.makedirs(extended_directory, exist_ok=True)
    try:
        checkpoint_payload = _checkpoint_bytes()
        with open(extended_checkpoint, "wb") as handle:
            handle.write(checkpoint_payload)
        metadata = package_module._validate_checkpoint_file(
            checkpoint,
            expected_symbol=SYMBOL,
            expected_step=STEP,
            require_data_identity=True,
        )
        assert metadata["dataset_id"] == f"sha256:{DATA_HASH}"
        assert package_module._read_file_bytes(checkpoint) == checkpoint_payload
    finally:
        shutil.rmtree(package_module._checkpoint_filesystem_path(base))


def test_rejects_strategy_data_identity_mismatch_before_publish(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)

    with pytest.raises(ValueError, match="dataset_id|data_sha256"):
        package_module.import_training_package(
            _secure_package(strategy=_strategy_bytes(data_hash="b" * 64)),
            "wrong-data.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_rejects_strategy_formula_mismatch_with_checkpoint(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    strategy = json.loads(_strategy_bytes())
    strategy["formula"] = [1]

    with pytest.raises(ValueError, match="checkpoint.best_formula"):
        package_module.import_training_package(
            _secure_package(strategy=json.dumps(strategy).encode("utf-8")),
            "wrong-formula.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_valid_package_replaces_one_symbol_only_after_full_validation(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)
    result = package_module.import_training_package(
        _secure_package(),
        "secure.zip",
        SYMBOL,
    )

    assert result["symbol"] == SYMBOL
    assert result["step"] == STEP
    installed_checkpoint = result["installed"][0]
    assert re.fullmatch(
        rf"checkpoints/H1/{DATA_HASH}/run_[0-9]{{20}}/ckpt_XAUUSD_step_0010\.pt",
        installed_checkpoint,
    )
    assert result["installed"][1:] == [STRATEGY, HISTORY]
    assert result["source_checkpoint_run_id"] == CHECKPOINT_RUN_ID
    assert result["checkpoint_run_id"] in installed_checkpoint
    assert (isolated_project / "checkpoints" / "ckpt_XAUUSD_step_0001.pt").is_file()
    assert (isolated_project / installed_checkpoint).is_file()
    assert json.loads((isolated_project / STRATEGY).read_text(encoding="utf-8"))["symbol"] == SYMBOL
    assert json.loads((isolated_project / HISTORY).read_text(encoding="utf-8"))["step"][-1] == 9
    other = isolated_project / "checkpoints" / "ckpt_BTCUSDT_step_0001.pt"
    assert other.read_bytes() == snapshot[other]
    assert not list((isolated_project / "scratch").glob("training-import-*"))


def test_publish_error_rolls_back_every_existing_artifact(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _seed_existing(isolated_project)
    real_replace = os.replace
    calls = 0

    def fail_once(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(package_module.os, "replace", fail_once)
    with pytest.raises(RuntimeError, match="已回滚"):
        package_module.import_training_package(
            _secure_package(),
            "rollback.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)
    identity_root = isolated_project / "checkpoints" / "H1" / DATA_HASH
    assert not identity_root.exists() or not list(identity_root.rglob("*.pt"))


def test_nested_checkpoint_identity_path_is_accepted_and_other_identities_survive(
    isolated_project: Path,
) -> None:
    _seed_existing(isolated_project)
    same_identity_old = (
        isolated_project
        / "checkpoints"
        / "H1"
        / DATA_HASH
        / CHECKPOINT_RUN_ID
        / "ckpt_XAUUSD_step_0001.pt"
    )
    other_run = (
        isolated_project
        / "checkpoints"
        / "H1"
        / DATA_HASH
        / "run_02000000000000000000"
        / "ckpt_XAUUSD_step_0001.pt"
    )
    other_dataset = (
        isolated_project
        / "checkpoints"
        / "H1"
        / ("b" * 64)
        / CHECKPOINT_RUN_ID
        / "ckpt_XAUUSD_step_0001.pt"
    )
    for path, content in (
        (same_identity_old, b"same"),
        (other_run, b"run"),
        (other_dataset, b"data"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    result = package_module.import_training_package(
        _nested_secure_package(),
        "nested.zip",
        SYMBOL,
    )

    installed_checkpoint = result["installed"][0]
    installed_run = PurePosixPath(installed_checkpoint).parts[3]
    assert installed_run != CHECKPOINT_RUN_ID
    assert installed_run > "run_02000000000000000000"
    assert result["source_checkpoint_run_id"] == CHECKPOINT_RUN_ID
    assert result["checkpoint_run_id"] == installed_run
    assert (isolated_project / installed_checkpoint).is_file()
    assert same_identity_old.read_bytes() == b"same"
    assert other_run.read_bytes() == b"run"
    assert other_dataset.read_bytes() == b"data"


def test_nested_checkpoint_path_must_match_manifest_data_identity(
    isolated_project: Path,
) -> None:
    snapshot = _seed_existing(isolated_project)

    with pytest.raises(ValueError, match="数据身份"):
        package_module.import_training_package(
            _nested_secure_package(path_hash="b" * 64),
            "wrong-nested-identity.zip",
            SYMBOL,
        )

    _assert_existing_unchanged(snapshot)


def test_exporter_rejects_flat_checkpoint_that_training_cannot_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "ckpt_XAUUSD_step_0010.pt"
    checkpoint.write_bytes(_checkpoint_bytes())
    history = source / HISTORY
    history.write_bytes(_history_bytes())
    strategy = json.loads(_strategy_bytes())
    monkeypatch.setattr(package_module, "get_published_bundle", lambda _symbol: None)
    monkeypatch.setattr(package_module, "checkpoint_glob", lambda _symbol: [checkpoint])
    monkeypatch.setattr(package_module, "_load_strategy", lambda _symbol: strategy)
    monkeypatch.setattr(package_module, "_history_path", lambda _symbol: history)

    with pytest.raises(ValueError, match="v2 训练包只接受"):
        package_module.build_training_export_zip(SYMBOL)


def test_exporter_preserves_nested_checkpoint_identity_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    checkpoint = source / CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(_checkpoint_bytes())
    history = source / HISTORY
    history.write_bytes(_history_bytes())
    strategy = json.loads(_strategy_bytes(run_id=SLURM_RUN_ID, timeframe="H1"))
    monkeypatch.setattr(package_module, "PROJECT_ROOT", source)
    monkeypatch.setattr(package_module, "get_published_bundle", lambda _symbol: None)
    monkeypatch.setattr(package_module, "checkpoint_glob", lambda _symbol: [checkpoint])
    monkeypatch.setattr(package_module, "_load_strategy", lambda _symbol: strategy)
    monkeypatch.setattr(package_module, "_history_path", lambda _symbol: history)

    content, _filename = package_module.build_training_export_zip(SYMBOL)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["checkpoint"] == CHECKPOINT
        assert manifest["timeframe"] == "H1"
        assert manifest["data_sha256"] == DATA_HASH
        assert manifest["run_id"] == SLURM_RUN_ID
        assert manifest["checkpoint_run_id"] == CHECKPOINT_RUN_ID

    target = tmp_path / "target"
    (target / "scratch").mkdir(parents=True)
    monkeypatch.setattr(package_module, "PROJECT_ROOT", target)
    imported = package_module.import_training_package(content, "nested.zip", SYMBOL)
    assert imported["installed"][0] != CHECKPOINT
    assert imported["source_checkpoint_run_id"] == CHECKPOINT_RUN_ID
