from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "backup_am_assets.py"
SPEC = importlib.util.spec_from_file_location("backup_am_assets_tested", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


def _run_promote(
    staging: Path,
    root: Path,
    *,
    script: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-", str(staging), str(root)],
        input=(script or backup.REMOTE_PROMOTE_SCRIPT).encode("utf-8"),
        capture_output=True,
        check=False,
    )


def _run_cleanup(staging: Path, root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-", str(staging), str(root)],
        input=backup.REMOTE_CLEANUP_SCRIPT.encode("utf-8"),
        capture_output=True,
        check=False,
    )


def _current_generation(root: Path) -> Path:
    generation_name = (root / "CURRENT").read_text("ascii").strip()
    generation = root / ".generations" / generation_name
    assert generation.is_dir()
    return generation


def test_sqlite_snapshot_is_self_contained_delete_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    database = project / "local_runs" / "queue.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE queue (value TEXT NOT NULL)")
        connection.execute("INSERT INTO queue VALUES ('snapshot-row')")
        connection.commit()

    monkeypatch.setattr(backup, "PROJECT_ROOT", project)
    monkeypatch.setattr(backup, "BACKUP_DIRS", ("local_runs",))
    snapshots = backup._sqlite_snapshots(tmp_path / "snapshots")

    main = snapshots["local_runs/queue.sqlite3"]
    assert snapshots["local_runs/queue.sqlite3-wal"].stat().st_size == 0
    assert snapshots["local_runs/queue.sqlite3-shm"].stat().st_size == 0
    with sqlite3.connect(main) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT value FROM queue").fetchone()[0] == (
            "snapshot-row"
        )


def test_remote_promotion_switches_complete_generation_atomically(
    tmp_path: Path,
) -> None:
    root = tmp_path / "backup"
    incoming = root / ".incoming"
    legacy = root / "local_runs"
    incoming.mkdir(parents=True)
    legacy.mkdir()
    (legacy / "queue.sqlite3").write_bytes(b"legacy-main")
    (legacy / "queue.sqlite3-wal").write_bytes(b"legacy-valid-wal")
    (legacy / "queue.sqlite3-shm").write_bytes(b"legacy-valid-shm")

    staging1 = incoming / "am-backup-aaaaaaaaaaaaaaaaaaaaaaaa"
    staged_db1 = staging1 / "local_runs"
    staged_db1.mkdir(parents=True)
    (staged_db1 / "queue.sqlite3").write_bytes(b"generation-one-main")
    (staged_db1 / "queue.sqlite3-wal").write_bytes(b"")
    (staged_db1 / "queue.sqlite3-shm").write_bytes(b"")

    first = _run_promote(staging1, root)
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    assert first.stdout.strip() == b"3"
    generation1 = _current_generation(root)
    assert (generation1 / "local_runs/queue.sqlite3").read_bytes() == (
        b"generation-one-main"
    )
    assert (generation1 / "local_runs/queue.sqlite3-wal").read_bytes() == b""
    assert (generation1 / "local_runs/queue.sqlite3-shm").read_bytes() == b""
    assert (legacy / "queue.sqlite3").read_bytes() == b"legacy-main"
    assert (legacy / "queue.sqlite3-wal").read_bytes() == b"legacy-valid-wal"

    staging2 = incoming / "am-backup-bbbbbbbbbbbbbbbbbbbbbbbb"
    staged_db2 = staging2 / "local_runs"
    staged_db2.mkdir(parents=True)
    (staged_db2 / "queue.sqlite3").write_bytes(b"generation-two-main")
    (staged_db2 / "queue.sqlite3-wal").write_bytes(b"")
    (staged_db2 / "queue.sqlite3-shm").write_bytes(b"")

    second = _run_promote(staging2, root)
    assert second.returncode == 0, second.stderr.decode("utf-8", "replace")
    generation2 = _current_generation(root)
    assert generation2 != generation1
    assert generation1.is_dir()
    assert (generation1 / "local_runs/queue.sqlite3").read_bytes() == (
        b"generation-one-main"
    )
    assert (generation2 / "local_runs/queue.sqlite3").read_bytes() == (
        b"generation-two-main"
    )


def test_failed_generation_does_not_move_current_pointer(tmp_path: Path) -> None:
    root = tmp_path / "backup"
    incoming = root / ".incoming"
    incoming.mkdir(parents=True)

    good = incoming / "am-backup-cccccccccccccccccccccccc"
    good_db = good / "local_runs"
    good_db.mkdir(parents=True)
    (good_db / "queue.sqlite3").write_bytes(b"good")
    first = _run_promote(good, root)
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    pointer_before = (root / "CURRENT").read_bytes()

    invalid = incoming / "am-backup-dddddddddddddddddddddddd"
    invalid_file = invalid / "outside-contract" / "file.bin"
    invalid_file.parent.mkdir(parents=True)
    invalid_file.write_bytes(b"invalid")
    failed = _run_promote(invalid, root)

    assert failed.returncode != 0
    assert (root / "CURRENT").read_bytes() == pointer_before
    assert (_current_generation(root) / "local_runs/queue.sqlite3").read_bytes() == (
        b"good"
    )


def test_mid_promotion_failure_removes_private_building(
    tmp_path: Path,
) -> None:
    root = tmp_path / "backup"
    incoming = root / ".incoming"
    incoming.mkdir(parents=True)

    good = incoming / "am-backup-eeeeeeeeeeeeeeeeeeeeeeee"
    good_file = good / "local_runs" / "conflict" / "seed.bin"
    good_file.parent.mkdir(parents=True)
    good_file.write_bytes(b"seed")
    first = _run_promote(good, root)
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    pointer_before = (root / "CURRENT").read_bytes()

    invalid = incoming / "am-backup-ffffffffffffffffffffffff"
    invalid_file = invalid / "local_runs" / "conflict"
    invalid_file.parent.mkdir(parents=True)
    invalid_file.write_bytes(b"cannot-replace-directory")
    failed = _run_promote(invalid, root)

    assert failed.returncode != 0
    assert (root / "CURRENT").read_bytes() == pointer_before
    assert not list((root / ".generations").glob("*.building"))


def test_pointer_commit_boundary_never_deletes_current_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "backup"
    incoming = root / ".incoming"
    incoming.mkdir(parents=True)
    staging = incoming / "am-backup-333333333333333333333333"
    staged_file = staging / "local_runs" / "queue.sqlite3"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_bytes(b"committed-before-interrupt")

    needle = 'publish_state["pointer_replaced"] = True'
    assert backup.REMOTE_PROMOTE_SCRIPT.count(needle) == 1
    fault_script = backup.REMOTE_PROMOTE_SCRIPT.replace(
        needle,
        'raise KeyboardInterrupt("commit-boundary-injection")',
    )
    interrupted = _run_promote(staging, root, script=fault_script)

    assert interrupted.returncode != 0
    current = _current_generation(root)
    assert (current / "local_runs/queue.sqlite3").read_bytes() == (
        b"committed-before-interrupt"
    )
    assert not list((root / ".generations").glob("*.building"))


def test_symlinked_control_directories_are_rejected_without_outside_delete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "backup"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    try:
        (root / ".incoming").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"当前平台不能创建目录符号链接: {exc}")

    staging = outside / "am-backup-111111111111111111111111"
    staged_file = staging / "local_runs" / "queue.sqlite3"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_bytes(b"must-survive")

    promoted = _run_promote(staging, root)
    cleaned = _run_cleanup(staging, root)

    assert promoted.returncode != 0
    assert cleaned.returncode != 0
    assert staged_file.read_bytes() == b"must-survive"


def test_symlinked_generations_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "backup"
    incoming = root / ".incoming"
    outside = tmp_path / "outside-generations"
    incoming.mkdir(parents=True)
    outside.mkdir()
    try:
        (root / ".generations").symlink_to(
            outside,
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"当前平台不能创建目录符号链接: {exc}")

    staging = incoming / "am-backup-222222222222222222222222"
    staged_file = staging / "local_runs" / "queue.sqlite3"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_bytes(b"not-published")

    promoted = _run_promote(staging, root)

    assert promoted.returncode != 0
    assert staged_file.read_bytes() == b"not-published"
    assert list(outside.iterdir()) == []
