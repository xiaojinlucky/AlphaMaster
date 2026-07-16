from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import slurm_control as sc


RUN_ID = "run_20260714T120000Z_deadbeef"
FILENAME = "XAUUSD_H1.parquet"


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "SLURM_BIN", Path("/fixed/slurm/bin"))
    monkeypatch.setattr(sc.getpass, "getuser", lambda: "jinqc")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "train_alphamaster.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
    return tmp_path


def _manifest(data: bytes) -> dict:
    return {
        "run_id": RUN_ID,
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "data_filename": FILENAME,
        "data_size": len(data),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "data_rows": 3000,
        "data_start": "2020-01-01T00:00:00Z",
        "data_end": "2026-01-01T00:00:00Z",
        "columns": ["time", "open", "high", "low", "close", "tick_volume"],
        "local_source": "mt5",
        "git_commit": "a" * 40,
        "training_parameters": {"train_steps": 10, "from_scratch": True},
        "requested_resources": {
            "partition": "cpu",
            "qos": "normal",
            "cpus_per_task": 4,
            "time_limit": "00:10:00",
            "memory": "8G",
        },
        "source_files": [
            {"path": path, "size": 1, "sha256": "b" * 64}
            for path in sc.REQUIRED_SOURCE_FILES
        ],
    }


def _finalized_run(root: Path) -> tuple[Path, dict]:
    prepared = sc.prepare_run(RUN_ID, FILENAME)
    data = b"PAR1-test-data"
    manifest = _manifest(data)
    Path(prepared["data_partial"]).write_bytes(data)
    Path(prepared["manifest_partial"]).write_text(json.dumps(manifest), encoding="utf-8")
    sc.finalize_upload(
        RUN_ID,
        FILENAME,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return root / "runs" / RUN_ID, manifest


@pytest.mark.parametrize(
    ("value", "validator"),
    [
        ("../escape", sc.validate_run_id),
        ("12;scancel", sc.validate_job_id),
        ("../../XAUUSD_H1.parquet", sc.validate_filename),
    ],
)
def test_identifier_validation_rejects_injection(value, validator) -> None:
    with pytest.raises(sc.ControlError):
        validator(value)


def test_resource_validation_is_closed() -> None:
    valid = {
        "partition": "cpu",
        "qos": "normal",
        "cpus_per_task": 1,
        "time_limit": "00:01:00",
        "memory": "512M",
    }
    assert sc.validate_resources(valid) == valid
    for key, value in (
        ("partition", "fat"),
        ("qos", "user_test"),
        ("cpus_per_task", 65),
        ("time_limit", "infinite"),
        ("memory", "1T"),
    ):
        broken = dict(valid)
        broken[key] = value
        with pytest.raises(sc.ControlError):
            sc.validate_resources(broken)


def test_manifest_rejects_unknown_local_source() -> None:
    data = b"PAR1-test-data"
    manifest = _manifest(data)
    manifest["local_source"] = "unknown"
    with pytest.raises(sc.ControlError, match="local_source"):
        sc._validate_manifest(manifest, RUN_ID, FILENAME)


@pytest.mark.parametrize(
    "source",
    [
        "mt5",
        "mt5_legacy_attested",
        "okx",
        "okx_legacy_attested",
        "ashare_local",
    ],
)
def test_manifest_accepts_known_local_sources(source: str) -> None:
    data = b"PAR1-test-data"
    manifest = _manifest(data)
    manifest["local_source"] = source
    assert sc._validate_manifest(manifest, RUN_ID, FILENAME)["local_source"] == source


def test_prepare_and_finalize_are_idempotent(isolated_root: Path) -> None:
    run_dir, manifest = _finalized_run(isolated_root)
    again = sc.prepare_run(RUN_ID, FILENAME)
    assert Path(again["run_dir"]) == run_dir
    result = sc.finalize_upload(
        RUN_ID,
        FILENAME,
        size_bytes=manifest["data_size"],
        sha256=manifest["data_sha256"],
    )
    assert result["finalized"] is True
    assert (run_dir / "input" / FILENAME).read_bytes() == b"PAR1-test-data"
    assert len((run_dir / "input" / "run_manifest.sha256").read_text().strip()) == 64


def test_finalize_cleans_identical_partials_after_response_loss(isolated_root: Path) -> None:
    run_dir, manifest = _finalized_run(isolated_root)
    input_dir = run_dir / "input"
    data_final = input_dir / FILENAME
    manifest_final = input_dir / "run_manifest.json"
    data_partial = input_dir / f"{FILENAME}.partial"
    manifest_partial = input_dir / "run_manifest.json.partial"
    data_partial.write_bytes(data_final.read_bytes())
    manifest_partial.write_bytes(manifest_final.read_bytes())

    result = sc.finalize_upload(
        RUN_ID,
        FILENAME,
        size_bytes=manifest["data_size"],
        sha256=manifest["data_sha256"],
    )

    assert result["finalized"] is True
    assert data_final.read_bytes() == b"PAR1-test-data"
    assert not data_partial.exists()
    assert not manifest_partial.exists()


def test_finalize_rejects_hash_mismatch(isolated_root: Path) -> None:
    prepared = sc.prepare_run(RUN_ID, FILENAME)
    data = b"PAR1-test-data"
    Path(prepared["data_partial"]).write_bytes(data)
    Path(prepared["manifest_partial"]).write_text(json.dumps(_manifest(data)), encoding="utf-8")
    with pytest.raises(sc.ControlError, match="SHA-256"):
        sc.finalize_upload(RUN_ID, FILENAME, size_bytes=len(data), sha256="0" * 64)


def test_submit_uses_fixed_argv_and_does_not_duplicate(isolated_root: Path) -> None:
    run_dir, _manifest_value = _finalized_run(isolated_root)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="12345\n", stderr="")

    first = sc.submit_run(RUN_ID, runner=fake_run)
    second = sc.submit_run(RUN_ID, runner=fake_run)
    assert first == {"run_id": RUN_ID, "job_id": "12345", "submitted": True, "idempotent": False}
    assert second["idempotent"] is True
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert Path(argv[0]).name == "sbatch"
    assert f"--chdir={run_dir}" in argv
    assert "--partition=cpu" in argv
    assert "--qos=normal" in argv
    assert not any(arg.startswith("--wrap") or arg.startswith("--nodelist") for arg in argv)
    assert kwargs["shell"] is False


def test_submit_rejects_non_numeric_job_id(isolated_root: Path) -> None:
    _finalized_run(isolated_root)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="12345;cluster\n", stderr="")

    with pytest.raises(sc.ControlError, match="纯数字"):
        sc.submit_run(RUN_ID, runner=fake_run)


def test_status_falls_back_to_sacct(isolated_root: Path) -> None:
    _finalized_run(isolated_root)

    def submit(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="12345\n", stderr="")

    sc.submit_run(RUN_ID, runner=submit)

    def query(args, **kwargs):
        name = Path(args[0]).name
        if name == "squeue":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        assert name == "sacct"
        if f"12345.batch" in args:
            line = "12345.batch|COMPLETED|0:0|00:01:00|00:03:40|120M\n"
            return subprocess.CompletedProcess(args, 0, stdout=line, stderr="")
        line = (
            f"12345|jinqc|alphamaster_{RUN_ID}|COMPLETED|0:0|"
            "2026-07-14T10:00:00|2026-07-14T10:01:00|00:01:00|4|cu19|120M\n"
        )
        return subprocess.CompletedProcess(args, 0, stdout=line, stderr="")

    result = sc.status_run(RUN_ID, "12345", runner=query)
    assert result["status"] == "COMPLETED"
    assert result["source"] == "sacct"
    assert result["allocated_cpus"] == 4
    assert result["total_cpu"] == "00:03:40"
    assert result["max_rss"] == "120M"


def test_cancel_checks_live_job_identity_before_scancel(isolated_root: Path) -> None:
    run_dir, _manifest_value = _finalized_run(isolated_root)

    def submit(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="12345\n", stderr="")

    sc.submit_run(RUN_ID, runner=submit)
    calls: list[str] = []

    def query(args, **kwargs):
        calls.append(Path(args[0]).name)
        line = f"12345|jinqc|wrong_name|RUNNING|{run_dir}|cu19\n"
        return subprocess.CompletedProcess(args, 0, stdout=line, stderr="")

    with pytest.raises(sc.ControlError, match="作业名"):
        sc.cancel_run(RUN_ID, "12345", runner=query)
    assert "scancel" not in calls


def test_tail_is_bounded_and_separate(isolated_root: Path) -> None:
    run_dir = Path(sc.prepare_run(RUN_ID, FILENAME)["run_dir"])
    (run_dir / "logs" / "slurm.out").write_text("a\nb\nc\n", encoding="utf-8")
    (run_dir / "logs" / "slurm.err").write_text("error\n", encoding="utf-8")
    result = sc.tail_run(RUN_ID, lines=2)
    assert result["stdout"] == ["b", "c"]
    assert result["stderr"] == ["error"]


def test_positional_cli_contract_matches_windows_client() -> None:
    parser = sc._parser()
    args = parser.parse_args(["finalize-upload", RUN_ID, FILENAME, "a" * 64, "123"])
    assert args.run_id == RUN_ID
    assert args.filename == FILENAME
    assert args.sha256 == "a" * 64
    assert args.size_bytes == 123
    tail = parser.parse_args(["tail", RUN_ID, "12345", "150"])
    assert tail.job_id == "12345" and tail.lines == 150


def test_result_rechecks_manifest_artifacts(isolated_root: Path) -> None:
    run_dir, _manifest_value = _finalized_run(isolated_root)

    def submit(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="12345\n", stderr="")

    sc.submit_run(RUN_ID, runner=submit)
    artifact = run_dir / "strategies" / "best_XAUUSD.json"
    artifact.write_text("{}", encoding="utf-8")
    history = run_dir / "training_history_XAUUSD.json"
    history.write_text("{}", encoding="utf-8")
    payload = {
        "run_id": RUN_ID,
        "slurm_job_id": "12345",
        "status": "COMPLETED",
        "exit_code": 0,
        "artifacts": [
            {
                "path": "strategies/best_XAUUSD.json",
                "size": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            {
                "path": "training_history_XAUUSD.json",
                "size": history.stat().st_size,
                "sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
            },
        ],
    }
    (run_dir / "output" / "result_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert sc.result_run(RUN_ID, "12345") == payload
