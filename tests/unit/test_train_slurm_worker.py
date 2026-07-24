from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from config import Config
from scripts import train_slurm_worker as worker


RUN_ID = "run_20260714T120000Z_deadbeef"
COMMIT = "a" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fingerprint(path: Path) -> tuple[int, str]:
    canonical = worker._canonical_source(path)
    return len(canonical), hashlib.sha256(canonical).hexdigest()


@pytest.fixture
def prepared_run(tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    tmp_path = tmp_path_factory.mktemp("worker")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    for relative in worker.REQUIRED_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (git / "refs" / "heads" / "main").write_text(COMMIT + "\n", encoding="ascii")

    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("python\n", encoding="ascii")

    run_dir = tmp_path / "runs" / RUN_ID
    for name in ("input", "logs", "checkpoints", "strategies", "output"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    data = b"PAR1-worker-data"
    data_path = run_dir / "input" / "XAUUSD_H1.parquet"
    data_path.write_bytes(data)
    manifest = {
        "run_id": RUN_ID,
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "data_filename": data_path.name,
        "data_size": len(data),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "data_rows": 3000,
        "data_start": "2020-01-01T00:00:00Z",
        "data_end": "2026-01-01T00:00:00Z",
        "columns": ["time", "open", "high", "low", "close", "tick_volume"],
        "dataset_id": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "local_source": "mt5",
        "periods_per_year": 6240,
        "minimum_bars": worker.GENERIC_MINIMUM_BARS,
        "git_commit": COMMIT,
        "training_parameters": {"train_steps": 10, "from_scratch": True},
        "requested_resources": {
            "partition": "cpu",
            "qos": "normal",
            "cpus_per_task": 4,
            "time_limit": "00:10:00",
            "memory": "8G",
        },
        "source_files": [],
    }
    for relative in worker.REQUIRED_SOURCE_FILES:
        size, digest = _source_fingerprint(tmp_path / relative)
        manifest["source_files"].append(
            {"path": relative, "size": size, "sha256": digest}
        )
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "input" / "run_manifest.sha256").write_text(_sha(manifest_path) + "\n", encoding="ascii")
    return run_dir, python, manifest


def _slurm_env() -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_NAME": f"alphamaster_{RUN_ID}",
        "SLURM_JOB_PARTITION": "cpu",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURMD_NODENAME": "cu19",
        "MT5_PASSWORD": "must-not-leak",
        "AI_API_KEY": "must-not-leak",
    }


def _write_checkpoint(run_dir: Path, manifest: dict, symbol: str) -> Path:
    path = (
        run_dir
        / "checkpoints"
        / manifest["timeframe"]
        / manifest["data_sha256"]
        / "run_00000000000000000001"
        / f"ckpt_{symbol}_step_0010.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    return path


def test_success_manifest_hashes_only_whitelisted_artifacts(prepared_run) -> None:
    run_dir, python, manifest = prepared_run
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        (run_dir / "strategies" / "best_XAUUSD.json").write_text("{}", encoding="utf-8")
        _write_checkpoint(run_dir, manifest, "XAUUSD")
        (run_dir / "training_history_XAUUSD.json").write_text("{}", encoding="utf-8")
        (run_dir / "output" / "not-an-artifact.txt").write_text("ignore", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    code, result = worker.run_worker(
        RUN_ID,
        runner=fake_run,
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )
    assert code == 0, result["error_message"]
    assert result["status"] == "COMPLETED"
    assert result["compute_node"] == "cu19"
    assert result["strategy_files"] == ["strategies/best_XAUUSD.json"]
    assert result["checkpoint_files"] == [
        f"checkpoints/H1/{manifest['data_sha256']}/"
        "run_00000000000000000001/ckpt_XAUUSD_step_0010.pt"
    ]
    assert result["symbol"] == "XAUUSD"
    assert result["timeframe"] == "H1"
    assert result["dataset_id"].startswith("sha256:")
    assert result["local_source"] == "mt5"
    assert result["periods_per_year"] == 6240
    assert result["data_start"] == manifest["data_start"]
    assert result["data_end"] == manifest["data_end"]
    assert result["columns"] == manifest["columns"]
    assert "output/not-an-artifact.txt" not in result["artifact_sha256"]
    argv, kwargs = calls[0]
    assert argv[0] == str(python.absolute())
    assert argv[argv.index("--train-steps") + 1] == "10"
    assert argv[argv.index("--periods-per-year") + 1] == "6240"
    assert argv[argv.index("--data-source") + 1] == "mt5"
    assert argv[argv.index("--minimum-bars") + 1] == str(Config.MIN_BARS)
    assert "--from-scratch" in argv
    assert "--train-steps" in argv
    assert kwargs["shell"] is False
    assert "MT5_PASSWORD" not in kwargs["env"]
    assert "AI_API_KEY" not in kwargs["env"]
    saved = json.loads((run_dir / "output" / "result_manifest.json").read_text(encoding="utf-8"))
    assert saved["artifact_sha256"] == result["artifact_sha256"]


def test_legacy_okx_source_is_forwarded_to_formal_training_loader(
    prepared_run,
) -> None:
    run_dir, python, manifest = prepared_run
    manifest["local_source"] = "okx_legacy_attested"
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "input" / "run_manifest.sha256").write_text(
        _sha(manifest_path) + "\n",
        encoding="ascii",
    )
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(list(args))
        (run_dir / "strategies" / "best_XAUUSD.json").write_text(
            "{}",
            encoding="utf-8",
        )
        _write_checkpoint(run_dir, manifest, "XAUUSD")
        (run_dir / "training_history_XAUUSD.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0)

    code, result = worker.run_worker(
        RUN_ID,
        runner=fake_run,
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )

    assert code == 0, result["error_message"]
    argv = calls[0]
    assert argv[argv.index("--data-source") + 1] == "okx_legacy_attested"


def test_data_hash_mismatch_fails_without_starting_training(prepared_run) -> None:
    run_dir, python, _manifest = prepared_run
    original = run_dir / "input" / "XAUUSD_H1.parquet"
    original.write_bytes(b"X" * original.stat().st_size)
    called = False

    def fake_run(args, **kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(args, 0)

    code, result = worker.run_worker(
        RUN_ID,
        runner=fake_run,
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )
    assert code == 1
    assert result["status"] == "FAILED"
    assert "SHA-256" in result["error_message"]
    assert called is False


def test_checkpoint_from_wrong_dataset_directory_is_not_collected(prepared_run) -> None:
    run_dir, python, _manifest = prepared_run

    def fake_run(args, **_kwargs):
        (run_dir / "strategies" / "best_XAUUSD.json").write_text("{}", encoding="utf-8")
        wrong = (
            run_dir
            / "checkpoints"
            / "H1"
            / ("0" * 64)
            / "run_00000000000000000001"
            / "ckpt_XAUUSD_step_0010.pt"
        )
        wrong.parent.mkdir(parents=True)
        wrong.write_bytes(b"wrong identity")
        (run_dir / "training_history_XAUUSD.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    code, result = worker.run_worker(
        RUN_ID,
        runner=fake_run,
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )
    assert code == 1
    assert result["status"] == "FAILED"
    assert "checkpoint" in result["error_message"]


def test_source_hash_mismatch_fails_closed(prepared_run) -> None:
    run_dir, python, _manifest = prepared_run
    root = run_dir.parents[1]
    source = root / "model_core" / "config.py"
    canonical = worker._canonical_source(source)
    source.write_bytes(b"X" + canonical[1:])
    code, result = worker.run_worker(
        RUN_ID,
        runner=lambda *args, **kwargs: pytest.fail("training must not start"),
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )
    assert code == 1
    assert "源码哈希不匹配" in result["error_message"]


def test_source_fingerprint_normalizes_windows_line_endings(prepared_run) -> None:
    run_dir, _python, manifest = prepared_run
    root = run_dir.parents[1]
    relative = "model_core/config.py"
    source = root / relative
    source.write_bytes(b"line1\r\nline2\r\n")
    canonical = worker._canonical_source(source)
    row = next(item for item in manifest["source_files"] if item["path"] == relative)
    row["size"] = len(canonical)
    row["sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "input" / "run_manifest.sha256").write_text(
        _sha(manifest_path) + "\n", encoding="ascii"
    )
    _manifest, _data, _manifest_hash = worker._verify_inputs(run_dir)


def test_ashare_period_contract_is_verified_before_training(prepared_run) -> None:
    run_dir, _python, manifest = prepared_run
    old_data = run_dir / "input" / manifest["data_filename"]
    new_data = run_dir / "input" / "000001_H1.parquet"
    old_data.rename(new_data)
    manifest.update(
        {
            "symbol": "000001",
            "timeframe": "H1",
            "data_filename": new_data.name,
            "data_rows": 1936,
            "local_source": "ashare_local",
            "periods_per_year": 6240,
            "minimum_bars": 1936,
        }
    )
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "input" / "run_manifest.sha256").write_text(
        _sha(manifest_path) + "\n", encoding="ascii"
    )

    with pytest.raises(worker.WorkerError, match="periods_per_year"):
        worker._verify_inputs(run_dir)


def test_ashare_worker_passes_verified_contract_to_train_file(prepared_run) -> None:
    run_dir, python, manifest = prepared_run
    old_data = run_dir / "input" / manifest["data_filename"]
    new_data = run_dir / "input" / "000001_H1.parquet"
    old_data.rename(new_data)
    manifest.update(
        {
            "symbol": "000001",
            "timeframe": "H1",
            "data_filename": new_data.name,
            "data_rows": 1936,
            "local_source": "ashare_local",
            "periods_per_year": 968,
            "minimum_bars": 1936,
        }
    )
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "input" / "run_manifest.sha256").write_text(
        _sha(manifest_path) + "\n", encoding="ascii"
    )
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(list(args))
        (run_dir / "strategies" / "best_000001.json").write_text("{}", encoding="utf-8")
        _write_checkpoint(run_dir, manifest, "000001")
        (run_dir / "training_history_000001.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    code, result = worker.run_worker(
        RUN_ID,
        runner=fake_run,
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )
    assert code == 0, result["error_message"]
    assert result["local_source"] == "ashare_local"
    argv = calls[0]
    assert argv[argv.index("--data-source") + 1] == "ashare_local"
    assert argv[argv.index("--periods-per-year") + 1] == "968"
    assert argv[argv.index("--minimum-bars") + 1] == "1936"


def test_nonzero_training_exit_is_recorded(prepared_run) -> None:
    run_dir, python, _manifest = prepared_run

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 17)

    code, result = worker.run_worker(
        RUN_ID,
        runner=fake_run,
        environ=_slurm_env(),
        cwd=run_dir,
        python_executable=python,
    )
    assert code == 17
    assert result["status"] == "FAILED"
    assert result["exit_code"] == 17
    assert "退出码为 17" in result["error_message"]


def test_wrong_working_directory_still_writes_failure_manifest(prepared_run) -> None:
    run_dir, python, _manifest = prepared_run
    code, result = worker.run_worker(
        RUN_ID,
        runner=lambda *args, **kwargs: pytest.fail("training must not start"),
        environ=_slurm_env(),
        cwd=run_dir.parent,
        python_executable=python,
    )
    assert code == 1
    assert "run 目录" in result["error_message"]
    assert (run_dir / "output" / "result_manifest.json").is_file()


def test_sbatch_script_has_fixed_clean_runtime() -> None:
    assert worker.GENERIC_MINIMUM_BARS == Config.MIN_BARS
    script = Path(__file__).parents[2] / "scripts" / "train_alphamaster.sbatch"
    text = script.read_text(encoding="utf-8")
    assert "umask 077" in text
    assert "/hwdata/home/jinqc/Quant/AlphaMaster/.venv/bin/python" not in text  # 由固定 ROOT 拼出
    assert 'readonly RUN_DIR="$(pwd -P)"' in text
    assert "BASH_SOURCE" not in text
    assert 'readonly PYTHON="${ROOT}/.venv/bin/python"' in text
    assert "exec /usr/bin/env -i" in text
    assert "OMP_NUM_THREADS" in text
    assert "--wrap" not in text
    assert "--nodelist" not in text


def test_clean_training_env_authorizes_only_the_slurm_worker(tmp_path: Path) -> None:
    env = worker._clean_training_env(
        tmp_path,
        12,
        {
            "SLURM_JOB_ID": "12345",
            "SLURMD_NODENAME": "cu16",
        },
    )

    assert env["ALPHAMASTER_TRAINING_RUNTIME"] == "slurm_worker_v1"
    assert env["SLURM_JOB_ID"] == "12345"
    assert env["SLURMD_NODENAME"] == "cu16"
