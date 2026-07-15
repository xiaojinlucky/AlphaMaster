"""Slurm 训练状态持久化与数据身份合同测试。"""
from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
import torch

import web.slurm_training_manager as manager_module
import web.progress as progress_module
import web.training_package as package_module


class FakeClient:
    def __init__(
        self,
        states: list[str] | None = None,
        *,
        transport_failures: dict[str, int] | None = None,
        cancel_error: Exception | None = None,
    ) -> None:
        self.states = list(states or [])
        self.transport_failures = dict(transport_failures or {})
        self.cancel_error = cancel_error
        self.prepared: list[str] = []
        self.uploaded: list[str] = []
        self.submitted: list[str] = []
        self.download_calls = 0
        self.cancelled: list[tuple[str, str]] = []

    def _maybe_interrupt(self, action: str) -> None:
        remaining = self.transport_failures.get(action, 0)
        if remaining:
            self.transport_failures[action] = remaining - 1
            raise manager_module.SlurmTransportError(f"{action} response lost")

    def prepare(self, run_id: str):
        self.prepared.append(run_id)
        self._maybe_interrupt("prepare")
        return {"ok": True}

    def upload_inputs(self, *, run_id: str, **_kwargs):
        self.uploaded.append(run_id)
        self._maybe_interrupt("upload")
        return {"ok": True}

    def submit(self, _run_id: str) -> str:
        self.submitted.append(_run_id)
        self._maybe_interrupt("submit")
        return "4321"

    def status(self, _run_id: str, _job_id: str):
        state = self.states.pop(0)
        return {
            "state": state,
            "node": "cpu-node-01" if state == "RUNNING" else "",
            "exit_code": "0:0" if state == "COMPLETED" else "",
            "started_at": "2026-07-14T14:00:00" if state == "COMPLETED" else None,
            "finished_at": "2026-07-14T14:00:03" if state == "COMPLETED" else None,
            "elapsed": "00:00:03" if state == "COMPLETED" else None,
            "allocated_cpus": 4 if state == "COMPLETED" else None,
            "total_cpu": "00:00:10" if state == "COMPLETED" else None,
            "max_rss": "128000K" if state == "COMPLETED" else None,
        }

    def download_result(
        self,
        *,
        run_id: str,
        job_id: str,
        expected_commit: str,
        local_artifact_root: Path,
        **_kwargs,
    ):
        self.download_calls += 1
        self._maybe_interrupt("download")
        run_manifest_path = local_artifact_root.parent / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        checkpoint = local_artifact_root / "checkpoints" / "ckpt_XAUUSD_step_10.pt"
        strategy = local_artifact_root / "strategies" / "best_XAUUSD.json"
        history = local_artifact_root / "training_history_XAUUSD.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        strategy.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vocab_version": manager_module.VOCAB_VERSION,
                "symbol": run_manifest["symbol"],
                "step": run_manifest["training_parameters"]["train_steps"],
                "best_score": 0.5,
                "best_formula": [1, 2, 3],
                "training_history": {},
            },
            checkpoint,
        )
        strategy.write_text(
            json.dumps(
                {
                    "vocab_version": manager_module.VOCAB_VERSION,
                    "symbol": run_manifest["symbol"],
                    "timeframe": run_manifest["timeframe"],
                    "data_file": f"/remote/input/{run_manifest['data_filename']}",
                    "formula": [1, 2, 3],
                    "best_score": 0.5,
                    "train_steps": run_manifest["training_parameters"]["train_steps"],
                }
            ),
            encoding="utf-8",
        )
        history.write_text("{}", encoding="utf-8")
        artifacts = []
        for path in (checkpoint, strategy, history):
            relative = path.relative_to(local_artifact_root).as_posix()
            artifacts.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        result = {
            "run_id": run_id,
            "slurm_job_id": job_id,
            "git_commit": expected_commit,
            "status": "COMPLETED",
            "exit_code": 0,
            "run_manifest_sha256": hashlib.sha256(run_manifest_path.read_bytes()).hexdigest(),
            "artifacts": artifacts,
            "checkpoint_files": [artifacts[0]["path"]],
            "strategy_files": [artifacts[1]["path"]],
            "artifact_sha256": {row["path"]: row["sha256"] for row in artifacts},
        }
        for field in (
            "symbol",
            "timeframe",
            "data_filename",
            "data_sha256",
            "data_size",
            "data_rows",
            "dataset_id",
            "local_source",
            "source_files",
            "training_parameters",
            "requested_resources",
        ):
            result[field] = run_manifest[field]
        result_path = local_artifact_root / "output" / "result_manifest.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return result

    def cancel(self, run_id: str, job_id: str):
        self.cancelled.append((run_id, job_id))
        if self.cancel_error is not None:
            raise self.cancel_error
        self._maybe_interrupt("cancel")
        return {"ok": True}

    def tail(self, _run_id: str, _job_id: str, _lines: int):
        return ["[10/20] smoke"]


def _dataset(
    tmp_path: Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    source: str = "MetaTrader5",
) -> Path:
    path = tmp_path / f"{symbol}_{timeframe}.parquet"
    path.write_bytes(b"PAR1-test")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "format": (
                    "alphamaster_mt5_dataset_v1"
                    if source == "MetaTrader5"
                    else "alphamaster_okx_dataset_v1"
                ),
                "data_filename": path.name,
                "data_sha256": digest,
                "data_rows": 3000,
                "data_start": "2020-01-01T00:00:00Z",
                "data_end": "2026-01-01T00:00:00Z",
                "data_timezone": "UTC",
                "time_unit": "unix_seconds",
                "columns": ["time", "open", "high", "low", "close", "tick_volume"],
                "source": source,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_TRAIN_STEPS", "10")
    monkeypatch.setattr(manager_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(manager_module, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        manager_module,
        "_source_files",
        lambda: [{"path": "train_file.py", "sha256": "b" * 64, "size": 1}],
    )
    monkeypatch.setattr(
        manager_module,
        "_inspect_parquet_contract",
        lambda _path: {
            "data_rows": 3000,
            "data_start": "2020-01-01T00:00:00Z",
            "data_end": "2026-01-01T00:00:00Z",
            "columns": ["time", "open", "high", "low", "close", "tick_volume"],
        },
    )
    return project


def test_submit_poll_download_and_restart_recovery(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["RUNNING", "COMPLETED"])
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    data_file = _dataset(tmp_path)

    job = manager.start(str(data_file), "XAUUSD", "H1", from_scratch=True)
    assert job["remote_state"] == "PENDING"
    assert job["slurm_job_id"] == "4321"
    assert manager.status()["job"]["remote_state"] == "RUNNING"
    ready = manager.status()
    assert ready["active"] is False
    assert ready["job"]["remote_state"] == "READY"
    assert ready["job"]["elapsed"] == "00:00:03"
    assert ready["job"]["allocated_cpus"] == 4
    assert ready["job"]["total_cpu"] == "00:00:10"
    assert ready["job"]["max_rss"] == "128000K"
    assert ready["job"]["training_parameters"] == {
        "train_steps": 10,
        "from_scratch": True,
    }
    assert ready["job"]["requested_resources"]["cpus_per_task"] == 1
    assert ready["job"]["artifact_count"] == 3
    assert (isolated_project / "checkpoints" / "ckpt_XAUUSD_step_10.pt").is_file()
    assert (isolated_project / "training_history_XAUUSD.json").is_file()
    assert json.loads((runs / "current.json").read_text(encoding="utf-8"))["run_id"] == job["run_id"]

    restored = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    assert restored.status()["job"]["remote_state"] == "READY"


def test_ready_publishes_one_run_bundle_and_replaces_old_higher_score_strategy(
    isolated_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_strategy = isolated_project / "strategies" / "best_XAUUSD.json"
    old_strategy.parent.mkdir(parents=True)
    old_strategy.write_text(
        json.dumps({"symbol": "XAUUSD", "formula": [4, 5], "best_score": 9.0}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_BACKEND", "slurm")
    monkeypatch.setattr(progress_module, "PROJECT_ROOT", isolated_project)

    client = FakeClient(["COMPLETED"])
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    started = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.status()["job"]["remote_state"] == "READY"

    canonical = json.loads(old_strategy.read_text(encoding="utf-8"))
    assert canonical["best_score"] == 0.5
    assert canonical["run_id"] == started["run_id"]
    bundle = progress_module.get_published_bundle("XAUUSD")
    assert bundle is not None
    assert bundle["run_id"] == started["run_id"]
    assert bundle["local_source"] == "mt5"
    assert bundle["artifact_root_path"] == (
        runs / started["run_id"] / "artifacts"
    ).resolve()
    assert bundle["strategy_path"].parent.name == "strategies"
    assert bundle["history_path"].parent == bundle["artifact_root_path"]
    assert all(path.parent.name == "checkpoints" for path in bundle["checkpoint_paths"])
    progress = progress_module.get_symbol_progress("XAUUSD")
    assert progress.current_step == 10
    assert progress.train_steps == 10
    assert progress.status == "completed"

    content, _ = package_module.build_training_export_zip("XAUUSD")
    with zipfile.ZipFile(BytesIO(content)) as archive:
        package_manifest = json.loads(archive.read("manifest.json"))
        packaged_strategy = json.loads(archive.read("strategies/best_XAUUSD.json"))
        assert package_manifest["run_id"] == started["run_id"]
        assert package_manifest["local_source"] == "mt5"
        assert packaged_strategy["local_source"] == "mt5"
        assert packaged_strategy["run_id"] == started["run_id"]
        assert set(package_manifest["files"]) == {
            "checkpoints/ckpt_XAUUSD_step_10.pt",
            "strategies/best_XAUUSD.json",
            "training_history_XAUUSD.json",
        }


@pytest.mark.parametrize(
    "artifact_field",
    ["checkpoint_files", "strategy_file", "history_file"],
)
def test_published_bundle_tampering_fails_closed_for_all_slurm_readers(
    isolated_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_field: str,
) -> None:
    monkeypatch.setenv("TRAINING_BACKEND", "slurm")
    monkeypatch.setattr(progress_module, "PROJECT_ROOT", isolated_project)
    client = FakeClient(["COMPLETED"])
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.status()["job"]["remote_state"] == "READY"

    pointer = json.loads(
        (isolated_project / "published_training" / "current_XAUUSD.json").read_text(
            encoding="utf-8"
        )
    )
    relative = pointer[artifact_field]
    if isinstance(relative, list):
        relative = relative[0]
    artifact = Path(pointer["artifact_root"]).joinpath(*relative.split("/"))
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    assert progress_module.get_published_bundle("XAUUSD") is None
    assert progress_module.checkpoint_glob("XAUUSD") == []
    assert progress_module._load_strategy("XAUUSD") is None


@pytest.mark.parametrize(
    ("interrupted_action", "expected_state"),
    [("prepare", "PREPARING"), ("upload", "UPLOADING")],
)
def test_restart_recovers_prepare_and_upload_response_loss(
    isolated_project: Path,
    tmp_path: Path,
    interrupted_action: str,
    expected_state: str,
) -> None:
    client = FakeClient(transport_failures={interrupted_action: 1})
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    started = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert started["remote_state"] == expected_state

    restored = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    recovered = restored.status()["job"]
    assert recovered["run_id"] == started["run_id"]
    assert recovered["remote_state"] == "PENDING"
    assert recovered["slurm_job_id"] == "4321"


def test_submit_response_loss_recovers_same_run_and_job_after_restart(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(transport_failures={"submit": 1})
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    started = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert started["remote_state"] == "SUBMITTING"
    assert started["slurm_job_id"] is None

    restored = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    recovered = restored.status()["job"]
    assert recovered["run_id"] == started["run_id"]
    assert recovered["remote_state"] == "PENDING"
    assert recovered["slurm_job_id"] == "4321"
    assert client.submitted == [started["run_id"], started["run_id"]]


def test_restart_recovers_persisted_submitted_state(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    started = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    state_path = runs / started["run_id"] / "state.json"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    persisted["remote_state"] = "SUBMITTED"
    state_path.write_text(json.dumps(persisted), encoding="utf-8")

    restored = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    recovered = restored.status()["job"]
    assert recovered["run_id"] == started["run_id"]
    assert recovered["slurm_job_id"] == "4321"
    assert recovered["remote_state"] == "PENDING"


def test_download_transport_failure_stays_retryable_across_restart(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"], transport_failures={"download": 1})
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    interrupted = manager.status()["job"]
    assert interrupted["remote_state"] == "DOWNLOADING"
    assert "download response lost" in interrupted["last_poll_error"]

    restored = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    assert restored.status()["job"]["remote_state"] == "READY"
    assert client.download_calls == 2


def test_download_integrity_failure_is_terminal(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"])

    def reject_integrity(**_kwargs):
        raise manager_module.SlurmClientError("结果校验失败")

    client.download_result = reject_integrity  # type: ignore[method-assign]
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert "结果校验失败" in failed["error"]


def test_result_identity_mismatch_cannot_become_ready(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"])
    original = client.download_result

    def wrong_identity(**kwargs):
        result = original(**kwargs)
        result["data_sha256"] = "d" * 64
        return result

    client.download_result = wrong_identity  # type: ignore[method-assign]
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert "data_sha256" in failed["error"]


def test_okx_source_is_preserved_and_tampering_cannot_become_ready(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"])
    original = client.download_result

    def wrong_source(**kwargs):
        result = original(**kwargs)
        result["local_source"] = "mt5"
        return result

    client.download_result = wrong_source  # type: ignore[method-assign]
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    manager.start(
        str(_dataset(tmp_path, symbol="BTCUSDT", source="OKX")),
        "BTCUSDT",
        "H1",
    )

    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert "local_source" in failed["error"]


def test_unknown_data_source_is_rejected(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(), local_runs_root=tmp_path / "runs"
    )
    with pytest.raises(RuntimeError, match="source"):
        manager.start(
            str(_dataset(tmp_path, source="Unverified")),
            "XAUUSD",
            "H1",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "alphamaster_mt5_dataset_v1", "format"),
        ("dataset_id", "sha256:" + "0" * 64, "dataset_id"),
        ("data_rows", 2999, "data_rows"),
        ("data_start", "2020-01-02T00:00:00Z", "data_start"),
        ("columns", ["time", "open"], "columns"),
    ],
)
def test_data_manifest_identity_tampering_is_rejected(
    isolated_project: Path,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = _dataset(tmp_path, source="OKX")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(), local_runs_root=tmp_path / "runs"
    )
    with pytest.raises(RuntimeError, match=message):
        manager.start(str(path), "XAUUSD", "H1")


def test_missing_strategy_cannot_become_ready(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"])
    original = client.download_result

    def missing_strategy(**kwargs):
        result = original(**kwargs)
        result["artifacts"] = [
            row for row in result["artifacts"] if not row["path"].startswith("strategies/")
        ]
        result["strategy_files"] = []
        result["artifact_sha256"] = {
            row["path"]: row["sha256"] for row in result["artifacts"]
        }
        return result

    client.download_result = missing_strategy  # type: ignore[method-assign]
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert "checkpoint、目标策略和训练历史" in failed["error"]


def test_start_rejects_symbol_or_timeframe_mismatch(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(), local_runs_root=tmp_path / "runs"
    )
    with pytest.raises(RuntimeError, match="不一致"):
        manager.start(str(_dataset(tmp_path)), "EURUSD", "H1")


def test_cancel_and_log_progress_are_remote_only(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["CANCELLED"])
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    job = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.parse_step_from_log() == 10
    assert manager.stop() is True
    assert client.cancelled == [(job["run_id"], "4321")]
    assert manager._job["remote_state"] == "CANCELLING"
    assert manager.status()["job"]["remote_state"] == "CANCELLED"


def test_cancel_transport_loss_is_retried_until_remote_terminal_state(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(
        ["RUNNING", "CANCELLED"],
        transport_failures={"cancel": 1},
    )
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    job = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.stop() is True
    assert manager.status()["job"]["remote_state"] == "CANCELLING"
    assert manager.status()["job"]["remote_state"] == "CANCELLED"
    assert client.cancelled == [
        (job["run_id"], "4321"),
        (job["run_id"], "4321"),
    ]


def test_cancel_race_with_natural_completion_downloads_result(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(
        ["COMPLETED"],
        cancel_error=manager_module.SlurmClientError("作业已不在活动队列"),
    )
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    job = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.stop() is True
    current = manager.status()["job"]
    assert current["run_id"] == job["run_id"]
    assert current["remote_state"] == "READY"
    assert client.download_calls == 1
