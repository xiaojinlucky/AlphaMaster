"""Slurm 训练状态持久化与数据身份合同测试。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
import torch

import web.progress as progress_module
import web.slurm_training_manager as manager_module
import web.training_package as package_module
from model_core.alphagpt import AlphaGPT
from scripts import slurm_control as control_module


@pytest.fixture
def tmp_path() -> Path:
    """身份隔离路径较深，使用短目录避免 Windows 旧路径上限干扰测试。"""
    with tempfile.TemporaryDirectory(prefix="alphamaster_slurm_") as directory:
        yield Path(directory)


_REAL_CHECKPOINT_STATES: tuple[dict, dict] | None = None


def test_training_source_fingerprint_isolated_from_frontend() -> None:
    paths = {row["path"] for row in manager_module._source_files()}
    assert "train_file.py" in paths
    assert "data_pipeline/a_share_data.py" in paths
    assert "model_core/engine.py" in paths
    assert "model_core/backtest.py" in paths
    assert "strategy_manager/signal.py" in paths
    assert "utils/training_runtime.py" in paths
    assert "scripts/train_slurm_worker.py" in paths
    assert "scripts/train_alphamaster.sbatch" in paths
    assert "web/app.py" not in paths
    assert "web/static/app.js" not in paths
    assert not any(path.startswith("portfolio_manager/") for path in paths)
    assert not any(path.startswith("tests/") for path in paths)


def _real_checkpoint_states() -> tuple[dict, dict]:
    global _REAL_CHECKPOINT_STATES
    if _REAL_CHECKPOINT_STATES is None:
        model = AlphaGPT()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _REAL_CHECKPOINT_STATES = (model.state_dict(), optimizer.state_dict())
    return _REAL_CHECKPOINT_STATES


class FakeClient:
    def __init__(
        self,
        states: list[str] | None = None,
        *,
        transport_failures: dict[str, int] | None = None,
        cancel_error: Exception | None = None,
        checkpoint_version: str | None = None,
        checkpoint_problem: str | None = None,
    ) -> None:
        self.states = list(states or [])
        self.transport_failures = dict(transport_failures or {})
        self.cancel_error = cancel_error
        self.checkpoint_version = checkpoint_version
        self.checkpoint_problem = checkpoint_problem
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
        checkpoint = (
            local_artifact_root
            / "checkpoints"
            / run_manifest["timeframe"]
            / run_manifest["data_sha256"]
            / "run_00000000000000000001"
            / "ckpt_XAUUSD_step_0010.pt"
        )
        strategy = local_artifact_root / "strategies" / "best_XAUUSD.json"
        history = local_artifact_root / "training_history_XAUUSD.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        strategy.parent.mkdir(parents=True, exist_ok=True)
        model_state, optimizer_state = _real_checkpoint_states()
        checkpoint_payload = {
            "vocab_version": (
                self.checkpoint_version or manager_module.VOCAB_VERSION
            ),
            "scoring_contract_version": (
                manager_module.SCORING_CONTRACT_VERSION
            ),
            "symbol": run_manifest["symbol"],
            "step": run_manifest["training_parameters"]["train_steps"],
            "best_score": 0.5,
            "best_formula": [1, 2, 3],
            "training_history": {},
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "timeframe": run_manifest["timeframe"],
            "dataset_id": run_manifest["dataset_id"],
            "data_sha256": run_manifest["data_sha256"],
            "local_source": run_manifest["local_source"],
            "periods_per_year": run_manifest["periods_per_year"],
            "minimum_bars": run_manifest["minimum_bars"],
        }
        if self.checkpoint_problem == "missing_version":
            checkpoint_payload.pop("vocab_version")
        elif self.checkpoint_problem == "missing_scoring_contract":
            checkpoint_payload.pop("scoring_contract_version")
        elif self.checkpoint_problem == "wrong_scoring_contract":
            checkpoint_payload["scoring_contract_version"] = "old-contract"
        elif self.checkpoint_problem == "identity_mismatch":
            checkpoint_payload["dataset_id"] = "sha256:" + "0" * 64
        if self.checkpoint_problem == "corrupt":
            checkpoint.write_bytes(b"not-a-checkpoint")
        else:
            with checkpoint.open("wb") as handle:
                torch.save(checkpoint_payload, handle)
        strategy.write_text(
            json.dumps(
                {
                    "vocab_version": manager_module.VOCAB_VERSION,
                    "scoring_contract_version": (
                        manager_module.SCORING_CONTRACT_VERSION
                    ),
                    "symbol": run_manifest["symbol"],
                    "timeframe": run_manifest["timeframe"],
                    "data_file": f"/remote/input/{run_manifest['data_filename']}",
                    "formula": [1, 2, 3],
                    "best_score": 0.5,
                    "train_steps": run_manifest["training_parameters"]["train_steps"],
                    "periods_per_year": run_manifest["periods_per_year"],
                    "minimum_bars": run_manifest["minimum_bars"],
                    "dataset_id": run_manifest["dataset_id"],
                    "data_sha256": run_manifest["data_sha256"],
                    "local_source": run_manifest["local_source"],
                    "data_rows": run_manifest["data_rows"],
                    "data_start": run_manifest["data_start"],
                    "data_end": run_manifest["data_end"],
                    "columns": run_manifest["columns"],
                }
            ),
            encoding="utf-8",
        )
        history.write_text(
            json.dumps(
                {
                    "scoring_contract_version": (
                        manager_module.SCORING_CONTRACT_VERSION
                    )
                }
            ),
            encoding="utf-8",
        )
        extra_checkpoint = None
        if self.checkpoint_problem == "extra_checkpoint":
            extra_checkpoint = local_artifact_root / "checkpoints" / "unvalidated_extra.pt"
            with extra_checkpoint.open("wb") as handle:
                torch.save({"vocab_version": "vprevious0000"}, handle)
        artifacts = []
        artifact_paths = [checkpoint, strategy, history]
        if extra_checkpoint is not None:
            artifact_paths.append(extra_checkpoint)
        for path in artifact_paths:
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
            "data_start",
            "data_end",
            "columns",
            "dataset_id",
            "local_source",
            "periods_per_year",
            "minimum_bars",
            "scoring_contract_version",
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
    formats = {
        "MetaTrader5": "alphamaster_mt5_dataset_v1",
        "OKX": "alphamaster_okx_dataset_v1",
        "AShareLocal": "alphamaster_ashare_local_dataset_v1",
        "AKShare": "alphamaster_ashare_akshare_sina_hfq_dataset_v1",
    }
    periods_per_year = (
        242
        if source == "AKShare"
        else (968 if source == "AShareLocal" else 6240)
    )
    minimum_bars = (
        484
        if source == "AKShare"
        else (
            1936
            if source == "AShareLocal"
            else manager_module.Config.MIN_BARS
        )
    )
    source_fields = (
        {
            "market": "CN_A_SHARE",
            "bar_timestamp_semantics": "bar_close",
            "source_timezone": "Asia/Shanghai",
            "source_time_encoding": "floor(china_local_wall_clock_unix_seconds/1000)",
            "session_close_times": ["10:30", "11:30", "14:00", "15:00"],
            "source_filename": f"{symbol}_60min.parquet",
            "source_sha256": "f" * 64,
            "periods_per_year": periods_per_year,
            "minimum_bars": minimum_bars,
        }
        if source == "AShareLocal"
        else (

                {
                    "source_id": "ashare_akshare_sina_hfq",
                    "market": "CN_A_SHARE",
                    "bar_timestamp_semantics": "bar_close",
                    "source_timezone": "Asia/Shanghai",
                    "session_close_times": ["15:00"],
                    "periods_per_year": periods_per_year,
                    "minimum_bars": minimum_bars,
                    "provider": "AKShare",
                    "provider_version": "1.18.64",
                    "provider_interface": "stock_zh_a_daily",
                    "adjustment": "hfq",
                    "adjustment_history_semantics": (
                        "cumulative_historical_factor_not_latest_price_normalized"
                    ),
                    "bar_completion": "completed_trading_days_only",
                    "source_response_sha256": "e" * 64,
                    "request": {
                        "canonical_symbol": symbol,
                        "symbol": (
                            f"sh{symbol}"
                            if symbol.startswith("6")
                            else f"sz{symbol}"
                        ),
                        "start_date": "20200101",
                        "end_date": "20260101",
                        "adjust": "hfq",
                    },
                }
                if source == "AKShare"
                else (
                    {
                        "source_family": "OKX",
                        "provenance_level": "downloader_verified",
                        "bar_completion": "confirmed_only",
                    }
                    if source == "OKX"
                    else {}
                )

        )
    )
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "format": formats.get(source, "alphamaster_unknown_dataset_v1"),
                "data_filename": path.name,
                "data_sha256": digest,
                "data_rows": 3000,
                "data_start": "2020-01-01T00:00:00Z",
                "data_end": "2026-01-01T00:00:00Z",
                "data_timezone": "UTC",
                "time_unit": "unix_seconds",
                "columns": ["time", "open", "high", "low", "close", "tick_volume"],
                "source": source,
                **source_fields,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_progress_reads_long_nested_checkpoint_via_file_handle(tmp_path: Path) -> None:
    long_root = tmp_path / ("nested_" + "x" * 100)
    checkpoint = (
        long_root
        / "checkpoints"
        / "H1"
        / ("a" * 64)
        / "run_00000000000000000001"
        / "ckpt_XAUUSD_step_0010.pt"
    )
    try:
        filesystem_path = progress_module._filesystem_path(checkpoint)
        filesystem_path.parent.mkdir(parents=True)
        with filesystem_path.open("wb") as handle:
            torch.save(
                {
                    "vocab_version": manager_module.VOCAB_VERSION,
                    "scoring_contract_version": (
                        manager_module.SCORING_CONTRACT_VERSION
                    ),
                    "step": 10,
                    "training_history": {},
                },
                handle,
            )
        if os.name == "nt":
            assert len(str(checkpoint.resolve())) > 260
            assert str(filesystem_path).startswith("\\\\?\\")
        progress_module.invalidate_checkpoint_cache()
        assert progress_module._load_checkpoint_meta(checkpoint)["step"] == 10
    finally:
        shutil.rmtree(progress_module._filesystem_path(long_root), ignore_errors=False)


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
        lambda: [
            {"path": path, "sha256": "b" * 64, "size": 1}
            for path in control_module.REQUIRED_SOURCE_FILES
        ],
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
    data_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()

    job = manager.start(str(data_file), "XAUUSD", "H1", from_scratch=True)
    assert job["remote_state"] == "PENDING"
    assert job["slurm_job_id"] == "4321"
    assert manager.status()["job"]["remote_state"] == "RUNNING"
    ready = manager.status()
    assert ready["active"] is False
    assert ready["job"]["remote_state"] == "READY", ready["job"].get("error")
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
    assert (
        isolated_project
        / "checkpoints"
        / "H1"
        / data_sha256
        / "run_00000000000000000001"
        / "ckpt_XAUUSD_step_0010.pt"
    ).is_file()
    assert (isolated_project / "training_history_XAUUSD.json").is_file()
    assert json.loads((runs / "current.json").read_text(encoding="utf-8"))["run_id"] == job["run_id"]

    restored = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    assert restored.status()["job"]["remote_state"] == "READY"


def test_missing_current_pointer_is_the_only_idle_recovery_state(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    runs = tmp_path / "runs"
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )

    status = manager.status()

    assert status["active"] is False
    assert status["status_unknown"] is False
    assert status["job"] is None
    started = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert started["remote_state"] == "PENDING"
    assert client.submitted == [started["run_id"]]


@pytest.mark.parametrize(
    "damage",
    (
        "broken-current-json",
        "missing-state",
        "broken-state-json",
        "broken-state-structure",
    ),
)
def test_existing_but_unrecoverable_current_pointer_fails_closed(
    isolated_project: Path,
    tmp_path: Path,
    damage: str,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    pointer = runs / "current.json"
    run_id = "run_20260724T010000Z_1234abcd"
    state_path = runs / run_id / "state.json"
    if damage == "broken-current-json":
        pointer.write_text("{", encoding="utf-8")
    else:
        pointer.write_text(
            json.dumps({"run_id": run_id}),
            encoding="utf-8",
        )
        if damage in {"broken-state-json", "broken-state-structure"}:
            state_path.parent.mkdir()
            state_path.write_text(
                (
                    "{"
                    if damage == "broken-state-json"
                    else json.dumps(
                        {
                            "run_id": run_id,
                            "remote_state": [],
                        }
                    )
                ),
                encoding="utf-8",
            )
    pointer_before = pointer.read_bytes()
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )

    status = manager.status()

    assert status["active"] is True
    assert status["status_unknown"] is True
    assert status["remote_status_stale"] is True
    assert status["job"]["remote_state"] == manager_module.RECOVERY_UNKNOWN
    assert status["job"]["state"] == "failed"
    assert "恢复失败" in status["job"]["recovery_error"]
    with pytest.raises(RuntimeError, match="恢复失败"):
        manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert client.prepared == []
    assert client.uploaded == []
    assert client.submitted == []
    assert pointer.read_bytes() == pointer_before


def test_unreadable_current_pointer_fails_closed(
    isolated_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = (tmp_path / "runs").resolve()
    runs.mkdir()
    pointer = runs / "current.json"
    pointer.write_text(
        json.dumps({"run_id": "run_20260724T010000Z_1234abcd"}),
        encoding="utf-8",
    )
    real_read_text = Path.read_text

    def fail_current_read(path: Path, *args, **kwargs):
        if path.resolve() == pointer:
            raise OSError("simulated read failure")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_current_read)
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )

    status = manager.status()

    assert status["active"] is True
    assert status["status_unknown"] is True
    assert status["job"]["remote_state"] == manager_module.RECOVERY_UNKNOWN
    assert "current.json" in status["job"]["recovery_error"]
    with pytest.raises(RuntimeError, match="恢复失败"):
        manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert client.prepared == []
    assert client.uploaded == []
    assert client.submitted == []


def test_planned_run_id_is_idempotent_and_never_resubmitted(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["PENDING"])
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    data_file = _dataset(tmp_path)
    planned = "run_20260724T010000Z_1234abcd"

    first = manager.start(
        str(data_file),
        "XAUUSD",
        "H1",
        from_scratch=True,
        planned_run_id=planned,
    )
    repeated = manager.start(
        str(data_file),
        "XAUUSD",
        "H1",
        from_scratch=True,
        planned_run_id=planned,
    )

    assert first["run_id"] == planned
    assert repeated["run_id"] == planned
    assert client.submitted == [planned]


def test_planned_run_retries_only_before_slurm_job_id_exists(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    class RejectFirstUploadClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.rejected = False

        def upload_inputs(self, *, run_id: str, **kwargs):
            self.uploaded.append(run_id)
            if not self.rejected:
                self.rejected = True
                raise manager_module.SlurmClientError(
                    "manifest local_source 不受支持"
                )
            return {"ok": True}

    client = RejectFirstUploadClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    data_file = _dataset(tmp_path)
    planned = "run_20260724T010000Z_1234abce"

    with pytest.raises(RuntimeError, match="提交失败"):
        manager.start(
            str(data_file),
            "XAUUSD",
            "H1",
            from_scratch=True,
            planned_run_id=planned,
        )
    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert failed["slurm_job_id"] is None

    retried = manager.start(
        str(data_file),
        "XAUUSD",
        "H1",
        from_scratch=True,
        planned_run_id=planned,
    )

    assert retried["run_id"] == planned
    assert retried["remote_state"] == "PENDING"
    assert retried["retry_count"] == 1
    assert retried["retry_history"][0]["error"] == (
        "manifest local_source 不受支持"
    )
    assert client.submitted == [planned]


def test_planned_run_id_rejects_identity_change(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["PENDING"])
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    data_file = _dataset(tmp_path)
    planned = "run_20260724T010000Z_1234abcd"
    manager.start(
        str(data_file),
        "XAUUSD",
        "H1",
        planned_run_id=planned,
    )

    with pytest.raises(RuntimeError, match="已绑定不同"):
        manager.start(
            str(data_file),
            "XAUUSD",
            "H1",
            from_scratch=True,
            planned_run_id=planned,
        )


def test_planned_run_id_freezes_explicit_training_budget(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["PENDING"])
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    data_file = _dataset(tmp_path)
    planned = "run_20260724T010000Z_8765abcd"

    job = manager.start(
        str(data_file),
        "XAUUSD",
        "H1",
        from_scratch=True,
        planned_run_id=planned,
        train_steps=200,
        cpus_per_task=12,
        memory="32G",
        time_limit="00:30:00",
    )

    assert job["training_parameters"] == {
        "train_steps": 200,
        "from_scratch": True,
    }
    assert job["requested_resources"] == {
        "partition": "cpu",
        "qos": "normal",
        "cpus_per_task": 12,
        "memory": "32G",
        "time_limit": "00:30:00",
    }
    with pytest.raises(RuntimeError, match="已绑定不同"):
        manager.start(
            str(data_file),
            "XAUUSD",
            "H1",
            from_scratch=True,
            planned_run_id=planned,
            train_steps=201,
            cpus_per_task=12,
            memory="32G",
            time_limit="00:30:00",
        )


def test_start_rejects_expected_source_drift_before_remote_prepare(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    runs = tmp_path / "runs"
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )
    actual = manager.current_source_sha256()
    drifted = "0" * 64 if actual != "0" * 64 else "1" * 64

    with pytest.raises(
        RuntimeError,
        match="expected_source_sha256 不一致",
    ):
        manager.start(
            str(_dataset(tmp_path)),
            "XAUUSD",
            "H1",
            expected_source_sha256=drifted,
        )

    assert client.prepared == []
    assert client.uploaded == []
    assert client.submitted == []
    assert not (runs / "current.json").exists()


def test_start_freezes_explicit_runtime_git_commit(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    runs = tmp_path / "runs"
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )
    expected_commit = "c" * 40

    started = manager.start(
        str(_dataset(tmp_path)),
        "XAUUSD",
        "H1",
        expected_git_commit=expected_commit,
    )

    manifest = json.loads(
        (
            runs
            / started["run_id"]
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert started["git_commit"] == expected_commit
    assert manifest["git_commit"] == expected_commit


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
    status = manager.status()["job"]
    assert status["remote_state"] == "READY", status.get("error")

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
    assert all(path.parent.name == "run_00000000000000000001" for path in bundle["checkpoint_paths"])
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
            f"checkpoints/H1/{bundle['data_sha256']}/"
            "run_00000000000000000001/ckpt_XAUUSD_step_0010.pt",
            "strategies/best_XAUUSD.json",
            "training_history_XAUUSD.json",
        }


def test_explicit_wrong_strategy_run_id_never_reaches_ready_or_publish(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"])
    original = client.download_result
    wrong_run_id = "run_20260724T020000Z_deadbeef"

    def wrong_strategy_run_id(**kwargs):
        result = original(**kwargs)
        strategy_path = (
            kwargs["local_artifact_root"]
            / "strategies"
            / "best_XAUUSD.json"
        )
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        strategy["run_id"] = wrong_run_id
        strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
        digest = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
        for row in result["artifacts"]:
            if row["path"] == "strategies/best_XAUUSD.json":
                row["size"] = strategy_path.stat().st_size
                row["sha256"] = digest
        result["artifact_sha256"]["strategies/best_XAUUSD.json"] = digest
        return result

    client.download_result = wrong_strategy_run_id  # type: ignore[method-assign]
    pointer_path = (
        isolated_project / "published_training" / "current_XAUUSD.json"
    )
    pointer_path.parent.mkdir(parents=True)
    pointer_before = b'{"bundle_id":"old"}'
    pointer_path.write_bytes(pointer_before)
    strategy_path = isolated_project / "strategies" / "best_XAUUSD.json"
    strategy_path.parent.mkdir(parents=True)
    strategy_before = b'{"best_score":9.9}'
    strategy_path.write_bytes(strategy_before)
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )

    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    failed = manager.status()["job"]

    assert failed["remote_state"] == "FAILED"
    assert "run_id" in failed["error"]
    assert pointer_path.read_bytes() == pointer_before
    assert strategy_path.read_bytes() == strategy_before


@pytest.mark.parametrize(
    ("checkpoint_version", "checkpoint_problem"),
    [
        ("vprevious0000", None),
        (None, "missing_version"),
        (None, "missing_scoring_contract"),
        (None, "wrong_scoring_contract"),
        (None, "identity_mismatch"),
        (None, "corrupt"),
        (None, "extra_checkpoint"),
    ],
    ids=(
        "previous-version",
        "missing-version",
        "missing-scoring-contract",
        "wrong-scoring-contract",
        "identity-mismatch",
        "corrupt",
        "extra-checkpoint",
    ),
)
def test_invalid_checkpoint_bundle_never_reaches_ready(
    isolated_project: Path,
    tmp_path: Path,
    checkpoint_version: str | None,
    checkpoint_problem: str | None,
) -> None:
    client = FakeClient(
        ["COMPLETED"],
        checkpoint_version=checkpoint_version,
        checkpoint_problem=checkpoint_problem,
    )
    runs = tmp_path / "local-runs"
    pointer_path = (
        isolated_project / "published_training" / "current_XAUUSD.json"
    )
    pointer_path.parent.mkdir(parents=True)
    pointer_before = b'{"bundle_id":"old"}'
    pointer_path.write_bytes(pointer_before)
    strategy_path = isolated_project / "strategies" / "best_XAUUSD.json"
    strategy_path.parent.mkdir(parents=True)
    strategy_before = b'{"best_score":9.9}'
    strategy_path.write_bytes(strategy_before)
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )

    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    status = manager.status()["job"]

    assert status["remote_state"] == "FAILED"
    assert "checkpoint" in status["error"]
    assert pointer_path.read_bytes() == pointer_before
    assert strategy_path.read_bytes() == strategy_before
    assert not (
        isolated_project / "checkpoints" / "unvalidated_extra.pt"
    ).exists()


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
    status = manager.status()["job"]
    assert status["remote_state"] == "READY", status.get("error")

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
    ("field", "value"),
    [
        ("timeframe", "M15"),
        ("data_sha256", "0" * 64),
        ("dataset_id", "sha256:" + "0" * 64),
        ("local_source", "unknown"),
        ("periods_per_year", 968),
        ("minimum_bars", 1),
        ("scoring_contract_version", "old-contract"),
    ],
)
def test_published_bundle_identity_tampering_fails_closed(
    isolated_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    monkeypatch.setenv("TRAINING_BACKEND", "slurm")
    monkeypatch.setattr(progress_module, "PROJECT_ROOT", isolated_project)
    client = FakeClient(["COMPLETED"])
    runs = tmp_path / "local-runs"
    manager = manager_module.SlurmTrainingManager(client=client, local_runs_root=runs)
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    status = manager.status()["job"]
    assert status["remote_state"] == "READY", status.get("error")

    pointer_path = isolated_project / "published_training" / "current_XAUUSD.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer[field] = value
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    assert progress_module.get_published_bundle("XAUUSD") is None


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
    assert recovered["remote_state"] == "PENDING", recovered.get("error")
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


@pytest.mark.parametrize(
    ("interrupted_action", "expected_state"),
    [
        ("prepare", "PREPARING"),
        ("upload", "UPLOADING"),
        ("submit", "SUBMITTING"),
    ],
)
def test_dispatch_recovery_rechecks_frozen_source_before_every_remote_action(
    isolated_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_action: str,
    expected_state: str,
) -> None:
    client = FakeClient(
        transport_failures={interrupted_action: 1},
    )
    runs = tmp_path / "runs"
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )
    expected_source = manager.current_source_sha256()
    observed_source = {"value": expected_source}
    monkeypatch.setattr(
        manager_module,
        "current_training_source_sha256",
        lambda: observed_source["value"],
    )

    started = manager.start(
        str(_dataset(tmp_path)),
        "XAUUSD",
        "H1",
        expected_source_sha256=expected_source,
    )
    assert started["remote_state"] == expected_state
    assert started["expected_source_sha256"] == expected_source
    calls_before_recovery = (
        len(client.prepared),
        len(client.uploaded),
        len(client.submitted),
    )
    observed_source["value"] = (
        "0" * 64 if expected_source != "0" * 64 else "1" * 64
    )

    restored = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=runs,
    )
    recovered = restored.status()["job"]

    assert recovered["run_id"] == started["run_id"]
    assert recovered["remote_state"] == "FAILED"
    assert "源码 SHA-256 已漂移" in recovered["error"]
    assert (
        len(client.prepared),
        len(client.uploaded),
        len(client.submitted),
    ) == calls_before_recovery


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_sha256", "d" * 64),
        ("data_start", "2020-01-02T00:00:00Z"),
        ("columns", ["time", "open"]),
    ],
)
def test_result_identity_mismatch_cannot_become_ready(
    isolated_project: Path,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    client = FakeClient(["COMPLETED"])
    original = client.download_result

    def wrong_identity(**kwargs):
        result = original(**kwargs)
        result[field] = value
        return result

    client.download_result = wrong_identity  # type: ignore[method-assign]
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert field in failed["error"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_start", "2020-01-02T00:00:00Z"),
        ("data_end", "2025-12-31T00:00:00Z"),
        ("columns", ["time", "open"]),
    ],
)
def test_strategy_range_or_columns_tampering_cannot_become_ready(
    isolated_project: Path,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    client = FakeClient(["COMPLETED"])
    original = client.download_result

    def wrong_strategy(**kwargs):
        result = original(**kwargs)
        strategy_path = (
            kwargs["local_artifact_root"] / "strategies" / "best_XAUUSD.json"
        )
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        strategy[field] = value
        strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
        digest = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
        for row in result["artifacts"]:
            if row["path"] == "strategies/best_XAUUSD.json":
                row["size"] = strategy_path.stat().st_size
                row["sha256"] = digest
        result["artifact_sha256"]["strategies/best_XAUUSD.json"] = digest
        return result

    client.download_result = wrong_strategy  # type: ignore[method-assign]
    manager = manager_module.SlurmTrainingManager(
        client=client, local_runs_root=tmp_path / "runs"
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    failed = manager.status()["job"]
    assert failed["remote_state"] == "FAILED"
    assert "回传策略身份" in failed["error"]


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


def test_legacy_okx_archive_enters_run_with_distinct_source_identity(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    path = _dataset(tmp_path, symbol="BTCUSDT", timeframe="H1", source="OKX")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("source_family", "provenance_level", "bar_completion"):
        manifest.pop(field)
    manifest.update(
        {
            "provenance_status": "legacy_archive_attestation",
            "source_instrument": "BTC-USDT-SWAP",
            "source_endpoint": "/api/v5/market/history-candles",
            "source_bar": "1H",
            "volume_semantics": "OKX contract volume mapped to tick_volume",
            "provenance": "user_provided_archive:OKX_K线数据.zip",
            "closed_bars_only": True,
            "derived_from": {
                "archive_member": "OKX_K线数据/BTCUSDT_H1.parquet",
                "data_sha256": "f" * 64,
            },
            "transform": {
                "dropped_trailing_unclosed_bars": 1,
                "cutoff_reference": "source_file_mtime",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(),
        local_runs_root=tmp_path / "runs",
    )

    started = manager.start(str(path), "BTCUSDT", "H1")
    run_manifest = json.loads(
        (
            tmp_path
            / "runs"
            / started["run_id"]
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert run_manifest["local_source"] == "okx_legacy_attested"
    assert control_module._validate_manifest(
        run_manifest,
        started["run_id"],
        path.name,
    )["local_source"] == "okx_legacy_attested"


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
        ("periods_per_year", 999, "periods_per_year"),
        ("bar_completion", "unconfirmed_allowed", "已完成 K 线"),
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


def test_ashare_periods_and_minimum_bars_enter_run_identity(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    path = _dataset(
        tmp_path,
        symbol="000001",
        timeframe="H1",
        source="AShareLocal",
    )
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(), local_runs_root=tmp_path / "runs"
    )
    started = manager.start(str(path), "000001", "H1")
    run_manifest = json.loads(
        (tmp_path / "runs" / started["run_id"] / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["local_source"] == "ashare_local"
    assert run_manifest["periods_per_year"] == 968
    assert run_manifest["minimum_bars"] == 1936


def test_akshare_hfq_enters_ashare_run_identity(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    path = _dataset(
        tmp_path,
        symbol="600519",
        timeframe="D1",
        source="AKShare",
    )
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(),
        local_runs_root=tmp_path / "runs",
    )
    started = manager.start(str(path), "600519", "D1")
    run_manifest = json.loads(
        (tmp_path / "runs" / started["run_id"] / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["local_source"] == "ashare_akshare_sina_hfq"
    assert run_manifest["periods_per_year"] == 242
    assert run_manifest["minimum_bars"] == 484


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


def test_snapshot_and_cached_log_never_poll_remote(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["RUNNING"])
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.tail_log(80) == ["[10/20] smoke"]

    def fail_status(*_args, **_kwargs):
        raise AssertionError("snapshot 不得查询远程 Slurm")

    def fail_tail(*_args, **_kwargs):
        raise AssertionError("cached_log_tail 不得查询远程日志")

    client.status = fail_status  # type: ignore[method-assign]
    client.tail = fail_tail  # type: ignore[method-assign]

    snapshot = manager.snapshot()
    assert snapshot["active"] is True
    assert snapshot["job"]["remote_state"] == "PENDING"
    assert manager.cached_log_tail(80) == ["[10/20] smoke"]
    assert manager.parse_step_from_log(refresh_remote=False) == 10


def test_cached_log_and_progress_are_bound_to_requested_run(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["RUNNING"])
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    current = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    assert manager.tail_log(80) == ["[10/20] smoke"]

    other_run_id = "run_20260724T020000Z_deadbeef"
    other_log = tmp_path / "runs" / other_run_id / "logs" / "tail.log"
    other_log.parent.mkdir(parents=True, exist_ok=True)
    other_log.write_text("[77/200] other run\n", encoding="utf-8")

    assert manager.cached_log_tail(80, run_id=current["run_id"]) == [
        "[10/20] smoke"
    ]
    assert manager.cached_log_tail(80, run_id=other_run_id) == [
        "[77/200] other run"
    ]
    assert (
        manager.parse_step_from_log(
            refresh_remote=False,
            run_id=other_run_id,
        )
        == 77
    )
    assert manager.tail_log(80, expected_run_id=other_run_id) == []


def test_terminal_log_is_marked_final_only_after_successful_remote_pull(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(["COMPLETED"])
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    run_id = manager.snapshot()["job"]["run_id"]

    def fail_tail(*_args, **_kwargs):
        raise manager_module.SlurmClientError("remote log unavailable")

    client.tail = fail_tail  # type: ignore[method-assign]
    assert manager.tail_log(80, expected_run_id=run_id, final=True) == []
    failed_snapshot = manager.snapshot()["job"]
    assert "final_log_refreshed_at" not in failed_snapshot
    assert failed_snapshot["last_log_poll_error"] == "remote log unavailable"

    client.tail = lambda *_args, **_kwargs: ["final line"]  # type: ignore[method-assign]
    assert manager.tail_log(80, expected_run_id=run_id, final=True) == [
        "final line"
    ]
    completed_snapshot = manager.snapshot()["job"]
    assert completed_snapshot["final_log_refreshed_at"]
    assert "last_log_poll_error" not in completed_snapshot


def test_unknown_slurm_state_remains_active_and_visible_as_stale(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    def unknown_status(*_args, **_kwargs):
        raise manager_module.SlurmClientError(
            "未知 Slurm 状态: POWER_DOWN_NODE"
        )

    client.status = unknown_status  # type: ignore[method-assign]
    status = manager.status()

    assert status["active"] is True
    assert status["job"]["remote_state"] == "PENDING"
    assert status["remote_status_stale"] is True
    assert "POWER_DOWN_NODE" in status["job"]["last_poll_error"]


def test_temporary_slurm_client_error_never_proves_remote_failure(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    client.status = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        manager_module.SlurmClientError("temporary sacct failure")
    )

    status = manager.status()

    assert status["active"] is True
    assert status["job"]["remote_state"] == "PENDING"
    assert status["remote_status_stale"] is True
    assert status["job"]["last_poll_error"] == "temporary sacct failure"


def test_expired_squeue_job_recovers_from_bound_result_manifest(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    client.status = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        manager_module.SlurmClientError(
            '{"ok": false, "error": '
            '"slurm_load_jobs error: Invalid job id specified"}'
        )
    )
    status = manager.status()

    assert status["active"] is False
    assert status["job"]["remote_state"] == "READY"
    assert client.download_calls == 1


def test_expired_squeue_job_retries_until_result_manifest_is_visible(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    submitted = tuple(client.submitted)
    client.status = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        manager_module.SlurmClientError(
            '{"ok": false, "error": '
            '"slurm_load_jobs error: Invalid job id specified"}'
        )
    )
    real_download = client.download_result
    attempts = 0

    def delayed_download(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            client.download_calls += 1
            raise manager_module.SlurmClientError("result manifest不存在")
        return real_download(**kwargs)

    client.download_result = delayed_download  # type: ignore[method-assign]

    waiting = manager.status()
    ready = manager.status()

    assert waiting["active"] is True
    assert waiting["job"]["remote_state"] == "DOWNLOADING"
    assert waiting["job"]["last_poll_error"] == "result manifest不存在"
    assert ready["active"] is False
    assert ready["job"]["remote_state"] == "READY"
    assert client.download_calls == 2
    assert tuple(client.submitted) == submitted


def test_raw_nonterminal_slurm_state_overrides_stale_failed_normalization(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    client.status = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "state": "SPECIAL_EXIT",
        "status": "FAILED",
        "reason": "legacy normalizer",
    }
    status = manager.status()

    assert status["active"] is True
    assert status["job"]["remote_state"] == "PENDING"
    assert status["job"]["slurm_state"] == "SPECIAL_EXIT"
    assert not status["job"].get("error")


def test_stop_rejects_changed_run_or_job_identity(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    job = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    with pytest.raises(RuntimeError, match="run_id 已变化"):
        manager.stop(
            expected_run_id="run_20260724T020000Z_deadbeef",
            expected_job_id=job["slurm_job_id"],
        )
    with pytest.raises(RuntimeError, match="job_id 已变化"):
        manager.stop(
            expected_run_id=job["run_id"],
            expected_job_id="999999",
        )

    assert client.cancelled == []
    assert manager.stop(
        expected_run_id=job["run_id"],
        expected_job_id=job["slurm_job_id"],
    )
    assert client.cancelled == [(job["run_id"], job["slurm_job_id"])]


def test_remote_log_wait_does_not_hold_training_state_lock(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = manager_module.SlurmTrainingManager(
        client=client,
        local_runs_root=tmp_path / "runs",
    )
    job = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")
    tail_started = threading.Event()
    release_tail = threading.Event()

    def blocking_tail(*_args, **_kwargs):
        tail_started.set()
        assert release_tail.wait(timeout=5)
        return ["done"]

    client.tail = blocking_tail  # type: ignore[method-assign]
    tail_thread = threading.Thread(
        target=manager.tail_log,
        kwargs={
            "expected_run_id": job["run_id"],
            "expected_job_id": job["slurm_job_id"],
        },
    )
    tail_thread.start()
    assert tail_started.wait(timeout=2)

    stop_results: list[bool] = []
    stop_thread = threading.Thread(
        target=lambda: stop_results.append(
            manager.stop(
                expected_run_id=job["run_id"],
                expected_job_id=job["slurm_job_id"],
            )
        )
    )
    stop_thread.start()
    stop_thread.join(timeout=1)
    assert not stop_thread.is_alive()
    assert stop_results == [True]

    release_tail.set()
    tail_thread.join(timeout=2)
    assert not tail_thread.is_alive()


def test_batch_source_hash_matches_frozen_run_manifest(
    isolated_project: Path,
    tmp_path: Path,
) -> None:
    manager = manager_module.SlurmTrainingManager(
        client=FakeClient(),
        local_runs_root=tmp_path / "runs",
    )
    job = manager.start(str(_dataset(tmp_path)), "XAUUSD", "H1")

    assert manager.current_source_sha256() == manager.run_source_sha256(
        job["run_id"]
    )


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
