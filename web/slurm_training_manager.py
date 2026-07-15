"""持久化的 Slurm 训练状态机，接口兼容现有 Web 训练管理器。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from utils.train_logging import strip_ansi
from web.slurm_training_client import (
    SlurmClientError,
    SlurmTrainingClient,
    SlurmTransportError,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE_CONTRACTS = {
    "MetaTrader5": ("mt5", "alphamaster_mt5_dataset_v1"),
    "OKX": ("okx", "alphamaster_okx_dataset_v1"),
}
REQUIRED_DATA_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")
LOCAL_RUNS_ROOT = PROJECT_ROOT / "local_runs"
PUBLISHED_BUNDLE_FORMAT = "alphamaster_published_bundle_v1"
ACTIVE_REMOTE_STATES = {
    "PREPARING",
    "UPLOADING",
    "SUBMITTING",
    "SUBMITTED",
    "PENDING",
    "RUNNING",
    "CANCELLING",
    "COMPLETED",
    "DOWNLOADING",
}
TERMINAL_REMOTE_STATES = {"READY", "FAILED", "CANCELLED"}
SLURM_PENDING = {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED"}
SLURM_RUNNING = {"RUNNING", "COMPLETING", "STAGE_OUT"}
SLURM_FAILED = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
}
ALLOWED_TRANSITIONS = {
    "PREPARING": {"UPLOADING", "CANCELLED"},
    "UPLOADING": {"SUBMITTING", "CANCELLED"},
    "SUBMITTING": {"SUBMITTED"},
    "SUBMITTED": {"PENDING", "CANCELLING"},
    "PENDING": {"PENDING", "RUNNING", "COMPLETED", "CANCELLING", "CANCELLED"},
    "RUNNING": {"PENDING", "RUNNING", "COMPLETED", "CANCELLING", "CANCELLED"},
    "CANCELLING": {"CANCELLING", "COMPLETED", "CANCELLED"},
    "COMPLETED": {"DOWNLOADING"},
    "DOWNLOADING": {"READY"},
    "READY": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temp)
    if sha256_file(temp) != sha256_file(source):
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"本机发布复制校验失败: {destination.name}")
    os.replace(temp, destination)


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    value = proc.stdout.strip()
    if proc.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError("无法确认本机源码提交")
    return value


def _source_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded = {
        ".git",
        ".venv",
        ".pytest_cache",
        "local_runs",
        "runs",
        "scratch",
        "tests",
        "__pycache__",
    }
    candidates = list(PROJECT_ROOT.rglob("*.py"))
    candidates.extend((PROJECT_ROOT / "scripts").glob("*.sbatch"))
    for path in sorted(set(candidates)):
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in excluded for part in rel.parts):
            continue
        try:
            canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"源码文件不是可规范化的 UTF-8 文本: {rel.as_posix()}") from exc
        rows.append({
            "path": rel.as_posix(),
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "size": len(canonical),
        })
    if not rows:
        raise RuntimeError("源码指纹列表为空")
    return rows


def _inspect_parquet_contract(data_file: Path) -> dict[str, Any]:
    try:
        parquet = pq.ParquetFile(data_file)
        rows = int(parquet.metadata.num_rows)
        columns = list(parquet.schema_arrow.names)
        first_group = parquet.read_row_group(0, columns=["time"])["time"]
        last_group = parquet.read_row_group(parquet.num_row_groups - 1, columns=["time"])["time"]
        first_time = first_group[0].as_py()
        last_time = last_group[len(last_group) - 1].as_py()
    except Exception as exc:
        raise RuntimeError("无法读取 Parquet 数据身份") from exc
    if rows <= 0 or not isinstance(first_time, int) or not isinstance(last_time, int):
        raise RuntimeError("Parquet time 必须是非空 unix_seconds 整数列")
    missing = [name for name in REQUIRED_DATA_COLUMNS if name not in columns]
    if missing:
        raise RuntimeError(f"Parquet 缺少训练列: {missing}")
    start = datetime.fromtimestamp(first_time, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    end = datetime.fromtimestamp(last_time, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return {"data_rows": rows, "data_start": start, "data_end": end, "columns": columns}


class SlurmTrainingManager:
    def __init__(
        self,
        client: SlurmTrainingClient | None = None,
        local_runs_root: Path | None = None,
    ) -> None:
        self.client = client or SlurmTrainingClient.from_environment()
        self.local_runs_root = (local_runs_root or LOCAL_RUNS_ROOT).resolve()
        self.local_runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = self._load_current_job()

    @classmethod
    def from_environment(cls) -> "SlurmTrainingManager":
        return cls()

    def _current_pointer(self) -> Path:
        return self.local_runs_root / "current.json"

    def _state_path(self, run_id: str) -> Path:
        return self.local_runs_root / run_id / "state.json"

    def _load_current_job(self) -> dict[str, Any] | None:
        pointer = self._current_pointer()
        if not pointer.is_file():
            return None
        try:
            run_id = json.loads(pointer.read_text(encoding="utf-8")).get("run_id")
            state_path = self._state_path(str(run_id))
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        return data if isinstance(data, dict) else None

    def _save(self) -> None:
        if not self._job:
            return
        _atomic_json(self._state_path(self._job["run_id"]), self._job)
        _atomic_json(self._current_pointer(), {"run_id": self._job["run_id"]})

    def _set_state(self, state: str, *, error: str | None = None) -> None:
        if not self._job:
            raise RuntimeError("没有当前 Slurm 任务")
        old = self._job["remote_state"]
        if state == "FAILED":
            if old in TERMINAL_REMOTE_STATES:
                return
        elif state not in ALLOWED_TRANSITIONS.get(old, set()):
            raise RuntimeError(f"非法状态跃迁: {old} -> {state}")
        self._job["remote_state"] = state
        self._job["updated_at"] = _utc_now()
        if state in TERMINAL_REMOTE_STATES:
            self._job["finished_at"] = _utc_now()
        if error:
            self._job["error"] = str(error)
        else:
            self._job.pop("last_poll_error", None)
        self._save()

    def _record_retryable_error(self, exc: Exception) -> None:
        if not self._job:
            return
        self._job["last_poll_error"] = str(exc)
        self._job["updated_at"] = _utc_now()
        self._save()

    @staticmethod
    def _legacy_state(remote_state: str) -> str:
        if remote_state in ACTIVE_REMOTE_STATES:
            return "running"
        if remote_state == "READY":
            return "completed"
        if remote_state == "CANCELLED":
            return "stopped"
        return "failed"

    def _public_job(self) -> dict[str, Any] | None:
        if not self._job:
            return None
        row = dict(self._job)
        row["state"] = self._legacy_state(row["remote_state"])
        return row

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            state = self._job.get("remote_state") if self._job else None
            return {
                "active": bool(state in ACTIVE_REMOTE_STATES),
                "backend": "slurm",
                "job": self._public_job(),
            }

    def _load_data_manifest(self, data_file: Path) -> dict[str, Any]:
        manifest_path = data_file.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError(
                f"远程训练要求数据 manifest: {manifest_path.name}"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("数据 manifest 无法读取") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("数据 manifest 必须是 JSON 对象")
        if payload.get("data_filename") != data_file.name:
            raise RuntimeError("数据 manifest 文件名不匹配")
        digest = sha256_file(data_file)
        if payload.get("data_sha256") != digest:
            raise RuntimeError("数据文件 SHA-256 与 manifest 不匹配")
        if payload.get("data_timezone") != "UTC":
            raise RuntimeError("数据 manifest 必须明确声明 UTC")
        source = payload.get("source")
        contract = DATA_SOURCE_CONTRACTS.get(source) if isinstance(source, str) else None
        if contract is None:
            raise RuntimeError("数据 manifest 的 source 不受支持")
        source_id, expected_format = contract
        if payload.get("format") != expected_format:
            raise RuntimeError("数据 manifest 的 format 与 source 不匹配")
        if payload.get("time_unit") != "unix_seconds":
            raise RuntimeError("数据 manifest 的 time_unit 必须是 unix_seconds")
        actual = _inspect_parquet_contract(data_file)
        for field in ("data_rows", "data_start", "data_end", "columns"):
            if payload.get(field) != actual[field]:
                raise RuntimeError(f"数据 manifest 的 {field} 与 Parquet 不匹配")
        expected_dataset_id = f"sha256:{digest}"
        if payload.get("dataset_id", expected_dataset_id) != expected_dataset_id:
            raise RuntimeError("数据 manifest 的 dataset_id 与文件哈希不匹配")
        return {**payload, "_source_id": source_id}

    def _build_run_manifest(
        self,
        *,
        run_id: str,
        data_file: Path,
        symbol: str,
        timeframe: str,
        from_scratch: bool,
    ) -> dict[str, Any]:
        data = self._load_data_manifest(data_file)
        if data.get("symbol") != symbol or data.get("timeframe") != timeframe:
            raise RuntimeError("请求的品种/周期与数据 manifest 不一致")
        train_steps = int(os.getenv("SLURM_TRAIN_STEPS", str(ModelConfig.TRAIN_STEPS)))
        cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
        if train_steps < 1 or train_steps > 1_000_000:
            raise RuntimeError("SLURM_TRAIN_STEPS 超出允许范围")
        if cpus < 1 or cpus > 64:
            raise RuntimeError("SLURM_CPUS_PER_TASK 超出允许范围")
        partition = os.getenv("SLURM_PARTITION", "cpu")
        qos = os.getenv("SLURM_QOS", "normal")
        time_limit = os.getenv("SLURM_TIME_LIMIT", "00:30:00")
        memory = os.getenv("SLURM_MEMORY", "")
        if partition != "cpu" or qos != "normal":
            raise RuntimeError("第一阶段只允许 cpu 分区和 normal QOS")
        if not re.fullmatch(r"(?:\d+-)?\d{2}:\d{2}:\d{2}", time_limit):
            raise RuntimeError("SLURM_TIME_LIMIT 必须是 [D-]HH:MM:SS")
        if memory and not re.fullmatch(r"[1-9]\d*(?:K|M|G|T)", memory):
            raise RuntimeError("SLURM_MEMORY 必须为空或使用 K/M/G/T 单位")
        requested_resources = {
            "partition": partition,
            "qos": qos,
            "cpus_per_task": cpus,
            "memory": memory,
            "time_limit": time_limit,
        }
        return {
            "run_id": run_id,
            "created_at": _utc_now(),
            "symbol": symbol,
            "timeframe": timeframe,
            "data_filename": data_file.name,
            "data_sha256": data["data_sha256"],
            "data_size": data_file.stat().st_size,
            "data_rows": int(data["data_rows"]),
            "data_start": data["data_start"],
            "data_end": data["data_end"],
            "data_timezone": "UTC",
            "dataset_id": data.get("dataset_id") or f"sha256:{data['data_sha256']}",
            "git_commit": _git_commit(),
            "source_files": _source_files(),
            "training_parameters": {
                "train_steps": train_steps,
                "from_scratch": bool(from_scratch),
            },
            "requested_resources": requested_resources,
            "local_source": data["_source_id"],
        }

    def start(
        self,
        data_file: str,
        symbol: str,
        timeframe: str,
        mode: str = "ftmo",
        *,
        from_scratch: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            if self._job and self._job.get("remote_state") in ACTIVE_REMOTE_STATES:
                raise RuntimeError(f"已有远程训练任务在运行: {self._job.get('run_id')}")
            path = Path(data_file).resolve()
            if not path.is_file():
                raise RuntimeError("训练数据文件不存在")
            run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + secrets.token_hex(4)
            run_dir = self.local_runs_root / run_id
            manifest = self._build_run_manifest(
                run_id=run_id,
                data_file=path,
                symbol=symbol,
                timeframe=timeframe,
                from_scratch=from_scratch,
            )
            manifest_path = run_dir / "run_manifest.json"
            _atomic_json(manifest_path, manifest)
            self._job = {
                "run_id": run_id,
                "slurm_job_id": None,
                "data_file": str(path),
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": mode,
                "backend": "slurm",
                "remote_state": "PREPARING",
                "pid": None,
                "log_path": f"local_runs/{run_id}/logs/tail.log",
                "artifact_dir": f"local_runs/{run_id}/artifacts",
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
                "finished_at": None,
                "exit_code": None,
                "error": None,
                "git_commit": manifest["git_commit"],
                "training_parameters": manifest["training_parameters"],
                "requested_resources": manifest["requested_resources"],
                "cancel_requested": False,
            }
            self._save()
            self._advance_pre_submission()
            if self._job.get("remote_state") == "FAILED":
                raise RuntimeError(f"Slurm 任务提交失败: {self._job.get('error')}")
            return self._public_job() or {}

    def _advance_pre_submission(self) -> None:
        if not self._job:
            return
        try:
            state = self._job.get("remote_state")
            run_id = self._job["run_id"]
            if state == "PREPARING":
                self.client.prepare(run_id)
                self._set_state("UPLOADING")
                state = "UPLOADING"
            if state == "UPLOADING":
                data_file = Path(self._job["data_file"]).resolve()
                manifest_file = self.local_runs_root / run_id / "run_manifest.json"
                if not data_file.is_file() or not manifest_file.is_file():
                    raise RuntimeError("恢复上传所需的本机输入不存在")
                try:
                    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("恢复上传所需的 run manifest 无法读取") from exc
                data_sha256 = manifest.get("data_sha256") if isinstance(manifest, dict) else None
                if not isinstance(data_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", data_sha256):
                    raise RuntimeError("恢复上传所需的数据哈希非法")
                self.client.upload_inputs(
                    run_id=run_id,
                    data_file=data_file,
                    manifest_file=manifest_file,
                    data_sha256=data_sha256,
                )
                self._set_state("SUBMITTING")
                state = "SUBMITTING"
            if state == "SUBMITTING":
                job_id = self.client.submit(run_id)
                self._job["slurm_job_id"] = job_id
                self._set_state("SUBMITTED")
                state = "SUBMITTED"
            if state == "SUBMITTED":
                self._set_state("PENDING")
                state = "PENDING"
            if state == "PENDING" and self._job.get("cancel_requested"):
                self._request_cancel()
        except SlurmTransportError as exc:
            self._record_retryable_error(exc)
        except Exception as exc:
            self._set_state("FAILED", error=str(exc))

    def _refresh_state(self) -> None:
        if not self._job:
            return
        state = self._job.get("remote_state")
        if state in {"PREPARING", "UPLOADING", "SUBMITTING", "SUBMITTED"}:
            self._advance_pre_submission()
            return
        job_id = self._job.get("slurm_job_id")
        if state not in {"PENDING", "RUNNING", "CANCELLING", "COMPLETED", "DOWNLOADING"}:
            return
        if state in {"COMPLETED", "DOWNLOADING"}:
            self._download()
            return
        if not job_id:
            self._set_state("FAILED", error="活动任务缺少 Slurm job ID")
            return
        try:
            remote = self.client.status(self._job["run_id"], job_id)
            slurm_state = str(remote.get("state") or "").upper().split("+")[0]
            self._job["slurm_state"] = slurm_state
            self._job["compute_node"] = remote.get("node")
            self._job["exit_code"] = remote.get("exit_code")
            self._job["remote_started_at"] = remote.get("started_at")
            self._job["remote_finished_at"] = remote.get("finished_at")
            self._job["elapsed"] = remote.get("elapsed")
            self._job["allocated_cpus"] = remote.get("allocated_cpus")
            self._job["total_cpu"] = remote.get("total_cpu")
            self._job["max_rss"] = remote.get("max_rss")
            self._job.pop("last_poll_error", None)
            self._save()
            if slurm_state in SLURM_PENDING:
                if state == "CANCELLING":
                    self._request_cancel(refresh_on_client_error=False)
                    return
                if state != "PENDING":
                    self._set_state("PENDING")
            elif slurm_state in SLURM_RUNNING:
                if state == "CANCELLING":
                    self._request_cancel(refresh_on_client_error=False)
                    return
                if state != "RUNNING":
                    self._set_state("RUNNING")
            elif slurm_state == "COMPLETED":
                self._set_state("COMPLETED")
                self._download()
            elif slurm_state == "CANCELLED":
                self._set_state("CANCELLED")
            elif slurm_state in SLURM_FAILED:
                reason = remote.get("reason") or slurm_state
                self._set_state("FAILED", error=f"Slurm {slurm_state}: {reason}")
            elif slurm_state:
                self._set_state("FAILED", error=f"未知 Slurm 终态: {slurm_state}")
        except SlurmTransportError as exc:
            self._record_retryable_error(exc)
        except Exception as exc:
            self._set_state("FAILED", error=f"Slurm 状态校验失败: {exc}")

    def _download(self) -> None:
        if not self._job or not self._job.get("slurm_job_id"):
            return
        if self._job["remote_state"] == "COMPLETED":
            self._set_state("DOWNLOADING")
        try:
            artifact_root = (
                self.local_runs_root / self._job["run_id"] / "artifacts"
            )
            manifest = self.client.download_result(
                run_id=self._job["run_id"],
                job_id=self._job["slurm_job_id"],
                local_artifact_root=artifact_root,
                expected_commit=self._job["git_commit"],
            )
            if manifest.get("status") != "COMPLETED" or int(manifest.get("exit_code", 1)) != 0:
                raise RuntimeError("Worker 结果 manifest 不是成功终态")
            self._validate_result_manifest(artifact_root, manifest)
            self._publish_verified_artifacts(artifact_root, manifest)
            result_manifest_path = artifact_root / "output" / "result_manifest.json"
            self._job["result_manifest_sha256"] = sha256_file(result_manifest_path)
            self._job["artifact_count"] = len(manifest["artifacts"])
            self._save()
            self._set_state("READY")
        except SlurmTransportError as exc:
            self._record_retryable_error(exc)
        except Exception as exc:
            self._set_state("FAILED", error=f"结果下载或校验失败: {exc}")

    def _validate_result_manifest(
        self, artifact_root: Path, manifest: dict[str, Any]
    ) -> None:
        if not self._job:
            raise RuntimeError("没有当前 Slurm 任务")
        run_manifest_path = self.local_runs_root / self._job["run_id"] / "run_manifest.json"
        try:
            expected = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("本机 run manifest 无法复核") from exc
        exact_fields = (
            "symbol",
            "timeframe",
            "data_filename",
            "data_sha256",
            "data_size",
            "data_rows",
            "dataset_id",
            "local_source",
            "git_commit",
            "source_files",
            "training_parameters",
            "requested_resources",
        )
        for field in exact_fields:
            if manifest.get(field) != expected.get(field):
                raise RuntimeError(f"结果 manifest 的 {field} 与本机 run 身份不匹配")
        if manifest.get("run_manifest_sha256") != sha256_file(run_manifest_path):
            raise RuntimeError("结果 manifest 的 run manifest 哈希不匹配")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("结果 manifest 缺少正式产物")
        rows: dict[str, dict[str, Any]] = {}
        for row in artifacts:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise RuntimeError("结果 manifest 产物结构非法")
            path = row["path"]
            if path in rows:
                raise RuntimeError("结果 manifest 含重复产物路径")
            rows[path] = row
        symbol = re.escape(str(expected["symbol"]))
        checkpoints = sorted(path for path in rows if re.fullmatch(rf"checkpoints/ckpt_{symbol}_step_\d+\.pt", path))
        strategies = sorted(path for path in rows if path == f"strategies/best_{expected['symbol']}.json")
        histories = sorted(path for path in rows if path == f"training_history_{expected['symbol']}.json")
        if not checkpoints or len(strategies) != 1 or len(histories) != 1:
            raise RuntimeError("结果必须同时包含 checkpoint、目标策略和训练历史")
        if sorted(manifest.get("checkpoint_files") or []) != checkpoints:
            raise RuntimeError("结果 manifest 的 checkpoint_files 不一致")
        if sorted(manifest.get("strategy_files") or []) != strategies:
            raise RuntimeError("结果 manifest 的 strategy_files 不一致")
        declared_hashes = manifest.get("artifact_sha256")
        actual_hashes = {path: row.get("sha256") for path, row in rows.items()}
        if declared_hashes != actual_hashes:
            raise RuntimeError("结果 manifest 的 artifact_sha256 不一致")

        strategy_path = artifact_root.joinpath(*strategies[0].split("/"))
        try:
            strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("回传策略不是合法 UTF-8 JSON") from exc
        formula = strategy.get("formula") if isinstance(strategy, dict) else None
        try:
            score = float(strategy.get("best_score"))
        except (TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError("回传策略 best_score 非法") from exc
        if (
            strategy.get("symbol") != expected["symbol"]
            or strategy.get("timeframe") != expected["timeframe"]
            or strategy.get("vocab_version") != VOCAB_VERSION
            or strategy.get("train_steps") != expected["training_parameters"]["train_steps"]
            or not isinstance(formula, list)
            or not formula
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= FORMULA_VOCAB.size
                for token in formula
            )
            or not math.isfinite(score)
        ):
            raise RuntimeError("回传策略身份、词表、公式或训练参数非法")
        data_file = strategy.get("data_file")
        if not isinstance(data_file, str) or Path(data_file).name != expected["data_filename"]:
            raise RuntimeError("回传策略的数据文件身份不匹配")

    def _publish_verified_artifacts(
        self, artifact_root: Path, manifest: dict[str, Any]
    ) -> None:
        if not self._job:
            raise RuntimeError("没有当前 Slurm 任务")
        for row in manifest["artifacts"]:
            rel = str(row["path"])
            source = artifact_root.joinpath(*rel.split("/"))
            if rel.startswith("checkpoints/"):
                _atomic_copy(source, PROJECT_ROOT / rel)
            elif rel.startswith("training_history_"):
                _atomic_copy(source, PROJECT_ROOT / rel)
        self._publish_strategy(artifact_root, manifest)

        checkpoint_files = list(manifest["checkpoint_files"])
        strategy_files = list(manifest["strategy_files"])
        history_files = [
            str(row["path"])
            for row in manifest["artifacts"]
            if str(row["path"]).startswith("training_history_")
        ]
        if len(strategy_files) != 1 or len(history_files) != 1:
            raise RuntimeError("无法为已验证产物建立唯一发布指针")
        symbol = str(manifest["symbol"])
        pointer = {
            "format": PUBLISHED_BUNDLE_FORMAT,
            "run_id": self._job["run_id"],
            "symbol": symbol,
            "timeframe": manifest["timeframe"],
            "dataset_id": manifest["dataset_id"],
            "local_source": manifest["local_source"],
            "data_filename": manifest["data_filename"],
            "data_sha256": manifest["data_sha256"],
            "data_file": self._job["data_file"],
            "artifact_root": str(artifact_root.resolve()),
            "checkpoint_files": checkpoint_files,
            "strategy_file": strategy_files[0],
            "history_file": history_files[0],
            "artifact_sha256": {
                rel: manifest["artifact_sha256"][rel]
                for rel in [*checkpoint_files, strategy_files[0], history_files[0]]
            },
            "result_manifest_sha256": sha256_file(
                artifact_root / "output" / "result_manifest.json"
            ),
            "published_at": _utc_now(),
        }
        safe_symbol = symbol.replace(".", "_")
        _atomic_json(
            PROJECT_ROOT / "published_training" / f"current_{safe_symbol}.json",
            pointer,
        )

    def _publish_strategy(self, artifact_root: Path, manifest: dict[str, Any]) -> None:
        if not self._job:
            raise RuntimeError("没有当前 Slurm 任务")
        found = False
        for row in manifest.get("artifacts") or []:
            rel = str(row.get("path") or "") if isinstance(row, dict) else ""
            if not rel.startswith("strategies/best_") or not rel.endswith(".json"):
                continue
            found = True
            source = artifact_root.joinpath(*rel.split("/"))
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("已验证策略在发布前无法读取") from exc
            payload["dataset_id"] = manifest.get("dataset_id")
            payload["local_source"] = manifest.get("local_source")
            payload["data_filename"] = Path(self._job["data_file"]).name
            payload["data_sha256"] = manifest.get("data_sha256")
            payload["data_file"] = self._job["data_file"]
            payload["run_id"] = self._job["run_id"]
            dest = PROJECT_ROOT / rel
            _atomic_json(dest, payload)
        if not found:
            raise RuntimeError("结果中没有可发布的目标策略")

    def _request_cancel(self, *, refresh_on_client_error: bool = True) -> bool:
        if not self._job or not self._job.get("slurm_job_id"):
            return False
        if self._job.get("remote_state") != "CANCELLING":
            self._set_state("CANCELLING")
        try:
            self.client.cancel(self._job["run_id"], self._job["slurm_job_id"])
            return True
        except SlurmTransportError as exc:
            self._record_retryable_error(exc)
            return True
        except SlurmClientError as exc:
            # cancel 与自然完成存在竞态；立即用 status/sacct 判定真实终态。
            self._record_retryable_error(exc)
            if refresh_on_client_error:
                self._refresh_state()
            return self._job.get("remote_state") != "FAILED"

    def stop(self) -> bool:
        with self._lock:
            if not self._job or self._job.get("remote_state") not in ACTIVE_REMOTE_STATES:
                return False
            state = self._job.get("remote_state")
            if state in {"COMPLETED", "DOWNLOADING"}:
                self._refresh_state()
                return False
            if state in {"PREPARING", "UPLOADING"}:
                self._set_state("CANCELLED")
                return True
            if state == "SUBMITTING":
                self._job["cancel_requested"] = True
                self._save()
                self._advance_pre_submission()
                return self._job.get("remote_state") != "FAILED"
            if state == "CANCELLING":
                return True
            job_id = self._job.get("slurm_job_id")
            if not job_id:
                self._set_state("FAILED", error="任务尚未获得 job ID，无法取消")
                return False
            return self._request_cancel()

    def tail_log(self, lines: int = 200) -> list[str]:
        with self._lock:
            if not self._job:
                return []
            log_path = self.local_runs_root / self._job["run_id"] / "logs" / "tail.log"
            rows: list[str] = []
            if self._job.get("slurm_job_id"):
                try:
                    rows = self.client.tail(
                        self._job["run_id"], self._job["slurm_job_id"], lines
                    )
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                except SlurmClientError:
                    rows = []
            if not rows and log_path.is_file():
                rows = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return [strip_ansi(row) for row in rows[-max(1, min(int(lines), 500)):]]

    def parse_step_from_log(self) -> int | None:
        for line in reversed(self.tail_log(80)):
            match = re.search(r"\[(\d+)/\d+\]", line)
            if match:
                return int(match.group(1))
        return None
