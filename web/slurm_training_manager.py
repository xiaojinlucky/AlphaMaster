"""持久化的 Slurm 训练状态机，接口兼容现有 Web 训练管理器。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from config import Config
from data_pipeline.a_share_akshare import AKSHARE_SLICE_SEALED_EVALUATION
from data_pipeline.a_share_data import (
    ASHARE_DATASET_FORMAT,
    ASHARE_SOURCE,
    ASHARE_SOURCE_ID,
    ASHARE_SPECS_BY_TIMEFRAME,
)
from data_pipeline.dataset_contracts import (
    AKSHARE_HFQ_FORMAT,
    AKSHARE_HFQ_SOURCE_ID,
    AKSHARE_SOURCE,
    MT5_LEGACY_SOURCE,
    MT5_LEGACY_SOURCE_ID,
    OKX_LEGACY_SOURCE_ID,
    OKX_SOURCE_ID,
    REMOTE_SOURCE_CONTRACTS,
    infer_periods_per_year,
    resolve_okx_source_id,
)
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
TRAINING_SOURCE_PATTERNS = (
    "config.py",
    "train_file.py",
    "data_pipeline/*.py",
    "model_core/*.py",
    "strategy_manager/__init__.py",
    "strategy_manager/signal.py",
    "utils/train_logging.py",
    "utils/training_runtime.py",
    "scripts/slurm_control.py",
    "scripts/train_slurm_worker.py",
    "scripts/train_alphamaster.sbatch",
)
DATA_SOURCE_CONTRACTS = {
    **REMOTE_SOURCE_CONTRACTS,
    ASHARE_SOURCE: (ASHARE_SOURCE_ID, ASHARE_DATASET_FORMAT),
    AKSHARE_SOURCE: (AKSHARE_HFQ_SOURCE_ID, AKSHARE_HFQ_FORMAT),
}
REQUIRED_DATA_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")
LOCAL_RUNS_ROOT = Path(
    os.getenv(
        "ALPHAMASTER_LOCAL_RUNS_ROOT",
        str(PROJECT_ROOT / "local_runs"),
    )
).expanduser().resolve()
PUBLISHED_BUNDLE_FORMAT = "alphamaster_published_bundle_v1"
RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$")
SQUEUE_EXPIRED_JOB_ERROR = "slurm_load_jobs error: Invalid job id specified"
RESULT_MANIFEST_NOT_READY_ERROR = "result manifest不存在"
RECOVERY_UNKNOWN = "RECOVERY_UNKNOWN"
ACTIVE_REMOTE_STATES = {
    RECOVERY_UNKNOWN,
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
SLURM_PENDING = {
    "PENDING",
    "CONFIGURING",
    "EXPEDITING",
    "POWER_UP_NODE",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESV_DEL_HOLD",
    "SPECIAL_EXIT",
    "SUSPENDED",
}
SLURM_RUNNING = {
    "RUNNING",
    "COMPLETING",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "UPDATE_DB",
}
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


def _age_seconds(value: object) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
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


def current_git_commit() -> str:
    """当前本机控制层对应的 Git 提交。"""
    return _git_commit()


def _source_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: set[Path] = set()
    for pattern in TRAINING_SOURCE_PATTERNS:
        candidates.update(PROJECT_ROOT.glob(pattern))
    for path in sorted(set(candidates)):
        rel = path.relative_to(PROJECT_ROOT)
        if not path.is_file():
            raise RuntimeError(f"训练运行时源码缺失: {rel.as_posix()}")
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


def _source_files_sha256(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("训练源码指纹列表为空或格式非法")
    raw = json.dumps(
        rows,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_training_source_sha256() -> str:
    """当前将被上传到 Slurm 的完整训练源码身份。"""
    return _source_files_sha256(_source_files())


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
    return {
        "data_rows": rows,
        "data_start": start,
        "data_end": end,
        "columns": columns,
        "periods_per_year": infer_periods_per_year(
            rows=rows,
            start_unix=first_time,
            end_unix=last_time,
        ),
    }


class SlurmTrainingManager:
    def __init__(
        self,
        client: SlurmTrainingClient | None = None,
        local_runs_root: Path | None = None,
    ) -> None:
        self.client = client or SlurmTrainingClient.from_environment()
        self.local_runs_root = (local_runs_root or LOCAL_RUNS_ROOT).resolve()
        self.local_runs_root.mkdir(parents=True, exist_ok=True)
        self._snapshot_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = self._load_current_job()
        self._cached_status = self._status_payload()

    @classmethod
    def from_environment(cls) -> SlurmTrainingManager:
        return cls()

    def _current_pointer(self) -> Path:
        return self.local_runs_root / "current.json"

    def _state_path(self, run_id: str) -> Path:
        return self.local_runs_root / run_id / "state.json"

    @staticmethod
    def _recovery_unknown_job(
        error: str,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "slurm_job_id": None,
            "backend": "slurm",
            "remote_state": RECOVERY_UNKNOWN,
            "started_at": None,
            "updated_at": _utc_now(),
            "finished_at": None,
            "error": error,
            "recovery_error": error,
        }

    def _load_current_job(self) -> dict[str, Any] | None:
        pointer = self._current_pointer()
        try:
            pointer_stat = pointer.stat()
        except FileNotFoundError:
            if os.path.lexists(pointer):
                return self._recovery_unknown_job(
                    "训练状态恢复失败：current.json 是失效链接"
                )
            return None
        except OSError as exc:
            return self._recovery_unknown_job(
                f"训练状态恢复失败：无法检查 current.json（{exc}）"
            )
        if not stat.S_ISREG(pointer_stat.st_mode):
            return self._recovery_unknown_job(
                "训练状态恢复失败：current.json 不是普通文件"
            )
        try:
            pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return self._recovery_unknown_job(
                f"训练状态恢复失败：current.json 无法读取或 JSON 损坏（{exc}）"
            )
        if not isinstance(pointer_data, dict):
            return self._recovery_unknown_job(
                "训练状态恢复失败：current.json 必须是 JSON 对象"
            )
        run_id = pointer_data.get("run_id")
        if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
            return self._recovery_unknown_job(
                "训练状态恢复失败：current.json 的 run_id 缺失或非法"
            )

        state_path = self._state_path(run_id)
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return self._recovery_unknown_job(
                f"训练状态恢复失败：current.json 指向的 state.json "
                f"缺失、无法读取或 JSON 损坏（{exc}）",
                run_id=run_id,
            )
        remote_state = (
            data.get("remote_state")
            if isinstance(data, dict)
            else None
        )
        required_text_fields = (
            "data_file",
            "symbol",
            "timeframe",
            "git_commit",
        )
        stored_source_sha256 = (
            data.get("expected_source_sha256")
            if isinstance(data, dict)
            else None
        )
        if (
            not isinstance(data, dict)
            or data.get("run_id") != run_id
            or not isinstance(remote_state, str)
            or remote_state not in ALLOWED_TRANSITIONS
            or data.get("backend") != "slurm"
            or any(
                not isinstance(data.get(field), str)
                or not data[field].strip()
                for field in required_text_fields
            )
            or re.fullmatch(r"[0-9a-f]{40}", data["git_commit"]) is None
            or not isinstance(data.get("training_parameters"), dict)
            or not isinstance(data.get("requested_resources"), dict)
            or "slurm_job_id" not in data
            or (
                stored_source_sha256 is not None
                and (
                    not isinstance(stored_source_sha256, str)
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        stored_source_sha256,
                    )
                    is None
                )
            )
            or (
                remote_state
                in {"PREPARING", "UPLOADING", "SUBMITTING"}
                and stored_source_sha256 is None
            )
        ):
            return self._recovery_unknown_job(
                "训练状态恢复失败：current.json 指向的 state.json "
                "结构或任务身份损坏",
                run_id=run_id,
            )
        return data

    def _save(self) -> None:
        if not self._job:
            return
        if self._job.get("remote_state") == RECOVERY_UNKNOWN:
            raise RuntimeError("恢复错误快照不可写回训练状态")
        _atomic_json(self._state_path(self._job["run_id"]), self._job)
        _atomic_json(self._current_pointer(), {"run_id": self._job["run_id"]})
        self._refresh_snapshot_cache()

    def _status_payload(self) -> dict[str, Any]:
        state = self._job.get("remote_state") if self._job else None
        remote_status_age_seconds = (
            _age_seconds(
                self._job.get("remote_polled_at")
                or self._job.get("started_at")
            )
            if self._job
            else None
        )
        return {
            "active": bool(state in ACTIVE_REMOTE_STATES),
            "backend": "slurm",
            "job": copy.deepcopy(self._public_job()),
            "status_unknown": bool(state == RECOVERY_UNKNOWN),
            "remote_status_age_seconds": remote_status_age_seconds,
            "remote_status_stale": bool(
                state == RECOVERY_UNKNOWN
                or (
                    state in {"PENDING", "RUNNING", "CANCELLING"}
                    and (
                        bool(self._job.get("last_poll_error"))
                        or remote_status_age_seconds is None
                        or remote_status_age_seconds > 120
                    )
                )
            ),
        }

    def _refresh_snapshot_cache(self) -> None:
        payload = self._status_payload()
        with self._snapshot_lock:
            self._cached_status = payload

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
        if remote_state == RECOVERY_UNKNOWN:
            return "failed"
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
            return self._status_payload()

    def snapshot(self) -> dict[str, Any]:
        """只读最近一次已确认状态；不等待节点选择、SSH 或 Slurm。"""
        with self._snapshot_lock:
            return copy.deepcopy(self._cached_status)

    def current_source_sha256(self) -> str:
        return current_training_source_sha256()

    def current_git_commit(self) -> str:
        return current_git_commit()

    @staticmethod
    def _normalize_source_sha256(value: object) -> str:
        normalized = str(value or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
            raise RuntimeError(
                "expected_source_sha256 必须是 64 位 SHA-256"
            )
        return normalized

    @staticmethod
    def _normalize_git_commit(value: object) -> str:
        normalized = str(value or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", normalized) is None:
            raise RuntimeError(
                "expected_git_commit 必须是 40 位 Git 提交"
            )
        return normalized

    def _resolve_expected_source_sha256(
        self,
        expected_source_sha256: str | None,
    ) -> str:
        current = current_training_source_sha256()
        expected = (
            current
            if expected_source_sha256 is None
            else self._normalize_source_sha256(expected_source_sha256)
        )
        if current != expected:
            raise RuntimeError(
                "当前训练源码 SHA-256 与 expected_source_sha256 不一致，"
                "拒绝提交"
            )
        return expected

    def _assert_dispatch_source_identity(self, action: str) -> None:
        if not self._job:
            raise RuntimeError("没有当前 Slurm 任务")
        expected = self._normalize_source_sha256(
            self._job.get("expected_source_sha256")
        )
        current = current_training_source_sha256()
        if current != expected:
            raise RuntimeError(
                "训练源码 SHA-256 已漂移，"
                f"拒绝在 {action} 前继续提交"
            )

    def run_source_sha256(self, run_id: str) -> str:
        if RUN_ID_RE.fullmatch(str(run_id)) is None:
            raise RuntimeError("训练源码身份的 run_id 非法")
        manifest_path = self.local_runs_root / run_id / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("无法读取 run 的训练源码身份") from exc
        return _source_files_sha256(manifest.get("source_files"))

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
        if source == MT5_LEGACY_SOURCE:
            if payload.get("source_family") != "MetaTrader5":
                raise RuntimeError("旧 MT5 manifest 的 source_family 不匹配")
            if payload.get("provenance_level") != "legacy_user_attested":
                raise RuntimeError("旧 MT5 manifest 的 provenance_level 不匹配")
            if payload.get("attestation_scope") != "exact_file_bytes":
                raise RuntimeError("旧 MT5 manifest 的 attestation_scope 不匹配")
            if payload.get("registration_method") != "legacy_sidecar_registration_v1":
                raise RuntimeError("旧 MT5 manifest 的 registration_method 不受支持")
            if payload.get("bar_timestamp_semantics") != "source_bar_open":
                raise RuntimeError("旧 MT5 manifest 的时间语义不匹配")
            plan_sha = payload.get("registration_plan_sha256")
            if not isinstance(plan_sha, str) or re.fullmatch(
                r"[0-9a-f]{64}", plan_sha
            ) is None:
                raise RuntimeError("旧 MT5 manifest 的注册计划哈希非法")
        if source_id == OKX_SOURCE_ID:
            try:
                source_id = resolve_okx_source_id(
                    payload,
                    symbol=str(payload.get("symbol") or ""),
                    timeframe=str(payload.get("timeframe") or "").upper(),
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        actual = _inspect_parquet_contract(data_file)
        for field in ("data_rows", "data_start", "data_end", "columns"):
            if payload.get(field) != actual[field]:
                raise RuntimeError(f"数据 manifest 的 {field} 与 Parquet 不匹配")
        expected_dataset_id = f"sha256:{digest}"
        if payload.get("dataset_id", expected_dataset_id) != expected_dataset_id:
            raise RuntimeError("数据 manifest 的 dataset_id 与文件哈希不匹配")
        if source in {ASHARE_SOURCE, AKSHARE_SOURCE}:
            timeframe = str(payload.get("timeframe") or "").upper()
            spec = ASHARE_SPECS_BY_TIMEFRAME.get(timeframe)
            if spec is None:
                raise RuntimeError("A 股 manifest 的 timeframe 不受支持")
            if payload.get("market") != "CN_A_SHARE":
                raise RuntimeError("A 股 manifest 的 market 必须是 CN_A_SHARE")
            if payload.get("bar_timestamp_semantics") != "bar_close":
                raise RuntimeError("A 股 manifest 必须声明 bar_close 时间语义")
            if payload.get("source_timezone") != "Asia/Shanghai":
                raise RuntimeError("A 股 manifest 的源时区必须是 Asia/Shanghai")
            expected_close_times = [
                value.strftime("%H:%M") for value in spec.close_times
            ]
            if payload.get("session_close_times") != expected_close_times:
                raise RuntimeError("A 股 manifest 的收盘时刻表不匹配")
            if payload.get("periods_per_year") != spec.periods_per_year:
                raise RuntimeError("A 股 manifest 的 periods_per_year 不匹配")
            if payload.get("minimum_bars") != spec.minimum_bars:
                raise RuntimeError("A 股 manifest 的 minimum_bars 不匹配")
            if int(payload["data_rows"]) < spec.minimum_bars:
                raise RuntimeError("A 股数据不足两个交易年")
            if source == ASHARE_SOURCE:
                if payload.get("source_time_encoding") != (
                    "floor(china_local_wall_clock_unix_seconds/1000)"
                ):
                    raise RuntimeError(
                        "AShareLocal manifest 的旧 time 编码声明不匹配"
                    )
                expected_source_filename = (
                    f"{payload.get('symbol')}_{spec.legacy_period}.parquet"
                )
                if payload.get("source_filename") != expected_source_filename:
                    raise RuntimeError(
                        "AShareLocal manifest 的 source_filename 不匹配"
                    )
                if not isinstance(
                    payload.get("source_sha256"),
                    str,
                ) or re.fullmatch(
                    r"[0-9a-f]{64}",
                    payload["source_sha256"],
                ) is None:
                    raise RuntimeError(
                        "AShareLocal manifest 的 source_sha256 非法"
                    )
            else:
                if timeframe != "D1":
                    raise RuntimeError("AKShare hfq 训练数据只支持 D1")
                if payload.get("source_id") != AKSHARE_HFQ_SOURCE_ID:
                    raise RuntimeError("AKShare hfq manifest 的 source_id 不匹配")
                if payload.get("provider") != "AKShare":
                    raise RuntimeError("AKShare hfq manifest 的 provider 不匹配")
                if payload.get("provider_interface") != "stock_zh_a_daily":
                    raise RuntimeError(
                        "AKShare hfq manifest 的 provider_interface 不匹配"
                    )
                if payload.get("adjustment") != "hfq":
                    raise RuntimeError("AKShare hfq manifest 的复权方式不匹配")
                if payload.get("bar_completion") != "completed_trading_days_only":
                    raise RuntimeError(
                        "AKShare hfq manifest 必须只包含已完成交易日"
                    )
                if payload.get("adjustment_history_semantics") != (
                    "cumulative_historical_factor_not_latest_price_normalized"
                ):
                    raise RuntimeError(
                        "AKShare hfq manifest 的历史版本语义不匹配"
                    )
                derivation = payload.get("derivation")
                if (
                    isinstance(derivation, dict)
                    and derivation.get("purpose")
                    == AKSHARE_SLICE_SEALED_EVALUATION
                ):
                    raise RuntimeError("封存评估数据禁止进入模型训练")
                version = payload.get("provider_version")
                if not isinstance(version, str) or re.fullmatch(
                    r"[0-9]+\.[0-9]+\.[0-9]+(?:[.+-].*)?",
                    version,
                ) is None:
                    raise RuntimeError(
                        "AKShare hfq manifest 的 provider_version 非法"
                    )
                source_hash = payload.get("source_response_sha256")
                if not isinstance(source_hash, str) or re.fullmatch(
                    r"[0-9a-f]{64}",
                    source_hash,
                ) is None:
                    raise RuntimeError(
                        "AKShare hfq manifest 的来源响应哈希非法"
                    )
                request = payload.get("request")
                canonical_symbol = str(payload.get("symbol") or "")
                provider_prefix = (
                    "sh"
                    if canonical_symbol.startswith("6")
                    else "sz"
                    if canonical_symbol.startswith(("0", "3"))
                    else ""
                )
                if not isinstance(request, dict) or request != {
                    "canonical_symbol": canonical_symbol,
                    "symbol": f"{provider_prefix}{canonical_symbol}",
                    "start_date": request.get("start_date")
                    if isinstance(request, dict)
                    else None,
                    "end_date": request.get("end_date")
                    if isinstance(request, dict)
                    else None,
                    "adjust": "hfq",
                }:
                    raise RuntimeError("AKShare hfq manifest 的 request 不匹配")
                if not provider_prefix:
                    raise RuntimeError(
                        "AKShare hfq manifest 的新浪股票代码前缀不受支持"
                    )
                for field in ("start_date", "end_date"):
                    if re.fullmatch(
                        r"[0-9]{8}",
                        str(request.get(field) or ""),
                    ) is None:
                        raise RuntimeError(
                            f"AKShare hfq manifest 的 request.{field} 非法"
                        )
                if request["start_date"] >= request["end_date"]:
                    raise RuntimeError(
                        "AKShare hfq manifest 的请求日期范围非法"
                    )
            periods_per_year = spec.periods_per_year
            minimum_bars: int | None = spec.minimum_bars
        else:
            inferred_periods_value = actual.get("periods_per_year")
            if inferred_periods_value is None:
                try:
                    start_unix = int(
                        datetime.fromisoformat(
                            str(actual["data_start"]).replace("Z", "+00:00")
                        ).timestamp()
                    )
                    end_unix = int(
                        datetime.fromisoformat(
                            str(actual["data_end"]).replace("Z", "+00:00")
                        ).timestamp()
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("无法从数据范围推导 periods_per_year") from exc
                inferred_periods_value = infer_periods_per_year(
                    rows=int(actual["data_rows"]),
                    start_unix=start_unix,
                    end_unix=end_unix,
                )
            inferred_periods = int(inferred_periods_value)
            if payload.get("periods_per_year", inferred_periods) != inferred_periods:
                raise RuntimeError(
                    "MT5/OKX manifest 的 periods_per_year 与数据范围不匹配"
                )
            if payload.get("minimum_bars", Config.MIN_BARS) != Config.MIN_BARS:
                raise RuntimeError(
                    f"MT5/OKX manifest 的 minimum_bars 必须是 {Config.MIN_BARS}"
                )
            if int(payload["data_rows"]) < Config.MIN_BARS:
                raise RuntimeError(f"MT5/OKX 数据不足 {Config.MIN_BARS} bars")
            periods_per_year = inferred_periods
            minimum_bars = Config.MIN_BARS
            if source == MT5_LEGACY_SOURCE and source_id != MT5_LEGACY_SOURCE_ID:
                raise RuntimeError("旧 MT5 manifest 的来源身份映射错误")
            if (
                source == "OKX"
                and source_id not in {OKX_SOURCE_ID, OKX_LEGACY_SOURCE_ID}
            ):
                raise RuntimeError("OKX manifest 的来源身份映射错误")
        return {
            **payload,
            "periods_per_year": periods_per_year,
            "minimum_bars": minimum_bars,
            "_source_id": source_id,
        }

    @staticmethod
    def _resolve_run_settings(
        *,
        from_scratch: bool,
        train_steps: int | None,
        cpus_per_task: int | None,
        memory: str | None,
        time_limit: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            resolved_steps = (
                int(os.getenv("SLURM_TRAIN_STEPS", str(ModelConfig.TRAIN_STEPS)))
                if train_steps is None
                else train_steps
            )
            resolved_cpus = (
                int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
                if cpus_per_task is None
                else cpus_per_task
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Slurm 训练步数或 CPU 数量不是整数") from exc
        if (
            isinstance(resolved_steps, bool)
            or not isinstance(resolved_steps, int)
            or not 1 <= resolved_steps <= 1_000_000
        ):
            raise RuntimeError("SLURM_TRAIN_STEPS 超出允许范围")
        if (
            isinstance(resolved_cpus, bool)
            or not isinstance(resolved_cpus, int)
            or not 1 <= resolved_cpus <= 64
        ):
            raise RuntimeError("SLURM_CPUS_PER_TASK 超出允许范围")

        partition = os.getenv("SLURM_PARTITION", "cpu")
        qos = os.getenv("SLURM_QOS", "normal")
        resolved_time_limit = (
            os.getenv("SLURM_TIME_LIMIT", "00:30:00")
            if time_limit is None
            else str(time_limit).strip()
        )
        resolved_memory = (
            os.getenv("SLURM_MEMORY", "")
            if memory is None
            else str(memory).strip().upper()
        )
        if partition != "cpu" or qos != "normal":
            raise RuntimeError("第一阶段只允许 cpu 分区和 normal QOS")
        if not re.fullmatch(
            r"(?:\d+-)?\d{2}:\d{2}:\d{2}",
            resolved_time_limit,
        ):
            raise RuntimeError("SLURM_TIME_LIMIT 必须是 [D-]HH:MM:SS")
        if resolved_memory and not re.fullmatch(
            r"[1-9]\d*(?:K|M|G|T)",
            resolved_memory,
        ):
            raise RuntimeError("SLURM_MEMORY 必须为空或使用 K/M/G/T 单位")
        return (
            {
                "train_steps": resolved_steps,
                "from_scratch": bool(from_scratch),
            },
            {
                "partition": partition,
                "qos": qos,
                "cpus_per_task": resolved_cpus,
                "memory": resolved_memory,
                "time_limit": resolved_time_limit,
            },
        )

    def _build_run_manifest(
        self,
        *,
        run_id: str,
        data_file: Path,
        symbol: str,
        timeframe: str,
        training_parameters: dict[str, Any],
        requested_resources: dict[str, Any],
        git_commit: str,
    ) -> dict[str, Any]:
        data = self._load_data_manifest(data_file)
        if data.get("symbol") != symbol or data.get("timeframe") != timeframe:
            raise RuntimeError("请求的品种/周期与数据 manifest 不一致")
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
            "columns": list(data["columns"]),
            "data_timezone": "UTC",
            "dataset_id": data.get("dataset_id") or f"sha256:{data['data_sha256']}",
            "periods_per_year": int(data["periods_per_year"]),
            "minimum_bars": data.get("minimum_bars"),
            "git_commit": git_commit,
            "source_files": _source_files(),
            "training_parameters": dict(training_parameters),
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
        planned_run_id: str | None = None,
        train_steps: int | None = None,
        cpus_per_task: int | None = None,
        memory: str | None = None,
        time_limit: str | None = None,
        expected_source_sha256: str | None = None,
        expected_git_commit: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            if (
                self._job
                and self._job.get("remote_state") == RECOVERY_UNKNOWN
            ):
                raise RuntimeError(
                    str(
                        self._job.get("recovery_error")
                        or "训练状态恢复失败，拒绝启动新任务"
                    )
                )
            resolved_source_sha256 = (
                self._resolve_expected_source_sha256(
                    expected_source_sha256
                )
            )
            resolved_git_commit = (
                _git_commit()
                if expected_git_commit is None
                else self._normalize_git_commit(expected_git_commit)
            )
            path = Path(data_file).resolve()
            (
                training_parameters,
                requested_resources,
            ) = self._resolve_run_settings(
                from_scratch=from_scratch,
                train_steps=train_steps,
                cpus_per_task=cpus_per_task,
                memory=memory,
                time_limit=time_limit,
            )
            if planned_run_id is not None:
                planned_run_id = str(planned_run_id).strip()
                if RUN_ID_RE.fullmatch(planned_run_id) is None:
                    raise RuntimeError("planned_run_id 格式非法")
                if self._job and self._job.get("run_id") == planned_run_id:
                    stored_source_sha256 = self._job.get(
                        "expected_source_sha256"
                    )
                    if (
                        stored_source_sha256 is not None
                        and self._normalize_source_sha256(
                            stored_source_sha256
                        )
                        != resolved_source_sha256
                    ):
                        raise RuntimeError(
                            "planned_run_id 已绑定不同的训练源码身份"
                        )
                    stored_git_commit = self._job.get("git_commit")
                    if (
                        stored_git_commit is not None
                        and self._normalize_git_commit(stored_git_commit)
                        != resolved_git_commit
                    ):
                        raise RuntimeError(
                            "planned_run_id 已绑定不同的运行提交"
                        )
                    expected_parameters = self._job.get("training_parameters") or {}
                    expected_resources = self._job.get("requested_resources") or {}
                    if (
                        Path(str(self._job.get("data_file") or "")).resolve() != path
                        or self._job.get("symbol") != symbol
                        or self._job.get("timeframe") != timeframe
                        or expected_parameters != training_parameters
                        or expected_resources != requested_resources
                    ):
                        raise RuntimeError(
                            "planned_run_id 已绑定不同的训练输入或参数"
                        )
                    if (
                        self._job.get("remote_state") == "FAILED"
                        and not self._job.get("slurm_job_id")
                    ):
                        return self._retry_pre_submission(
                            path=path,
                            symbol=symbol,
                            timeframe=timeframe,
                            training_parameters=training_parameters,
                            requested_resources=requested_resources,
                            expected_source_sha256=resolved_source_sha256,
                        )
                    return self._public_job() or {}
            if self._job and self._job.get("remote_state") in ACTIVE_REMOTE_STATES:
                raise RuntimeError(f"已有远程训练任务在运行: {self._job.get('run_id')}")
            if not path.is_file():
                raise RuntimeError("训练数据文件不存在")
            run_id = planned_run_id or (
                datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_")
                + secrets.token_hex(4)
            )
            run_dir = self.local_runs_root / run_id
            if run_dir.exists():
                raise RuntimeError(f"run_id 运行目录已存在，拒绝覆盖: {run_id}")
            manifest = self._build_run_manifest(
                run_id=run_id,
                data_file=path,
                symbol=symbol,
                timeframe=timeframe,
                training_parameters=training_parameters,
                requested_resources=requested_resources,
                git_commit=resolved_git_commit,
            )
            if (
                _source_files_sha256(manifest.get("source_files"))
                != resolved_source_sha256
            ):
                raise RuntimeError(
                    "训练源码在 run manifest 构建期间发生漂移，拒绝提交"
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
                "expected_source_sha256": resolved_source_sha256,
                "training_parameters": manifest["training_parameters"],
                "requested_resources": manifest["requested_resources"],
                "cancel_requested": False,
            }
            self._save()
            self._advance_pre_submission()
            if self._job.get("remote_state") == "FAILED":
                raise RuntimeError(f"Slurm 任务提交失败: {self._job.get('error')}")
            return self._public_job() or {}

    def _retry_pre_submission(
        self,
        *,
        path: Path,
        symbol: str,
        timeframe: str,
        training_parameters: dict[str, Any],
        requested_resources: dict[str, Any],
        expected_source_sha256: str,
    ) -> dict[str, Any]:
        """复用未取得 job ID 的 run；提交过的 run 永不在这里重投。"""
        if (
            not self._job
            or self._job.get("remote_state") != "FAILED"
            or self._job.get("slurm_job_id")
        ):
            raise RuntimeError("当前 run 不满足提交前恢复条件")
        run_id = str(self._job["run_id"])
        manifest_path = self.local_runs_root / run_id / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("提交前恢复所需的 run manifest 无法读取") from exc
        expected = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_filename": path.name,
            "data_sha256": sha256_file(path),
            "source_files": _source_files(),
            "training_parameters": training_parameters,
            "requested_resources": requested_resources,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise RuntimeError(
                    f"提交前恢复拒绝：冻结的 {field} 已变化"
                )
        if (
            _source_files_sha256(manifest.get("source_files"))
            != expected_source_sha256
        ):
            raise RuntimeError("提交前恢复拒绝：冻结的训练源码身份已变化")

        history = list(self._job.get("retry_history") or [])
        history.append(
            {
                "failed_at": self._job.get("finished_at")
                or self._job.get("updated_at"),
                "error": self._job.get("error"),
            }
        )
        self._job.update(
            {
                "remote_state": "PREPARING",
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
                "finished_at": None,
                "exit_code": None,
                "error": None,
                "cancel_requested": False,
                "expected_source_sha256": expected_source_sha256,
                "retry_count": int(self._job.get("retry_count") or 0) + 1,
                "retry_history": history[-20:],
            }
        )
        self._save()
        self._advance_pre_submission()
        if self._job.get("remote_state") == "FAILED":
            raise RuntimeError(f"Slurm 任务恢复失败: {self._job.get('error')}")
        return self._public_job() or {}

    def _advance_pre_submission(self) -> None:
        if not self._job:
            return
        try:
            state = self._job.get("remote_state")
            run_id = self._job["run_id"]
            if state == "PREPARING":
                self._assert_dispatch_source_identity("prepare")
                self.client.prepare(run_id)
                self._set_state("UPLOADING")
                state = "UPLOADING"
            if state == "UPLOADING":
                self._assert_dispatch_source_identity("upload")
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
                self._assert_dispatch_source_identity("submit")
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
            normalized_status = str(remote.get("status") or "").upper()
            self._job["slurm_state"] = slurm_state
            self._job["compute_node"] = remote.get("node")
            self._job["exit_code"] = remote.get("exit_code")
            self._job["remote_started_at"] = remote.get("started_at")
            self._job["remote_finished_at"] = remote.get("finished_at")
            self._job["elapsed"] = remote.get("elapsed")
            self._job["allocated_cpus"] = remote.get("allocated_cpus")
            self._job["total_cpu"] = remote.get("total_cpu")
            self._job["max_rss"] = remote.get("max_rss")
            self._job["remote_polled_at"] = _utc_now()
            self._job.pop("last_poll_error", None)
            self._save()
            if slurm_state in SLURM_PENDING:
                effective_status = "PENDING"
            elif slurm_state in SLURM_RUNNING:
                effective_status = "RUNNING"
            elif slurm_state == "COMPLETED":
                effective_status = "COMPLETED"
            elif slurm_state == "CANCELLED":
                effective_status = "CANCELLED"
            elif slurm_state in SLURM_FAILED:
                effective_status = "FAILED"
            elif normalized_status in {
                "PENDING",
                "RUNNING",
                "COMPLETED",
                "CANCELLED",
                "FAILED",
            }:
                effective_status = normalized_status
            else:
                effective_status = ""

            if effective_status == "PENDING":
                if state == "CANCELLING":
                    self._request_cancel(refresh_on_client_error=False)
                    return
                if state != "PENDING":
                    self._set_state("PENDING")
            elif effective_status == "RUNNING":
                if state == "CANCELLING":
                    self._request_cancel(refresh_on_client_error=False)
                    return
                if state != "RUNNING":
                    self._set_state("RUNNING")
            elif effective_status == "COMPLETED":
                self._set_state("COMPLETED")
                self._download()
            elif effective_status == "CANCELLED":
                self._set_state("CANCELLED")
            elif effective_status == "FAILED":
                reason = remote.get("reason") or slurm_state
                self._set_state("FAILED", error=f"Slurm {slurm_state}: {reason}")
            elif slurm_state or normalized_status:
                self._record_retryable_error(
                    RuntimeError(
                        "暂不识别的 Slurm 活动状态: "
                        f"{normalized_status or slurm_state}"
                    )
                )
        except SlurmTransportError as exc:
            self._record_retryable_error(exc)
        except SlurmClientError as exc:
            if SQUEUE_EXPIRED_JOB_ERROR in str(exc):
                # squeue 会在作业离开活动队列一段时间后对旧 job_id 返回此错误。
                # 远端 result 接口仍会按 run/job 绑定校验结果 manifest；只有完整
                # 下载并通过本机哈希与身份复核后才会进入 READY。
                self._set_state("COMPLETED")
                self._download()
                return
            # 查询失败只代表本轮没有取得可信观测，不能证明远端作业失败。
            # 只有成功返回的 Slurm 明确终态才有权结束当前 run。
            self._record_retryable_error(exc)
        except Exception as exc:
            self._record_retryable_error(
                RuntimeError(f"Slurm 状态监控异常: {exc}")
            )

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
        except SlurmClientError as exc:
            if RESULT_MANIFEST_NOT_READY_ERROR in str(exc):
                # Slurm 已结束与共享文件系统发布 result manifest 之间可能有短暂窗口。
                # 保持 DOWNLOADING，下一轮继续读取同一 run/job，绝不重新提交。
                self._record_retryable_error(exc)
                return
            self._set_state("FAILED", error=f"结果下载或校验失败: {exc}")
        except Exception as exc:
            self._set_state("FAILED", error=f"结果下载或校验失败: {exc}")

    def _validate_result_manifest(
        self, artifact_root: Path, manifest: dict[str, Any]
    ) -> None:
        if not self._job:
            raise RuntimeError("没有当前 Slurm 任务")
        manifest_run_id = manifest.get("run_id")
        if (
            manifest_run_id not in (None, "")
            and manifest_run_id != self._job["run_id"]
        ):
            raise RuntimeError("结果 manifest 的 run_id 与当前 run 不匹配")
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
            "data_start",
            "data_end",
            "columns",
            "dataset_id",
            "local_source",
            "periods_per_year",
            "minimum_bars",
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
        timeframe = re.escape(str(expected["timeframe"]))
        data_sha256 = re.escape(str(expected["data_sha256"]))
        all_checkpoint_artifacts = sorted(
            path for path in rows if path.startswith("checkpoints/")
        )
        checkpoints = sorted(
            path
            for path in rows
            if re.fullmatch(
                rf"checkpoints/{timeframe}/{data_sha256}/run_[0-9]{{20}}/"
                rf"ckpt_{symbol}_step_[0-9]{{4,}}\.pt",
                path,
            )
        )
        strategies = sorted(path for path in rows if path == f"strategies/best_{expected['symbol']}.json")
        histories = sorted(path for path in rows if path == f"training_history_{expected['symbol']}.json")
        if not checkpoints or len(strategies) != 1 or len(histories) != 1:
            raise RuntimeError("结果必须同时包含 checkpoint、目标策略和训练历史")
        if all_checkpoint_artifacts != checkpoints:
            raise RuntimeError("结果包含未登记或路径非法的 checkpoint")
        if sorted(manifest.get("checkpoint_files") or []) != checkpoints:
            raise RuntimeError("结果 manifest 的 checkpoint_files 不一致")
        if sorted(manifest.get("strategy_files") or []) != strategies:
            raise RuntimeError("结果 manifest 的 strategy_files 不一致")
        declared_hashes = manifest.get("artifact_sha256")
        actual_hashes = {path: row.get("sha256") for path, row in rows.items()}
        if declared_hashes != actual_hashes:
            raise RuntimeError("结果 manifest 的 artifact_sha256 不一致")

        checkpoint_identity = {
            "symbol": expected["symbol"],
            "timeframe": expected["timeframe"],
            "dataset_id": expected["dataset_id"],
            "data_sha256": expected["data_sha256"],
            "local_source": expected["local_source"],
            "periods_per_year": expected["periods_per_year"],
            "minimum_bars": expected["minimum_bars"],
        }
        for relative in checkpoints:
            checkpoint_path = artifact_root.joinpath(*relative.split("/"))
            try:
                with checkpoint_path.open("rb") as handle:
                    checkpoint = torch.load(
                        handle,
                        map_location="cpu",
                        weights_only=True,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"回传 checkpoint 无法安全读取: {relative}"
                ) from exc
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"回传 checkpoint 顶层不是对象: {relative}")
            if checkpoint.get("vocab_version") != VOCAB_VERSION:
                raise RuntimeError(
                    f"回传 checkpoint 公式执行版本不匹配: {relative}"
                )
            for field, expected_value in checkpoint_identity.items():
                actual_value = checkpoint.get(field)
                if (
                    type(actual_value) is not type(expected_value)
                    or actual_value != expected_value
                ):
                    raise RuntimeError(
                        f"回传 checkpoint 的 {field} 与 run 身份不匹配: "
                        f"{relative}"
                    )

        strategy_path = artifact_root.joinpath(*strategies[0].split("/"))
        try:
            strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("回传策略不是合法 UTF-8 JSON") from exc
        strategy_run_id = (
            strategy.get("run_id")
            if isinstance(strategy, dict)
            else None
        )
        if (
            strategy_run_id not in (None, "")
            and strategy_run_id != self._job["run_id"]
        ):
            raise RuntimeError("回传策略 run_id 与当前 run 不匹配")
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
            or strategy.get("periods_per_year") != expected["periods_per_year"]
            or strategy.get("minimum_bars") != expected.get("minimum_bars")
            or strategy.get("dataset_id") != expected["dataset_id"]
            or strategy.get("data_sha256") != expected["data_sha256"]
            or strategy.get("local_source") != expected["local_source"]
            or strategy.get("data_rows") != expected["data_rows"]
            or strategy.get("data_start") != expected["data_start"]
            or strategy.get("data_end") != expected["data_end"]
            or strategy.get("columns") != expected["columns"]
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
        verified_checkpoints = set(manifest["checkpoint_files"])
        for row in manifest["artifacts"]:
            rel = str(row["path"])
            if rel in verified_checkpoints:
                source = artifact_root.joinpath(*rel.split("/"))
                _atomic_copy(source, PROJECT_ROOT / rel)
            elif rel.startswith("training_history_"):
                source = artifact_root.joinpath(*rel.split("/"))
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
            "periods_per_year": manifest["periods_per_year"],
            "minimum_bars": manifest.get("minimum_bars"),
            "data_filename": manifest["data_filename"],
            "data_sha256": manifest["data_sha256"],
            "data_rows": manifest["data_rows"],
            "data_start": manifest["data_start"],
            "data_end": manifest["data_end"],
            "columns": list(manifest["columns"]),
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
            payload["periods_per_year"] = manifest.get("periods_per_year")
            payload["minimum_bars"] = manifest.get("minimum_bars")
            payload["data_filename"] = Path(self._job["data_file"]).name
            payload["data_sha256"] = manifest.get("data_sha256")
            payload["data_rows"] = manifest.get("data_rows")
            payload["data_start"] = manifest.get("data_start")
            payload["data_end"] = manifest.get("data_end")
            payload["columns"] = list(manifest.get("columns") or [])
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

    def stop(
        self,
        *,
        expected_run_id: str | None = None,
        expected_job_id: str | int | None = None,
    ) -> bool:
        with self._lock:
            if (
                self._job
                and self._job.get("remote_state") == RECOVERY_UNKNOWN
            ):
                raise RuntimeError(
                    "训练状态恢复失败，拒绝取消未知任务"
                )
            if not self._job or self._job.get("remote_state") not in ACTIVE_REMOTE_STATES:
                return False
            current_run_id = str(self._job.get("run_id") or "")
            current_job_id = str(self._job.get("slurm_job_id") or "")
            if (
                expected_run_id is not None
                and current_run_id != str(expected_run_id)
            ):
                raise RuntimeError("训练任务 run_id 已变化，拒绝取消")
            if (
                expected_job_id is not None
                and current_job_id != str(expected_job_id)
            ):
                raise RuntimeError("训练任务 job_id 已变化，拒绝取消")
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
                self._job["cancel_requested"] = True
                self._record_retryable_error(
                    RuntimeError("取消请求等待 Slurm job ID 恢复")
                )
                return True
            return self._request_cancel()

    def tail_log(
        self,
        lines: int = 200,
        *,
        expected_run_id: str | None = None,
        expected_job_id: str | int | None = None,
        final: bool = False,
    ) -> list[str]:
        with self._lock:
            if not self._job:
                return []
            run_id = str(self._job["run_id"])
            if expected_run_id is not None and run_id != expected_run_id:
                return []
            log_path = self.local_runs_root / run_id / "logs" / "tail.log"
            job_id = str(self._job.get("slurm_job_id") or "")
            if (
                expected_job_id is not None
                and job_id != str(expected_job_id)
            ):
                return []

        rows: list[str] = []
        if job_id:
            with self._log_lock:
                try:
                    rows = self.client.tail(run_id, job_id, lines)
                    _atomic_text(
                        log_path,
                        "\n".join(rows) + ("\n" if rows else ""),
                    )
                    with self._lock:
                        if (
                            self._job
                            and str(self._job.get("run_id") or "") == run_id
                            and str(self._job.get("slurm_job_id") or "") == job_id
                        ):
                            self._job.pop("last_log_poll_error", None)
                            if final:
                                self._job["final_log_refreshed_at"] = _utc_now()
                            self._save()
                except SlurmClientError as exc:
                    with self._lock:
                        if (
                            self._job
                            and str(self._job.get("run_id") or "") == run_id
                            and str(self._job.get("slurm_job_id") or "") == job_id
                        ):
                            self._job["last_log_poll_error"] = str(exc)
                            self._save()
                    rows = []
        if not rows and log_path.is_file():
            rows = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        return [
            strip_ansi(row)
            for row in rows[-max(1, min(int(lines), 500)):]
        ]

    def cached_log_tail(
        self,
        lines: int = 200,
        *,
        run_id: str | None = None,
    ) -> list[str]:
        """只读本机日志缓存；不等待远程日志拉取。"""
        resolved_run_id = str(run_id or "").strip()
        if not resolved_run_id:
            status = self.snapshot()
            job = status.get("job") if isinstance(status, dict) else None
            resolved_run_id = (
                str(job.get("run_id") or "")
                if isinstance(job, dict)
                else ""
            )
        if not resolved_run_id:
            return []
        log_path = self.local_runs_root / resolved_run_id / "logs" / "tail.log"
        if not log_path.is_file():
            return []
        try:
            rows = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            return []
        return [
            strip_ansi(row)
            for row in rows[-max(1, min(int(lines), 500)):]
        ]

    def parse_step_from_log(
        self,
        *,
        refresh_remote: bool = True,
        run_id: str | None = None,
    ) -> int | None:
        rows = (
            self.tail_log(80, expected_run_id=run_id)
            if refresh_remote
            else self.cached_log_tail(80, run_id=run_id)
        )
        for line in reversed(rows):
            match = re.search(r"\[(\d+)/\d+\]", line)
            if match:
                return int(match.group(1))
        return None
