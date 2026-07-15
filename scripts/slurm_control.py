"""AlphaMaster 服务器端固定 Slurm 控制器。

本模块只接受结构化参数，所有路径和 Slurm 命令均固定。它不会执行客户端
提供的 shell 片段，也不会选择或固定具体计算节点。
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ROOT = Path("/hwdata/home/jinqc/Quant/AlphaMaster")
SLURM_BIN = Path("/opt/gridview/slurm/bin")

RUN_ID_RE = re.compile(r"^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
FILENAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}_"
    r"(?:M1|M5|M15|M30|H1|H4|D1|W1|MN1)\.parquet$",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
TIME_RE = re.compile(r"^(?:(?P<days>[0-7])-)?(?P<hours>[0-9]{1,2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$")
MEMORY_RE = re.compile(r"^(?P<amount>[1-9][0-9]{0,5})(?P<unit>[MG])$")
ALLOWED_LOCAL_SOURCES = frozenset({"mt5", "okx"})

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_DATA_BYTES = 64 * 1024**3
MIN_MEMORY_MIB = 512
MAX_MEMORY_MIB = 512 * 1024
MAX_TAIL_LINES = 1000
MAX_TAIL_BYTES = 2 * 1024 * 1024

REQUIRED_SOURCE_FILES = (
    "train_file.py",
    "model_core/config.py",
    "scripts/train_slurm_worker.py",
    "scripts/train_alphamaster.sbatch",
)

_ACTIVE_PENDING = {"PENDING", "CONFIGURING", "SUSPENDED"}
_ACTIVE_RUNNING = {"RUNNING", "COMPLETING", "RESIZING", "STAGE_OUT"}
_FAILED_STATES = {
    "BOOT_FAIL",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


class ControlError(RuntimeError):
    """拒绝不满足固定控制合同的请求。"""


def validate_run_id(value: str) -> str:
    if not isinstance(value, str) or not RUN_ID_RE.fullmatch(value):
        raise ControlError("run_id 必须符合 run_YYYYMMDDTHHMMSSZ_八位十六进制格式")
    return value


def validate_job_id(value: str) -> str:
    if not isinstance(value, str) or not JOB_ID_RE.fullmatch(value):
        raise ControlError("job_id 必须是纯数字正整数")
    return value


def validate_filename(value: str) -> str:
    if not isinstance(value, str) or not FILENAME_RE.fullmatch(value):
        raise ControlError("数据文件名必须是合法的 {symbol}_{timeframe}.parquet 基名")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ControlError("数据文件名不得包含路径")
    return value


def validate_sha256(value: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ControlError("SHA-256 必须是 64 位十六进制字符串")
    return value.lower()


def validate_resources(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlError("requested_resources 必须是对象")
    partition = value.get("partition")
    qos = value.get("qos")
    cpus = value.get("cpus_per_task")
    time_limit = value.get("time_limit")
    memory = value.get("memory")

    if partition != "cpu":
        raise ControlError("Slurm 分区只允许 cpu")
    if qos != "normal":
        raise ControlError("Slurm QOS 只允许 normal")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or not 1 <= cpus <= 64:
        raise ControlError("cpus_per_task 必须在 1..64")
    if not isinstance(time_limit, str):
        raise ControlError("time_limit 必须是 Slurm 时间字符串")
    match = TIME_RE.fullmatch(time_limit)
    if match is None:
        raise ControlError("time_limit 必须为 HH:MM:SS 或 D-HH:MM:SS，且不超过 7 天")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours"))
    total_seconds = ((days * 24 + hours) * 60 + int(match.group("minutes"))) * 60 + int(match.group("seconds"))
    if hours > 23 or not 60 <= total_seconds <= 7 * 24 * 3600:
        raise ControlError("time_limit 必须在 1 分钟到 7 天之间")

    if not isinstance(memory, str):
        raise ControlError("memory 必须使用 M 或 G 单位")
    if memory == "":
        return {
            "partition": partition,
            "qos": qos,
            "cpus_per_task": cpus,
            "time_limit": time_limit,
            "memory": memory,
        }
    mem_match = MEMORY_RE.fullmatch(memory)
    if mem_match is None:
        raise ControlError("memory 必须是正整数加 M/G，例如 8192M 或 8G")
    amount = int(mem_match.group("amount"))
    memory_mib = amount * (1024 if mem_match.group("unit") == "G" else 1)
    if not MIN_MEMORY_MIB <= memory_mib <= MAX_MEMORY_MIB:
        raise ControlError("memory 必须在 512M 到 512G 之间")

    return {
        "partition": partition,
        "qos": qos,
        "cpus_per_task": cpus,
        "time_limit": time_limit,
        "memory": memory,
    }


def _runs_root() -> Path:
    return ROOT / "runs"


def _run_dir(run_id: str, *, require_exists: bool = True) -> Path:
    validate_run_id(run_id)
    runs_root = _runs_root().resolve()
    path = runs_root / run_id
    if path.parent != runs_root:
        raise ControlError("run 目录越界")
    if require_exists:
        _require_directory(path, "run 目录")
    return path


def _require_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ControlError(f"{label}不存在") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ControlError(f"{label}必须是真实目录，不能是符号链接")


def _require_regular_file(path: Path, label: str, *, max_bytes: int | None = None) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ControlError(f"{label}不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ControlError(f"{label}必须是普通文件，不能是符号链接")
    if max_bytes is not None and info.st_size > max_bytes:
        raise ControlError(f"{label}超过大小上限")
    return info.st_size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    token = secrets.token_hex(8)
    temp = path.with_name(f".{path.name}.{token}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
    token = secrets.token_hex(8)
    temp = path.with_name(f".{path.name}.{token}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        if temp.exists():
            temp.unlink()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label, max_bytes=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise ControlError(f"{label}不是合法 UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ControlError(f"{label}顶层必须是对象")
    return value


def _validate_source_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 512:
        raise ControlError("source_files 必须是 1..512 项列表")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            raise ControlError("source_files 项必须是对象")
        relative = row.get("path")
        path = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
        if (
            not isinstance(relative, str)
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or len(relative) > 240
            or path.suffix not in {".py", ".sbatch"}
            or relative in seen
        ):
            raise ControlError("source_files 含非法或重复路径")
        size = row.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 16 * 1024 * 1024:
            raise ControlError("source_files 文件大小非法")
        seen.add(relative)
        rows.append({"path": relative, "size": size, "sha256": validate_sha256(row.get("sha256"))})
    if not set(REQUIRED_SOURCE_FILES).issubset(seen):
        raise ControlError("source_files 缺少固定关键源码")
    return rows


def _validate_manifest(manifest: Mapping[str, Any], run_id: str, filename: str) -> dict[str, Any]:
    if manifest.get("run_id") != run_id:
        raise ControlError("manifest run_id 不匹配")
    if manifest.get("data_filename") != filename:
        raise ControlError("manifest data_filename 不匹配")
    validate_sha256(manifest.get("data_sha256"))
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or not GIT_COMMIT_RE.fullmatch(commit):
        raise ControlError("manifest git_commit 必须是 40 位十六进制提交")
    rows = manifest.get("data_rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ControlError("manifest data_rows 必须是正整数")
    if manifest.get("local_source") not in ALLOWED_LOCAL_SOURCES:
        raise ControlError("manifest local_source 不受支持")

    training = manifest.get("training_parameters")
    if not isinstance(training, dict):
        raise ControlError("training_parameters 必须是对象")
    steps = training.get("train_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1_000_000:
        raise ControlError("train_steps 必须在 1..1000000")
    from_scratch = training.get("from_scratch", False)
    if not isinstance(from_scratch, bool):
        raise ControlError("from_scratch 必须是布尔值")

    validate_resources(manifest.get("requested_resources"))
    _validate_source_files(manifest.get("source_files"))
    return dict(manifest)


def prepare_run(run_id: str, filename: str | None = None) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    if filename is not None:
        filename = validate_filename(filename)
    if not ROOT.exists() or not ROOT.is_dir():
        raise ControlError("固定 AlphaMaster 根目录不存在")

    runs_root = _runs_root()
    runs_root.mkdir(mode=0o700, exist_ok=True)
    _require_directory(runs_root, "runs 根目录")
    run_dir = _run_dir(run_id, require_exists=False)
    marker = run_dir / ".run_control.json"
    created = False
    try:
        run_dir.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        _require_directory(run_dir, "run 目录")

    if created:
        _atomic_write_json(
            marker,
            {"run_id": run_id, "data_filename": filename, "owner": getpass.getuser()},
        )
    else:
        saved = _read_json(marker, "run 控制标记")
        if saved.get("run_id") != run_id or (filename is not None and saved.get("data_filename") not in {None, filename}):
            raise ControlError("现有 run 目录与请求不匹配")
        if saved.get("owner") != getpass.getuser():
            raise ControlError("现有 run 目录不属于当前用户")
    os.chmod(run_dir, 0o700)

    for name in ("input", "logs", "checkpoints", "strategies", "output"):
        child = run_dir / name
        child.mkdir(mode=0o700, exist_ok=True)
        _require_directory(child, f"{name} 目录")
        os.chmod(child, 0o700)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "data_partial": str(run_dir / "input" / f"{filename}.partial") if filename else None,
        "manifest_partial": str(run_dir / "input" / "run_manifest.json.partial"),
    }


def finalize_upload(
    run_id: str,
    filename: str,
    *,
    size_bytes: int,
    sha256: str,
) -> dict[str, Any]:
    filename = validate_filename(filename)
    expected_hash = validate_sha256(sha256)
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or not 0 < size_bytes <= MAX_DATA_BYTES:
        raise ControlError("size_bytes 必须在 1 字节到 64 GiB 之间")
    run_dir = _run_dir(run_id)
    marker = _read_json(run_dir / ".run_control.json", "run 控制标记")
    if marker.get("run_id") != run_id or marker.get("data_filename") not in {None, filename}:
        raise ControlError("上传文件名与 prepare 阶段不匹配")
    if marker.get("owner") != getpass.getuser():
        raise ControlError("run 目录不属于当前用户")

    input_dir = run_dir / "input"
    _require_directory(input_dir, "input 目录")
    data_partial = input_dir / f"{filename}.partial"
    data_final = input_dir / filename
    manifest_partial = input_dir / "run_manifest.json.partial"
    manifest_final = input_dir / "run_manifest.json"

    data_candidates = [path for path in (data_final, data_partial) if path.exists()]
    if not data_candidates:
        raise ControlError("上传数据不存在")
    data_hashes: set[str] = set()
    for source in data_candidates:
        actual_size = _require_regular_file(source, "上传数据", max_bytes=MAX_DATA_BYTES)
        if actual_size != size_bytes:
            raise ControlError("上传数据大小与声明不匹配")
        actual_hash = _sha256_file(source)
        if actual_hash != expected_hash:
            raise ControlError("上传数据 SHA-256 不匹配")
        data_hashes.add(actual_hash)
    if len(data_hashes) != 1:
        raise ControlError("数据 partial 与正式文件内容不一致")

    manifest_candidates = [path for path in (manifest_final, manifest_partial) if path.exists()]
    if not manifest_candidates:
        raise ControlError("上传 manifest 不存在")
    manifest_hashes: set[str] = set()
    for source in manifest_candidates:
        candidate = _read_json(source, "上传 manifest")
        _validate_manifest(candidate, run_id, filename)
        if validate_sha256(candidate["data_sha256"]) != expected_hash:
            raise ControlError("manifest 与命令声明的数据 SHA-256 不匹配")
        manifest_size = candidate.get("data_size", candidate.get("data_size_bytes"))
        if manifest_size is not None and manifest_size != size_bytes:
            raise ControlError("manifest data_size_bytes 不匹配")
        manifest_hashes.add(_sha256_file(source))
    if len(manifest_hashes) != 1:
        raise ControlError("manifest partial 与正式文件内容不一致")
    manifest_hash = next(iter(manifest_hashes))

    if not data_final.exists():
        os.replace(data_partial, data_final)
        os.chmod(data_final, 0o600)
    elif data_partial.exists():
        data_partial.unlink()
    if not manifest_final.exists():
        os.replace(manifest_partial, manifest_final)
        os.chmod(manifest_final, 0o600)
    elif manifest_partial.exists():
        manifest_partial.unlink()
    if marker.get("data_filename") is None:
        marker["data_filename"] = filename
        _atomic_write_json(run_dir / ".run_control.json", marker)
    _atomic_write_text(input_dir / "run_manifest.sha256", manifest_hash + "\n")

    return {
        "run_id": run_id,
        "data_filename": filename,
        "data_size_bytes": size_bytes,
        "data_sha256": expected_hash,
        "run_manifest_sha256": manifest_hash,
        "finalized": True,
    }


def _load_finalized_manifest(run_id: str) -> tuple[Path, dict[str, Any], str]:
    run_dir = _run_dir(run_id)
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest = _read_json(manifest_path, "run manifest")
    filename = validate_filename(manifest.get("data_filename"))
    _validate_manifest(manifest, run_id, filename)
    hash_path = run_dir / "input" / "run_manifest.sha256"
    _require_regular_file(hash_path, "manifest 哈希", max_bytes=128)
    expected = validate_sha256(hash_path.read_text(encoding="ascii").strip())
    actual = _sha256_file(manifest_path)
    if actual != expected:
        raise ControlError("run manifest 哈希不匹配")
    return run_dir, manifest, actual


def _clean_slurm_env() -> dict[str, str]:
    return {
        "PATH": f"{SLURM_BIN}:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _run_slurm(args: Sequence[str], *, runner: Runner | None = None) -> subprocess.CompletedProcess[str]:
    execute = runner or subprocess.run
    result = execute(
        list(args),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_clean_slurm_env(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Slurm 命令失败").strip()[:1000]
        raise ControlError(detail)
    return result


def _job_name(run_id: str) -> str:
    return f"alphamaster_{run_id}"


def _binding_path(run_dir: Path) -> Path:
    return run_dir / ".slurm_job.json"


def submit_run(run_id: str, *, runner: Runner | None = None) -> dict[str, Any]:
    run_dir, manifest, manifest_hash = _load_finalized_manifest(run_id)
    resources = validate_resources(manifest["requested_resources"])
    binding_path = _binding_path(run_dir)
    if binding_path.exists():
        binding = _read_json(binding_path, "Slurm 作业绑定")
        if binding.get("run_id") != run_id or binding.get("manifest_sha256") != manifest_hash:
            raise ControlError("现有 Slurm 作业绑定与本次 run 不匹配")
        job_id = validate_job_id(binding.get("job_id"))
        _load_binding(run_id, job_id)
        return {"run_id": run_id, "job_id": job_id, "submitted": False, "idempotent": True}

    lock_path = run_dir / ".submit.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ControlError("该 run 正在提交或上次提交结果不确定，拒绝重复提交") from exc
    os.write(lock_fd, (json.dumps({"run_id": run_id, "owner": getpass.getuser()}) + "\n").encode("utf-8"))
    os.fsync(lock_fd)
    submission_may_exist = False

    try:
        # 取得互斥后再次检查，覆盖两个控制器同时进入 submit 的竞态。
        if binding_path.exists():
            binding = _read_json(binding_path, "Slurm 作业绑定")
            job_id = validate_job_id(binding.get("job_id"))
            _load_binding(run_id, job_id)
            os.close(lock_fd)
            lock_fd = -1
            lock_path.unlink()
            return {"run_id": run_id, "job_id": job_id, "submitted": False, "idempotent": True}

        script = ROOT / "scripts" / "train_alphamaster.sbatch"
        _require_regular_file(script, "固定 sbatch 脚本", max_bytes=256 * 1024)
        command = [
            str(SLURM_BIN / "sbatch"),
            "--parsable",
            f"--job-name={_job_name(run_id)}",
            f"--partition={resources['partition']}",
            f"--qos={resources['qos']}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resources['cpus_per_task']}",
            f"--time={resources['time_limit']}",
            f"--chdir={run_dir}",
            f"--output={run_dir / 'logs' / 'slurm.out'}",
            f"--error={run_dir / 'logs' / 'slurm.err'}",
            "--export=NONE",
            str(script),
            run_id,
        ]
        if resources["memory"]:
            command.insert(command.index(f"--chdir={run_dir}"), f"--mem={resources['memory']}")
        if any(arg == "--wrap" or arg.startswith("--wrap=") or arg == "--nodelist" or arg.startswith("--nodelist=") for arg in command):
            raise ControlError("固定提交命令包含禁止参数")
        result = _run_slurm(command, runner=runner)
        submission_may_exist = True
        job_text = result.stdout.strip()
        job_id = validate_job_id(job_text)
        os.ftruncate(lock_fd, 0)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        os.write(lock_fd, (json.dumps({"run_id": run_id, "job_id": job_id}) + "\n").encode("utf-8"))
        os.fsync(lock_fd)
        binding = {
            "run_id": run_id,
            "job_id": job_id,
            "job_name": _job_name(run_id),
            "owner": getpass.getuser(),
            "run_dir": str(run_dir.resolve()),
            "manifest_sha256": manifest_hash,
            "requested_resources": resources,
        }
        _atomic_write_json(binding_path, binding)
        os.close(lock_fd)
        lock_fd = -1
        lock_path.unlink()
        return {"run_id": run_id, "job_id": job_id, "submitted": True, "idempotent": False}
    except Exception:
        # sbatch 成功后若绑定落盘失败，保留锁并拒绝后续重投，避免重复作业。
        if not submission_may_exist and lock_path.exists():
            if lock_fd >= 0:
                os.close(lock_fd)
                lock_fd = -1
            lock_path.unlink()
        raise
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)


def _load_binding(run_id: str, job_id: str) -> tuple[Path, dict[str, Any]]:
    job_id = validate_job_id(job_id)
    run_dir = _run_dir(run_id)
    binding = _read_json(_binding_path(run_dir), "Slurm 作业绑定")
    if binding.get("run_id") != run_id or binding.get("job_id") != job_id:
        raise ControlError("job_id 与 run_id 的服务器绑定不匹配")
    if binding.get("owner") != getpass.getuser():
        raise ControlError("作业不属于当前用户")
    if binding.get("job_name") != _job_name(run_id):
        raise ControlError("作业名与 run_id 不匹配")
    if Path(str(binding.get("run_dir", ""))).resolve() != run_dir.resolve():
        raise ControlError("作业工作目录与 run_id 不匹配")
    return run_dir, binding


def _normalize_state(raw: str) -> str:
    state = _base_state(raw)
    if state in _ACTIVE_PENDING:
        return "PENDING"
    if state in _ACTIVE_RUNNING:
        return "RUNNING"
    if state == "COMPLETED":
        return "COMPLETED"
    if state.startswith("CANCELLED"):
        return "CANCELLED"
    if state in _FAILED_STATES:
        return "FAILED"
    raise ControlError(f"未知 Slurm 状态: {raw}")


def _base_state(raw: str) -> str:
    return raw.strip().split()[0].rstrip("+").upper() if raw.strip() else ""


def _active_job(run_dir: Path, run_id: str, job_id: str, *, runner: Runner | None = None) -> dict[str, str] | None:
    result = _run_slurm(
        [str(SLURM_BIN / "squeue"), "-h", "-j", job_id, "-o", "%i|%u|%j|%T|%Z|%N"],
        runner=runner,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) != 1:
        raise ControlError("squeue 返回了歧义记录")
    fields = lines[0].split("|", 5)
    if len(fields) != 6:
        raise ControlError("squeue 输出格式异常")
    found_id, owner, name, state_value, workdir, node = fields
    if found_id != job_id or owner != getpass.getuser():
        raise ControlError("squeue 作业身份不匹配")
    if name != _job_name(run_id):
        raise ControlError("squeue 作业名与 run_id 不匹配")
    if Path(workdir).resolve() != run_dir.resolve():
        raise ControlError("squeue 工作目录与 run_id 不匹配")
    return {"job_id": found_id, "owner": owner, "job_name": name, "raw_state": state_value, "workdir": workdir, "node": node}


def status_run(run_id: str, job_id: str, *, runner: Runner | None = None) -> dict[str, Any]:
    run_dir, _binding = _load_binding(run_id, job_id)
    active = _active_job(run_dir, run_id, job_id, runner=runner)
    if active is not None:
        return {
            "run_id": run_id,
            "job_id": job_id,
            "status": _normalize_state(active["raw_state"]),
            "state": _base_state(active["raw_state"]),
            "slurm_state": active["raw_state"],
            "node": active["node"] or None,
            "exit_code": None,
            "source": "squeue",
        }

    result = _run_slurm(
        [
            str(SLURM_BIN / "sacct"),
            "-n",
            "-P",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,User,JobName,State,ExitCode,Start,End,Elapsed,AllocCPUS,NodeList,MaxRSS",
        ],
        runner=runner,
    )
    matches: list[list[str]] = []
    for line in result.stdout.splitlines():
        fields = line.rstrip("\r\n").split("|")
        if fields and fields[0] == job_id:
            matches.append(fields)
    if len(matches) != 1 or len(matches[0]) != 11:
        raise ControlError("sacct 未返回唯一完整的作业记录")
    fields = matches[0]
    if fields[1] != getpass.getuser() or fields[2] != _job_name(run_id):
        raise ControlError("sacct 作业身份与 run_id 不匹配")
    usage_result = _run_slurm(
        [
            str(SLURM_BIN / "sacct"),
            "-n",
            "-P",
            "-j",
            f"{job_id}.batch",
            "--format=JobIDRaw,State,ExitCode,Elapsed,TotalCPU,MaxRSS",
        ],
        runner=runner,
    )
    usage_rows = [
        line.rstrip("\r\n").split("|")
        for line in usage_result.stdout.splitlines()
        if line.strip()
    ]
    if len(usage_rows) != 1 or len(usage_rows[0]) != 6 or usage_rows[0][0] != f"{job_id}.batch":
        raise ControlError("sacct 未返回唯一完整的 batch 用量记录")
    usage = usage_rows[0]
    return {
        "run_id": run_id,
        "job_id": job_id,
        "status": _normalize_state(fields[3]),
        "state": _base_state(fields[3]),
        "slurm_state": fields[3],
        "exit_code": fields[4],
        "started_at": fields[5] or None,
        "finished_at": fields[6] or None,
        "elapsed": fields[7] or None,
        "allocated_cpus": int(fields[8]) if fields[8].isdigit() else None,
        "compute_nodes": fields[9] or None,
        "node": fields[9] or None,
        "total_cpu": usage[4] or None,
        "max_rss": usage[5] or None,
        "source": "sacct",
    }


def _tail_file(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    _require_regular_file(path, "Slurm 日志")
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunks: list[bytes] = []
        collected = 0
        newlines = 0
        while position > 0 and collected < MAX_TAIL_BYTES and newlines <= lines:
            size = min(64 * 1024, position, MAX_TAIL_BYTES - collected)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            chunks.append(chunk)
            collected += len(chunk)
            newlines += chunk.count(b"\n")
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return [line[-16_384:] for line in text.splitlines()[-lines:]]


def tail_run(run_id: str, *, stream: str = "both", lines: int = 200) -> dict[str, Any]:
    if stream not in {"stdout", "stderr", "both"}:
        raise ControlError("stream 只允许 stdout、stderr 或 both")
    if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= MAX_TAIL_LINES:
        raise ControlError(f"lines 必须在 1..{MAX_TAIL_LINES}")
    run_dir = _run_dir(run_id)
    result: dict[str, Any] = {"run_id": run_id}
    if stream in {"stdout", "both"}:
        result["stdout"] = _tail_file(run_dir / "logs" / "slurm.out", lines)
    if stream in {"stderr", "both"}:
        result["stderr"] = _tail_file(run_dir / "logs" / "slurm.err", lines)
    return result


def tail_job(run_id: str, job_id: str, *, lines: int = 200) -> dict[str, Any]:
    _load_binding(run_id, job_id)
    tails = tail_run(run_id, stream="both", lines=lines)
    merged = list(tails.get("stdout", []))
    merged.extend(f"[stderr] {line}" for line in tails.get("stderr", []))
    return {"run_id": run_id, "job_id": job_id, "lines": merged[-lines:]}


def cancel_run(run_id: str, job_id: str, *, runner: Runner | None = None) -> dict[str, Any]:
    run_dir, _binding = _load_binding(run_id, job_id)
    active = _active_job(run_dir, run_id, job_id, runner=runner)
    if active is None:
        terminal = status_run(run_id, job_id, runner=runner)
        raise ControlError(f"作业已不在活动队列，当前状态为 {terminal['status']}")
    _run_slurm([str(SLURM_BIN / "scancel"), job_id], runner=runner)
    return {"run_id": run_id, "job_id": job_id, "cancel_requested": True}


def result_run(run_id: str, job_id: str) -> dict[str, Any]:
    run_dir, _binding = _load_binding(run_id, job_id)
    result = _read_json(run_dir / "output" / "result_manifest.json", "result manifest")
    if result.get("run_id") != run_id or str(result.get("slurm_job_id")) != job_id:
        raise ControlError("result manifest 的 run/job 身份不匹配")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) > 64:
        raise ControlError("result manifest 产物列表非法")
    for row in artifacts:
        if not isinstance(row, dict):
            raise ControlError("result manifest 产物结构非法")
        relative = row.get("path")
        posix = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
        allowed_history = (
            len(posix.parts) == 1
            and bool(re.fullmatch(r"training_history_[A-Za-z0-9._-]+\.json", posix.name))
        )
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or not posix.parts
            or (posix.parts[0] not in {"checkpoints", "strategies"} and not allowed_history)
        ):
            raise ControlError("result manifest 产物路径越界")
        artifact = run_dir.joinpath(*posix.parts)
        size = _require_regular_file(artifact, "结果产物", max_bytes=8 * 1024**3)
        declared_size = row.get("size", row.get("size_bytes"))
        if declared_size != size or validate_sha256(row.get("sha256")) != _sha256_file(artifact):
            raise ControlError("result manifest 产物完整性校验失败")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaMaster 固定 Slurm 控制器")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("run_id")

    finalize = sub.add_parser("finalize-upload")
    finalize.add_argument("run_id")
    finalize.add_argument("filename")
    finalize.add_argument("sha256")
    finalize.add_argument("size_bytes", type=int)

    submit = sub.add_parser("submit")
    submit.add_argument("run_id")

    status = sub.add_parser("status")
    status.add_argument("run_id")
    status.add_argument("job_id")

    tail = sub.add_parser("tail")
    tail.add_argument("run_id")
    tail.add_argument("job_id")
    tail.add_argument("lines", type=int, nargs="?", default=200)

    cancel = sub.add_parser("cancel")
    cancel.add_argument("run_id")
    cancel.add_argument("job_id")

    result = sub.add_parser("result")
    result.add_argument("run_id")
    result.add_argument("job_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_run(args.run_id)
        elif args.command == "finalize-upload":
            result = finalize_upload(
                args.run_id,
                args.filename,
                size_bytes=args.size_bytes,
                sha256=args.sha256,
            )
        elif args.command == "submit":
            result = submit_run(args.run_id)
        elif args.command == "status":
            result = status_run(args.run_id, args.job_id)
        elif args.command == "tail":
            result = tail_job(args.run_id, args.job_id, lines=args.lines)
        elif args.command == "cancel":
            result = cancel_run(args.run_id, args.job_id)
        elif args.command == "result":
            result = result_run(args.run_id, args.job_id)
        else:  # pragma: no cover - argparse 已限制
            raise ControlError("未知命令")
    except ControlError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
