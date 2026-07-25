"""在 Slurm 计算节点执行单个 AlphaMaster 训练 run。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_PYTHON = Path("/hwdata/home/jinqc/.local/bin/python3.11")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.dataset_contracts import (
    FREE_STOCKDB_QFQ_FORMAT,
    FREE_STOCKDB_QFQ_SOURCE_ID,
    FREE_STOCKDB_SOURCE,
    TRAINING_SOURCE_IDS,
    source_family,
)
from model_core.target_contract import SCORING_CONTRACT_VERSION

RUN_ID_RE = re.compile(r"^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
FILENAME_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9][A-Za-z0-9._-]{0,95})_"
    r"(?P<timeframe>M1|M5|M15|M30|H1|H4|D1|W1|MN1)\.parquet$",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"^(?:(?P<days>[0-7])-)?(?P<hours>[0-9]{1,2}):[0-5][0-9]:[0-5][0-9]$")
MEMORY_RE = re.compile(r"^[1-9][0-9]{0,5}[MG]$")
ALLOWED_LOCAL_SOURCES = TRAINING_SOURCE_IDS
REQUIRED_DATA_COLUMNS = {"time", "open", "high", "low", "close", "tick_volume"}
ASHARE_PERIOD_CONTRACTS = {
    "M5": (11_616, 23_232),
    "M15": (3_872, 7_744),
    "H1": (968, 1_936),
    "D1": (242, 484),
}
GENERIC_MINIMUM_BARS = 3_000  # 必须与根 config.Config.MIN_BARS 保持一致

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_DATA_BYTES = 64 * 1024**3
DATA_SOURCE_MANIFEST_FILENAME = "data_source_manifest.json"
MAX_ARTIFACT_BYTES = 8 * 1024**3
MAX_ARTIFACT_TOTAL_BYTES = 32 * 1024**3
REQUIRED_SOURCE_FILES = (
    "train_file.py",
    "data_pipeline/dataset_contracts.py",
    "model_core/config.py",
    "utils/training_runtime.py",
    "scripts/train_slurm_worker.py",
    "scripts/train_alphamaster.sbatch",
)
ARTIFACT_WHITELIST = (
    "checkpoints/{timeframe}/{data_sha256}/run_*/ckpt_{symbol}_step_*.pt",
    "strategies/best_{symbol}.json",
    "training_history_{symbol}.json",
)

Runner = Callable[..., subprocess.CompletedProcess[Any]]


class WorkerError(RuntimeError):
    """训练输入或运行环境违反固定合同。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(value: str) -> str:
    if not isinstance(value, str) or not RUN_ID_RE.fullmatch(value):
        raise WorkerError("run_id 非法")
    return value


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise WorkerError(f"{label} 不是合法 SHA-256")
    return value.lower()


def _require_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise WorkerError(f"{label}不存在") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise WorkerError(f"{label}必须是真实目录")


def _require_file(path: Path, label: str, max_bytes: int | None = None) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise WorkerError(f"{label}不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WorkerError(f"{label}必须是普通文件")
    if max_bytes is not None and info.st_size > max_bytes:
        raise WorkerError(f"{label}超过大小上限")
    return info.st_size


def _require_worker_python(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise WorkerError("Worker Python 不存在") from exc
    if stat.S_ISREG(info.st_mode):
        return
    if not stat.S_ISLNK(info.st_mode):
        raise WorkerError("Worker Python 必须是普通文件或固定虚拟环境符号链接")
    try:
        target = path.resolve(strict=True)
    except OSError as exc:
        raise WorkerError("Worker Python 符号链接失效") from exc
    if target != BASE_PYTHON.resolve(strict=True) or not target.is_file():
        raise WorkerError("Worker Python 符号链接未指向固定 Python 3.11")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_source(path: Path) -> bytes:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkerError(f"源码不是合法 UTF-8 文本: {path.name}") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label, MAX_MANIFEST_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"{label}不是合法 UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{label}顶层必须是对象")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
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


def _materialize_data_source_manifest(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise WorkerError("来源 manifest 在物化前发生漂移")
    if destination.exists():
        _require_file(destination, "训练数据 sidecar", MAX_MANIFEST_BYTES)
        if _sha256_file(destination) != expected_sha256:
            raise WorkerError("训练数据 sidecar 已存在但身份不匹配")
        return
    temp = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, destination)
        os.chmod(destination, 0o600)
    finally:
        if temp.exists():
            temp.unlink()


def _read_git_commit() -> str:
    git_dir = ROOT / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise WorkerError(".git 指针格式异常")
        candidate = Path(text[8:])
        git_dir = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
    _require_directory(git_dir, ".git 目录")
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if GIT_COMMIT_RE.fullmatch(head):
        return head.lower()
    if not head.startswith("ref: "):
        raise WorkerError("Git HEAD 格式异常")
    ref = head[5:]
    if not re.fullmatch(r"refs/[A-Za-z0-9._/-]+", ref) or ".." in ref:
        raise WorkerError("Git ref 非法")
    ref_path = git_dir / ref
    if ref_path.is_file():
        commit = ref_path.read_text(encoding="ascii").strip()
    else:
        commit = ""
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line and not line.startswith(("#", "^")):
                    value, name = line.split(" ", 1)
                    if name == ref:
                        commit = value
                        break
    if not GIT_COMMIT_RE.fullmatch(commit):
        raise WorkerError("无法解析当前 Git 提交")
    return commit.lower()


def _verify_inputs(run_dir: Path) -> tuple[dict[str, Any], Path, str]:
    manifest_path = run_dir / "input" / "run_manifest.json"
    manifest = _read_json(manifest_path, "run manifest")
    run_id = run_dir.name
    if manifest.get("run_id") != run_id:
        raise WorkerError("manifest run_id 不匹配")
    if manifest.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise WorkerError("manifest scoring_contract_version 不匹配")

    filename = manifest.get("data_filename")
    if not isinstance(filename, str):
        raise WorkerError("manifest 缺少 data_filename")
    name_match = FILENAME_RE.fullmatch(filename)
    if name_match is None or Path(filename).name != filename:
        raise WorkerError("data_filename 非法")
    if manifest.get("symbol") != name_match.group("symbol"):
        raise WorkerError("manifest symbol 与文件名不匹配")
    if str(manifest.get("timeframe", "")).upper() != name_match.group("timeframe").upper():
        raise WorkerError("manifest timeframe 与文件名不匹配")
    local_source = manifest.get("local_source")
    if local_source not in ALLOWED_LOCAL_SOURCES:
        raise WorkerError("manifest local_source 不受支持")
    periods_per_year = manifest.get("periods_per_year")
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise WorkerError("manifest periods_per_year 非法")
    timestamps: list[datetime] = []
    for field in ("data_start", "data_end"):
        value = manifest.get(field)
        if not isinstance(value, str):
            raise WorkerError(f"manifest {field} 必须是 UTC 时间")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise WorkerError(f"manifest {field} 必须是 UTC 时间") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise WorkerError(f"manifest {field} 必须是 UTC 时间")
        timestamps.append(parsed)
    if timestamps[0] >= timestamps[1]:
        raise WorkerError("manifest data_start 必须早于 data_end")
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(name, str) or not name for name in columns)
        or len(columns) != len(set(columns))
        or not REQUIRED_DATA_COLUMNS.issubset(columns)
    ):
        raise WorkerError("manifest columns 缺少训练列或包含重复项")
    timeframe = name_match.group("timeframe").upper()
    if source_family(str(local_source)) == "ashare":
        contract = ASHARE_PERIOD_CONTRACTS.get(timeframe)
        if contract is None:
            raise WorkerError("A 股数据 timeframe 不受支持")
        expected_periods, expected_minimum = contract
        if periods_per_year != expected_periods:
            raise WorkerError("A 股数据 periods_per_year 不匹配")
        if manifest.get("minimum_bars") != expected_minimum:
            raise WorkerError("A 股数据 minimum_bars 不匹配")
        data_rows = manifest.get("data_rows")
        if isinstance(data_rows, bool) or not isinstance(data_rows, int) or data_rows < expected_minimum:
            raise WorkerError("A 股数据不足两个交易年")
    else:
        if manifest.get("minimum_bars") != GENERIC_MINIMUM_BARS:
            raise WorkerError(f"MT5/OKX minimum_bars 必须是 {GENERIC_MINIMUM_BARS}")
        data_rows = manifest.get("data_rows")
        if (
            isinstance(data_rows, bool)
            or not isinstance(data_rows, int)
            or data_rows < GENERIC_MINIMUM_BARS
        ):
            raise WorkerError(f"MT5/OKX 数据不足 {GENERIC_MINIMUM_BARS} bars")

    hash_path = run_dir / "input" / "run_manifest.sha256"
    _require_file(hash_path, "manifest 哈希", 128)
    expected_manifest_hash = _validate_sha256(hash_path.read_text(encoding="ascii").strip(), "manifest 哈希")
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise WorkerError("run manifest 哈希不匹配")

    data_path = run_dir / "input" / filename
    data_size = _require_file(data_path, "训练数据", MAX_DATA_BYTES)
    expected_size = manifest.get("data_size", manifest.get("data_size_bytes"))
    if expected_size is not None and expected_size != data_size:
        raise WorkerError("训练数据大小不匹配")
    expected_data_hash = _validate_sha256(manifest.get("data_sha256"), "数据哈希")
    if _sha256_file(data_path) != expected_data_hash:
        raise WorkerError("训练数据 SHA-256 不匹配")
    source_manifest_fields = (
        manifest.get("data_source_manifest_filename"),
        manifest.get("data_source_manifest_sha256"),
        manifest.get("data_source_manifest_size"),
        manifest.get("data_source_manifest"),
    )
    if local_source == FREE_STOCKDB_QFQ_SOURCE_ID:
        if (
            manifest.get("data_source_manifest_filename")
            != DATA_SOURCE_MANIFEST_FILENAME
        ):
            raise WorkerError("free-stockdb 来源 manifest 文件名不匹配")
        expected_source_hash = _validate_sha256(
            manifest.get("data_source_manifest_sha256"),
            "来源 manifest 哈希",
        )
        expected_source_size = manifest.get("data_source_manifest_size")
        if (
            isinstance(expected_source_size, bool)
            or not isinstance(expected_source_size, int)
            or not 0 < expected_source_size <= MAX_MANIFEST_BYTES
        ):
            raise WorkerError("来源 manifest 大小非法")
        source_manifest = manifest.get("data_source_manifest")
        if not isinstance(source_manifest, dict):
            raise WorkerError("free-stockdb 来源 manifest 身份缺失")
        expected_source_fields = {
            "source": FREE_STOCKDB_SOURCE,
            "format": FREE_STOCKDB_QFQ_FORMAT,
            "source_id": FREE_STOCKDB_QFQ_SOURCE_ID,
            "symbol": manifest.get("symbol"),
            "timeframe": manifest.get("timeframe"),
            "data_filename": filename,
            "data_sha256": manifest.get("data_sha256"),
            "dataset_id": manifest.get("dataset_id"),
            "data_rows": manifest.get("data_rows"),
            "data_start": manifest.get("data_start"),
            "data_end": manifest.get("data_end"),
            "columns": manifest.get("columns"),
            "periods_per_year": manifest.get("periods_per_year"),
            "minimum_bars": manifest.get("minimum_bars"),
        }
        for field, expected in expected_source_fields.items():
            if source_manifest.get(field) != expected:
                raise WorkerError(
                    f"free-stockdb 来源 manifest 的 {field} 与 run manifest 不一致"
                )
        source_manifest_path = (
            run_dir / "input" / DATA_SOURCE_MANIFEST_FILENAME
        )
        actual_source_size = _require_file(
            source_manifest_path,
            "来源 manifest",
            MAX_MANIFEST_BYTES,
        )
        if actual_source_size != expected_source_size:
            raise WorkerError("来源 manifest 大小不匹配")
        if _sha256_file(source_manifest_path) != expected_source_hash:
            raise WorkerError("来源 manifest SHA-256 不匹配")
        if _read_json(source_manifest_path, "来源 manifest") != source_manifest:
            raise WorkerError("来源 manifest 与 run manifest 冻结内容不一致")
        _materialize_data_source_manifest(
            source_manifest_path,
            data_path.with_suffix(".manifest.json"),
            expected_sha256=expected_source_hash,
        )
    elif any(value is not None for value in source_manifest_fields):
        raise WorkerError("非 free-stockdb run 不接受来源 manifest 输入")

    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or not GIT_COMMIT_RE.fullmatch(commit):
        raise WorkerError("manifest git_commit 非法")
    if _read_git_commit() != commit.lower():
        raise WorkerError("当前源码提交与 manifest 不匹配")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not 1 <= len(source_files) <= 512:
        raise WorkerError("source_files 列表非法")
    seen: set[str] = set()
    for row in source_files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise WorkerError("source_files 项非法")
        relative = row["path"]
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or "\\" in relative or posix.suffix not in {".py", ".sbatch"} or relative in seen:
            raise WorkerError("source_files 路径非法或重复")
        seen.add(relative)
        source = ROOT.joinpath(*posix.parts)
        _require_file(source, f"源码 {relative}", 16 * 1024 * 1024)
        canonical = _canonical_source(source)
        if row.get("size") != len(canonical):
            raise WorkerError(f"源码大小不匹配: {relative}")
        expected = _validate_sha256(row.get("sha256"), f"源码哈希 {relative}")
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise WorkerError(f"源码哈希不匹配: {relative}")
    if not set(REQUIRED_SOURCE_FILES).issubset(seen):
        raise WorkerError("source_files 缺少固定关键源码")

    resources = manifest.get("requested_resources")
    if not isinstance(resources, dict):
        raise WorkerError("requested_resources 非法")
    if resources.get("partition") != "cpu" or resources.get("qos") != "normal":
        raise WorkerError("Worker 只允许 cpu/normal")
    cpus = resources.get("cpus_per_task")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or not 1 <= cpus <= 64:
        raise WorkerError("cpus_per_task 非法")
    time_limit = resources.get("time_limit")
    time_match = TIME_RE.fullmatch(time_limit) if isinstance(time_limit, str) else None
    if time_match is None or int(time_match.group("hours")) > 23:
        raise WorkerError("time_limit 非法")
    memory = resources.get("memory")
    if not isinstance(memory, str) or (memory and MEMORY_RE.fullmatch(memory) is None):
        raise WorkerError("memory 非法")

    training = manifest.get("training_parameters")
    if not isinstance(training, dict):
        raise WorkerError("training_parameters 非法")
    steps = training.get("train_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1_000_000:
        raise WorkerError("train_steps 非法")
    if not isinstance(training.get("from_scratch", False), bool):
        raise WorkerError("from_scratch 非法")
    return manifest, data_path, actual_manifest_hash


def _environment_versions(python_executable: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "numpy", "pandas", "scipy", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_executable": str(python_executable),
        "platform": platform.platform(),
        "packages": packages,
    }


def _clean_training_env(run_dir: Path, cpus: int, source: Mapping[str, str]) -> dict[str, str]:
    env = {
        "HOME": str(run_dir),
        "PATH": f"{ROOT / '.venv' / 'bin'}:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": str(cpus),
        "MKL_NUM_THREADS": str(cpus),
        "OPENBLAS_NUM_THREADS": str(cpus),
        "NUMEXPR_NUM_THREADS": str(cpus),
        "ALPHAMASTER_TRAINING_RUNTIME": "slurm_worker_v1",
    }
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURMD_NODENAME", "SLURM_JOB_NODELIST"):
        value = source.get(key)
        if value and len(value) <= 512 and "\x00" not in value:
            env[key] = value
    return env


def _collect_artifacts(
    run_dir: Path,
    symbol: str,
    timeframe: str,
    data_sha256: str,
) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for template in ARTIFACT_WHITELIST:
        paths.update(
            run_dir.glob(
                template.format(
                    symbol=symbol,
                    timeframe=timeframe,
                    data_sha256=data_sha256,
                )
            )
        )
    artifacts: list[dict[str, Any]] = []
    total = 0
    for path in sorted(paths):
        size = _require_file(path, f"训练产物 {path.name}", MAX_ARTIFACT_BYTES)
        total += size
        if total > MAX_ARTIFACT_TOTAL_BYTES:
            raise WorkerError("训练产物总大小超过 32 GiB")
        relative = path.relative_to(run_dir).as_posix()
        if relative.startswith("checkpoints/") and re.fullmatch(
            rf"checkpoints/{re.escape(timeframe)}/{re.escape(data_sha256)}/"
            rf"run_[0-9]{{20}}/ckpt_{re.escape(symbol)}_step_[0-9]{{4,}}\.pt",
            relative,
        ) is None:
            raise WorkerError("checkpoint 路径不符合数据身份隔离合同")
        artifacts.append({"path": relative, "size": size, "size_bytes": size, "sha256": _sha256_file(path)})
    return artifacts


def run_worker(
    run_id: str,
    *,
    runner: Runner | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    python_executable: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    run_id = _validate_run_id(run_id)
    runs_root = ROOT / "runs"
    _require_directory(runs_root, "runs 根目录")
    run_dir = runs_root / run_id
    _require_directory(run_dir, "run 目录")
    if run_dir.resolve().parent != runs_root.resolve():
        raise WorkerError("run 目录越界")
    output_dir = run_dir / "output"
    _require_directory(output_dir, "output 目录")
    source_env = dict(environ or os.environ)
    job_id = source_env.get("SLURM_JOB_ID", "")
    started_at = _utc_now()
    node = source_env.get("SLURMD_NODENAME") or socket.gethostname()
    # 保留虚拟环境入口的词法路径；resolve() 会越过 pyvenv.cfg，导致子进程退回基础解释器。
    executable = Path(python_executable or sys.executable).absolute()
    expected_python = (ROOT / ".venv" / "bin" / "python").absolute()
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    artifacts: list[dict[str, Any]] = []
    exit_code = 1
    error_message: str | None = None
    try:
        for name in ("input", "logs", "checkpoints", "strategies"):
            _require_directory(run_dir / name, f"{name} 目录")
        current_dir = (cwd or Path.cwd()).resolve()
        if current_dir != run_dir.resolve():
            raise WorkerError("Worker 必须以唯一 run 目录作为工作目录")
        if not JOB_ID_RE.fullmatch(job_id):
            raise WorkerError("Worker 必须由 Slurm 作业启动")
        if executable != expected_python:
            raise WorkerError("Worker 必须使用项目 .venv 的绝对 Python")
        _require_worker_python(executable)
        manifest, data_path, manifest_hash = _verify_inputs(run_dir)
        resources = manifest["requested_resources"]
        cpus = resources["cpus_per_task"]
        actual_cpus = source_env.get("SLURM_CPUS_PER_TASK")
        if actual_cpus is None or not actual_cpus.isdigit() or int(actual_cpus) != cpus:
            raise WorkerError("实际 SLURM_CPUS_PER_TASK 与 manifest 不匹配")
        training = manifest["training_parameters"]
        command = [
            str(executable),
            str(ROOT / "train_file.py"),
            "--data-file",
            str(data_path),
            "--train-steps",
            str(training["train_steps"]),
            "--periods-per-year",
            str(manifest["periods_per_year"]),
            "--data-source",
            str(manifest["local_source"]),
        ]
        if manifest.get("minimum_bars") is not None:
            command.extend(["--minimum-bars", str(manifest["minimum_bars"])])
        if training.get("from_scratch", False):
            command.append("--from-scratch")
        execute = runner or subprocess.run
        completed = execute(
            command,
            shell=False,
            check=False,
            cwd=run_dir,
            env=_clean_training_env(run_dir, cpus, source_env),
        )
        exit_code = int(completed.returncode)
        if exit_code != 0:
            raise WorkerError(f"train_file.py 退出码为 {exit_code}")
        artifacts = _collect_artifacts(
            run_dir,
            str(manifest["symbol"]),
            str(manifest["timeframe"]),
            str(manifest["data_sha256"]),
        )
        required_artifact_kinds = {
            "checkpoint": any(item["path"].startswith("checkpoints/") for item in artifacts),
            "strategy": any(item["path"].startswith("strategies/") for item in artifacts),
            "history": any(item["path"].startswith("training_history_") for item in artifacts),
        }
        missing = [name for name, present in required_artifact_kinds.items() if not present]
        if missing:
            raise WorkerError(f"训练进程成功退出但缺少正式产物: {', '.join(missing)}")
    except Exception as exc:  # 失败也必须形成真实 manifest
        if isinstance(exc, KeyboardInterrupt):
            raise
        error_message = str(exc)[:2000]
        if manifest is not None:
            try:
                artifacts = _collect_artifacts(
                    run_dir,
                    str(manifest.get("symbol", "")),
                    str(manifest.get("timeframe", "")),
                    str(manifest.get("data_sha256", "")),
                )
            except Exception as artifact_exc:
                error_message = f"{error_message}; 产物审计失败: {artifact_exc}"[:2000]
        if exit_code == 0:
            exit_code = 1

    finished_at = _utc_now()
    status_value = "COMPLETED" if exit_code == 0 and error_message is None else "FAILED"
    checkpoint_files = [item["path"] for item in artifacts if item["path"].startswith("checkpoints/")]
    strategy_files = [item["path"] for item in artifacts if item["path"].startswith("strategies/")]
    result = {
        "run_id": run_id,
        "slurm_job_id": job_id,
        "status": status_value,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "git_commit": manifest.get("git_commit") if manifest else None,
        "scoring_contract_version": (
            manifest.get("scoring_contract_version") if manifest else None
        ),
        "run_manifest_sha256": manifest_hash,
        "source_files": manifest.get("source_files") if manifest else None,
        "symbol": manifest.get("symbol") if manifest else None,
        "timeframe": manifest.get("timeframe") if manifest else None,
        "data_filename": manifest.get("data_filename") if manifest else None,
        "data_sha256": manifest.get("data_sha256") if manifest else None,
        "data_size": manifest.get("data_size") if manifest else None,
        "data_rows": manifest.get("data_rows") if manifest else None,
        "data_start": manifest.get("data_start") if manifest else None,
        "data_end": manifest.get("data_end") if manifest else None,
        "columns": manifest.get("columns") if manifest else None,
        "dataset_id": manifest.get("dataset_id") if manifest else None,
        "local_source": manifest.get("local_source") if manifest else None,
        "periods_per_year": manifest.get("periods_per_year") if manifest else None,
        "minimum_bars": manifest.get("minimum_bars") if manifest else None,
        "data_source_manifest_filename": (
            manifest.get("data_source_manifest_filename") if manifest else None
        ),
        "data_source_manifest_sha256": (
            manifest.get("data_source_manifest_sha256") if manifest else None
        ),
        "data_source_manifest_size": (
            manifest.get("data_source_manifest_size") if manifest else None
        ),
        "data_source_manifest": (
            manifest.get("data_source_manifest") if manifest else None
        ),
        "requested_resources": manifest.get("requested_resources") if manifest else None,
        "training_parameters": manifest.get("training_parameters") if manifest else None,
        "compute_node": node,
        "environment_versions": _environment_versions(executable),
        "artifact_whitelist": [
            pattern.format(
                symbol=str(manifest.get("symbol", "")),
                timeframe=str(manifest.get("timeframe", "")),
                data_sha256=str(manifest.get("data_sha256", "")),
            )
            for pattern in ARTIFACT_WHITELIST
        ] if manifest else [],
        "artifacts": artifacts,
        "checkpoint_files": checkpoint_files,
        "strategy_files": strategy_files,
        "artifact_sha256": {item["path"]: item["sha256"] for item in artifacts},
        "error_message": error_message,
    }
    _atomic_write_json(output_dir / "result_manifest.json", result)
    return exit_code, result


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("用法: train_slurm_worker.py RUN_ID", file=sys.stderr)
        return 2
    try:
        code, _result = run_worker(args[0])
        return code
    except WorkerError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
