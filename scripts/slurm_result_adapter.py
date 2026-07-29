"""兼容旧式多 checkpoint 结果清单，只发布最高训练步 checkpoint。

该适配器位于训练源码冻结集合之外。它不修改远端 result manifest 或训练产物，
只在完整校验旧清单后返回一个最多 64 项的内存视图。
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


MAX_SOURCE_ARTIFACTS = 1024
MAX_PUBLISHED_ARTIFACTS = 64
RUN_ID_RE = re.compile(r"^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRAINING_HISTORY_RE = re.compile(
    r"^training_history_[A-Za-z0-9._-]+\.json$"
)


class AdapterError(RuntimeError):
    pass


def _read_pinned_file(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    max_bytes: int,
    retain_bytes: bool = False,
) -> tuple[int, str, bytes | None]:
    """用 openat 固定整条目录链，并从同一个文件句柄读取和哈希。"""
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise AdapterError(f"{label} 路径越界")
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise AdapterError("运行环境不支持安全的目录链固定")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open(root, directory_flags)
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(relative.name, file_flags, dir_fd=directory_fd)
        with os.fdopen(file_fd, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AdapterError(f"{label} 不是普通文件")
            if before.st_size < 0 or before.st_size > max_bytes:
                raise AdapterError(f"{label} 大小越界")
            digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if retain_bytes else None
            total = 0
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise AdapterError(f"{label} 读取大小越界")
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if total != before.st_size or identity_after != identity_before:
                raise AdapterError(f"{label} 在读取期间发生变化")
            payload = b"".join(chunks) if chunks is not None else None
            return total, digest.hexdigest(), payload
    except OSError as exc:
        raise AdapterError(f"{label} 无法安全读取") from exc
    finally:
        os.close(directory_fd)


def _load_control(remote_root: Path):
    path = remote_root / "scripts" / "slurm_control.py"
    spec = importlib.util.spec_from_file_location(
        "alphamaster_frozen_slurm_control", path
    )
    if spec is None or spec.loader is None:
        raise AdapterError("无法加载冻结 Slurm 控制器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.ROOT).resolve(strict=True) != remote_root:
        raise AdapterError("冻结 Slurm 控制器根目录身份不符")
    return module


def _validate_remote_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise AdapterError("remote_root 必须是无上跳绝对路径")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise AdapterError("remote_root 不是目录")
    return resolved


def adapt(remote_root_value: str, run_id: str, job_id: str) -> dict[str, Any]:
    remote_root = _validate_remote_root(remote_root_value)
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise AdapterError("run_id 非法")
    if JOB_ID_RE.fullmatch(job_id) is None:
        raise AdapterError("job_id 非法")
    control = _load_control(remote_root)
    run_dir, _binding = control._load_binding(run_id, job_id)
    (
        _manifest_size,
        source_result_manifest_sha256,
        manifest_bytes,
    ) = _read_pinned_file(
        run_dir,
        PurePosixPath("output/result_manifest.json"),
        label="result manifest",
        max_bytes=8 * 1024 * 1024,
        retain_bytes=True,
    )
    try:
        result = json.loads((manifest_bytes or b"").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError("result manifest 不是合法 UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise AdapterError("result manifest 顶层必须是对象")
    if (
        result.get("run_id") != run_id
        or str(result.get("slurm_job_id")) != job_id
    ):
        raise AdapterError("result manifest 的 run/job 身份不匹配")

    artifacts = result.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or len(artifacts) > MAX_SOURCE_ARTIFACTS
    ):
        raise AdapterError("result manifest 原始产物列表非法")

    symbol = str(result.get("symbol") or "")
    timeframe = str(result.get("timeframe") or "")
    data_sha256 = str(result.get("data_sha256") or "")
    if (
        not symbol
        or not timeframe
        or SHA256_RE.fullmatch(data_sha256) is None
    ):
        raise AdapterError("result manifest 训练身份不完整")
    checkpoint_pattern = re.compile(
        rf"checkpoints/{re.escape(timeframe)}/{re.escape(data_sha256)}/"
        rf"(?P<run>run_[0-9]{{20}})/"
        rf"ckpt_{re.escape(symbol)}_step_(?P<step>[0-9]{{4,}})\.pt"
    )

    checkpoint_rows: list[tuple[int, str, dict[str, Any]]] = []
    non_checkpoint_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    full_hashes: dict[str, str] = {}
    for row in artifacts:
        if not isinstance(row, dict):
            raise AdapterError("result manifest 产物结构非法")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in seen_paths:
            raise AdapterError("result manifest 产物路径非法或重复")
        seen_paths.add(relative)
        posix = PurePosixPath(relative)
        allowed_history = (
            len(posix.parts) == 1
            and TRAINING_HISTORY_RE.fullmatch(posix.name) is not None
        )
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or not posix.parts
            or (
                posix.parts[0] not in {"checkpoints", "strategies"}
                and not allowed_history
            )
        ):
            raise AdapterError("result manifest 产物路径越界")
        size, actual_digest, _payload = _read_pinned_file(
            run_dir,
            posix,
            label="结果产物",
            max_bytes=8 * 1024**3,
        )
        declared_size = row.get("size", row.get("size_bytes"))
        digest = row.get("sha256")
        if (
            declared_size != size
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or digest != actual_digest
        ):
            raise AdapterError("result manifest 产物完整性校验失败")
        full_hashes[relative] = digest
        if relative.startswith("checkpoints/"):
            match = checkpoint_pattern.fullmatch(relative)
            if match is None:
                raise AdapterError("result manifest 含非法 checkpoint 路径")
            checkpoint_rows.append(
                (int(match.group("step")), match.group("run"), row)
            )
        else:
            non_checkpoint_rows.append(row)

    if not checkpoint_rows:
        raise AdapterError("result manifest 缺少 checkpoint")
    if len({item[1] for item in checkpoint_rows}) != 1:
        raise AdapterError("result manifest 含多个 checkpoint 运行目录")
    highest_step = max(item[0] for item in checkpoint_rows)
    latest = [
        row for step, _run, row in checkpoint_rows if step == highest_step
    ]
    if len(latest) != 1:
        raise AdapterError("最高训练步 checkpoint 不唯一")

    full_checkpoint_paths = sorted(
        str(row["path"]) for _step, _run, row in checkpoint_rows
    )
    declared_checkpoints = result.get("checkpoint_files")
    if (
        not isinstance(declared_checkpoints, list)
        or not all(isinstance(path, str) for path in declared_checkpoints)
        or sorted(declared_checkpoints) != full_checkpoint_paths
    ):
        raise AdapterError("result manifest 的 checkpoint_files 不一致")
    declared_hashes = result.get("artifact_sha256")
    if not isinstance(declared_hashes, dict) or declared_hashes != full_hashes:
        raise AdapterError("result manifest 的 artifact_sha256 不一致")

    published = sorted(
        [*non_checkpoint_rows, latest[0]],
        key=lambda row: str(row["path"]),
    )
    if len(published) > MAX_PUBLISHED_ARTIFACTS:
        raise AdapterError("归一化后的产物列表仍然超限")
    strategies = [
        str(row["path"])
        for row in published
        if str(row["path"]).startswith("strategies/")
    ]
    histories = [
        str(row["path"])
        for row in published
        if TRAINING_HISTORY_RE.fullmatch(str(row["path"])) is not None
    ]
    if len(strategies) != 1 or len(histories) != 1:
        raise AdapterError("归一化结果必须各含一个策略和训练历史")

    normalized = dict(result)
    normalized["artifacts"] = published
    normalized["checkpoint_files"] = [str(latest[0]["path"])]
    normalized["strategy_files"] = strategies
    normalized["artifact_sha256"] = {
        str(row["path"]): str(row["sha256"]) for row in published
    }
    normalized["artifact_selection"] = {
        "policy": "highest_step_single_run_v1",
        "source_result_manifest_sha256": source_result_manifest_sha256,
        "source_artifact_count": len(artifacts),
        "published_artifact_count": len(published),
        "omitted_checkpoint_count": len(checkpoint_rows) - 1,
        "published_checkpoint_step": highest_step,
    }
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            json.dumps(
                {"ok": False, "error": "用法: REMOTE_ROOT RUN_ID JOB_ID"},
                ensure_ascii=False,
            )
        )
        return 2
    try:
        payload = adapt(args[0], args[1], args[2])
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)[:2000]},
                ensure_ascii=False,
            )
        )
        return 2
    print(
        json.dumps(
            {"ok": True, **payload},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
