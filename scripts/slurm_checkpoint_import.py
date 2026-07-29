"""在新 Slurm run 提交前导入一个已验证的父 checkpoint。

脚本字节由 Windows 客户端固定 SHA-256 后经 SSH stdin 执行。它不修改父
run，只把同字节 checkpoint 写入尚未提交的新 run，并留下不可变导入回执。
"""

from __future__ import annotations

import hashlib
import importlib.util
import getpass
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


RUN_ID_RE = re.compile(r"^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_CHECKPOINT_BYTES = 8 * 1024**3
MAX_RECEIPT_BYTES = 1024 * 1024
IDENTITY_FIELDS = (
    "symbol",
    "timeframe",
    "dataset_id",
    "data_sha256",
    "local_source",
    "periods_per_year",
    "minimum_bars",
)
MANIFEST_BINDING_FIELDS = (
    *IDENTITY_FIELDS,
    "git_commit",
    "scoring_contract_version",
    "source_files",
)


class ImportError(RuntimeError):
    pass


def _load_control(remote_root: Path):
    path = remote_root / "scripts" / "slurm_control.py"
    spec = importlib.util.spec_from_file_location(
        "alphamaster_checkpoint_import_control",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("无法加载冻结 Slurm 控制器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.ROOT).resolve(strict=True) != remote_root:
        raise ImportError("冻结 Slurm 控制器根目录身份不符")
    return module


def _validate_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ImportError("remote_root 必须是无上跳绝对路径")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ImportError("remote_root 不是目录")
    return resolved


def _validate_checkpoint_path(
    value: str,
    *,
    symbol: str,
    timeframe: str,
    data_sha256: str,
    step: int,
) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.as_posix() != value
    ):
        raise ImportError("checkpoint 路径越界")
    matched = re.fullmatch(
        rf"checkpoints/{re.escape(timeframe)}/{re.escape(data_sha256)}/"
        rf"run_[0-9]{{20}}/ckpt_{re.escape(symbol)}_step_"
        rf"([0-9]{{4,}})\.pt",
        value,
    )
    if matched is None or int(matched.group(1)) != step:
        raise ImportError("checkpoint 路径与训练身份不一致")
    return relative


def _require_parent_node_fail(
    control: Any,
    parent_run_id: str,
    parent_job_id: str,
) -> None:
    result = control._run_slurm(
        [
            str(control.SLURM_BIN / "sacct"),
            "-n",
            "-P",
            "-X",
            "-j",
            parent_job_id,
            (
                "--format=JobIDRaw,User,JobName,State,ExitCode,Start,End,"
                "Elapsed,AllocCPUS,NodeList,MaxRSS"
            ),
        ]
    )
    matches: list[list[str]] = []
    for line in result.stdout.splitlines():
        fields = line.rstrip("\r\n").split("|")
        if fields and fields[0] == parent_job_id:
            matches.append(fields)
    if len(matches) != 1 or len(matches[0]) != 11:
        raise ImportError("sacct 未返回父作业唯一完整终态")
    fields = matches[0]
    if (
        fields[1] != getpass.getuser()
        or fields[2] != control._job_name(parent_run_id)
        or control._base_state(fields[3]) != "NODE_FAIL"
    ):
        raise ImportError("父 Slurm 作业不是明确的 NODE_FAIL 终态")


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ImportError("运行环境不支持安全的 openat 目录固定")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_chain(
    root: Path,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    flags = _directory_flags()
    directory_fd = os.open(root, flags)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _write_all(file_fd: int, chunk: bytes) -> None:
    view = memoryview(chunk)
    while view:
        written = os.write(file_fd, view)
        if written <= 0:
            raise ImportError("checkpoint 写入中断")
        view = view[written:]


def _hash_open_file(
    file_fd: int,
    *,
    expected_size: int,
) -> tuple[str, tuple[int, int, int, int, int]]:
    before = os.fstat(file_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
        raise ImportError("checkpoint 大小或文件类型不匹配")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    total = 0
    os.lseek(file_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise ImportError("checkpoint 读取大小越界")
        digest.update(chunk)
    after = os.fstat(file_fd)
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if total != expected_size or after_identity != identity:
        raise ImportError("checkpoint 在读取期间发生变化")
    return digest.hexdigest(), identity


def _verify_existing(
    directory_fd: int,
    filename: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        file_fd = os.open(filename, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    try:
        digest, _identity = _hash_open_file(
            file_fd,
            expected_size=expected_size,
        )
    finally:
        os.close(file_fd)
    if digest != expected_sha256:
        raise ImportError("目标 checkpoint 已存在但字节不同")
    return True


def _copy_checkpoint(
    source_root: Path,
    target_root: Path,
    relative: PurePosixPath,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    source_dir_fd = -1
    target_dir_fd = -1
    source_fd = -1
    target_fd = -1
    temporary = f".{relative.name}.{os.getpid()}.{os.urandom(8).hex()}.partial"
    try:
        source_dir_fd = _open_directory_chain(
            source_root,
            relative.parts[:-1],
            create=False,
        )
        target_dir_fd = _open_directory_chain(
            target_root,
            relative.parts[:-1],
            create=True,
        )
        if _verify_existing(
            target_dir_fd,
            relative.name,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        ):
            return True
        source_fd = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=source_dir_fd,
        )
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ImportError("父 checkpoint 大小或文件类型不匹配")
        source_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        target_fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=target_dir_fd,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise ImportError("父 checkpoint 读取大小越界")
            digest.update(chunk)
            _write_all(target_fd, chunk)
        after = os.fstat(source_fd)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if total != expected_size or after_identity != source_identity:
            raise ImportError("父 checkpoint 在复制期间发生变化")
        if digest.hexdigest() != expected_sha256:
            raise ImportError("父 checkpoint SHA-256 不匹配")
        os.fsync(target_fd)
        os.close(target_fd)
        target_fd = -1
        try:
            os.link(
                temporary,
                relative.name,
                src_dir_fd=target_dir_fd,
                dst_dir_fd=target_dir_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if not _verify_existing(
                target_dir_fd,
                relative.name,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            ):
                raise
        os.unlink(temporary, dir_fd=target_dir_fd)
        os.fsync(target_dir_fd)
        return False
    except OSError as exc:
        raise ImportError("checkpoint 无法安全导入") from exc
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if source_fd >= 0:
            os.close(source_fd)
        if target_dir_fd >= 0:
            try:
                os.unlink(temporary, dir_fd=target_dir_fd)
            except (FileNotFoundError, OSError):
                pass
        if source_dir_fd >= 0:
            os.close(source_dir_fd)
        if target_dir_fd >= 0:
            os.close(target_dir_fd)


def _load_checkpoint(
    target_root: Path,
    relative: PurePosixPath,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    directory_fd = _open_directory_chain(
        target_root,
        relative.parts[:-1],
        create=False,
    )
    try:
        file_fd = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(file_fd, "rb", closefd=True) as handle:
                import torch

                before = os.fstat(handle.fileno())
                identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_size != expected_size
                ):
                    raise ImportError("导入 checkpoint 大小或文件类型不匹配")
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_size:
                        raise ImportError("导入 checkpoint 读取大小越界")
                    digest.update(chunk)
                if (
                    total != expected_size
                    or digest.hexdigest() != expected_sha256
                ):
                    raise ImportError("导入 checkpoint SHA-256 不匹配")
                handle.seek(0)
                payload = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=True,
                )
                after = os.fstat(handle.fileno())
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if after_identity != identity:
                    raise ImportError("导入 checkpoint 在语义读取期间发生变化")
        except Exception as exc:
            if isinstance(exc, ImportError):
                raise
            raise ImportError("导入 checkpoint 无法安全读取") from exc
    finally:
        os.close(directory_fd)
    if not isinstance(payload, dict):
        raise ImportError("导入 checkpoint 顶层不是对象")
    return payload


def _verify_checkpoint_path(
    target_root: Path,
    relative: PurePosixPath,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    directory_fd = _open_directory_chain(
        target_root,
        relative.parts[:-1],
        create=False,
    )
    try:
        return _verify_existing(
            directory_fd,
            relative.name,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    finally:
        os.close(directory_fd)


def _write_receipt(target_run_dir: Path, receipt: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    input_fd = _open_directory_chain(target_run_dir, ("input",), create=False)
    filename = "checkpoint_recovery.json"
    temporary = f".{filename}.{os.getpid()}.{os.urandom(8).hex()}.partial"
    file_fd = -1
    try:
        try:
            existing_fd = os.open(
                filename,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=input_fd,
            )
        except FileNotFoundError:
            existing_fd = -1
        if existing_fd >= 0:
            try:
                existing = bytearray()
                while True:
                    chunk = os.read(existing_fd, 1024 * 1024)
                    if not chunk:
                        break
                    existing.extend(chunk)
                    if len(existing) > MAX_RECEIPT_BYTES:
                        raise ImportError("checkpoint 恢复回执大小越界")
            finally:
                os.close(existing_fd)
            if bytes(existing) != payload:
                raise ImportError("checkpoint 恢复回执已存在但内容不同")
            return
        file_fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=input_fd,
        )
        _write_all(file_fd, payload)
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = -1
        os.link(
            temporary,
            filename,
            src_dir_fd=input_fd,
            dst_dir_fd=input_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=input_fd)
        os.fsync(input_fd)
    except OSError as exc:
        raise ImportError("checkpoint 恢复回执无法原子写入") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        try:
            os.unlink(temporary, dir_fd=input_fd)
        except (FileNotFoundError, OSError):
            pass
        os.close(input_fd)


def import_checkpoint(
    remote_root_value: str,
    target_run_id: str,
    parent_run_id: str,
    parent_job_id: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_size_value: str,
    checkpoint_step_value: str,
) -> dict[str, Any]:
    remote_root = _validate_root(remote_root_value)
    if (
        RUN_ID_RE.fullmatch(target_run_id) is None
        or RUN_ID_RE.fullmatch(parent_run_id) is None
        or target_run_id == parent_run_id
    ):
        raise ImportError("恢复 run 身份非法")
    if JOB_ID_RE.fullmatch(parent_job_id) is None:
        raise ImportError("父 job ID 非法")
    if SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ImportError("checkpoint SHA-256 非法")
    if (
        not checkpoint_size_value.isascii()
        or not checkpoint_size_value.isdigit()
        or not checkpoint_step_value.isascii()
        or not checkpoint_step_value.isdigit()
    ):
        raise ImportError("checkpoint 大小或步数非法")
    checkpoint_size = int(checkpoint_size_value)
    checkpoint_step = int(checkpoint_step_value)
    if not 0 < checkpoint_size <= MAX_CHECKPOINT_BYTES or checkpoint_step <= 0:
        raise ImportError("checkpoint 大小或步数越界")

    control = _load_control(remote_root)
    parent_run_dir, _binding = control._load_binding(
        parent_run_id,
        parent_job_id,
    )
    _require_parent_node_fail(control, parent_run_id, parent_job_id)
    target_run_dir, target_manifest, target_manifest_sha256 = (
        control._load_finalized_manifest(target_run_id)
    )
    loaded_parent_dir, parent_manifest, parent_manifest_sha256 = (
        control._load_finalized_manifest(parent_run_id)
    )
    if loaded_parent_dir.resolve(strict=True) != parent_run_dir.resolve(strict=True):
        raise ImportError("父 run 目录身份不一致")
    if (
        (target_run_dir / ".slurm_job.json").exists()
        or (target_run_dir / ".submit.lock").exists()
    ):
        raise ImportError("目标 run 已进入提交阶段，拒绝导入 checkpoint")
    for field in MANIFEST_BINDING_FIELDS:
        if target_manifest.get(field) != parent_manifest.get(field):
            raise ImportError(f"父子 run 的 {field} 不一致")
    target_training = target_manifest.get("training_parameters")
    parent_training = parent_manifest.get("training_parameters")
    if (
        not isinstance(target_training, dict)
        or not isinstance(parent_training, dict)
        or target_training.get("from_scratch") is not False
        or parent_training.get("from_scratch") is not True
        or target_training.get("train_steps") != parent_training.get("train_steps")
        or checkpoint_step >= int(target_training.get("train_steps") or 0)
    ):
        raise ImportError("父子 run 的训练参数不符合续训合同")
    recovery = {
        "parent_run_id": parent_run_id,
        "parent_job_id": parent_job_id,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "checkpoint_step": checkpoint_step,
    }
    if target_manifest.get("checkpoint_recovery") != recovery:
        raise ImportError("目标 run manifest 的 checkpoint 恢复合同不匹配")
    relative = _validate_checkpoint_path(
        checkpoint_path,
        symbol=str(target_manifest["symbol"]),
        timeframe=str(target_manifest["timeframe"]),
        data_sha256=str(target_manifest["data_sha256"]),
        step=checkpoint_step,
    )
    idempotent = _copy_checkpoint(
        parent_run_dir,
        target_run_dir,
        relative,
        expected_size=checkpoint_size,
        expected_sha256=checkpoint_sha256,
    )
    checkpoint = _load_checkpoint(
        target_run_dir,
        relative,
        expected_size=checkpoint_size,
        expected_sha256=checkpoint_sha256,
    )
    for field in IDENTITY_FIELDS:
        if (
            field not in checkpoint
            or type(checkpoint[field]) is not type(target_manifest[field])
            or checkpoint[field] != target_manifest[field]
        ):
            raise ImportError(f"checkpoint 的 {field} 与目标 run 不一致")
    if checkpoint.get("step") != checkpoint_step:
        raise ImportError("checkpoint 内部 step 不一致")
    if (
        checkpoint.get("scoring_contract_version")
        != target_manifest["scoring_contract_version"]
        or not isinstance(checkpoint.get("vocab_version"), str)
        or not checkpoint["vocab_version"]
    ):
        raise ImportError("checkpoint 的评分合同或词表版本不一致")
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "best_snapshot",
        "training_history",
    ):
        if field not in checkpoint:
            raise ImportError(f"checkpoint 缺少续训状态: {field}")
    receipt = {
        "schema_version": 1,
        "target_run_id": target_run_id,
        "target_manifest_sha256": target_manifest_sha256,
        "parent_run_id": parent_run_id,
        "parent_job_id": parent_job_id,
        "parent_manifest_sha256": parent_manifest_sha256,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "checkpoint_step": checkpoint_step,
    }
    _write_receipt(target_run_dir, receipt)
    if not _verify_checkpoint_path(
        target_run_dir,
        relative,
        expected_size=checkpoint_size,
        expected_sha256=checkpoint_sha256,
    ):
        raise ImportError("导入 checkpoint 在回执落盘后消失")
    return {**receipt, "idempotent": idempotent, "imported": True}


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 8:
        print(
            json.dumps(
                {"ok": False, "error": "checkpoint import 参数数量错误"},
                ensure_ascii=False,
            )
        )
        return 2
    try:
        result = import_checkpoint(*values)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
