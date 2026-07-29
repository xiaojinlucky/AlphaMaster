"""AlphaMaster 模型资产异地备份（WO-AM-08）。

把本机 checkpoints / local_runs / published_training 增量同步到 HPC
`/hwdata/home/jinqc/AlphaMaster-backup/`。设计约束：
- 不做镜像删除：只发布本轮清单中的文件；本轮远端临时目录会受控清理，
  本地项目数据绝不写入。
- 增量判定：远端已存在同路径且同字节数、且体积 > 10 MiB 的文件跳过
  （checkpoint 内容寻址、只增不改）；小文件（清单/SQLite/JSON）每次都重传，
  保证账本类可变文件始终最新。
- 传输：先解到远端唯一临时目录，完整核对后构造一个完整只读版本；
  版本闭合后原子替换 `CURRENT` 指针，上一完整版本始终保留。
- 失败关闭：ssh/tar 任一步非零退出即报错退出，不静默吞。

用法（任何模型/人接手都一样）：
    .venv/Scripts/python scripts/backup_am_assets.py           # 增量同步
    .venv/Scripts/python scripts/backup_am_assets.py --dry-run # 只看清单
依赖：本机 OpenSSH（ssh，BatchMode 免密）+ Git Bash tar；远端 GNU find/tar。
"""
from __future__ import annotations

import argparse
import secrets
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIRS = ("checkpoints", "local_runs", "published_training")
REMOTE_HOST = "compute-node-11"
REMOTE_ROOT = "/hwdata/home/jinqc/AlphaMaster-backup"
REMOTE_PYTHON = "/hwdata/home/jinqc/.local/bin/python3.11"
SKIP_IF_SAME_SIZE_OVER = 10 * 1024 * 1024  # 10 MiB
SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", REMOTE_HOST]
SQLITE_SIDECAR_SUFFIXES = (".sqlite3-wal", ".sqlite3-shm")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # encoding 强制 UTF-8 + replace：Windows 默认 GBK 解码会被子进程输出的
    # 非法字节直接干崩 reader 线程（实测教训：WSL bash 的 UTF-16 错误文本）。
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", **kw
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"命令失败({proc.returncode}): {' '.join(cmd[:4])}…\n"
            f"{(proc.stderr or '')[-500:]}"
        )
    return proc


def remote_inventory(
    remote_root: str = REMOTE_ROOT,
    *,
    create: bool = False,
    allow_missing: bool = False,
) -> dict[str, int]:
    """读取远端相对路径与字节数；任何 find 错误都失败关闭。"""
    quoted_root = shlex.quote(remote_root)
    if create:
        _run_remote_python(REMOTE_ENSURE_ROOT_SCRIPT, remote_root)
    missing_clause = ":" if allow_missing else (
        f"echo '远端备份目录不存在' >&2; exit 2"
    )
    proc = _run(
        SSH_BASE
        + [
            f"if test -d {quoted_root} && test ! -L {quoted_root}; then "
            f"target={quoted_root}; "
            f"if test -e {quoted_root}/CURRENT || "
            f"test -L {quoted_root}/CURRENT; then "
            f"test -f {quoted_root}/CURRENT && "
            f"test ! -L {quoted_root}/CURRENT || "
            "{ echo 'CURRENT 不是实体普通文件' >&2; exit 2; }; "
            f"generation=$(cat -- {quoted_root}/CURRENT) && "
            "printf '%s' \"$generation\" | "
            "LC_ALL=C grep -Eq '^gen-[0-9]+-[0-9]+-[0-9a-f]{24}$' || "
            "{ echo 'CURRENT 内容非法' >&2; exit 2; }; "
            f"target={quoted_root}/.generations/$generation; "
            f"test -d {quoted_root}/.generations && "
            f"test ! -L {quoted_root}/.generations || "
            "{ echo '.generations 不是实体目录' >&2; exit 2; }; "
            "test -d \"$target\" && test ! -L \"$target\" || "
            "{ echo 'CURRENT 指向的版本不存在' >&2; exit 2; }; "
            "fi; "
            "cd \"$target\" && find . -type f -printf '%P\\t%s\\n'; "
            f"elif test -e {quoted_root}; then "
            "echo '远端备份路径不是目录' >&2; exit 2; "
            f"else {missing_clause}; fi"
        ]
    )
    inv: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            raise RuntimeError("远端库存输出格式非法")
        rel, size = line.rsplit("\t", 1)
        normalized = rel.replace("\\", "/")
        if normalized in inv:
            raise RuntimeError("远端库存含重复路径")
        try:
            inv[normalized] = int(size)
        except ValueError as exc:
            raise RuntimeError("远端库存文件大小非法") from exc
    return inv


def _sqlite_snapshots(staging_root: Path) -> dict[str, Path]:
    """生成 SQLite 一致性主库及用于中和远端旧日志的空 sidecar。"""
    snapshots: dict[str, Path] = {}
    for top in BACKUP_DIRS:
        base = PROJECT_ROOT / top
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_symlink():
                raise RuntimeError(f"备份集合禁止符号链接: {p}")
            if not p.is_file() or p.suffix.lower() != ".sqlite3":
                continue
            rel = f"{top}/{p.relative_to(base).as_posix()}"
            snapshot = staging_root.joinpath(*Path(rel).parts)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            source_uri = f"file:{p.resolve().as_posix()}?mode=ro"
            try:
                with (
                    closing(
                        sqlite3.connect(source_uri, uri=True)
                    ) as source,
                    closing(sqlite3.connect(snapshot)) as target,
                ):
                    source.backup(target)
                    journal_mode = target.execute(
                        "PRAGMA journal_mode=DELETE"
                    ).fetchone()
                    integrity = target.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
            except sqlite3.Error as exc:
                raise RuntimeError(f"SQLite 在线快照失败: {rel}") from exc
            if (
                journal_mode is None
                or str(journal_mode[0]).lower() != "delete"
            ):
                raise RuntimeError(
                    f"SQLite 快照未转为自包含 DELETE 模式: {rel}"
                )
            if integrity is None or integrity[0] != "ok":
                raise RuntimeError(f"SQLite 在线快照完整性失败: {rel}")
            snapshots[rel] = snapshot
            for suffix in ("-wal", "-shm"):
                sidecar_rel = rel + suffix
                sidecar = Path(str(snapshot) + suffix)
                sidecar.touch(exist_ok=False)
                if sidecar.stat().st_size != 0:
                    raise RuntimeError(
                        f"SQLite 空 sidecar 创建失败: {sidecar_rel}"
                    )
                snapshots[sidecar_rel] = sidecar
    return snapshots


def local_manifest(
    snapshots: dict[str, Path],
) -> list[tuple[str, int, Path]]:
    files: list[tuple[str, int, Path]] = []
    seen_sqlite: set[str] = set()
    sqlite_main_paths = {
        rel
        for rel in snapshots
        if rel.lower().endswith(".sqlite3")
    }
    for top in BACKUP_DIRS:
        base = PROJECT_ROOT / top
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_symlink():
                raise RuntimeError(f"备份集合禁止符号链接: {p}")
            if (
                not p.is_file()
                or p.name.lower().endswith(SQLITE_SIDECAR_SUFFIXES)
            ):
                continue
            rel = f"{top}/{p.relative_to(base).as_posix()}"
            if p.suffix.lower() == ".sqlite3":
                if rel not in snapshots:
                    raise RuntimeError(
                        f"SQLite 文件在快照阶段后出现: {rel}"
                    )
                seen_sqlite.add(rel)
            source = snapshots.get(rel, p)
            files.append((rel, source.stat().st_size, source))
    if seen_sqlite != sqlite_main_paths:
        raise RuntimeError("SQLite 文件集合在快照期间发生变化")
    for rel in sorted(set(snapshots) - sqlite_main_paths):
        source = snapshots[rel]
        if source.stat().st_size != 0:
            raise RuntimeError(f"SQLite sidecar 必须为空: {rel}")
        files.append((rel, 0, source))
    return files


def _transfer_tar(
    *,
    source_root: Path,
    relative_paths: list[str],
    listfile: Path,
    remote_target: str,
) -> None:
    if not relative_paths:
        return
    listfile.write_text(
        "\n".join(relative_paths) + "\n",
        encoding="utf-8",
    )
    with tempfile.TemporaryFile() as tar_stderr:
        p_tar = subprocess.Popen(
            [
                "tar",
                "czf",
                "-",
                "-C",
                str(source_root),
                "-T",
                str(listfile),
            ],
            stdout=subprocess.PIPE,
            stderr=tar_stderr,
        )
        if p_tar.stdout is None:
            p_tar.terminate()
            p_tar.wait()
            raise RuntimeError("无法建立本机 tar 管道")
        try:
            p_ssh = subprocess.Popen(
                SSH_BASE
                + [
                    "tar xzf - -C "
                    + shlex.quote(remote_target)
                ],
                stdin=p_tar.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            p_tar.stdout.close()
            p_tar.terminate()
            p_tar.wait()
            raise
        p_tar.stdout.close()
        _ssh_out, ssh_err = p_ssh.communicate()
        p_tar.wait()
        tar_stderr.seek(0)
        tar_err = tar_stderr.read()
    if p_tar.returncode != 0 or p_ssh.returncode != 0:
        raise RuntimeError(
            f"tar/ssh 管道失败: tar={p_tar.returncode} "
            f"ssh={p_ssh.returncode}\n"
            f"{(tar_err or b'').decode('utf-8', 'replace')[-300:]}\n"
            f"{(ssh_err or b'').decode('utf-8', 'replace')[-300:]}"
        )


REMOTE_ENSURE_ROOT_SCRIPT = r"""
import os
import stat
import sys
from pathlib import Path

root_raw = Path(os.path.abspath(sys.argv[1]))
try:
    root_info = os.lstat(root_raw)
except FileNotFoundError:
    parent_info = os.lstat(root_raw.parent)
    if not stat.S_ISDIR(parent_info.st_mode):
        raise SystemExit("备份根父目录不是实体目录")
    os.mkdir(root_raw, 0o700)
    root_info = os.lstat(root_raw)
if not stat.S_ISDIR(root_info.st_mode):
    raise SystemExit("备份根不是实体目录")
root = root_raw.resolve(strict=True)

incoming_raw = root / ".incoming"
try:
    incoming_info = os.lstat(incoming_raw)
except FileNotFoundError:
    os.mkdir(incoming_raw, 0o700)
    incoming_info = os.lstat(incoming_raw)
if not stat.S_ISDIR(incoming_info.st_mode):
    raise SystemExit(".incoming 不是实体目录")
incoming = incoming_raw.resolve(strict=True)
if incoming.parent != root:
    raise SystemExit(".incoming 解析越界")
"""


REMOTE_CREATE_STAGING_SCRIPT = r"""
import os
import re
import stat
import sys
from pathlib import Path

root_raw = Path(os.path.abspath(sys.argv[1]))
root_info = os.lstat(root_raw)
if not stat.S_ISDIR(root_info.st_mode):
    raise SystemExit("备份根不是实体目录")
root = root_raw.resolve(strict=True)
incoming_raw = root / ".incoming"
incoming_info = os.lstat(incoming_raw)
if not stat.S_ISDIR(incoming_info.st_mode):
    raise SystemExit(".incoming 不是实体目录")
incoming = incoming_raw.resolve(strict=True)
if incoming.parent != root:
    raise SystemExit(".incoming 解析越界")

name = sys.argv[2]
if re.fullmatch(r"am-backup-[0-9a-f]{24}", name) is None:
    raise SystemExit("staging 名称非法")
staging_raw = incoming_raw / name
os.mkdir(staging_raw, 0o700)
staging_info = os.lstat(staging_raw)
if not stat.S_ISDIR(staging_info.st_mode):
    raise SystemExit("staging 不是实体目录")
staging = staging_raw.resolve(strict=True)
if staging.parent != incoming:
    raise SystemExit("staging 解析越界")
print(staging_raw)
"""


REMOTE_PROMOTE_SCRIPT = r"""
import atexit
import os
import re
import secrets
import shutil
import stat
import sys
import time
from pathlib import Path

root_raw = Path(os.path.abspath(sys.argv[2]))
root_info = os.lstat(root_raw)
if not stat.S_ISDIR(root_info.st_mode):
    raise SystemExit("远端备份根不是实体目录")
root = root_raw.resolve(strict=True)

incoming_raw = root / ".incoming"
incoming_info = os.lstat(incoming_raw)
if not stat.S_ISDIR(incoming_info.st_mode):
    raise SystemExit("远端 .incoming 不是实体目录")
incoming = incoming_raw.resolve(strict=True)
if incoming.parent != root:
    raise SystemExit("远端 .incoming 解析越界")

staging_raw = Path(os.path.abspath(sys.argv[1]))
if staging_raw.parent != incoming_raw:
    raise SystemExit("远端 staging 原始路径越界")
staging_info = os.lstat(staging_raw)
if not stat.S_ISDIR(staging_info.st_mode):
    raise SystemExit("远端 staging 不是实体目录")
staging = staging_raw.resolve(strict=True)
if (
    staging.parent != incoming
    or re.fullmatch(r"am-backup-[0-9a-f]{24}", staging.name) is None
):
    raise SystemExit("远端 staging 身份非法")

files = []
for current, directories, names in os.walk(
    staging,
    topdown=True,
    followlinks=False,
):
    current_path = Path(current)
    for name in directories:
        candidate = current_path / name
        if candidate.is_symlink():
            raise SystemExit("远端 staging 含目录符号链接")
    for name in names:
        source = current_path / name
        info = os.lstat(source)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit("远端 staging 含非普通文件")
        relative = source.relative_to(staging)
        if not relative.parts or relative.parts[0] not in {
            "checkpoints",
            "local_runs",
            "published_training",
        }:
            raise SystemExit("远端发布路径越界")
        files.append((source, relative))

generations_raw = root / ".generations"
try:
    generations_info = os.lstat(generations_raw)
except FileNotFoundError:
    os.mkdir(generations_raw, 0o700)
    generations_info = os.lstat(generations_raw)
if not stat.S_ISDIR(generations_info.st_mode):
    raise SystemExit("远端 versions 根身份非法")
generations = generations_raw.resolve(strict=True)
if generations.parent != root:
    raise SystemExit("远端 versions 根解析越界")

generation_name = (
    f"gen-{time.time_ns()}-{os.getpid()}-{secrets.token_hex(12)}"
)
if re.fullmatch(r"gen-[0-9]+-[0-9]+-[0-9a-f]{24}", generation_name) is None:
    raise SystemExit("版本名生成失败")
building = generations / (generation_name + ".building")
generation = generations / generation_name
publish_state = {
    "building_created": False,
    "renamed": False,
    "pointer_replaced": False,
    "pointer_tmp": None,
}

def fsync_directory(path):
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def cleanup_private_publish_paths():
    pointer_tmp = publish_state["pointer_tmp"]
    if pointer_tmp is not None:
        try:
            info = os.lstat(pointer_tmp)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(info.st_mode):
                os.unlink(pointer_tmp)
    if publish_state["pointer_replaced"]:
        return
    pointer = root / "CURRENT"
    try:
        pointer_info = os.lstat(pointer)
    except FileNotFoundError:
        current_value = None
    except OSError:
        return
    else:
        if not stat.S_ISREG(pointer_info.st_mode):
            return
        try:
            current_value = pointer.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return
    if current_value == generation_name:
        return
    if current_value is not None and re.fullmatch(
        r"gen-[0-9]+-[0-9]+-[0-9a-f]{24}",
        current_value,
    ) is None:
        return
    private_path = generation if publish_state["renamed"] else building
    if not publish_state["building_created"]:
        return
    try:
        info = os.lstat(private_path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        shutil.rmtree(private_path)
        fsync_directory(generations)

atexit.register(cleanup_private_publish_paths)
os.mkdir(building, 0o700)
publish_state["building_created"] = True

def read_current_generation():
    pointer = root / "CURRENT"
    try:
        info = os.lstat(pointer)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("CURRENT 不是实体普通文件")
    value = pointer.read_text(encoding="ascii").strip()
    if re.fullmatch(r"gen-[0-9]+-[0-9]+-[0-9a-f]{24}", value) is None:
        raise SystemExit("CURRENT 内容非法")
    candidate = generations / value
    candidate_info = os.lstat(candidate)
    if not stat.S_ISDIR(candidate_info.st_mode):
        raise SystemExit("CURRENT 指向的版本不是实体目录")
    candidate_resolved = candidate.resolve(strict=True)
    if candidate_resolved.parent != generations:
        raise SystemExit("CURRENT 指向的版本解析越界")
    return candidate_resolved

def ensure_destination_parent(base, relative):
    current = base
    for part in relative.parent.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode):
            raise SystemExit("远端发布目录祖先不是实体目录")

def clone_regular_tree(source_root, relative_root):
    source_base = source_root / relative_root
    if not source_base.exists():
        return
    source_info = os.lstat(source_base)
    if not stat.S_ISDIR(source_info.st_mode):
        raise SystemExit("上一版本顶层不是实体目录")
    for current, directories, names in os.walk(
        source_base,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        relative_current = current_path.relative_to(source_root)
        target_current = building / relative_current
        target_current.mkdir(parents=True, exist_ok=True)
        for name in directories:
            if (current_path / name).is_symlink():
                raise SystemExit("上一版本含目录符号链接")
        for name in names:
            source = current_path / name
            info = os.lstat(source)
            if not stat.S_ISREG(info.st_mode):
                raise SystemExit("上一版本含非普通文件")
            os.link(source, target_current / name, follow_symlinks=False)

previous = read_current_generation()
if previous is None:
    for top in ("checkpoints", "local_runs", "published_training"):
        clone_regular_tree(root, Path(top))
else:
    for top in ("checkpoints", "local_runs", "published_training"):
        clone_regular_tree(previous, Path(top))

for source, relative in sorted(files, key=lambda item: item[1].as_posix()):
    destination = building / relative
    ensure_destination_parent(building, relative)
    os.replace(source, destination)

for current, _directories, _names in os.walk(
    staging,
    topdown=False,
    followlinks=False,
):
    Path(current).rmdir()
fsync_directory(incoming)

directories_to_sync = []
for current, directories, names in os.walk(
    building,
    topdown=True,
    followlinks=False,
):
    current_path = Path(current)
    directories_to_sync.append(current_path)
    for name in directories:
        info = os.lstat(current_path / name)
        if not stat.S_ISDIR(info.st_mode):
            raise SystemExit("待发布版本含目录符号链接")
    for name in names:
        file_path = current_path / name
        descriptor = os.open(
            file_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise SystemExit("待发布版本含非普通文件")
            if os.name != "nt":
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
for directory in reversed(directories_to_sync):
    fsync_directory(directory)

os.rename(building, generation)
publish_state["renamed"] = True
fsync_directory(generations)

pointer_tmp = root / (
    ".CURRENT-" + generation_name + "-" + secrets.token_hex(12)
)
publish_state["pointer_tmp"] = pointer_tmp
pointer_fd = os.open(
    pointer_tmp,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o600,
)
try:
    os.write(pointer_fd, (generation_name + "\n").encode("ascii"))
    os.fsync(pointer_fd)
finally:
    os.close(pointer_fd)
os.replace(pointer_tmp, root / "CURRENT")
publish_state["pointer_tmp"] = None
publish_state["pointer_replaced"] = True
fsync_directory(root)
print(len(files))
"""


REMOTE_CLEANUP_SCRIPT = r"""
import os
import re
import shutil
import stat
import sys
from pathlib import Path

root_raw = Path(os.path.abspath(sys.argv[2]))
root_info = os.lstat(root_raw)
if not stat.S_ISDIR(root_info.st_mode):
    raise SystemExit("备份根不是实体目录")
root = root_raw.resolve(strict=True)
incoming_raw = root / ".incoming"
incoming_info = os.lstat(incoming_raw)
if not stat.S_ISDIR(incoming_info.st_mode):
    raise SystemExit(".incoming 不是实体目录")
incoming = incoming_raw.resolve(strict=True)
if incoming.parent != root:
    raise SystemExit(".incoming 解析越界")

candidate_raw = Path(os.path.abspath(sys.argv[1]))
if (
    candidate_raw.parent != incoming_raw
    or re.fullmatch(r"am-backup-[0-9a-f]{24}", candidate_raw.name) is None
):
    raise SystemExit("拒绝清理非本轮 staging")
try:
    candidate_info = os.lstat(candidate_raw)
except FileNotFoundError:
    raise SystemExit(0)
if not stat.S_ISDIR(candidate_info.st_mode):
    raise SystemExit("拒绝清理非实体 staging")
resolved = candidate_raw.resolve(strict=True)
if resolved.parent != incoming:
    raise SystemExit("拒绝清理越界 staging")
if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
    raise SystemExit("当前 Python 不支持抗符号链接 rmtree")
incoming_fd = os.open(
    incoming,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
)
try:
    shutil.rmtree(candidate_raw.name, dir_fd=incoming_fd)
finally:
    os.close(incoming_fd)
"""


def _run_remote_python(script: str, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        SSH_BASE + [REMOTE_PYTHON, "-", *args],
        input=script.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "远端 Python 失败"
            f"({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')[-500:]}"
        )
    return proc


def _cleanup_remote_staging(remote_staging: str) -> None:
    _run_remote_python(
        REMOTE_CLEANUP_SCRIPT,
        remote_staging,
        REMOTE_ROOT,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    scratch = PROJECT_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="am_backup_sqlite_",
        dir=scratch,
    ) as staging_value:
        staging_root = Path(staging_value)
        snapshots = _sqlite_snapshots(staging_root)
        remote = remote_inventory(
            create=not args.dry_run,
            allow_missing=args.dry_run,
        )
        todo = [
            (rel, size, source)
            for rel, size, source in local_manifest(snapshots)
            if not (
                size > SKIP_IF_SAME_SIZE_OVER
                and remote.get(rel) == size
            )
        ]
        total_mb = sum(size for _rel, size, _source in todo) / 1e6
        print(
            f"待传 {len(todo)} 个文件，共 {total_mb:.1f} MB"
            "（SQLite 快照 "
            f"{sum(rel.lower().endswith('.sqlite3') for rel in snapshots)} 个；"
            f"远端已有 {len(remote)} 个）"
        )
        if args.dry_run or not todo:
            return 0

        regular_paths = [
            rel
            for rel, _size, _source in todo
            if rel not in snapshots
        ]
        snapshot_paths = [
            rel
            for rel, _size, _source in todo
            if rel in snapshots
        ]
        listfile_root = staging_root / ".lists"
        listfile_root.mkdir()
        regular_listfile = listfile_root / "regular.txt"
        sqlite_listfile = listfile_root / "sqlite.txt"
        staging_name = f"am-backup-{secrets.token_hex(12)}"
        created = _run_remote_python(
            REMOTE_CREATE_STAGING_SCRIPT,
            REMOTE_ROOT,
            staging_name,
        )
        remote_staging = created.stdout.decode("utf-8").strip()
        expected_staging = f"{REMOTE_ROOT}/.incoming/{staging_name}"
        if remote_staging != expected_staging:
            raise RuntimeError("远端 staging 创建回执非法")
        try:
            _transfer_tar(
                source_root=PROJECT_ROOT,
                relative_paths=regular_paths,
                listfile=regular_listfile,
                remote_target=remote_staging,
            )
            _transfer_tar(
                source_root=staging_root,
                relative_paths=snapshot_paths,
                listfile=sqlite_listfile,
                remote_target=remote_staging,
            )
            staged = remote_inventory(remote_staging)
            expected = {
                rel: size for rel, size, _source in todo
            }
            if staged != expected:
                missing = sorted(set(expected) - set(staged))
                extra = sorted(set(staged) - set(expected))
                changed = sorted(
                    rel
                    for rel in set(expected) & set(staged)
                    if expected[rel] != staged[rel]
                )
                raise RuntimeError(
                    "远端 staging 校验失败: "
                    f"缺失={missing[:5]} 多余={extra[:5]} "
                    f"大小差异={changed[:5]}"
                )
            promoted = _run_remote_python(
                REMOTE_PROMOTE_SCRIPT,
                remote_staging,
                REMOTE_ROOT,
            )
            try:
                promoted_count = int(
                    promoted.stdout.decode("utf-8").strip()
                )
            except ValueError as exc:
                raise RuntimeError("远端发布计数非法") from exc
            if promoted_count != len(todo):
                raise RuntimeError("远端发布计数与待传清单不一致")
        except Exception:
            _cleanup_remote_staging(remote_staging)
            raise

        after = remote_inventory()
        missing = [
            rel for rel, size, _source in todo if after.get(rel) != size
        ]
        if missing:
            raise RuntimeError(
                f"备份校验失败，{len(missing)} 个文件远端大小不符: "
                f"{missing[:5]}"
            )
        print(
            f"完成：{len(todo)} 个文件已同步并逐一核对大小，"
            f"耗时 {time.time()-t0:.0f}s；远端现有 {len(after)} 个文件"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
