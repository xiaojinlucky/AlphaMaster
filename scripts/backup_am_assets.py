"""AlphaMaster 模型资产异地备份（WO-AM-08）。

把本机 checkpoints / local_runs / published_training 增量同步到 HPC
`/hwdata/home/jinqc/AlphaMaster-backup/`。设计约束：
- 只新增、不删除：远端永不删除文件（防误删）；本地也绝不写入项目数据。
- 增量判定：远端已存在同路径且同字节数、且体积 > 10 MiB 的文件跳过
  （checkpoint 内容寻址、只增不改）；小文件（清单/SQLite/JSON）每次都重传，
  保证账本类可变文件始终最新。
- 传输：把待传清单打成 tar 流经 ssh 管道解到远端，避免逐文件 scp 往返。
- 失败关闭：ssh/tar 任一步非零退出即报错退出，不静默吞。

用法（任何模型/人接手都一样）：
    .venv/Scripts/python scripts/backup_am_assets.py           # 增量同步
    .venv/Scripts/python scripts/backup_am_assets.py --dry-run # 只看清单
依赖：本机 OpenSSH（ssh，BatchMode 免密）+ Git Bash tar；远端 GNU find/tar。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIRS = ("checkpoints", "local_runs", "published_training")
REMOTE_HOST = "compute-node-11"
REMOTE_ROOT = "/hwdata/home/jinqc/AlphaMaster-backup"
SKIP_IF_SAME_SIZE_OVER = 10 * 1024 * 1024  # 10 MiB
SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", REMOTE_HOST]


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


def remote_inventory() -> dict[str, int]:
    """远端 相对路径 -> 字节数（目录不存在按空处理并创建）。"""
    _run(SSH_BASE + [f"mkdir -p {REMOTE_ROOT}"])
    proc = _run(
        SSH_BASE
        + [f"cd {REMOTE_ROOT} && find . -type f -printf '%P\\t%s\\n' 2>/dev/null || true"]
    )
    inv: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if "\t" in line:
            rel, size = line.rsplit("\t", 1)
            inv[rel.replace("\\", "/")] = int(size)
    return inv


def local_manifest() -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    for top in BACKUP_DIRS:
        base = PROJECT_ROOT / top
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel = f"{top}/{p.relative_to(base).as_posix()}"
                files.append((rel, p.stat().st_size))
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    remote = remote_inventory()
    todo = [
        (rel, size)
        for rel, size in local_manifest()
        if not (
            size > SKIP_IF_SAME_SIZE_OVER and remote.get(rel) == size
        )
    ]
    total_mb = sum(s for _, s in todo) / 1e6
    print(f"待传 {len(todo)} 个文件，共 {total_mb:.1f} MB（远端已有 {len(remote)} 个）")
    if args.dry_run or not todo:
        return 0

    listfile = PROJECT_ROOT / "scratch" / "backup_transfer_list.txt"
    listfile.parent.mkdir(exist_ok=True)
    listfile.write_text("\n".join(rel for rel, _ in todo) + "\n", encoding="utf-8")

    # tar 流式管道（纯 Python 双进程，不经 bash——PATH 上的 bash 可能命中
    # WSL 而非 Git Bash）：本地 Windows 自带 bsdtar 打包清单 → stdout 直接
    # 接远端 GNU tar 解包（覆盖同名小文件）。
    p_tar = subprocess.Popen(
        [
            "tar", "czf", "-",
            "-C", str(PROJECT_ROOT),
            "-T", str(listfile),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p_ssh = subprocess.Popen(
        SSH_BASE + [f"tar xzf - -C {REMOTE_ROOT}"],
        stdin=p_tar.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p_tar.stdout.close()  # 让 ssh 端收到 EOF
    ssh_out, ssh_err = p_ssh.communicate()
    tar_err = p_tar.stderr.read()  # stdout 已交给 ssh，不能再用 communicate 读
    p_tar.stderr.close()
    p_tar.wait()
    if p_tar.returncode != 0 or p_ssh.returncode != 0:
        raise RuntimeError(
            f"tar/ssh 管道失败: tar={p_tar.returncode} ssh={p_ssh.returncode}\n"
            f"{(tar_err or b'').decode('utf-8', 'replace')[-300:]}\n"
            f"{(ssh_err or b'').decode('utf-8', 'replace')[-300:]}"
        )

    after = remote_inventory()
    missing = [rel for rel, size in todo if after.get(rel) != size]
    if missing:
        raise RuntimeError(f"备份校验失败，{len(missing)} 个文件远端大小不符: {missing[:5]}")
    print(
        f"完成：{len(todo)} 个文件已同步并逐一核对大小，耗时 {time.time()-t0:.0f}s；"
        f"远端现有 {len(after)} 个文件"
    )
    listfile.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
