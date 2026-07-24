"""研究层差分实验的共享输入与确定性输出工具。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def load_fixture(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    required = {
        "contract_version",
        "case_id",
        "clock",
        "constituents",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"fixture 缺少字段: {missing}")
    return payload, hashlib.sha256(raw).hexdigest()


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_git_source(root: Path, expected_commit: str) -> str:
    resolved = root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != expected_commit:
        raise RuntimeError(
            f"第三方源码提交不匹配: expected={expected_commit}, actual={head}"
        )
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    if status.strip():
        raise RuntimeError(f"第三方源码工作树不干净: {resolved}")
    return head


def verify_repo_paths_clean(root: Path, paths: list[str]) -> None:
    result = subprocess.run(
        ["git", "-C", str(root.resolve()), "diff", "--quiet", "HEAD", "--"]
        + paths,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"AlphaMaster 目标源码存在未提交改动: {paths}")
