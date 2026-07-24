"""模型训练运行位置的硬门禁。"""
from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping


SLURM_WORKER_RUNTIME = "slurm_worker_v1"
_JOB_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")


def require_slurm_training_runtime(
    environ: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
) -> None:
    """只允许受控 Slurm Worker 在 Linux 计算节点启动模型训练。"""
    env = environ or os.environ
    platform_value = platform_name or sys.platform
    if (
        not platform_value.startswith("linux")
        or env.get("ALPHAMASTER_TRAINING_RUNTIME") != SLURM_WORKER_RUNTIME
        or _JOB_ID_RE.fullmatch(str(env.get("SLURM_JOB_ID") or "")) is None
        or not str(env.get("SLURMD_NODENAME") or "").strip()
    ):
        raise RuntimeError(
            "AlphaMaster 模型训练只能由服务器 Slurm Worker 启动；"
            "Windows 本机和普通命令行训练已禁用"
        )


__all__ = ["SLURM_WORKER_RUNTIME", "require_slurm_training_runtime"]
