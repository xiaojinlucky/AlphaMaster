from __future__ import annotations

import pytest

import train_file
from utils.training_runtime import require_slurm_training_runtime


def test_training_runtime_accepts_only_linux_slurm_worker() -> None:
    require_slurm_training_runtime(
        {
            "ALPHAMASTER_TRAINING_RUNTIME": "slurm_worker_v1",
            "SLURM_JOB_ID": "568306",
            "SLURMD_NODENAME": "cu16",
        },
        platform_name="linux",
    )


@pytest.mark.parametrize(
    ("environment", "platform_name"),
    [
        ({}, "win32"),
        (
            {
                "ALPHAMASTER_TRAINING_RUNTIME": "slurm_worker_v1",
                "SLURM_JOB_ID": "568306",
                "SLURMD_NODENAME": "cu16",
            },
            "win32",
        ),
        (
            {
                "SLURM_JOB_ID": "568306",
                "SLURMD_NODENAME": "cu16",
            },
            "linux",
        ),
        (
            {
                "ALPHAMASTER_TRAINING_RUNTIME": "slurm_worker_v1",
                "SLURM_JOB_ID": "bad",
                "SLURMD_NODENAME": "cu16",
            },
            "linux",
        ),
    ],
)
def test_training_runtime_rejects_local_or_uncontrolled_entry(
    environment: dict[str, str],
    platform_name: str,
) -> None:
    with pytest.raises(RuntimeError, match="只能由服务器 Slurm Worker 启动"):
        require_slurm_training_runtime(
            environment,
            platform_name=platform_name,
        )


def test_train_file_rejects_windows_before_reading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALPHAMASTER_TRAINING_RUNTIME", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)

    with pytest.raises(RuntimeError, match="Windows 本机和普通命令行训练已禁用"):
        train_file.train_from_file("missing_D1.parquet")
