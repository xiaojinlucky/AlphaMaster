"""模型训练只能使用 Slurm，任何本机后端配置都必须失败关闭。"""
from __future__ import annotations

import pytest

from web import training_manager as manager_module


def test_missing_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAINING_BACKEND", raising=False)

    with pytest.raises(RuntimeError, match="只允许 TRAINING_BACKEND=slurm"):
        manager_module._build_training_manager()


def test_local_backend_is_rejected_even_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_BACKEND", "local")

    with pytest.raises(RuntimeError, match="禁止在 Windows 本机训练"):
        manager_module._build_training_manager()
