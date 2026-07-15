"""训练后端必须显式选择，远程配置缺失时禁止退回本机。"""
from __future__ import annotations

import pytest

from web import training_manager as manager_module


def test_missing_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAINING_BACKEND", raising=False)

    with pytest.raises(RuntimeError, match="拒绝隐式本机训练"):
        manager_module._build_training_manager()


def test_local_backend_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAINING_BACKEND", "local")

    manager = manager_module._build_training_manager()

    assert type(manager) is manager_module.TrainingManager
