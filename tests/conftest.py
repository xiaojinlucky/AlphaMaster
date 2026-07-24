"""Shared pytest fixtures."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_PYTEST_RUNTIME = tempfile.TemporaryDirectory(
    prefix="alphamaster-pytest-runtime-"
)
_PYTEST_RUNTIME_DIR = Path(_PYTEST_RUNTIME.name)
os.environ["ALPHAMASTER_LOCAL_RUNS_ROOT"] = str(
    _PYTEST_RUNTIME_DIR / "local_runs"
)
os.environ["ALPHAMASTER_TRAINING_QUEUE_DB"] = str(
    _PYTEST_RUNTIME_DIR / "training_queue.sqlite3"
)


@pytest.fixture(autouse=True)
def isolate_web_settings(tmp_path, monkeypatch):
    """Never let unit tests overwrite the real web_settings.json."""
    settings_path = tmp_path / "web_settings.json"
    monkeypatch.setattr("web.settings.SETTINGS_PATH", settings_path)
