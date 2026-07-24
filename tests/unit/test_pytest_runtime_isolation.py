from __future__ import annotations

import os
from pathlib import Path


def test_global_web_managers_use_pytest_runtime_root() -> None:
    from web.a_share_pipeline import a_share_pipeline_manager
    from web.training_manager import training_manager

    expected = Path(os.environ["ALPHAMASTER_LOCAL_RUNS_ROOT"]).resolve()

    assert training_manager.local_runs_root == expected
    assert a_share_pipeline_manager.local_runs_root == expected
    assert training_manager.snapshot()["job"] is None
