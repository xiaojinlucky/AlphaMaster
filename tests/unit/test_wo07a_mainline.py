from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import wo07a_mainline_core as core


def test_mainline_core_builds_expected_qfq_and_d1() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06"]
            ),
            "open": [10.0, 10.2, 20.4],
            "high": [10.5, 10.6, 21.0],
            "low": [9.8, 10.0, 20.0],
            "close": [10.1, 10.3, 20.6],
            "volume": [100, 200, 300],
        }
    )
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-06"]),
            "factor": [1.0, 2.0],
            "factor_decimal": ["1.0", "2.0"],
        }
    )

    qfq, switches, large_switches, violations = core.build_qfq(
        history, factors
    )
    assert switches == ("2020-01-06",)
    assert large_switches == ("2020-01-06",)
    assert violations == ()
    assert np.allclose(qfq["close"], [10.1, 10.3, 10.3])

    d1 = core.make_d1(qfq)
    assert tuple(d1.columns) == core.D1_COLUMNS
    assert [str(d1[column].dtype) for column in d1.columns] == [
        "int64",
        "float32",
        "float32",
        "float32",
        "float32",
        "int64",
    ]
    assert d1["time"].is_monotonic_increasing
    assert d1["tick_volume"].tolist() == [100, 200, 300]


def test_mainline_core_rejects_duplicate_factor_json_keys() -> None:
    raw = (
        b'var sz000001qfq='
        b'{"total":1,"data":[{"d":"2020-01-01","f":1.0,"f":2.0}]};'
    )
    with pytest.raises(core.MainlineCoreError, match="重复键"):
        core.decode_factor(raw, "000001")


def test_mainline_release_files_do_not_import_retired_branch() -> None:
    build_text = (SCRIPTS / "wo07a_build_v4_mainline.py").read_text("utf-8")
    audit_text = (SCRIPTS / "wo07a_audit_mainline.py").read_text("utf-8")
    entry_text = (SCRIPTS / "wo07a_audit.py").read_text("utf-8")

    assert "wo07a_build_v4.py" not in build_text
    assert "wo07a_contract" not in build_text
    assert "wo07a_publish_guard" not in build_text
    assert "wo07a_mainline_core" not in audit_text
    assert "wo07a_build_v4" not in audit_text
    assert "extended-adjudication" not in entry_text


def test_mainline_cli_exposes_all_artifact_roots() -> None:
    build_text = (SCRIPTS / "wo07a_build_v4_mainline.py").read_text("utf-8")
    audit_text = (SCRIPTS / "wo07a_audit_mainline.py").read_text("utf-8")

    for option in ("--v3", "--output", "--capture", "--check-only"):
        assert option in build_text
    for option in ("--v3", "--v4"):
        assert option in audit_text
