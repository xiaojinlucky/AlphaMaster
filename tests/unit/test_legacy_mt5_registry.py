from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.dataset_contracts import MT5_LEGACY_SOURCE_ID
from data_pipeline.legacy_mt5_registry import (
    LegacyRegistrationError,
    apply_registration_plan,
    build_registration_plan,
)
from data_pipeline.parquet_manager import inspect_parquet_file


def test_registration_cli_runs_directly_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "register_legacy_mt5_parquet.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "旧 MT5 Parquet" in completed.stdout


def _write_parquet(path: Path, rows: int = 3000) -> None:
    frame = pd.DataFrame(
        {
            "time": [1_700_000_000 + index * 300 for index in range(rows)],
            "open": [100.0 + index / 1000 for index in range(rows)],
            "high": [101.0 + index / 1000 for index in range(rows)],
            "low": [99.0 + index / 1000 for index in range(rows)],
            "close": [100.5 + index / 1000 for index in range(rows)],
            "tick_volume": [100 + index for index in range(rows)],
        }
    )
    frame.to_parquet(path, index=False)


def test_plan_marks_valid_bare_mt5_file_as_eligible(tmp_path: Path) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)

    plan = build_registration_plan(tmp_path, feed_id="legacy-bulk")

    assert plan["summary"] == {
        "total": 1,
        "eligible": 1,
        "already_registered": 0,
        "rejected": 0,
    }
    row = plan["files"][0]
    assert row["status"] == "eligible"
    assert row["symbol"] == "NVDA"
    assert row["timeframe"] == "M5"
    assert len(plan["plan_sha256"]) == 64
    assert not data.with_suffix(".manifest.json").exists()


def test_plan_binds_multiple_source_reports_by_hash(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "NVDA_M5.parquet")
    first = tmp_path / "_bulk_sync_report.json"
    retry = tmp_path / "_bulk_sync_retry_report.json"
    first.write_text('{"jobs_ok": 642, "jobs_failed": 26}', encoding="utf-8")
    retry.write_text('{"jobs_ok": 26, "jobs_failed": 0}', encoding="utf-8")

    plan = build_registration_plan(
        tmp_path,
        source_report=[first, retry],
    )

    assert [row["filename"] for row in plan["source_reports"]] == [
        first.name,
        retry.name,
    ]
    assert all(len(row["sha256"]) == 64 for row in plan["source_reports"])


def test_apply_requires_exact_source_acknowledgement(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "NVDA_M5.parquet")
    plan = build_registration_plan(tmp_path)

    with pytest.raises(LegacyRegistrationError, match="明确确认"):
        apply_registration_plan(
            plan,
            expected_plan_sha256=plan["plan_sha256"],
            source_acknowledgement="",
        )


def test_apply_publishes_legacy_manifest_and_loader_rechecks_it(
    tmp_path: Path,
) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    source_report = tmp_path / "_bulk_sync_report.json"
    source_report.write_text('{"jobs_ok": 1}', encoding="utf-8")
    plan = build_registration_plan(
        tmp_path,
        feed_id="legacy-bulk",
        source_report=[source_report],
    )

    report = apply_registration_plan(
        plan,
        expected_plan_sha256=plan["plan_sha256"],
        source_acknowledgement="MetaTrader5",
    )

    assert report["summary"]["registered"] == 1
    assert report["summary"]["failed"] == 0
    manifest_path = data.with_suffix(".manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["provenance_level"] == "legacy_user_attested"
    assert payload["registration_plan_sha256"] == plan["plan_sha256"]
    assert payload["source_reports"] == plan["source_reports"]
    info = inspect_parquet_file(data)
    assert info["source"] == MT5_LEGACY_SOURCE_ID
    assert info["registration"] == "registered"
    assert info["capabilities"]["remote_training"] is True
    remote_info = inspect_parquet_file(
        data,
        expected_source_id=MT5_LEGACY_SOURCE_ID,
        expected_periods_per_year=payload["periods_per_year"],
        expected_minimum_bars=payload["minimum_bars"],
    )
    assert remote_info["source"] == MT5_LEGACY_SOURCE_ID


def test_existing_manifest_is_never_overwritten(tmp_path: Path) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    first_plan = build_registration_plan(tmp_path)
    apply_registration_plan(
        first_plan,
        expected_plan_sha256=first_plan["plan_sha256"],
        source_acknowledgement="MetaTrader5",
    )
    manifest_path = data.with_suffix(".manifest.json")
    original = manifest_path.read_bytes()

    second_plan = build_registration_plan(tmp_path)

    assert second_plan["summary"]["already_registered"] == 1
    report = apply_registration_plan(
        second_plan,
        expected_plan_sha256=second_plan["plan_sha256"],
        source_acknowledgement="MetaTrader5",
    )
    assert report["summary"]["already_registered"] == 1
    assert manifest_path.read_bytes() == original


def test_file_drift_after_plan_fails_without_sidecar(tmp_path: Path) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    plan = build_registration_plan(tmp_path)
    _write_parquet(data, rows=3001)

    report = apply_registration_plan(
        plan,
        expected_plan_sha256=plan["plan_sha256"],
        source_acknowledgement="MetaTrader5",
    )

    assert report["summary"]["failed"] == 1
    assert "漂移" in report["results"][0]["message"]
    assert not data.with_suffix(".manifest.json").exists()


def test_insufficient_file_is_rejected_during_plan(tmp_path: Path) -> None:
    data = tmp_path / "SPCX_D1.parquet"
    _write_parquet(data, rows=17)

    plan = build_registration_plan(tmp_path)

    assert plan["summary"]["rejected"] == 1
    assert "数据不足" in plan["files"][0]["reason"]
    assert not data.with_suffix(".manifest.json").exists()
