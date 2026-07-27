from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_csi300_historical_am_inputs_v3_resume_v2 as RESUME_V2
from scripts.build_csi300_historical_am_inputs_v3 import (
    build_membership_evidence,
    classify_status,
    validate_coverage_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INITIAL_SCRIPT = (
    PROJECT_ROOT / "scripts" / "build_csi300_historical_am_inputs_v3.py"
)
FORMAL_RESUME_SCRIPT = (
    PROJECT_ROOT / "scripts" / "build_csi300_historical_am_inputs_v3_resume_v2.py"
)
RELEASE_BINDING = (
    PROJECT_ROOT
    / "docs"
    / "evidence"
    / "csi300_historical_am_inputs_v3_release_binding_20260726.json"
)
EXPECTED_V3_MANIFEST_SHA256 = (
    "e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c"
)
EXPECTED_INITIAL_SCRIPT_SHA256 = (
    "ffab3102c5afbab0a0ddbf1e34a3928057bdea2e217addedd259aeb72b306b7a"
)
EXPECTED_RESUME_SCRIPT_SHA256 = (
    "ae7684fc15f70ea39891a4c8ee673f7ceaf532687e7e5a0bc9239ff7d902b7f2"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_eol_attribute(relative_path: str) -> str:
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", relative_path],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip().rsplit(": ", maxsplit=1)[-1]


def test_classify_status_is_mutually_exclusive() -> None:
    assert classify_status(
        source_rows=0,
        qfq_factor_points=0,
        continuity_violations=0,
    ) == ("source_missing", ["source_missing"])
    assert classify_status(
        source_rows=600,
        qfq_factor_points=3,
        continuity_violations=0,
    ) == ("available", [])
    assert classify_status(
        source_rows=100,
        qfq_factor_points=0,
        continuity_violations=2,
    ) == (
        "quarantine",
        [
            "minimum_bars",
            "missing_qfq_factor_points",
            "qfq_continuity",
        ],
    )


def test_membership_evidence_hash_is_order_independent() -> None:
    history = pd.DataFrame(
        [
            {
                "code": "000002",
                "date": date(2020, 2, 28),
                "weight": 1.2,
                "display_name": "万科A",
            },
            {
                "code": "000001",
                "date": date(2020, 1, 31),
                "weight": 1.0,
                "display_name": "平安银行",
            },
            {
                "code": "000001",
                "date": date(2020, 2, 28),
                "weight": 1.1,
                "display_name": "平安银行",
            },
        ]
    )
    first = build_membership_evidence(history)
    second = build_membership_evidence(
        history.sample(frac=1.0, random_state=7)
    )
    assert first == second
    assert first["000001"]["membership_rows"] == 2
    assert first["000001"]["membership_first"] == "2020-01-31"
    assert first["000001"]["membership_last"] == "2020-02-28"


def test_coverage_validator_requires_exact_historical_union() -> None:
    records = [
        {
            "code": "000001",
            "baseline_status": "available",
            "status": "available",
            "retrospective_only": True,
            "point_in_time_eligible": False,
            "sealed_evaluation_eligible": False,
        },
        {
            "code": "000002",
            "baseline_status": "not_exported",
            "status": "source_missing",
            "retrospective_only": True,
            "point_in_time_eligible": False,
            "sealed_evaluation_eligible": False,
        },
    ]
    counts = validate_coverage_records(
        records,
        historical_codes={"000001", "000002"},
        v2_codes={"000001"},
    )
    assert counts == {"available": 1, "source_missing": 1}

    with pytest.raises(RuntimeError, match="精确全集"):
        validate_coverage_records(
            records[:1],
            historical_codes={"000001", "000002"},
            v2_codes={"000001"},
        )


def test_resume_v2_quarantines_600228_invalid_source_record() -> None:
    records = [
        {
            "date": 20260623,
            "code": "600228",
            "open": 8.8,
            "high": 9.0,
            "low": 8.7,
            "close": 8.9,
            "volume": 100,
        },
        {
            "date": 20260624,
            "code": "600228",
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 8.92,
            "volume": 200,
        },
    ]
    audit = RESUME_V2.audit_record_quality(
        "600228",
        records,
        stage="raw",
    )
    assert audit["passed"] is False
    assert audit["violation_count"] == 1
    assert audit["examples"] == [
        {
            "row_number": 1,
            "date": "20260624",
            "stage": "raw",
            "reasons": [
                "nonpositive_or_nonfinite_price",
                "high_below_open_or_close",
            ],
            "values": {
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 8.92,
                "volume": 200.0,
            },
        }
    ]
    status, reasons = RESUME_V2.classify_with_record_audit(
        source_rows=len(records),
        qfq_factor_points=1,
        continuity_violations=0,
        raw_audit=audit,
        adjusted_audit=None,
    )
    assert status == "quarantine"
    assert reasons == ["invalid_source_record"]


def test_resume_v2_append_only_writer_refuses_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    RESUME_V2.write_json_append_only(target, {"version": 1})
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        RESUME_V2.write_json_append_only(target, {"version": 2})


def test_rnd03_release_binding_uses_formal_resume_script() -> None:
    binding = json.loads(RELEASE_BINDING.read_text(encoding="utf-8"))
    assert binding["format"] == (
        "alphamaster_csi300_historical_am_inputs_v3_release_binding_v1"
    )
    assert binding["release_id"] == (
        "20260726_csi300_historical_am_inputs_v3"
    )
    assert binding["immutable_artifact"] == {
        "manifest_relative_path": (
            "20260726_csi300_historical_am_inputs_v3/manifest.json"
        ),
        "manifest_sha256": EXPECTED_V3_MANIFEST_SHA256,
        "mutation_policy": "immutable_do_not_rebuild_or_rewrite",
    }
    assert binding["repository_sources"]["initial_script"] == {
        "relative_path": (
            "scripts/build_csi300_historical_am_inputs_v3.py"
        ),
        "sha256": EXPECTED_INITIAL_SCRIPT_SHA256,
        "artifact_relative_path": (
            "source/build_csi300_historical_am_inputs_v3.py"
        ),
        "applies_new_code_indices": [1, 360],
        "provenance_strength": (
            "available_sidecar_embedded_other_receipts_post_hoc"
        ),
    }
    assert binding["repository_sources"]["resume_v2_script"] == {
        "relative_path": (
            "scripts/build_csi300_historical_am_inputs_v3_resume_v2.py"
        ),
        "sha256": EXPECTED_RESUME_SCRIPT_SHA256,
        "artifact_relative_path": (
            "source/build_csi300_historical_am_inputs_v3_resume_v2.py"
        ),
        "applies_new_code_indices": [361, 649],
        "provenance_strength": "embedded_at_receipt_creation",
    }
    assert binding["binding_scope"] == {
        "artifact_status": "already_published_immutable",
        "repository_action": "publish_exact_source_bytes_only",
        "artifact_rewrite_authorized": False,
    }
    assert _sha256_file(INITIAL_SCRIPT) == EXPECTED_INITIAL_SCRIPT_SHA256
    assert _sha256_file(FORMAL_RESUME_SCRIPT) == (
        EXPECTED_RESUME_SCRIPT_SHA256
    )


def test_rnd03_frozen_extraction_scripts_force_lf_checkout() -> None:
    assert _git_eol_attribute(
        "scripts/build_csi300_historical_am_inputs_v3.py"
    ) == "lf"
    assert _git_eol_attribute(
        "scripts/build_csi300_historical_am_inputs_v3_resume_v2.py"
    ) == "lf"
