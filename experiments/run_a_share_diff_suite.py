"""在全新输出目录中顺序执行两组 A 股隔离差分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_DIR = REPO_ROOT / "experiments" / "a_share_execution_diff"
RESEARCH_DIR = REPO_ROOT / "experiments" / "a_share_research_layer_diff"

AKQUANT_COMMIT = "30054523fb905adb1c3f250749e1b5ff61cf8452"
RQALPHA_COMMIT = "3503ab57932540cd36bf8375134e52c6923bf0d2"
QLIB_COMMIT = "79633dd9506ea689e5400dea0197717b5b3d74b7"
VNPY_COMMIT = "1b78494979deb4c4996f6b864f234d9839f2f239"

ENVIRONMENT_INVENTORY_SCRIPT = """
import importlib.metadata
import json

inventory = []
for distribution in importlib.metadata.distributions():
    direct_url = distribution.read_text("direct_url.json")
    inventory.append(
        {
            "name": distribution.metadata["Name"],
            "version": distribution.version,
            "direct_url": json.loads(direct_url) if direct_url else None,
        }
    )
inventory.sort(key=lambda item: (item["name"].lower(), item["version"]))
print(json.dumps(inventory, ensure_ascii=True, sort_keys=True))
"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _source_contract() -> tuple[dict[str, str], str]:
    paths = [
        Path(__file__).resolve(),
        *sorted(EXECUTION_DIR.glob("*.py")),
        EXECUTION_DIR / "fixture.json",
        *sorted(RESEARCH_DIR.glob("*.py")),
        RESEARCH_DIR / "fixture.json",
    ]
    hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path)
        for path in paths
    }
    contract_sha256 = _sha256_bytes(
        json.dumps(
            hashes,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    )
    return hashes, contract_sha256


def _environment(
    name: str,
    python: Path,
    output_dir: Path,
) -> dict[str, Any]:
    version = subprocess.run(
        [str(python), "--version"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    inventory = subprocess.run(
        [str(python), "-c", ENVIRONMENT_INVENTORY_SCRIPT],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    version_text = (version.stdout or version.stderr).decode(
        "utf-8",
        errors="replace",
    ).strip()
    inventory_path = output_dir / f"environment_{name}.json"
    inventory_path.write_bytes(inventory.stdout)
    return {
        "python": str(python.resolve()),
        "version": version_text,
        "distribution_inventory": str(inventory_path),
        "distribution_inventory_sha256": _sha256_bytes(
            inventory.stdout
        ),
        "distribution_inventory_bytes": len(inventory.stdout),
        "distribution_count": len(json.loads(inventory.stdout)),
    }


def _run_step(
    name: str,
    command: list[str],
    expected_output: Path,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    finished = datetime.now(timezone.utc)
    stdout_path = expected_output.parent / f"{name}.stdout.log"
    stderr_path = expected_output.parent / f"{name}.stderr.log"
    stdout_path.write_bytes(process.stdout)
    stderr_path.write_bytes(process.stderr)
    result = {
        "name": name,
        "command": command,
        "command_sha256": _sha256_bytes(
            json.dumps(command, ensure_ascii=False).encode("utf-8")
        ),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "returncode": process.returncode,
        "stdout_sha256": _sha256_bytes(process.stdout),
        "stdout_bytes": len(process.stdout),
        "stdout_path": str(stdout_path),
        "stderr_sha256": _sha256_bytes(process.stderr),
        "stderr_bytes": len(process.stderr),
        "stderr_path": str(stderr_path),
        "expected_output": str(expected_output),
    }
    if process.returncode != 0:
        tail = process.stderr.decode("utf-8", errors="replace")[-2000:]
        result["status"] = "FAILED"
        result["error"] = f"{name} 失败:\n{tail}"
        return result
    if not expected_output.is_file():
        result["status"] = "FAILED"
        result["error"] = (
            f"{name} 未生成结果文件: {expected_output}"
        )
        return result
    result["status"] = "PASSED"
    result["output_sha256"] = _sha256_file(expected_output)
    result["output_bytes"] = expected_output.stat().st_size
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--akquant-python", required=True, type=Path)
    parser.add_argument("--akquant-source", required=True, type=Path)
    parser.add_argument("--rqalpha-python", required=True, type=Path)
    parser.add_argument("--rqalpha-source", required=True, type=Path)
    parser.add_argument("--qlib-python", required=True, type=Path)
    parser.add_argument("--qlib-source", required=True, type=Path)
    parser.add_argument("--vnpy-python", required=True, type=Path)
    parser.add_argument("--vnpy-source", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "contract_version": "a-share-diff-suite-v3",
        "run_id": output_dir.name,
        "status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "source_files": {},
        "source_contract_sha256": "",
        "environments": {},
        "steps": [],
    }
    source_files, source_contract_sha256 = _source_contract()
    manifest["source_files"] = source_files
    manifest["source_contract_sha256"] = source_contract_sha256
    _write_manifest(manifest_path, manifest)

    pythons = {
        "alphamaster": Path(sys.executable),
        "akquant": args.akquant_python,
        "rqalpha": args.rqalpha_python,
        "qlib": args.qlib_python,
        "vnpy": args.vnpy_python,
    }
    try:
        manifest["environments"] = {
            name: _environment(name, path, output_dir)
            for name, path in pythons.items()
        }
        execution_fixture = EXECUTION_DIR / "fixture.json"
        research_fixture = RESEARCH_DIR / "fixture.json"
        step_specs = [
            (
                "alphamaster_execution",
                [
                    str(pythons["alphamaster"]),
                    str(EXECUTION_DIR / "run_alphamaster.py"),
                    "--fixture",
                    str(execution_fixture),
                    "--output",
                    str(output_dir / "alphamaster.json"),
                ],
                output_dir / "alphamaster.json",
            ),
            (
                "akquant_execution",
                [
                    str(pythons["akquant"]),
                    str(EXECUTION_DIR / "run_akquant.py"),
                    "--fixture",
                    str(execution_fixture),
                    "--source-root",
                    str(args.akquant_source),
                    "--source-test-commit",
                    AKQUANT_COMMIT,
                    "--output",
                    str(output_dir / "akquant.json"),
                ],
                output_dir / "akquant.json",
            ),
            (
                "rqalpha_execution",
                [
                    str(pythons["rqalpha"]),
                    str(EXECUTION_DIR / "run_rqalpha.py"),
                    "--fixture",
                    str(execution_fixture),
                    "--source-root",
                    str(args.rqalpha_source),
                    "--source-commit",
                    RQALPHA_COMMIT,
                    "--output",
                    str(output_dir / "rqalpha.json"),
                ],
                output_dir / "rqalpha.json",
            ),
            (
                "execution_comparison",
                [
                    str(pythons["alphamaster"]),
                    str(EXECUTION_DIR / "compare_results.py"),
                    "--fixture",
                    str(execution_fixture),
                    "--alphamaster",
                    str(output_dir / "alphamaster.json"),
                    "--akquant",
                    str(output_dir / "akquant.json"),
                    "--rqalpha",
                    str(output_dir / "rqalpha.json"),
                    "--output",
                    str(output_dir / "execution_comparison.json"),
                ],
                output_dir / "execution_comparison.json",
            ),
            (
                "alphamaster_clock",
                [
                    str(pythons["alphamaster"]),
                    str(RESEARCH_DIR / "run_alphamaster_clock.py"),
                    "--fixture",
                    str(research_fixture),
                    "--output",
                    str(output_dir / "alphamaster_clock.json"),
                ],
                output_dir / "alphamaster_clock.json",
            ),
            (
                "qlib_clock",
                [
                    str(pythons["qlib"]),
                    str(RESEARCH_DIR / "inspect_qlib_clock.py"),
                    "--fixture",
                    str(research_fixture),
                    "--source-root",
                    str(args.qlib_source),
                    "--source-commit",
                    QLIB_COMMIT,
                    "--output",
                    str(output_dir / "qlib_clock.json"),
                ],
                output_dir / "qlib_clock.json",
            ),
            (
                "vnpy_constituents",
                [
                    str(pythons["vnpy"]),
                    str(RESEARCH_DIR / "run_vnpy_constituents.py"),
                    "--fixture",
                    str(research_fixture),
                    "--source-root",
                    str(args.vnpy_source),
                    "--source-commit",
                    VNPY_COMMIT,
                    "--output",
                    str(output_dir / "vnpy_constituents.json"),
                ],
                output_dir / "vnpy_constituents.json",
            ),
            (
                "research_comparison",
                [
                    str(pythons["alphamaster"]),
                    str(RESEARCH_DIR / "compare_results.py"),
                    "--fixture",
                    str(research_fixture),
                    "--alphamaster",
                    str(output_dir / "alphamaster_clock.json"),
                    "--qlib",
                    str(output_dir / "qlib_clock.json"),
                    "--vnpy",
                    str(output_dir / "vnpy_constituents.json"),
                    "--output",
                    str(output_dir / "research_comparison.json"),
                ],
                output_dir / "research_comparison.json",
            ),
        ]
        for name, command, expected_output in step_specs:
            step_result = _run_step(name, command, expected_output)
            manifest["steps"].append(step_result)
            _write_manifest(manifest_path, manifest)
            if step_result["status"] != "PASSED":
                raise RuntimeError(step_result["error"])
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        manifest["status"] = "FAILED"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "PASSED"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    print(output_dir / "execution_comparison.json")
    print(output_dir / "research_comparison.json")
    print(manifest_path)


if __name__ == "__main__":
    main()
