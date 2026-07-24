"""Windows OpenSSH Slurm 客户端的边界与完整性测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import web.slurm_training_client as client_module


def _client() -> client_module.SlurmTrainingClient:
    return client_module.SlurmTrainingClient(
        remote_root="/hwdata/home/jinqc/Quant/AlphaMaster",
        selector_script=r"D:\Desktop\codex-remote-tools\check-best-node.ps1",
    )


@pytest.mark.parametrize(
    "value",
    ["../bad", "run_bad", "run_20260714T120000Z_deadbeeg", "run_20260714T120000Z_deadbeef;x"],
)
def test_run_id_rejects_non_contract_values(value: str) -> None:
    with pytest.raises(client_module.SlurmClientError):
        client_module.SlurmTrainingClient.validate_run_id(value)


def test_data_filename_rejects_paths_and_unknown_timeframes() -> None:
    for value in ("../XAUUSD_H1.parquet", "XAUUSD_1H.parquet", "XAUUSD_H1.csv"):
        with pytest.raises(client_module.SlurmClientError):
            client_module.SlurmTrainingClient.validate_data_filename(value)


def test_selector_accepts_only_compute_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    observed: dict[str, object] = {}

    def first_run(command, **_kwargs):
        observed["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout="Recommended now: compute-node-12\n",
            stderr="",
        )

    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        first_run,
    )
    assert client.select_compute_host() == "compute-node-12"
    command = observed["command"]
    assert isinstance(command, list)
    assert Path(command[0]).name.lower() == "pwsh.exe"

    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="Recommended now: login-node\n",
            stderr="",
        ),
    )
    with pytest.raises(client_module.SlurmClientError, match="允许的计算节点"):
        client.select_compute_host()


def test_remote_call_uses_fixed_controller_on_compute_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    monkeypatch.setattr(client, "select_compute_host", lambda: "compute-node-13")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"state":"PENDING"}', stderr="")

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    payload = client.status("run_20260714T120000Z_deadbeef", "12345")
    assert payload["state"] == "PENDING"
    command = observed["command"]
    assert isinstance(command, list)
    assert "compute-node-13" in command
    assert "login-node" not in command
    assert "scripts/slurm_control.py status run_20260714T120000Z_deadbeef 12345" in command[-1]
    assert observed["kwargs"]["shell"] is False


@pytest.mark.parametrize("returncode", [255])
def test_remote_transport_failure_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    client = _client()
    monkeypatch.setattr(client, "select_compute_host", lambda: "compute-node-11")
    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=returncode, stdout="", stderr="connection lost"
        ),
    )
    with pytest.raises(client_module.SlurmTransportError):
        client.status("run_20260714T120000Z_deadbeef", "12345")


def test_remote_validation_failure_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    monkeypatch.setattr(client, "select_compute_host", lambda: "compute-node-11")
    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="invalid run manifest"
        ),
    )
    with pytest.raises(client_module.SlurmClientError) as caught:
        client.status("run_20260714T120000Z_deadbeef", "12345")
    assert not isinstance(caught.value, client_module.SlurmTransportError)


def test_download_rejects_manifest_path_traversal(tmp_path: Path, monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client,
        "_remote_call",
        lambda *_args: {
            "run_id": "run_20260714T120000Z_deadbeef",
            "slurm_job_id": "123",
            "git_commit": "a" * 40,
            "artifacts": [
                {"path": "strategies/../../escape.json", "size": 1, "sha256": "0" * 64}
            ],
        },
    )
    with pytest.raises(client_module.SlurmClientError, match="越界"):
        client.download_result(
            run_id="run_20260714T120000Z_deadbeef",
            job_id="123",
            local_artifact_root=tmp_path,
            expected_commit="a" * 40,
        )


def test_download_stages_and_verifies_artifact(tmp_path: Path, monkeypatch) -> None:
    client = _client()
    content = b"verified-strategy"
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "run_id": "run_20260714T120000Z_deadbeef",
        "slurm_job_id": "123",
        "git_commit": "a" * 40,
        "artifacts": [
            {"path": "strategies/best_XAUUSD.json", "size": len(content), "sha256": digest}
        ],
    }
    monkeypatch.setattr(client, "_remote_call", lambda *_args: manifest)
    monkeypatch.setattr(client, "select_compute_host", lambda: "compute-node-11")

    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(content)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(client_module.subprocess, "run", fake_run)
    returned = client.download_result(
        run_id=manifest["run_id"],
        job_id="123",
        local_artifact_root=tmp_path,
        expected_commit="a" * 40,
    )
    assert returned == manifest
    assert (tmp_path / "strategies" / "best_XAUUSD.json").read_bytes() == content
    assert not list(tmp_path.rglob("*.partial"))


def test_download_scp_failure_is_retryable(tmp_path: Path, monkeypatch) -> None:
    client = _client()
    manifest = {
        "run_id": "run_20260714T120000Z_deadbeef",
        "slurm_job_id": "123",
        "git_commit": "a" * 40,
        "artifacts": [
            {"path": "strategies/best_XAUUSD.json", "size": 1, "sha256": "0" * 64}
        ],
    }
    monkeypatch.setattr(client, "_remote_call", lambda *_args: manifest)
    monkeypatch.setattr(client, "select_compute_host", lambda: "compute-node-11")
    monkeypatch.setattr(
        client_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="connection reset"
        ),
    )
    with pytest.raises(client_module.SlurmTransportError):
        client.download_result(
            run_id=manifest["run_id"],
            job_id="123",
            local_artifact_root=tmp_path,
            expected_commit="a" * 40,
        )


def test_training_history_root_artifact_is_explicitly_allowed() -> None:
    path = client_module.SlurmTrainingClient._validate_artifact_path(
        "training_history_XAUUSD.json"
    )
    assert path.name == "training_history_XAUUSD.json"
