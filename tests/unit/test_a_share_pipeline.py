from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import web.a_share_pipeline as pipeline_module
from web.a_share_pipeline import ASharePipelineManager

RUN_ID = "run_20260723T151419Z_bdc5e5a0"
DATA_HASH = "a" * 64


def _write_fake_backtest(tmp_path: Path) -> dict:
    report_path = (
        tmp_path
        / "local_runs"
        / RUN_ID
        / "postprocess"
        / "backtest_output"
        / "multi_factor_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_mode": "replay",
        "symbol": "600519",
        "timeframe": "D1",
        "data_sha256": DATA_HASH,
        "portfolio": {"total_return": 1.2, "sharpe": 0.5},
        "symbols": {"600519": {"n_trades": 10}},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {
        **report,
        "report_path": str(report_path.relative_to(tmp_path)).replace("\\", "/"),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }


def _write_fake_signal(tmp_path: Path) -> dict:
    output_path = (
        tmp_path
        / "local_runs"
        / RUN_ID
        / "postprocess"
        / "signal_simulation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "alphamaster_signal_simulation_v3",
        "run_id": RUN_ID,
        "symbol": "600519",
        "timeframe": "1d",
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "lifecycle_event": {
            "action": "BUY",
            "previous_exposure": 0.0,
            "resulting_exposure": 0.6,
        },
        "output_path": str(output_path.relative_to(tmp_path)).replace("\\", "/"),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    strategy_run_id: str | None = RUN_ID,
) -> tuple[ASharePipelineManager, dict]:
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    runs = tmp_path / "local_runs"
    run_dir = runs / RUN_ID
    run_dir.mkdir(parents=True)
    artifact_root = run_dir / "artifacts"
    strategy_dir = artifact_root / "strategies"
    strategy_dir.mkdir(parents=True)
    data_file = tmp_path / "600519_D1.parquet"
    data_file.write_bytes(b"parquet")
    manifest = {
        "run_id": RUN_ID,
        "symbol": "600519",
        "timeframe": "D1",
        "local_source": "ashare_akshare_sina_hfq",
        "data_sha256": DATA_HASH,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    strategy_file = strategy_dir / "best_600519.json"
    strategy = {
        "symbol": "600519",
        "timeframe": "D1",
        "data_sha256": DATA_HASH,
        "best_score": 0.8,
        "formula": [1, 2],
        "vocab_version": "unused-in-mocked-signal",
    }
    if strategy_run_id is not None:
        strategy["run_id"] = strategy_run_id
    strategy_file.write_text(json.dumps(strategy), encoding="utf-8")
    result_manifest_sha256 = "b" * 64
    monkeypatch.setattr(
        pipeline_module,
        "get_published_bundle",
        lambda _symbol: {
            "run_id": RUN_ID,
            "symbol": "600519",
            "timeframe": "D1",
            "data_sha256": DATA_HASH,
            "data_file": str(data_file),
            "result_manifest_sha256": result_manifest_sha256,
            "artifact_root_path": artifact_root,
            "strategy_path": strategy_file,
        },
    )
    training = {
        "active": False,
        "job": {
            "run_id": RUN_ID,
            "remote_state": "READY",
            "slurm_job_id": "568306",
            "compute_node": "cu16",
            "elapsed": "00:01:25",
            "allocated_cpus": 12,
            "max_rss": "1254524K",
            "total_cpu": "10:09.454",
            "data_file": str(data_file),
            "result_manifest_sha256": result_manifest_sha256,
        },
    }
    return ASharePipelineManager(local_runs_root=runs), training


def test_ready_training_runs_backtest_then_signal_and_persists_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    calls: list[str] = []

    def fake_backtest(**_kwargs):
        calls.append("backtest")
        return _write_fake_backtest(tmp_path)

    def fake_signal(**_kwargs):
        calls.append("signal")
        return _write_fake_signal(tmp_path)

    monkeypatch.setattr(manager, "_run_backtest", fake_backtest)
    monkeypatch.setattr(manager, "_run_signal", fake_signal)

    manager.observe(training)
    state = manager.wait(RUN_ID, timeout=5)

    assert state is not None
    assert state["status"] == "READY"
    assert state["stages"]["training"]["backend"] == "slurm"
    assert state["stages"]["training"]["allocated_cpus"] == 12
    assert state["stages"]["backtest"]["evaluation_mode"] == "replay"
    assert state["stages"]["signal"]["lifecycle_event"]["action"] == "BUY"
    assert calls == ["backtest", "signal"]


def test_published_strategy_must_match_current_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(
        tmp_path,
        monkeypatch,
        strategy_run_id="run_20260723T151500Z_deadbeef",
    )

    manager.observe(training)
    state = manager.wait(RUN_ID, timeout=5)

    assert state is not None
    assert state["status"] == "FAILED"
    assert "run_id 不一致" in state["error"]
    assert state["stages"]["backtest"]["status"] == "PENDING"


def test_published_strategy_without_run_id_uses_verified_bundle_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(
        tmp_path,
        monkeypatch,
        strategy_run_id=None,
    )
    monkeypatch.setattr(
        manager,
        "_run_backtest",
        lambda **_kwargs: _write_fake_backtest(tmp_path),
    )
    monkeypatch.setattr(
        manager,
        "_run_signal",
        lambda **_kwargs: _write_fake_signal(tmp_path),
    )

    manager.observe(training)
    state = manager.wait(RUN_ID, timeout=5)

    assert state is not None
    assert state["status"] == "READY"
    strategy_snapshot = (
        tmp_path
        / "local_runs"
        / RUN_ID
        / "postprocess"
        / "published_strategy.json"
    )
    payload = json.loads(strategy_snapshot.read_text(encoding="utf-8"))
    assert payload["run_id"] == RUN_ID


def test_ready_training_does_not_restart_permanent_pipeline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(
        tmp_path,
        monkeypatch,
        strategy_run_id="run_20260723T151500Z_deadbeef",
    )
    training["job"]["retry_count"] = 1

    manager.observe(training)
    first = manager.wait(RUN_ID, timeout=5)
    assert first is not None
    assert first["status"] == "FAILED"
    attempts = first["attempts"]

    second = manager.observe(training)

    assert second is not None
    assert second["status"] == "FAILED"
    assert second["attempts"] == attempts


def test_failed_pipeline_requires_exact_error_for_explicit_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(
        tmp_path,
        monkeypatch,
        strategy_run_id="run_20260723T151500Z_deadbeef",
    )
    manager.observe(training)
    failed = manager.wait(RUN_ID, timeout=5)
    assert failed is not None
    assert failed["status"] == "FAILED"

    with pytest.raises(RuntimeError, match="预期错误不一致"):
        manager.recover_failed(RUN_ID, expected_error="另一个错误")

    recovered = manager.recover_failed(
        RUN_ID,
        expected_error="发布策略与当前训练 run_id 不一致",
    )

    assert recovered["status"] == "WAITING_TRAINING"
    assert recovered["error"] is None
    assert recovered["recovery_count"] == 1
    assert recovered["last_recovery"]["previous_error"] == (
        "发布策略与当前训练 run_id 不一致"
    )


def test_non_ashare_training_is_not_claimed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    manifest_path = tmp_path / "local_runs" / RUN_ID / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_source"] = "okx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert manager.observe(training) is None


def test_d1_generic_ashare_source_is_not_claimed_as_sina_hfq(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    manifest_path = tmp_path / "local_runs" / RUN_ID / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_source"] = "ashare_local"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert manager.observe(training) is None


def test_akshare_hfq_training_is_claimed_as_ashare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    manifest_path = tmp_path / "local_runs" / RUN_ID / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_source"] = "ashare_akshare_sina_hfq"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    training["job"]["remote_state"] = "FAILED"
    training["job"]["error"] = "expected-test-stop"

    state = manager.observe(training)

    assert state is not None
    assert state["status"] == "FAILED"
    assert state["stages"]["training"]["error"] == "expected-test-stop"


def test_training_failure_reason_is_exposed_in_pipeline_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    training["job"]["remote_state"] = "FAILED"
    training["job"]["error"] = "Slurm OUT_OF_MEMORY"

    state = manager.observe(training)

    assert state is not None
    assert state["status"] == "FAILED"
    assert state["stages"]["training"]["error"] == "Slurm OUT_OF_MEMORY"


def test_pre_submission_retry_clears_stale_pipeline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    training["job"]["remote_state"] = "FAILED"
    training["job"]["error"] = "manifest local_source 不受支持"
    failed = manager.observe(training)
    assert failed is not None
    assert failed["status"] == "FAILED"

    training["active"] = True
    training["job"].update(
        {
            "remote_state": "PENDING",
            "slurm_job_id": "568571",
            "error": None,
            "retry_count": 1,
        }
    )
    recovered = manager.observe(training)

    assert recovered is not None
    assert recovered["status"] == "WAITING_TRAINING"
    assert recovered["error"] is None
    assert recovered["stages"]["training"]["status"] == "RUNNING"


def test_missing_or_tampered_published_bundle_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_module, "get_published_bundle", lambda _symbol: None)

    manager.observe(training)
    state = manager.wait(RUN_ID, timeout=5)

    assert state is not None
    assert state["status"] == "FAILED"
    assert "产物哈希校验失败" in state["error"]


def test_transient_signal_failure_retries_without_repeating_ready_backtest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    manager.retry_delay_seconds = 0.0
    calls: list[str] = []

    def fake_backtest(**_kwargs):
        calls.append("backtest")
        return _write_fake_backtest(tmp_path)

    def fake_signal(**_kwargs):
        calls.append("signal")
        if calls.count("signal") == 1:
            raise pipeline_module.PipelineTransientError("通达信暂时不可用")
        return _write_fake_signal(tmp_path)

    monkeypatch.setattr(manager, "_run_backtest", fake_backtest)
    monkeypatch.setattr(manager, "_run_signal", fake_signal)

    manager.observe(training)
    first = manager.wait(RUN_ID, timeout=5)
    assert first is not None
    assert first["status"] == "RETRY_WAIT"

    manager.observe(training)
    second = manager.wait(RUN_ID, timeout=5)

    assert second is not None
    assert second["status"] == "READY"
    assert second["attempts"] == 2
    assert calls == ["backtest", "signal", "signal"]


def test_forged_ready_state_cannot_skip_backtest_and_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, training = _fixture(tmp_path, monkeypatch)
    state_path = (
        tmp_path / "local_runs" / RUN_ID / "pipeline_state.json"
    )
    state_path.write_text(
        json.dumps(
            {
                "format": pipeline_module.PIPELINE_FORMAT,
                "run_id": RUN_ID,
                "status": "READY",
            }
        ),
        encoding="utf-8",
    )

    state = manager.observe(training)

    assert state is not None
    assert state["status"] == "FAILED"
    assert state["state_integrity_error"] is True
    assert "完整性校验失败" in state["error"]


def test_two_managers_share_one_postprocess_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, training = _fixture(tmp_path, monkeypatch)
    second = ASharePipelineManager(
        local_runs_root=tmp_path / "local_runs"
    )
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def fake_backtest(**_kwargs):
        calls.append("backtest")
        entered.set()
        assert release.wait(timeout=5)
        return _write_fake_backtest(tmp_path)

    def fake_signal(**_kwargs):
        calls.append("signal")
        return _write_fake_signal(tmp_path)

    for manager in (first, second):
        monkeypatch.setattr(manager, "_run_backtest", fake_backtest)
        monkeypatch.setattr(manager, "_run_signal", fake_signal)

    first.observe(training)
    assert entered.wait(timeout=5)
    second_state = second.observe(training)
    assert second_state is not None
    assert second_state["status"] == "POSTPROCESSING"
    release.set()

    assert first.wait(RUN_ID, timeout=5)["status"] == "READY"
    assert second.wait(RUN_ID, timeout=5)["status"] == "READY"
    assert calls == ["backtest", "signal"]


def test_run_lease_is_exclusive_across_processes(tmp_path: Path) -> None:
    """2026-07-27 环境类修复：原 timeout=20 秒过紧——子进程需冷启动 Python 并
    import 整条 web.a_share_pipeline 依赖链（空载实测 ~2.7s，机器被训练批次/
    Web 控制台占满时实测可超 20s，触发 TimeoutExpired 闪断）。超时只是防挂死
    护栏，放宽到 120s；互斥断言（第二进程抢锁必须返回 False）强度不变。
    """
    lock_path = tmp_path / "pipeline.lock"
    lease = pipeline_module._RunLease(lock_path)
    assert lease.acquire() is True
    code = (
        "from pathlib import Path;"
        "from web.a_share_pipeline import _RunLease;"
        f"lease=_RunLease(Path({str(lock_path)!r}));"
        "print(lease.acquire())"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    finally:
        lease.release()

    assert result.stdout.strip() == "False"
