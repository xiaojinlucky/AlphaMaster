"""Web 本机控制边界与敏感响应的回归测试。"""
from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.app as app_module
from run_web import _is_alphamaster_health, _validate_bind_host


BASE_URL = "http://127.0.0.1:8765"
ORIGIN_HEADERS = {"Origin": BASE_URL}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app_module.app, base_url=BASE_URL, raise_server_exceptions=False)


def _control_headers(client: TestClient) -> dict[str, str]:
    response = client.get("/api/session", headers=ORIGIN_HEADERS)
    assert response.status_code == 200
    token = response.json()["control_token"]
    assert isinstance(token, str) and len(token) >= 32
    return {**ORIGIN_HEADERS, "X-AlphaMaster-Control": token}


def test_background_monitor_advances_pipeline_without_browser_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = {"active": False, "job": {"run_id": "run_20260723T151419Z_bdc5e5a0"}}
    observed: list[dict] = []
    batch_calls: list[dict] = []
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: training,
    )
    monkeypatch.setattr(
        app_module.a_share_pipeline_manager,
        "observe",
        lambda payload: observed.append(payload) or {"status": "READY"},
    )
    monkeypatch.setattr(
        app_module.training_batch_controller,
        "advance_once",
        lambda **kwargs: batch_calls.append(kwargs) or {},
    )

    result = app_module._advance_a_share_pipeline_once()

    assert result == {"status": "READY"}
    assert observed == [training]
    assert batch_calls == [
        {
            "training": training,
            "pipeline": {"status": "READY"},
        }
    ]


@pytest.mark.parametrize(
    ("training", "expected_final"),
    [
        (
            {
                "active": True,
                "job": {
                    "run_id": "run_20260723T151419Z_bdc5e5a0",
                    "slurm_job_id": "568306",
                    "remote_state": "RUNNING",
                },
            },
            False,
        ),
        (
            {
                "active": False,
                "job": {
                    "run_id": "run_20260723T151419Z_bdc5e5a0",
                    "slurm_job_id": "568306",
                    "remote_state": "FAILED",
                },
            },
            True,
        ),
    ],
)
def test_background_monitor_refreshes_exact_run_log(
    monkeypatch: pytest.MonkeyPatch,
    training: dict,
    expected_final: bool,
) -> None:
    log_calls: list[dict] = []
    monkeypatch.setattr(
        app_module.training_manager,
        "tail_log",
        lambda lines, **kwargs: log_calls.append(
            {"lines": lines, **kwargs}
        )
        or [],
    )
    app_module._refresh_training_log_once(training)

    assert log_calls == [
        {
            "lines": 200,
            "expected_run_id": "run_20260723T151419Z_bdc5e5a0",
            "expected_job_id": "568306",
            **({"final": True} if expected_final else {}),
        }
    ]


def test_log_refresh_scheduler_returns_while_remote_tail_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_refresh(_training: dict) -> None:
        started.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(
        app_module,
        "_refresh_training_log_once",
        blocking_refresh,
    )
    with app_module._TRAINING_LOG_REFRESH_LOCK:
        app_module._TRAINING_LOG_REFRESH_THREAD = None

    app_module._schedule_training_log_refresh(
        {"active": True, "job": {"run_id": "run_test"}}
    )

    assert started.wait(timeout=2)
    thread = app_module._TRAINING_LOG_REFRESH_THREAD
    assert thread is not None and thread.is_alive()
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_training_stop_is_locked_and_bound_to_requested_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingLock:
        entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_args):
            self.entered = False

    lock = RecordingLock()
    status_lock_states: list[bool] = []
    stop_calls: list[dict] = []
    training = {
        "active": True,
        "job": {
            "run_id": "run_20260723T151419Z_bdc5e5a0",
            "slurm_job_id": "568306",
        },
    }

    def fake_status() -> dict:
        status_lock_states.append(lock.entered)
        return training

    def fake_stop(**kwargs) -> bool:
        assert lock.entered
        stop_calls.append(kwargs)
        return True

    monkeypatch.setattr(
        app_module.training_batch_controller,
        "admission_lock",
        lock,
    )
    monkeypatch.setattr(app_module.training_manager, "status", fake_status)
    monkeypatch.setattr(app_module.training_manager, "stop", fake_stop)

    result = app_module.api_training_stop(
        app_module.StopTrainingRequest(
            run_id="run_20260723T151419Z_bdc5e5a0",
            slurm_job_id="568306",
        )
    )

    assert result["ok"] is True
    assert stop_calls == [
        {
            "expected_run_id": "run_20260723T151419Z_bdc5e5a0",
            "expected_job_id": "568306",
        }
    ]
    assert status_lock_states == [True, False]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"run_id": "run_20260723T151419Z_bdc5e5a0"},
        {"slurm_job_id": "568306"},
        {"run_id": "", "slurm_job_id": "568306"},
        {
            "run_id": "run_20260723T151419Z_bdc5e5a0",
            "slurm_job_id": "   ",
        },
    ],
)
def test_training_stop_requires_both_displayed_identity_fields(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict | None,
) -> None:
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: pytest.fail("身份字段无效时不得读取当前训练作业"),
    )
    request_kwargs = {} if payload is None else {"json": payload}

    response = client.post(
        "/api/training/stop",
        headers=_control_headers(client),
        **request_kwargs,
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("requested_run_id", "requested_job_id"),
    [
        ("run_20260723T151419Z_old00000", "568740"),
        ("run_20260723T151419Z_f4a3f24c", "568306"),
    ],
)
def test_training_stop_rejects_stale_displayed_identity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    requested_run_id: str,
    requested_job_id: str,
) -> None:
    stop_calls: list[dict] = []
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: {
            "active": True,
            "job": {
                "run_id": "run_20260723T151419Z_f4a3f24c",
                "slurm_job_id": "568740",
            },
        },
    )
    monkeypatch.setattr(
        app_module.training_manager,
        "stop",
        lambda **kwargs: stop_calls.append(kwargs) or True,
    )

    response = client.post(
        "/api/training/stop",
        headers=_control_headers(client),
        json={
            "run_id": requested_run_id,
            "slurm_job_id": requested_job_id,
        },
    )

    assert response.status_code == 409
    assert "已变化" in response.json()["detail"]
    assert stop_calls == []


def test_training_stop_frontend_sends_last_displayed_identity() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "web" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    start = script.index("async function stopTraining()")
    end = script.index("\nfunction switchPage(", start)
    stop_source = script[start:end]

    assert "const job = lastTrainingSnapshot?.job || {};" in stop_source
    assert "run_id: expectedRunId" in stop_source
    assert "slurm_job_id: expectedJobId" in stop_source
    assert 'headers: { "Content-Type": "application/json" }' in stop_source
    assert "body: JSON.stringify({" in stop_source


def test_session_is_loopback_only_and_not_cached(client: TestClient) -> None:
    response = client.get("/api/session", headers=ORIGIN_HEADERS)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in response.headers

    bad_host = client.get(
        "/api/session",
        headers={"Host": "attacker.example:8765"},
    )
    assert bad_host.status_code == 403

    bad_origin = client.get(
        "/api/session",
        headers={"Origin": "http://127.0.0.1:9999"},
    )
    assert bad_origin.status_code == 403


def test_all_write_requests_require_control_token(client: TestClient) -> None:
    missing = client.post("/api/training/import")
    assert missing.status_code == 403
    assert "控制令牌" in missing.json()["detail"]

    disabled = client.post(
        "/api/training/import",
        headers=_control_headers(client),
    )
    assert disabled.status_code == 403
    assert "已禁用" in disabled.json()["detail"]


def test_side_effecting_gets_also_require_control_token(client: TestClient) -> None:
    response = client.get("/api/data-file/browse", headers=ORIGIN_HEADERS)
    assert response.status_code == 403
    assert "控制令牌" in response.json()["detail"]


def test_sensitive_settings_are_never_returned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = {
        "debug_mode": False,
        "ai_provider": "deepseek",
        "ai_api_key": "ai-secret-value",
        "feishu_enabled": True,
        "feishu_webhook_url": "https://example.invalid/hook-secret",
        "feishu_secret": "feishu-secret-value",
        "last_data_file": "",
        "last_strategy_file": "",
        "bt_commission_pct": 0.02,
        "bt_slippage_pct": 0.01,
    }
    monkeypatch.setattr(app_module, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(app_module, "save_settings", lambda _: dict(settings))

    control_headers = _control_headers(client)
    for path in ("/api/settings", "/api/config", "/api/realtime/feishu"):
        response = client.get(path, headers=control_headers)
        assert response.status_code == 200
        body = response.text
        assert "ai-secret-value" not in body
        assert "hook-secret" not in body
        assert "feishu-secret-value" not in body

    saved = client.put(
        "/api/realtime/feishu",
        headers={**_control_headers(client), "Content-Type": "application/json"},
        json={"enabled": True},
    )
    assert saved.status_code == 200
    assert "hook-secret" not in saved.text
    assert "feishu-secret-value" not in saved.text


def test_internal_errors_do_not_return_traceback_or_exception_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash() -> dict:
        raise RuntimeError("private-exception-detail")

    monkeypatch.setattr(app_module, "load_settings", crash)
    response = client.get("/api/settings", headers=_control_headers(client))
    assert response.status_code == 500
    assert response.json() == {"detail": "服务器内部错误"}
    assert "traceback" not in response.text.lower()
    assert "private-exception-detail" not in response.text


def test_debug_logs_are_disabled_and_feishu_test_is_constrained(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "is_debug_mode", lambda: False)
    logs = client.get("/api/debug/logs", headers=_control_headers(client))
    assert logs.status_code == 403

    sent: list[tuple[str, str | None, str | None]] = []

    def fake_send(
        text: str,
        *,
        webhook_url: str | None = None,
        secret: str | None = None,
    ) -> tuple[bool, str]:
        sent.append((text, webhook_url, secret))
        return True, "ok"

    monkeypatch.setattr(app_module, "send_feishu_text", fake_send)
    feishu = client.post(
        "/api/realtime/feishu/test",
        headers=_control_headers(client),
        json={
            "webhook_url": (
                "https://open.feishu.cn/open-apis/bot/v2/hook/"
                "12345678-1234-1234-1234-123456789abc"
            ),
            "secret": "test-only-secret",
        },
    )
    assert feishu.status_code == 200
    assert feishu.json()["ok"] is True
    assert len(sent) == 1
    assert "test-only-secret" not in feishu.text

    invalid = client.put(
        "/api/realtime/feishu",
        headers={**_control_headers(client), "Content-Type": "application/json"},
        json={"webhook_url": "http://127.0.0.1/internal"},
    )
    assert invalid.status_code == 400
    assert "飞书官方" in invalid.json()["detail"]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_run_web_rejects_non_loopback_bind_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="只允许绑定"):
        _validate_bind_host(host)


def test_shortcut_health_probe_only_accepts_alphamaster() -> None:
    assert _is_alphamaster_health({"status": "ok", "version": "1.1.0"}) is True
    assert _is_alphamaster_health({"status": "ok", "version": "other"}) is False
    assert _is_alphamaster_health("ok") is False


def test_production_torch_loads_are_weights_only() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = (
        root / "model_core" / "engine.py",
        root / "web" / "progress.py",
        root / "web" / "training_package.py",
    )
    found = 0
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "load":
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "torch":
                continue
            found += 1
            weights_only = next(
                (kw.value for kw in node.keywords if kw.arg == "weights_only"), None
            )
            assert isinstance(weights_only, ast.Constant) and weights_only.value is True
    assert found == 3
