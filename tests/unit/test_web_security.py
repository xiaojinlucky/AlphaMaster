"""Web 本机控制边界与敏感响应的回归测试。"""
from __future__ import annotations

import ast
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


def test_debug_logs_and_feishu_test_are_disabled_by_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "is_debug_mode", lambda: False)
    logs = client.get("/api/debug/logs", headers=_control_headers(client))
    assert logs.status_code == 403

    feishu = client.post(
        "/api/realtime/feishu/test",
        headers=_control_headers(client),
    )
    assert feishu.status_code == 403
    assert "已禁用" in feishu.json()["detail"]


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
