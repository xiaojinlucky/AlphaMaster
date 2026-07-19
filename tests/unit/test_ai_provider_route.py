"""AI 分析接口保存选择但不把共享 env 密钥写入设置。"""

from __future__ import annotations

from fastapi.testclient import TestClient

import web.ai_analyze as ai_analyze
import web.app as app_module

BASE_URL = "http://127.0.0.1:8765"
ORIGIN_HEADERS = {"Origin": BASE_URL}


def _control_headers(client: TestClient) -> dict[str, str]:
    session = client.get("/api/session", headers=ORIGIN_HEADERS)
    return {
        **ORIGIN_HEADERS,
        "X-AlphaMaster-Control": session.json()["control_token"],
    }


def test_analyze_route_forwards_provider_controls_without_env_secret(
    monkeypatch,
) -> None:
    settings = {
        "ai_provider": "deepseek",
        "ai_api_key": "saved-deepseek-key",
        "ai_api_key_provider": "deepseek",
        "ai_model": "deepseek-v4-pro",
        "ai_thinking": True,
        "ai_reasoning_effort": "high",
    }
    saved: dict = {}
    forwarded: dict = {}
    monkeypatch.setattr(app_module, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda payload: saved.update(payload) or {**settings, **payload},
    )

    def fake_stream(**kwargs):
        forwarded.update(kwargs)
        yield {"type": "done", "answer": "OK"}

    monkeypatch.setattr(ai_analyze, "analyze_training_stream", fake_stream)
    client = TestClient(app_module.app, base_url=BASE_URL)

    response = client.post(
        "/api/ai/analyze-training",
        headers={
            **_control_headers(client),
            "Content-Type": "application/json",
        },
        json={
            "provider": "kimi",
            "api_key": None,
            "model": "kimi-k2.6",
            "thinking": False,
            "reasoning_effort": "high",
            "symbol": "XAUUSD",
        },
    )

    assert response.status_code == 200
    assert forwarded["provider"] == "kimi"
    assert forwarded["api_key"] is None
    assert forwarded["model"] == "kimi-k2.6"
    assert forwarded["thinking"] is False
    assert "ai_api_key" not in saved
    assert "saved-deepseek-key" not in response.text


def test_analyze_route_does_not_save_failed_provider_selection(
    monkeypatch,
) -> None:
    settings = {
        "ai_provider": "deepseek",
        "ai_api_key": "",
        "ai_api_key_provider": "",
        "ai_model": "",
        "ai_thinking": True,
        "ai_reasoning_effort": "high",
    }
    saved: dict = {}
    monkeypatch.setattr(app_module, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda payload: saved.update(payload) or {**settings, **payload},
    )
    monkeypatch.setattr(
        ai_analyze,
        "analyze_training_stream",
        lambda **_kwargs: iter(
            [{"type": "error", "message": "不支持的 AI 通道"}]
        ),
    )
    client = TestClient(app_module.app, base_url=BASE_URL)

    response = client.post(
        "/api/ai/analyze-training",
        headers={
            **_control_headers(client),
            "Content-Type": "application/json",
        },
        json={
            "provider": "openclaw",
            "api_key": "openclaw",
            "symbol": "XAUUSD",
        },
    )

    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert saved == {}
