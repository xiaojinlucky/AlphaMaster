"""Codex、DeepSeek、Kimi、MiMo 供应商路由测试。"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from web.ai_providers import (
    ResolvedProvider,
    _provider_environment,
    resolve_provider,
    stream_chat_completions,
)
from web.codex_subscription import CodexStatus


def test_shared_env_utf8_bom_does_not_hide_first_key(
    tmp_path,
    monkeypatch,
) -> None:
    env_path = tmp_path / "env"
    env_path.write_text("\ufeffDEEPSEEK_API_KEY=secret\n", encoding="utf-8")
    monkeypatch.setattr("web.ai_providers.SHARED_ENV_PATH", env_path)

    values = _provider_environment()

    assert values["DEEPSEEK_API_KEY"] == "secret"


def test_deepseek_resolves_shared_env_and_reasoning() -> None:
    with patch(
        "web.ai_providers._provider_environment",
        return_value={
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        },
    ):
        resolved = resolve_provider(
            "deepseek",
            thinking=True,
            reasoning_effort="max",
        )

    assert resolved.model == "deepseek-v4-pro"
    assert resolved.api_key == "deepseek-secret"
    assert resolved.thinking is True
    assert resolved.reasoning_effort == "max"


def test_explicit_api_key_and_model_override_shared_defaults() -> None:
    with patch(
        "web.ai_providers._provider_environment",
        return_value={
            "MOONSHOT_API_KEY": "env-secret",
            "MOONSHOT_MODEL": "kimi-k2.6",
        },
    ):
        resolved = resolve_provider(
            "kimi",
            "explicit-secret",
            model="kimi-k3",
            thinking=False,
            reasoning_effort="low",
        )

    assert resolved.api_key == "explicit-secret"
    assert resolved.model == "kimi-k3"
    assert resolved.thinking is True
    assert resolved.reasoning_effort == "max"


def test_codex_requires_official_login_and_forces_thinking() -> None:
    with patch(
        "web.codex_subscription.codex_status",
        return_value=CodexStatus(True, True, "ok"),
    ):
        resolved = resolve_provider(
            "codex",
            model="auto",
            thinking=False,
            reasoning_effort="xhigh",
        )

    assert resolved.transport == "codex_cli"
    assert resolved.thinking is True
    assert resolved.reasoning_effort == "xhigh"


class _FakeStream:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        content = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "hidden reasoning",
                        "content": "answer",
                    }
                }
            ]
        }
        return iter(
            [
                f"data: {json.dumps(content)}\n".encode(),
                b"data: [DONE]\n",
            ]
        )


@pytest.mark.parametrize(
    ("provider", "model", "expected_field"),
    [
        ("deepseek", "deepseek-v4-pro", "thinking"),
        ("kimi", "kimi-k2.6", "thinking"),
        ("mimo", "mimo-v2.5-pro", "max_completion_tokens"),
    ],
)
def test_api_stream_payload_matches_provider(
    provider: str,
    model: str,
    expected_field: str,
) -> None:
    resolved = ResolvedProvider(
        provider=provider,
        model=model,
        base_url="https://provider.example/v1",
        api_key="secret",
        label=provider,
        thinking=True,
        reasoning_effort="high",
    )
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _FakeStream()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        chunks = list(
            stream_chat_completions(
                resolved,
                [{"role": "user", "content": "hello"}],
            )
        )

    assert chunks == ["answer"]
    assert expected_field in captured["payload"]
    assert "hidden reasoning" not in "".join(chunks)


def test_codex_stream_uses_subscription_runner() -> None:
    resolved = ResolvedProvider(
        provider="codex",
        model="auto",
        base_url="",
        api_key="",
        label="Codex",
        transport="codex_cli",
        reasoning_effort="high",
    )
    with patch(
        "web.codex_subscription.run_codex_subscription",
        return_value="CODEX_OK",
    ) as run:
        chunks = list(
            stream_chat_completions(
                resolved,
                [{"role": "user", "content": "hello"}],
            )
        )

    assert chunks == ["CODEX_OK"]
    run.assert_called_once()
