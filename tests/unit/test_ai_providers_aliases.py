"""AI 供应商白名单和第三方会话凭据隔离测试。"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import web.ai_providers as providers_module
from web.ai_providers import PROVIDERS, provider_status, resolve_provider
from web.codex_subscription import CodexStatus


def test_only_four_explicit_providers_are_registered() -> None:
    assert PROVIDERS == ("codex", "deepseek", "kimi", "mimo")


@pytest.mark.parametrize("provider", ["openclaw", "openclaw_wb", "workbuddy"])
def test_legacy_third_party_provider_is_rejected(provider: str) -> None:
    with pytest.raises(ValueError, match="不支持的 AI 通道"):
        resolve_provider(provider, "untrusted-session-token")


@pytest.mark.parametrize(
    "alias",
    ["openclaw", "openclaw/main", "openclaw_wb", "openclaw_wb/auto"],
)
def test_legacy_key_alias_is_rejected(alias: str) -> None:
    with pytest.raises(ValueError, match="已停用第三方客户端 Key 别名"):
        resolve_provider("deepseek", alias)


def test_provider_status_does_not_expose_third_party_discovery_paths() -> None:
    with (
        patch(
            "web.codex_subscription.codex_status",
            return_value=CodexStatus(True, True, "ok"),
        ),
        patch("web.ai_providers._provider_environment", return_value={}),
    ):
        status = provider_status()

    assert [row["id"] for row in status["providers"]] == list(PROVIDERS)
    assert all("api_key" not in row for row in status["providers"])
    assert not hasattr(providers_module, "_workbuddy_token")
    assert not hasattr(providers_module, "_extract_electron_token")
    assert not hasattr(providers_module, "detect_qclaw")
