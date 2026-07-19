"""AI 供应商最小前端合同。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ai_panel_exposes_provider_model_thinking_and_effort_controls() -> None:
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "aiProviderSelect",
        "aiModelInput",
        "aiThinkingSelect",
        "aiEffortSelect",
        "aiApiKeyInput",
    ):
        assert f'id="{element_id}"' in html
    for provider in ("codex", "deepseek", "kimi", "mimo"):
        assert f'value="{provider}"' in html
    assert 'value="openclaw"' not in html
    assert 'value="openclaw_wb"' not in html


def test_ai_request_sends_provider_specific_fields() -> None:
    script = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "provider: resolved.provider" in script
    assert "model: model || null" in script
    assert "thinking," in script
    assert "reasoning_effort: reasoningEffort" in script


def test_saved_key_is_bound_to_its_provider() -> None:
    script = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "window.__aiStoredKeyProvider" in script
    assert "function hasStoredAiKeyFor(provider)" in script
    assert "!hasStoredAiKeyFor(resolved.provider)" in script
    assert "window.__aiHasStoredKey" not in script
