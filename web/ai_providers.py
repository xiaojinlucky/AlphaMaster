"""AlphaMaster 训练分析使用的 AI 供应商路由。

只允许四个明确通道：

- Codex：使用本机当前 ChatGPT 订阅登录。
- DeepSeek、Kimi、小米 MiMo：使用各自官方兼容接口和独立 API Key。

本模块不会读取浏览器、IDE、Electron 或其他客户端的会话凭据。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ENV_PATH = PROJECT_ROOT.parent / "env"


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    label: str
    default_model: str
    default_base_url: str
    api_key_env_names: tuple[str, ...]
    model_env_names: tuple[str, ...]
    base_url_env_names: tuple[str, ...]


_API_PROVIDER_SPECS = {
    "deepseek": ProviderSpec(
        provider="deepseek",
        label="DeepSeek",
        default_model=DEEPSEEK_MODEL,
        default_base_url=DEEPSEEK_BASE_URL,
        api_key_env_names=("DEEPSEEK_API_KEY",),
        model_env_names=("DEEPSEEK_MODEL",),
        base_url_env_names=("DEEPSEEK_BASE_URL",),
    ),
    "kimi": ProviderSpec(
        provider="kimi",
        label="Kimi",
        default_model="kimi-k2.6",
        default_base_url="https://api.moonshot.cn/v1",
        api_key_env_names=("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        model_env_names=("KIMI_MODEL", "MOONSHOT_MODEL"),
        base_url_env_names=("KIMI_BASE_URL", "MOONSHOT_BASE_URL"),
    ),
    "mimo": ProviderSpec(
        provider="mimo",
        label="小米 MiMo",
        default_model="mimo-v2.5-pro",
        default_base_url="https://api.xiaomimimo.com/v1",
        api_key_env_names=("MIMO_API_KEY",),
        model_env_names=("MIMO_MODEL",),
        base_url_env_names=("MIMO_BASE_URL",),
    ),
}

PROVIDERS = ("codex", "deepseek", "kimi", "mimo")


@dataclass
class ResolvedProvider:
    provider: str
    model: str
    base_url: str
    api_key: str
    label: str
    needs_user_key: bool = False
    transport: str = "openai_chat"
    thinking: bool = True
    reasoning_effort: str = "high"


def _provider_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    if SHARED_ENV_PATH.is_file():
        values.update(
            {
                str(key).lstrip("\ufeff"): str(value)
                for key, value in dotenv_values(SHARED_ENV_PATH).items()
                if key and value
            }
        )
    for key, value in os.environ.items():
        if value:
            values[key] = value
    return values


def _first_value(values: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = str(values.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _api_provider_defaults(spec: ProviderSpec) -> tuple[str, str, str]:
    values = _provider_environment()
    model = _first_value(values, spec.model_env_names) or spec.default_model
    base_url = (
        _first_value(values, spec.base_url_env_names)
        or spec.default_base_url
    )
    api_key = _first_value(values, spec.api_key_env_names)
    return model, base_url, api_key


def _thinking_capabilities(
    provider: str,
    model: str,
) -> tuple[bool, bool, tuple[str, ...]]:
    model_id = str(model or "").strip().lower()
    if provider == "codex":
        return True, False, ("low", "medium", "high", "xhigh", "max", "ultra")
    if provider == "deepseek":
        return True, True, ("high", "max")
    if provider == "kimi":
        if model_id.startswith("kimi-k3"):
            return True, False, ("max",)
        if model_id.startswith("kimi-k2.7-code"):
            return True, False, ()
        return True, True, ()
    if provider == "mimo":
        return True, True, ()
    return False, True, ()


def provider_status() -> dict[str, Any]:
    """返回固定白名单内的供应商状态，不读取第三方客户端凭据。"""
    from web.codex_subscription import codex_status

    codex = codex_status()
    providers: list[dict[str, Any]] = [
        {
            "id": "codex",
            "label": "Codex 订阅（ChatGPT 登录）",
            "available": codex.logged_in,
            "configured": codex.logged_in,
            "needs_user_key": False,
            "model": "auto",
            "supports_thinking_on": True,
            "supports_thinking_off": False,
            "supported_efforts": list(
                _thinking_capabilities("codex", "auto")[2]
            ),
            "hint": codex.message,
        }
    ]
    for provider_id, spec in _API_PROVIDER_SPECS.items():
        model, base_url, api_key = _api_provider_defaults(spec)
        supports_on, supports_off, efforts = _thinking_capabilities(
            provider_id,
            model,
        )
        providers.append(
            {
                "id": provider_id,
                "label": f"{spec.label} ({model})",
                "available": bool(api_key),
                "configured": bool(api_key),
                "needs_user_key": not bool(api_key),
                "model": model,
                "base_url": base_url,
                "supports_thinking_on": supports_on,
                "supports_thinking_off": supports_off,
                "supported_efforts": list(efforts),
                "hint": (
                    f"已从 Quant/env 读取配置 · {base_url}"
                    if api_key
                    else f"需要 API Key · {base_url}"
                ),
            }
        )
    return {"providers": providers}


def resolve_provider(
    provider: str,
    api_key: str | None = None,
    *,
    model: str | None = None,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> ResolvedProvider:
    pid = (provider or "deepseek").strip().lower()
    key = (api_key or "").strip()
    key_lower = key.lower()
    if (
        key_lower in {"openclaw", "openclaw_wb"}
        or key_lower.startswith("openclaw/")
        or key_lower.startswith("openclaw_wb/")
    ):
        raise ValueError("已停用第三方客户端 Key 别名，请选择四个正式 AI 通道。")

    if pid not in PROVIDERS:
        raise ValueError(
            f"不支持的 AI 通道: {provider}（可选: {', '.join(PROVIDERS)}）"
        )

    if pid == "codex":
        from web.codex_subscription import codex_status

        status = codex_status()
        if not status.installed:
            raise ValueError("未安装官方 Codex CLI。")
        if not status.logged_in:
            raise ValueError("Codex CLI 尚未登录 ChatGPT，请先运行 codex login。")
        _, _, efforts = _thinking_capabilities("codex", model or "auto")
        effort = str(reasoning_effort or "high").strip().lower()
        if effort not in efforts:
            effort = "high"
        return ResolvedProvider(
            provider="codex",
            model=str(model or "auto").strip() or "auto",
            base_url="",
            api_key="",
            label="Codex 订阅",
            needs_user_key=False,
            transport="codex_cli",
            thinking=True,
            reasoning_effort=effort,
        )

    spec = _API_PROVIDER_SPECS[pid]
    default_model, base_url, env_key = _api_provider_defaults(spec)
    key = key or env_key
    if not key:
        raise ValueError(
            f"未找到 {spec.label} API Key；请在界面填写，"
            f"或在 Quant/env 配置 {spec.api_key_env_names[0]}。"
        )

    selected_model = str(model or default_model).strip() or default_model
    supports_on, supports_off, efforts = _thinking_capabilities(
        pid,
        selected_model,
    )
    selected_thinking = True if thinking is None else bool(thinking)
    if supports_on and not supports_off:
        selected_thinking = True
    effort = str(reasoning_effort or "high").strip().lower()
    if efforts and effort not in efforts:
        effort = "high" if "high" in efforts else efforts[0]
    if not efforts:
        effort = ""

    return ResolvedProvider(
        provider=pid,
        model=selected_model,
        base_url=base_url,
        api_key=key,
        label=spec.label,
        needs_user_key=not bool(env_key),
        transport="openai_chat",
        thinking=selected_thinking,
        reasoning_effort=effort,
    )


def chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> str:
    """调用供应商并返回完整回答。"""
    content = "".join(
        stream_chat_completions(
            resolved,
            messages,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    ).strip()
    if not content:
        raise RuntimeError("AI 返回内容为空")
    return content


def stream_chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 180.0,
):
    """输出 Codex 最终正文或兼容接口的流式正文片段。"""
    if resolved.transport == "codex_cli":
        from web.codex_subscription import run_codex_subscription

        yield run_codex_subscription(
            messages,
            model=resolved.model,
            reasoning_effort=resolved.reasoning_effort,
            timeout=timeout,
        )
        return

    import urllib.error
    import urllib.request

    url = resolved.base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": resolved.model,
        "messages": messages,
        "stream": True,
    }
    if resolved.provider == "mimo":
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    if resolved.provider == "deepseek":
        payload["thinking"] = {
            "type": "enabled" if resolved.thinking else "disabled"
        }
        if resolved.thinking and resolved.reasoning_effort:
            payload["reasoning_effort"] = resolved.reasoning_effort
    elif resolved.provider == "kimi":
        model_id = resolved.model.lower()
        if model_id.startswith("kimi-k3"):
            payload["reasoning_effort"] = "max"
        elif not model_id.startswith("kimi-k2.7-code"):
            payload["thinking"] = {
                "type": "enabled" if resolved.thinking else "disabled"
            }
    elif resolved.provider == "mimo":
        payload["thinking"] = {
            "type": "enabled" if resolved.thinking else "disabled"
        }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {resolved.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "AlphaMaster-AI-Analyze",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:800]
        if resolved.api_key:
            error_body = error_body.replace(resolved.api_key, "***")
        raise RuntimeError(
            f"AI 请求失败 HTTP {exc.code}: {error_body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"AI 请求失败: {exc}") from exc
