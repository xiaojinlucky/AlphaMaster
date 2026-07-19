"""AlphaMaster Codex 订阅客户端隔离测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from web.codex_subscription import (
    CodexStatus,
    _sanitized_environment,
    codex_status,
    run_codex_subscription,
)


def test_codex_environment_excludes_broker_and_model_secrets(monkeypatch) -> None:
    monkeypatch.setenv("LONGBRIDGE_ACCESS_TOKEN", "broker-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "model-secret")
    monkeypatch.setenv("PATH", "safe")

    env = _sanitized_environment()

    assert env["PATH"] == "safe"
    assert "LONGBRIDGE_ACCESS_TOKEN" not in env
    assert "DEEPSEEK_API_KEY" not in env


def test_codex_status_accepts_chatgpt_subscription() -> None:
    completed = subprocess.CompletedProcess(
        args=["codex", "login", "status"],
        returncode=0,
        stdout="",
        stderr="Logged in using ChatGPT\n",
    )
    with (
        patch("web.codex_subscription._codex_executable", return_value="codex"),
        patch(
            "web.codex_subscription.subprocess.run",
            return_value=completed,
        ),
    ):
        status = codex_status()

    assert status.logged_in is True
    assert "ChatGPT" in status.message


def test_codex_status_rejects_api_key_login() -> None:
    completed = subprocess.CompletedProcess(
        args=["codex", "login", "status"],
        returncode=0,
        stdout="Logged in using API key\n",
        stderr="",
    )
    with (
        patch("web.codex_subscription._codex_executable", return_value="codex"),
        patch(
            "web.codex_subscription.subprocess.run",
            return_value=completed,
        ),
    ):
        status = codex_status()

    assert status.logged_in is False
    assert "不是 ChatGPT 订阅登录" in status.message


def test_codex_command_disables_tools() -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "AM_CODEX_OK"},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {}}),
        ]
    )
    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = (stdout, "")
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return process

    with (
        patch(
            "web.codex_subscription.codex_status",
            return_value=CodexStatus(True, True, "ok"),
        ),
        patch("web.codex_subscription._codex_executable", return_value="codex"),
        patch("web.codex_subscription.subprocess.Popen", side_effect=fake_popen),
    ):
        answer = run_codex_subscription(
            [{"role": "user", "content": "只回复 OK"}],
        )

    assert answer == "AM_CODEX_OK"
    assert "shell_tool" in captured["command"]
    assert "web_search" not in captured["command"]
    assert "--ephemeral" in captured["command"]
    assert "--ignore-user-config" in captured["command"]
    assert "--strict-config" in captured["command"]
    sandbox_index = captured["command"].index("--sandbox")
    assert captured["command"][sandbox_index + 1] == "read-only"
    assert "--skip-git-repo-check" in captured["command"]
    assert 'web_search="disabled"' in captured["command"]
    assert "skills.include_instructions=false" in captured["command"]
    assert "browser_use" in captured["command"]
    assert "computer_use" in captured["command"]
    assert "image_generation" in captured["command"]
    assert "plugins" in captured["command"]
    assert captured["env"].get("DEEPSEEK_API_KEY") is None
    assert "alphamaster-codex-" in Path(captured["cwd"]).name
    assert Path(captured["cwd"]).resolve() != Path.cwd().resolve()
    assert not Path(captured["cwd"]).exists()


def test_codex_command_rejects_forbidden_tool_event() -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "file_change", "path": "unexpected.txt"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "SHOULD_NOT_RETURN"},
                }
            ),
        ]
    )
    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = (stdout, "")

    with (
        patch(
            "web.codex_subscription.codex_status",
            return_value=CodexStatus(True, True, "ok"),
        ),
        patch("web.codex_subscription._codex_executable", return_value="codex"),
        patch(
            "web.codex_subscription.subprocess.Popen",
            return_value=process,
        ),
    ):
        with pytest.raises(RuntimeError, match="被禁止的工具调用"):
            run_codex_subscription(
                [{"role": "user", "content": "只回复 OK"}],
            )


def test_codex_command_timeout_terminates_process() -> None:
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0

    with (
        patch(
            "web.codex_subscription.codex_status",
            return_value=CodexStatus(True, True, "ok"),
        ),
        patch("web.codex_subscription._codex_executable", return_value="codex"),
        patch(
            "web.codex_subscription.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "web.codex_subscription.time.monotonic",
            side_effect=(0.0, 1.0),
        ),
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            run_codex_subscription(
                [{"role": "user", "content": "只回复 OK"}],
                timeout=0.01,
            )

    process.terminate.assert_called_once()
    process.wait.assert_called_once()
