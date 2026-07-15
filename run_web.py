"""
run_web.py — 启动训练 Web 控制台

用法:
    python run_web.py
    python run_web.py --port 8765

浏览器打开 http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.settings import load_settings

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _validate_bind_host(host: str) -> str:
    normalized = str(host or "").strip().lower()
    if normalized not in _LOOPBACK_HOSTS:
        raise ValueError("Web 控制台只允许绑定 127.0.0.1 或 localhost")
    return normalized


def _is_alphamaster_health(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("version") == "1.1.0"
    )


def _probe_alphamaster(base_url: str, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=timeout) as response:
            if response.status != 200:
                return False
            return _is_alphamaster_health(json.load(response))
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
        return False


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _open_browser_when_ready(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _probe_alphamaster(base_url):
            webbrowser.open_new_tab(base_url)
            return
        time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaMaster Training Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="服务就绪后自动打开本机浏览器",
    )
    args = parser.parse_args()
    try:
        args.host = _validate_bind_host(args.host)
    except ValueError as exc:
        parser.error(str(exc))

    base_url = f"http://{args.host}:{args.port}"
    if _probe_alphamaster(base_url):
        print(f"AlphaMaster 已运行: {base_url}")
        if args.open_browser:
            webbrowser.open_new_tab(base_url)
        return
    if _port_is_open(args.host, args.port):
        parser.error(f"端口 {args.port} 已被其他程序占用")

    try:
        import uvicorn
    except ImportError:
        print("请先安装依赖: pip install fastapi uvicorn[standard]")
        sys.exit(1)

    debug = bool(load_settings().get("debug_mode", False))

    print(f"\n  AlphaMaster 量化因子挖掘中心")
    print(f"  → {base_url}")
    print(f"  调试模式: {'开启' if debug else '关闭（默认）'}\n")

    if args.open_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(base_url,),
            daemon=True,
            name="alphamaster-browser-opener",
        ).start()

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="debug" if debug else "warning",
        access_log=debug,
    )


if __name__ == "__main__":
    main()
