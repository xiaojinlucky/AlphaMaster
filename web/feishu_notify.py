"""飞书自定义机器人通知。

支持只做多交易动作事件与旧版方向转折文本，使用 webhook + 可选签名校验。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from strategy_manager.signal_lifecycle import ACTION_CN
from web.settings import load_settings

_DIR_CN = {
    "LONG": "看涨",
    "SHORT": "看跌",
    "FLAT": "不确定",
}
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_FEISHU_WEBHOOK_HOST = "open.feishu.cn"
_FEISHU_WEBHOOK_PATH = "/open-apis/bot/v2/hook/"
_FEISHU_WEBHOOK_TOKEN = re.compile(r"^[A-Za-z0-9_-]{20,128}$")


def direction_cn(direction: str | None) -> str:
    if not direction:
        return "未知"
    return _DIR_CN.get(str(direction).upper(), str(direction))


def strength_cn(strength: float | None, direction: str | None) -> str:
    if direction == "FLAT" or direction is None:
        return "没把握"
    s = max(0.0, min(1.0, float(strength or 0.0)))
    if s < 0.2:
        return "一点把握"
    if s < 0.4:
        return "把握不大"
    if s < 0.6:
        return "一半把握"
    if s < 0.8:
        return "比较有把握"
    return "很有把握"


def _gen_sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def validate_feishu_webhook_url(webhook_url: str) -> str:
    """只接受飞书官方自定义机器人地址，阻止后端访问任意网址。"""
    url = str(webhook_url or "").strip()
    if not url:
        raise ValueError("未配置 Webhook URL")
    if len(url) > 512:
        raise ValueError("Webhook URL 过长")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Webhook URL 格式无效") from exc
    token = (
        parsed.path[len(_FEISHU_WEBHOOK_PATH):]
        if parsed.path.startswith(_FEISHU_WEBHOOK_PATH)
        else ""
    )
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != _FEISHU_WEBHOOK_HOST
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not _FEISHU_WEBHOOK_TOKEN.fullmatch(token)
    ):
        raise ValueError(
            "Webhook URL 必须是飞书官方 "
            "https://open.feishu.cn/open-apis/bot/v2/hook/... 地址"
        )
    return url


def send_text(
    text: str,
    *,
    webhook_url: str | None = None,
    secret: str | None = None,
    timeout_s: float = 10.0,
    max_attempts: int = 3,
) -> tuple[bool, str]:
    """向飞书群发送纯文本；网络错误和服务端错误最多重试三次。"""
    settings = load_settings()
    configured_url = (
        webhook_url
        if webhook_url is not None
        else settings.get("feishu_webhook_url") or ""
    )
    try:
        url = validate_feishu_webhook_url(configured_url)
    except ValueError as exc:
        return False, str(exc)
    sec = (
        secret
        if secret is not None
        else settings.get("feishu_secret") or ""
    ).strip()
    attempts = int(max_attempts)
    if not 1 <= attempts <= 5:
        raise ValueError("max_attempts 必须位于 [1, 5]")

    payload: dict[str, Any] = {
        "msg_type": "text",
        "content": {"text": text},
    }
    if sec:
        ts = int(time.time())
        payload["timestamp"] = str(ts)
        payload["sign"] = _gen_sign(sec, ts)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_detail = "飞书请求失败"
    for attempt in range(attempts):
        retryable = False
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            last_detail = f"HTTP {exc.code}: {detail}"
            retryable = exc.code == 429 or exc.code >= 500
        except Exception as exc:  # noqa: BLE001
            last_detail = str(exc)
            retryable = True
        else:
            if data.get("code") == 0 or data.get("StatusCode") == 0:
                return True, "ok"

            code = data.get("code", data.get("StatusCode", "?"))
            msg = data.get("msg", data.get("StatusMessage", ""))
            hint = ""
            if code == 19021:
                hint = "（签名校验失败，请检查密钥或留空禁用签名）"
            elif code == 19024:
                hint = "（关键词校验失败，请检查机器人自定义关键词）"
            elif code == 19022:
                hint = "（IP 不在白名单）"
            return False, f"飞书返回 code={code} msg={msg}{hint}"

        if not retryable or attempt + 1 >= attempts:
            return False, last_detail
        time.sleep(0.5 * (2**attempt))

    return False, last_detail


def feishu_configured() -> tuple[bool, str]:
    """返回飞书是否已启用并具备 webhook。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"
    return True, "ok"


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "—"


def _price(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def format_signal_event(event: dict[str, Any]) -> str:
    """把结构化信号事件格式化为用户可直接照着执行的中文消息。"""
    action = str(event.get("action") or "")
    action_cn = ACTION_CN.get(action, action or "未知")
    bar_ts = int(event.get("bar_ts") or 0)
    bar_time = (
        datetime.fromtimestamp(bar_ts, tz=_SHANGHAI).strftime("%Y-%m-%d %H:%M")
        if bar_ts > 0
        else "—"
    )
    created_at = float(event.get("created_at") or 0.0)
    created_time = (
        datetime.fromtimestamp(created_at, tz=_SHANGHAI).strftime("%Y-%m-%d %H:%M:%S")
        if created_at > 0
        else "—"
    )
    lines = [
        "【AlphaMaster 大A交易信号】",
        "",
        f"股票：{event.get('symbol') or '—'}",
        f"周期：{event.get('timeframe') or '—'}",
        f"动作：{action_cn}",
        (
            "虚拟仓位："
            f"{_pct(event.get('previous_exposure'))} → "
            f"{_pct(event.get('resulting_exposure'))}"
        ),
        f"模型目标仓位：{_pct(event.get('requested_exposure'))}",
        f"参考价格：{_price(event.get('price'))}",
        f"虚拟成本：{_price(event.get('entry_price'))}",
        f"止损参考：{_price(event.get('stop_price'))}",
        f"止盈参考：{_price(event.get('take_profit_price'))}",
        f"信号强度：{_pct(event.get('strength'))}",
        f"K线开盘时间：{bar_time}",
        f"信号生成时间：{created_time}",
        f"原因：{event.get('reason') or '—'}",
        f"策略：{event.get('strategy_name') or '—'}",
        f"信号编号：{event.get('event_id') or '—'}",
        "执行边界：减仓或离场请以账户实际可卖数量为准（普通 A 股 T+1）。",
    ]
    return "\n".join(lines)


def notify_signal_event(event: dict[str, Any]) -> tuple[bool, str]:
    """推送一条已经持久化的交易动作事件。"""
    configured, detail = feishu_configured()
    if not configured:
        return False, detail
    return send_text(format_signal_event(event))


def notify_direction_flip(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    prev_direction: str,
    new_direction: str,
    strength: float | None = None,
    factor_value: float | None = None,
) -> tuple[bool, str]:
    """信号方向发生转折时推送提醒。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    prev_cn = direction_cn(prev_direction)
    new_cn = direction_cn(new_direction)
    grasp = strength_cn(strength, new_direction)
    factor_s = f"{factor_value:+.4f}" if factor_value is not None else "—"

    text = (
        f"【AlphaMaster 信号转折】\n"
        f"{symbol} · {timeframe}\n"
        f"上次判断：{prev_cn}\n"
        f"本次判断：{new_cn}（{grasp}）\n"
        f"策略：{strategy_name}\n"
        f"因子：{factor_s}"
    )
    return send_text(text)
