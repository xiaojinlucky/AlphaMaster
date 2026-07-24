from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from web.data_sources.base import Bar
from web.feishu_notify import (
    format_signal_event,
    send_text,
    validate_feishu_webhook_url,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_VOCAB_VERSION = "test-vocab-version"
TEST_SCORING_CONTRACT = "test-scoring-contract"


def _load_realtime_module(monkeypatch: pytest.MonkeyPatch):
    """隔离本机损坏的 PyTorch DLL，只替换本测试不执行的公式依赖。"""

    class TestVocabMismatchError(ValueError):
        pass

    class TestFormulaVocab:
        @staticmethod
        def verify(version: str) -> None:
            if version != TEST_VOCAB_VERSION:
                raise TestVocabMismatchError(version)

    vocab_stub = types.ModuleType("model_core.vocab")
    vocab_stub.FORMULA_VOCAB = TestFormulaVocab()
    vocab_stub.VOCAB_VERSION = TEST_VOCAB_VERSION
    vocab_stub.VocabVersionMismatchError = TestVocabMismatchError
    target_contract_stub = types.ModuleType("model_core.target_contract")
    target_contract_stub.SCORING_CONTRACT_VERSION = TEST_SCORING_CONTRACT

    live_signal_stub = types.ModuleType("strategy_manager.live_signal")
    live_signal_stub.evaluate_signal = lambda *_args, **_kwargs: {}
    live_signal_stub.min_exposure = lambda: 0.05

    class TestConfig:
        SIGNAL_EVENT_DB = "local_data/test-signals.sqlite3"
        SIGNAL_REBALANCE_DELTA = 0.10
        SIGNAL_TAKE_PROFIT_REMAINING_RATIO = 0.50
        MIN_TRADE_EXPOSURE = 0.05
        STOP_LOSS_PCT = -0.02
        TAKE_PROFIT_PCT = 0.04

    config_stub = types.ModuleType("config")
    config_stub.Config = TestConfig

    monkeypatch.setitem(sys.modules, "model_core.vocab", vocab_stub)
    monkeypatch.setitem(
        sys.modules,
        "model_core.target_contract",
        target_contract_stub,
    )
    monkeypatch.setitem(sys.modules, "strategy_manager.live_signal", live_signal_stub)
    monkeypatch.setitem(sys.modules, "config", config_stub)

    module_name = f"_realtime_manager_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "web" / "realtime_manager.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "bars_to_raw_dict", lambda _bars: {})
    return module


def _strategy_file(tmp_path: Path) -> Path:
    path = tmp_path / "best_600519.json"
    path.write_text(
        json.dumps(
            {
                "formula": [0],
                "vocab_version": TEST_VOCAB_VERSION,
                "scoring_contract_version": TEST_SCORING_CONTRACT,
                "symbol": "600519",
                "timeframe": "15m",
                "best_score": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _manager_and_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    realtime_module = _load_realtime_module(monkeypatch)
    manager = realtime_module.RealtimeManager(tmp_path / "signals.sqlite3")
    task = manager._add_task_internal(
        "tongdaxin",
        "600519",
        "15m",
        str(_strategy_file(tmp_path)),
        persist=False,
    )
    return realtime_module, manager, task


def _ok_result(position: float) -> dict:
    direction = "LONG" if position >= 0.05 else "SHORT" if position <= -0.05 else "FLAT"
    return {
        "state": "ok",
        "direction": direction,
        "strength_raw": abs(position),
        "strength": abs(position),
        "position_raw": position,
        "position": position,
        "factor_value_raw": position,
        "factor_value": position,
        "bars_used": 500,
        "message": "",
    }


def test_realtime_pipeline_emits_once_per_closed_bar_and_delivers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realtime_module, manager, task = _manager_and_task(tmp_path, monkeypatch)
    bars = [Bar(ts=1000, open=100, high=101, low=99, close=100, volume=1)]
    result = _ok_result(0.40)
    evaluate_calls = 0
    sent: list[dict] = []

    def fake_bars(*_args, **_kwargs):
        return list(bars)

    def fake_evaluate(*_args, **_kwargs):
        nonlocal evaluate_calls
        evaluate_calls += 1
        return dict(result)

    monkeypatch.setattr(manager, "_get_bars", fake_bars)
    monkeypatch.setattr(realtime_module, "evaluate_signal", fake_evaluate)
    monkeypatch.setattr(
        "web.feishu_notify.feishu_configured",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        "web.feishu_notify.notify_signal_event",
        lambda event: (sent.append(dict(event)) is None, "ok"),
    )

    manager._evaluate_task(task)
    assert task.latest_event is not None
    assert task.latest_event["action"] == "BUY"
    assert task.latest_event["delivery_status"] == "DELIVERED"
    assert evaluate_calls == 1
    assert len(sent) == 1

    manager._evaluate_task(task)
    assert evaluate_calls == 1
    assert len(sent) == 1

    bars[0] = Bar(ts=2000, open=101, high=102, low=100, close=101, volume=1)
    result.update(_ok_result(0.70))
    manager._evaluate_task(task)

    assert task.latest_event is not None
    assert task.latest_event["action"] == "ADD"
    assert task.latest_event["delivery_status"] == "DELIVERED"
    assert evaluate_calls == 2
    assert len(sent) == 2
    assert [row["action"] for row in manager.signal_events(limit=10)] == ["ADD", "BUY"]


def test_realtime_pipeline_records_skipped_and_failed_delivery(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realtime_module, manager, task = _manager_and_task(tmp_path, monkeypatch)
    bars = [Bar(ts=1000, open=100, high=101, low=99, close=100, volume=1)]
    monkeypatch.setattr(manager, "_get_bars", lambda *_args, **_kwargs: list(bars))
    monkeypatch.setattr(
        realtime_module,
        "evaluate_signal",
        lambda *_args, **_kwargs: _ok_result(0.40),
    )
    monkeypatch.setattr(
        "web.feishu_notify.feishu_configured",
        lambda: (False, "飞书通知未启用"),
    )

    manager._evaluate_task(task)
    assert task.latest_event is not None
    assert task.latest_event["delivery_status"] == "SKIPPED"

    bars[0] = Bar(ts=2000, open=101, high=102, low=100, close=101, volume=1)
    monkeypatch.setattr(
        realtime_module,
        "evaluate_signal",
        lambda *_args, **_kwargs: _ok_result(0.70),
    )
    monkeypatch.setattr(
        "web.feishu_notify.feishu_configured",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        "web.feishu_notify.notify_signal_event",
        lambda _event: (False, "network down"),
    )

    manager._evaluate_task(task)
    assert task.latest_event is not None
    assert task.latest_event["delivery_status"] == "FAILED"
    assert task.latest_event["delivery_detail"] == "network down"
    assert "飞书投递失败" in task.warn


def test_negative_a_share_signal_is_recorded_but_not_pushed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realtime_module, manager, task = _manager_and_task(tmp_path, monkeypatch)
    monkeypatch.setattr(
        manager,
        "_get_bars",
        lambda *_args, **_kwargs: [
            Bar(ts=1000, open=100, high=101, low=99, close=100, volume=1)
        ],
    )
    monkeypatch.setattr(
        realtime_module,
        "evaluate_signal",
        lambda *_args, **_kwargs: _ok_result(-0.80),
    )
    notified = False

    def unexpected_notify(_event):
        nonlocal notified
        notified = True
        return True, "ok"

    monkeypatch.setattr(
        "web.feishu_notify.notify_signal_event",
        unexpected_notify,
    )

    manager._evaluate_task(task)

    assert task.latest_event is not None
    assert task.latest_event["action"] == "HOLD"
    assert task.latest_event["requested_exposure"] == 0.0
    assert task.latest_event["delivery_status"] == "NOT_REQUIRED"
    assert notified is False


def test_realtime_pipeline_uses_full_precision_position_for_entry_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realtime_module, manager, task = _manager_and_task(tmp_path, monkeypatch)
    monkeypatch.setattr(
        manager,
        "_get_bars",
        lambda *_args, **_kwargs: [
            Bar(ts=1000, open=100, high=101, low=99, close=100, volume=1)
        ],
    )
    result = _ok_result(0.04996)
    result["position"] = 0.05
    result["strength"] = 0.05
    monkeypatch.setattr(
        realtime_module,
        "evaluate_signal",
        lambda *_args, **_kwargs: dict(result),
    )

    manager._evaluate_task(task)

    assert task.position == 0.05
    assert task.latest_event is not None
    assert task.latest_event["action"] == "HOLD"
    assert task.latest_event["requested_exposure"] == 0.0
    assert task.latest_event["raw_position"] == pytest.approx(0.04996)


def test_feishu_trade_signal_message_contains_auditable_fields() -> None:
    text = format_signal_event(
        {
            "event_id": "AM-ABC",
            "symbol": "600519",
            "timeframe": "15m",
            "action": "ADD",
            "previous_exposure": 0.30,
            "requested_exposure": 0.65,
            "resulting_exposure": 0.65,
            "price": 1482.50,
            "entry_price": 1460.0,
            "stop_price": 1430.8,
            "take_profit_price": 1518.4,
            "strength": 0.65,
            "bar_ts": 1_753_263_900,
            "reason": "目标长仓提高至少 10%",
            "strategy_name": "best_600519",
        }
    )

    for expected in (
        "AlphaMaster 大A交易信号",
        "股票：600519",
        "动作：加仓",
        "虚拟仓位：30% → 65%",
        "模型目标仓位：65%",
        "止损参考：1430.8",
        "信号编号：AM-ABC",
    ):
        assert expected in text


def test_feishu_webhook_only_accepts_official_robot_endpoint() -> None:
    official = (
        "https://open.feishu.cn/open-apis/bot/v2/hook/"
        "12345678-1234-1234-1234-123456789abc"
    )
    assert validate_feishu_webhook_url(official) == official

    for invalid in (
        "http://open.feishu.cn/open-apis/bot/v2/hook/12345678901234567890",
        "https://127.0.0.1/open-apis/bot/v2/hook/12345678901234567890",
        (
            "https://open.feishu.cn.evil.example/open-apis/bot/v2/hook/"
            "12345678901234567890"
        ),
        "https://open.feishu.cn/open-apis/bot/v2/hook/short",
    ):
        with pytest.raises(ValueError, match="飞书官方"):
            validate_feishu_webhook_url(invalid)


def test_feishu_transport_retries_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official = (
        "https://open.feishu.cn/open-apis/bot/v2/hook/"
        "12345678-1234-1234-1234-123456789abc"
    )
    attempts = 0
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read() -> bytes:
            return b'{"code":0,"msg":"success"}'

    def fake_urlopen(_request, *, timeout):
        nonlocal attempts
        assert timeout == 1.0
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                official,
                503,
                "busy",
                {},
                io.BytesIO(b"temporary"),
            )
        return FakeResponse()

    monkeypatch.setattr("web.feishu_notify.load_settings", lambda: {})
    monkeypatch.setattr("web.feishu_notify.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("web.feishu_notify.time.sleep", sleeps.append)

    ok, detail = send_text(
        "AlphaMaster test",
        webhook_url=official,
        timeout_s=1.0,
        max_attempts=3,
    )

    assert ok is True
    assert detail == "ok"
    assert attempts == 2
    assert sleeps == [0.5]
