"""Tests for per-symbol training time accounting."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from web import training_time as tt


@pytest.fixture
def isolated_training_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tt, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tt, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    tt._backfilled.clear()
    yield tmp_path


def test_record_and_summarize_session(isolated_training_time: Path) -> None:
    """2026-07-27 修复时间依赖：旧版把 started_at 写死为 2026-07-14 的墙钟日期，
    而活跃任务（finished_at=None）的 session_seconds = now - started_at，只在
    写测当天 ≈0，此后随真实日期线性增长（07-27 实测已达 ~1,079,824 秒）。
    改为相对 datetime.now(timezone.utc) 锚定；活跃会话断言用真实流逝时间上界，
    高负载下也不闪断。断言语义不变：刚启动的活跃任务 session 时长应≈0。
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=1)          # 昨天完成的 2.5h 历史会话
    end = start + timedelta(hours=2, minutes=30)
    tt.record_training_session(
        symbol="XAUUSD",
        started_at=start.isoformat(),
        finished_at=end.isoformat(),
        log_path="logs/train_XAUUSD_20260714_100000.log",
    )

    summary = tt.get_training_time_summary("XAUUSD")
    assert summary.history_total_seconds == 9000
    assert summary.session_seconds is None

    job_start = datetime.now(timezone.utc)   # 活跃任务：刚刚启动
    job = {
        "symbol": "XAUUSD",
        "started_at": job_start.isoformat(),
        "finished_at": None,
    }
    live = tt.get_training_time_summary("XAUUSD", job=job, active=True)
    elapsed_ceil = (datetime.now(timezone.utc) - job_start).total_seconds() + 2.0
    assert live.session_seconds is not None
    assert 0 <= live.session_seconds <= elapsed_ceil, (
        f"刚启动的活跃任务 session_seconds 应在 [0, {elapsed_ceil:.1f}]，"
        f"实际 {live.session_seconds}"
    )
    assert live.history_total_seconds >= 9000


def test_backfill_from_logs(isolated_training_time: Path) -> None:
    log = isolated_training_time / "logs" / "train_XAUUSD_20260714_120000.log"
    log.write_text("train\n", encoding="utf-8")
    start_ts = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    end_ts = start_ts + 3600
    import os

    os.utime(log, (start_ts, end_ts))

    summary = tt.get_training_time_summary("XAUUSD")
    assert summary.history_total_seconds >= 3500
    data = json.loads((isolated_training_time / "training_time_XAUUSD.json").read_text(encoding="utf-8"))
    assert len(data["sessions"]) >= 1
