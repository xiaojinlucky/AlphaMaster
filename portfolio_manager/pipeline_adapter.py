"""把经过身份复核的单股流水线产物转换为组合控制器输入。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from data_pipeline.dataset_contracts import (
    AKSHARE_HFQ_SOURCE_ID as TRAINING_SOURCE_ID,
)
from model_core.target_contract import SCORING_CONTRACT_VERSION
from portfolio_manager.controller import ModelSignalSnapshot
from web.data_sources.sina_hfq_daily import SOURCE_ID as SINA_HFQ_SOURCE_ID

SIGNAL_SIMULATION_FORMAT = "alphamaster_signal_simulation_v3"
PIPELINE_FORMAT = "alphamaster_a_share_pipeline_v2"
SIGNAL_INPUT_FORMAT = "alphamaster_signal_input_v1"
CALIBRATION_FORMAT = "alphamaster_rolling_factor_calibration_v1"
PORTFOLIO_TIMEFRAME = "1d"
_SESSION_CLOSE = time(15, 0)
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_FORCED_EXIT_ACTIONS = frozenset({"EXIT", "STOP_LOSS"})
_KNOWN_ACTIONS = frozenset(
    {
        "HOLD",
        "BUY",
        "ADD",
        "REDUCE",
        "EXIT",
        "TAKE_PROFIT",
        "STOP_LOSS",
    }
)


@dataclass(frozen=True)
class _PipelineSignalExpectation:
    """从冻结 run、策略和行情快照独立取得的预期身份。"""

    run_id: str
    symbol: str
    strategy_fingerprint: str
    market_data_sha256: str
    signal_input_sha256: str
    last_bar_ts: int
    market_source: str = SINA_HFQ_SOURCE_ID
    timeframe: str = PORTFOLIO_TIMEFRAME


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} 必须是非空且无首尾空白的文本")
    return value


def _sha256(name: str, value: object) -> str:
    text = _text(name, value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return text


def _integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} 必须至少为 {minimum}")
    return number


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_expectation(
    expected: _PipelineSignalExpectation,
) -> _PipelineSignalExpectation:
    if not isinstance(expected, _PipelineSignalExpectation):
        raise ValueError("必须提供内部验证的流水线信号身份")
    if _RUN_ID_RE.fullmatch(_text("expected.run_id", expected.run_id)) is None:
        raise ValueError("expected.run_id 格式非法")
    if _SYMBOL_RE.fullmatch(_text("expected.symbol", expected.symbol)) is None:
        raise ValueError("expected.symbol 必须是 6 位数字")
    _sha256("expected.strategy_fingerprint", expected.strategy_fingerprint)
    _sha256("expected.market_data_sha256", expected.market_data_sha256)
    _sha256("expected.signal_input_sha256", expected.signal_input_sha256)
    _integer("expected.last_bar_ts", expected.last_bar_ts)
    if expected.market_source != SINA_HFQ_SOURCE_ID:
        raise ValueError("组合控制器只接受新浪后复权同源日线")
    if expected.timeframe != PORTFOLIO_TIMEFRAME:
        raise ValueError("组合控制器只接受 1d 已收盘信号")
    return expected


def _calibration_history(
    payload: Mapping[str, Any],
    *,
    bar_ts: int,
) -> tuple[tuple[float, ...], str, str]:
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration 必须是对象")
    version = _text("calibration.version", calibration.get("version"))
    if version != CALIBRATION_FORMAT:
        raise ValueError(f"calibration.version 必须为 {CALIBRATION_FORMAT}")
    _integer(
        "calibration.window_bars",
        calibration.get("window_bars"),
        minimum=200,
    )
    if calibration.get("window_bars") != 500:
        raise ValueError("calibration.window_bars 必须等于 500")
    history_raw = calibration.get("history")
    if not isinstance(history_raw, list) or len(history_raw) != 252:
        raise ValueError("calibration.history 必须恰好包含 252 个历史点")

    points: list[dict[str, object]] = []
    scores: list[float] = []
    previous_ts = 0
    for index, raw_point in enumerate(history_raw):
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"calibration.history[{index}] 必须是对象")
        point_ts = _integer(
            f"calibration.history[{index}].bar_ts",
            raw_point.get("bar_ts"),
        )
        if point_ts <= previous_ts or point_ts >= bar_ts:
            raise ValueError("校准历史时间必须严格递增且早于当前 K 线")
        score = _finite(
            f"calibration.history[{index}].raw_score",
            raw_point.get("raw_score"),
        )
        points.append({"bar_ts": point_ts, "raw_score": score})
        scores.append(score)
        previous_ts = point_ts

    expected_hash = _sha256(
        "calibration.history_sha256",
        calibration.get("history_sha256"),
    )
    body = {
        "version": version,
        "window_bars": calibration["window_bars"],
        "history": points,
    }
    if _canonical_sha256(body) != expected_hash:
        raise ValueError("calibration.history_sha256 与历史内容不一致")
    return tuple(scores), version, expected_hash


def _model_signal_from_pipeline_payload(
    payload: Mapping[str, Any],
    *,
    expected: _PipelineSignalExpectation,
) -> ModelSignalSnapshot:
    """严格转换一个经过外部身份约束的单股虚拟信号产物。"""
    expected = _validate_expectation(expected)
    if payload.get("format") != SIGNAL_SIMULATION_FORMAT:
        raise ValueError(f"只接受 {SIGNAL_SIMULATION_FORMAT}")
    raw_signal = payload.get("raw_signal")
    lifecycle = payload.get("lifecycle_event")
    if not isinstance(raw_signal, Mapping):
        raise ValueError("raw_signal 必须是对象")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("lifecycle_event 必须是对象")
    if raw_signal.get("state") != "ok":
        raise ValueError("raw_signal.state 必须为 ok")

    run_id = _text("run_id", payload.get("run_id"))
    symbol = _text("symbol", payload.get("symbol"))
    market_source = _text("market_source", payload.get("market_source"))
    timeframe = _text("timeframe", payload.get("timeframe"))
    strategy_fingerprint = _sha256(
        "strategy_fingerprint",
        payload.get("strategy_fingerprint"),
    )
    market_data_sha256 = _sha256(
        "market_data_sha256",
        payload.get("market_data_sha256"),
    )
    signal_input_sha256 = _sha256(
        "signal_input_sha256",
        payload.get("signal_input_sha256"),
    )
    actual_identity = (
        run_id,
        symbol,
        strategy_fingerprint,
        market_data_sha256,
        signal_input_sha256,
        market_source,
        timeframe,
    )
    expected_identity = (
        expected.run_id,
        expected.symbol,
        expected.strategy_fingerprint,
        expected.market_data_sha256,
        expected.signal_input_sha256,
        expected.market_source,
        expected.timeframe,
    )
    if actual_identity != expected_identity:
        raise ValueError("信号产物与预期 run、股票、策略或行情身份不一致")
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run_id 格式非法")
    if _SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError("symbol 必须是 6 位数字")
    if str(lifecycle.get("symbol") or "") != symbol:
        raise ValueError("顶层 symbol 与 lifecycle_event.symbol 不一致")

    bar_ts = _integer("last_bar_ts", payload.get("last_bar_ts"))
    lifecycle_bar_ts = _integer(
        "lifecycle_event.bar_ts",
        lifecycle.get("bar_ts"),
    )
    if lifecycle_bar_ts != bar_ts:
        raise ValueError("顶层 last_bar_ts 与 lifecycle_event.bar_ts 不一致")
    if bar_ts != expected.last_bar_ts:
        raise ValueError("信号时间与冻结行情合同的最后 K 线不一致")
    close_at = datetime.fromtimestamp(bar_ts, tz=_SHANGHAI)
    if close_at.time() != _SESSION_CLOSE or close_at.weekday() >= 5:
        raise ValueError("日线 bar_ts 必须是工作日上海时区 15:00 收盘时刻")
    session_date = close_at.date().isoformat()
    if str(lifecycle.get("strategy_fingerprint") or "") != strategy_fingerprint:
        raise ValueError("策略指纹与 lifecycle_event 不一致")

    history, calibration_version, calibration_hash = _calibration_history(
        payload,
        bar_ts=bar_ts,
    )
    display_position = _finite(
        "raw_signal.position",
        raw_signal.get("position"),
    )
    raw_position = _finite(
        "raw_signal.position_raw",
        raw_signal.get("position_raw"),
    )
    factor_value = _finite(
        "raw_signal.factor_value_raw",
        raw_signal.get("factor_value_raw"),
    )
    display_strength = _finite(
        "raw_signal.strength",
        raw_signal.get("strength"),
    )
    strength = _finite(
        "raw_signal.strength_raw",
        raw_signal.get("strength_raw"),
    )
    if not -1.0 <= raw_position <= 1.0:
        raise ValueError("raw_signal.position_raw 必须位于 [-1, 1]")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("raw_signal.strength_raw 必须位于 [0, 1]")
    if not math.isclose(strength, abs(raw_position), abs_tol=5e-5):
        raise ValueError("raw_signal.strength_raw 必须等于 position_raw 的绝对值")
    if not math.isclose(
        display_position,
        round(raw_position, 4),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        display_strength,
        round(strength, 4),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("展示仓位或强度与完整精度字段不一致")

    action = _text("lifecycle_event.action", lifecycle.get("action"))
    if action not in _KNOWN_ACTIONS:
        raise ValueError("lifecycle_event.action 不在允许集合")
    return ModelSignalSnapshot(
        run_id=run_id,
        symbol=symbol,
        bar_ts=bar_ts,
        session_date=session_date,
        timeframe=timeframe,
        market_source=market_source,
        raw_score=factor_value,
        requested_exposure=max(0.0, raw_position),
        confidence=strength,
        model_version=strategy_fingerprint,
        data_version=market_data_sha256,
        calibration_version=calibration_version,
        calibration_history_sha256=calibration_hash,
        history_scores=history,
        model_exit=action in _FORCED_EXIT_ACTIONS,
    )


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} 不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} 顶层必须是对象")
    return payload


def _resolve_artifact_reference(
    value: object,
    *,
    project_root: Path,
    name: str,
) -> Path:
    reference = Path(_text(name, value))
    resolved = (
        reference.resolve()
        if reference.is_absolute()
        else (project_root / reference).resolve()
    )
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{name} 指向项目目录之外") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} 指向的文件不存在")
    return resolved


def _strategy_fingerprint(
    strategy: Mapping[str, Any],
    *,
    run_id: str,
    symbol: str,
    training_timeframe: str,
    data_sha256: str,
) -> str:
    if (
        strategy.get("run_id") != run_id
        or strategy.get("symbol") != symbol
        or strategy.get("timeframe") != training_timeframe
        or strategy.get("data_sha256") != data_sha256
        or strategy.get("scoring_contract_version")
        != SCORING_CONTRACT_VERSION
    ):
        raise ValueError("发布策略与 run、股票、周期、数据或评分合同不一致")
    formula = strategy.get("formula")
    if (
        not isinstance(formula, list)
        or not formula
        or any(
            isinstance(token, bool) or not isinstance(token, Integral)
            for token in formula
        )
    ):
        raise ValueError("发布策略 formula 必须是非空整数列表")
    vocab_version = _text(
        "published_strategy.vocab_version",
        strategy.get("vocab_version"),
    )
    body = {
        "formula": [int(token) for token in formula],
        "vocab_version": vocab_version,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "symbol": symbol,
        "timeframe": training_timeframe,
    }
    return _canonical_sha256(body)


def _verified_ledger_event(
    ledger_path: Path,
    *,
    event_id: str,
) -> dict[str, Any]:
    if not ledger_path.is_file():
        raise ValueError("虚拟信号 SQLite 账本不存在")
    try:
        uri = f"file:{ledger_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            integrity = conn.execute("PRAGMA quick_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise ValueError("虚拟信号 SQLite 账本完整性检查失败")
            row = conn.execute(
                "SELECT * FROM signal_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
    except (sqlite3.Error, OSError) as exc:
        raise ValueError("无法只读复核虚拟信号 SQLite 账本") from exc
    if row is None:
        raise ValueError("虚拟信号 SQLite 账本缺少对应事件")
    return dict(row)


def _verified_pipeline_payload(
    run_dir: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], _PipelineSignalExpectation]:
    run_id = run_dir.name
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run 目录名不是合法 run_id")
    expected_run_dir = (project_root / "local_runs" / run_id).resolve()
    if run_dir != expected_run_dir:
        raise ValueError("run 目录不在项目 local_runs 下")

    state = _read_json_object(run_dir / "pipeline_state.json", "pipeline_state")
    manifest = _read_json_object(run_dir / "run_manifest.json", "run_manifest")
    post_dir = run_dir / "postprocess"
    signal_path = (post_dir / "signal_simulation.json").resolve()
    signal_input_path = (post_dir / "signal_input.json").resolve()
    strategy_path = (post_dir / "published_strategy.json").resolve()
    ledger_path = (post_dir / "signal_simulation.sqlite3").resolve()

    if (
        state.get("format") != PIPELINE_FORMAT
        or state.get("run_id") != run_id
        or state.get("status") != "READY"
    ):
        raise ValueError("pipeline_state 不是当前格式的 READY 终态")
    symbol = _text("pipeline_state.symbol", state.get("symbol"))
    if _SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError("pipeline_state.symbol 必须是 6 位数字")
    training_timeframe = _text(
        "pipeline_state.timeframe",
        state.get("timeframe"),
    )
    if training_timeframe != "D1":
        raise ValueError("组合控制器只接受 D1 训练流水线")
    if (
        manifest.get("run_id") != run_id
        or manifest.get("symbol") != symbol
        or manifest.get("timeframe") != training_timeframe
        or manifest.get("local_source") != TRAINING_SOURCE_ID
    ):
        raise ValueError("run_manifest 与流水线终态或新浪后复权训练源不一致")
    training_data_sha256 = _sha256(
        "run_manifest.data_sha256",
        manifest.get("data_sha256"),
    )

    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("pipeline_state.stages 必须是对象")
    backtest_stage = stages.get("backtest")
    signal_stage = stages.get("signal")
    if (
        not isinstance(backtest_stage, Mapping)
        or backtest_stage.get("status") != "READY"
        or not isinstance(signal_stage, Mapping)
        or signal_stage.get("status") != "READY"
    ):
        raise ValueError("pipeline_state 的回测和信号阶段必须都为 READY")

    referenced_signal = _resolve_artifact_reference(
        signal_stage.get("output_path"),
        project_root=project_root,
        name="pipeline_state.signal.output_path",
    )
    if referenced_signal != signal_path:
        raise ValueError("pipeline_state 没有指向当前 run 的信号产物")
    expected_signal_sha256 = _sha256(
        "pipeline_state.signal.output_sha256",
        signal_stage.get("output_sha256"),
    )
    if hashlib.sha256(signal_path.read_bytes()).hexdigest() != expected_signal_sha256:
        raise ValueError("信号产物与 pipeline_state SHA-256 不一致")

    report_path = _resolve_artifact_reference(
        backtest_stage.get("report_path"),
        project_root=project_root,
        name="pipeline_state.backtest.report_path",
    )
    expected_report_sha256 = _sha256(
        "pipeline_state.backtest.report_sha256",
        backtest_stage.get("report_sha256"),
    )
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != expected_report_sha256:
        raise ValueError("回测报告与 pipeline_state SHA-256 不一致")
    report = _read_json_object(report_path, "回测报告")
    if (
        report.get("symbol") != symbol
        or report.get("timeframe") != training_timeframe
        or report.get("data_sha256") != training_data_sha256
        or backtest_stage.get("data_sha256") != training_data_sha256
    ):
        raise ValueError("回测报告与训练数据身份不一致")

    strategy = _read_json_object(strategy_path, "发布策略")
    strategy_fingerprint = _strategy_fingerprint(
        strategy,
        run_id=run_id,
        symbol=symbol,
        training_timeframe=training_timeframe,
        data_sha256=training_data_sha256,
    )
    signal_input = _read_json_object(signal_input_path, "冻结信号输入")
    input_sha256 = _sha256(
        "signal_input.input_sha256",
        signal_input.get("input_sha256"),
    )
    input_body = {
        key: value for key, value in signal_input.items() if key != "input_sha256"
    }
    if _canonical_sha256(input_body) != input_sha256:
        raise ValueError("冻结信号输入 SHA-256 不一致")
    market_contract = signal_input.get("market_contract")
    if not isinstance(market_contract, Mapping):
        raise ValueError("冻结信号输入缺少行情合同")
    market_data_sha256 = _sha256(
        "signal_input.market_data_sha256",
        signal_input.get("market_data_sha256"),
    )
    if _canonical_sha256(market_contract) != market_data_sha256:
        raise ValueError("冻结行情合同 SHA-256 不一致")
    market_bars = market_contract.get("bars")
    if not isinstance(market_bars, list) or len(market_bars) != 752:
        raise ValueError("冻结行情合同必须包含 752 根日线")
    market_timestamps: list[int] = []
    for index, bar in enumerate(market_bars):
        if not isinstance(bar, list) or len(bar) != 6:
            raise ValueError(f"冻结行情 bars[{index}] 必须是 6 字段列表")
        market_timestamps.append(_integer(f"冻结行情 bars[{index}].bar_ts", bar[0]))
    if any(
        current <= previous
        for previous, current in zip(
            market_timestamps,
            market_timestamps[1:],
        )
    ):
        raise ValueError("冻结行情时间必须严格递增")
    frozen_last_bar_ts = _integer(
        "signal_input.last_bar_ts",
        signal_input.get("last_bar_ts"),
    )
    if market_timestamps[-1] != frozen_last_bar_ts:
        raise ValueError("冻结信号时间与行情合同最后 K 线不一致")
    frozen_close_at = datetime.fromtimestamp(
        frozen_last_bar_ts,
        tz=_SHANGHAI,
    )
    if frozen_close_at.time() != _SESSION_CLOSE or frozen_close_at.weekday() >= 5:
        raise ValueError("冻结行情最后 K 线不是工作日上海 15:00")
    if (
        signal_input.get("format") != SIGNAL_INPUT_FORMAT
        or signal_input.get("run_id") != run_id
        or signal_input.get("symbol") != symbol
        or signal_input.get("training_timeframe") != training_timeframe
        or signal_input.get("strategy_fingerprint") != strategy_fingerprint
        or signal_input.get("market_source") != SINA_HFQ_SOURCE_ID
        or signal_input.get("timeframe") != PORTFOLIO_TIMEFRAME
        or market_contract.get("source") != SINA_HFQ_SOURCE_ID
        or market_contract.get("symbol") != symbol
        or market_contract.get("timeframe") != "D1"
    ):
        raise ValueError("冻结信号输入与 run、策略或新浪日线合同不一致")

    payload = _read_json_object(signal_path, "信号产物")
    referenced_input = _resolve_artifact_reference(
        payload.get("signal_input_path"),
        project_root=project_root,
        name="signal_input_path",
    )
    referenced_strategy = _resolve_artifact_reference(
        payload.get("strategy_file"),
        project_root=project_root,
        name="strategy_file",
    )
    if referenced_input != signal_input_path or referenced_strategy != strategy_path:
        raise ValueError("信号产物没有指向当前 run 的冻结输入或发布策略")
    if (
        payload.get("signal_input_sha256") != input_sha256
        or payload.get("last_bar_ts") != frozen_last_bar_ts
        or payload.get("raw_signal") != signal_input.get("raw_signal")
        or payload.get("calibration") != signal_input.get("calibration")
        or signal_stage.get("signal_input_sha256") != input_sha256
        or signal_stage.get("strategy_fingerprint") != strategy_fingerprint
        or signal_stage.get("market_data_sha256") != market_data_sha256
    ):
        raise ValueError("信号产物、冻结输入与 pipeline_state 不一致")
    lifecycle = payload.get("lifecycle_event")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("信号产物缺少 lifecycle_event")
    event_id = _text("lifecycle_event.event_id", lifecycle.get("event_id"))
    ledger_event = _verified_ledger_event(ledger_path, event_id=event_id)
    if ledger_event != dict(lifecycle):
        raise ValueError("信号产物 lifecycle_event 与 SQLite 账本不一致")
    raw_signal = signal_input.get("raw_signal")
    if not isinstance(raw_signal, Mapping):
        raise ValueError("冻结信号输入缺少 raw_signal")
    expected_watch_id = (
        f"pipeline:{run_id}:{SINA_HFQ_SOURCE_ID}:{symbol}:"
        f"{PORTFOLIO_TIMEFRAME}:{strategy_fingerprint[:12]}"
    )
    if (
        ledger_event.get("watch_id") != expected_watch_id
        or ledger_event.get("source") != SINA_HFQ_SOURCE_ID
        or ledger_event.get("symbol") != symbol
        or ledger_event.get("timeframe") != PORTFOLIO_TIMEFRAME
        or ledger_event.get("strategy_name") != strategy_path.stem
        or ledger_event.get("strategy_fingerprint") != strategy_fingerprint
        or ledger_event.get("bar_ts") != frozen_last_bar_ts
        or not math.isclose(
            _finite("ledger_event.price", ledger_event.get("price")),
            _finite("signal_input.last_close", signal_input.get("last_close")),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                "ledger_event.raw_position",
                ledger_event.get("raw_position"),
            ),
            _finite(
                "raw_signal.position_raw",
                raw_signal.get("position_raw"),
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                "ledger_event.factor_value",
                ledger_event.get("factor_value"),
            ),
            _finite(
                "raw_signal.factor_value_raw",
                raw_signal.get("factor_value_raw"),
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                "ledger_event.strength",
                ledger_event.get("strength"),
            ),
            _finite(
                "raw_signal.strength_raw",
                raw_signal.get("strength_raw"),
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("SQLite 生命周期事件与冻结信号输入身份不一致")
    expected = _PipelineSignalExpectation(
        run_id=run_id,
        symbol=symbol,
        strategy_fingerprint=strategy_fingerprint,
        market_data_sha256=market_data_sha256,
        signal_input_sha256=input_sha256,
        last_bar_ts=frozen_last_bar_ts,
    )
    return payload, expected


def load_pipeline_signal(
    run_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ModelSignalSnapshot:
    """从完整 READY 流水线证据链加载信号；不接受调用方自报身份。"""
    resolved_run_dir = Path(run_dir).resolve()
    resolved_root = (
        Path(project_root).resolve()
        if project_root is not None
        else resolved_run_dir.parent.parent.resolve()
    )
    payload, expected = _verified_pipeline_payload(
        resolved_run_dir,
        project_root=resolved_root,
    )
    return _model_signal_from_pipeline_payload(payload, expected=expected)


__all__ = [
    "CALIBRATION_FORMAT",
    "PORTFOLIO_TIMEFRAME",
    "SIGNAL_SIMULATION_FORMAT",
    "load_pipeline_signal",
]
