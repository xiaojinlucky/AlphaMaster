"""确定性的多标的信号校准与 Top-K 组合选择。

首版只解决一个问题：把同一根已收盘 K 线上的异构单股模型输出，
转换成可比较、可复现、可审计的目标持仓。交易费用、涨跌停、停牌、
T+1 和整手约束属于后续虚拟执行层，不在本模块中猜测或静默处理。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, time
from numbers import Integral, Real
from zoneinfo import ZoneInfo

from portfolio_manager.universe import (
    UniverseContract,
    load_csi_a50_universe_contract,
)

_SYMBOL_RE = re.compile(r"^\d{6}$")
_RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$")
_SESSION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WEIGHT_EPSILON = 1e-9
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_DAILY_CLOSE = time(15, 0)
_CSI_A50_TIMEFRAME = "1d"
_CSI_A50_MARKET_SOURCE = "akshare_sina_hfq_ohlcv"


@dataclass(frozen=True)
class ModelSignalSnapshot:
    """一只股票在同一决策时点的模型输出与历史校准样本。"""

    run_id: str
    symbol: str
    bar_ts: int
    session_date: str
    timeframe: str
    market_source: str
    raw_score: float
    requested_exposure: float
    confidence: float
    model_version: str
    data_version: str
    calibration_version: str
    calibration_history_sha256: str
    history_scores: tuple[float, ...]
    model_exit: bool = False


@dataclass(frozen=True)
class PortfolioPolicy:
    """首版组合选择规则。"""

    top_k: int
    dropout_rank: int
    minimum_history: int = 20
    minimum_model_exposure: float = 0.05
    minimum_confidence: float = 0.0
    minimum_calibrated_score: float = 0.5
    gross_exposure: float = 1.0


@dataclass(frozen=True)
class CalibratedSignal:
    """校准后的单股信号及其横截面排名。"""

    run_id: str
    symbol: str
    raw_score: float
    calibrated_score: float
    confidence: float
    requested_exposure: float
    model_version: str
    data_version: str
    calibration_version: str
    calibration_history_sha256: str
    eligible: bool
    rejection_reason: str
    rank: int | None


@dataclass(frozen=True)
class PortfolioDecision:
    """一次确定性的组合目标持仓决策。"""

    decision_id: str
    universe: UniverseContract
    policy: PortfolioPolicy
    bar_ts: int
    session_date: str
    timeframe: str
    market_source: str
    account_snapshot_sha256: str
    current_weights: tuple[tuple[str, float], ...]
    selected_symbols: tuple[str, ...]
    target_weights: tuple[tuple[str, float], ...]
    entered_symbols: tuple[str, ...]
    retained_symbols: tuple[str, ...]
    exited_symbols: tuple[str, ...]
    cash_weight: float
    ranking: tuple[CalibratedSignal, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "universe": self.universe.to_dict(),
            "policy": asdict(self.policy),
            "bar_ts": self.bar_ts,
            "session_date": self.session_date,
            "timeframe": self.timeframe,
            "market_source": self.market_source,
            "account_snapshot_sha256": self.account_snapshot_sha256,
            "current_weights": dict(self.current_weights),
            "selected_symbols": list(self.selected_symbols),
            "target_weights": dict(self.target_weights),
            "entered_symbols": list(self.entered_symbols),
            "retained_symbols": list(self.retained_symbols),
            "exited_symbols": list(self.exited_symbols),
            "cash_weight": self.cash_weight,
            "ranking": [asdict(row) for row in self.ranking],
        }


def _checked_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _checked_unit_interval(name: str, value: object) -> float:
    number = _checked_real(name, value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]")
    return number


def _checked_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} 必须至少为 {minimum}")
    return number


def _checked_symbol(symbol: object) -> str:
    if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError(f"股票代码必须是 6 位数字文本: {symbol!r}")
    return symbol


def _checked_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} 不能为空")
    return value


def _checked_sha256(name: str, value: object) -> str:
    text = _checked_text(name, value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} 必须是小写 SHA-256")
    return text


def _validate_policy(policy: PortfolioPolicy, universe_size: int) -> None:
    if universe_size <= 0:
        raise ValueError("冻结股票池不能为空")
    top_k = _checked_integer("top_k", policy.top_k)
    dropout_rank = _checked_integer("dropout_rank", policy.dropout_rank)
    minimum_history = _checked_integer(
        "minimum_history",
        policy.minimum_history,
        minimum=2,
    )
    if top_k > universe_size:
        raise ValueError("top_k 必须位于 [1, 股票池大小]")
    if not top_k <= dropout_rank <= universe_size:
        raise ValueError("dropout_rank 必须位于 [top_k, 股票池大小]")
    if minimum_history < 2:
        raise ValueError("minimum_history 必须至少为 2")
    _checked_unit_interval(
        "minimum_model_exposure",
        policy.minimum_model_exposure,
    )
    _checked_unit_interval("minimum_confidence", policy.minimum_confidence)
    _checked_unit_interval(
        "minimum_calibrated_score",
        policy.minimum_calibrated_score,
    )
    gross = _checked_real("gross_exposure", policy.gross_exposure)
    if not math.isfinite(gross) or not 0.0 < gross <= 1.0:
        raise ValueError("gross_exposure 必须位于 (0, 1]")


def _empirical_percentile(
    raw_score: float,
    history_scores: Iterable[float],
    *,
    minimum_history: int,
) -> float:
    """用单标的历史中位秩校准当前分数，不混用不同模型的原始量纲。"""
    current = _checked_real("raw_score", raw_score)
    history = tuple(
        _checked_real(f"history_scores[{index}]", value)
        for index, value in enumerate(history_scores)
    )
    if len(history) < minimum_history:
        raise ValueError(f"历史校准样本不足: {len(history)} < {minimum_history}")
    less = sum(value < current for value in history)
    equal = sum(value == current for value in history)
    return (less + 0.5 * equal) / len(history)


def _validate_current_weights(
    current_weights: Mapping[str, float],
    expected_symbols: frozenset[str],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_symbol, raw_weight in current_weights.items():
        symbol = _checked_symbol(raw_symbol)
        if symbol not in expected_symbols:
            raise ValueError(f"当前持仓不属于冻结股票池: {symbol}")
        weight = _checked_unit_interval(
            f"current_weights[{symbol}]",
            raw_weight,
        )
        if weight > 0.0:
            normalized[symbol] = weight
    if sum(normalized.values()) > 1.0 + _WEIGHT_EPSILON:
        raise ValueError("当前持仓权重合计不能超过 1")
    return normalized


def _canonical_decision_id(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "AM-PORT-" + hashlib.sha256(raw).hexdigest()[:24].upper()


def _build_portfolio_decision(
    signals: Iterable[ModelSignalSnapshot],
    *,
    universe: UniverseContract,
    current_weights: Mapping[str, float],
    account_snapshot_sha256: str,
    policy: PortfolioPolicy,
    previous_decision_ts: int | None = None,
) -> PortfolioDecision:
    """校准同一时点的完整股票池，并生成 Top-K + Dropout 目标持仓。

    这里故意要求完整股票池和完全一致的 ``bar_ts``。少一只股票、混入
    不同 K 线或时间倒退都会直接失败，避免把陈旧信号静默拼成一个组合。
    """
    if not isinstance(universe, UniverseContract):
        raise ValueError("universe 必须是经过验证的冻结股票池合同")
    expected_list = tuple(_checked_symbol(symbol) for symbol in universe.symbols)
    expected_set = frozenset(expected_list)
    _validate_policy(policy, len(expected_set))

    rows = tuple(signals)
    if not rows:
        raise ValueError("signals 不能为空")
    by_symbol: dict[str, ModelSignalSnapshot] = {}
    timestamps: set[int] = set()
    session_dates: set[str] = set()
    timeframes: set[str] = set()
    market_sources: set[str] = set()
    run_ids: set[str] = set()
    for row in rows:
        symbol = _checked_symbol(row.symbol)
        if symbol in by_symbol:
            raise ValueError(f"signals 出现重复股票: {symbol}")
        if symbol not in expected_set:
            raise ValueError(f"signals 包含冻结股票池外标的: {symbol}")
        bar_ts = _checked_integer(f"{symbol}.bar_ts", row.bar_ts)
        run_id = _checked_text(f"{symbol}.run_id", row.run_id)
        if _RUN_ID_RE.fullmatch(run_id) is None:
            raise ValueError(f"{symbol}.run_id 格式非法")
        if run_id in run_ids:
            raise ValueError(f"不同股票不能复用同一训练 run: {run_id}")
        run_ids.add(run_id)
        session_date = _checked_text(
            f"{symbol}.session_date",
            row.session_date,
        )
        if _SESSION_DATE_RE.fullmatch(session_date) is None:
            raise ValueError(f"{symbol}.session_date 必须是 YYYY-MM-DD")
        timeframe = _checked_text(f"{symbol}.timeframe", row.timeframe)
        market_source = _checked_text(
            f"{symbol}.market_source",
            row.market_source,
        )
        if timeframe == "1d":
            close_at = datetime.fromtimestamp(bar_ts, tz=_SHANGHAI)
            if (
                close_at.time() != _DAILY_CLOSE
                or close_at.date().isoformat() != session_date
            ):
                raise ValueError(f"{symbol}.bar_ts 与日线交易日或 15:00 收盘语义不一致")
        timestamps.add(bar_ts)
        session_dates.add(session_date)
        timeframes.add(timeframe)
        market_sources.add(market_source)
        by_symbol[symbol] = row

    actual_set = frozenset(by_symbol)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"signals 必须完整覆盖冻结股票池；缺少={missing}，多出={extra}"
        )
    if len(timestamps) != 1:
        raise ValueError("所有 signals 必须来自同一根已收盘 K 线")
    if len(session_dates) != 1:
        raise ValueError("所有 signals 必须来自同一交易日")
    if len(timeframes) != 1:
        raise ValueError("所有 signals 必须使用同一周期")
    if len(market_sources) != 1:
        raise ValueError("所有 signals 必须使用同一行情来源")
    bar_ts = next(iter(timestamps))
    session_date = next(iter(session_dates))
    timeframe = next(iter(timeframes))
    market_source = next(iter(market_sources))
    if previous_decision_ts is not None:
        previous_ts = _checked_integer(
            "previous_decision_ts",
            previous_decision_ts,
        )
        if bar_ts <= previous_ts:
            raise ValueError(f"组合决策时间未前进: {bar_ts} <= {previous_ts}")

    current = _validate_current_weights(current_weights, expected_set)
    account_snapshot = _checked_sha256(
        "account_snapshot_sha256",
        account_snapshot_sha256,
    )
    minimum_history = _checked_integer(
        "minimum_history",
        policy.minimum_history,
        minimum=2,
    )
    staged: list[CalibratedSignal] = []
    for symbol in sorted(by_symbol):
        row = by_symbol[symbol]
        confidence = _checked_unit_interval(
            f"{symbol}.confidence",
            row.confidence,
        )
        exposure = _checked_unit_interval(
            f"{symbol}.requested_exposure",
            row.requested_exposure,
        )
        score = _empirical_percentile(
            row.raw_score,
            row.history_scores,
            minimum_history=minimum_history,
        )
        model_version = _checked_sha256(
            f"{symbol}.model_version",
            row.model_version,
        )
        data_version = _checked_sha256(
            f"{symbol}.data_version",
            row.data_version,
        )
        calibration_version = _checked_text(
            f"{symbol}.calibration_version",
            row.calibration_version,
        )
        calibration_history_sha256 = _checked_sha256(
            f"{symbol}.calibration_history_sha256",
            row.calibration_history_sha256,
        )
        if not isinstance(row.model_exit, bool):
            raise ValueError(f"{symbol}.model_exit 必须是布尔值")
        rejection_reason = ""
        if row.model_exit:
            rejection_reason = "单股模型主动离场"
        elif exposure < float(policy.minimum_model_exposure):
            rejection_reason = "单股目标仓位未达到入场门槛"
        elif confidence < float(policy.minimum_confidence):
            rejection_reason = "信号置信度未达到门槛"
        elif score < float(policy.minimum_calibrated_score):
            rejection_reason = "历史分位分数未达到门槛"
        staged.append(
            CalibratedSignal(
                run_id=row.run_id,
                symbol=symbol,
                raw_score=float(row.raw_score),
                calibrated_score=round(score, 12),
                confidence=confidence,
                requested_exposure=exposure,
                model_version=model_version,
                data_version=data_version,
                calibration_version=calibration_version,
                calibration_history_sha256=calibration_history_sha256,
                eligible=not rejection_reason,
                rejection_reason=rejection_reason,
                rank=None,
            )
        )

    eligible = sorted(
        (row for row in staged if row.eligible),
        key=lambda row: (
            -row.calibrated_score,
            row.symbol,
        ),
    )
    ranks = {row.symbol: index for index, row in enumerate(eligible, start=1)}
    ranked = tuple(
        CalibratedSignal(
            **{
                **asdict(row),
                "rank": ranks.get(row.symbol),
            }
        )
        for row in sorted(
            staged,
            key=lambda row: (
                row.symbol not in ranks,
                ranks.get(row.symbol, len(expected_set) + 1),
                row.symbol,
            ),
        )
    )

    held = frozenset(current)
    top_k = _checked_integer("top_k", policy.top_k)
    dropout_rank = _checked_integer("dropout_rank", policy.dropout_rank)
    retained = [
        row.symbol
        for row in eligible
        if row.symbol in held and ranks[row.symbol] <= dropout_rank
    ][:top_k]
    selected = list(retained)
    for row in eligible:
        if len(selected) >= top_k:
            break
        if row.symbol not in selected:
            selected.append(row.symbol)

    selected_tuple = tuple(selected)
    target_per_slot = _checked_real("gross_exposure", policy.gross_exposure) / top_k
    target_weights = tuple(
        (symbol, round(target_per_slot, 12)) for symbol in selected_tuple
    )
    target_total = sum(weight for _, weight in target_weights)
    entered = tuple(symbol for symbol in selected_tuple if symbol not in held)
    retained_final = tuple(symbol for symbol in selected_tuple if symbol in held)
    exited = tuple(sorted(symbol for symbol in held if symbol not in selected_tuple))

    current_tuple = tuple(sorted(current.items()))
    identity = {
        "universe": universe.to_dict(),
        "policy": asdict(policy),
        "bar_ts": bar_ts,
        "session_date": session_date,
        "timeframe": timeframe,
        "market_source": market_source,
        "account_snapshot_sha256": account_snapshot,
        "current_weights": dict(current_tuple),
        "selected_symbols": list(selected_tuple),
        "target_weights": dict(target_weights),
        "entered_symbols": list(entered),
        "retained_symbols": list(retained_final),
        "exited_symbols": list(exited),
        "cash_weight": round(max(0.0, 1.0 - target_total), 12),
        "ranking": [asdict(row) for row in ranked],
    }
    return PortfolioDecision(
        decision_id=_canonical_decision_id(identity),
        universe=universe,
        policy=policy,
        bar_ts=bar_ts,
        session_date=session_date,
        timeframe=timeframe,
        market_source=market_source,
        account_snapshot_sha256=account_snapshot,
        current_weights=current_tuple,
        selected_symbols=selected_tuple,
        target_weights=target_weights,
        entered_symbols=entered,
        retained_symbols=retained_final,
        exited_symbols=exited,
        cash_weight=round(max(0.0, 1.0 - target_total), 12),
        ranking=ranked,
    )


def build_csi_a50_portfolio_decision(
    signals: Iterable[ModelSignalSnapshot],
    *,
    current_weights: Mapping[str, float],
    account_snapshot_sha256: str,
    policy: PortfolioPolicy,
    previous_decision_ts: int | None = None,
) -> PortfolioDecision:
    """从经过官方合同复核的 50 股快照生成 A50 目标组合。"""
    rows = tuple(signals)
    for row in rows:
        if not isinstance(row, ModelSignalSnapshot):
            raise ValueError("A50 signals 必须由 ModelSignalSnapshot 组成")
        if row.timeframe != _CSI_A50_TIMEFRAME:
            raise ValueError("A50 正式组合只接受 1d 日线信号")
        if row.market_source != _CSI_A50_MARKET_SOURCE:
            raise ValueError("A50 正式组合只接受 akshare_sina_hfq_ohlcv 行情来源")
    universe = load_csi_a50_universe_contract()
    return _build_portfolio_decision(
        rows,
        universe=universe,
        current_weights=current_weights,
        account_snapshot_sha256=account_snapshot_sha256,
        policy=policy,
        previous_decision_ts=previous_decision_ts,
    )


__all__ = [
    "CalibratedSignal",
    "ModelSignalSnapshot",
    "PortfolioDecision",
    "PortfolioPolicy",
    "build_csi_a50_portfolio_decision",
]
