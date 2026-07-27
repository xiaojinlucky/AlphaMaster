"""组合目标决策与确定性虚拟执行的 SQLite 审计账本。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from portfolio_manager.controller import (
    CalibratedSignal,
    PortfolioDecision,
    PortfolioPolicy,
    _canonical_decision_id,
)
from portfolio_manager.universe import (
    UNIVERSE_CONTRACT_TYPE_HISTORICAL,
    UNIVERSE_CONTRACT_TYPE_UNTRUSTED,
    HistoricalUniverseContract,
    UniverseContract,
    WeightedUniverseConstituent,
)

_DECISION_ID_RE = re.compile(r"^AM-PORT-[0-9A-F]{24}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$")
_DATE_RE = re.compile(r"^\d{8}$")
_SESSION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "decision_id",
        "universe",
        "policy",
        "bar_ts",
        "session_date",
        "timeframe",
        "market_source",
        "account_snapshot_sha256",
        "current_weights",
        "selected_symbols",
        "target_weights",
        "entered_symbols",
        "retained_symbols",
        "exited_symbols",
        "cash_weight",
        "ranking",
    }
)
_UNIVERSE_KEYS = frozenset(
    {
        "universe_id",
        "snapshot_date",
        "constituent_count",
        "universe_sha256",
        "symbols",
        "contract_type",
        "query_mode",
        "point_in_time_safe",
        "sealed_oos_eligible",
        "provenance_identity",
        "contract_sha256",
    }
)
_HISTORICAL_UNIVERSE_KEYS = _UNIVERSE_KEYS | frozenset(
    {
        "selection_format",
        "source_format",
        "source_index",
        "source_trust_policy",
        "as_of_date",
        "mode",
        "source_effective_date",
        "source_effective_until_exclusive",
        "observed_at",
        "receipt_at",
        "strict_available_at",
        "source_history_root",
        "query_at",
        "reconstructed",
        "source_data_sha256",
        "source_receipt_sha256",
        "constituents",
    }
)
_HISTORICAL_CONSTITUENT_KEYS = frozenset(
    {"symbol", "weight", "display_name"}
)
_POLICY_KEYS = frozenset(
    {
        "top_k",
        "dropout_rank",
        "minimum_history",
        "minimum_model_exposure",
        "minimum_confidence",
        "minimum_calibrated_score",
        "gross_exposure",
    }
)
_RANKING_KEYS = frozenset(
    {
        "run_id",
        "symbol",
        "raw_score",
        "calibrated_score",
        "confidence",
        "requested_exposure",
        "model_version",
        "data_version",
        "calibration_version",
        "calibration_history_sha256",
        "eligible",
        "rejection_reason",
        "rank",
    }
)
_EPSILON = 1e-9
_EXECUTION_LEDGER_VERSION = "portfolio-execution-ledger-v1"
_EXECUTION_ID_RE = re.compile(r"^AM-EXEC-[0-9A-F]{24}$")
_EXECUTION_INPUT_KEYS = frozenset(
    {
        "contract_version",
        "decision_id",
        "execution_session",
        "account_before",
        "decision_quotes",
        "execution_quotes",
        "fee_schedule",
    }
)
_ACCOUNT_KEYS = frozenset({"cash", "lots"})
_POSITION_LOT_KEYS = frozenset(
    {"symbol", "acquired_session", "shares", "unit_cost"}
)
_QUOTE_KEYS = frozenset(
    {"symbol", "session_date", "price", "status", "lot_size"}
)
_FEE_KEYS = frozenset(
    {
        "commission_rate",
        "minimum_commission",
        "stamp_duty_rate",
        "transfer_fee_rate",
        "slippage_rate",
    }
)
# RND-04C：replay 运行身份绑定（把 overlay/universe/价格身份链进账本，
# 与既有执行行 row_sha256 交叉锁定；语义重算核验在 replay 层完成）。
REPLAY_BINDING_VERSION = "portfolio-replay-binding-v1"
_REPLAY_BINDING_KEYS = frozenset(
    {
        "contract_version",
        "replay_contract_version",
        "replay_run_id",
        "execution_id",
        "decision_id",
        "decision_session",
        "execution_session",
        "universe",
        "overlay_identity_sha256",
        "derivation_rule_version",
        "freestockdb_price_identity",
        "decision_quote_provenance",
        "execution_quote_provenance",
        "engineering_signal",
        "fee_schedule",
        "execution_row_sha256",
    }
)


def _strict_object(
    name: str,
    value: object,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} 必须是 JSON 对象")
    actual_keys = frozenset(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(f"{name} 字段集合非法；缺少={missing}，多出={extra}")
    return value


def _strict_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeError(f"{name} 必须是非空文本")
    return value


def _strict_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(f"{name} 必须是整数")
    number = int(value)
    if number < minimum:
        raise RuntimeError(f"{name} 必须至少为 {minimum}")
    return number


def _strict_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"{name} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} 必须是有限数")
    return number


def _strict_unit_interval(name: str, value: object) -> float:
    number = _strict_real(name, value)
    if not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{name} 必须位于 [0, 1]")
    return number


def _strict_symbol(name: str, value: object) -> str:
    text = _strict_text(name, value)
    if _SYMBOL_RE.fullmatch(text) is None:
        raise RuntimeError(f"{name} 必须是 6 位股票代码")
    return text


def _strict_sha256(name: str, value: object) -> str:
    text = _strict_text(name, value)
    if _SHA256_RE.fullmatch(text) is None:
        raise RuntimeError(f"{name} 必须是小写 SHA-256")
    return text


def _strict_symbol_list(
    name: str,
    value: object,
    *,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{name} 必须是列表")
    symbols = tuple(
        _strict_symbol(f"{name}[{index}]", symbol) for index, symbol in enumerate(value)
    )
    if len(set(symbols)) != len(symbols):
        raise RuntimeError(f"{name} 不能包含重复股票")
    if not set(symbols).issubset(allowed):
        raise RuntimeError(f"{name} 包含冻结股票池外标的")
    return symbols


def _strict_weights(
    name: str,
    value: object,
    *,
    allowed: frozenset[str],
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} 必须是 JSON 对象")
    weights: dict[str, float] = {}
    for raw_symbol, raw_weight in value.items():
        symbol = _strict_symbol(f"{name}.symbol", raw_symbol)
        if symbol not in allowed:
            raise RuntimeError(f"{name} 包含冻结股票池外标的: {symbol}")
        weights[symbol] = _strict_unit_interval(
            f"{name}[{symbol}]",
            raw_weight,
        )
    if sum(weights.values()) > 1.0 + _EPSILON:
        raise RuntimeError(f"{name} 权重合计不能超过 1")
    return weights


def _universe_from_payload(
    universe: dict[str, Any],
) -> UniverseContract:
    base_kwargs = {
        "universe_id": universe["universe_id"],
        "snapshot_date": universe["snapshot_date"],
        "constituent_count": universe["constituent_count"],
        "universe_sha256": universe["universe_sha256"],
        "symbols": tuple(universe["symbols"]),
        "contract_type": universe["contract_type"],
        "query_mode": universe["query_mode"],
        "point_in_time_safe": universe["point_in_time_safe"],
        "sealed_oos_eligible": universe["sealed_oos_eligible"],
        "provenance_identity": universe["provenance_identity"],
        "contract_sha256": universe["contract_sha256"],
    }
    try:
        if universe["contract_type"] == UNIVERSE_CONTRACT_TYPE_HISTORICAL:
            raw_constituents = universe["constituents"]
            if not isinstance(raw_constituents, list):
                raise RuntimeError("historical constituents 必须是列表")
            constituents = tuple(
                WeightedUniverseConstituent(
                    **_strict_object(
                        f"historical constituents[{index}]",
                        raw_row,
                        _HISTORICAL_CONSTITUENT_KEYS,
                    )
                )
                for index, raw_row in enumerate(raw_constituents)
            )
            contract: UniverseContract = HistoricalUniverseContract(
                **base_kwargs,
                as_of_date=universe["as_of_date"],
                source_trust_policy=universe["source_trust_policy"],
                source_effective_date=universe["source_effective_date"],
                source_effective_until_exclusive=universe[
                    "source_effective_until_exclusive"
                ],
                observed_at=universe["observed_at"],
                receipt_at=universe["receipt_at"],
                strict_available_at=universe["strict_available_at"],
                reconstructed=universe["reconstructed"],
                source_data_sha256=universe["source_data_sha256"],
                source_receipt_sha256=universe["source_receipt_sha256"],
                constituents=constituents,
                source_history_root=universe["source_history_root"],
                query_at=universe["query_at"],
            )
        else:
            contract = UniverseContract(**base_kwargs)
        contract.validate_contract_identity()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("universe canonical 合同校验失败") from exc
    if contract.to_dict() != universe:
        raise RuntimeError("universe payload 无法无损重建")
    return contract


def _validate_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _strict_object("组合决策 payload", payload, _TOP_LEVEL_KEYS)
    decision_id = _strict_text("decision_id", payload["decision_id"])
    if _DECISION_ID_RE.fullmatch(decision_id) is None:
        raise RuntimeError("decision_id 格式非法")

    raw_universe = payload["universe"]
    if not isinstance(raw_universe, dict):
        raise RuntimeError("universe 必须是 JSON 对象")
    expected_universe_keys = (
        _HISTORICAL_UNIVERSE_KEYS
        if raw_universe.get("contract_type")
        == UNIVERSE_CONTRACT_TYPE_HISTORICAL
        else _UNIVERSE_KEYS
    )
    universe = _strict_object(
        "universe",
        raw_universe,
        expected_universe_keys,
    )
    _strict_text("universe.universe_id", universe["universe_id"])
    snapshot_date = _strict_text(
        "universe.snapshot_date",
        universe["snapshot_date"],
    )
    if _DATE_RE.fullmatch(snapshot_date) is None:
        raise RuntimeError("universe.snapshot_date 必须是 YYYYMMDD")
    constituent_count = _strict_integer(
        "universe.constituent_count",
        universe["constituent_count"],
    )
    _strict_sha256(
        "universe.universe_sha256",
        universe["universe_sha256"],
    )
    raw_universe_symbols = universe["symbols"]
    if not isinstance(raw_universe_symbols, list):
        raise RuntimeError("universe.symbols 必须是列表")
    universe_symbols = tuple(
        _strict_symbol(f"universe.symbols[{index}]", symbol)
        for index, symbol in enumerate(raw_universe_symbols)
    )
    if len(universe_symbols) != constituent_count:
        raise RuntimeError("universe.constituent_count 与 symbols 数量不一致")
    if len(set(universe_symbols)) != len(universe_symbols):
        raise RuntimeError("universe.symbols 不能包含重复股票")
    contract_type = _strict_text(
        "universe.contract_type",
        universe["contract_type"],
    )
    query_mode = _strict_text("universe.query_mode", universe["query_mode"])
    point_in_time_safe = universe["point_in_time_safe"]
    sealed_oos_eligible = universe["sealed_oos_eligible"]
    if not isinstance(point_in_time_safe, bool):
        raise RuntimeError("universe.point_in_time_safe 必须是布尔值")
    if not isinstance(sealed_oos_eligible, bool):
        raise RuntimeError("universe.sealed_oos_eligible 必须是布尔值")
    provenance_identity = _strict_text(
        "universe.provenance_identity",
        universe["provenance_identity"],
    )
    contract_sha256 = _strict_sha256(
        "universe.contract_sha256",
        universe["contract_sha256"],
    )
    _universe_from_payload(universe)
    allowed = frozenset(universe_symbols)

    policy = _strict_object("policy", payload["policy"], _POLICY_KEYS)
    top_k = _strict_integer("policy.top_k", policy["top_k"])
    dropout_rank = _strict_integer(
        "policy.dropout_rank",
        policy["dropout_rank"],
    )
    _strict_integer(
        "policy.minimum_history",
        policy["minimum_history"],
        minimum=2,
    )
    _strict_unit_interval(
        "policy.minimum_model_exposure",
        policy["minimum_model_exposure"],
    )
    _strict_unit_interval(
        "policy.minimum_confidence",
        policy["minimum_confidence"],
    )
    _strict_unit_interval(
        "policy.minimum_calibrated_score",
        policy["minimum_calibrated_score"],
    )
    gross_exposure = _strict_unit_interval(
        "policy.gross_exposure",
        policy["gross_exposure"],
    )
    if gross_exposure <= 0.0:
        raise RuntimeError("policy.gross_exposure 必须大于 0")
    if top_k > constituent_count or not top_k <= dropout_rank <= constituent_count:
        raise RuntimeError("policy 的 top_k 或 dropout_rank 超出股票池范围")

    _strict_integer("bar_ts", payload["bar_ts"])
    session_date = _strict_text("session_date", payload["session_date"])
    if _SESSION_DATE_RE.fullmatch(session_date) is None:
        raise RuntimeError("session_date 必须是 YYYY-MM-DD")
    _strict_text("timeframe", payload["timeframe"])
    _strict_text("market_source", payload["market_source"])
    _strict_sha256(
        "account_snapshot_sha256",
        payload["account_snapshot_sha256"],
    )

    current_weights = _strict_weights(
        "current_weights",
        payload["current_weights"],
        allowed=allowed,
    )
    target_weights = _strict_weights(
        "target_weights",
        payload["target_weights"],
        allowed=allowed,
    )
    selected = _strict_symbol_list(
        "selected_symbols",
        payload["selected_symbols"],
        allowed=allowed,
    )
    entered = _strict_symbol_list(
        "entered_symbols",
        payload["entered_symbols"],
        allowed=allowed,
    )
    retained = _strict_symbol_list(
        "retained_symbols",
        payload["retained_symbols"],
        allowed=allowed,
    )
    exited = _strict_symbol_list(
        "exited_symbols",
        payload["exited_symbols"],
        allowed=allowed,
    )
    if set(selected) != set(target_weights):
        raise RuntimeError("selected_symbols 与 target_weights 不一致")
    if set(entered) | set(retained) != set(selected):
        raise RuntimeError(
            "entered_symbols、retained_symbols 与 selected_symbols 不一致"
        )
    if set(entered) & set(retained):
        raise RuntimeError("entered_symbols 与 retained_symbols 不能重叠")
    if set(exited) & set(selected):
        raise RuntimeError("exited_symbols 与 selected_symbols 不能重叠")
    if not set(retained).issubset(current_weights):
        raise RuntimeError("retained_symbols 必须来自当前持仓")
    cash_weight = _strict_unit_interval("cash_weight", payload["cash_weight"])
    if not math.isclose(
        sum(target_weights.values()) + cash_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=_EPSILON,
    ):
        raise RuntimeError("target_weights 与 cash_weight 合计必须为 1")

    ranking = payload["ranking"]
    if not isinstance(ranking, list):
        raise RuntimeError("ranking 必须是列表")
    if len(ranking) != constituent_count:
        raise RuntimeError("ranking 必须完整覆盖冻结股票池")
    ranking_symbols: list[str] = []
    ranking_runs: list[str] = []
    ranks: list[int] = []
    for index, raw_row in enumerate(ranking):
        row = _strict_object(
            f"ranking[{index}]",
            raw_row,
            _RANKING_KEYS,
        )
        run_id = _strict_text(f"ranking[{index}].run_id", row["run_id"])
        if _RUN_ID_RE.fullmatch(run_id) is None:
            raise RuntimeError(f"ranking[{index}].run_id 格式非法")
        ranking_runs.append(run_id)
        ranking_symbols.append(
            _strict_symbol(f"ranking[{index}].symbol", row["symbol"])
        )
        _strict_real(f"ranking[{index}].raw_score", row["raw_score"])
        _strict_unit_interval(
            f"ranking[{index}].calibrated_score",
            row["calibrated_score"],
        )
        _strict_unit_interval(
            f"ranking[{index}].confidence",
            row["confidence"],
        )
        _strict_unit_interval(
            f"ranking[{index}].requested_exposure",
            row["requested_exposure"],
        )
        _strict_sha256(
            f"ranking[{index}].model_version",
            row["model_version"],
        )
        _strict_sha256(
            f"ranking[{index}].data_version",
            row["data_version"],
        )
        _strict_text(
            f"ranking[{index}].calibration_version",
            row["calibration_version"],
        )
        _strict_sha256(
            f"ranking[{index}].calibration_history_sha256",
            row["calibration_history_sha256"],
        )
        if not isinstance(row["eligible"], bool):
            raise RuntimeError(f"ranking[{index}].eligible 必须是布尔值")
        if not isinstance(row["rejection_reason"], str):
            raise RuntimeError(f"ranking[{index}].rejection_reason 必须是文本")
        rank = row["rank"]
        if rank is not None:
            ranks.append(
                _strict_integer(
                    f"ranking[{index}].rank",
                    rank,
                )
            )
    if frozenset(ranking_symbols) != allowed:
        raise RuntimeError("ranking 股票集合与冻结股票池不一致")
    if len(set(ranking_runs)) != len(ranking_runs):
        raise RuntimeError("ranking 不能复用同一训练 run")
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("ranking.rank 不能重复")

    identity = dict(payload)
    identity.pop("decision_id")
    if _canonical_decision_id(identity) != decision_id:
        raise RuntimeError("组合决策 payload 与 decision_id 不一致")
    return payload


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("执行账本 payload 不能规范化为 JSON") from exc


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _decode_json_object(name: str, raw: object) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise RuntimeError(f"{name} 不是 JSON 文本")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} 不是合法 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} 不是 JSON 对象")
    return payload


def _decision_from_payload(payload: dict[str, Any]) -> PortfolioDecision:
    validated = _validate_decision_payload(payload)
    universe = _universe_from_payload(validated["universe"])
    policy = PortfolioPolicy(**dict(validated["policy"]))
    ranking = tuple(
        CalibratedSignal(**dict(row)) for row in validated["ranking"]
    )
    decision = PortfolioDecision(
        decision_id=validated["decision_id"],
        universe=universe,
        policy=policy,
        bar_ts=validated["bar_ts"],
        session_date=validated["session_date"],
        timeframe=validated["timeframe"],
        market_source=validated["market_source"],
        account_snapshot_sha256=validated["account_snapshot_sha256"],
        current_weights=tuple(validated["current_weights"].items()),
        selected_symbols=tuple(validated["selected_symbols"]),
        target_weights=tuple(validated["target_weights"].items()),
        entered_symbols=tuple(validated["entered_symbols"]),
        retained_symbols=tuple(validated["retained_symbols"]),
        exited_symbols=tuple(validated["exited_symbols"]),
        cash_weight=validated["cash_weight"],
        ranking=ranking,
    )
    if decision.to_dict() != validated:
        raise RuntimeError("组合决策 payload 无法无损重建")
    return decision


def _execution_types():
    # execution.py 在执行函数内部导入本模块；这里延迟导入以避免循环依赖。
    from portfolio_manager.execution import (
        AShareFeeSchedule,
        ExecutionQuote,
        PortfolioExecutionResult,
        PositionLot,
        VirtualAccount,
        execute_portfolio_decision,
    )

    return (
        AShareFeeSchedule,
        ExecutionQuote,
        PortfolioExecutionResult,
        PositionLot,
        VirtualAccount,
        execute_portfolio_decision,
    )


def _account_from_payload(payload: object):
    _, _, _, PositionLot, VirtualAccount, _ = _execution_types()
    account_data = _strict_object(
        "执行账户",
        payload,
        _ACCOUNT_KEYS,
    )
    raw_lots = account_data["lots"]
    if not isinstance(raw_lots, list):
        raise RuntimeError("执行账户 lots 必须是列表")
    lots = []
    for index, raw_lot in enumerate(raw_lots):
        lot_data = _strict_object(
            f"执行账户 lots[{index}]",
            raw_lot,
            _POSITION_LOT_KEYS,
        )
        try:
            lot = PositionLot(**lot_data)
            lot.validated()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"执行账户 lots[{index}] 非法") from exc
        lots.append(lot)
    try:
        account = VirtualAccount(
            cash=account_data["cash"],
            lots=tuple(lots),
        )
        canonical = account.to_dict()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("执行账户 payload 非法") from exc
    if canonical != account_data:
        raise RuntimeError("执行账户 payload 不是 canonical 形式")
    return account


def _quotes_from_payload(name: str, payload: object):
    _, ExecutionQuote, _, _, _, _ = _execution_types()
    if not isinstance(payload, list):
        raise RuntimeError(f"{name} 必须是列表")
    quotes = []
    for index, raw_quote in enumerate(payload):
        quote_data = _strict_object(
            f"{name}[{index}]",
            raw_quote,
            _QUOTE_KEYS,
        )
        try:
            quote = ExecutionQuote(**quote_data)
            quote.validated()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{name}[{index}] 非法") from exc
        quotes.append(quote)
    symbols = [quote.symbol for quote in quotes]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise RuntimeError(f"{name} 必须按代码排序且不能重复")
    return tuple(quotes)


def _fee_from_payload(payload: object):
    AShareFeeSchedule, _, _, _, _, _ = _execution_types()
    fee_data = _strict_object("执行费用", payload, _FEE_KEYS)
    try:
        schedule = AShareFeeSchedule(**fee_data)
        schedule.validated()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("执行费用 payload 非法") from exc
    if schedule.to_dict() != fee_data:
        raise RuntimeError("执行费用 payload 不是 canonical 形式")
    return schedule


def _execution_input_payload(
    *,
    decision_id: str,
    execution_session: str,
    account_before: object,
    decision_quotes: tuple[object, ...],
    execution_quotes: tuple[object, ...],
    fee_schedule: object,
) -> dict[str, Any]:
    return {
        "contract_version": _EXECUTION_LEDGER_VERSION,
        "decision_id": decision_id,
        "execution_session": execution_session,
        "account_before": account_before.to_dict(),
        "decision_quotes": [
            quote.to_dict()
            for quote in sorted(
                decision_quotes,
                key=lambda item: item.symbol,
            )
        ],
        "execution_quotes": [
            quote.to_dict()
            for quote in sorted(
                execution_quotes,
                key=lambda item: item.symbol,
            )
        ],
        "fee_schedule": fee_schedule.to_dict(),
    }


def _fill_payloads(
    execution_id: str,
    orders: tuple[object, ...],
) -> tuple[dict[str, Any], ...]:
    fills: list[dict[str, Any]] = []
    for order in orders:
        if order.filled_shares <= 0:
            continue
        fill_index = len(fills) + 1
        fills.append(
            {
                "fill_id": f"{order.order_id}-FILL-001",
                "execution_id": execution_id,
                "order_id": order.order_id,
                "fill_index": fill_index,
                "symbol": order.symbol,
                "side": order.side,
                "filled_shares": order.filled_shares,
                "fill_price": order.fill_price,
                "gross_amount": order.gross_amount,
                "fees": order.fees,
                "cash_change": order.cash_change,
            }
        )
    return tuple(fills)


def _execution_identity_payload(
    *,
    decision_payload: dict[str, Any],
    input_payload: dict[str, Any],
    result_payload: dict[str, Any],
    order_payloads: tuple[dict[str, Any], ...],
    fill_payloads: tuple[dict[str, Any], ...],
    bootstrap_account: bool,
) -> dict[str, Any]:
    return {
        "contract_version": _EXECUTION_LEDGER_VERSION,
        "bootstrap_account": bootstrap_account,
        "decision": decision_payload,
        "input": input_payload,
        "result": result_payload,
        "orders": list(order_payloads),
        "fills": list(fill_payloads),
        "account_before_sha256": _payload_sha256(
            input_payload["account_before"]
        ),
        "account_after_sha256": _payload_sha256(
            result_payload["account_after"]
        ),
    }


def _execution_row_identity_payload(
    *,
    execution_id: str,
    decision_id: str,
    decision_session: str,
    execution_session: str,
    bootstrap_account: bool,
    created_at: float,
    input_payload: dict[str, Any],
    input_sha256: str,
    result_payload: dict[str, Any],
    result_sha256: str,
    identity_sha256: str,
    account_before_sha256: str,
    account_after_sha256: str,
    order_count: int,
    fill_count: int,
) -> dict[str, Any]:
    return {
        "contract_version": _EXECUTION_LEDGER_VERSION,
        "execution_id": execution_id,
        "decision_id": decision_id,
        "decision_session": decision_session,
        "execution_session": execution_session,
        "bootstrap_account": bootstrap_account,
        "created_at": created_at,
        "input": input_payload,
        "input_sha256": input_sha256,
        "result": result_payload,
        "result_sha256": result_sha256,
        "identity_sha256": identity_sha256,
        "account_before_sha256": account_before_sha256,
        "account_after_sha256": account_after_sha256,
        "order_count": order_count,
        "fill_count": fill_count,
    }


def _account_state_identity_payload(
    *,
    execution_id: str,
    execution_session: str,
    account_payload: dict[str, Any],
    account_sha256: str,
    updated_at: float,
    execution_row_sha256: str,
) -> dict[str, Any]:
    return {
        "contract_version": _EXECUTION_LEDGER_VERSION,
        "singleton_id": 1,
        "execution_id": execution_id,
        "execution_session": execution_session,
        "account": account_payload,
        "account_sha256": account_sha256,
        "updated_at": updated_at,
        "execution_row_sha256": execution_row_sha256,
    }


class PortfolioDecisionLedger:
    """原子记录组合决策、执行、订单、成交与当前账户。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS portfolio_decisions (
                    decision_id TEXT PRIMARY KEY,
                    bar_ts INTEGER NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS target_portfolio_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    decision_id TEXT NOT NULL,
                    bar_ts INTEGER NOT NULL,
                    target_weights_json TEXT NOT NULL,
                    cash_weight REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (decision_id)
                        REFERENCES portfolio_decisions(decision_id)
                );

                CREATE INDEX IF NOT EXISTS idx_portfolio_decisions_bar_ts
                ON portfolio_decisions(bar_ts DESC);

                CREATE TABLE IF NOT EXISTS portfolio_executions (
                    execution_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL UNIQUE,
                    decision_session TEXT NOT NULL,
                    execution_session TEXT NOT NULL UNIQUE,
                    bootstrap_account INTEGER NOT NULL
                        CHECK (bootstrap_account IN (0, 1)),
                    created_at REAL NOT NULL,
                    input_payload_json TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    result_payload_json TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    identity_sha256 TEXT NOT NULL,
                    row_sha256 TEXT NOT NULL,
                    account_before_sha256 TEXT NOT NULL,
                    account_after_sha256 TEXT NOT NULL,
                    order_count INTEGER NOT NULL CHECK (order_count >= 0),
                    fill_count INTEGER NOT NULL CHECK (fill_count >= 0),
                    FOREIGN KEY (decision_id)
                        REFERENCES portfolio_decisions(decision_id)
                );

                CREATE TABLE IF NOT EXISTS portfolio_execution_orders (
                    order_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL CHECK (order_index > 0),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE (execution_id, order_index),
                    UNIQUE (order_id, execution_id),
                    FOREIGN KEY (execution_id)
                        REFERENCES portfolio_executions(execution_id)
                );

                CREATE TABLE IF NOT EXISTS portfolio_execution_fills (
                    fill_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    order_id TEXT NOT NULL UNIQUE,
                    fill_index INTEGER NOT NULL CHECK (fill_index > 0),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE (execution_id, fill_index),
                    FOREIGN KEY (order_id, execution_id)
                        REFERENCES portfolio_execution_orders(
                            order_id, execution_id
                        )
                );

                CREATE TABLE IF NOT EXISTS portfolio_account_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    execution_id TEXT NOT NULL UNIQUE,
                    execution_session TEXT NOT NULL,
                    account_payload_json TEXT NOT NULL,
                    account_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    FOREIGN KEY (execution_id)
                        REFERENCES portfolio_executions(execution_id)
                );

                CREATE INDEX IF NOT EXISTS idx_portfolio_executions_session
                ON portfolio_executions(execution_session DESC);

                CREATE TABLE IF NOT EXISTS portfolio_replay_bindings (
                    execution_id TEXT PRIMARY KEY,
                    replay_run_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    execution_row_sha256 TEXT NOT NULL,
                    FOREIGN KEY (execution_id)
                        REFERENCES portfolio_executions(execution_id)
                );
                """
            )
            required_audit_columns = (
                ("portfolio_executions", "row_sha256"),
                ("portfolio_account_state", "state_sha256"),
            )
            missing_columns: list[tuple[str, str]] = []
            for table, column in required_audit_columns:
                columns = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in columns:
                    if conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]:
                        raise RuntimeError(
                            "非空执行账本缺少审计身份列，拒绝自动迁移"
                        )
                    missing_columns.append((table, column))
            for table, column in missing_columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    @staticmethod
    def _decode_payload(
        raw: str,
        *,
        expected_decision_id: str | None = None,
        expected_bar_ts: int | None = None,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("组合决策账本 payload 不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("组合决策账本 payload 不是 JSON 对象")
        validated = _validate_decision_payload(payload)
        if (
            validated["universe"]["contract_type"]
            == UNIVERSE_CONTRACT_TYPE_UNTRUSTED
        ):
            raise RuntimeError("untrusted universe 不得从组合决策账本读取")
        if (
            expected_decision_id is not None
            and validated["decision_id"] != expected_decision_id
        ):
            raise RuntimeError("组合决策 payload 与账本 decision_id 不一致")
        if expected_bar_ts is not None and validated["bar_ts"] != expected_bar_ts:
            raise RuntimeError("组合决策 payload 与账本 bar_ts 不一致")
        return validated

    def record_decision(
        self,
        decision: PortfolioDecision,
    ) -> tuple[dict[str, Any], bool]:
        """写入一次决策；完全相同的重复请求返回原记录。"""
        if (
            decision.universe.contract_type
            == UNIVERSE_CONTRACT_TYPE_UNTRUSTED
        ):
            raise RuntimeError("untrusted universe 不得写入组合决策账本")
        payload = decision.to_dict()
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT decision_id, bar_ts, payload_json
                FROM portfolio_decisions
                WHERE decision_id = ?
                """,
                (decision.decision_id,),
            ).fetchone()
            if existing is not None:
                stored = self._decode_payload(
                    existing["payload_json"],
                    expected_decision_id=existing["decision_id"],
                    expected_bar_ts=existing["bar_ts"],
                )
                if stored != payload:
                    raise RuntimeError("同一 decision_id 对应不同组合决策内容")
                return stored, False
            _validate_decision_payload(payload)

            same_bar = conn.execute(
                """
                SELECT decision_id
                FROM portfolio_decisions
                WHERE bar_ts = ?
                """,
                (decision.bar_ts,),
            ).fetchone()
            if same_bar is not None:
                raise ValueError(
                    f"同一根 K 线已经存在不同组合决策: {same_bar['decision_id']}"
                )

            latest = conn.execute(
                """
                SELECT MAX(bar_ts) AS bar_ts
                FROM portfolio_decisions
                """
            ).fetchone()
            if (
                latest is not None
                and latest["bar_ts"] is not None
                and int(decision.bar_ts) <= int(latest["bar_ts"])
            ):
                raise ValueError(
                    f"组合决策时间未前进: {decision.bar_ts} <= {int(latest['bar_ts'])}"
                )

            conn.execute(
                """
                INSERT INTO portfolio_decisions (
                    decision_id, bar_ts, created_at, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    int(decision.bar_ts),
                    now,
                    payload_json,
                ),
            )
            target_weights_json = json.dumps(
                dict(decision.target_weights),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            conn.execute(
                """
                INSERT INTO target_portfolio_state (
                    singleton_id, decision_id, bar_ts,
                    target_weights_json, cash_weight, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    decision_id = excluded.decision_id,
                    bar_ts = excluded.bar_ts,
                    target_weights_json = excluded.target_weights_json,
                    cash_weight = excluded.cash_weight,
                    updated_at = excluded.updated_at
                """,
                (
                    decision.decision_id,
                    int(decision.bar_ts),
                    target_weights_json,
                    float(decision.cash_weight),
                    now,
                ),
            )
            return payload, True

    @staticmethod
    def _decision_from_connection(
        conn: sqlite3.Connection,
        decision_id: str,
    ) -> tuple[dict[str, Any], PortfolioDecision]:
        row = conn.execute(
            """
            SELECT decision_id, bar_ts, payload_json
            FROM portfolio_decisions
            WHERE decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"未知组合决策: {decision_id}")
        payload = PortfolioDecisionLedger._decode_payload(
            row["payload_json"],
            expected_decision_id=row["decision_id"],
            expected_bar_ts=row["bar_ts"],
        )
        return payload, _decision_from_payload(payload)

    @staticmethod
    def _decode_execution_input(
        raw: object,
        *,
        expected_decision_id: str,
        expected_execution_session: str,
    ) -> tuple[
        dict[str, Any],
        object,
        tuple[object, ...],
        tuple[object, ...],
        object,
    ]:
        payload = _strict_object(
            "执行输入",
            _decode_json_object("执行输入", raw),
            _EXECUTION_INPUT_KEYS,
        )
        if payload["contract_version"] != _EXECUTION_LEDGER_VERSION:
            raise RuntimeError("执行输入合同版本不一致")
        if payload["decision_id"] != expected_decision_id:
            raise RuntimeError("执行输入与 decision_id 不一致")
        if payload["execution_session"] != expected_execution_session:
            raise RuntimeError("执行输入与 execution_session 不一致")
        account = _account_from_payload(payload["account_before"])
        decision_quotes = _quotes_from_payload(
            "decision_quotes",
            payload["decision_quotes"],
        )
        execution_quotes = _quotes_from_payload(
            "execution_quotes",
            payload["execution_quotes"],
        )
        fee_schedule = _fee_from_payload(payload["fee_schedule"])
        canonical = _execution_input_payload(
            decision_id=expected_decision_id,
            execution_session=expected_execution_session,
            account_before=account,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            fee_schedule=fee_schedule,
        )
        if payload != canonical or raw != _canonical_json(canonical):
            raise RuntimeError("执行输入 payload 不是 canonical 形式")
        return (
            canonical,
            account,
            decision_quotes,
            execution_quotes,
            fee_schedule,
        )

    @staticmethod
    def _foreign_keys_fail_closed(conn: sqlite3.Connection) -> None:
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("执行账本外键完整性校验失败")

    @classmethod
    def _read_execution_row(
        cls,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        (
            _,
            _,
            PortfolioExecutionResult,
            _,
            _,
            execute_portfolio_decision,
        ) = _execution_types()
        execution_id = _strict_text("execution_id", row["execution_id"])
        if _EXECUTION_ID_RE.fullmatch(execution_id) is None:
            raise RuntimeError("execution_id 格式非法")
        decision_id = _strict_text("execution.decision_id", row["decision_id"])
        decision_payload, decision = cls._decision_from_connection(
            conn,
            decision_id,
        )
        decision_session = _strict_text(
            "execution.decision_session",
            row["decision_session"],
        )
        execution_session = _strict_text(
            "execution.execution_session",
            row["execution_session"],
        )
        bootstrap_value = row["bootstrap_account"]
        if (
            isinstance(bootstrap_value, bool)
            or not isinstance(bootstrap_value, Integral)
            or int(bootstrap_value) not in (0, 1)
        ):
            raise RuntimeError("bootstrap_account 账本值非法")
        bootstrap_account = bool(bootstrap_value)
        created_at = _strict_real("execution.created_at", row["created_at"])
        input_sha256 = _strict_sha256(
            "execution.input_sha256",
            row["input_sha256"],
        )
        result_sha256 = _strict_sha256(
            "execution.result_sha256",
            row["result_sha256"],
        )
        identity_sha256 = _strict_sha256(
            "execution.identity_sha256",
            row["identity_sha256"],
        )
        row_sha256 = _strict_sha256(
            "execution.row_sha256",
            row["row_sha256"],
        )
        account_before_sha256 = _strict_sha256(
            "execution.account_before_sha256",
            row["account_before_sha256"],
        )
        account_after_sha256 = _strict_sha256(
            "execution.account_after_sha256",
            row["account_after_sha256"],
        )
        order_count = _strict_integer(
            "execution.order_count",
            row["order_count"],
            minimum=0,
        )
        fill_count = _strict_integer(
            "execution.fill_count",
            row["fill_count"],
            minimum=0,
        )
        raw_input = row["input_payload_json"]
        if (
            not isinstance(raw_input, str)
            or hashlib.sha256(raw_input.encode("utf-8")).hexdigest()
            != input_sha256
        ):
            raise RuntimeError("执行输入 payload 哈希不一致")
        (
            input_payload,
            account_before,
            decision_quotes,
            execution_quotes,
            fee_schedule,
        ) = cls._decode_execution_input(
            raw_input,
            expected_decision_id=decision_id,
            expected_execution_session=execution_session,
        )
        try:
            recomputed = execute_portfolio_decision(
                decision,
                execution_session=execution_session,
                account=account_before,
                decision_quotes=decision_quotes,
                quotes=execution_quotes,
                fee_schedule=fee_schedule,
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("执行账本输入无法重算") from exc
        if not isinstance(recomputed, PortfolioExecutionResult):
            raise RuntimeError("执行重算没有返回 PortfolioExecutionResult")
        result_payload = recomputed.to_dict()
        raw_result = row["result_payload_json"]
        canonical_result = _canonical_json(result_payload)
        if (
            not isinstance(raw_result, str)
            or hashlib.sha256(raw_result.encode("utf-8")).hexdigest()
            != result_sha256
            or raw_result != canonical_result
        ):
            raise RuntimeError("执行结果 payload 与重算结果不一致")
        if (
            recomputed.execution_id != execution_id
            or recomputed.decision_id != decision_id
            or recomputed.decision_session != decision_session
            or recomputed.execution_session != execution_session
            or decision.session_date != decision_session
        ):
            raise RuntimeError("执行结果身份与账本主键不一致")

        expected_orders = tuple(order.to_dict() for order in recomputed.orders)
        order_rows = conn.execute(
            """
            SELECT order_id, execution_id, order_index,
                   payload_json, payload_sha256
            FROM portfolio_execution_orders
            WHERE execution_id = ?
            ORDER BY order_index
            """,
            (execution_id,),
        ).fetchall()
        if len(order_rows) != order_count or order_count != len(expected_orders):
            raise RuntimeError("执行订单行数与重算结果不一致")
        for expected_index, (order_row, expected_order) in enumerate(
            zip(order_rows, expected_orders),
            start=1,
        ):
            raw_order = order_row["payload_json"]
            canonical_order = _canonical_json(expected_order)
            if (
                order_row["execution_id"] != execution_id
                or order_row["order_index"] != expected_index
                or order_row["order_id"] != expected_order["order_id"]
                or not isinstance(raw_order, str)
                or hashlib.sha256(raw_order.encode("utf-8")).hexdigest()
                != _strict_sha256(
                    "order.payload_sha256",
                    order_row["payload_sha256"],
                )
                or raw_order != canonical_order
            ):
                raise RuntimeError("执行订单身份、顺序或 payload 不一致")

        expected_fills = _fill_payloads(execution_id, recomputed.orders)
        fill_rows = conn.execute(
            """
            SELECT fill_id, execution_id, order_id, fill_index,
                   payload_json, payload_sha256
            FROM portfolio_execution_fills
            WHERE execution_id = ?
            ORDER BY fill_index
            """,
            (execution_id,),
        ).fetchall()
        if len(fill_rows) != fill_count or fill_count != len(expected_fills):
            raise RuntimeError("执行成交行数与重算结果不一致")
        for expected_index, (fill_row, expected_fill) in enumerate(
            zip(fill_rows, expected_fills),
            start=1,
        ):
            raw_fill = fill_row["payload_json"]
            canonical_fill = _canonical_json(expected_fill)
            if (
                fill_row["execution_id"] != execution_id
                or fill_row["fill_index"] != expected_index
                or fill_row["fill_id"] != expected_fill["fill_id"]
                or fill_row["order_id"] != expected_fill["order_id"]
                or not isinstance(raw_fill, str)
                or hashlib.sha256(raw_fill.encode("utf-8")).hexdigest()
                != _strict_sha256(
                    "fill.payload_sha256",
                    fill_row["payload_sha256"],
                )
                or raw_fill != canonical_fill
            ):
                raise RuntimeError("执行成交身份、顺序或 payload 不一致")

        expected_account_before_sha256 = _payload_sha256(
            input_payload["account_before"]
        )
        expected_account_after_sha256 = _payload_sha256(
            result_payload["account_after"]
        )
        if (
            account_before_sha256 != expected_account_before_sha256
            or account_after_sha256 != expected_account_after_sha256
        ):
            raise RuntimeError("执行账户前后身份不一致")
        identity_payload = _execution_identity_payload(
            decision_payload=decision_payload,
            input_payload=input_payload,
            result_payload=result_payload,
            order_payloads=expected_orders,
            fill_payloads=expected_fills,
            bootstrap_account=bootstrap_account,
        )
        if _payload_sha256(identity_payload) != identity_sha256:
            raise RuntimeError("执行 canonical identity 不一致")
        row_identity_payload = _execution_row_identity_payload(
            execution_id=execution_id,
            decision_id=decision_id,
            decision_session=decision_session,
            execution_session=execution_session,
            bootstrap_account=bootstrap_account,
            created_at=created_at,
            input_payload=input_payload,
            input_sha256=input_sha256,
            result_payload=result_payload,
            result_sha256=result_sha256,
            identity_sha256=identity_sha256,
            account_before_sha256=account_before_sha256,
            account_after_sha256=account_after_sha256,
            order_count=order_count,
            fill_count=fill_count,
        )
        if _payload_sha256(row_identity_payload) != row_sha256:
            raise RuntimeError("执行行审计身份不一致")
        return {
            "execution_id": execution_id,
            "decision_id": decision_id,
            "decision_bar_ts": decision_payload["bar_ts"],
            "decision_session": decision_session,
            "execution_session": execution_session,
            "bootstrap_account": bootstrap_account,
            "created_at": created_at,
            "identity_sha256": identity_sha256,
            "row_sha256": row_sha256,
            "account_before_sha256": account_before_sha256,
            "account_after_sha256": account_after_sha256,
            "input": input_payload,
            "result": result_payload,
            "orders": list(expected_orders),
            "fills": list(expected_fills),
        }

    @classmethod
    def _verified_execution_chain(
        cls,
        conn: sqlite3.Connection,
    ) -> list[dict[str, Any]]:
        cls._foreign_keys_fail_closed(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM portfolio_executions
            ORDER BY execution_session, execution_id
            """
        ).fetchall()
        records = [cls._read_execution_row(conn, row) for row in rows]
        previous: dict[str, Any] | None = None
        for index, record in enumerate(records):
            if index == 0 and record["bootstrap_account"] is not True:
                raise RuntimeError("首笔执行缺少 bootstrap 账户声明")
            if index > 0 and record["bootstrap_account"] is not False:
                raise RuntimeError("非首笔执行不能重新 bootstrap 账户")
            if record["execution_session"] <= record["decision_session"]:
                raise RuntimeError("执行日期没有晚于组合决策日期")
            if previous is not None:
                if (
                    record["decision_session"]
                    <= previous["decision_session"]
                ):
                    raise RuntimeError("组合决策日期未严格前进")
                if (
                    record["decision_bar_ts"]
                    <= previous["decision_bar_ts"]
                ):
                    raise RuntimeError("组合决策 bar_ts 未严格前进")
                if (
                    record["decision_session"]
                    < previous["execution_session"]
                ):
                    raise RuntimeError("组合决策日期早于前次执行日期")
                if (
                    record["execution_session"]
                    <= previous["execution_session"]
                ):
                    raise RuntimeError("执行日期未严格前进")
                if (
                    _canonical_json(record["input"]["account_before"])
                    != _canonical_json(previous["result"]["account_after"])
                ):
                    raise RuntimeError("连续执行的账户前后状态不一致")
            previous = record
        return records

    @classmethod
    def _verified_account_state(
        cls,
        conn: sqlite3.Connection,
        records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT singleton_id, execution_id, execution_session,
                   account_payload_json, account_sha256, updated_at,
                   state_sha256
            FROM portfolio_account_state
            WHERE singleton_id = 1
            """
        ).fetchone()
        if not records:
            if row is not None:
                raise RuntimeError("没有执行历史却存在账户状态")
            return None
        if row is None:
            raise RuntimeError("执行历史存在，但当前账户状态缺失")
        latest = records[-1]
        if (
            row["singleton_id"] != 1
            or row["execution_id"] != latest["execution_id"]
            or row["execution_session"] != latest["execution_session"]
        ):
            raise RuntimeError("当前账户没有指向最新执行")
        raw_account = row["account_payload_json"]
        account_payload = latest["result"]["account_after"]
        canonical_account = _canonical_json(account_payload)
        account_sha256 = _strict_sha256(
            "account_state.account_sha256",
            row["account_sha256"],
        )
        if (
            not isinstance(raw_account, str)
            or raw_account != canonical_account
            or hashlib.sha256(raw_account.encode("utf-8")).hexdigest()
            != account_sha256
            or account_sha256 != latest["account_after_sha256"]
        ):
            raise RuntimeError("当前账户 payload 或身份与最新执行不一致")
        updated_at = _strict_real(
            "account_state.updated_at",
            row["updated_at"],
        )
        state_sha256 = _strict_sha256(
            "account_state.state_sha256",
            row["state_sha256"],
        )
        state_identity_payload = _account_state_identity_payload(
            execution_id=latest["execution_id"],
            execution_session=latest["execution_session"],
            account_payload=account_payload,
            account_sha256=account_sha256,
            updated_at=updated_at,
            execution_row_sha256=latest["row_sha256"],
        )
        if (
            updated_at != latest["created_at"]
            or _payload_sha256(state_identity_payload) != state_sha256
        ):
            raise RuntimeError("当前账户审计身份不一致")
        return {
            "execution_id": latest["execution_id"],
            "execution_session": latest["execution_session"],
            "account": account_payload,
            "account_sha256": account_sha256,
            "updated_at": updated_at,
            "state_sha256": state_sha256,
        }

    def record_execution(
        self,
        result: object,
        *,
        decision_quotes: object,
        execution_quotes: object,
        fee_schedule: object,
        bootstrap_account: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """原子持久化一次确定性执行；同一完整 payload 重试幂等。"""
        (
            AShareFeeSchedule,
            ExecutionQuote,
            PortfolioExecutionResult,
            _,
            _,
            execute_portfolio_decision,
        ) = _execution_types()
        if not isinstance(result, PortfolioExecutionResult):
            raise ValueError("result 必须是 PortfolioExecutionResult")
        if not isinstance(bootstrap_account, bool):
            raise ValueError("bootstrap_account 必须是布尔值")
        if not isinstance(fee_schedule, AShareFeeSchedule):
            raise ValueError("fee_schedule 必须是 AShareFeeSchedule")
        raw_decision_quotes = tuple(decision_quotes)
        raw_execution_quotes = tuple(execution_quotes)
        if any(
            not isinstance(quote, ExecutionQuote)
            for quote in raw_decision_quotes + raw_execution_quotes
        ):
            raise ValueError("执行行情必须由 ExecutionQuote 组成")

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            decision_payload, decision = self._decision_from_connection(
                conn,
                result.decision_id,
            )
            try:
                recomputed = execute_portfolio_decision(
                    decision,
                    execution_session=result.execution_session,
                    account=result.account_before,
                    decision_quotes=raw_decision_quotes,
                    quotes=raw_execution_quotes,
                    fee_schedule=fee_schedule,
                )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError("传入执行条件无法从已记录决策重算") from exc
            result_payload = result.to_dict()
            recomputed_payload = recomputed.to_dict()
            if _canonical_json(result_payload) != _canonical_json(
                recomputed_payload
            ):
                raise RuntimeError("传入执行结果与账本重算结果不一致")
            if (
                result.execution_id != recomputed.execution_id
                or result.decision_id != decision.decision_id
                or result.decision_session != decision.session_date
            ):
                raise RuntimeError("传入执行结果身份与账本重算不一致")

            input_payload = _execution_input_payload(
                decision_id=decision.decision_id,
                execution_session=recomputed.execution_session,
                account_before=recomputed.account_before,
                decision_quotes=raw_decision_quotes,
                execution_quotes=raw_execution_quotes,
                fee_schedule=fee_schedule,
            )
            canonical_input = _canonical_json(input_payload)
            canonical_result = _canonical_json(recomputed_payload)
            order_payloads = tuple(
                order.to_dict() for order in recomputed.orders
            )
            fill_payloads = _fill_payloads(
                recomputed.execution_id,
                recomputed.orders,
            )
            identity_payload = _execution_identity_payload(
                decision_payload=decision_payload,
                input_payload=input_payload,
                result_payload=recomputed_payload,
                order_payloads=order_payloads,
                fill_payloads=fill_payloads,
                bootstrap_account=bootstrap_account,
            )
            identity_sha256 = _payload_sha256(identity_payload)

            existing_row = conn.execute(
                """
                SELECT *
                FROM portfolio_executions
                WHERE execution_id = ?
                """,
                (recomputed.execution_id,),
            ).fetchone()
            if existing_row is not None:
                records = self._verified_execution_chain(conn)
                self._verified_account_state(conn, records)
                existing = next(
                    record
                    for record in records
                    if record["execution_id"] == recomputed.execution_id
                )
                if (
                    _canonical_json(existing["input"]) != canonical_input
                    or _canonical_json(existing["result"]) != canonical_result
                    or existing["identity_sha256"] != identity_sha256
                ):
                    raise RuntimeError(
                        "同一 execution_id 对应不同 canonical identity"
                    )
                return existing, False

            same_decision = conn.execute(
                """
                SELECT execution_id
                FROM portfolio_executions
                WHERE decision_id = ?
                """,
                (decision.decision_id,),
            ).fetchone()
            if same_decision is not None:
                raise RuntimeError(
                    "同一组合决策已经存在执行结果: "
                    f"{same_decision['execution_id']}"
                )

            records = self._verified_execution_chain(conn)
            current_state = self._verified_account_state(conn, records)
            if not records:
                if not bootstrap_account:
                    raise RuntimeError("首笔执行必须显式 bootstrap 账户")
            else:
                previous = records[-1]
                if decision.session_date <= previous["decision_session"]:
                    raise RuntimeError("组合决策日期未严格前进")
                if decision.bar_ts <= previous["decision_bar_ts"]:
                    raise RuntimeError("组合决策 bar_ts 未严格前进")
                if decision.session_date < previous["execution_session"]:
                    raise RuntimeError("组合决策日期早于前次执行日期")
                if bootstrap_account:
                    raise RuntimeError("已有持久账户时不得重新 bootstrap")
                if current_state is None:
                    raise RuntimeError("当前持久账户缺失")
                if (
                    _canonical_json(current_state["account"])
                    != _canonical_json(input_payload["account_before"])
                ):
                    raise RuntimeError("account_before 不等于当前持久账户")
                if (
                    recomputed.execution_session
                    <= current_state["execution_session"]
                ):
                    raise RuntimeError("执行日期未前进")

            now = time.time()
            input_sha256 = hashlib.sha256(
                canonical_input.encode("utf-8")
            ).hexdigest()
            result_sha256 = hashlib.sha256(
                canonical_result.encode("utf-8")
            ).hexdigest()
            account_before_sha256 = _payload_sha256(
                input_payload["account_before"]
            )
            account_after_sha256 = _payload_sha256(
                recomputed_payload["account_after"]
            )
            row_identity_payload = _execution_row_identity_payload(
                execution_id=recomputed.execution_id,
                decision_id=decision.decision_id,
                decision_session=decision.session_date,
                execution_session=recomputed.execution_session,
                bootstrap_account=bootstrap_account,
                created_at=now,
                input_payload=input_payload,
                input_sha256=input_sha256,
                result_payload=recomputed_payload,
                result_sha256=result_sha256,
                identity_sha256=identity_sha256,
                account_before_sha256=account_before_sha256,
                account_after_sha256=account_after_sha256,
                order_count=len(order_payloads),
                fill_count=len(fill_payloads),
            )
            row_sha256 = _payload_sha256(row_identity_payload)

            conn.execute(
                """
                INSERT INTO portfolio_executions (
                    execution_id, decision_id, decision_session,
                    execution_session, bootstrap_account, created_at,
                    input_payload_json, input_sha256,
                    result_payload_json, result_sha256, identity_sha256,
                    row_sha256, account_before_sha256, account_after_sha256,
                    order_count, fill_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recomputed.execution_id,
                    decision.decision_id,
                    decision.session_date,
                    recomputed.execution_session,
                    int(bootstrap_account),
                    now,
                    canonical_input,
                    input_sha256,
                    canonical_result,
                    result_sha256,
                    identity_sha256,
                    row_sha256,
                    account_before_sha256,
                    account_after_sha256,
                    len(order_payloads),
                    len(fill_payloads),
                ),
            )
            for order_index, order_payload in enumerate(
                order_payloads,
                start=1,
            ):
                raw_order = _canonical_json(order_payload)
                conn.execute(
                    """
                    INSERT INTO portfolio_execution_orders (
                        order_id, execution_id, order_index,
                        payload_json, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        order_payload["order_id"],
                        recomputed.execution_id,
                        order_index,
                        raw_order,
                        hashlib.sha256(raw_order.encode("utf-8")).hexdigest(),
                    ),
                )
            for fill_payload in fill_payloads:
                raw_fill = _canonical_json(fill_payload)
                conn.execute(
                    """
                    INSERT INTO portfolio_execution_fills (
                        fill_id, execution_id, order_id, fill_index,
                        payload_json, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill_payload["fill_id"],
                        recomputed.execution_id,
                        fill_payload["order_id"],
                        fill_payload["fill_index"],
                        raw_fill,
                        hashlib.sha256(raw_fill.encode("utf-8")).hexdigest(),
                    ),
                )
            account_after_json = _canonical_json(
                recomputed_payload["account_after"]
            )
            state_identity_payload = _account_state_identity_payload(
                execution_id=recomputed.execution_id,
                execution_session=recomputed.execution_session,
                account_payload=recomputed_payload["account_after"],
                account_sha256=account_after_sha256,
                updated_at=now,
                execution_row_sha256=row_sha256,
            )
            state_sha256 = _payload_sha256(state_identity_payload)
            conn.execute(
                """
                INSERT INTO portfolio_account_state (
                    singleton_id, execution_id, execution_session,
                    account_payload_json, account_sha256, updated_at,
                    state_sha256
                ) VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    execution_id = excluded.execution_id,
                    execution_session = excluded.execution_session,
                    account_payload_json = excluded.account_payload_json,
                    account_sha256 = excluded.account_sha256,
                    updated_at = excluded.updated_at,
                    state_sha256 = excluded.state_sha256
                """,
                (
                    recomputed.execution_id,
                    recomputed.execution_session,
                    account_after_json,
                    account_after_sha256,
                    now,
                    state_sha256,
                ),
            )
            verified_records = self._verified_execution_chain(conn)
            self._verified_account_state(conn, verified_records)
            stored = next(
                record
                for record in verified_records
                if record["execution_id"] == recomputed.execution_id
            )
            if (
                stored["identity_sha256"] != identity_sha256
                or stored["row_sha256"] != row_sha256
            ):
                raise RuntimeError("执行写入后 canonical identity 不一致")
            return stored, True

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            records = self._verified_execution_chain(conn)
            self._verified_account_state(conn, records)
        for record in records:
            if record["execution_id"] == str(execution_id):
                return record
        return None

    def get_account_state(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            records = self._verified_execution_chain(conn)
            return self._verified_account_state(conn, records)

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT decision_id, bar_ts, payload_json
                FROM portfolio_decisions
                WHERE decision_id = ?
                """,
                (str(decision_id),),
            ).fetchone()
        if row is None:
            return None
        return self._decode_payload(
            row["payload_json"],
            expected_decision_id=row["decision_id"],
            expected_bar_ts=row["bar_ts"],
        )

    def get_target_state(self) -> dict[str, Any] | None:
        """返回最新目标组合；它不是已成交账户状态。"""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT state.decision_id, state.bar_ts,
                       state.target_weights_json, state.cash_weight,
                       state.updated_at, decision.payload_json
                FROM target_portfolio_state AS state
                JOIN portfolio_decisions AS decision
                  ON decision.decision_id = state.decision_id
                WHERE state.singleton_id = 1
                """
            ).fetchone()
            latest = conn.execute(
                """
                SELECT decision_id, bar_ts
                FROM portfolio_decisions
                ORDER BY bar_ts DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            if latest is not None:
                raise RuntimeError("组合历史存在，但最新目标状态缺失")
            return None
        if (
            latest is None
            or latest["decision_id"] != row["decision_id"]
            or latest["bar_ts"] != row["bar_ts"]
        ):
            raise RuntimeError("最新目标状态没有指向最新组合决策")
        payload = self._decode_payload(
            row["payload_json"],
            expected_decision_id=row["decision_id"],
            expected_bar_ts=row["bar_ts"],
        )
        try:
            target_weights = json.loads(row["target_weights_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("最新目标状态权重不是合法 JSON") from exc
        allowed = frozenset(payload["universe"]["symbols"])
        validated_target_weights = _strict_weights(
            "最新目标状态权重",
            target_weights,
            allowed=allowed,
        )
        row_bar_ts = _strict_integer("最新目标状态 bar_ts", row["bar_ts"])
        row_cash_weight = _strict_unit_interval(
            "最新目标状态 cash_weight",
            row["cash_weight"],
        )
        updated_at = _strict_real(
            "最新目标状态 updated_at",
            row["updated_at"],
        )
        if (
            payload["decision_id"] != row["decision_id"]
            or payload["bar_ts"] != row_bar_ts
            or payload["target_weights"] != target_weights
            or payload["target_weights"] != validated_target_weights
            or payload["cash_weight"] != row_cash_weight
        ):
            raise RuntimeError("最新目标状态与组合决策历史不一致")
        return {
            "decision_id": row["decision_id"],
            "bar_ts": row_bar_ts,
            "target_weights": validated_target_weights,
            "cash_weight": row_cash_weight,
            "updated_at": updated_at,
        }

    def list_decisions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit 必须是整数")
        bounded = max(1, min(limit, 500))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT decision_id, bar_ts, payload_json
                FROM portfolio_decisions
                ORDER BY bar_ts DESC
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        return [
            self._decode_payload(
                row["payload_json"],
                expected_decision_id=row["decision_id"],
                expected_bar_ts=row["bar_ts"],
            )
            for row in rows
        ]

    # -- RND-04C replay 身份绑定（最小扩展，不触碰既有写入/读回路径）------

    def list_executions(self) -> list[dict[str, Any]]:
        """返回完整核验后的执行链记录（读取即核验，失败关闭）。"""
        with self._lock, self._connect() as conn:
            records = self._verified_execution_chain(conn)
            self._verified_account_state(conn, records)
        return records

    @staticmethod
    def _validate_replay_binding_payload(payload: object) -> dict[str, Any]:
        binding = _strict_object("replay 绑定", payload, _REPLAY_BINDING_KEYS)
        if binding["contract_version"] != REPLAY_BINDING_VERSION:
            raise RuntimeError("replay 绑定合同版本不一致")
        _strict_text(
            "replay 绑定 replay_contract_version",
            binding["replay_contract_version"],
        )
        _strict_text("replay 绑定 replay_run_id", binding["replay_run_id"])
        execution_id = _strict_text(
            "replay 绑定 execution_id",
            binding["execution_id"],
        )
        if _EXECUTION_ID_RE.fullmatch(execution_id) is None:
            raise RuntimeError("replay 绑定 execution_id 格式非法")
        decision_id = _strict_text(
            "replay 绑定 decision_id",
            binding["decision_id"],
        )
        if _DECISION_ID_RE.fullmatch(decision_id) is None:
            raise RuntimeError("replay 绑定 decision_id 格式非法")
        for field in ("decision_session", "execution_session"):
            value = _strict_text(f"replay 绑定 {field}", binding[field])
            if _SESSION_DATE_RE.fullmatch(value) is None:
                raise RuntimeError(f"replay 绑定 {field} 必须是 YYYY-MM-DD")
        _strict_sha256(
            "replay 绑定 overlay_identity_sha256",
            binding["overlay_identity_sha256"],
        )
        _strict_text(
            "replay 绑定 derivation_rule_version",
            binding["derivation_rule_version"],
        )
        _strict_sha256(
            "replay 绑定 execution_row_sha256",
            binding["execution_row_sha256"],
        )
        for field in (
            "universe",
            "freestockdb_price_identity",
            "decision_quote_provenance",
            "execution_quote_provenance",
            "engineering_signal",
            "fee_schedule",
        ):
            if not isinstance(binding[field], dict):
                raise RuntimeError(f"replay 绑定 {field} 必须是 JSON 对象")
        return binding

    def _cross_check_replay_binding(
        self,
        binding: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record = next(
            (
                item
                for item in records
                if item["execution_id"] == binding["execution_id"]
            ),
            None,
        )
        if record is None:
            raise RuntimeError(
                "replay 绑定引用的执行不存在于已核验执行链"
            )
        if (
            binding["decision_id"] != record["decision_id"]
            or binding["decision_session"] != record["decision_session"]
            or binding["execution_session"] != record["execution_session"]
            or binding["execution_row_sha256"] != record["row_sha256"]
        ):
            raise RuntimeError("replay 绑定与执行链身份不一致")
        return record

    def record_replay_binding(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """原子写入一条 replay 绑定；同一 canonical payload 重试幂等。"""
        binding = self._validate_replay_binding_payload(payload)
        canonical = _canonical_json(binding)
        payload_sha256 = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            records = self._verified_execution_chain(conn)
            self._verified_account_state(conn, records)
            self._cross_check_replay_binding(binding, records)
            existing = conn.execute(
                """
                SELECT execution_id, replay_run_id, payload_json,
                       payload_sha256, execution_row_sha256
                FROM portfolio_replay_bindings
                WHERE execution_id = ?
                """,
                (binding["execution_id"],),
            ).fetchone()
            if existing is not None:
                if (
                    existing["payload_json"] != canonical
                    or existing["payload_sha256"] != payload_sha256
                    or existing["replay_run_id"] != binding["replay_run_id"]
                    or existing["execution_row_sha256"]
                    != binding["execution_row_sha256"]
                ):
                    raise RuntimeError(
                        "同一执行已存在不同 canonical replay 绑定"
                    )
                return binding, False
            conn.execute(
                """
                INSERT INTO portfolio_replay_bindings (
                    execution_id, replay_run_id, created_at,
                    payload_json, payload_sha256, execution_row_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    binding["execution_id"],
                    binding["replay_run_id"],
                    time.time(),
                    canonical,
                    payload_sha256,
                    binding["execution_row_sha256"],
                ),
            )
            return binding, True

    def get_replay_binding(
        self,
        execution_id: str,
    ) -> dict[str, Any] | None:
        """读取并核验一条 replay 绑定；任何篡改失败关闭。"""
        with self._lock, self._connect() as conn:
            records = self._verified_execution_chain(conn)
            self._verified_account_state(conn, records)
            row = conn.execute(
                """
                SELECT execution_id, replay_run_id, created_at,
                       payload_json, payload_sha256, execution_row_sha256
                FROM portfolio_replay_bindings
                WHERE execution_id = ?
                """,
                (str(execution_id),),
            ).fetchone()
        if row is None:
            return None
        raw = row["payload_json"]
        if (
            not isinstance(raw, str)
            or hashlib.sha256(raw.encode("utf-8")).hexdigest()
            != _strict_sha256(
                "replay 绑定 payload_sha256",
                row["payload_sha256"],
            )
        ):
            raise RuntimeError("replay 绑定 payload 哈希不一致")
        binding = self._validate_replay_binding_payload(
            _decode_json_object("replay 绑定", raw)
        )
        if raw != _canonical_json(binding):
            raise RuntimeError("replay 绑定 payload 不是 canonical 形式")
        if (
            binding["execution_id"] != row["execution_id"]
            or binding["replay_run_id"] != row["replay_run_id"]
            or binding["execution_row_sha256"]
            != row["execution_row_sha256"]
        ):
            raise RuntimeError("replay 绑定行与 payload 身份不一致")
        _strict_real("replay 绑定 created_at", row["created_at"])
        self._cross_check_replay_binding(binding, records)
        return binding
