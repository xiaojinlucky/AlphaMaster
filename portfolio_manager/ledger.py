"""组合目标决策的 SQLite 审计账本。"""

from __future__ import annotations

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
    PortfolioDecision,
    _canonical_decision_id,
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
    }
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


def _validate_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _strict_object("组合决策 payload", payload, _TOP_LEVEL_KEYS)
    decision_id = _strict_text("decision_id", payload["decision_id"])
    if _DECISION_ID_RE.fullmatch(decision_id) is None:
        raise RuntimeError("decision_id 格式非法")

    universe = _strict_object("universe", payload["universe"], _UNIVERSE_KEYS)
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


class PortfolioDecisionLedger:
    """原子记录组合决策，并保存最新目标组合供重启恢复。"""

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
                """
            )

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
