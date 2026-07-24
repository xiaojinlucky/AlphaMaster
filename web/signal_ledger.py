"""实时信号事件与虚拟长仓的 SQLite 账本。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from strategy_manager.signal_lifecycle import (
    HOLD,
    PUSH_ACTIONS,
    LongOnlyState,
    decide_long_only,
)


@dataclass(frozen=True)
class SignalEvent:
    event_id: str
    watch_id: str
    source: str
    symbol: str
    timeframe: str
    strategy_name: str
    strategy_fingerprint: str
    bar_ts: int
    created_at: float
    action: str
    reason: str
    previous_exposure: float
    requested_exposure: float
    resulting_exposure: float
    price: float
    entry_price: float | None
    stop_price: float | None
    take_profit_price: float | None
    factor_value: float
    strength: float
    raw_position: float
    delivery_status: str
    delivery_detail: str
    delivery_attempts: int

    @property
    def should_push(self) -> bool:
        return self.action in PUSH_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalLedger:
    """每个方法使用独立连接，并用进程内锁串行化写事务。"""

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
                CREATE TABLE IF NOT EXISTS virtual_positions (
                    watch_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_fingerprint TEXT NOT NULL,
                    exposure REAL NOT NULL,
                    entry_price REAL,
                    take_profit_done INTEGER NOT NULL,
                    last_bar_ts INTEGER NOT NULL,
                    last_price REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signal_events (
                    event_id TEXT PRIMARY KEY,
                    watch_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strategy_fingerprint TEXT NOT NULL,
                    bar_ts INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    previous_exposure REAL NOT NULL,
                    requested_exposure REAL NOT NULL,
                    resulting_exposure REAL NOT NULL,
                    price REAL NOT NULL,
                    entry_price REAL,
                    stop_price REAL,
                    take_profit_price REAL,
                    factor_value REAL NOT NULL,
                    strength REAL NOT NULL,
                    raw_position REAL NOT NULL,
                    delivery_status TEXT NOT NULL,
                    delivery_detail TEXT NOT NULL,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (watch_id, bar_ts)
                );

                CREATE INDEX IF NOT EXISTS idx_signal_events_created
                ON signal_events(created_at DESC);
                """
            )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> SignalEvent:
        return SignalEvent(**dict(row))

    @staticmethod
    def _event_id(payload: dict[str, Any]) -> str:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "AM-" + hashlib.sha256(raw).hexdigest()[:24].upper()

    @staticmethod
    def _state_from_row(row: sqlite3.Row | None) -> LongOnlyState:
        if row is None:
            return LongOnlyState()
        return LongOnlyState(
            exposure=float(row["exposure"]),
            entry_price=(
                float(row["entry_price"]) if row["entry_price"] is not None else None
            ),
            take_profit_done=bool(row["take_profit_done"]),
            last_bar_ts=int(row["last_bar_ts"]),
            last_price=float(row["last_price"]),
        )

    def process_bar(
        self,
        *,
        watch_id: str,
        source: str,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        strategy_fingerprint: str,
        bar_ts: int,
        price: float,
        raw_position: float,
        factor_value: float,
        strength: float,
        rebalance_delta: float,
        minimum_exposure: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        take_profit_remaining_ratio: float,
    ) -> tuple[SignalEvent, bool]:
        """原子地产生一根 K 线的决策；重复 K 线返回原事件。"""
        bar_time = int(bar_ts)
        created_at = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM signal_events WHERE watch_id = ? AND bar_ts = ?",
                (watch_id, bar_time),
            ).fetchone()
            if existing is not None:
                return self._event_from_row(existing), False

            position_row = conn.execute(
                "SELECT * FROM virtual_positions WHERE watch_id = ?",
                (watch_id,),
            ).fetchone()
            state = self._state_from_row(position_row)
            if state.last_bar_ts is not None and bar_time <= state.last_bar_ts:
                raise ValueError(
                    f"K 线时间未前进: {bar_time} <= {state.last_bar_ts}"
                )

            decision = decide_long_only(
                state,
                raw_position=raw_position,
                price=price,
                minimum_exposure=minimum_exposure,
                rebalance_delta=rebalance_delta,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                take_profit_remaining_ratio=take_profit_remaining_ratio,
            )
            identity = {
                "watch_id": watch_id,
                "strategy_fingerprint": strategy_fingerprint,
                "bar_ts": bar_time,
                "action": decision.action,
                "requested_exposure": round(decision.requested_exposure, 6),
                "resulting_exposure": round(decision.resulting_exposure, 6),
            }
            event_id = self._event_id(identity)
            delivery_status = "PENDING" if decision.action in PUSH_ACTIONS else "NOT_REQUIRED"

            conn.execute(
                """
                INSERT INTO signal_events (
                    event_id, watch_id, source, symbol, timeframe, strategy_name,
                    strategy_fingerprint, bar_ts, created_at, action, reason,
                    previous_exposure, requested_exposure, resulting_exposure,
                    price, entry_price, stop_price, take_profit_price,
                    factor_value, strength, raw_position,
                    delivery_status, delivery_detail, delivery_attempts
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, '', 0
                )
                """,
                (
                    event_id,
                    watch_id,
                    source,
                    symbol,
                    timeframe,
                    strategy_name,
                    strategy_fingerprint,
                    bar_time,
                    created_at,
                    decision.action,
                    decision.reason,
                    decision.previous_exposure,
                    decision.requested_exposure,
                    decision.resulting_exposure,
                    float(price),
                    decision.entry_price,
                    decision.stop_price,
                    decision.take_profit_price,
                    float(factor_value),
                    float(strength),
                    float(raw_position),
                    delivery_status,
                ),
            )
            conn.execute(
                """
                INSERT INTO virtual_positions (
                    watch_id, source, symbol, timeframe, strategy_fingerprint,
                    exposure, entry_price, take_profit_done, last_bar_ts,
                    last_price, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(watch_id) DO UPDATE SET
                    source = excluded.source,
                    symbol = excluded.symbol,
                    timeframe = excluded.timeframe,
                    strategy_fingerprint = excluded.strategy_fingerprint,
                    exposure = excluded.exposure,
                    entry_price = excluded.entry_price,
                    take_profit_done = excluded.take_profit_done,
                    last_bar_ts = excluded.last_bar_ts,
                    last_price = excluded.last_price,
                    updated_at = excluded.updated_at
                """,
                (
                    watch_id,
                    source,
                    symbol,
                    timeframe,
                    strategy_fingerprint,
                    decision.resulting_exposure,
                    decision.entry_price,
                    int(decision.take_profit_done),
                    bar_time,
                    float(price),
                    created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM signal_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("信号事件写入后无法复读")
            return self._event_from_row(row), True

    def record_delivery(self, event_id: str, status: str, detail: str = "") -> None:
        allowed = {"DELIVERED", "FAILED", "SKIPPED"}
        if status not in allowed:
            raise ValueError(f"非法投递状态: {status}")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE signal_events
                SET delivery_status = ?,
                    delivery_detail = ?,
                    delivery_attempts = delivery_attempts + 1
                WHERE event_id = ?
                """,
                (status, str(detail), event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"未知信号事件: {event_id}")

    def list_events(
        self,
        *,
        limit: int = 100,
        watch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 500))
        query = "SELECT * FROM signal_events"
        params: list[Any] = []
        if watch_id:
            query += " WHERE watch_id = ?"
            params.append(watch_id)
        query += " ORDER BY bar_ts DESC, created_at DESC LIMIT ?"
        params.append(bounded)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_from_row(row).to_dict() for row in rows]

    def get_position(self, watch_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM virtual_positions WHERE watch_id = ?",
                (watch_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM signal_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._event_from_row(row).to_dict() if row is not None else None


__all__ = ["HOLD", "SignalEvent", "SignalLedger"]
