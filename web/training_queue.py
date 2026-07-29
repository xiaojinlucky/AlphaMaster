"""AlphaMaster 持久化批训练队列的 SQLite 存储核心。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


DEFAULT_DB_PATH = Path("local_runs/training_queue.sqlite3")
MAX_BATCH_ITEMS = 50

QUEUED = "QUEUED"
DISPATCHING = "DISPATCHING"
TRAINING = "TRAINING"
POSTPROCESSING = "POSTPROCESSING"
READY = "READY"
NEEDS_ATTENTION = "NEEDS_ATTENTION"

ACTIVE_ITEM_STATUSES = (DISPATCHING, TRAINING, POSTPROCESSING)
ITEM_STATUSES = (
    QUEUED,
    DISPATCHING,
    TRAINING,
    POSTPROCESSING,
    READY,
    NEEDS_ATTENTION,
)
BATCH_STATUSES = (QUEUED, "ACTIVE", READY, NEEDS_ATTENTION)
NEXT_ITEM_STATUS = {
    DISPATCHING: TRAINING,
    TRAINING: POSTPROCESSING,
    POSTPROCESSING: READY,
}

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:K|M|G|T)$")
_TIME_LIMIT_RE = re.compile(r"^(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$")
_RUN_ID_RE = re.compile(r"^run_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
_JOB_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")


class TrainingQueueError(RuntimeError):
    """批训练队列基础异常。"""


class QueueValidationError(TrainingQueueError, ValueError):
    """批次或项目输入不符合冻结要求。"""


class IdempotencyConflictError(TrainingQueueError):
    """相同幂等键对应了不同的冻结请求。"""


class QueueNotFoundError(TrainingQueueError, LookupError):
    """指定批次或训练项目不存在。"""


class StateTransitionError(TrainingQueueError):
    """训练项目发生非法状态跳转。"""


class DataHashDriftError(TrainingQueueError):
    """训练数据已不再匹配入队时冻结的 SHA-256。"""


class SourceHashDriftError(TrainingQueueError):
    """训练源码已不再匹配批次冻结的 SHA-256。"""


@dataclass(frozen=True)
class BatchItemSpec:
    """创建批次时冻结的单只标的训练输入。"""

    symbol: str
    timeframe: str
    data_file: str | Path
    data_sha256: str
    planned_run_id: str
    train_steps: int = 200
    cpus_per_task: int = 12
    memory: str = "32G"
    time_limit: str = "00:30:00"


@dataclass(frozen=True)
class BatchRecord:
    batch_id: str
    idempotency_key: str
    status: str
    contract_sha256: str
    source_sha256: str | None
    runtime_git_commit: str | None
    request_sha256: str
    item_count: int
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class BatchItemRecord:
    batch_id: str
    ordinal: int
    symbol: str
    timeframe: str
    data_file: str
    data_sha256: str
    planned_run_id: str
    execution_run_id: str
    attempt_number: int
    parent_run_id: str | None
    parent_job_id: str | None
    resume_checkpoint_path: str | None
    resume_checkpoint_sha256: str | None
    resume_checkpoint_size: int | None
    resume_checkpoint_step: int | None
    train_steps: int
    cpus_per_task: int
    memory: str
    time_limit: str
    status: str
    job_id: str | None
    error: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class CreateBatchResult:
    batch: BatchRecord
    created: bool


@dataclass(frozen=True)
class DataHashCheck:
    planned_run_id: str
    data_file: str
    expected_sha256: str
    actual_sha256: str | None
    matches: bool


@dataclass(frozen=True)
class _FrozenItem:
    ordinal: int
    symbol: str
    timeframe: str
    data_file: str
    data_sha256: str
    planned_run_id: str
    train_steps: int
    cpus_per_task: int
    memory: str
    time_limit: str


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256，避免一次读入整个训练文件。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrainingQueue:
    """只负责持久化、串行领取和状态推进，不执行任何外部提交。"""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL
                        CHECK (status IN {BATCH_STATUSES}),
                    contract_sha256 TEXT NOT NULL,
                    source_sha256 TEXT,
                    runtime_git_commit TEXT,
                    request_sha256 TEXT NOT NULL,
                    item_count INTEGER NOT NULL
                        CHECK (item_count BETWEEN 1 AND {MAX_BATCH_ITEMS}),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS batch_items (
                    batch_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL
                        CHECK (ordinal BETWEEN 0 AND {MAX_BATCH_ITEMS - 1}),
                    symbol TEXT NOT NULL COLLATE NOCASE,
                    timeframe TEXT NOT NULL,
                    data_file TEXT NOT NULL COLLATE NOCASE,
                    data_sha256 TEXT NOT NULL,
                    planned_run_id TEXT NOT NULL UNIQUE,
                    execution_run_id TEXT NOT NULL UNIQUE,
                    attempt_number INTEGER NOT NULL DEFAULT 0
                        CHECK (attempt_number IN (0, 1)),
                    parent_run_id TEXT,
                    parent_job_id TEXT,
                    resume_checkpoint_path TEXT,
                    resume_checkpoint_sha256 TEXT,
                    resume_checkpoint_size INTEGER,
                    resume_checkpoint_step INTEGER,
                    train_steps INTEGER NOT NULL
                        CHECK (train_steps BETWEEN 1 AND 1000000),
                    cpus_per_task INTEGER NOT NULL
                        CHECK (cpus_per_task BETWEEN 1 AND 64),
                    memory TEXT NOT NULL,
                    time_limit TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN {ITEM_STATUSES}),
                    job_id TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (batch_id, ordinal),
                    UNIQUE (batch_id, symbol),
                    UNIQUE (batch_id, data_file),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                        ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_batch_items_status
                ON batch_items(status);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_training_item
                ON batch_items ((1))
                WHERE status IN ('{DISPATCHING}', '{TRAINING}', '{POSTPROCESSING}');
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(batch_items)")
            }
            migrations = {
                "train_steps": "INTEGER NOT NULL DEFAULT 200",
                "cpus_per_task": "INTEGER NOT NULL DEFAULT 12",
                "memory": "TEXT NOT NULL DEFAULT '32G'",
                "time_limit": "TEXT NOT NULL DEFAULT '00:30:00'",
                "execution_run_id": "TEXT",
                "attempt_number": "INTEGER NOT NULL DEFAULT 0",
                "parent_run_id": "TEXT",
                "parent_job_id": "TEXT",
                "resume_checkpoint_path": "TEXT",
                "resume_checkpoint_sha256": "TEXT",
                "resume_checkpoint_size": "INTEGER",
                "resume_checkpoint_step": "INTEGER",
            }
            for column, declaration in migrations.items():
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE batch_items "
                        f"ADD COLUMN {column} {declaration}"
                    )
            conn.execute(
                """
                UPDATE batch_items
                SET execution_run_id = planned_run_id
                WHERE execution_run_id IS NULL OR execution_run_id = ''
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_batch_items_execution_run_id
                ON batch_items(execution_run_id)
                """
            )
            batch_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(batches)")
            }
            if "source_sha256" not in batch_columns:
                conn.execute(
                    "ALTER TABLE batches ADD COLUMN source_sha256 TEXT"
                )
            if "runtime_git_commit" not in batch_columns:
                conn.execute(
                    "ALTER TABLE batches ADD COLUMN runtime_git_commit TEXT"
                )

    @staticmethod
    def _normalize_text(value: object, field: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise QueueValidationError(f"{field} 不能为空")
        return normalized

    @staticmethod
    def _normalize_identifier(value: object, field: str) -> str:
        normalized = TrainingQueue._normalize_text(value, field)
        if not _IDENTIFIER_RE.fullmatch(normalized):
            raise QueueValidationError(f"{field} 格式非法")
        return normalized

    @staticmethod
    def _normalize_git_commit(value: object) -> str:
        normalized = str(value or "").strip().lower()
        if _GIT_COMMIT_RE.fullmatch(normalized) is None:
            raise QueueValidationError(
                "runtime_git_commit 必须是 40 位 Git 提交"
            )
        return normalized

    @staticmethod
    def _normalize_sha256(value: object, field: str) -> str:
        normalized = TrainingQueue._normalize_text(value, field).lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise QueueValidationError(f"{field} 必须是 64 位 SHA-256")
        return normalized

    @staticmethod
    def normalize_data_path(value: str | Path) -> str:
        """解析相对路径和 ``..``，再按操作系统规则统一大小写。"""
        raw = os.fspath(value)
        if not str(raw).strip():
            raise QueueValidationError("data_file 不能为空")
        resolved = Path(raw).expanduser().resolve(strict=False)
        return os.path.normcase(os.path.normpath(str(resolved)))

    @staticmethod
    def _normalize_positive_int(
        value: object,
        field: str,
        *,
        maximum: int,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise QueueValidationError(
                f"{field} 必须是 1 至 {maximum} 的整数"
            )
        return value

    @classmethod
    def _freeze_items(
        cls,
        items: Sequence[BatchItemSpec],
    ) -> tuple[_FrozenItem, ...]:
        if not items:
            raise QueueValidationError("批次至少需要一个训练项目")
        if len(items) > MAX_BATCH_ITEMS:
            raise QueueValidationError(
                f"单批最多 {MAX_BATCH_ITEMS} 个训练项目"
            )

        frozen: list[_FrozenItem] = []
        symbols: set[str] = set()
        paths: set[str] = set()
        run_ids: set[str] = set()
        for ordinal, spec in enumerate(items):
            if not isinstance(spec, BatchItemSpec):
                raise QueueValidationError("items 必须由 BatchItemSpec 组成")
            symbol = cls._normalize_text(spec.symbol, "symbol").upper()
            symbol_key = symbol.casefold()
            if symbol_key in symbols:
                raise QueueValidationError(f"同批 symbol 重复: {symbol}")
            symbols.add(symbol_key)

            timeframe = cls._normalize_text(spec.timeframe, "timeframe")
            data_file = cls.normalize_data_path(spec.data_file)
            path_key = os.path.normcase(data_file).casefold()
            if path_key in paths:
                raise QueueValidationError(f"同批 data_file 重复: {data_file}")
            paths.add(path_key)

            planned_run_id = cls._normalize_identifier(
                spec.planned_run_id,
                "planned_run_id",
            )
            if planned_run_id in run_ids:
                raise QueueValidationError(
                    f"同批 planned_run_id 重复: {planned_run_id}"
                )
            run_ids.add(planned_run_id)
            train_steps = cls._normalize_positive_int(
                spec.train_steps,
                "train_steps",
                maximum=1_000_000,
            )
            cpus_per_task = cls._normalize_positive_int(
                spec.cpus_per_task,
                "cpus_per_task",
                maximum=64,
            )
            memory = cls._normalize_text(spec.memory, "memory").upper()
            if _MEMORY_RE.fullmatch(memory) is None:
                raise QueueValidationError(
                    "memory 必须使用 K/M/G/T 单位，例如 32G"
                )
            time_limit = cls._normalize_text(
                spec.time_limit,
                "time_limit",
            )
            if _TIME_LIMIT_RE.fullmatch(time_limit) is None:
                raise QueueValidationError(
                    "time_limit 必须是 [D-]HH:MM:SS"
                )

            frozen.append(
                _FrozenItem(
                    ordinal=ordinal,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_file=data_file,
                    data_sha256=cls._normalize_sha256(
                        spec.data_sha256,
                        "data_sha256",
                    ),
                    planned_run_id=planned_run_id,
                    train_steps=train_steps,
                    cpus_per_task=cpus_per_task,
                    memory=memory,
                    time_limit=time_limit,
                )
            )
        return tuple(frozen)

    @staticmethod
    def _request_sha256(
        contract_sha256: str,
        source_sha256: str,
        runtime_git_commit: str | None,
        items: Sequence[_FrozenItem],
    ) -> str:
        payload = {
            "contract_sha256": contract_sha256,
            "source_sha256": source_sha256,
            "runtime_git_commit": runtime_git_commit,
            "items": [
                {
                    "ordinal": item.ordinal,
                    "symbol": item.symbol,
                    "timeframe": item.timeframe,
                    "data_file": item.data_file,
                    "data_sha256": item.data_sha256,
                    "planned_run_id": item.planned_run_id,
                    "train_steps": item.train_steps,
                    "cpus_per_task": item.cpus_per_task,
                    "memory": item.memory,
                    "time_limit": item.time_limit,
                }
                for item in items
            ],
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _batch_from_row(row: sqlite3.Row) -> BatchRecord:
        return BatchRecord(**dict(row))

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> BatchItemRecord:
        return BatchItemRecord(**dict(row))

    @staticmethod
    def _hash_check_from_item(item: BatchItemRecord) -> DataHashCheck:
        path = Path(item.data_file)
        actual = sha256_file(path) if path.is_file() else None
        return DataHashCheck(
            planned_run_id=item.planned_run_id,
            data_file=item.data_file,
            expected_sha256=item.data_sha256,
            actual_sha256=actual,
            matches=actual == item.data_sha256,
        )

    def _existing_by_idempotency_key(
        self,
        conn: sqlite3.Connection,
        idempotency_key: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM batches WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()

    def _return_existing_or_raise(
        self,
        row: sqlite3.Row,
        request_sha256: str,
    ) -> CreateBatchResult:
        if row["request_sha256"] != request_sha256:
            raise IdempotencyConflictError(
                "相同 idempotency_key 已绑定不同的冻结批次"
            )
        return CreateBatchResult(
            batch=self._batch_from_row(row),
            created=False,
        )

    def create_batch(
        self,
        *,
        idempotency_key: str,
        contract_sha256: str,
        source_sha256: str,
        runtime_git_commit: str | None = None,
        items: Sequence[BatchItemSpec],
        batch_id: str | None = None,
    ) -> CreateBatchResult:
        """幂等创建批次；所有 run ID 和输入身份在提交前一次性落库。"""
        key = self._normalize_text(idempotency_key, "idempotency_key")
        if len(key) > 256:
            raise QueueValidationError("idempotency_key 过长")
        contract_hash = self._normalize_sha256(
            contract_sha256,
            "contract_sha256",
        )
        source_hash = self._normalize_sha256(
            source_sha256,
            "source_sha256",
        )
        runtime_commit = (
            self._normalize_git_commit(runtime_git_commit)
            if runtime_git_commit is not None
            else None
        )
        frozen_items = self._freeze_items(items)
        request_hash = self._request_sha256(
            contract_hash,
            source_hash,
            runtime_commit,
            frozen_items,
        )

        with self._connect() as conn:
            existing = self._existing_by_idempotency_key(conn, key)
            if existing is not None:
                return self._return_existing_or_raise(existing, request_hash)

        for item in frozen_items:
            check = self._hash_check_from_item(
                BatchItemRecord(
                    batch_id="",
                    ordinal=item.ordinal,
                    symbol=item.symbol,
                    timeframe=item.timeframe,
                    data_file=item.data_file,
                    data_sha256=item.data_sha256,
                    planned_run_id=item.planned_run_id,
                    execution_run_id=item.planned_run_id,
                    attempt_number=0,
                    parent_run_id=None,
                    parent_job_id=None,
                    resume_checkpoint_path=None,
                    resume_checkpoint_sha256=None,
                    resume_checkpoint_size=None,
                    resume_checkpoint_step=None,
                    train_steps=item.train_steps,
                    cpus_per_task=item.cpus_per_task,
                    memory=item.memory,
                    time_limit=item.time_limit,
                    status=QUEUED,
                    job_id=None,
                    error=None,
                    created_at=0.0,
                    updated_at=0.0,
                )
            )
            if not check.matches:
                raise DataHashDriftError(
                    f"训练数据与冻结 SHA-256 不一致: {item.data_file}"
                )

        resolved_batch_id = (
            self._normalize_identifier(batch_id, "batch_id")
            if batch_id is not None
            else f"batch_{uuid.uuid4().hex}"
        )
        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = self._existing_by_idempotency_key(conn, key)
                if existing is not None:
                    return self._return_existing_or_raise(existing, request_hash)

                conn.execute(
                    """
                    INSERT INTO batches (
                        batch_id, idempotency_key, status, contract_sha256,
                        source_sha256, runtime_git_commit, request_sha256,
                        item_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_batch_id,
                        key,
                        QUEUED,
                        contract_hash,
                        source_hash,
                        runtime_commit,
                        request_hash,
                        len(frozen_items),
                        now,
                        now,
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO batch_items (
                        batch_id, ordinal, symbol, timeframe, data_file,
                        data_sha256, planned_run_id, execution_run_id,
                        attempt_number, parent_run_id, parent_job_id,
                        resume_checkpoint_path, resume_checkpoint_sha256,
                        resume_checkpoint_size, resume_checkpoint_step, train_steps,
                        cpus_per_task, memory, time_limit, status,
                        job_id, error, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, 0,
                        NULL, NULL, NULL, NULL, NULL, NULL,
                        ?, ?, ?, ?, ?,
                        NULL, NULL, ?, ?
                    )
                    """,
                    [
                        (
                            resolved_batch_id,
                            item.ordinal,
                            item.symbol,
                            item.timeframe,
                            item.data_file,
                            item.data_sha256,
                            item.planned_run_id,
                            item.planned_run_id,
                            item.train_steps,
                            item.cpus_per_task,
                            item.memory,
                            item.time_limit,
                            QUEUED,
                            now,
                            now,
                        )
                        for item in frozen_items
                    ],
                )
                row = conn.execute(
                    "SELECT * FROM batches WHERE batch_id = ?",
                    (resolved_batch_id,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise QueueValidationError(
                "batch_id、planned_run_id 或冻结输入已存在"
            ) from exc
        if row is None:  # pragma: no cover - SQLite 成功插入后不应发生
            raise TrainingQueueError("批次写入后无法读取")
        return CreateBatchResult(batch=self._batch_from_row(row), created=True)

    def get_batch(self, batch_id: str) -> BatchRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return self._batch_from_row(row) if row is not None else None

    def bind_batch_source(
        self,
        batch_id: str,
        source_sha256: str,
    ) -> BatchRecord:
        """只为旧批次一次性绑定首个真实 run 的训练源码身份。"""
        source_hash = self._normalize_sha256(
            source_sha256,
            "source_sha256",
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is None:
                raise QueueNotFoundError(f"批次不存在: {batch_id}")
            if row["source_sha256"] is not None:
                if row["source_sha256"] != source_hash:
                    raise SourceHashDriftError(
                        "批次训练源码 SHA-256 已冻结，不能改写"
                    )
                return self._batch_from_row(row)

            item_rows = conn.execute(
                """
                SELECT * FROM batch_items
                WHERE batch_id = ?
                ORDER BY ordinal ASC
                """,
                (batch_id,),
            ).fetchall()
            frozen_items = tuple(
                _FrozenItem(
                    ordinal=int(item["ordinal"]),
                    symbol=str(item["symbol"]),
                    timeframe=str(item["timeframe"]),
                    data_file=str(item["data_file"]),
                    data_sha256=str(item["data_sha256"]),
                    planned_run_id=str(item["planned_run_id"]),
                    train_steps=int(item["train_steps"]),
                    cpus_per_task=int(item["cpus_per_task"]),
                    memory=str(item["memory"]),
                    time_limit=str(item["time_limit"]),
                )
                for item in item_rows
            )
            request_hash = self._request_sha256(
                str(row["contract_sha256"]),
                source_hash,
                (
                    str(row["runtime_git_commit"])
                    if row["runtime_git_commit"] is not None
                    else None
                ),
                frozen_items,
            )
            now = time.time()
            conn.execute(
                """
                UPDATE batches
                SET source_sha256 = ?, request_sha256 = ?, updated_at = ?
                WHERE batch_id = ? AND source_sha256 IS NULL
                """,
                (source_hash, request_hash, now, batch_id),
            )
            updated = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover
            raise TrainingQueueError("批次源码身份绑定后无法读取")
        return self._batch_from_row(updated)

    def get_batch_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> BatchRecord | None:
        with self._connect() as conn:
            row = self._existing_by_idempotency_key(conn, idempotency_key)
        return self._batch_from_row(row) if row is not None else None

    def list_batches(
        self,
        *,
        statuses: Sequence[str] | None = None,
    ) -> list[BatchRecord]:
        """按创建顺序列出批次，可用于进程重启后的持久化恢复。"""
        normalized: tuple[str, ...] | None = None
        if statuses is not None:
            normalized = tuple(
                self._normalize_text(status, "status").upper()
                for status in statuses
            )
            if not normalized:
                return []
            invalid = sorted(set(normalized) - set(BATCH_STATUSES))
            if invalid:
                raise QueueValidationError(f"批次状态非法: {invalid}")
        with self._connect() as conn:
            if normalized is None:
                rows = conn.execute(
                    """
                    SELECT * FROM batches
                    ORDER BY created_at ASC, batch_id ASC
                    """
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in normalized)
                rows = conn.execute(
                    f"""
                    SELECT * FROM batches
                    WHERE status IN ({placeholders})
                    ORDER BY created_at ASC, batch_id ASC
                    """,
                    normalized,
                ).fetchall()
        return [self._batch_from_row(row) for row in rows]

    def has_pending_work(self) -> bool:
        """是否还有排队、运行或等待人工处理的批次。"""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM batches
                WHERE status IN (?, ?, ?)
                LIMIT 1
                """,
                (QUEUED, "ACTIVE", NEEDS_ATTENTION),
            ).fetchone()
        return row is not None

    def list_items(self, batch_id: str) -> list[BatchItemRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM batch_items
                WHERE batch_id = ?
                ORDER BY ordinal ASC
                """,
                (batch_id,),
            ).fetchall()
        return [self._item_from_row(row) for row in rows]

    def get_item(self, planned_run_id: str) -> BatchItemRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def get_item_by_execution_run_id(
        self,
        execution_run_id: str,
    ) -> BatchItemRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_items WHERE execution_run_id = ?",
                (execution_run_id,),
            ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def get_active_item(self) -> BatchItemRecord | None:
        placeholders = ", ".join("?" for _ in ACTIVE_ITEM_STATUSES)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM batch_items
                WHERE status IN ({placeholders})
                ORDER BY updated_at ASC
                LIMIT 1
                """,
                ACTIVE_ITEM_STATUSES,
            ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def check_data_hash(self, planned_run_id: str) -> DataHashCheck:
        item = self.get_item(planned_run_id)
        if item is None:
            raise QueueNotFoundError(f"训练项目不存在: {planned_run_id}")
        return self._hash_check_from_item(item)

    @staticmethod
    def _first_unfinished_item(
        conn: sqlite3.Connection,
        batch_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM batch_items
            WHERE batch_id = ? AND status != ?
            ORDER BY ordinal ASC
            LIMIT 1
            """,
            (batch_id, READY),
        ).fetchone()

    def can_claim_next(self, batch_id: str) -> bool:
        """只做瞬时判断；真正的互斥保证由 :meth:`claim_next` 完成。"""
        with self._connect() as conn:
            batch = conn.execute(
                "SELECT status FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if batch is None:
                raise QueueNotFoundError(f"批次不存在: {batch_id}")
            if batch["status"] not in (QUEUED, "ACTIVE"):
                return False
            placeholders = ", ".join("?" for _ in ACTIVE_ITEM_STATUSES)
            active = conn.execute(
                f"""
                SELECT 1 FROM batch_items
                WHERE status IN ({placeholders})
                LIMIT 1
                """,
                ACTIVE_ITEM_STATUSES,
            ).fetchone()
            if active is not None:
                return False
            row = self._first_unfinished_item(conn, batch_id)
        if row is None or row["status"] != QUEUED:
            return False
        return self._hash_check_from_item(self._item_from_row(row)).matches

    def claim_next(
        self,
        batch_id: str,
        *,
        source_sha256: str | None = None,
    ) -> BatchItemRecord | None:
        """原子领取同批最早未完成项目，并在返回前持久化 DISPATCHING。"""
        drift_message: str | None = None
        source_drift_message: str | None = None
        claimed: BatchItemRecord | None = None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            batch = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if batch is None:
                raise QueueNotFoundError(f"批次不存在: {batch_id}")
            if batch["status"] not in (QUEUED, "ACTIVE"):
                return None
            frozen_source = str(batch["source_sha256"] or "")
            if not frozen_source:
                raise SourceHashDriftError("批次尚未冻结训练源码 SHA-256")
            observed_source = (
                self._normalize_sha256(source_sha256, "source_sha256")
                if source_sha256 is not None
                else frozen_source
            )

            placeholders = ", ".join("?" for _ in ACTIVE_ITEM_STATUSES)
            active = conn.execute(
                f"""
                SELECT 1 FROM batch_items
                WHERE status IN ({placeholders})
                LIMIT 1
                """,
                ACTIVE_ITEM_STATUSES,
            ).fetchone()
            if active is not None:
                return None

            row = self._first_unfinished_item(conn, batch_id)
            if row is None or row["status"] != QUEUED:
                return None

            item = self._item_from_row(row)
            now = time.time()
            if observed_source != frozen_source:
                source_drift_message = (
                    f"训练源码 SHA-256 漂移: {item.planned_run_id}"
                )
                conn.execute(
                    """
                    UPDATE batch_items
                    SET status = ?, error = ?, updated_at = ?
                    WHERE planned_run_id = ?
                    """,
                    (
                        NEEDS_ATTENTION,
                        source_drift_message,
                        now,
                        item.planned_run_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE batches
                    SET status = ?, updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (NEEDS_ATTENTION, now, batch_id),
                )
            else:
                check = self._hash_check_from_item(item)
            if source_drift_message is not None:
                pass
            elif not check.matches:
                drift_message = (
                    f"训练数据 SHA-256 漂移: {item.planned_run_id}"
                )
                conn.execute(
                    """
                    UPDATE batch_items
                    SET status = ?, error = ?, updated_at = ?
                    WHERE planned_run_id = ?
                    """,
                    (
                        NEEDS_ATTENTION,
                        drift_message,
                        now,
                        item.planned_run_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE batches
                    SET status = ?, updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (NEEDS_ATTENTION, now, batch_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE batch_items
                    SET status = ?, updated_at = ?
                    WHERE planned_run_id = ? AND status = ?
                    """,
                    (
                        DISPATCHING,
                        now,
                        item.planned_run_id,
                        QUEUED,
                    ),
                )
                conn.execute(
                    """
                    UPDATE batches
                    SET status = 'ACTIVE', updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (now, batch_id),
                )
                claimed_row = conn.execute(
                    """
                    SELECT * FROM batch_items
                    WHERE planned_run_id = ?
                    """,
                    (item.planned_run_id,),
                ).fetchone()
                if claimed_row is not None:
                    claimed = self._item_from_row(claimed_row)
        if drift_message is not None:
            raise DataHashDriftError(drift_message)
        if source_drift_message is not None:
            raise SourceHashDriftError(source_drift_message)
        return claimed

    def claim_next_available(
        self,
        *,
        source_sha256: str | None = None,
    ) -> BatchItemRecord | None:
        """从最早未完成批次领取下一项；真正互斥仍由数据库唯一索引保证。"""
        for batch in self.list_batches(statuses=(QUEUED, "ACTIVE")):
            claimed = self.claim_next(
                batch.batch_id,
                source_sha256=source_sha256,
            )
            if claimed is not None:
                return claimed
        return None

    def advance_item(
        self,
        planned_run_id: str,
        next_status: str,
        *,
        job_id: str | int | None = None,
    ) -> BatchItemRecord:
        """把当前活动项目严格推进一格，不允许跨状态跳转。"""
        target = str(next_status).strip().upper()
        if target not in (TRAINING, POSTPROCESSING, READY):
            raise StateTransitionError(f"不允许推进到状态: {next_status}")
        normalized_job_id = str(job_id).strip() if job_id is not None else None
        if normalized_job_id == "":
            raise QueueValidationError("job_id 不能为空字符串")
        if target == TRAINING and normalized_job_id is None:
            raise QueueValidationError(
                "进入 TRAINING 前必须冻结 Slurm job_id"
            )

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
            if row is None:
                raise QueueNotFoundError(f"训练项目不存在: {planned_run_id}")
            current = self._item_from_row(row)
            expected = NEXT_ITEM_STATUS.get(current.status)
            if expected != target:
                raise StateTransitionError(
                    f"非法状态跳转: {current.status} -> {target}"
                )
            if (
                current.job_id is not None
                and normalized_job_id is not None
                and current.job_id != normalized_job_id
            ):
                raise QueueValidationError("job_id 已冻结，不能改写")
            stored_job_id = current.job_id or normalized_job_id
            if stored_job_id is None:
                raise QueueValidationError(
                    f"进入 {target} 前必须已有 Slurm job_id"
                )
            now = time.time()
            conn.execute(
                """
                UPDATE batch_items
                SET status = ?, job_id = ?, error = NULL, updated_at = ?
                WHERE planned_run_id = ?
                """,
                (target, stored_job_id, now, planned_run_id),
            )

            if target == READY:
                unfinished = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM batch_items
                    WHERE batch_id = ? AND status != ?
                    """,
                    (current.batch_id, READY),
                ).fetchone()["count"]
                batch_status = READY if unfinished == 0 else "ACTIVE"
                conn.execute(
                    """
                    UPDATE batches
                    SET status = ?, updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (batch_status, now, current.batch_id),
                )

            updated = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover - 更新成功后不应发生
            raise TrainingQueueError("状态推进后无法读取训练项目")
        return self._item_from_row(updated)

    def fail_item(
        self,
        planned_run_id: str,
        error: str,
    ) -> BatchItemRecord:
        """暂停当前活动项目和所属批次，不自动领取后续项目。"""
        detail = self._normalize_text(error, "error")
        if len(detail) > 4000:
            detail = detail[:4000]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
            if row is None:
                raise QueueNotFoundError(f"训练项目不存在: {planned_run_id}")
            current = self._item_from_row(row)
            if current.status not in ACTIVE_ITEM_STATUSES:
                raise StateTransitionError(
                    f"只有活动项目可以失败: {current.status}"
                )
            now = time.time()
            conn.execute(
                """
                UPDATE batch_items
                SET status = ?, error = ?, updated_at = ?
                WHERE planned_run_id = ?
                """,
                (NEEDS_ATTENTION, detail, now, planned_run_id),
            )
            conn.execute(
                """
                UPDATE batches
                SET status = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (NEEDS_ATTENTION, now, current.batch_id),
            )
            updated = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover - 更新成功后不应发生
            raise TrainingQueueError("失败状态写入后无法读取训练项目")
        return self._item_from_row(updated)

    def retry_pre_submission(self, planned_run_id: str) -> BatchItemRecord:
        """只恢复尚未取得 Slurm job ID 的失败项目，避免重复提交远端作业。"""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
            if row is None:
                raise QueueNotFoundError(f"训练项目不存在: {planned_run_id}")
            current = self._item_from_row(row)
            if current.status != NEEDS_ATTENTION:
                raise StateTransitionError(
                    f"只有 NEEDS_ATTENTION 项目可以恢复: {current.status}"
                )
            if current.job_id is not None:
                raise StateTransitionError(
                    "已取得 Slurm job ID 的失败项目禁止复用原 run_id"
                )
            placeholders = ", ".join("?" for _ in ACTIVE_ITEM_STATUSES)
            active = conn.execute(
                f"""
                SELECT 1 FROM batch_items
                WHERE status IN ({placeholders})
                LIMIT 1
                """,
                ACTIVE_ITEM_STATUSES,
            ).fetchone()
            if active is not None:
                raise StateTransitionError("已有活动项目，不能恢复失败项")

            now = time.time()
            conn.execute(
                """
                UPDATE batch_items
                SET status = ?, error = NULL, updated_at = ?
                WHERE planned_run_id = ?
                """,
                (DISPATCHING, now, planned_run_id),
            )
            conn.execute(
                """
                UPDATE batches
                SET status = 'ACTIVE', updated_at = ?
                WHERE batch_id = ?
                """,
                (now, current.batch_id),
            )
            updated = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover - 更新成功后不应发生
            raise TrainingQueueError("恢复状态写入后无法读取训练项目")
        return self._item_from_row(updated)

    def recover_submitted_item(
        self,
        planned_run_id: str,
        job_id: str | int,
    ) -> BatchItemRecord:
        """用已存在的同一 Slurm 作业恢复误报失败；绝不重新提交。"""
        normalized_job_id = self._normalize_text(job_id, "job_id")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
            if row is None:
                raise QueueNotFoundError(f"训练项目不存在: {planned_run_id}")
            current = self._item_from_row(row)
            if current.status != NEEDS_ATTENTION:
                raise StateTransitionError(
                    f"只有 NEEDS_ATTENTION 项目可以恢复: {current.status}"
                )
            if current.job_id is None:
                raise QueueValidationError(
                    "未冻结 job_id 的失败项目禁止自动恢复"
                )
            if current.job_id != normalized_job_id:
                raise QueueValidationError("恢复作业的 job_id 与冻结记录不一致")
            placeholders = ", ".join("?" for _ in ACTIVE_ITEM_STATUSES)
            active = conn.execute(
                f"""
                SELECT 1 FROM batch_items
                WHERE status IN ({placeholders})
                LIMIT 1
                """,
                ACTIVE_ITEM_STATUSES,
            ).fetchone()
            if active is not None:
                raise StateTransitionError("已有活动项目，不能恢复误报失败项")

            now = time.time()
            conn.execute(
                """
                UPDATE batch_items
                SET status = ?, job_id = ?, error = NULL, updated_at = ?
                WHERE planned_run_id = ?
                """,
                (TRAINING, normalized_job_id, now, planned_run_id),
            )
            conn.execute(
                """
                UPDATE batches
                SET status = 'ACTIVE', updated_at = ?
                WHERE batch_id = ?
                """,
                (now, current.batch_id),
            )
            updated = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover - 更新成功后不应发生
            raise TrainingQueueError("恢复活动作业后无法读取训练项目")
        return self._item_from_row(updated)

    def begin_checkpoint_recovery(
        self,
        planned_run_id: str,
        *,
        recovery_run_id: str,
        parent_job_id: str | int,
        checkpoint_path: str,
        checkpoint_sha256: str,
        checkpoint_size: int,
        checkpoint_step: int,
    ) -> BatchItemRecord:
        """为一次 NODE_FAIL 创建新的物理 run；旧 run/job 作为父谱系保留。"""
        normalized_recovery_run = self._normalize_text(
            recovery_run_id,
            "recovery_run_id",
        )
        if _RUN_ID_RE.fullmatch(normalized_recovery_run) is None:
            raise QueueValidationError("recovery_run_id 格式非法")
        normalized_parent_job = self._normalize_text(
            parent_job_id,
            "parent_job_id",
        )
        if _JOB_ID_RE.fullmatch(normalized_parent_job) is None:
            raise QueueValidationError("parent_job_id 格式非法")
        normalized_checkpoint_sha = self._normalize_sha256(
            checkpoint_sha256,
            "checkpoint_sha256",
        )
        if (
            isinstance(checkpoint_size, bool)
            or not isinstance(checkpoint_size, int)
            or not 0 < checkpoint_size <= 8 * 1024**3
        ):
            raise QueueValidationError("checkpoint_size 超出允许范围")
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step <= 0
        ):
            raise QueueValidationError("checkpoint_step 必须是正整数")
        normalized_checkpoint_path = self._normalize_text(
            checkpoint_path,
            "checkpoint_path",
        )
        posix = PurePosixPath(normalized_checkpoint_path)
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or not posix.parts
            or posix.as_posix() != normalized_checkpoint_path
        ):
            raise QueueValidationError("checkpoint_path 必须是规范的相对 POSIX 路径")

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
            if row is None:
                raise QueueNotFoundError(f"训练项目不存在: {planned_run_id}")
            current = self._item_from_row(row)
            if current.status != NEEDS_ATTENTION:
                raise StateTransitionError(
                    f"只有 NEEDS_ATTENTION 项目可以续训: {current.status}"
                )
            if current.attempt_number != 0:
                raise StateTransitionError("一次性 checkpoint 恢复已经使用")
            if re.fullmatch(
                r"Slurm NODE_FAIL(?::.*)?",
                str(current.error or "").strip(),
                flags=re.IGNORECASE,
            ) is None:
                raise StateTransitionError("只有明确的 Slurm NODE_FAIL 可以续训")
            if current.job_id != normalized_parent_job:
                raise QueueValidationError("父 job ID 与当前冻结记录不一致")
            if current.execution_run_id != current.planned_run_id:
                raise StateTransitionError("初始物理 run 身份已发生变化")
            if normalized_recovery_run == current.planned_run_id:
                raise QueueValidationError("恢复 run 必须使用新的物理 run ID")
            if checkpoint_step >= current.train_steps:
                raise QueueValidationError("checkpoint_step 必须小于目标总步数")
            expected_checkpoint = re.fullmatch(
                rf"checkpoints/{re.escape(current.timeframe)}/"
                rf"{re.escape(current.data_sha256)}/run_[0-9]{{20}}/"
                rf"ckpt_{re.escape(current.symbol)}_step_([0-9]{{4,}})\.pt",
                normalized_checkpoint_path,
            )
            if (
                expected_checkpoint is None
                or int(expected_checkpoint.group(1)) != checkpoint_step
            ):
                raise QueueValidationError("checkpoint_path 与队列训练身份不一致")
            collision = conn.execute(
                """
                SELECT 1 FROM batch_items
                WHERE planned_run_id = ? OR execution_run_id = ?
                LIMIT 1
                """,
                (normalized_recovery_run, normalized_recovery_run),
            ).fetchone()
            if collision is not None:
                raise QueueValidationError("恢复物理 run ID 已存在")
            placeholders = ", ".join("?" for _ in ACTIVE_ITEM_STATUSES)
            active = conn.execute(
                f"""
                SELECT 1 FROM batch_items
                WHERE status IN ({placeholders})
                LIMIT 1
                """,
                ACTIVE_ITEM_STATUSES,
            ).fetchone()
            if active is not None:
                raise StateTransitionError("已有活动项目，不能启动 checkpoint 恢复")

            now = time.time()
            conn.execute(
                """
                UPDATE batch_items
                SET execution_run_id = ?, attempt_number = 1,
                    parent_run_id = planned_run_id, parent_job_id = ?,
                    resume_checkpoint_path = ?,
                    resume_checkpoint_sha256 = ?,
                    resume_checkpoint_size = ?, resume_checkpoint_step = ?,
                    status = ?, job_id = NULL, error = NULL, updated_at = ?
                WHERE planned_run_id = ?
                """,
                (
                    normalized_recovery_run,
                    normalized_parent_job,
                    normalized_checkpoint_path,
                    normalized_checkpoint_sha,
                    checkpoint_size,
                    checkpoint_step,
                    DISPATCHING,
                    now,
                    planned_run_id,
                ),
            )
            conn.execute(
                """
                UPDATE batches
                SET status = 'ACTIVE', updated_at = ?
                WHERE batch_id = ?
                """,
                (now, current.batch_id),
            )
            updated = conn.execute(
                "SELECT * FROM batch_items WHERE planned_run_id = ?",
                (planned_run_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover
            raise TrainingQueueError("checkpoint 恢复状态写入后无法读取")
        return self._item_from_row(updated)
