from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from web.training_queue import (
    DISPATCHING,
    NEEDS_ATTENTION,
    POSTPROCESSING,
    QUEUED,
    READY,
    TRAINING,
    BatchItemSpec,
    DataHashDriftError,
    IdempotencyConflictError,
    QueueValidationError,
    SourceHashDriftError,
    StateTransitionError,
    TrainingQueue,
)


CONTRACT_SHA256 = "c" * 64
SOURCE_SHA256 = "f" * 64
RUNTIME_GIT_COMMIT = "a" * 40


def _write_data(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _spec(
    path: Path,
    *,
    symbol: str = "600519",
    planned_run_id: str = "run_600519",
) -> BatchItemSpec:
    return BatchItemSpec(
        symbol=symbol,
        timeframe="D1",
        data_file=path,
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        planned_run_id=planned_run_id,
    )


def _create_two_item_batch(
    queue: TrainingQueue,
    tmp_path: Path,
    *,
    key: str = "leaders-v1",
) -> str:
    first = tmp_path / "600519_D1.parquet"
    second = tmp_path / "000858_D1.parquet"
    _write_data(first, b"maotai")
    _write_data(second, b"wuliangye")
    result = queue.create_batch(
        idempotency_key=key,
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        items=[
            _spec(first),
            _spec(
                second,
                symbol="000858",
                planned_run_id="run_000858",
            ),
        ],
    )
    return result.batch.batch_id


def test_create_batch_is_idempotent_and_freezes_identity(tmp_path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    data_sha256 = _write_data(data_file, b"frozen-data")
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    spec = BatchItemSpec(
        symbol="600519",
        timeframe="D1",
        data_file=data_file,
        data_sha256=data_sha256,
        planned_run_id="run_600519",
    )

    first = queue.create_batch(
        idempotency_key="leaders-v1",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        runtime_git_commit=RUNTIME_GIT_COMMIT,
        items=[spec],
    )
    repeated = queue.create_batch(
        idempotency_key="leaders-v1",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        runtime_git_commit=RUNTIME_GIT_COMMIT,
        items=[spec],
    )

    assert first.created is True
    assert repeated.created is False
    assert repeated.batch.batch_id == first.batch.batch_id
    assert repeated.batch.source_sha256 == SOURCE_SHA256
    assert repeated.batch.runtime_git_commit == RUNTIME_GIT_COMMIT
    item = queue.list_items(first.batch.batch_id)[0]
    assert item.status == QUEUED
    assert item.data_sha256 == data_sha256
    assert item.planned_run_id == "run_600519"
    assert item.data_file == TrainingQueue.normalize_data_path(data_file)
    assert item.train_steps == 200
    assert item.cpus_per_task == 12
    assert item.memory == "32G"
    assert item.time_limit == "00:30:00"

    with pytest.raises(IdempotencyConflictError):
        queue.create_batch(
            idempotency_key="leaders-v1",
            contract_sha256="d" * 64,
            source_sha256=SOURCE_SHA256,
            runtime_git_commit=RUNTIME_GIT_COMMIT,
            items=[spec],
        )

    changed_budget = BatchItemSpec(
        symbol=spec.symbol,
        timeframe=spec.timeframe,
        data_file=spec.data_file,
        data_sha256=spec.data_sha256,
        planned_run_id=spec.planned_run_id,
        train_steps=201,
    )
    with pytest.raises(IdempotencyConflictError):
        queue.create_batch(
            idempotency_key="leaders-v1",
            contract_sha256=CONTRACT_SHA256,
            source_sha256=SOURCE_SHA256,
            runtime_git_commit=RUNTIME_GIT_COMMIT,
            items=[changed_budget],
        )
    with pytest.raises(IdempotencyConflictError):
        queue.create_batch(
            idempotency_key="leaders-v1",
            contract_sha256=CONTRACT_SHA256,
            source_sha256="e" * 64,
            runtime_git_commit=RUNTIME_GIT_COMMIT,
            items=[spec],
        )
    with pytest.raises(IdempotencyConflictError):
        queue.create_batch(
            idempotency_key="leaders-v1",
            contract_sha256=CONTRACT_SHA256,
            source_sha256=SOURCE_SHA256,
            runtime_git_commit="b" * 40,
            items=[spec],
        )


def test_legacy_queue_migration_backfills_execution_identity_without_state_change(
    tmp_path,
) -> None:
    database = tmp_path / "legacy.sqlite3"
    planned_run_id = "run_20260723T235959Z_867bfc69"
    with sqlite3.connect(database) as conn:
        conn.executescript(
            """
            CREATE TABLE batches (
                batch_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                contract_sha256 TEXT NOT NULL,
                source_sha256 TEXT,
                runtime_git_commit TEXT,
                request_sha256 TEXT NOT NULL,
                item_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE batch_items (
                batch_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                symbol TEXT NOT NULL COLLATE NOCASE,
                timeframe TEXT NOT NULL,
                data_file TEXT NOT NULL COLLATE NOCASE,
                data_sha256 TEXT NOT NULL,
                planned_run_id TEXT NOT NULL UNIQUE,
                train_steps INTEGER NOT NULL,
                cpus_per_task INTEGER NOT NULL,
                memory TEXT NOT NULL,
                time_limit TEXT NOT NULL,
                status TEXT NOT NULL,
                job_id TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (batch_id, ordinal)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO batches VALUES (
                'batch-1', 'legacy', 'NEEDS_ATTENTION', ?, ?, ?, ?, 1, 1.0, 2.0
            )
            """,
            (
                CONTRACT_SHA256,
                SOURCE_SHA256,
                RUNTIME_GIT_COMMIT,
                "d" * 64,
            ),
        )
        conn.execute(
            """
            INSERT INTO batch_items VALUES (
                'batch-1', 0, '000617', 'D1', 'G:/data/000617_D1.parquet',
                ?, ?, 9000, 12, '32G', '06:00:00', 'NEEDS_ATTENTION',
                '581389', 'Slurm NODE_FAIL: NODE_FAIL', 1.0, 2.0
            )
            """,
            ("6" * 64, planned_run_id),
        )

    queue = TrainingQueue(database)
    item = queue.get_item(planned_run_id)

    assert item is not None
    assert item.planned_run_id == planned_run_id
    assert item.execution_run_id == planned_run_id
    assert item.attempt_number == 0
    assert item.status == NEEDS_ATTENTION
    assert item.job_id == "581389"
    assert item.error == "Slurm NODE_FAIL: NODE_FAIL"
    with sqlite3.connect(database) as conn:
        index = conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'index'
              AND name = 'idx_batch_items_execution_run_id'
            """
        ).fetchone()
    assert index == (1,)


def test_concurrent_claim_allows_only_one_active_item(tmp_path) -> None:
    db = tmp_path / "queue.sqlite3"
    queue = TrainingQueue(db)
    batch_id = _create_two_item_batch(queue, tmp_path)
    barrier = threading.Barrier(2)

    def claim() -> object:
        local_queue = TrainingQueue(db)
        barrier.wait()
        return local_queue.claim_next(batch_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))

    claimed = [item for item in results if item is not None]
    assert len(claimed) == 1
    assert claimed[0].ordinal == 0
    assert claimed[0].status == DISPATCHING
    assert queue.get_active_item() == claimed[0]


def test_batch_rejects_more_than_fifty_items(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    items = [
        BatchItemSpec(
            symbol=f"{index:06d}",
            timeframe="D1",
            data_file=tmp_path / f"{index:06d}.parquet",
            data_sha256="a" * 64,
            planned_run_id=f"run_{index:06d}",
        )
        for index in range(51)
    ]

    with pytest.raises(QueueValidationError, match="最多 50"):
        queue.create_batch(
            idempotency_key="too-many",
            contract_sha256=CONTRACT_SHA256,
            source_sha256=SOURCE_SHA256,
            items=items,
        )


def test_batch_rejects_duplicate_symbol(tmp_path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_data(first, b"first")
    _write_data(second, b"second")
    queue = TrainingQueue(tmp_path / "queue.sqlite3")

    with pytest.raises(QueueValidationError, match="symbol 重复"):
        queue.create_batch(
            idempotency_key="duplicate-symbol",
            contract_sha256=CONTRACT_SHA256,
            source_sha256=SOURCE_SHA256,
            items=[
                _spec(first, symbol="abc", planned_run_id="run_first"),
                _spec(second, symbol="ABC", planned_run_id="run_second"),
            ],
        )


def test_batch_rejects_path_alias_and_normalizes_case(tmp_path) -> None:
    data_dir = tmp_path / "DATA"
    data_file = data_dir / "sample.parquet"
    _write_data(data_file, b"same-file")
    alias = data_dir / "unused" / ".." / "sample.parquet"
    queue = TrainingQueue(tmp_path / "queue.sqlite3")

    assert TrainingQueue.normalize_data_path(alias) == os.path.normcase(
        str(data_file.resolve())
    )
    with pytest.raises(QueueValidationError, match="data_file 重复"):
        queue.create_batch(
            idempotency_key="duplicate-path",
            contract_sha256=CONTRACT_SHA256,
            source_sha256=SOURCE_SHA256,
            items=[
                _spec(data_file, symbol="600519", planned_run_id="run_first"),
                BatchItemSpec(
                    symbol="000858",
                    timeframe="D1",
                    data_file=alias,
                    data_sha256=hashlib.sha256(b"same-file").hexdigest(),
                    planned_run_id="run_second",
                ),
            ],
        )


def test_state_machine_rejects_skips_and_preserves_order(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)

    first = queue.claim_next(batch_id)
    assert first is not None
    assert first.planned_run_id == "run_600519"
    with pytest.raises(StateTransitionError, match="非法状态跳转"):
        queue.advance_item(first.planned_run_id, POSTPROCESSING)

    training = queue.advance_item(
        first.planned_run_id,
        TRAINING,
        job_id="568405",
    )
    assert training.job_id == "568405"
    assert queue.claim_next(batch_id) is None
    queue.advance_item(first.planned_run_id, POSTPROCESSING)
    queue.advance_item(first.planned_run_id, READY)

    second = queue.claim_next(batch_id)
    assert second is not None
    assert second.ordinal == 1
    assert second.status == DISPATCHING


def test_active_item_blocks_other_batches(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    first_file = tmp_path / "first.parquet"
    second_file = tmp_path / "second.parquet"
    _write_data(first_file, b"first")
    _write_data(second_file, b"second")
    first_batch = queue.create_batch(
        idempotency_key="batch-one",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        items=[_spec(first_file, planned_run_id="run_first")],
    ).batch.batch_id
    second_batch = queue.create_batch(
        idempotency_key="batch-two",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        items=[
            _spec(
                second_file,
                symbol="000858",
                planned_run_id="run_second",
            )
        ],
    ).batch.batch_id

    assert queue.claim_next(first_batch) is not None
    assert queue.can_claim_next(second_batch) is False
    assert queue.claim_next(second_batch) is None
    assert queue.list_items(second_batch)[0].status == QUEUED


def test_failed_item_pauses_batch_without_skipping(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    current = queue.claim_next(batch_id)
    assert current is not None

    failed = queue.fail_item(current.planned_run_id, "远端训练失败")

    assert failed.status == NEEDS_ATTENTION
    assert failed.error == "远端训练失败"
    assert queue.get_batch(batch_id).status == NEEDS_ATTENTION
    assert queue.has_pending_work() is True
    assert queue.claim_next(batch_id) is None
    assert [item.status for item in queue.list_items(batch_id)] == [
        NEEDS_ATTENTION,
        QUEUED,
    ]


def test_pre_submission_failure_can_resume_without_new_run_id(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    current = queue.claim_next(batch_id)
    assert current is not None
    queue.fail_item(current.planned_run_id, "上传前校验失败")

    resumed = queue.retry_pre_submission(current.planned_run_id)

    assert resumed.status == DISPATCHING
    assert resumed.job_id is None
    assert resumed.error is None
    assert queue.get_batch(batch_id).status == "ACTIVE"


def test_submitted_failure_cannot_reuse_original_run_id(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    current = queue.claim_next(batch_id)
    assert current is not None
    queue.advance_item(current.planned_run_id, TRAINING, job_id="568405")
    queue.fail_item(current.planned_run_id, "远端作业失败")

    with pytest.raises(StateTransitionError, match="禁止复用"):
        queue.retry_pre_submission(current.planned_run_id)


def test_recover_submitted_item_reuses_exact_remote_job(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    current = queue.claim_next(batch_id)
    assert current is not None
    queue.advance_item(current.planned_run_id, TRAINING, job_id="568571")
    queue.fail_item(current.planned_run_id, "批训练对应的 Slurm run 状态丢失")

    recovered = queue.recover_submitted_item(
        current.planned_run_id,
        "568571",
    )

    assert recovered.status == TRAINING
    assert recovered.job_id == "568571"
    assert recovered.error is None
    assert queue.get_batch(batch_id).status == "ACTIVE"


def test_recover_submitted_item_rejects_different_job_id(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    current = queue.claim_next(batch_id)
    assert current is not None
    queue.advance_item(current.planned_run_id, TRAINING, job_id="568571")
    queue.fail_item(current.planned_run_id, "误报失败")

    with pytest.raises(QueueValidationError, match="job_id"):
        queue.recover_submitted_item(current.planned_run_id, "568572")


def test_recover_submitted_item_rejects_unbound_legacy_job(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    current = queue.claim_next(batch_id)
    assert current is not None
    queue.fail_item(current.planned_run_id, "提交响应丢失")

    with pytest.raises(QueueValidationError, match="未冻结 job_id"):
        queue.recover_submitted_item(current.planned_run_id, "568571")


def test_source_hash_drift_pauses_batch_before_next_dispatch(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)

    with pytest.raises(SourceHashDriftError, match="训练源码"):
        queue.claim_next(batch_id, source_sha256="e" * 64)

    batch = queue.get_batch(batch_id)
    items = queue.list_items(batch_id)
    assert batch is not None
    assert batch.status == NEEDS_ATTENTION
    assert items[0].status == NEEDS_ATTENTION
    assert items[1].status == QUEUED
    assert queue.has_pending_work() is True


def test_legacy_batch_source_can_only_be_bound_once(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    original = queue.get_batch(batch_id)
    assert original is not None
    with queue._connect() as conn:
        conn.execute(
            """
            UPDATE batches
            SET source_sha256 = NULL, request_sha256 = ?
            WHERE batch_id = ?
            """,
            ("0" * 64, batch_id),
        )

    bound = queue.bind_batch_source(batch_id, SOURCE_SHA256)

    assert bound.source_sha256 == SOURCE_SHA256
    assert bound.request_sha256 == original.request_sha256
    with pytest.raises(SourceHashDriftError, match="不能改写"):
        queue.bind_batch_source(batch_id, "e" * 64)


def test_file_hash_drift_is_detected_and_pauses_batch(tmp_path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_data(data_file, b"original")
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = queue.create_batch(
        idempotency_key="hash-drift",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        items=[_spec(data_file)],
    ).batch.batch_id

    data_file.write_bytes(b"changed")
    check = queue.check_data_hash("run_600519")
    assert check.matches is False
    assert check.actual_sha256 != check.expected_sha256
    assert queue.can_claim_next(batch_id) is False

    with pytest.raises(DataHashDriftError):
        queue.claim_next(batch_id)

    assert queue.get_item("run_600519").status == NEEDS_ATTENTION
    assert queue.get_batch(batch_id).status == NEEDS_ATTENTION


def test_happy_path_finishes_batch(tmp_path) -> None:
    data_file = tmp_path / "600519_D1.parquet"
    _write_data(data_file, b"data")
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = queue.create_batch(
        idempotency_key="ready",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        items=[_spec(data_file)],
    ).batch.batch_id

    item = queue.claim_next(batch_id)
    assert item is not None
    queue.advance_item(item.planned_run_id, TRAINING, job_id="568405")
    queue.advance_item(item.planned_run_id, POSTPROCESSING)
    ready = queue.advance_item(item.planned_run_id, READY)

    assert ready.status == READY
    assert ready.job_id == "568405"
    assert queue.get_batch(batch_id).status == READY
    assert queue.get_active_item() is None


def test_claim_next_available_keeps_oldest_batch_contiguous(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    first_batch = _create_two_item_batch(queue, tmp_path, key="first-batch")
    third_file = tmp_path / "third.parquet"
    _write_data(third_file, b"third")
    second_batch = queue.create_batch(
        idempotency_key="second-batch",
        contract_sha256=CONTRACT_SHA256,
        source_sha256=SOURCE_SHA256,
        items=[
            _spec(
                third_file,
                symbol="000001",
                planned_run_id="run_third",
            )
        ],
    ).batch.batch_id

    first = queue.claim_next_available()
    assert first is not None
    assert first.batch_id == first_batch
    queue.advance_item(first.planned_run_id, TRAINING, job_id="1")
    queue.advance_item(first.planned_run_id, POSTPROCESSING)
    queue.advance_item(first.planned_run_id, READY)

    next_item = queue.claim_next_available()

    assert next_item is not None
    assert next_item.batch_id == first_batch
    assert queue.get_batch(second_batch).status == QUEUED
    assert queue.has_pending_work() is True


def test_node_fail_checkpoint_recovery_preserves_logical_identity(
    tmp_path,
) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    queue.advance_item(claimed.planned_run_id, TRAINING, job_id="581389")
    queue.fail_item(claimed.planned_run_id, "Slurm NODE_FAIL: NODE_FAIL")
    recovery_run_id = "run_20260729T130000Z_1234abcd"
    checkpoint_path = (
        f"checkpoints/{claimed.timeframe}/{claimed.data_sha256}/"
        "run_01785293593994423452/"
        f"ckpt_{claimed.symbol}_step_0100.pt"
    )

    recovered = queue.begin_checkpoint_recovery(
        claimed.planned_run_id,
        recovery_run_id=recovery_run_id,
        parent_job_id="581389",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256="5" * 64,
        checkpoint_size=5_965_894,
        checkpoint_step=100,
    )

    assert recovered.planned_run_id == claimed.planned_run_id
    assert recovered.execution_run_id == recovery_run_id
    assert recovered.attempt_number == 1
    assert recovered.parent_run_id == claimed.planned_run_id
    assert recovered.parent_job_id == "581389"
    assert recovered.resume_checkpoint_path == checkpoint_path
    assert recovered.resume_checkpoint_sha256 == "5" * 64
    assert recovered.resume_checkpoint_size == 5_965_894
    assert recovered.resume_checkpoint_step == 100
    assert recovered.status == DISPATCHING
    assert recovered.job_id is None
    assert recovered.error is None
    assert queue.get_batch(batch_id).status == "ACTIVE"
    assert (
        queue.get_item_by_execution_run_id(recovery_run_id)
        == recovered
    )

    active = queue.advance_item(
        recovered.planned_run_id,
        TRAINING,
        job_id="581999",
    )
    assert active.job_id == "581999"
    assert active.parent_job_id == "581389"


def test_checkpoint_recovery_is_one_time_and_node_fail_only(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _create_two_item_batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    queue.advance_item(claimed.planned_run_id, TRAINING, job_id="581389")
    queue.fail_item(claimed.planned_run_id, "not NODE_FAIL")
    checkpoint_path = (
        f"checkpoints/{claimed.timeframe}/{claimed.data_sha256}/"
        "run_01785293593994423452/"
        f"ckpt_{claimed.symbol}_step_0100.pt"
    )
    args = {
        "recovery_run_id": "run_20260729T130000Z_1234abcd",
        "parent_job_id": "581389",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": "5" * 64,
        "checkpoint_size": 5_965_894,
        "checkpoint_step": 100,
    }
    with pytest.raises(StateTransitionError, match="NODE_FAIL"):
        queue.begin_checkpoint_recovery(claimed.planned_run_id, **args)

    with queue._connect() as conn:
        conn.execute(
            """
            UPDATE batch_items
            SET error = ?
            WHERE planned_run_id = ?
            """,
            ("Slurm NODE_FAIL: NODE_FAIL", claimed.planned_run_id),
        )
    recovered = queue.begin_checkpoint_recovery(
        claimed.planned_run_id,
        **args,
    )
    queue.advance_item(
        recovered.planned_run_id,
        TRAINING,
        job_id="581999",
    )
    queue.fail_item(recovered.planned_run_id, "Slurm NODE_FAIL: NODE_FAIL")
    with pytest.raises(StateTransitionError, match="一次性"):
        queue.begin_checkpoint_recovery(
            recovered.planned_run_id,
            recovery_run_id="run_20260729T130100Z_8765abcd",
            parent_job_id="581999",
            checkpoint_path=checkpoint_path,
            checkpoint_sha256="5" * 64,
            checkpoint_size=5_965_894,
            checkpoint_step=100,
        )
