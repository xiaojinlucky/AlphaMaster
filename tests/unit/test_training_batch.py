from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from web.training_batch import TrainingBatchController
from web.training_batch import default_queue_path
from web.training_queue import (
    DISPATCHING,
    NEEDS_ATTENTION,
    POSTPROCESSING,
    QUEUED,
    READY,
    TRAINING,
    BatchItemSpec,
    TrainingQueue,
)


RUN_ONE = "run_20260724T010000Z_11111111"
RUN_TWO = "run_20260724T010001Z_22222222"
SOURCE_SHA256 = "f" * 64
RUNTIME_GIT_COMMIT = "a" * 40


class FakeTrainingManager:
    def __init__(self) -> None:
        self.payload: dict = {"active": False, "job": None}
        self.starts: list[dict] = []
        self.status_calls = 0
        self.source_sha256 = SOURCE_SHA256

    def status(self) -> dict:
        self.status_calls += 1
        return copy.deepcopy(self.payload)

    def start(self, **kwargs) -> dict:
        self.starts.append(dict(kwargs))
        job = {
            "run_id": kwargs["planned_run_id"],
            "slurm_job_id": "568500",
            "remote_state": "PENDING",
            "data_file": kwargs["data_file"],
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
        }
        self.payload = {"active": True, "job": job}
        return copy.deepcopy(job)

    def current_source_sha256(self) -> str:
        return self.source_sha256

    def current_git_commit(self) -> str:
        return RUNTIME_GIT_COMMIT

    def run_source_sha256(self, _run_id: str) -> str:
        return SOURCE_SHA256


class FakePipelineManager:
    def __init__(self) -> None:
        self.status = "WAITING_TRAINING"
        self.error: str | None = None

    def observe(self, _training: dict) -> dict:
        return {"status": self.status, "error": self.error}


def _write(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _batch(queue: TrainingQueue, tmp_path: Path) -> str:
    first = tmp_path / "600519_D1.parquet"
    second = tmp_path / "000858_D1.parquet"
    first_hash = _write(first, b"first")
    second_hash = _write(second, b"second")
    return queue.create_batch(
        idempotency_key="a50-v1",
        contract_sha256="a" * 64,
        source_sha256=SOURCE_SHA256,
        runtime_git_commit=RUNTIME_GIT_COMMIT,
        items=[
            BatchItemSpec(
                symbol="600519",
                timeframe="D1",
                data_file=first,
                data_sha256=first_hash,
                planned_run_id=RUN_ONE,
            ),
            BatchItemSpec(
                symbol="000858",
                timeframe="D1",
                data_file=second,
                data_sha256=second_hash,
                planned_run_id=RUN_TWO,
            ),
        ],
    ).batch.batch_id


def test_external_training_is_never_interrupted_or_claimed_over(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    training = FakeTrainingManager()
    pipeline = FakePipelineManager()
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=pipeline,
    )
    training.payload = {
        "active": True,
        "job": {
            "run_id": "run_20260724T000000Z_deadbeef",
            "remote_state": "PENDING",
        },
    }

    controller.advance_once()

    assert training.starts == []
    assert [item.status for item in queue.list_items(batch_id)] == [QUEUED, QUEUED]


def test_previous_run_postprocessing_finishes_before_batch_claim(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    training = FakeTrainingManager()
    pipeline = FakePipelineManager()
    pipeline.status = "POSTPROCESSING"
    training.payload = {
        "active": False,
        "job": {
            "run_id": "run_20260724T000000Z_deadbeef",
            "remote_state": READY,
        },
    }
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=pipeline,
    )

    controller.advance_once()

    assert training.starts == []
    assert [item.status for item in queue.list_items(batch_id)] == [QUEUED, QUEUED]


def test_batch_runs_training_then_postprocessing_then_next_item(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    training = FakeTrainingManager()
    pipeline = FakePipelineManager()
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=pipeline,
    )

    first = controller.advance_once()
    assert first["active_item"]["status"] == TRAINING
    assert training.starts[0]["planned_run_id"] == RUN_ONE
    assert training.starts[0]["from_scratch"] is True
    assert training.starts[0]["train_steps"] == 200
    assert training.starts[0]["cpus_per_task"] == 12
    assert training.starts[0]["memory"] == "32G"
    assert training.starts[0]["time_limit"] == "00:30:00"
    assert training.starts[0]["expected_source_sha256"] == SOURCE_SHA256
    assert (
        training.starts[0]["expected_git_commit"]
        == RUNTIME_GIT_COMMIT
    )

    training.payload["active"] = False
    training.payload["job"]["remote_state"] = READY
    pipeline.status = "POSTPROCESSING"
    controller.advance_once()
    assert queue.get_item(RUN_ONE).status == POSTPROCESSING

    pipeline.status = READY
    controller.advance_once()
    assert queue.get_item(RUN_ONE).status == READY

    controller.advance_once()
    assert queue.get_item(RUN_TWO).status == TRAINING
    assert queue.get_batch(batch_id).status == "ACTIVE"


def test_provided_training_snapshot_avoids_duplicate_remote_poll(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    queue.claim_next(batch_id)
    queue.advance_item(RUN_ONE, TRAINING, job_id="568500")
    training = FakeTrainingManager()
    training.payload = {
        "active": True,
        "job": {
            "run_id": RUN_ONE,
            "slurm_job_id": "568500",
            "remote_state": "RUNNING",
        },
    }
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once(training=copy.deepcopy(training.payload))

    assert training.status_calls == 0
    assert queue.get_item(RUN_ONE).status == TRAINING


def test_missing_snapshot_does_not_fail_submitted_item(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    queue.claim_next(batch_id)
    queue.advance_item(RUN_ONE, TRAINING, job_id="568571")
    controller = TrainingBatchController(
        queue=queue,
        training_manager=FakeTrainingManager(),
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once(
        training={"active": False, "job": None},
        pipeline={"status": "WAITING_TRAINING"},
    )

    assert queue.get_item(RUN_ONE).status == TRAINING
    assert queue.get_batch(batch_id).status == "ACTIVE"


def test_same_run_with_different_job_cannot_finish_frozen_item(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    queue.claim_next(batch_id)
    queue.advance_item(RUN_ONE, TRAINING, job_id="111")
    controller = TrainingBatchController(
        queue=queue,
        training_manager=FakeTrainingManager(),
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once(
        training={
            "active": False,
            "job": {
                "run_id": RUN_ONE,
                "slurm_job_id": "222",
                "remote_state": "FAILED",
                "error": "wrong job",
            },
        },
        pipeline={"status": "FAILED"},
    )

    assert queue.get_item(RUN_ONE).status == TRAINING
    assert queue.get_item(RUN_ONE).job_id == "111"
    assert queue.get_batch(batch_id).status == "ACTIVE"


def test_dispatch_waits_for_exact_job_id_after_submit_response_loss(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    training = FakeTrainingManager()
    training.payload = {
        "active": True,
        "job": {
            "run_id": RUN_ONE,
            "slurm_job_id": None,
            "remote_state": "SUBMITTING",
        },
    }
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once()

    persisted = queue.get_item(RUN_ONE)
    assert persisted is not None
    assert persisted.status == DISPATCHING
    assert persisted.job_id is None
    assert training.starts == []


def test_active_remote_job_recovers_false_attention_without_resubmit(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    queue.claim_next(batch_id)
    queue.advance_item(RUN_ONE, TRAINING, job_id="568571")
    queue.fail_item(RUN_ONE, "批训练对应的 Slurm run 状态丢失")
    training = FakeTrainingManager()
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once(
        training={
            "active": True,
            "job": {
                "run_id": RUN_ONE,
                "slurm_job_id": "568571",
                "remote_state": "PENDING",
            },
        },
        pipeline={"status": "WAITING_TRAINING"},
    )

    assert queue.get_item(RUN_ONE).status == TRAINING
    assert queue.get_batch(batch_id).status == "ACTIVE"
    assert training.starts == []


@pytest.mark.parametrize("pipeline_status", ["FAILED", "CANCELLED"])
def test_remote_ready_does_not_revive_permanent_postprocessing_failure(
    tmp_path,
    pipeline_status,
) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    queue.claim_next(batch_id)
    queue.advance_item(RUN_ONE, TRAINING, job_id="568571")
    queue.advance_item(RUN_ONE, POSTPROCESSING, job_id="568571")
    queue.fail_item(RUN_ONE, "永久后处理错误")
    training = FakeTrainingManager()
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )
    ready_training = {
        "active": False,
        "job": {
            "run_id": RUN_ONE,
            "slurm_job_id": "568571",
            "remote_state": READY,
        },
    }

    for _ in range(3):
        controller.advance_once(
            training=ready_training,
            pipeline={
                "status": pipeline_status,
                "error": "后处理不可恢复",
            },
        )
        item = queue.get_item(RUN_ONE)
        assert item.status == NEEDS_ATTENTION
        assert item.error == "永久后处理错误"
        assert queue.get_batch(batch_id).status == NEEDS_ATTENTION

    assert training.starts == []


def test_restart_recovers_dispatching_item_without_duplicate_start(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    training = FakeTrainingManager()
    training.payload = {
        "active": True,
        "job": {
            "run_id": RUN_ONE,
            "slurm_job_id": "568500",
            "remote_state": "PENDING",
        },
    }
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once()

    assert queue.get_item(RUN_ONE).status == TRAINING
    assert training.starts == []


def test_restart_dispatch_source_drift_never_calls_start(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    assert claimed.status == DISPATCHING
    restarted_training = FakeTrainingManager()
    restarted_training.source_sha256 = "e" * 64
    controller = TrainingBatchController(
        queue=queue,
        training_manager=restarted_training,
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once(
        training={"active": False, "job": None},
        pipeline={"status": "WAITING_TRAINING"},
    )

    item = queue.get_item(RUN_ONE)
    assert item.status == NEEDS_ATTENTION
    assert item.job_id is None
    assert queue.get_batch(batch_id).status == NEEDS_ATTENTION
    assert restarted_training.starts == []


def test_pre_submission_failure_resumes_same_frozen_run(tmp_path) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    queue.fail_item(RUN_ONE, "manifest local_source 不受支持")
    resumed = queue.retry_pre_submission(RUN_ONE)
    assert resumed.status == DISPATCHING

    training = FakeTrainingManager()
    training.payload = {
        "active": False,
        "job": {
            "run_id": RUN_ONE,
            "slurm_job_id": None,
            "remote_state": "FAILED",
        },
    }
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )

    controller.advance_once()

    assert queue.get_item(RUN_ONE).status == TRAINING
    assert [call["planned_run_id"] for call in training.starts] == [RUN_ONE]
    assert training.starts[0]["expected_source_sha256"] == SOURCE_SHA256


def test_default_queue_path_honors_explicit_runtime_override(
    tmp_path,
    monkeypatch,
) -> None:
    override = tmp_path / "isolated" / "training_queue.sqlite3"
    monkeypatch.setenv("ALPHAMASTER_TRAINING_QUEUE_DB", str(override))

    assert default_queue_path(tmp_path / "project") == override.resolve()


def test_checkpoint_recovery_dispatches_new_physical_run_without_scratch(
    tmp_path,
) -> None:
    queue = TrainingQueue(tmp_path / "queue.sqlite3")
    batch_id = _batch(queue, tmp_path)
    claimed = queue.claim_next(batch_id)
    assert claimed is not None
    queue.advance_item(RUN_ONE, TRAINING, job_id="581389")
    queue.fail_item(RUN_ONE, "Slurm NODE_FAIL: NODE_FAIL")
    recovery_run_id = "run_20260729T130000Z_1234abcd"
    checkpoint_path = (
        f"checkpoints/{claimed.timeframe}/{claimed.data_sha256}/"
        "run_01785293593994423452/"
        f"ckpt_{claimed.symbol}_step_0100.pt"
    )
    queue.begin_checkpoint_recovery(
        RUN_ONE,
        recovery_run_id=recovery_run_id,
        parent_job_id="581389",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256="5" * 64,
        checkpoint_size=5_965_894,
        checkpoint_step=100,
    )
    training = FakeTrainingManager()
    controller = TrainingBatchController(
        queue=queue,
        training_manager=training,
        pipeline_manager=FakePipelineManager(),
    )

    snapshot = controller.advance_once(
        training={"active": False, "job": None},
        pipeline={"status": "WAITING_TRAINING"},
    )

    assert len(training.starts) == 1
    started = training.starts[0]
    assert started["planned_run_id"] == recovery_run_id
    assert started["from_scratch"] is False
    assert started["checkpoint_recovery"] == {
        "parent_run_id": RUN_ONE,
        "parent_job_id": "581389",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": "5" * 64,
        "checkpoint_size": 5_965_894,
        "checkpoint_step": 100,
    }
    persisted = queue.get_item(RUN_ONE)
    assert persisted is not None
    assert persisted.execution_run_id == recovery_run_id
    assert persisted.job_id == "568500"
    assert persisted.status == TRAINING
    assert snapshot["active_item"]["planned_run_id"] == RUN_ONE
    assert snapshot["active_item"]["execution_run_id"] == recovery_run_id
