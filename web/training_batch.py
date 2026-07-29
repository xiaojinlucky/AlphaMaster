"""把持久化批训练队列接到现有 Slurm 与大 A 后处理状态机。"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from web.training_queue import (
    DISPATCHING,
    NEEDS_ATTENTION,
    POSTPROCESSING,
    READY,
    TRAINING,
    BatchItemRecord,
    SourceHashDriftError,
    TrainingQueue,
)


class TrainingBatchController:
    """单步推进批队列；调用方可安全地在后台循环中反复调用。"""

    def __init__(
        self,
        *,
        queue: TrainingQueue,
        training_manager: Any,
        pipeline_manager: Any,
    ) -> None:
        self.queue = queue
        self.training_manager = training_manager
        self.pipeline_manager = pipeline_manager
        self.admission_lock = threading.RLock()

    @staticmethod
    def _job(training: dict[str, Any]) -> dict[str, Any]:
        job = training.get("job") if isinstance(training, dict) else None
        return job if isinstance(job, dict) else {}

    @staticmethod
    def _remote_state(job: dict[str, Any]) -> str:
        return str(job.get("remote_state") or "").upper()

    @staticmethod
    def _job_matches_item(
        item: BatchItemRecord,
        job: dict[str, Any],
    ) -> bool:
        return (
            str(job.get("run_id") or "") == item.execution_run_id
            and item.job_id is not None
            and str(job.get("slurm_job_id") or "") == item.job_id
        )

    def _fail(self, item: BatchItemRecord, message: str) -> BatchItemRecord:
        return self.queue.fail_item(item.planned_run_id, message)

    def _ensure_source_identity(
        self,
        item: BatchItemRecord,
        job: dict[str, Any],
    ) -> BatchItemRecord:
        if str(job.get("run_id") or "") != item.execution_run_id:
            return item
        try:
            run_source = self.training_manager.run_source_sha256(
                item.execution_run_id
            )
            batch = self.queue.get_batch(item.batch_id)
            if batch is None:
                return self._fail(item, "批次记录不存在")
            if batch.source_sha256 is None:
                batch = self.queue.bind_batch_source(
                    item.batch_id,
                    run_source,
                )
            if batch.source_sha256 != run_source:
                return self._fail(
                    item,
                    "当前 run 的训练源码与批次冻结身份不一致",
                )
        except Exception as exc:
            return self._fail(item, f"训练源码身份校验失败: {exc}")
        return item

    def _dispatch(
        self,
        item: BatchItemRecord,
        training: dict[str, Any],
    ) -> BatchItemRecord:
        job = self._job(training)
        run_id = str(job.get("run_id") or "")
        if bool(training.get("active")) and run_id != item.execution_run_id:
            # 可能是 API 状态读取后，另一个进程先提交了任务；不抢占、不取消。
            return item
        retryable_pre_submission_failure = (
            run_id == item.execution_run_id
            and self._remote_state(job) == "FAILED"
            and not job.get("slurm_job_id")
        )
        should_start = (
            run_id != item.execution_run_id
            or retryable_pre_submission_failure
        )
        if should_start:
            try:
                batch = self.queue.get_batch(item.batch_id)
                if batch is None:
                    return self._fail(item, "批次记录不存在")
                expected_source_sha256 = batch.source_sha256
                if expected_source_sha256 is None:
                    return self._fail(
                        item,
                        "批次尚未冻结训练源码 SHA-256",
                    )
                expected_git_commit = batch.runtime_git_commit
                if expected_git_commit is None:
                    return self._fail(
                        item,
                        "批次尚未冻结服务器运行提交",
                    )
                current_source_sha256 = (
                    self.training_manager.current_source_sha256()
                )
            except Exception as exc:
                return self._fail(item, f"训练源码身份校验失败: {exc}")
            if current_source_sha256 != expected_source_sha256:
                return self._fail(
                    item,
                    "当前训练源码与批次冻结身份不一致",
                )
        try:
            if should_start:
                job = self.training_manager.start(
                    data_file=item.data_file,
                    symbol=item.symbol,
                    timeframe=item.timeframe,
                    mode="ftmo",
                    from_scratch=item.attempt_number == 0,
                    planned_run_id=item.execution_run_id,
                    checkpoint_recovery=(
                        {
                            "parent_run_id": item.parent_run_id,
                            "parent_job_id": item.parent_job_id,
                            "checkpoint_path": item.resume_checkpoint_path,
                            "checkpoint_sha256": item.resume_checkpoint_sha256,
                            "checkpoint_size": item.resume_checkpoint_size,
                            "checkpoint_step": item.resume_checkpoint_step,
                        }
                        if item.attempt_number == 1
                        else None
                    ),
                    expected_source_sha256=expected_source_sha256,
                    expected_git_commit=expected_git_commit,
                    train_steps=item.train_steps,
                    cpus_per_task=item.cpus_per_task,
                    memory=item.memory,
                    time_limit=item.time_limit,
                )
            else:
                job = dict(job)
        except Exception as exc:
            refreshed = self.training_manager.status()
            refreshed_job = self._job(refreshed)
            if (
                str(refreshed_job.get("run_id") or "")
                == item.execution_run_id
            ):
                job = refreshed_job
            elif bool(refreshed.get("active")):
                return item
            else:
                return self._fail(item, f"Slurm 提交失败: {exc}")

        state = self._remote_state(job)
        if state in {"FAILED", "CANCELLED"}:
            return self._fail(
                item,
                str(job.get("error") or f"Slurm 训练进入 {state}"),
            )
        job_id = job.get("slurm_job_id")
        if not job_id:
            # 提交响应丢失时继续保持 DISPATCHING，等同一 run 恢复出
            # 确切 job_id 后再冻结到队列，绝不以空 job_id 进入 TRAINING。
            return item
        return self.queue.advance_item(
            item.planned_run_id,
            TRAINING,
            job_id=job_id,
        )

    def _observe_training(
        self,
        item: BatchItemRecord,
        training: dict[str, Any],
    ) -> BatchItemRecord:
        job = self._job(training)
        if not self._job_matches_item(item, job):
            # 空快照、测试替身或无关 run 不能把已提交作业写成失败。
            # 只有同一 run/job 的明确终态才有权改变队列状态。
            return item
        state = self._remote_state(job)
        if state in {"FAILED", "CANCELLED"}:
            return self._fail(
                item,
                str(job.get("error") or f"Slurm 训练进入 {state}"),
            )
        if state == READY:
            return self.queue.advance_item(
                item.planned_run_id,
                POSTPROCESSING,
                job_id=job.get("slurm_job_id"),
            )
        return item

    def _observe_postprocessing(
        self,
        item: BatchItemRecord,
        training: dict[str, Any],
        pipeline: dict[str, Any] | None = None,
    ) -> BatchItemRecord:
        job = self._job(training)
        if not self._job_matches_item(item, job):
            return item
        if pipeline is None:
            pipeline = self.pipeline_manager.observe(training)
        if not isinstance(pipeline, dict):
            return self._fail(item, "当前训练不是可识别的大 A 后处理任务")
        status = str(pipeline.get("status") or "").upper()
        if status == READY:
            return self.queue.advance_item(
                item.planned_run_id,
                READY,
                job_id=job.get("slurm_job_id"),
            )
        if status in {"FAILED", "CANCELLED"}:
            return self._fail(
                item,
                str(pipeline.get("error") or f"大 A 后处理进入 {status}"),
            )
        return item

    def advance_once(
        self,
        *,
        training: dict[str, Any] | None = None,
        pipeline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """推进一格，不阻塞等待 Slurm 或本机后处理完成。"""
        with self.admission_lock:
            training = training or self.training_manager.status()
            if pipeline is None:
                pipeline = self.pipeline_manager.observe(training)
            current_job = self._job(training)
            current_run_id = str(current_job.get("run_id") or "")
            current_job_id = current_job.get("slurm_job_id")
            current_remote_state = self._remote_state(current_job)
            if (
                current_run_id
                and current_job_id
                and current_remote_state
                in {
                    "PENDING",
                    "RUNNING",
                    "COMPLETED",
                    "DOWNLOADING",
                }
            ):
                persisted = self.queue.get_item_by_execution_run_id(
                    current_run_id
                )
                if (
                    persisted is not None
                    and persisted.status == NEEDS_ATTENTION
                ):
                    self.queue.recover_submitted_item(
                        persisted.planned_run_id,
                        current_job_id,
                    )
            item = self.queue.get_active_item()
            if item is None:
                if bool(training.get("active")):
                    return self.snapshot()
                pipeline_status = (
                    str(pipeline.get("status") or "").upper()
                    if isinstance(pipeline, dict)
                    else ""
                )
                current_job = self._job(training)
                if (
                    current_job.get("run_id")
                    and pipeline_status
                    and pipeline_status
                    not in {READY, "FAILED", "CANCELLED"}
                ):
                    # 当前 run 的回测和信号模拟尚未结束，不能让下一次训练
                    # 覆盖它正在核对的发布策略。
                    return self.snapshot()
                try:
                    item = self.queue.claim_next_available(
                        source_sha256=(
                            self.training_manager.current_source_sha256()
                        ),
                    )
                except SourceHashDriftError:
                    return self.snapshot()
                if item is None:
                    return self.snapshot()

            job = self._job(training)
            if str(job.get("run_id") or "") == item.execution_run_id:
                checked = self._ensure_source_identity(item, job)
                if checked.status == NEEDS_ATTENTION:
                    return self.snapshot(item.batch_id)
                item = checked

            if item.status == DISPATCHING:
                item = self._dispatch(item, training)
            elif item.status == TRAINING:
                item = self._observe_training(item, training)
            elif item.status == POSTPROCESSING:
                item = self._observe_postprocessing(
                    item,
                    training,
                    pipeline,
                )
            elif item.status == NEEDS_ATTENTION:
                return self.snapshot(item.batch_id)
            return self.snapshot(item.batch_id)

    def snapshot(self, batch_id: str | None = None) -> dict[str, Any]:
        active = self.queue.get_active_item()
        batch = self.queue.get_batch(batch_id) if batch_id else None
        items = self.queue.list_items(batch_id) if batch is not None else []
        return {
            "active_item": asdict(active) if active is not None else None,
            "batch": asdict(batch) if batch is not None else None,
            "items": [asdict(item) for item in items],
            "has_pending_work": self.queue.has_pending_work(),
        }


def default_queue_path(project_root: str | Path) -> Path:
    override = os.getenv("ALPHAMASTER_TRAINING_QUEUE_DB", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(project_root).resolve() / "local_runs" / "training_queue.sqlite3"
