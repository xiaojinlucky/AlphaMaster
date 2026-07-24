"""从严格验证的 A50 切分合同创建 50 股持久化 Slurm 训练队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_a50_sealed_split import (
    SealedSplitError,
    load_a50_sealed_split,
)
from web.training_queue import (
    BatchItemSpec,
    CreateBatchResult,
    TrainingQueue,
    TrainingQueueError,
)
from web.slurm_training_manager import current_training_source_sha256


_MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:K|M|G|T)$")
_TIME_LIMIT_RE = re.compile(r"^(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$")


class A50BatchPlanError(RuntimeError):
    """A50 批训练计划无法冻结或写入持久队列。"""


@dataclass(frozen=True)
class A50BatchPlan:
    contract_sha256: str
    source_sha256: str
    batch_id: str
    idempotency_key: str
    train_steps: int
    cpus_per_task: int
    memory: str
    time_limit: str
    items: tuple[BatchItemSpec, ...]


def _resolve_project_path(value: str) -> Path:
    raw = Path(value)
    return (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()


def _positive_int(value: Any, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise A50BatchPlanError(f"{label} 必须是 1 至 {maximum} 的整数")
    return value


def _campaign_sha256(
    *,
    contract_sha256: str,
    train_steps: int,
    cpus_per_task: int,
    memory: str,
    time_limit: str,
) -> str:
    payload = {
        "contract_sha256": contract_sha256,
        "train_steps": train_steps,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "time_limit": time_limit,
        "from_scratch": True,
        "partition": "cpu",
        "qos": "normal",
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_a50_batch_plan(
    split_contract: str | Path,
    *,
    train_steps: int = 200,
    cpus_per_task: int = 12,
    memory: str = "32G",
    time_limit: str = "00:30:00",
) -> A50BatchPlan:
    """验证全部物理数据，并冻结 50 个互不重复的 run ID 与资源预算。"""
    try:
        contract = load_a50_sealed_split(split_contract)
    except SealedSplitError as exc:
        raise A50BatchPlanError(str(exc)) from exc
    train_steps = _positive_int(train_steps, "train_steps", 1_000_000)
    cpus_per_task = _positive_int(cpus_per_task, "cpus_per_task", 64)
    memory = str(memory).strip().upper()
    time_limit = str(time_limit).strip()
    if _MEMORY_RE.fullmatch(memory) is None:
        raise A50BatchPlanError("memory 必须使用 K/M/G/T 单位，例如 32G")
    if _TIME_LIMIT_RE.fullmatch(time_limit) is None:
        raise A50BatchPlanError("time_limit 必须是 [D-]HH:MM:SS")
    source_sha256 = current_training_source_sha256()

    campaign_hash = _campaign_sha256(
        contract_sha256=contract["contract_sha256"],
        train_steps=train_steps,
        cpus_per_task=cpus_per_task,
        memory=memory,
        time_limit=time_limit,
    )
    training_root = _resolve_project_path(contract["training_data_dir"])
    timestamp = f"{contract['universe_snapshot_date']}T235959Z"
    items: list[BatchItemSpec] = []
    run_ids: set[str] = set()
    for ordinal, item in enumerate(contract["items"]):
        symbol = item["symbol"]
        suffix = hashlib.sha256(
            f"{campaign_hash}:{ordinal}:{symbol}".encode("ascii")
        ).hexdigest()[:8]
        run_id = f"run_{timestamp}_{suffix}"
        if run_id in run_ids:
            raise A50BatchPlanError("确定性 planned_run_id 发生碰撞")
        run_ids.add(run_id)
        training = item["training"]
        items.append(
            BatchItemSpec(
                symbol=symbol,
                timeframe="D1",
                data_file=training_root / training["data_filename"],
                data_sha256=training["data_sha256"],
                planned_run_id=run_id,
                train_steps=train_steps,
                cpus_per_task=cpus_per_task,
                memory=memory,
                time_limit=time_limit,
            )
        )
    if len(items) != 50:
        raise A50BatchPlanError("A50 批训练必须恰好包含 50 个标的")
    return A50BatchPlan(
        contract_sha256=contract["contract_sha256"],
        source_sha256=source_sha256,
        batch_id=f"batch_a50_{campaign_hash[:16]}",
        idempotency_key=f"a50:{campaign_hash}",
        train_steps=train_steps,
        cpus_per_task=cpus_per_task,
        memory=memory,
        time_limit=time_limit,
        items=tuple(items),
    )


def enqueue_a50_batch(
    plan: A50BatchPlan,
    *,
    queue_db: str | Path,
) -> CreateBatchResult:
    queue = TrainingQueue(queue_db)
    return queue.create_batch(
        idempotency_key=plan.idempotency_key,
        contract_sha256=plan.contract_sha256,
        source_sha256=plan.source_sha256,
        items=plan.items,
        batch_id=plan.batch_id,
    )


def _summary(
    plan: A50BatchPlan,
    result: CreateBatchResult | None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "check_only": result is None,
        "created": result.created if result is not None else False,
        "batch_id": plan.batch_id,
        "contract_sha256": plan.contract_sha256,
        "source_sha256": plan.source_sha256,
        "item_count": len(plan.items),
        "train_steps": plan.train_steps,
        "cpus_per_task": plan.cpus_per_task,
        "memory": plan.memory,
        "time_limit": plan.time_limit,
        "first_symbol": plan.items[0].symbol,
        "last_symbol": plan.items[-1].symbol,
        "status": result.batch.status if result is not None else "VALIDATED",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证 A50 训练/封存合同并创建 50 股串行训练队列"
    )
    parser.add_argument("--split-contract", required=True)
    parser.add_argument(
        "--queue-db",
        default=str(PROJECT_ROOT / "local_runs" / "training_queue.sqlite3"),
    )
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--cpus-per-task", type=int, default=12)
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--time-limit", default="00:30:00")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        plan = build_a50_batch_plan(
            args.split_contract,
            train_steps=args.train_steps,
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            time_limit=args.time_limit,
        )
        result = (
            None
            if args.check_only
            else enqueue_a50_batch(plan, queue_db=args.queue_db)
        )
    except (A50BatchPlanError, TrainingQueueError) as exc:
        print(f"A50 批训练入队失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_summary(plan, result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
