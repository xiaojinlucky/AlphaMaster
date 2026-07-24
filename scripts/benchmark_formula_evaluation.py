"""隔离基准：同一批公式在串行与线程并行下的评估速度和数值一致性。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.config import ModelConfig
from model_core.engine import (
    AlphaEngine,
    _build_walk_forward_folds,
    _repetition_penalty,
)
from model_core.target_contract import valid_target_length
from strategy_manager.signal import compute_target_positions_stateless


def _sample_formulas(engine: AlphaEngine, count: int, seed: int) -> list[list[int]]:
    random.seed(seed)
    torch.manual_seed(seed)
    inp = torch.zeros((count, 1), dtype=torch.long, device=ModelConfig.DEVICE)
    stack_depths = [0] * count
    previous_tokens: list[int | None] = [None] * count
    infected_chain_lengths = [0] * count
    sampled: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(ModelConfig.MAX_FORMULA_LEN):
            logits, _, _ = engine.model(inp)
            logits = engine.sampler.apply_mask_to_logits(
                logits,
                stack_depths,
                step,
                ModelConfig.MAX_FORMULA_LEN,
                prev_tokens=previous_tokens,
                infected_chain_lens=infected_chain_lengths,
            )
            token = Categorical(logits=logits).sample()
            sampled.append(token)
            inp = torch.cat([inp, token.unsqueeze(1)], dim=1)
            for index in range(count):
                value = int(token[index].item())
                stack_depths[index] += engine.sampler.delta[value]
                previous_tokens[index] = value
                infected_chain_lengths[index] = engine.sampler.update_infection(
                    value,
                    infected_chain_lengths[index],
                )
    return torch.stack(sampled, dim=1).tolist()


def _evaluate_one(
    engine: AlphaEngine,
    index: int,
    formula: list[int],
    features: torch.Tensor,
    target_returns: torch.Tensor,
    folds: list[dict[str, int]],
    use_walk_forward: bool,
) -> dict[str, Any]:
    with torch.no_grad():
        factor = engine.vm.execute(formula, features)
    if factor is None:
        return {
            "index": index,
            "status": "none",
            "reward": torch.tensor(-5.0, device=features.device),
            "validation": torch.tensor(-5.0, device=features.device),
            "factor": None,
        }
    valid_steps = valid_target_length(target_returns.shape[1])
    factor_valid = factor[:, :valid_steps]
    target_valid = target_returns[:, :valid_steps]
    if factor_valid.std() < 1e-4:
        return {
            "index": index,
            "status": "constant",
            "reward": torch.tensor(-2.0, device=features.device),
            "validation": torch.tensor(-2.0, device=features.device),
            "factor": None,
        }

    with torch.no_grad():
        if use_walk_forward:
            train_scores: list[torch.Tensor] = []
            validation_scores: list[torch.Tensor] = []
            for fold in folds:
                train_score, validation_score = engine.bt.evaluate_fold(
                    factor,
                    target_returns,
                    fold["train_start"],
                    fold["train_end"],
                    fold["val_start"],
                    fold["val_end"],
                )
                train_ic, _ = AlphaEngine._compute_ic_aligned(
                    factor_valid[:, fold["train_start"] : fold["train_end"]],
                    target_valid[:, fold["train_start"] : fold["train_end"]],
                )
                train_scores.append(
                    ModelConfig.REWARD_ALPHA
                    * AlphaEngine._apply_ic_gate(train_score, train_ic)
                )
                validation_scores.append(validation_score)
            reward = torch.stack(train_scores).mean()
            validation = torch.stack(validation_scores).mean()
        else:
            reward, _ = engine.bt.evaluate(factor, {}, target_returns)
            full_ic, _ = AlphaEngine._compute_ic_aligned(
                factor_valid,
                target_valid,
            )
            reward = AlphaEngine._apply_ic_gate(
                ModelConfig.REWARD_ALPHA * reward,
                full_ic,
            )
            validation = reward

    return {
        "index": index,
        "status": "ok",
        "reward": reward,
        "validation": validation,
        "factor": factor_valid,
    }


def _finalize_in_formula_order(
    engine: AlphaEngine,
    formulas: list[list[int]],
    rows: list[dict[str, Any]],
    starting_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按生产顺序施加相关性惩罚并更新冠军池，保留真实顺序依赖。"""
    if starting_state is None:
        engine.best_score = -float("inf")
        engine.best_formula = None
        engine.factor_pool = []
        engine._factor_pool_counter = 0
        engine._elite_pool = []
        engine._elite_counter = 0
    else:
        engine.best_score = starting_state["best_score"]
        engine.best_formula = starting_state["best_formula"]
        engine.factor_pool = [
            (score, counter, factor.clone())
            for score, counter, factor in starting_state["factor_pool"]
        ]
        engine._factor_pool_counter = starting_state["factor_pool_counter"]
        engine._elite_pool = [
            (score, counter, list(tokens), birth)
            for score, counter, tokens, birth in starting_state["elite_pool"]
        ]
        engine._elite_counter = starting_state["elite_counter"]
    finalized: list[dict[str, Any]] = []
    for formula, row in zip(formulas, rows, strict=True):
        if row["status"] != "ok":
            finalized.append(
                {
                    "index": row["index"],
                    "status": row["status"],
                    "reward": float(row["reward"].item()),
                    "validation": float(row["validation"].item()),
                }
            )
            continue
        factor = row["factor"]
        reward = row["reward"]
        validation = row["validation"]
        penalty = _repetition_penalty(formula)
        if penalty > 0:
            reward = reward - penalty
            validation = validation - penalty
        reward = engine._apply_corr_penalty(reward, factor)
        validation = engine._apply_corr_penalty(validation, factor)
        train_value = float(reward.item())
        validation_value = float(validation.item())
        if validation_value > engine.best_score:
            overfit = train_value > 0.5 and validation_value < train_value * 0.5
            exposure = float(
                compute_target_positions_stateless(factor).abs().mean().item()
            )
            if not overfit and exposure >= 0.05:
                engine.best_score = validation_value
                engine.best_formula = list(formula)
                engine._update_factor_pool(validation_value, factor)
        engine._update_elite_pool(validation_value, formula, step=1)
        finalized.append(
            {
                "index": row["index"],
                "status": "ok",
                "reward": train_value,
                "validation": validation_value,
            }
        )
    pool_identity = []
    for score, counter, factor in sorted(engine.factor_pool, key=lambda item: item[:2]):
        factor_bytes = factor.detach().cpu().contiguous().numpy().tobytes()
        pool_identity.append(
            {
                "score": score,
                "counter": counter,
                "factor_sha256": hashlib.sha256(factor_bytes).hexdigest(),
            }
        )
    state = {
        "best_score": engine.best_score,
        "best_formula": engine.best_formula,
        "factor_pool_counter": engine._factor_pool_counter,
        "factor_pool": pool_identity,
        "elite_counter": engine._elite_counter,
        "elite_pool": [
            {
                "score": score,
                "counter": counter,
                "tokens": tokens,
                "birth": birth,
            }
            for score, counter, tokens, birth in sorted(engine._elite_pool)
        ],
    }
    return finalized, state


def _capture_engine_state(engine: AlphaEngine) -> dict[str, Any]:
    return {
        "best_score": engine.best_score,
        "best_formula": (
            list(engine.best_formula) if engine.best_formula is not None else None
        ),
        "factor_pool": [
            (score, counter, factor.clone())
            for score, counter, factor in engine.factor_pool
        ],
        "factor_pool_counter": engine._factor_pool_counter,
        "elite_pool": [
            (score, counter, list(tokens), birth)
            for score, counter, tokens, birth in engine._elite_pool
        ],
        "elite_counter": engine._elite_counter,
    }


def _run_mode(
    *,
    engine: AlphaEngine,
    formulas: list[list[int]],
    features: torch.Tensor,
    target_returns: torch.Tensor,
    folds: list[dict[str, int]],
    use_walk_forward: bool,
    workers: int,
    intra_threads: int,
    starting_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], float, float]:
    torch.set_num_threads(intra_threads)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if workers == 1:
        results = [
            _evaluate_one(
                engine,
                index,
                formula,
                features,
                target_returns,
                folds,
                use_walk_forward,
            )
            for index, formula in enumerate(formulas)
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="alpha-eval-benchmark",
        ) as pool:
            futures = [
                pool.submit(
                    _evaluate_one,
                    engine,
                    index,
                    formula,
                    features,
                    target_returns,
                    folds,
                    use_walk_forward,
                )
                for index, formula in enumerate(formulas)
            ]
            results_by_index = {
                result["index"]: result for result in (future.result() for future in futures)
            }
        results = [results_by_index[index] for index in range(len(formulas))]
    finalized, state = _finalize_in_formula_order(
        engine,
        formulas,
        results,
        starting_state=starting_state,
    )
    return (
        finalized,
        state,
        time.perf_counter() - wall_start,
        time.process_time() - cpu_start,
    )


def _result_digest(results: list[dict[str, Any]]) -> str:
    payload = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _max_delta(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> float:
    maximum = 0.0
    for left, right in zip(expected, actual, strict=True):
        if left["index"] != right["index"] or left["status"] != right["status"]:
            return math.inf
        maximum = max(
            maximum,
            abs(left["reward"] - right["reward"]),
            abs(left["validation"] - right["validation"]),
        )
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--periods-per-year", type=int, required=True)
    parser.add_argument("--minimum-bars", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--expected-cpus", type=int, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--score-tolerance", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != args.expected_cpus:
        raise RuntimeError(
            f"CPU affinity 不匹配: {len(affinity)} != {args.expected_cpus}"
        )
    if args.rounds < 5 or args.rounds % 5 != 0:
        raise ValueError("rounds 必须是 5 的正整数倍，确保五种模式位置平衡")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manager = ParquetDataManager(
        args.data_file,
        expected_source_id=args.source_id,
        expected_periods_per_year=args.periods_per_year,
        expected_minimum_bars=args.minimum_bars,
    )
    manager.load()
    engine = AlphaEngine(
        data_manager=manager,
        use_lord_regularization=False,
        target_symbol=manager.symbol,
    )
    features = manager.feat_tensor.to(ModelConfig.DEVICE)
    target_returns = manager.target_ret.to(ModelConfig.DEVICE)
    valid_steps = valid_target_length(int(target_returns.shape[1]))
    folds = _build_walk_forward_folds(
        valid_steps,
        engine.n_folds,
        gap=ModelConfig.WF_GAP,
    )
    use_walk_forward = len(folds) > 1 and not (
        folds[0]["train_start"] == 0
        and folds[0]["train_end"] == valid_steps
    )

    new_count = ModelConfig.BATCH_SIZE - max(
        1,
        int(ModelConfig.BATCH_SIZE * ModelConfig.ELITE_REPLAY_FRAC),
    )
    new_formulas = _sample_formulas(engine, new_count, args.seed)
    elite_count = ModelConfig.BATCH_SIZE - new_count
    formulas = new_formulas + new_formulas[:elite_count]

    modes = [
        {"name": "serial_12x1", "workers": 1, "intra_threads": 12},
        {"name": "parallel_2x6", "workers": 2, "intra_threads": 6},
        {"name": "parallel_4x3", "workers": 4, "intra_threads": 3},
        {"name": "parallel_6x2", "workers": 6, "intra_threads": 2},
        {"name": "parallel_12x1", "workers": 12, "intra_threads": 1},
    ]

    # 先做一批串行公式，构造与真实训练后续 step 一致的非空冠军/因子/精英池。
    _run_mode(
        engine=engine,
        formulas=formulas[:32],
        features=features,
        target_returns=target_returns,
        folds=folds,
        use_walk_forward=use_walk_forward,
        workers=1,
        intra_threads=12,
    )
    starting_state = _capture_engine_state(engine)
    if not starting_state["factor_pool"] or not starting_state["elite_pool"]:
        raise RuntimeError("预热未形成非空因子池和精英池，基准不具备后续 step 语义")

    # 预热后再从完全相同的非空状态运行串行基线。
    baseline, baseline_state, _, _ = _run_mode(
        engine=engine,
        formulas=formulas,
        features=features,
        target_returns=target_returns,
        folds=folds,
        use_walk_forward=use_walk_forward,
        workers=1,
        intra_threads=12,
        starting_state=starting_state,
    )
    measurements: dict[str, list[dict[str, float]]] = {
        mode["name"]: [] for mode in modes
    }
    deltas: dict[str, list[float]] = {mode["name"]: [] for mode in modes}
    digests: dict[str, list[str]] = {mode["name"]: [] for mode in modes}
    state_digests: dict[str, list[str]] = {mode["name"]: [] for mode in modes}

    for round_index in range(args.rounds):
        ordered_modes = modes[round_index:] + modes[:round_index]
        for mode in ordered_modes:
            result, state, wall_seconds, cpu_seconds = _run_mode(
                engine=engine,
                formulas=formulas,
                features=features,
                target_returns=target_returns,
                folds=folds,
                use_walk_forward=use_walk_forward,
                workers=mode["workers"],
                intra_threads=mode["intra_threads"],
                starting_state=starting_state,
            )
            measurements[mode["name"]].append(
                {
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "allocated_cpu_utilization": cpu_seconds
                    / max(wall_seconds * 12, 1e-9),
                }
            )
            deltas[mode["name"]].append(_max_delta(baseline, result))
            digests[mode["name"]].append(_result_digest(result))
            state_digests[mode["name"]].append(_result_digest([state]))

    summary: list[dict[str, Any]] = []
    serial_median = statistics.median(
        row["wall_seconds"] for row in measurements["serial_12x1"]
    )
    for mode in modes:
        rows = measurements[mode["name"]]
        median_wall = statistics.median(row["wall_seconds"] for row in rows)
        summary.append(
            {
                **mode,
                "wall_seconds": [row["wall_seconds"] for row in rows],
                "median_wall_seconds": median_wall,
                "median_cpu_seconds": statistics.median(
                    row["cpu_seconds"] for row in rows
                ),
                "median_allocated_cpu_utilization": statistics.median(
                    row["allocated_cpu_utilization"] for row in rows
                ),
                "speedup_vs_serial": serial_median / median_wall,
                "max_abs_score_delta": max(deltas[mode["name"]]),
                "result_digests": digests[mode["name"]],
                "state_digests": state_digests[mode["name"]],
            }
        )

    formula_digest = hashlib.sha256(
        json.dumps(formulas, separators=(",", ":")).encode()
    ).hexdigest()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = {
        "format": "alphamaster_formula_eval_benchmark_v2",
        "pid": os.getpid(),
        "data_file": str(Path(args.data_file).resolve()),
        "data_sha256": manager.data_sha256,
        "symbol": manager.symbol,
        "timeframe": manager.timeframe,
        "bars": manager.data_rows,
        "seed": args.seed,
        "formula_sha256": formula_digest,
        "formula_count": len(formulas),
        "fold_count": len(folds),
        "rounds": args.rounds,
        "torch_version": torch.__version__,
        "cpu_count_visible": os.cpu_count(),
        "cpu_affinity": affinity,
        "source_sha256": args.source_sha256,
        "benchmark_script_sha256": script_sha256,
        "model_config": {
            "batch_size": ModelConfig.BATCH_SIZE,
            "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
            "reward_alpha": ModelConfig.REWARD_ALPHA,
            "corr_threshold": ModelConfig.CORR_THRESHOLD,
            "corr_penalty": ModelConfig.CORR_PENALTY,
            "factor_top_k": ModelConfig.FACTOR_TOP_K,
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "baseline_digest": _result_digest(baseline),
        "baseline_state_digest": _result_digest([baseline_state]),
        "starting_factor_pool_size": len(starting_state["factor_pool"]),
        "starting_elite_pool_size": len(starting_state["elite_pool"]),
        "modes": summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    max_delta = max(row["max_abs_score_delta"] for row in summary)
    if not math.isfinite(max_delta) or max_delta > args.score_tolerance:
        raise RuntimeError(
            f"串并行数值差异超限: {max_delta} > {args.score_tolerance}"
        )
    expected_state = output["baseline_state_digest"]
    if any(
        digest != expected_state
        for row in summary
        for digest in row["state_digests"]
    ):
        raise RuntimeError("串并行状态更新结果不一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
