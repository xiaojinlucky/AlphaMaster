"""大 A 训练完成后的回测与虚拟信号流水线。"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from config import Config
from data_pipeline.dataset_contracts import source_family
from strategy_manager.live_signal import evaluate_signal, min_exposure
from web.data_sources.base import bars_to_raw_dict
from web.data_sources.tongdaxin_source import TongdaxinSource
from web.progress import get_published_bundle
from web.realtime_manager import _load_strategy_meta
from web.signal_ledger import SignalLedger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS_ROOT = PROJECT_ROOT / "local_runs"
PIPELINE_FORMAT = "alphamaster_a_share_pipeline_v1"
RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$")
TIMEFRAME_TO_TDX = {
    "M5": "5m",
    "M15": "15m",
    "H1": "1h",
    "D1": "1d",
}
TERMINAL_STATES = {"READY", "FAILED", "CANCELLED"}
MAX_POSTPROCESS_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 15.0


class PipelinePermanentError(RuntimeError):
    """身份、合同或代码错误；重复执行不会自行恢复。"""


class PipelineTransientError(RuntimeError):
    """外部行情或本机子进程的短暂故障；允许有限重试。"""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class ASharePipelineManager:
    """同一训练 run 内依次执行 replay 回测和通达信虚拟信号。"""

    def __init__(
        self,
        local_runs_root: Path | None = None,
        *,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        self.local_runs_root = (local_runs_root or LOCAL_RUNS_ROOT).resolve()
        self.local_runs_root.mkdir(parents=True, exist_ok=True)
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(str(run_id)):
            raise RuntimeError("流水线 run_id 非法")
        run_dir = (self.local_runs_root / run_id).resolve()
        if run_dir.parent != self.local_runs_root:
            raise RuntimeError("流水线运行目录越界")
        return run_dir

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "pipeline_state.json"

    def _read_state(self, run_id: str) -> dict[str, Any] | None:
        path = self._state_path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now()
        _atomic_json(self._state_path(str(state["run_id"])), state)

    def snapshot(self, run_id: str | None) -> dict[str, Any] | None:
        """只读本机流水线快照，不写状态，也不启动后处理线程。"""
        normalized = str(run_id or "").strip()
        if not normalized:
            return None
        with self._lock:
            return self._read_state(normalized)

    def recover_failed(self, run_id: str, *, expected_error: str) -> dict[str, Any]:
        """在代码缺陷修复后，显式恢复同一 run 的后处理，不重新训练。"""
        normalized_error = str(expected_error or "").strip()
        if not normalized_error:
            raise RuntimeError("恢复后处理必须提供预期错误")
        with self._lock:
            thread = self._threads.get(run_id)
            if thread is not None and thread.is_alive():
                raise RuntimeError("后处理仍在运行，不能恢复")
            state = self._read_state(run_id)
            if state is None:
                raise RuntimeError("后处理状态不存在")
            if str(state.get("status") or "").upper() != "FAILED":
                raise RuntimeError("只有 FAILED 后处理可以恢复")
            actual_error = str(state.get("error") or "").strip()
            if actual_error != normalized_error:
                raise RuntimeError("当前后处理错误与预期错误不一致")

            recovered_at = _utc_now()
            state["status"] = "WAITING_TRAINING"
            state["error"] = None
            state.pop("finished_at", None)
            state.pop("next_retry_at", None)
            state["recovery_count"] = int(state.get("recovery_count") or 0) + 1
            state["last_recovery"] = {
                "recovered_at": recovered_at,
                "previous_error": actual_error,
            }
            for name in ("backtest", "signal"):
                stage = state.setdefault("stages", {}).setdefault(
                    name,
                    {"status": "PENDING"},
                )
                if str(stage.get("status") or "").upper() in {
                    "FAILED",
                    "RETRY_WAIT",
                    "RUNNING",
                }:
                    state["stages"][name] = {"status": "PENDING"}
            self._write_state(state)
            return dict(state)

    @staticmethod
    def _training_stage(job: dict[str, Any]) -> dict[str, Any]:
        remote_state = str(job.get("remote_state") or "").upper()
        if remote_state == "READY":
            status = "READY"
        elif remote_state in {"FAILED", "CANCELLED"}:
            status = remote_state
        else:
            status = "RUNNING"
        return {
            "status": status,
            "backend": "slurm",
            "job_id": job.get("slurm_job_id"),
            "node": job.get("compute_node"),
            "elapsed": job.get("elapsed"),
            "allocated_cpus": job.get("allocated_cpus"),
            "max_rss": job.get("max_rss"),
            "total_cpu": job.get("total_cpu"),
            "best_score": None,
            "error": job.get("error") or job.get("last_poll_error"),
        }

    def observe(self, training: dict[str, Any]) -> dict[str, Any] | None:
        """读取训练状态；READY 的大 A run 自动进入后处理。"""
        job = training.get("job") if isinstance(training, dict) else None
        if not isinstance(job, dict) or not job.get("run_id"):
            return None
        run_id = str(job["run_id"])
        run_dir = self._run_dir(run_id)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict) or source_family(
            str(manifest.get("local_source") or "")
        ) != "ashare":
            return None

        with self._lock:
            state = self._read_state(run_id)
            if state is None:
                state = {
                    "format": PIPELINE_FORMAT,
                    "run_id": run_id,
                    "symbol": manifest.get("symbol"),
                    "timeframe": manifest.get("timeframe"),
                    "status": "WAITING_TRAINING",
                    "created_at": _utc_now(),
                    "stages": {
                        "training": self._training_stage(job),
                        "backtest": {"status": "PENDING"},
                        "signal": {"status": "PENDING"},
                    },
                    "error": None,
                    "attempts": 0,
                }
            else:
                state.setdefault("attempts", 0)
                previous_score = (
                    state.get("stages", {})
                    .get("training", {})
                    .get("best_score")
                )
                training_stage = self._training_stage(job)
                if previous_score is not None:
                    training_stage["best_score"] = previous_score
                state.setdefault("stages", {})["training"] = training_stage

            remote_state = str(job.get("remote_state") or "").upper()
            backtest_status = str(
                state.get("stages", {}).get("backtest", {}).get("status") or ""
            ).upper()
            signal_status = str(
                state.get("stages", {}).get("signal", {}).get("status") or ""
            ).upper()
            if (
                int(job.get("retry_count") or 0) > 0
                and state.get("status") in {"FAILED", "CANCELLED"}
                and remote_state not in {"READY", "FAILED", "CANCELLED"}
                and backtest_status == "PENDING"
                and signal_status == "PENDING"
            ):
                # 训练提交前失败后恢复，同一 run 的旧错误不能继续污染新状态。
                state["status"] = "WAITING_TRAINING"
                state["error"] = None
            self._write_state(state)

            if remote_state == "READY":
                self._ensure_started(run_id, job, manifest)
            elif (
                remote_state in {"FAILED", "CANCELLED"}
                and state.get("status") not in TERMINAL_STATES
            ):
                state["status"] = remote_state
                state["error"] = job.get("error")
                self._write_state(state)
            return self._read_state(run_id)

    def _ensure_started(
        self,
        run_id: str,
        job: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        with self._lock:
            state = self._read_state(run_id) or {}
            if state.get("status") in TERMINAL_STATES:
                return
            if (
                state.get("status") == "RETRY_WAIT"
                and time.time() < float(state.get("next_retry_at") or 0.0)
            ):
                return
            thread = self._threads.get(run_id)
            if thread is not None and thread.is_alive():
                return
            state["status"] = "POSTPROCESSING"
            state["error"] = None
            state["attempts"] = int(state.get("attempts") or 0) + 1
            state.pop("next_retry_at", None)
            self._write_state(state)
            thread = threading.Thread(
                target=self._run_pipeline,
                args=(run_id, dict(job), dict(manifest)),
                name=f"ashare-pipeline-{run_id}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()

    def _run_backtest(
        self,
        *,
        run_id: str,
        strategy_file: Path,
        data_file: Path,
    ) -> dict[str, Any]:
        post_dir = self._run_dir(run_id) / "postprocess"
        post_dir.mkdir(parents=True, exist_ok=True)
        log_path = post_dir / "backtest.log"
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "run_backtest.py"),
            "--strategy-file",
            str(strategy_file),
            "--data-file",
            str(data_file),
            "--evaluation-mode",
            "replay",
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["LOGURU_COLORIZE"] = "0"
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                result = subprocess.run(
                    command,
                    cwd=post_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    shell=False,
                    timeout=1800,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise PipelineTransientError("流水线回测超时，等待自动重试") from exc
        if result.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            raise PipelinePermanentError(
                f"流水线回测失败 (exit_code={result.returncode}): "
                + " | ".join(tail)
            )
        report_path = post_dir / "backtest_output" / "multi_factor_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelinePermanentError("流水线回测未生成合法报告") from exc
        if report.get("evaluation_mode") != "replay":
            raise PipelinePermanentError("流水线首轮回测必须明确标记为 replay")
        return {
            "report_path": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "chart_path": str(
                (post_dir / "backtest_output" / "portfolio_equity.png").relative_to(
                    PROJECT_ROOT
                )
            ).replace("\\", "/"),
            "evaluation_mode": "replay",
            "score_start": report.get("score_start"),
            "score_end": report.get("score_end"),
            "symbol": report.get("symbol"),
            "timeframe": report.get("timeframe"),
            "data_sha256": report.get("data_sha256"),
            "dataset_id": report.get("dataset_id"),
            "portfolio": report.get("portfolio") or {},
            "symbols": report.get("symbols") or {},
        }

    def _run_signal(
        self,
        *,
        run_id: str,
        symbol: str,
        timeframe: str,
        strategy_file: Path,
    ) -> dict[str, Any]:
        tdx_timeframe = TIMEFRAME_TO_TDX.get(timeframe)
        if tdx_timeframe is None:
            raise PipelinePermanentError(
                f"通达信虚拟信号尚不支持训练周期 {timeframe}"
            )
        meta = _load_strategy_meta(str(strategy_file))
        source = TongdaxinSource()
        try:
            try:
                bars = source.fetch_bars(
                    symbol,
                    tdx_timeframe,
                    500,
                    drop_forming=True,
                )
            except Exception as exc:
                raise PipelineTransientError(
                    f"通达信行情暂时不可用: {exc}"
                ) from exc
        finally:
            source.disconnect()
        result = evaluate_signal(meta["formula"], bars_to_raw_dict(bars))
        if result.get("state") != "ok":
            raise PipelinePermanentError(
                f"虚拟信号计算失败: {result.get('message') or result}"
            )

        post_dir = self._run_dir(run_id) / "postprocess"
        ledger = SignalLedger(post_dir / "signal_simulation.sqlite3")
        watch_id = (
            f"pipeline:{run_id}:tongdaxin:{symbol}:{tdx_timeframe}:"
            f"{meta['fingerprint'][:12]}"
        )
        event, created = ledger.process_bar(
            watch_id=watch_id,
            source="tongdaxin",
            symbol=symbol,
            timeframe=tdx_timeframe,
            strategy_name=strategy_file.stem,
            strategy_fingerprint=str(meta["fingerprint"]),
            bar_ts=int(bars[-1].ts),
            price=float(bars[-1].close),
            raw_position=float(result["position"]),
            factor_value=float(result["factor_value"]),
            strength=float(result["strength"]),
            minimum_exposure=float(
                getattr(Config, "MIN_TRADE_EXPOSURE", min_exposure())
            ),
            rebalance_delta=float(
                getattr(Config, "SIGNAL_REBALANCE_DELTA", 0.10)
            ),
            stop_loss_pct=float(getattr(Config, "STOP_LOSS_PCT", -0.02)),
            take_profit_pct=float(getattr(Config, "TAKE_PROFIT_PCT", 0.04)),
            take_profit_remaining_ratio=float(
                getattr(Config, "SIGNAL_TAKE_PROFIT_REMAINING_RATIO", 0.50)
            ),
        )
        if created and event.should_push:
            ledger.record_delivery(
                event.event_id,
                "SKIPPED",
                "流水线虚拟信号验证：未发送飞书",
            )
        event_payload = ledger.get_event(event.event_id)
        if event_payload is None:
            raise RuntimeError("虚拟信号事件写入后无法复读")
        payload = {
            "run_id": run_id,
            "strategy_file": str(strategy_file),
            "strategy_fingerprint": meta["fingerprint"],
            "market_source": "tongdaxin",
            "symbol": symbol,
            "timeframe": tdx_timeframe,
            "bars_used": len(bars),
            "last_bar_ts": int(bars[-1].ts),
            "last_close": float(bars[-1].close),
            "raw_signal": result,
            "lifecycle_event": event_payload,
            "created": created,
        }
        output_path = post_dir / "signal_simulation.json"
        _atomic_json(output_path, payload)
        return {
            **payload,
            "output_path": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }

    def _run_pipeline(
        self,
        run_id: str,
        job: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        state = self._read_state(run_id)
        if state is None:
            return
        try:
            symbol = str(manifest["symbol"])
            timeframe = str(manifest["timeframe"])
            data_file = Path(str(job["data_file"])).resolve()
            if not data_file.is_file():
                raise PipelinePermanentError("流水线训练数据文件不存在")
            bundle = get_published_bundle(symbol)
            if bundle is None:
                raise PipelinePermanentError("Slurm 发布包缺失或产物哈希校验失败")
            expected_artifact_root = (self._run_dir(run_id) / "artifacts").resolve()
            if (
                bundle.get("run_id") != run_id
                or bundle.get("timeframe") != timeframe
                or bundle.get("data_sha256") != manifest.get("data_sha256")
                or bundle.get("result_manifest_sha256")
                != job.get("result_manifest_sha256")
                or Path(str(bundle.get("data_file") or "")).resolve() != data_file
                or Path(bundle["artifact_root_path"]).resolve()
                != expected_artifact_root
            ):
                raise PipelinePermanentError(
                    "Slurm 发布包与当前训练 run、数据或结果清单不一致"
                )
            strategy_file = Path(bundle["strategy_path"]).resolve()
            strategy = json.loads(strategy_file.read_text(encoding="utf-8"))
            if not isinstance(strategy, dict):
                raise PipelinePermanentError("发布策略格式非法")
            strategy_run_id = strategy.get("run_id")
            if strategy_run_id is not None and strategy_run_id != run_id:
                raise PipelinePermanentError("发布策略与当前训练 run_id 不一致")
            if strategy.get("data_sha256") != manifest.get("data_sha256"):
                raise PipelinePermanentError("发布策略与当前训练数据哈希不一致")
            strategy = dict(strategy)
            strategy["run_id"] = run_id
            post_dir = self._run_dir(run_id) / "postprocess"
            strategy_snapshot = post_dir / "published_strategy.json"
            _atomic_json(strategy_snapshot, strategy)

            with self._lock:
                state = self._read_state(run_id) or state
                state["stages"]["training"]["best_score"] = strategy.get("best_score")
                prior_backtest = state["stages"].get("backtest") or {}
                if prior_backtest.get("status") != "READY":
                    state["stages"]["backtest"] = {
                        "status": "RUNNING",
                        "started_at": _utc_now(),
                    }
                self._write_state(state)
            if prior_backtest.get("status") == "READY":
                backtest = dict(prior_backtest)
            else:
                backtest = self._run_backtest(
                    run_id=run_id,
                    strategy_file=strategy_snapshot,
                    data_file=data_file,
                )
                if backtest.get("symbols", {}).get(symbol) is None:
                    raise PipelinePermanentError("回测报告缺少当前大 A 品种")
                if (
                    backtest.get("symbol") != symbol
                    or backtest.get("timeframe") != timeframe
                    or backtest.get("data_sha256") != manifest.get("data_sha256")
                ):
                    raise PipelinePermanentError(
                        "回测报告与当前训练数据身份不一致"
                    )
            with self._lock:
                state = self._read_state(run_id) or state
                if prior_backtest.get("status") != "READY":
                    state["stages"]["backtest"] = {
                        "status": "READY",
                        "finished_at": _utc_now(),
                        **backtest,
                    }
                state["stages"]["signal"] = {
                    "status": "RUNNING",
                    "started_at": _utc_now(),
                }
                self._write_state(state)

            signal = self._run_signal(
                run_id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                strategy_file=strategy_snapshot,
            )
            with self._lock:
                state = self._read_state(run_id) or state
                state["stages"]["signal"] = {
                    "status": "READY",
                    "finished_at": _utc_now(),
                    **signal,
                }
                state["status"] = "READY"
                state["finished_at"] = _utc_now()
                state["error"] = None
                self._write_state(state)
        except PipelineTransientError as exc:
            with self._lock:
                state = self._read_state(run_id) or state
                attempts = int(state.get("attempts") or 0)
                state["finished_at"] = _utc_now()
                state["error"] = str(exc)
                stages = state.setdefault("stages", {})
                for name in ("backtest", "signal"):
                    stage = stages.setdefault(name, {"status": "PENDING"})
                    if stage.get("status") == "RUNNING":
                        stage["status"] = "RETRY_WAIT"
                        stage["error"] = str(exc)
                        stage["finished_at"] = _utc_now()
                        break
                if attempts < MAX_POSTPROCESS_ATTEMPTS:
                    state["status"] = "RETRY_WAIT"
                    state["next_retry_at"] = time.time() + self.retry_delay_seconds
                else:
                    state["status"] = "FAILED"
                self._write_state(state)
        except Exception as exc:
            with self._lock:
                state = self._read_state(run_id) or state
                state["status"] = "FAILED"
                state["finished_at"] = _utc_now()
                state["error"] = str(exc)
                stages = state.setdefault("stages", {})
                for name in ("backtest", "signal"):
                    stage = stages.setdefault(name, {"status": "PENDING"})
                    if stage.get("status") == "RUNNING":
                        stage["status"] = "FAILED"
                        stage["error"] = str(exc)
                        stage["finished_at"] = _utc_now()
                        break
                self._write_state(state)
        finally:
            with self._lock:
                self._threads.pop(run_id, None)

    def wait(self, run_id: str, timeout: float = 60.0) -> dict[str, Any] | None:
        """测试/命令行使用：等待当前后处理线程结束。"""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout=timeout)
        return self._read_state(run_id)


a_share_pipeline_manager = ASharePipelineManager()
