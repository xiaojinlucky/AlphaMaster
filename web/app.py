"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.legacy_mt5_registry import (
    apply_registration_plan,
    build_single_file_registration_plan,
    write_registration_report,
)
from data_pipeline.a_share_akshare import AKSHARE_SLICE_TRAINING
from data_pipeline.dataset_contracts import AKSHARE_HFQ_SOURCE_ID, source_family
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from web.file_dialog import pick_parquet_file, pick_strategy_file
from web.progress import (
    get_symbol_progress,
    get_strategy_for_export,
    invalidate_checkpoint_cache,
    list_strategies,
    build_strategy_export_filename,
)
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import load_settings, save_settings
from web.strategy_file import (
    inspect_strategy_file,
    resolve_strategy_file,
    strategy_path_for_symbol,
    sync_best_strategy_for_symbol,
)
from web.training_manager import training_manager
from web.training_batch import TrainingBatchController, default_queue_path
from web.training_queue import (
    BatchItemSpec,
    DataHashDriftError,
    IdempotencyConflictError,
    QueueValidationError,
    TrainingQueue,
)
from web.training_time import get_training_time_summary
from web.training_package import build_training_export_zip
from web.backtest_manager import backtest_manager
from web.a_share_pipeline import a_share_pipeline_manager
from web.realtime_manager import realtime_manager
from web.data_sources.factory import list_sources
from web.feishu_notify import send_text as send_feishu_text
from web.feishu_notify import validate_feishu_webhook_url
from strategy_manager.live_signal import min_exposure

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"

setup_logging()
logger = get_logger()

app = FastAPI(title="AlphaMaster Training", version="1.1.0")
training_queue = TrainingQueue(default_queue_path(ROOT))
training_batch_controller = TrainingBatchController(
    queue=training_queue,
    training_manager=training_manager,
    pipeline_manager=a_share_pipeline_manager,
)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
_TOKEN_EXEMPT_API_PATHS = frozenset({"/api/health", "/api/session"})
_CONTROL_TOKEN = secrets.token_urlsafe(32)
_REGISTRATION_PLAN_LOCK = threading.Lock()
_REGISTRATION_PLANS: dict[str, dict[str, Any]] = {}
_PIPELINE_MONITOR_INTERVAL_SECONDS = 5.0
_PIPELINE_MONITOR_STOP = threading.Event()
_PIPELINE_MONITOR_THREAD: threading.Thread | None = None
_TRAINING_LOG_REFRESH_LOCK = threading.Lock()
_TRAINING_LOG_REFRESH_THREAD: threading.Thread | None = None
_SENSITIVE_SETTING_KEYS = frozenset(
    {
        "ai_api_key",
        "feishu_webhook_url",
        "feishu_secret",
        "slurm_ssh_key",
        "ssh_key",
        "ssh_path",
    }
)


def _effective_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return port
    return 443 if scheme == "https" else 80 if scheme == "http" else None


def _parse_host_header(raw_host: str, scheme: str) -> tuple[str, int | None]:
    try:
        parsed = urlsplit(f"//{raw_host}")
        port = _effective_port(scheme, parsed.port)
    except ValueError as exc:
        raise HTTPException(403, "非法 Host") from exc
    hostname = (parsed.hostname or "").lower()
    if (
        hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(403, "仅允许本机回环地址访问")
    return hostname, port


def _validate_origin(origin: str, *, request_scheme: str, host_port: int) -> None:
    try:
        parsed = urlsplit(origin)
        origin_port = _effective_port(parsed.scheme.lower(), parsed.port)
    except ValueError as exc:
        raise HTTPException(403, "非法 Origin") from exc
    if (
        parsed.scheme.lower() != request_scheme.lower()
        or (parsed.hostname or "").lower() not in _LOOPBACK_HOSTS
        or origin_port != host_port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(403, "仅允许同端口回环来源访问")


def _public_settings(settings: dict[str, Any]) -> dict[str, Any]:
    def is_sensitive(key: str) -> bool:
        normalized = key.lower()
        return normalized in _SENSITIVE_SETTING_KEYS or (
            "ssh" in normalized
            and any(marker in normalized for marker in ("path", "key", "config", "known_hosts"))
        )

    public = {k: v for k, v in settings.items() if not is_sensitive(k)}
    public["has_ai_api_key"] = bool(settings.get("ai_api_key")) and (
        settings.get("ai_api_key_provider") == settings.get("ai_provider")
    )
    public["feishu_webhook_configured"] = bool(settings.get("feishu_webhook_url"))
    public["feishu_secret_configured"] = bool(settings.get("feishu_secret"))
    return public


@app.middleware("http")
async def enforce_local_control_boundary(request: Request, call_next):
    try:
        _, host_port = _parse_host_header(
            request.headers.get("host", ""), request.url.scheme
        )
        origin = request.headers.get("origin")
        if origin:
            _validate_origin(
                origin,
                request_scheme=request.url.scheme,
                host_port=host_port,
            )
        if (
            request.url.path.startswith("/api/")
            and request.url.path not in _TOKEN_EXEMPT_API_PATHS
        ):
            supplied = request.headers.get("x-alphamaster-control", "")
            if not supplied or not secrets.compare_digest(supplied, _CONTROL_TOKEN):
                raise HTTPException(403, "缺少或无效的本机控制令牌")
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


class StartTrainingRequest(BaseModel):
    data_file: str
    from_scratch: bool = False


class StopTrainingRequest(BaseModel):
    run_id: str
    slurm_job_id: str


class BatchTrainingItemRequest(BaseModel):
    data_file: str
    planned_run_id: str
    train_steps: int = 200
    cpus_per_task: int = 12
    memory: str = "32G"
    time_limit: str = "00:30:00"


class CreateTrainingBatchRequest(BaseModel):
    idempotency_key: str
    contract_sha256: str
    runtime_git_commit: str | None = None
    items: list[BatchTrainingItemRequest]


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    last_backtest_data_file: str | None = None
    last_strategy_file: str | None = None
    debug_mode: bool | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    ai_model: str | None = None
    ai_thinking: bool | None = None
    ai_reasoning_effort: str | None = None
    bt_commission_pct: float | None = None
    bt_slippage_pct: float | None = None


class AnalyzeTrainingRequest(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    model: str | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None
    symbol: str | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str
    data_file: str | None = None
    evaluation_mode: str = "auto"
    score_start: str | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None


class RegisterLegacyMt5Request(BaseModel):
    data_file: str
    plan_sha256: str
    source_acknowledgement: str


class AddWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str


class RemoveWatchRequest(BaseModel):
    id: str


class FeishuSettingsRequest(BaseModel):
    enabled: bool | None = None
    webhook_url: str | None = None
    secret: str | None = None


class FeishuTestRequest(BaseModel):
    webhook_url: str | None = None
    secret: str | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_error(f"{request.method} {request.url.path} unhandled", exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    if is_debug_mode():
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    if response.status_code >= 400:
        log_error(f"{request.method} {request.url.path} -> HTTP {response.status_code}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_error(f"{request.method} {request.url.path} HTTP {exc.status_code}: {exc.detail}")
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_error(f"{request.method} {request.url.path} crashed", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"},
    )


def _decorate_data_info(
    info: dict[str, Any],
    *,
    include_registration_plan: bool = False,
) -> dict[str, Any]:
    payload = dict(info)
    capabilities = dict(payload.get("capabilities") or {})
    # 数据能力装饰只依赖固定后端配置，不能为了展示文件卡片阻塞远端轮询锁。
    backend = os.getenv("TRAINING_BACKEND", "").strip().lower()
    training_compatible = bool(
        capabilities.get("remote_training")
        if backend == "slurm"
        else capabilities.get("local_training")
    )
    capabilities["training"] = training_compatible
    payload["capabilities"] = capabilities
    payload["training_backend"] = backend
    if not training_compatible and backend == "slurm":
        payload["reason_code"] = "manifest_required_for_slurm"
        payload["message"] = (
            "当前是远程训练：该文件尚未注册。"
            "确认它来自旧 MT5 后，可生成受审计 manifest。"
        )
    else:
        payload["reason_code"] = ""
    if include_registration_plan and capabilities.get("legacy_registration"):
        plan = build_single_file_registration_plan(payload["data_file"])
        if plan["summary"]["eligible"] == 1:
            plan_sha = str(plan["plan_sha256"])
            with _REGISTRATION_PLAN_LOCK:
                _REGISTRATION_PLANS[plan_sha] = plan
                while len(_REGISTRATION_PLANS) > 32:
                    _REGISTRATION_PLANS.pop(next(iter(_REGISTRATION_PLANS)))
            payload["registration_plan_sha256"] = plan_sha
    return payload


def _inspect_or_http(
    path: str,
    *,
    include_registration_plan: bool = False,
) -> dict[str, Any]:
    try:
        return _decorate_data_info(
            inspect_parquet_file(path),
            include_registration_plan=include_registration_plan,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _job_to_dict(job: Any) -> dict[str, Any]:
    if isinstance(job, dict):
        return dict(job)
    converter = getattr(job, "to_dict", None)
    if callable(converter):
        payload = converter()
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("训练管理器返回了非法任务结构")


def _browse_data_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening native file picker")
    try:
        path = pick_parquet_file()
    except Exception as exc:
        log_error("File picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("File picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected file: %s", path)
    info = _inspect_or_http(path, include_registration_plan=True)
    save_settings({"last_data_file": info["data_file"]})
    return {"ok": True, "cancelled": False, **info}


def _browse_backtest_data_file() -> dict[str, Any]:
    try:
        path = pick_parquet_file()
    except Exception as exc:
        log_error("Backtest file picker failed", exc)
        raise HTTPException(500, f"回测数据选择失败: {exc}") from exc
    if not path:
        return {"ok": False, "cancelled": True}
    info = _inspect_or_http(path)
    save_settings({"last_backtest_data_file": info["data_file"]})
    return {"ok": True, "cancelled": False, **info}


def _strategy_context() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    train_symbol = None
    if data_file:
        try:
            train_symbol = inspect_parquet_file(data_file).get("symbol")
        except Exception:
            pass

    resolved = resolve_strategy_file(
        settings.get("last_strategy_file") or "",
        train_symbol,
    )
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(
                resolved,
                data_file_hint=settings.get("last_data_file") or None,
            )
        except Exception as e:
            strategy_info = {
                "strategy_file": resolved,
                "valid": False,
                "message": str(e),
            }
    return {
        "last_strategy_file": resolved,
        "strategy_file": strategy_info,
        "train_symbol": train_symbol,
    }


def _browse_strategy_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening strategy file picker")
    try:
        path = pick_strategy_file()
    except Exception as exc:
        log_error("Strategy file picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("Strategy file picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected strategy: %s", path)
    info = _inspect_strategy_or_http(path)
    save_settings({"last_strategy_file": info["strategy_file"]})
    return {"ok": True, "cancelled": False, **info}


def _inspect_strategy_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_strategy_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _resolve_train_symbol(symbol: str | None = None) -> str | None:
    if symbol:
        return symbol.strip() or None
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return inspect_parquet_file(data_file).get("symbol")
    except Exception:
        return None


def _wait_training_idle(timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        training_manager.status()
        if not training_manager.status().get("active"):
            return
        time.sleep(0.2)


def _sync_and_persist_best_strategy(
    symbol: str,
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any] | None:
    invalidate_checkpoint_cache()
    hint = data_file_hint
    if not hint:
        job = training_manager.snapshot().get("job") or {}
        if str(job.get("symbol") or "") == symbol:
            hint = job.get("data_file") or None
    if not hint:
        hint = load_settings().get("last_data_file") or None
    info = sync_best_strategy_for_symbol(symbol, data_file_hint=hint)
    if info:
        save_settings({"last_strategy_file": info["strategy_file"]})
    return info


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.1.0"}


@app.get("/api/session")
def api_session() -> JSONResponse:
    return JSONResponse(
        content={"control_token": _CONTROL_TOKEN},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/routes")
def api_routes() -> dict[str, Any]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": sorted(routes, key=lambda r: r["path"])}


@app.get("/api/debug/logs")
def api_debug_logs(lines: int = 200) -> dict[str, Any]:
    if not is_debug_mode():
        raise HTTPException(403, "调试模式未开启")
    return debug_snapshot(max(1, min(int(lines), 500)))


@app.post("/api/debug/client-log")
def api_client_log(req: ClientLogRequest) -> dict[str, bool]:
    msg = req.message
    if req.context:
        msg = f"{msg} | context={req.context}"
    if req.level == "error":
        log_error(f"[client] {msg}")
    elif is_debug_mode():
        logger.info("[client] %s", msg)
    return {"ok": True}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return _public_settings(load_settings())


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.last_data_file is not None:
        payload["last_data_file"] = req.last_data_file
    if req.last_backtest_data_file is not None:
        payload["last_backtest_data_file"] = req.last_backtest_data_file
    if req.last_strategy_file is not None:
        payload["last_strategy_file"] = req.last_strategy_file
    if req.debug_mode is not None:
        payload["debug_mode"] = req.debug_mode
    if req.ai_provider is not None:
        payload["ai_provider"] = req.ai_provider
    if req.ai_api_key is not None:
        payload["ai_api_key"] = req.ai_api_key
        payload["ai_api_key_provider"] = req.ai_provider or load_settings().get(
            "ai_provider",
            "deepseek",
        )
    if req.ai_model is not None:
        payload["ai_model"] = req.ai_model
    if req.ai_thinking is not None:
        payload["ai_thinking"] = req.ai_thinking
    if req.ai_reasoning_effort is not None:
        payload["ai_reasoning_effort"] = req.ai_reasoning_effort
    if req.bt_commission_pct is not None:
        payload["bt_commission_pct"] = req.bt_commission_pct
    if req.bt_slippage_pct is not None:
        payload["bt_slippage_pct"] = req.bt_slippage_pct
    saved = save_settings(payload)
    if req.debug_mode is not None:
        set_debug_mode(req.debug_mode)
    return {"ok": True, **_public_settings(saved)}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    backtest_data_file = settings.get("last_backtest_data_file") or ""
    file_info = None
    backtest_file_info = None
    if data_file:
        try:
            file_info = _decorate_data_info(
                inspect_parquet_file(data_file),
                include_registration_plan=True,
            )
        except Exception as e:
            file_info = {
                "data_file": data_file,
                "valid": False,
                "message": str(e),
            }
    if backtest_data_file:
        try:
            backtest_file_info = _decorate_data_info(
                inspect_parquet_file(backtest_data_file)
            )
        except Exception as e:
            backtest_file_info = {
                "data_file": backtest_data_file,
                "valid": False,
                "message": str(e),
            }
    strat_ctx = _strategy_context()
    return {
        "train_steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "reward_mode": ModelConfig.REWARD_MODE,
        "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        "device": str(ModelConfig.DEVICE),
        "training_backend": os.getenv("TRAINING_BACKEND", "").strip().lower(),
        "last_data_file": data_file,
        "data_file": file_info,
        "last_backtest_data_file": settings.get("last_backtest_data_file") or "",
        "backtest_data_file": backtest_file_info,
        "last_strategy_file": strat_ctx["last_strategy_file"],
        "strategy_file": strat_ctx["strategy_file"],
        "debug_mode": settings.get("debug_mode", False),
        "ai_provider": settings.get("ai_provider", "deepseek"),
        "ai_model": settings.get("ai_model", ""),
        "ai_thinking": settings.get("ai_thinking", True),
        "ai_reasoning_effort": settings.get("ai_reasoning_effort", "high"),
        "has_ai_api_key": bool(settings.get("ai_api_key")) and (
            settings.get("ai_api_key_provider") == settings.get("ai_provider")
        ),
        "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
        "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
    }


@app.get("/api/ai/providers")
def api_ai_providers() -> dict[str, Any]:
    from web.ai_providers import provider_status

    status = provider_status()
    settings = load_settings()
    status["selected"] = settings.get("ai_provider", "deepseek")
    status["model"] = settings.get("ai_model", "")
    status["thinking"] = settings.get("ai_thinking", True)
    status["reasoning_effort"] = settings.get("ai_reasoning_effort", "high")
    status["has_api_key"] = bool(settings.get("ai_api_key")) and (
        settings.get("ai_api_key_provider") == settings.get("ai_provider")
    )
    return status


@app.post("/api/ai/analyze-training")
def api_ai_analyze_training(req: AnalyzeTrainingRequest):
    from fastapi.responses import StreamingResponse

    from web.ai_analyze import analyze_training_stream

    settings = load_settings()
    requested_provider = (
        req.provider or settings.get("ai_provider") or "deepseek"
    ).strip().lower()
    if req.api_key is not None:
        raw_key = str(req.api_key).strip()
    elif settings.get("ai_api_key_provider") == requested_provider:
        raw_key = str(settings.get("ai_api_key") or "").strip()
    else:
        raw_key = ""
    provider = requested_provider

    save_payload = {
        "ai_provider": provider,
        "ai_model": str(
            req.model if req.model is not None else settings.get("ai_model") or ""
        ).strip(),
        "ai_thinking": (
            req.thinking
            if req.thinking is not None
            else bool(settings.get("ai_thinking", True))
        ),
        "ai_reasoning_effort": str(
            req.reasoning_effort
            if req.reasoning_effort is not None
            else settings.get("ai_reasoning_effort") or "high"
        ).strip(),
    }
    if req.api_key is not None and str(req.api_key).strip():
        save_payload["ai_api_key"] = str(req.api_key).strip()
        save_payload["ai_api_key_provider"] = provider

    def event_gen():
        settings_saved = False
        try:
            for event in analyze_training_stream(
                provider=provider,
                api_key=str(raw_key).strip() or None,
                model=save_payload["ai_model"] or None,
                thinking=bool(save_payload["ai_thinking"]),
                reasoning_effort=save_payload["ai_reasoning_effort"],
                symbol=req.symbol,
            ):
                if not settings_saved and event.get("type") != "error":
                    save_settings(save_payload)
                    settings_saved = True
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file() -> dict[str, Any]:
    return _browse_data_file()


@app.post("/api/backtest/data-file/browse")
@app.get("/api/backtest/data-file/browse")
def api_browse_backtest_data_file() -> dict[str, Any]:
    return _browse_backtest_data_file()


@app.post("/api/data-file/register-legacy-mt5")
def api_register_legacy_mt5(req: RegisterLegacyMt5Request) -> dict[str, Any]:
    with _REGISTRATION_PLAN_LOCK:
        plan = _REGISTRATION_PLANS.get(req.plan_sha256)
    if plan is None:
        raise HTTPException(409, "注册计划已失效，请重新选择数据文件")
    planned = plan.get("files") or []
    if len(planned) != 1:
        raise HTTPException(409, "前端注册计划必须只包含一个文件")
    planned_path = (
        Path(str(plan["input_root"])) / str(planned[0]["relative_path"])
    ).resolve()
    if planned_path != Path(req.data_file).resolve():
        raise HTTPException(409, "注册计划与当前数据文件不匹配")
    report = apply_registration_plan(
        plan,
        expected_plan_sha256=req.plan_sha256,
        source_acknowledgement=req.source_acknowledgement,
    )
    report_dir = ROOT / "scratch" / "legacy_mt5_ui_reports"
    report_path = report_dir / f"{req.plan_sha256[:16]}.json"
    write_registration_report(report_path, report)
    if report["summary"]["failed"]:
        raise HTTPException(
            409,
            report["results"][0].get("message") or "旧 MT5 数据注册失败",
        )
    with _REGISTRATION_PLAN_LOCK:
        _REGISTRATION_PLANS.pop(req.plan_sha256, None)
    info = _inspect_or_http(req.data_file)
    save_settings({"last_data_file": info["data_file"]})
    return {
        "ok": True,
        "data_file": info,
        "registration_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
    }


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file() -> dict[str, Any]:
    return _browse_strategy_file()


@app.post("/api/strategy-file/sync-best")
@app.get("/api/strategy-file/sync-best")
def api_sync_best_strategy(symbol: str | None = None) -> dict[str, Any]:
    sym = _resolve_train_symbol(symbol)
    if not sym:
        raise HTTPException(400, "请先选择训练数据文件或指定品种")
    info = _sync_and_persist_best_strategy(sym)
    if not info:
        raise HTTPException(404, f"未找到 {sym} 的可用策略")
    return {"ok": True, **info}


def _progress_with_live_step(
    symbol: str,
    active: bool,
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    current_step = p.current_step
    train_steps = p.train_steps
    if active:
        run_id = str((job or {}).get("run_id") or "") or None
        live = training_manager.parse_step_from_log(
            refresh_remote=False,
            run_id=run_id,
        )
        current_step = live if live is not None else 0
        live_parameters = (job or {}).get("training_parameters") or {}
        live_train_steps = live_parameters.get("train_steps")
        if (
            isinstance(live_train_steps, int)
            and not isinstance(live_train_steps, bool)
            and live_train_steps > 0
        ):
            train_steps = live_train_steps
        else:
            train_steps = ModelConfig.TRAIN_STEPS
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    val_score = None
    hist = p.history or {}
    vals = hist.get("val_score") or []
    if vals:
        try:
            val_score = float(vals[-1])
        except (TypeError, ValueError):
            val_score = None
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "val_score": val_score,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
    }


def _attach_training_time(
    row: dict[str, Any] | None,
    *,
    symbol: str | None,
    job: dict[str, Any] | None,
    active: bool,
) -> dict[str, Any] | None:
    if not row or not symbol:
        return row
    summary = get_training_time_summary(symbol, job=job, active=active)
    row = dict(row)
    row["session_seconds"] = summary.session_seconds
    row["history_total_seconds"] = summary.history_total_seconds
    return row


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    training = training_manager.snapshot()
    job = training.get("job")
    pipeline = a_share_pipeline_manager.snapshot(
        (job or {}).get("run_id") if isinstance(job, dict) else None
    )
    active = bool(training.get("active"))

    if data_file:
        try:
            file_info = _decorate_data_info(inspect_parquet_file(data_file))
            sym = file_info.get("symbol")
            row = _progress_with_live_step(sym, active=False, job=job)
            progress = {
                "symbol": row["symbol"],
                "status": row["status"],
                "current_step": row["current_step"],
                "train_steps": row["train_steps"],
                "progress_pct": row["progress_pct"],
                "best_score": row["best_score"],
                "val_score": row.get("val_score"),
                "formula_decoded": row["formula_decoded"],
                "has_checkpoint": row.get("has_checkpoint", False),
                "has_strategy": row.get("has_strategy", False),
            }
            progress = _attach_training_time(
                progress, symbol=sym, job=job, active=active and job and job.get("symbol") == sym
            )
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    if job and job.get("symbol") and active:
        sym = job["symbol"]
        row = _progress_with_live_step(sym, active=True, job=job)
        progress = {
            "symbol": row["symbol"],
            "status": "running_job",
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "val_score": row.get("val_score"),
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
        }
        progress = _attach_training_time(progress, symbol=sym, job=job, active=True)

    return {
        "data_file": file_info,
        "progress": progress,
        "training": training,
        "pipeline": pipeline,
    }


@app.get("/api/symbols/{symbol}")
def api_symbol(symbol: str) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    return {
        "symbol": p.symbol,
        "status": p.status,
        "current_step": p.current_step,
        "train_steps": p.train_steps,
        "progress_pct": round(p.progress_pct, 1),
        "best_score": p.best_score,
        "best_formula": p.best_formula,
        "formula_decoded": p.formula_decoded,
        "has_strategy": p.has_strategy,
        "strategy_score": p.strategy_score,
        "checkpoint_path": p.checkpoint_path,
        "history": p.history,
    }


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list_strategies()}


@app.get("/api/strategies/{symbol}/export")
def api_export_strategy(symbol: str):
    import json

    from fastapi.responses import Response

    try:
        payload = get_strategy_for_export(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    progress = get_symbol_progress(symbol)
    step = progress.current_step
    score = payload.get("best_score")
    if score is None:
        score = progress.strategy_score if progress.strategy_score is not None else progress.best_score
    filename = build_strategy_export_filename(symbol, step, score)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.get("/api/training/{symbol}/export")
def api_export_training(symbol: str):
    from fastapi.responses import Response

    try:
        body, zip_name = build_training_export_zip(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.post("/api/training/import")
async def api_import_training() -> dict[str, Any]:
    raise HTTPException(403, "第一阶段已禁用外部 ZIP/PT 训练包导入")


@app.get("/api/training/status")
def api_training_status() -> dict[str, Any]:
    status = training_manager.snapshot()
    job = status.get("job") if isinstance(status, dict) else None
    run_id = (job or {}).get("run_id") if isinstance(job, dict) else None
    status["log_tail"] = training_manager.cached_log_tail(
        150,
        run_id=run_id,
    )
    status["pipeline"] = a_share_pipeline_manager.snapshot(run_id)
    return status


@app.get("/api/pipeline/status")
def api_pipeline_status() -> dict[str, Any]:
    training = training_manager.snapshot()
    run_id = (training.get("job") or {}).get("run_id")
    return {
        "pipeline": a_share_pipeline_manager.snapshot(run_id),
        "training_run_id": run_id,
    }


@app.get("/api/training/batches")
def api_training_batches() -> dict[str, Any]:
    return {
        "batches": [
            {
                "batch_id": batch.batch_id,
                "idempotency_key": batch.idempotency_key,
                "status": batch.status,
                "contract_sha256": batch.contract_sha256,
                "source_sha256": batch.source_sha256,
                "runtime_git_commit": batch.runtime_git_commit,
                "item_count": batch.item_count,
                "created_at": batch.created_at,
                "updated_at": batch.updated_at,
            }
            for batch in training_queue.list_batches()
        ],
        "active_item": training_batch_controller.snapshot()["active_item"],
    }


@app.get("/api/training/batches/{batch_id}")
def api_training_batch(batch_id: str) -> dict[str, Any]:
    snapshot = training_batch_controller.snapshot(batch_id)
    if snapshot["batch"] is None:
        raise HTTPException(404, "批训练不存在")
    return snapshot


@app.post("/api/training/batches")
def api_training_batch_create(
    req: CreateTrainingBatchRequest,
) -> dict[str, Any]:
    if not 1 <= len(req.items) <= 50:
        raise HTTPException(400, "批训练必须包含 1 至 50 个标的")
    specs: list[BatchItemSpec] = []
    for item in req.items:
        if re.fullmatch(
            r"run_\d{8}T\d{6}Z_[0-9a-f]{8}",
            item.planned_run_id,
        ) is None:
            raise HTTPException(400, "planned_run_id 格式非法")
        info = _inspect_or_http(item.data_file)
        if (
            info.get("source") != AKSHARE_HFQ_SOURCE_ID
            or source_family(str(info.get("source") or "")) != "ashare"
            or info.get("timeframe") != "D1"
            or info.get("dataset_purpose") != AKSHARE_SLICE_TRAINING
        ):
            raise HTTPException(
                400,
                "A50 批训练只接受已冻结的 AKShare 后复权 D1 训练切片",
            )
        specs.append(
            BatchItemSpec(
                symbol=str(info["symbol"]),
                timeframe=str(info["timeframe"]),
                data_file=str(info["data_file"]),
                data_sha256=str(info["data_sha256"]),
                planned_run_id=item.planned_run_id,
                train_steps=item.train_steps,
                cpus_per_task=item.cpus_per_task,
                memory=item.memory,
                time_limit=item.time_limit,
            )
        )
    try:
        result = training_queue.create_batch(
            idempotency_key=req.idempotency_key,
            contract_sha256=req.contract_sha256,
            source_sha256=training_manager.current_source_sha256(),
            runtime_git_commit=(
                req.runtime_git_commit
                or training_manager.current_git_commit()
            ),
            items=specs,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except DataHashDriftError as exc:
        raise HTTPException(409, str(exc)) from exc
    except QueueValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "ok": True,
        "created": result.created,
        **training_batch_controller.snapshot(result.batch.batch_id),
    }


@app.post("/api/training/start")
def api_training_start(req: StartTrainingRequest) -> dict[str, Any]:
    info = _inspect_or_http(req.data_file)
    if not bool((info.get("capabilities") or {}).get("training")):
        raise HTTPException(
            400,
            info.get("message")
            or "当前数据不满足所选训练后端的要求",
        )
    save_settings({"last_data_file": info["data_file"]})
    with training_batch_controller.admission_lock:
        if training_queue.has_pending_work():
            raise HTTPException(409, "批训练队列正在运行，禁止插入单标的任务")
        try:
            job = training_manager.start(
                data_file=info["data_file"],
                symbol=info["symbol"],
                timeframe=info["timeframe"],
                mode="ftmo",
                from_scratch=bool(req.from_scratch),
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
    if req.from_scratch:
        invalidate_checkpoint_cache()
    return {
        "ok": True,
        "job": _job_to_dict(job),
        "data_file": info,
        "from_scratch": bool(req.from_scratch),
        "pipeline": a_share_pipeline_manager.observe(
            {"active": True, "job": _job_to_dict(job)}
        ),
    }


@app.post("/api/training/stop")
def api_training_stop(req: StopTrainingRequest) -> dict[str, Any]:
    expected_run_id = req.run_id.strip()
    expected_job_id = req.slurm_job_id.strip()
    if not expected_run_id or not expected_job_id:
        raise HTTPException(422, "停止训练必须提供页面展示的 run_id 和 slurm_job_id")

    with training_batch_controller.admission_lock:
        training = training_manager.status()
        job = training.get("job") or {}
        current_run_id = str(job.get("run_id") or "")
        current_job_id = str(job.get("slurm_job_id") or "")
        if (
            current_run_id != expected_run_id
            or current_job_id != expected_job_id
        ):
            raise HTTPException(409, "页面展示的训练任务已变化，拒绝停止")
        symbol = job.get("symbol")
        data_file_hint = job.get("data_file")
        try:
            stopped = training_manager.stop(
                expected_run_id=expected_run_id,
                expected_job_id=expected_job_id,
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    strategy_file = None
    if symbol:
        _wait_training_idle()
        strategy_file = _sync_and_persist_best_strategy(
            symbol,
            data_file_hint=data_file_hint,
        )
    return {
        "ok": stopped,
        "training": training_manager.status(),
        "strategy_file": strategy_file,
    }


# ─────────────────────────────────────────────────────────────────────
# 回测 API
# ─────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "profit_loss_ratio",
    "n_trades", "win_rate", "avg_hold_bars",
)


def _load_backtest_report() -> dict[str, Any] | None:
    import json

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _backtest_focus_symbol(symbol: str | None = None) -> str | None:
    """Resolve the symbol used to filter backtest charts/report for the web UI."""
    if symbol:
        return symbol.strip() or None

    job = backtest_manager.status().get("job") or {}
    if job.get("symbol"):
        return str(job["symbol"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("symbol"):
        return str(strat["symbol"])

    report = _load_backtest_report()
    if report:
        keys = list((report.get("symbols") or {}).keys())
        if len(keys) == 1:
            return keys[0]
    return None


def _filter_report_for_symbol(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = report.get("symbols") or {}
    if symbol not in symbols:
        return report

    sym_data = symbols[symbol]
    return {
        **report,
        "focus_symbol": symbol,
        "symbols": {symbol: sym_data},
        "portfolio": {
            "total_return": sym_data.get("total_return"),
            "sharpe": sym_data.get("sharpe"),
            "sortino": sym_data.get("sortino"),
            "profit_loss_ratio": sym_data.get("profit_loss_ratio"),
            "n_trades": sym_data.get("n_trades"),
            "win_rate": sym_data.get("win_rate"),
        },
    }


def _list_backtest_charts(symbol: str | None = None) -> list[dict[str, str]]:
    """列出回测输出目录下的图表；单品种模式只返回该品种相关文件。"""
    if not BACKTEST_OUTPUT_DIR.exists():
        return []

    if symbol:
        charts: list[dict[str, str]] = []
        equity = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
        if equity.exists():
            charts.append(
                {"name": equity.name, "label": f"{symbol} 资金曲线", "kind": "equity"}
            )
        return charts

    charts = []
    portfolio = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
    if portfolio.exists():
        charts.append({"name": "portfolio_equity.png", "label": "组合资金曲线", "kind": "portfolio"})
    for path in sorted(BACKTEST_OUTPUT_DIR.glob("equity_*.png")):
        sym = path.stem.replace("equity_", "", 1)
        charts.append({"name": path.name, "label": f"{sym} 资金曲线", "kind": "symbol"})
    return charts


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    status = backtest_manager.status()
    status["log_tail"] = backtest_manager.tail_log(200)
    return status


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    info = _inspect_strategy_or_http(req.strategy_file)
    settings = load_settings()
    commission = (
        float(req.commission_pct)
        if req.commission_pct is not None
        else float(settings.get("bt_commission_pct", 0.02))
    )
    slippage = (
        float(req.slippage_pct)
        if req.slippage_pct is not None
        else float(settings.get("bt_slippage_pct", 0.01))
    )
    if commission < 0 or slippage < 0:
        raise HTTPException(400, "手续费和滑点不能为负数")

    save_settings({
        "last_strategy_file": info["strategy_file"],
        "bt_commission_pct": commission,
        "bt_slippage_pct": slippage,
    })

    data_file: str | None = None
    data_info: dict[str, Any] | None = None
    data_resolution = ""
    candidates = (
        ("explicit_evaluation", str(req.data_file or "").strip(), True),
        (
            "backtest_selection",
            str(settings.get("last_backtest_data_file") or "").strip(),
            False,
        ),
        ("strategy_recorded", str(info.get("data_file") or "").strip(), False),
        ("training_selection", str(settings.get("last_data_file") or "").strip(), False),
    )
    for resolution, candidate, strict in candidates:
        if not candidate:
            continue
        try:
            inspected = _decorate_data_info(inspect_parquet_file(candidate))
            if inspected.get("symbol") != info.get("symbol"):
                raise ValueError(
                    f"品种不一致: {inspected.get('symbol')} != {info.get('symbol')}"
                )
            data_file = inspected["data_file"]
            data_info = inspected
            data_resolution = resolution
            break
        except Exception as exc:
            if strict:
                raise HTTPException(
                    400,
                    f"所选回测数据无法加载: {candidate}\n{exc}",
                ) from exc

    if not data_file:
        raise HTTPException(
            400,
            "没有可用的回测数据。请在「策略回测」页选择测试 Parquet。",
        )

    try:
        manager = ParquetDataManager(data_file)
        manager.load()
        from run_backtest import _validate_strategy_data_contract

        evaluation = _validate_strategy_data_contract(
            info,
            manager,
            evaluation_mode=req.evaluation_mode,
            score_start=req.score_start,
        )
    except ValueError as exc:
        raise HTTPException(400, f"策略与回测数据不兼容: {exc}") from exc

    save_settings({"last_backtest_data_file": data_file})

    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
            evaluation_mode=req.evaluation_mode,
            score_start=req.score_start,
            commission_pct=commission,
            slippage_pct=slippage,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {
        "ok": True,
        "job": job.to_dict(),
        "strategy_file": info,
        "data_file": data_info,
        "data_resolution": data_resolution,
        "evaluation": evaluation,
    }


@app.post("/api/backtest/stop")
def api_backtest_stop() -> dict[str, Any]:
    stopped = backtest_manager.stop()
    return {"ok": stopped, "backtest": backtest_manager.status()}


@app.get("/api/backtest/report")
def api_backtest_report(symbol: str | None = None) -> dict[str, Any]:
    report = _load_backtest_report()
    focus = _backtest_focus_symbol(symbol)
    if report and focus:
        report = _filter_report_for_symbol(report, focus)
    return {
        "available": report is not None,
        "report": report,
        "charts": _list_backtest_charts(focus),
        "focus_symbol": focus,
    }


@app.get("/api/backtest/equity")
def api_backtest_equity(symbol: str | None = None) -> dict[str, Any]:
    """资金曲线原始数据（供前端渲染交互式 HTML 图表）。"""
    import json

    path = BACKTEST_OUTPUT_DIR / "equity_curve.json"
    focus = _backtest_focus_symbol(symbol)
    if not path.exists():
        return {"available": False, "focus_symbol": focus, "data": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "focus_symbol": focus, "data": None}

    # 单品种模式：只保留聚焦品种，去掉无关序列
    if focus and isinstance(data.get("symbols"), dict) and focus in data["symbols"]:
        data = {
            **data,
            "symbols": {focus: data["symbols"][focus]},
        }
        data.pop("portfolio", None)

    return {"available": True, "focus_symbol": focus, "data": data}


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    # 防止路径穿越：仅允许输出目录内的 png 文件
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(400, "非法文件名")
    path = (BACKTEST_OUTPUT_DIR / name).resolve()
    try:
        path.relative_to(BACKTEST_OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "非法路径") from None
    if not path.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────
# 实时行情分析 API
# ─────────────────────────────────────────────────────────────────────


@app.on_event("startup")
def _startup_realtime() -> None:
    global _PIPELINE_MONITOR_THREAD
    try:
        realtime_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("realtime load_persisted failed", exc)
    if _PIPELINE_MONITOR_THREAD is None or not _PIPELINE_MONITOR_THREAD.is_alive():
        _PIPELINE_MONITOR_STOP.clear()
        _PIPELINE_MONITOR_THREAD = threading.Thread(
            target=_pipeline_monitor_loop,
            name="alphamaster-pipeline-monitor",
            daemon=True,
        )
        _PIPELINE_MONITOR_THREAD.start()


def _pipeline_monitor_loop() -> None:
    """即使没有打开浏览器，也持续推进当前 Slurm run 的后处理。"""
    while not _PIPELINE_MONITOR_STOP.is_set():
        try:
            _advance_a_share_pipeline_once()
        except Exception as exc:  # noqa: BLE001
            log_error("a-share pipeline monitor failed", exc)
        _PIPELINE_MONITOR_STOP.wait(_PIPELINE_MONITOR_INTERVAL_SECONDS)


def _refresh_training_log_once(training: dict[str, Any]) -> None:
    job = training.get("job") if isinstance(training, dict) else None
    if not isinstance(job, dict):
        return
    run_id = str(job.get("run_id") or "")
    job_id = str(job.get("slurm_job_id") or "")
    if not run_id or not job_id:
        return
    remote_state = str(job.get("remote_state") or "").upper()
    if bool(training.get("active")) and remote_state == "RUNNING":
        training_manager.tail_log(
            200,
            expected_run_id=run_id,
            expected_job_id=job_id,
        )
    elif (
        remote_state
        in {
            "COMPLETED",
            "DOWNLOADING",
            "READY",
            "FAILED",
            "CANCELLED",
        }
        and not job.get("final_log_refreshed_at")
    ):
        training_manager.tail_log(
            200,
            expected_run_id=run_id,
            expected_job_id=job_id,
            final=True,
        )


def _schedule_training_log_refresh(training: dict[str, Any]) -> None:
    global _TRAINING_LOG_REFRESH_THREAD
    with _TRAINING_LOG_REFRESH_LOCK:
        thread = _TRAINING_LOG_REFRESH_THREAD
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=_refresh_training_log_once,
            args=(training,),
            name="alphamaster-training-log-refresh",
            daemon=True,
        )
        _TRAINING_LOG_REFRESH_THREAD = thread
        thread.start()


def _advance_a_share_pipeline_once() -> dict[str, Any] | None:
    training = training_manager.status()
    pipeline = a_share_pipeline_manager.observe(training)
    training_batch_controller.advance_once(
        training=training,
        pipeline=pipeline,
    )
    _schedule_training_log_refresh(training)
    return pipeline


@app.on_event("shutdown")
def _shutdown_pipeline_monitor() -> None:
    global _PIPELINE_MONITOR_THREAD
    _PIPELINE_MONITOR_STOP.set()
    thread = _PIPELINE_MONITOR_THREAD
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    _PIPELINE_MONITOR_THREAD = None


@app.get("/api/realtime/sources")
def api_realtime_sources() -> dict[str, Any]:
    return {"sources": list_sources(), "min_exposure": min_exposure()}


@app.post("/api/realtime/tradingview/probe")
def api_realtime_tradingview_probe() -> dict[str, Any]:
    """Probe TradingView reachability (same behavior as PA_Agent before fetch)."""
    from web.data_sources.tradingview_connectivity import (
        TV_CLOUD_SERVER_WIKI_URL,
        TV_CONNECTIVITY_MESSAGE,
        check_tradingview_connectivity,
    )

    ok, detail = check_tradingview_connectivity(
        timeout_s=15.0, max_attempts=2, retry_delay_s=2.0
    )
    return {
        "ok": ok,
        "detail": detail,
        "blocked": not ok,
        "title": "无法使用 TradingView",
        "message": None if ok else TV_CONNECTIVITY_MESSAGE,
        "wiki_url": TV_CLOUD_SERVER_WIKI_URL,
    }


@app.get("/api/realtime/strategies")
def api_realtime_strategies() -> dict[str, Any]:
    """已保存的 best_*.json 策略，供因子来源下拉。"""
    rows = []
    for s in list_strategies():
        sym = s.get("symbol")
        if not sym:
            continue
        path = strategy_path_for_symbol(sym)
        if not path.exists():
            continue
        rows.append(
            {
                "symbol": sym,
                "timeframe": s.get("timeframe"),
                "best_score": s.get("best_score"),
                "formula_decoded": s.get("formula_decoded"),
                "strategy_file": str(path.resolve()),
            }
        )
    return {"strategies": rows}


@app.get("/api/realtime/status")
def api_realtime_status() -> dict[str, Any]:
    return realtime_manager.status()


@app.get("/api/realtime/signals")
def api_realtime_signals(
    limit: int = 100,
    watch_id: str | None = None,
) -> dict[str, Any]:
    return {
        "signals": realtime_manager.signal_events(
            limit=limit,
            watch_id=watch_id,
        )
    }


@app.post("/api/realtime/watch")
def api_realtime_watch(req: AddWatchRequest) -> dict[str, Any]:
    try:
        watch = realtime_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "watch": watch}


@app.post("/api/realtime/unwatch")
def api_realtime_unwatch(req: RemoveWatchRequest) -> dict[str, Any]:
    removed = realtime_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/realtime/start")
def api_realtime_start() -> dict[str, Any]:
    realtime_manager.start()
    return {"ok": True, **realtime_manager.status()}


@app.post("/api/realtime/stop")
def api_realtime_stop() -> dict[str, Any]:
    realtime_manager.stop()
    return {"ok": True, "running": False}


@app.get("/api/realtime/feishu")
def api_realtime_feishu_get() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("feishu_enabled")),
        "webhook_configured": bool(s.get("feishu_webhook_url")),
        "secret_configured": bool(s.get("feishu_secret")),
    }


@app.put("/api/realtime/feishu")
def api_realtime_feishu_put(req: FeishuSettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.enabled is not None:
        payload["feishu_enabled"] = bool(req.enabled)
    if req.webhook_url is not None:
        webhook_url = req.webhook_url.strip()
        if webhook_url:
            try:
                webhook_url = validate_feishu_webhook_url(webhook_url)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        payload["feishu_webhook_url"] = webhook_url
    if req.secret is not None:
        payload["feishu_secret"] = req.secret
    saved = save_settings(payload)
    return {
        "ok": True,
        "enabled": bool(saved.get("feishu_enabled")),
        "webhook_configured": bool(saved.get("feishu_webhook_url")),
        "secret_configured": bool(saved.get("feishu_secret")),
    }


@app.post("/api/realtime/feishu/test")
def api_realtime_feishu_test(req: FeishuTestRequest | None = None) -> dict[str, Any]:
    request = req or FeishuTestRequest()
    ok, detail = send_feishu_text(
        "【AlphaMaster 飞书连通测试】\n连接成功，后续交易动作信号将发送到本群。",
        webhook_url=request.webhook_url,
        secret=request.secret,
    )
    if not ok:
        raise HTTPException(502, f"飞书测试失败：{detail}")
    return {"ok": True, "message": "飞书测试消息已发送"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
