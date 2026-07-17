"""Read training progress from checkpoints and strategy files."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import torch

from config import Config
from data_pipeline.a_share_data import ASHARE_SPECS_BY_TIMEFRAME
from data_pipeline.dataset_contracts import TRAINING_SOURCE_IDS
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB, VocabVersionMismatchError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
PUBLISHED_BUNDLE_FORMAT = "alphamaster_published_bundle_v1"


def _safe_symbol_tag(symbol: str) -> str:
    return symbol.replace(".", "_")


def _uses_slurm_backend() -> bool:
    return os.environ.get("TRAINING_BACKEND", "").strip().lower() == "slurm"


def published_pointer_path(symbol: str) -> Path:
    return PROJECT_ROOT / "published_training" / f"current_{_safe_symbol_tag(symbol)}.json"


def _published_artifact_path(artifact_root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("发布指针含非法产物路径")
    resolved = artifact_root.joinpath(*posix.parts).resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError("发布指针产物越出运行目录")
    return resolved


def _filesystem_path(path: Path) -> Path:
    """Windows 下用扩展路径执行文件 I/O，避免嵌套 checkpoint 撞 MAX_PATH。"""
    if os.name != "nt":
        return path
    raw = os.path.normpath(str(path if path.is_absolute() else Path.cwd() / path))
    if raw.startswith("\\\\?\\"):
        return Path(raw)
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


def _read_text(path: Path) -> str:
    with _filesystem_path(path).open("r", encoding="utf-8") as handle:
        return handle.read()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _filesystem_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_published_bundle(symbol: str) -> dict[str, Any] | None:
    """读取 Slurm 原子发布指针；任何字段不完整时都拒绝该整套产物。"""
    if not _uses_slurm_backend():
        return None
    pointer = published_pointer_path(symbol)
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("format") != PUBLISHED_BUNDLE_FORMAT
            or payload.get("symbol") != symbol
            or not re.fullmatch(r"run_[A-Za-z0-9_-]+", str(payload.get("run_id") or ""))
        ):
            return None
        artifact_root = Path(str(payload["artifact_root"])).resolve()
        checkpoint_rel = payload.get("checkpoint_files")
        strategy_rel = payload.get("strategy_file")
        history_rel = payload.get("history_file")
        hashes = payload.get("artifact_sha256")
        timeframe = payload.get("timeframe")
        data_sha256 = payload.get("data_sha256")
        dataset_id = payload.get("dataset_id")
        local_source = payload.get("local_source")
        periods_per_year = payload.get("periods_per_year")
        minimum_bars = payload.get("minimum_bars")
        if (
            not isinstance(checkpoint_rel, list)
            or not checkpoint_rel
            or not all(isinstance(item, str) for item in checkpoint_rel)
            or not isinstance(strategy_rel, str)
            or not isinstance(history_rel, str)
            or not isinstance(hashes, dict)
            or not isinstance(timeframe, str)
            or re.fullmatch(r"[A-Z0-9]+", timeframe) is None
            or not isinstance(data_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", data_sha256) is None
            or dataset_id != f"sha256:{data_sha256}"
            or local_source not in TRAINING_SOURCE_IDS
            or isinstance(periods_per_year, bool)
            or not isinstance(periods_per_year, int)
            or periods_per_year <= 0
            or isinstance(minimum_bars, bool)
            or not isinstance(minimum_bars, int)
            or minimum_bars <= 0
            or (
                local_source in TRAINING_SOURCE_IDS - {"ashare_local"}
                and minimum_bars != Config.MIN_BARS
            )
            or (
                local_source == "ashare_local"
                and (
                    timeframe not in ASHARE_SPECS_BY_TIMEFRAME
                    or periods_per_year
                    != ASHARE_SPECS_BY_TIMEFRAME[timeframe].periods_per_year
                    or minimum_bars
                    != ASHARE_SPECS_BY_TIMEFRAME[timeframe].minimum_bars
                )
            )
            or any(
                not re.fullmatch(
                    rf"checkpoints/{re.escape(timeframe)}/{re.escape(data_sha256)}/"
                    rf"run_[0-9]{{20}}/ckpt_{re.escape(symbol)}_step_[0-9]{{4,}}\.pt",
                    item,
                )
                for item in checkpoint_rel
            )
            or strategy_rel != f"strategies/best_{symbol}.json"
            or history_rel != f"training_history_{symbol}.json"
        ):
            return None
        checkpoints = [_published_artifact_path(artifact_root, item) for item in checkpoint_rel]
        strategy = _published_artifact_path(artifact_root, strategy_rel)
        history = _published_artifact_path(artifact_root, history_rel)
        required = [*checkpoint_rel, strategy_rel, history_rel]
        artifact_paths = [*checkpoints, strategy, history]
        if (
            any(not _filesystem_path(path).is_file() for path in artifact_paths)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(hashes.get(item) or "")) for item in required)
            or any(
                _sha256_file(path) != hashes[relative]
                for relative, path in zip(required, artifact_paths)
            )
        ):
            return None
        strategy_payload = json.loads(_read_text(strategy))
        if not isinstance(strategy_payload, dict) or any(
            strategy_payload.get(field) != payload.get(field)
            for field in (
                "symbol",
                "timeframe",
                "dataset_id",
                "data_sha256",
                "local_source",
                "periods_per_year",
                "minimum_bars",
            )
        ):
            return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return {
        **payload,
        "artifact_root_path": artifact_root,
        "checkpoint_paths": sorted(checkpoints, key=_step_from_name),
        "strategy_path": strategy,
        "history_path": history,
    }


def checkpoint_glob(symbol: str) -> list[Path]:
    if _uses_slurm_backend():
        bundle = get_published_bundle(symbol)
        return list(bundle["checkpoint_paths"]) if bundle else []
    tag = _safe_symbol_tag(symbol)
    patterns = [
        f"ckpt_{symbol}_step_*.pt",
        f"ckpt_{tag}_step_*.pt",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(CHECKPOINT_DIR.glob(pattern))
        found.extend(CHECKPOINT_DIR.glob(f"*/*/run_*/{pattern}"))
    return sorted(set(found), key=lambda p: _filesystem_path(p).stat().st_mtime)


def _step_from_name(path: Path) -> int:
    m = re.search(r"_step_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 0


@dataclass
class SymbolProgress:
    symbol: str
    train_steps: int
    current_step: int
    best_score: float | None
    best_formula: list[int] | None
    formula_decoded: str | None
    has_strategy: bool
    strategy_score: float | None
    checkpoint_path: str | None
    checkpoint_mtime: float | None
    history: dict[str, Any] | None

    @property
    def progress_pct(self) -> float:
        if self.train_steps <= 0:
            return 0.0
        return min(100.0, 100.0 * self.current_step / self.train_steps)

    @property
    def status(self) -> str:
        if self.current_step >= self.train_steps and self.has_strategy:
            return "completed"
        if self.current_step > 0:
            return "in_progress"
        if self.has_strategy:
            return "strategy_only"
        return "idle"


_ckpt_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_checkpoint_cache() -> None:
    _ckpt_cache.clear()


def _load_checkpoint_meta(path: Path) -> dict[str, Any]:
    filesystem_path = _filesystem_path(path)
    mtime = filesystem_path.stat().st_mtime
    key = str(path)
    cached = _ckpt_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    with filesystem_path.open("rb") as handle:
        ckpt = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint {path} 顶层不是对象")
    FORMULA_VOCAB.verify(ckpt.get("vocab_version"))
    meta = {
        "step": int(ckpt.get("step", _step_from_name(path))),
        "best_score": ckpt.get("best_score"),
        "best_formula": ckpt.get("best_formula"),
        "training_history": ckpt.get("training_history") or {},
    }
    _ckpt_cache[key] = (mtime, meta)
    return meta


def _decode_formula(tokens: list[int] | None) -> str | None:
    if not tokens:
        return None
    names = FORMULA_VOCAB.token_names
    try:
        return " → ".join(names[t] for t in tokens)
    except (IndexError, TypeError):
        return str(tokens)


def _load_strategy(symbol: str) -> dict[str, Any] | None:
    bundle = get_published_bundle(symbol) if _uses_slurm_backend() else None
    if _uses_slurm_backend() and not bundle:
        return None
    path = bundle["strategy_path"] if bundle else STRATEGIES_DIR / f"best_{symbol}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(_read_text(path))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        FORMULA_VOCAB.verify(payload.get("vocab_version"))
    except VocabVersionMismatchError:
        return None
    if bundle:
        payload = dict(payload)
        payload["run_id"] = bundle["run_id"]
        payload["dataset_id"] = bundle.get("dataset_id")
        payload["local_source"] = bundle.get("local_source")
        payload["periods_per_year"] = bundle.get("periods_per_year")
        payload["minimum_bars"] = bundle.get("minimum_bars")
        payload["data_sha256"] = bundle.get("data_sha256")
        payload["data_filename"] = bundle.get("data_filename")
        payload["data_file"] = bundle.get("data_file")
    return payload


def _pick_training_history(
    file_history: dict[str, Any] | None,
    ckpt_history: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """取步数更多的那份历史，避免旧 checkpoint 覆盖较新的 json 曲线。"""
    if not file_history and not ckpt_history:
        return None
    if not file_history:
        return ckpt_history
    if not ckpt_history:
        return file_history
    file_n = len(file_history.get("step") or [])
    ckpt_n = len(ckpt_history.get("step") or [])
    return file_history if file_n >= ckpt_n else ckpt_history


def get_symbol_progress(symbol: str) -> SymbolProgress:
    train_steps = ModelConfig.TRAIN_STEPS
    strategy = _load_strategy(symbol)
    ckpts = checkpoint_glob(symbol)

    current_step = 0
    best_score = None
    best_formula = None
    history: dict[str, Any] | None = None
    ckpt_path: str | None = None
    ckpt_mtime: float | None = None

    bundle = get_published_bundle(symbol) if _uses_slurm_backend() else None
    if bundle and strategy:
        published_train_steps = strategy.get("train_steps")
        if (
            isinstance(published_train_steps, int)
            and not isinstance(published_train_steps, bool)
            and published_train_steps > 0
        ):
            train_steps = published_train_steps
    if _uses_slurm_backend():
        hist_file = (
            bundle["history_path"]
            if bundle
            else PROJECT_ROOT / "published_training" / "__missing_history__"
        )
    else:
        hist_file = PROJECT_ROOT / f"training_history_{symbol}.json"
    file_history: dict[str, Any] | None = None
    if hist_file.exists():
        try:
            file_history = json.loads(_read_text(hist_file))
            steps = file_history.get("step") or []
            if steps:
                # history 存的是 0 起算的训练步索引，展示与日志 [N/5000] 对齐用 N
                current_step = max(current_step, int(steps[-1]) + 1)
            bests = file_history.get("best_score") or []
            if bests:
                best_score = float(bests[-1])
            history = file_history
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    if ckpts:
        latest = ckpts[-1]
        try:
            ckpt_path = str(latest.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            ckpt_path = str(latest)
        ckpt_mtime = _filesystem_path(latest).stat().st_mtime
        try:
            meta = _load_checkpoint_meta(latest)
            current_step = max(current_step, int(meta["step"]))
            if meta.get("best_score") is not None:
                best_score = float(meta["best_score"])
            best_formula = meta.get("best_formula")
            history = _pick_training_history(file_history, meta.get("training_history"))
        except Exception:
            current_step = max(current_step, _step_from_name(latest))

    if strategy:
        if best_score is None and strategy.get("best_score") is not None:
            best_score = float(strategy["best_score"])
        if best_formula is None and strategy.get("formula"):
            best_formula = strategy["formula"]

    return SymbolProgress(
        symbol=symbol,
        train_steps=train_steps,
        current_step=current_step,
        best_score=best_score,
        best_formula=best_formula,
        formula_decoded=_decode_formula(best_formula),
        has_strategy=strategy is not None,
        strategy_score=float(strategy["best_score"]) if strategy and strategy.get("best_score") is not None else None,
        checkpoint_path=ckpt_path,
        checkpoint_mtime=ckpt_mtime,
        history=history,
    )


def get_strategy_for_export(symbol: str) -> dict[str, Any]:
    data = _load_strategy(symbol)
    if not data:
        raise FileNotFoundError(f"未找到 {symbol} 的策略，请先完成训练")
    out = dict(data)
    formula = out.get("formula")
    if formula and not out.get("formula_decoded"):
        out["formula_decoded"] = _decode_formula(formula)
    return out


def build_strategy_export_filename(
    symbol: str,
    step: int,
    score: float | None,
) -> str:
    """e.g. strategy_ADAUSD_step0084_score2.4021.json"""
    safe = symbol.replace(".", "_")
    step_part = f"step{max(0, int(step)):04d}"
    if score is not None:
        return f"strategy_{safe}_{step_part}_score{float(score):.4f}.json"
    return f"strategy_{safe}_{step_part}.json"


def list_strategies() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if _uses_slurm_backend():
        pointer_dir = PROJECT_ROOT / "published_training"
        if not pointer_dir.exists():
            return rows
        for pointer in sorted(pointer_dir.glob("current_*.json")):
            try:
                symbol = str(json.loads(pointer.read_text(encoding="utf-8"))["symbol"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            data = _load_strategy(symbol)
            if not data:
                continue
            formula = data.get("formula")
            rows.append({
                "file": f"best_{symbol}.json",
                "symbol": symbol,
                "timeframe": data.get("timeframe"),
                "best_score": data.get("best_score"),
                "formula_decoded": data.get("formula_decoded") or _decode_formula(formula),
                "train_steps": data.get("train_steps"),
                "mode": data.get("mode"),
            })
        return rows
    if not STRATEGIES_DIR.exists():
        return rows
    for path in sorted(STRATEGIES_DIR.glob("best_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        formula = data.get("formula")
        rows.append({
            "file": path.name,
            "symbol": data.get("symbol") or path.stem.replace("best_", "", 1),
            "timeframe": data.get("timeframe"),
            "best_score": data.get("best_score"),
            "formula_decoded": data.get("formula_decoded") or _decode_formula(formula),
            "train_steps": data.get("train_steps"),
            "mode": data.get("mode"),
        })
    return rows
