"""Export / import training checkpoints as portable zip packages."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from data_pipeline.dataset_contracts import DATA_SOURCE_IDS
from model_core.alphagpt import AlphaGPT
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import FORMULA_VOCAB
from web.progress import (
    PROJECT_ROOT,
    _decode_formula,
    _load_strategy,
    checkpoint_glob,
    get_published_bundle,
    invalidate_checkpoint_cache,
)

_PACKAGE_FORMAT = "alphamaster_training_v2"
_LEGACY_PACKAGE_FORMAT = "alphamaster_training_v1"
_CKPT_NAME_RE = re.compile(r"^ckpt_(.+)_step_(\d+)\.pt$", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TIMEFRAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,15}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9_-]+$")
_LOCAL_SOURCES = DATA_SOURCE_IDS
_CHECKPOINT_IDENTITY_FIELDS = (
    "symbol",
    "timeframe",
    "dataset_id",
    "data_sha256",
    "local_source",
    "periods_per_year",
    "minimum_bars",
)

# 当前检查点约 7 MB。上限既给模型增长留足余量，也避免压缩炸弹在校验阶段耗尽内存/磁盘。
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MEMBERS = 4
_IMPORT_LOCK = threading.Lock()


def _history_path(symbol: str) -> Path:
    bundle = get_published_bundle(symbol)
    if bundle:
        return bundle["history_path"]
    return PROJECT_ROOT / f"training_history_{symbol}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_checkpoint_filesystem_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_file_bytes(path: Path) -> bytes:
    with open(_checkpoint_filesystem_path(path), "rb") as handle:
        return handle.read()


def _verify_published_bundle(bundle: dict[str, Any]) -> None:
    hashes = bundle["artifact_sha256"]
    pairs = [
        *zip(bundle["checkpoint_files"], bundle["checkpoint_paths"]),
        (bundle["strategy_file"], bundle["strategy_path"]),
        (bundle["history_file"], bundle["history_path"]),
    ]
    for relative, path in pairs:
        if _sha256_file(path) != hashes[relative]:
            raise ValueError(f"发布训练包完整性校验失败: {relative}")


def _symbol_from_ckpt_name(name: str) -> str | None:
    m = _CKPT_NAME_RE.match(Path(name).name)
    return m.group(1) if m else None


def _checkpoint_filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def _validate_checkpoint_resume_state(ckpt: dict[str, Any], name: str) -> None:
    model_state = ckpt.get("model_state_dict")
    optimizer_state = ckpt.get("optimizer_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"检查点 {name} 缺少可加载的 model_state_dict")
    if (
        not isinstance(optimizer_state, dict)
        or not isinstance(optimizer_state.get("state"), dict)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state["param_groups"]
    ):
        raise ValueError(f"检查点 {name} 缺少可加载的 optimizer_state_dict")

    model = AlphaGPT()
    expected_state = model.state_dict()
    if set(model_state) != set(expected_state):
        raise ValueError(f"检查点 {name} 的模型参数键与当前 AlphaGPT 不一致")
    for key, expected in expected_state.items():
        actual = model_state[key]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            raise ValueError(f"检查点 {name} 的模型参数 {key} 形状或 dtype 不一致")
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(optimizer_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"检查点 {name} 无法加载到当前模型/优化器") from exc

    for parameter, state_payload in optimizer.state.items():
        if not isinstance(state_payload, dict):
            raise ValueError(f"检查点 {name} 的 optimizer state 非法")
        for state_key, value in state_payload.items():
            if (
                isinstance(value, torch.Tensor)
                and state_key != "step"
                and value.shape != parameter.shape
            ):
                raise ValueError(
                    f"检查点 {name} 的 optimizer state {state_key} 形状不一致"
                )

    best_score = ckpt.get("best_score")
    best_formula = ckpt.get("best_formula")
    if (
        isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise ValueError(f"检查点 {name} 的 best_score 非法")
    if (
        not isinstance(best_formula, list)
        or not best_formula
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= FORMULA_VOCAB.size
            for token in best_formula
        )
    ):
        raise ValueError(f"检查点 {name} 的 best_formula 非法")
    training_history = ckpt.get("training_history")
    if (
        not isinstance(training_history, dict)
        or any(
            not isinstance(key, str) or not isinstance(values, list)
            for key, values in training_history.items()
        )
    ):
        raise ValueError(f"检查点 {name} 的 training_history 非法")

    best_snapshot = ckpt.get("best_snapshot")
    if best_snapshot is not None and not isinstance(best_snapshot, dict):
        raise ValueError(f"检查点 {name} 的 best_snapshot 非法")
    factor_pool = ckpt.get("factor_pool", [])
    elite_pool = ckpt.get("elite_pool", [])
    if not isinstance(factor_pool, list) or not isinstance(elite_pool, list):
        raise ValueError(f"检查点 {name} 的 factor_pool/elite_pool 非法")
    for field in ("factor_pool_counter", "elite_counter", "restart_count"):
        value = ckpt.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"检查点 {name} 的 {field} 非法")


def _validate_checkpoint_file(
    path: Path,
    *,
    expected_symbol: str | None = None,
    expected_step: int | None = None,
    require_data_identity: bool = False,
) -> dict[str, Any]:
    with open(_checkpoint_filesystem_path(path), "rb") as handle:
        ckpt = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"检查点 {path.name} 不是合法字典")
    if ckpt.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise ValueError(f"检查点 {path.name} 的评分合同不兼容")
    artifact_version = ckpt.get("vocab_version")
    if artifact_version is None:
        raise ValueError(
            f"检查点 {path.name} 过旧（无 vocab_version），"
            f"当前词表 {FORMULA_VOCAB.version!r}，请重新训练"
        )
    FORMULA_VOCAB.verify(artifact_version)
    raw_step = ckpt.get("step")
    if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step <= 0:
        raise ValueError(f"检查点 {path.name} 的 step 非法")
    step = raw_step
    symbol = ckpt.get("symbol")
    if symbol is not None and (not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol)):
        raise ValueError(f"检查点 {path.name} 的 symbol 非法")
    if expected_symbol is not None and symbol is not None and symbol != expected_symbol:
        raise ValueError(f"检查点 {path.name} 的品种与 manifest 不一致")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"检查点 {path.name} 的 step 与 manifest 不一致")
    metadata = {
        "step": step,
        "symbol": symbol,
        "best_score": ckpt.get("best_score"),
        "best_formula": ckpt.get("best_formula"),
        "timeframe": ckpt.get("timeframe"),
        "dataset_id": ckpt.get("dataset_id"),
        "data_sha256": ckpt.get("data_sha256"),
        "local_source": ckpt.get("local_source"),
        "periods_per_year": ckpt.get("periods_per_year"),
        "minimum_bars": ckpt.get("minimum_bars"),
        "scoring_contract_version": ckpt.get("scoring_contract_version"),
    }
    if require_data_identity:
        missing = [field for field in _CHECKPOINT_IDENTITY_FIELDS if field not in ckpt]
        if missing:
            raise ValueError(
                f"检查点 {path.name} 缺少完整数据身份: {', '.join(missing)}"
            )
        if (
            metadata["symbol"] != expected_symbol
            or not isinstance(metadata["timeframe"], str)
            or re.fullmatch(r"[A-Z0-9]+", metadata["timeframe"]) is None
            or metadata["local_source"] not in _LOCAL_SOURCES
        ):
            raise ValueError(f"检查点 {path.name} 的 timeframe/local_source 非法")
        digest = metadata["data_sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"检查点 {path.name} 的 data_sha256 非法")
        if metadata["dataset_id"] != f"sha256:{digest}":
            raise ValueError(f"检查点 {path.name} 的 dataset_id 非法")
        for key in ("periods_per_year", "minimum_bars"):
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"检查点 {path.name} 的 {key} 非法")
        _validate_checkpoint_resume_state(ckpt, path.name)
    return metadata


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _artifact_row(relative: str, content: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "size": len(content),
        "sha256": _sha256_bytes(content),
    }


def _checkpoint_relative_for_export(
    checkpoint: Path,
    bundle: dict[str, Any] | None,
) -> str:
    if bundle:
        for relative, path in zip(bundle["checkpoint_files"], bundle["checkpoint_paths"]):
            if path.resolve() == checkpoint.resolve():
                return relative
    try:
        relative = checkpoint.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        if relative.startswith("checkpoints/"):
            return relative
    except ValueError:
        pass
    return f"checkpoints/{checkpoint.name}"


def _validate_checkpoint_path_contract(
    relative: str,
    symbol: str,
    step: int,
    manifest: dict[str, Any],
) -> None:
    nested_match = re.fullmatch(
        rf"checkpoints/([^/]+)/([0-9a-f]{{64}})/(run_[0-9]{{20}})/"
        rf"ckpt_{re.escape(symbol)}_step_(\d+)\.pt",
        relative,
    )
    if not nested_match or int(nested_match.group(4)) != step:
        raise ValueError(
            "v2 训练包只接受按 timeframe+data_sha256+checkpoint_run_id "
            "隔离的 checkpoint 路径"
        )
    timeframe, data_sha256, run_id = nested_match.group(1, 2, 3)
    if (
        not _TIMEFRAME_RE.fullmatch(timeframe)
        or manifest.get("timeframe") != timeframe
        or manifest.get("data_sha256") != data_sha256
        or manifest.get("checkpoint_run_id") != run_id
    ):
        raise ValueError("训练包嵌套 checkpoint 路径的数据身份与 manifest 不一致")


def _checkpoint_run_id_from_path(relative: str) -> str | None:
    parts = PurePosixPath(relative).parts
    if len(parts) == 5 and re.fullmatch(r"run_[0-9]{20}", parts[3]):
        return parts[3]
    return None


def _validate_symbol(value: Any) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValueError("训练包 symbol 非法")
    return value


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON 含重复字段: {key}")
        payload[key] = value
    return payload


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"JSON 含非有限数值: {value}")


def _load_json_bytes(content: bytes, label: str) -> Any:
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不是合法 UTF-8 JSON") from exc


def _validate_data_identity(
    manifest: dict[str, Any],
    strategy: dict[str, Any] | None,
    checkpoint: dict[str, Any],
) -> None:
    missing = [field for field in _CHECKPOINT_IDENTITY_FIELDS if manifest.get(field) is None]
    if missing:
        raise ValueError(f"训练包 manifest 缺少完整数据身份: {', '.join(missing)}")
    data_sha256 = manifest.get("data_sha256")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(data_sha256, str) or not _SHA256_RE.fullmatch(data_sha256):
        raise ValueError("训练包 data_sha256 非法")
    if dataset_id != f"sha256:{data_sha256}":
        raise ValueError("训练包 dataset_id 与 data_sha256 不一致")

    result_hash = manifest.get("result_manifest_sha256")
    if result_hash is not None and (
        not isinstance(result_hash, str) or not _SHA256_RE.fullmatch(result_hash)
    ):
        raise ValueError("训练包 result_manifest_sha256 非法")
    run_id = manifest.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id)):
        raise ValueError("训练包 run_id 非法")

    for payload_name, payload in (("检查点", checkpoint), ("策略", strategy)):
        if payload is None:
            continue
        for key in _CHECKPOINT_IDENTITY_FIELDS:
            claimed = payload.get(key)
            expected = manifest.get(key)
            if type(claimed) is not type(expected) or claimed != expected:
                raise ValueError(f"{payload_name}的 {key} 与 manifest 不一致")
        if (
            payload.get("scoring_contract_version")
            != manifest.get("scoring_contract_version")
        ):
            raise ValueError(f"{payload_name}的评分合同与 manifest 不一致")
    if strategy is not None:
        if strategy.get("formula") != checkpoint.get("best_formula"):
            raise ValueError("训练策略 formula 与 checkpoint.best_formula 不一致")
        strategy_score = float(strategy["best_score"])
        checkpoint_score = float(checkpoint["best_score"])
        if not math.isclose(strategy_score, checkpoint_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("训练策略 best_score 与 checkpoint.best_score 不一致")


def _validate_strategy_bytes(
    content: bytes,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    strategy = _load_json_bytes(content, "训练策略")
    if not isinstance(strategy, dict):
        raise ValueError("训练策略必须是 JSON 对象")
    if strategy.get("symbol") != manifest["symbol"]:
        raise ValueError("训练策略 symbol 与 manifest 不一致")
    artifact_version = strategy.get("vocab_version")
    if artifact_version is None:
        raise ValueError("训练策略缺少 vocab_version")
    FORMULA_VOCAB.verify(artifact_version)
    if strategy.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise ValueError("训练策略评分合同不兼容")
    formula = strategy.get("formula")
    if (
        not isinstance(formula, list)
        or not formula
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= FORMULA_VOCAB.size
            for token in formula
        )
    ):
        raise ValueError("训练策略 formula 非法")
    best_score = strategy.get("best_score")
    if (
        isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise ValueError("训练策略 best_score 非法")

    for key in (
        "run_id",
        "timeframe",
        "dataset_id",
        "data_sha256",
        "local_source",
        "periods_per_year",
        "minimum_bars",
    ):
        if manifest.get(key) is not None and strategy.get(key) != manifest[key]:
            raise ValueError(f"训练策略 {key} 与 manifest 不一致")
        if manifest.get(key) is None and strategy.get(key) is not None and key in {
            "run_id",
            "dataset_id",
            "data_sha256",
        }:
            raise ValueError(f"训练策略声明了 manifest 未绑定的 {key}")
    return strategy


def _validate_history_bytes(content: bytes, manifest: dict[str, Any]) -> None:
    history = _load_json_bytes(content, "训练历史")
    if not isinstance(history, dict):
        raise ValueError("训练历史必须是 JSON 对象")
    if history.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise ValueError("训练历史评分合同不兼容")
    steps = history.get("step")
    if steps is None:
        return
    if (
        not isinstance(steps, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in steps
        )
        or any(left >= right for left, right in zip(steps, steps[1:]))
        or (steps and steps[-1] >= manifest["step"])
    ):
        raise ValueError("训练历史 step 序列非法或超出 checkpoint")


def build_training_export_zip(symbol: str) -> tuple[bytes, str]:
    symbol = _validate_symbol(symbol)
    bundle = get_published_bundle(symbol)
    if bundle:
        _verify_published_bundle(bundle)
    ckpts = checkpoint_glob(symbol)
    if not ckpts:
        raise FileNotFoundError(f"未找到 {symbol} 的训练检查点，请先训练并保存 checkpoint")

    latest = ckpts[-1]
    name_match = _CKPT_NAME_RE.fullmatch(latest.name)
    if not name_match or name_match.group(1) != symbol:
        raise ValueError("检查点文件名与导出品种不一致")
    step = int(name_match.group(2))
    checkpoint_meta = _validate_checkpoint_file(
        latest,
        expected_symbol=symbol,
        expected_step=step,
        require_data_identity=True,
    )

    strategy = _load_strategy(symbol)
    history_path = _history_path(symbol)
    checkpoint_relative = _checkpoint_relative_for_export(latest, bundle)

    safe = symbol.replace(".", "_")
    zip_name = f"training_{safe}_step{step:04d}.zip"

    manifest = {
        "format": _PACKAGE_FORMAT,
        "symbol": symbol,
        "step": step,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "checkpoint": checkpoint_relative,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    for key in _CHECKPOINT_IDENTITY_FIELDS:
        if key != "symbol":
            manifest[key] = checkpoint_meta[key]
    if bundle:
        manifest.update({
            "run_id": bundle["run_id"],
            "timeframe": bundle.get("timeframe"),
            "dataset_id": bundle.get("dataset_id"),
            "local_source": bundle.get("local_source"),
            "periods_per_year": bundle.get("periods_per_year"),
            "minimum_bars": bundle.get("minimum_bars"),
            "data_sha256": bundle.get("data_sha256"),
            "result_manifest_sha256": bundle.get("result_manifest_sha256"),
        })
    elif strategy:
        # 本机训练若已经携带可验证的数据身份，也必须写入 manifest，不能只藏在策略中。
        for key in (
            "run_id",
            "timeframe",
            "dataset_id",
            "data_sha256",
            "local_source",
            "periods_per_year",
            "minimum_bars",
        ):
            if strategy.get(key) is not None:
                manifest[key] = strategy[key]

    checkpoint_run_id = _checkpoint_run_id_from_path(checkpoint_relative)
    if checkpoint_run_id is not None:
        manifest["checkpoint_run_id"] = checkpoint_run_id
    _validate_checkpoint_path_contract(checkpoint_relative, symbol, step, manifest)

    artifacts: dict[str, bytes] = {
        manifest["checkpoint"]: _read_file_bytes(latest),
    }
    strategy_payload: dict[str, Any] | None = None
    if strategy:
        strat_name = f"strategies/best_{symbol}.json"
        strategy_payload = dict(strategy)
        if strategy_payload.get("formula") and not strategy_payload.get("formula_decoded"):
            strategy_payload["formula_decoded"] = _decode_formula(strategy_payload["formula"])
        artifacts[strat_name] = json.dumps(
            strategy_payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")

    if history_path.exists():
        hist_name = f"training_history_{symbol}.json"
        artifacts[hist_name] = _read_file_bytes(history_path)

    manifest["files"] = list(artifacts)
    manifest["artifacts"] = [
        _artifact_row(relative, payload) for relative, payload in artifacts.items()
    ]
    if strategy_payload is not None:
        _validate_strategy_bytes(artifacts[f"strategies/best_{symbol}.json"], manifest)
    _validate_data_identity(manifest, strategy_payload, checkpoint_meta)
    if history_path.exists():
        _validate_history_bytes(artifacts[f"training_history_{symbol}.json"], manifest)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for relative, payload in artifacts.items():
            zf.writestr(relative, payload)
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        )

    return buf.getvalue(), zip_name


def import_training_package(
    content: bytes,
    filename: str,
    expected_symbol: str | None = None,
) -> dict[str, Any]:
    name = Path(filename).name.lower()
    if not name.endswith(".zip"):
        raise ValueError("仅支持含完整 manifest、哈希与数据身份的 v2 ZIP 训练包")
    result = _import_zip(content, expected_symbol)
    symbol = result["symbol"]
    step = result["step"]
    installed = result["installed"]

    invalidate_checkpoint_cache()
    return {
        "ok": True,
        "symbol": symbol,
        "step": step,
        "installed": installed,
        "source_checkpoint_run_id": result["source_checkpoint_run_id"],
        "checkpoint_run_id": result["checkpoint_run_id"],
        "message": f"已导入 {symbol} 的训练文件（step {step}），下次训练将从断点续训",
    }


def _validate_zip_members(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = zf.infolist()
    if not infos or len(infos) > _MAX_MEMBERS:
        raise ValueError("训练包成员数量非法")

    found: dict[str, zipfile.ZipInfo] = {}
    casefolded: set[str] = set()
    total_size = 0
    for info in infos:
        name = info.filename
        parts = name.split("/")
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if not name or "\x00" in name:
            raise ValueError("训练包含空成员名或 NUL 字符")
        if "\\" in name:
            raise ValueError(f"训练包成员禁止使用反斜杠: {name}")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
            raise ValueError(f"训练包成员禁止绝对路径或盘符: {name}")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"训练包成员路径非法: {name}")
        posix = PurePosixPath(name)
        if posix.is_absolute():
            raise ValueError(f"训练包成员禁止绝对路径: {name}")
        is_dos_directory = info.create_system == 0 and bool(info.external_attr & 0x10)
        if info.is_dir() or name.endswith("/") or file_type == stat.S_IFDIR or is_dos_directory:
            raise ValueError(f"训练包禁止目录成员: {name}")
        if file_type == stat.S_IFLNK:
            raise ValueError(f"训练包禁止链接成员: {name}")
        if file_type not in {0, stat.S_IFREG}:
            raise ValueError(f"训练包禁止特殊文件成员: {name}")
        if info.flag_bits & 0x1:
            raise ValueError(f"训练包禁止加密成员: {name}")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError(f"训练包成员压缩格式不受支持: {name}")
        folded = name.casefold()
        if name in found or folded in casefolded:
            raise ValueError(f"训练包含重复成员: {name}")
        if info.file_size < 0 or info.file_size > _MAX_MEMBER_BYTES:
            raise ValueError(f"训练包成员大小非法: {name}")
        total_size += info.file_size
        if total_size > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("训练包解压后总大小超限")
        found[name] = info
        casefolded.add(folded)
    return found


def _read_member_limited(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limit: int,
) -> bytes:
    if info.file_size > limit:
        raise ValueError(f"训练包成员过大: {info.filename}")
    chunks: list[bytes] = []
    total = 0
    with zf.open(info, "r") as source:
        while True:
            block = source.read(min(1024 * 1024, limit - total + 1))
            if not block:
                break
            total += len(block)
            if total > limit:
                raise ValueError(f"训练包成员解压大小超限: {info.filename}")
            chunks.append(block)
    if total != info.file_size:
        raise ValueError(f"训练包成员实际大小与 ZIP 目录不一致: {info.filename}")
    return b"".join(chunks)


def _validate_import_manifest(
    payload: Any,
    infos: dict[str, zipfile.ZipInfo],
    expected_symbol: str | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("训练包 manifest 必须是 JSON 对象")
    package_format = payload.get("format")
    if package_format == _LEGACY_PACKAGE_FORMAT:
        raise ValueError("旧版训练包没有强制大小/哈希清单，已拒绝导入；请重新导出")
    if package_format != _PACKAGE_FORMAT:
        raise ValueError("不支持的训练包格式")
    if payload.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise ValueError("训练包评分合同不兼容")

    symbol = _validate_symbol(payload.get("symbol"))
    if expected_symbol and symbol != expected_symbol:
        raise ValueError(f"训练包品种为 {symbol}，与当前选择的 {expected_symbol} 不一致")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("训练包 manifest 的 step 非法")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise ValueError("训练包 manifest 缺少 checkpoint")
    _validate_checkpoint_path_contract(checkpoint, symbol, step, payload)

    files = payload.get("files")
    if (
        not isinstance(files, list)
        or not files
        or len(files) > _MAX_MEMBERS - 1
        or any(not isinstance(item, str) for item in files)
        or len(files) != len(set(files))
        or len(files) != len({item.casefold() for item in files})
    ):
        raise ValueError("训练包 manifest 的 files 非法或重复")
    actual_files = set(infos) - {"manifest.json"}
    if set(files) != actual_files or len(infos) != len(files) + 1:
        raise ValueError("训练包 manifest 与 ZIP 实际成员不一致")

    allowed = {
        checkpoint,
        f"strategies/best_{symbol}.json",
        f"training_history_{symbol}.json",
    }
    if any(relative not in allowed for relative in files):
        raise ValueError("训练包含严格 allowlist 之外的成员")
    if checkpoint not in files:
        raise ValueError("训练包缺少 manifest 指定的 checkpoint")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(files):
        raise ValueError("训练包缺少完整大小/哈希清单")
    rows: dict[str, dict[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise ValueError("训练包大小/哈希清单结构非法")
        relative = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(relative, str) or relative in rows:
            raise ValueError("训练包大小/哈希清单含重复路径")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"训练包声明的成员大小非法: {relative}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"训练包声明的成员哈希非法: {relative}")
        if relative not in infos or infos[relative].file_size != size:
            raise ValueError(f"训练包声明大小与 ZIP 成员不一致: {relative}")
        rows[relative] = row
    if set(rows) != set(files):
        raise ValueError("训练包大小/哈希清单与 files 不一致")

    for key in ("periods_per_year", "minimum_bars"):
        value = payload.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"训练包 {key} 非法")
    timeframe = payload.get("timeframe")
    if timeframe is not None and (
        not isinstance(timeframe, str) or not _TIMEFRAME_RE.fullmatch(timeframe)
    ):
        raise ValueError("训练包 timeframe 非法")
    local_source = payload.get("local_source")
    if local_source is not None and local_source not in _LOCAL_SOURCES:
        raise ValueError("训练包 local_source 非法")
    return payload


def _staged_path(stage_root: Path, relative: str) -> Path:
    # 暂存名不复刻深层身份路径，避免 Windows MAX_PATH；relative 已在 ZIP/manifest 层严格校验。
    staging_name = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    candidate = stage_root / "payloads" / staging_name
    if not candidate.resolve().is_relative_to(stage_root.resolve()):
        raise ValueError("训练包暂存路径越界")
    return candidate


def _stage_verified_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    row: dict[str, Any],
    stage_root: Path,
) -> None:
    destination = _staged_path(stage_root, info.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with zf.open(info, "r") as source, destination.open("xb") as target:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > row["size"] or total > _MAX_MEMBER_BYTES:
                raise ValueError(f"训练包成员解压大小超出声明: {info.filename}")
            digest.update(block)
            target.write(block)
    if total != row["size"] or total != info.file_size:
        raise ValueError(f"训练包成员大小校验失败: {info.filename}")
    if digest.hexdigest() != row["sha256"]:
        raise ValueError(f"训练包成员 SHA-256 校验失败: {info.filename}")


def _localized_publish_paths(
    manifest: dict[str, Any],
) -> tuple[dict[str, str], str]:
    timeframe = manifest["timeframe"]
    data_sha256 = manifest["data_sha256"]
    identity_root = PROJECT_ROOT.resolve() / "checkpoints" / timeframe / data_sha256
    if identity_root.is_symlink() or (
        identity_root.exists()
        and (
            not identity_root.is_dir()
            or not identity_root.resolve().is_relative_to(PROJECT_ROOT.resolve())
        )
    ):
        raise ValueError("本机 checkpoint 身份目录非法")

    existing_numbers: list[int] = []
    if identity_root.exists():
        for path in identity_root.iterdir():
            match = re.fullmatch(r"run_([0-9]{20})", path.name)
            if match:
                existing_numbers.append(int(match.group(1)))
    next_number = max([time.time_ns(), *(value + 1 for value in existing_numbers)])
    if next_number > 99_999_999_999_999_999_999:
        raise ValueError("本机 checkpoint run 序号已耗尽")
    local_run_id = f"run_{next_number:020d}"

    source_checkpoint = manifest["checkpoint"]
    checkpoint_name = PurePosixPath(source_checkpoint).name
    local_checkpoint = (
        f"checkpoints/{timeframe}/{data_sha256}/{local_run_id}/{checkpoint_name}"
    )
    publish_paths = {relative: relative for relative in manifest["files"]}
    publish_paths[source_checkpoint] = local_checkpoint
    return publish_paths, local_run_id


def _atomic_publish_staged(
    stage_root: Path,
    publish_paths: dict[str, str],
    symbol: str,
) -> list[str]:
    project_root = PROJECT_ROOT.resolve()
    destinations: dict[Path, Path] = {}
    installed: list[str] = []
    for source_relative, destination_relative in publish_paths.items():
        final_path = project_root.joinpath(*PurePosixPath(destination_relative).parts)
        if not final_path.parent.resolve().is_relative_to(project_root):
            raise ValueError("训练包发布路径越出项目目录")
        destinations[final_path] = _staged_path(stage_root, source_relative)
        installed.append(destination_relative)

    checkpoint_relative = next(
        relative for relative in installed if relative.endswith(".pt")
    )
    checkpoint_parts = PurePosixPath(checkpoint_relative).parts
    checkpoint_scope = project_root.joinpath(*checkpoint_parts[:-1])
    if checkpoint_scope.exists() and (
        checkpoint_scope.is_symlink()
        or not checkpoint_scope.is_dir()
        or not checkpoint_scope.resolve().is_relative_to(project_root)
    ):
        raise ValueError("项目 checkpoint 身份目录非法")
    old_checkpoints = [
        path
        for path in checkpoint_scope.glob("ckpt_*_step_*.pt")
        if path.is_file() and _symbol_from_ckpt_name(path.name) == symbol
    ] if checkpoint_scope.exists() else []
    removals = [path for path in old_checkpoints if path not in destinations]
    history_path = project_root / f"training_history_{symbol}.json"
    if f"training_history_{symbol}.json" not in installed and history_path.exists():
        removals.append(history_path)

    affected = list(dict.fromkeys([*destinations, *removals]))
    backup_dir = stage_root / "__backup__"
    removed_dir = stage_root / "__removed__"
    backup_dir.mkdir()
    removed_dir.mkdir()
    existed: dict[Path, bool] = {}
    backups: dict[Path, Path] = {}
    for index, path in enumerate(affected):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"现有训练产物不是普通文件: {path.name}")
        existed[path] = path.exists()
        if path.exists():
            backup = backup_dir / f"{index:04d}.bak"
            shutil.copy2(path, backup)
            backups[path] = backup

    created_dirs: list[Path] = []
    changed_paths: list[Path] = []
    try:
        for destination in destinations:
            missing: list[Path] = []
            parent = destination.parent
            while parent != project_root and not parent.exists():
                missing.append(parent)
                parent = parent.parent
            if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
                raise OSError(f"发布父目录非法: {parent}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            created_dirs.extend(reversed(missing))

        for destination, staged in destinations.items():
            os.replace(staged, destination)
            changed_paths.append(destination)
        for index, path in enumerate(removals):
            if path.exists():
                os.replace(path, removed_dir / f"{index:04d}.removed")
                changed_paths.append(path)
    except Exception as exc:
        rollback_errors: list[str] = []
        for path in reversed(changed_paths):
            try:
                if existed[path]:
                    backup = backups[path]
                    if not backup.exists():
                        raise OSError("回滚备份缺失")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, path)
                elif path.exists() or path.is_symlink():
                    path.unlink()
            except OSError as rollback_exc:
                rollback_errors.append(f"{path.name}: {rollback_exc}")
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
        if rollback_errors:
            raise RuntimeError(
                f"训练包原子发布失败且回滚不完整: {'; '.join(rollback_errors)}"
            ) from exc
        raise RuntimeError("训练包原子发布失败，现有产物已回滚") from exc

    return installed


def _import_zip(content: bytes, expected_symbol: str | None) -> dict[str, Any]:
    if not isinstance(content, bytes) or not content:
        raise ValueError("训练包为空")
    if len(content) > _MAX_ARCHIVE_BYTES:
        raise ValueError("训练包压缩文件过大")

    project_root = PROJECT_ROOT.resolve()
    scratch_root = PROJECT_ROOT / "scratch"
    if scratch_root.is_symlink() or (
        scratch_root.exists()
        and (
            not scratch_root.is_dir()
            or not scratch_root.resolve().is_relative_to(project_root)
        )
    ):
        raise ValueError("项目 scratch 暂存目录非法")
    scratch_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="training-import-",
            dir=scratch_root,
            ignore_cleanup_errors=True,
        ) as tmp:
            stage_root = Path(tmp)
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    infos = _validate_zip_members(zf)
                    manifest_info = infos.get("manifest.json")
                    if manifest_info is None:
                        raise ValueError("训练包缺少 manifest.json")
                    manifest_bytes = _read_member_limited(
                        zf,
                        manifest_info,
                        _MAX_MANIFEST_BYTES,
                    )
                    manifest = _validate_import_manifest(
                        _load_json_bytes(manifest_bytes, "训练包 manifest"),
                        infos,
                        expected_symbol,
                    )
                    artifact_rows = {
                        row["path"]: row for row in manifest["artifacts"]
                    }
                    for relative in manifest["files"]:
                        _stage_verified_member(
                            zf,
                            infos[relative],
                            artifact_rows[relative],
                            stage_root,
                        )
            except zipfile.BadZipFile as exc:
                raise ValueError("训练包 ZIP 结构或 CRC 非法") from exc

            symbol = manifest["symbol"]
            step = manifest["step"]
            checkpoint_meta = _validate_checkpoint_file(
                _staged_path(stage_root, manifest["checkpoint"]),
                expected_symbol=symbol,
                expected_step=step,
                require_data_identity=True,
            )
            strategy_rel = f"strategies/best_{symbol}.json"
            strategy = (
                _validate_strategy_bytes(
                    _staged_path(stage_root, strategy_rel).read_bytes(),
                    manifest,
                )
                if strategy_rel in manifest["files"]
                else None
            )
            history_rel = f"training_history_{symbol}.json"
            if history_rel in manifest["files"]:
                _validate_history_bytes(
                    _staged_path(stage_root, history_rel).read_bytes(),
                    manifest,
                )
            _validate_data_identity(manifest, strategy, checkpoint_meta)
            with _IMPORT_LOCK:
                publish_paths, local_checkpoint_run_id = _localized_publish_paths(
                    manifest
                )
                installed = _atomic_publish_staged(
                    stage_root,
                    publish_paths,
                    symbol,
                )
            return {
                "symbol": symbol,
                "step": step,
                "installed": installed,
                "source_checkpoint_run_id": manifest["checkpoint_run_id"],
                "checkpoint_run_id": local_checkpoint_run_id,
            }
    except OSError as exc:
        raise ValueError(f"训练包暂存或发布失败: {exc}") from exc
