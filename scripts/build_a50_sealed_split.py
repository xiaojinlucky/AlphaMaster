"""把冻结的 50 股后复权数据物理拆成训练集与一次性封存评估集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.a_share_akshare import (
    AKSHARE_SLICE_SEALED_EVALUATION,
    AKSHARE_SLICE_TRAINING,
    AKShareDataError,
    load_akshare_hfq_manifest,
    publish_akshare_hfq_slice,
)
from data_pipeline.dataset_contracts import AKSHARE_HFQ_SOURCE_ID
from scripts.freeze_csi_a50_universe import (
    UniverseContractError,
    load_frozen_universe,
    write_json_exclusive,
)


LEGACY_SPLIT_FORMAT = "alphamaster_a50_sealed_split_v1"
SPLIT_FORMAT = "alphamaster_a50_sealed_split_v2"
MIN_SYMBOL_TEST_BARS = 200
DEFAULT_WARMUP_BARS = 252
_DATE_RE = re.compile(r"^[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
SPLIT_KEYS = (
    "format",
    "universe_file",
    "universe_contract_sha256",
    "universe_snapshot_date",
    "source_id",
    "requested_test_start",
    "test_start",
    "test_end",
    "common_test_bar_count",
    "minimum_symbol_test_bar_count",
    "maximum_symbol_test_bar_count",
    "warmup_bars",
    "symbol_count",
    "training_data_dir",
    "evaluation_data_dir",
    "items",
    "contract_sha256",
)
SPLIT_ITEM_KEYS = (
    "symbol",
    "name",
    "parent_data_sha256",
    "training",
    "sealed_evaluation",
)
TRAINING_ITEM_KEYS = (
    "data_filename",
    "data_sha256",
    "data_manifest_sha256",
    "dataset_id",
    "data_rows",
    "data_start",
    "data_end",
)
LEGACY_TRAINING_ITEM_KEYS = tuple(
    field
    for field in TRAINING_ITEM_KEYS
    if field != "data_manifest_sha256"
)
SEALED_EVALUATION_ITEM_KEYS = (
    *TRAINING_ITEM_KEYS,
    "score_start",
    "warmup_bars",
)
LEGACY_SEALED_EVALUATION_ITEM_KEYS = (
    *LEGACY_TRAINING_ITEM_KEYS,
    "score_start",
    "warmup_bars",
)


class SealedSplitError(RuntimeError):
    """50 股训练/封存切分无法满足统一时间与身份合同。"""


def _canonical_hash(payload: dict[str, Any]) -> str:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _utc_iso(unix_seconds: int) -> str:
    return (
        datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _requested_start_seconds(value: str) -> int:
    if _DATE_RE.fullmatch(str(value)) is None:
        raise SealedSplitError("requested_test_start 必须是 YYYYMMDD")
    try:
        day = datetime.strptime(str(value), "%Y%m%d").date()
    except ValueError as exc:
        raise SealedSplitError("requested_test_start 不是合法日期") from exc
    local_close = datetime.combine(
        day,
        time(hour=15),
        tzinfo=_SHANGHAI_TIMEZONE,
    )
    return int(local_close.astimezone(timezone.utc).timestamp())


def _require_exact_dict(
    value: Any,
    keys: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or len(value) != len(keys)
        or set(value) != set(keys)
    ):
        raise SealedSplitError(f"{label} 字段合同发生变化")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SealedSplitError(f"{label} 必须是小写 SHA-256")
    return value


def _require_positive_int(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise SealedSplitError(f"{label} 必须是不小于 {minimum} 的整数")
    return value


def _parse_utc_iso(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise SealedSplitError(f"{label} 必须是 UTC 时间")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise SealedSplitError(
            f"{label} 必须是 YYYY-MM-DDTHH:MM:SSZ"
        ) from exc
    return parsed


def _resolve_contract_reference(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SealedSplitError(f"{label} 不能为空")
    raw = Path(value)
    return (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()


def _load_slice_manifest(
    *,
    root: Path,
    filename: str,
    symbol: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    expected_filename = f"{symbol}_D1.parquet"
    if filename != expected_filename or Path(filename).name != filename:
        raise SealedSplitError(f"{label}.data_filename 不匹配")
    path = (root / filename).resolve()
    if path.parent != root:
        raise SealedSplitError(f"{label}.data_filename 越出冻结目录")
    try:
        frame = pd.read_parquet(path)
        manifest = load_akshare_hfq_manifest(path, frame)
    except Exception as exc:
        raise SealedSplitError(
            f"{label} 数据验证失败: {type(exc).__name__}: {exc}"
        ) from exc
    if manifest is None:
        raise SealedSplitError(f"{label} 不属于 AKShare 后复权合同")
    return path, manifest


def load_a50_sealed_split(path: str | Path) -> dict[str, Any]:
    """严格复核 50 股训练/封存合同及其全部 100 份物理数据。"""
    contract_path = Path(path).resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedSplitError("切分合同不是合法 UTF-8 JSON") from exc
    payload = _require_exact_dict(payload, SPLIT_KEYS, "切分合同")
    split_format = payload["format"]
    if split_format not in {LEGACY_SPLIT_FORMAT, SPLIT_FORMAT}:
        raise SealedSplitError("切分合同 format 不匹配")
    training_item_keys = (
        TRAINING_ITEM_KEYS
        if split_format == SPLIT_FORMAT
        else LEGACY_TRAINING_ITEM_KEYS
    )
    evaluation_item_keys = (
        SEALED_EVALUATION_ITEM_KEYS
        if split_format == SPLIT_FORMAT
        else LEGACY_SEALED_EVALUATION_ITEM_KEYS
    )
    contract_hash = _require_sha256(
        payload["contract_sha256"],
        "contract_sha256",
    )
    body = {key: payload[key] for key in SPLIT_KEYS[:-1]}
    if _canonical_hash(body) != contract_hash:
        raise SealedSplitError("contract_sha256 与切分合同内容不一致")
    if payload["source_id"] != AKSHARE_HFQ_SOURCE_ID:
        raise SealedSplitError("切分合同只接受 AKShare 新浪后复权来源")

    requested_seconds = _requested_start_seconds(
        payload["requested_test_start"]
    )
    test_start = _parse_utc_iso(payload["test_start"], "test_start")
    test_end = _parse_utc_iso(payload["test_end"], "test_end")
    if int(test_start.timestamp()) != requested_seconds:
        raise SealedSplitError("test_start 与 requested_test_start 不一致")
    if test_end <= test_start:
        raise SealedSplitError("test_end 必须晚于 test_start")

    common_count = _require_positive_int(
        payload["common_test_bar_count"],
        "common_test_bar_count",
        minimum=2,
    )
    minimum_count = _require_positive_int(
        payload["minimum_symbol_test_bar_count"],
        "minimum_symbol_test_bar_count",
        minimum=MIN_SYMBOL_TEST_BARS,
    )
    maximum_count = _require_positive_int(
        payload["maximum_symbol_test_bar_count"],
        "maximum_symbol_test_bar_count",
        minimum=minimum_count,
    )
    warmup_bars = _require_positive_int(
        payload["warmup_bars"],
        "warmup_bars",
        minimum=DEFAULT_WARMUP_BARS,
    )
    if common_count > minimum_count:
        raise SealedSplitError(
            "共同交易日数量不能大于任一标的的真实测试 K 线数"
        )
    if payload["symbol_count"] != 50:
        raise SealedSplitError("symbol_count 必须恰好为 50")

    universe_path = _resolve_contract_reference(
        payload["universe_file"],
        "universe_file",
    )
    try:
        universe = load_frozen_universe(universe_path)
    except UniverseContractError as exc:
        raise SealedSplitError(f"冻结股票池验证失败: {exc}") from exc
    if (
        universe["contract_sha256"]
        != payload["universe_contract_sha256"]
        or universe["snapshot_date"] != payload["universe_snapshot_date"]
    ):
        raise SealedSplitError("切分合同与冻结股票池身份不一致")

    training_root = _resolve_contract_reference(
        payload["training_data_dir"],
        "training_data_dir",
    )
    evaluation_root = _resolve_contract_reference(
        payload["evaluation_data_dir"],
        "evaluation_data_dir",
    )
    if training_root == evaluation_root:
        raise SealedSplitError("训练目录与封存评估目录必须物理分离")

    items = payload["items"]
    if not isinstance(items, list) or len(items) != 50:
        raise SealedSplitError("items 必须恰好包含 50 个标的")
    expected_constituents = universe["constituents"]
    test_counts: list[int] = []
    for position, (raw, constituent) in enumerate(
        zip(items, expected_constituents, strict=True),
        start=1,
    ):
        item = _require_exact_dict(
            raw,
            SPLIT_ITEM_KEYS,
            f"第 {position} 个标的",
        )
        symbol = item["symbol"]
        if (
            not isinstance(symbol, str)
            or _SYMBOL_RE.fullmatch(symbol) is None
            or symbol != constituent["symbol"]
            or item["name"] != constituent["name"]
        ):
            raise SealedSplitError(
                f"第 {position} 个标的与冻结股票池顺序或身份不一致"
            )
        parent_hash = _require_sha256(
            item["parent_data_sha256"],
            f"{symbol}.parent_data_sha256",
        )
        training = _require_exact_dict(
            item["training"],
            training_item_keys,
            f"{symbol}.training",
        )
        evaluation = _require_exact_dict(
            item["sealed_evaluation"],
            evaluation_item_keys,
            f"{symbol}.sealed_evaluation",
        )
        training_hash = _require_sha256(
            training["data_sha256"],
            f"{symbol}.training.data_sha256",
        )
        evaluation_hash = _require_sha256(
            evaluation["data_sha256"],
            f"{symbol}.sealed_evaluation.data_sha256",
        )
        training_manifest_hash = (
            _require_sha256(
                training["data_manifest_sha256"],
                f"{symbol}.training.data_manifest_sha256",
            )
            if split_format == SPLIT_FORMAT
            else None
        )
        evaluation_manifest_hash = (
            _require_sha256(
                evaluation["data_manifest_sha256"],
                f"{symbol}.sealed_evaluation.data_manifest_sha256",
            )
            if split_format == SPLIT_FORMAT
            else None
        )
        if (
            training["dataset_id"] != f"sha256:{training_hash}"
            or evaluation["dataset_id"] != f"sha256:{evaluation_hash}"
        ):
            raise SealedSplitError(f"{symbol} 的 dataset_id 不匹配")
        if training_hash == evaluation_hash:
            raise SealedSplitError(f"{symbol} 的训练与封存数据哈希相同")

        training_rows = _require_positive_int(
            training["data_rows"],
            f"{symbol}.training.data_rows",
            minimum=484,
        )
        evaluation_rows = _require_positive_int(
            evaluation["data_rows"],
            f"{symbol}.sealed_evaluation.data_rows",
            minimum=warmup_bars + MIN_SYMBOL_TEST_BARS,
        )
        if evaluation["warmup_bars"] != warmup_bars:
            raise SealedSplitError(f"{symbol} 的 warmup_bars 不一致")
        item_test_count = evaluation_rows - warmup_bars
        test_counts.append(item_test_count)

        training_start = _parse_utc_iso(
            training["data_start"],
            f"{symbol}.training.data_start",
        )
        training_end = _parse_utc_iso(
            training["data_end"],
            f"{symbol}.training.data_end",
        )
        evaluation_start = _parse_utc_iso(
            evaluation["data_start"],
            f"{symbol}.sealed_evaluation.data_start",
        )
        evaluation_end = _parse_utc_iso(
            evaluation["data_end"],
            f"{symbol}.sealed_evaluation.data_end",
        )
        score_start = _parse_utc_iso(
            evaluation["score_start"],
            f"{symbol}.sealed_evaluation.score_start",
        )
        if not training_start < training_end < test_start:
            raise SealedSplitError(f"{symbol} 的训练时间范围越过封存边界")
        if not evaluation_start < score_start == test_start:
            raise SealedSplitError(f"{symbol} 的封存预热或评分起点不一致")
        if evaluation_end != test_end:
            raise SealedSplitError(f"{symbol} 的封存评估终点不一致")

        training_path, training_manifest = _load_slice_manifest(
            root=training_root,
            filename=training["data_filename"],
            symbol=symbol,
            label=f"{symbol}.training",
        )
        evaluation_path, evaluation_manifest = _load_slice_manifest(
            root=evaluation_root,
            filename=evaluation["data_filename"],
            symbol=symbol,
            label=f"{symbol}.sealed_evaluation",
        )
        if split_format == SPLIT_FORMAT and (
            hashlib.sha256(
                training_path.with_suffix(".manifest.json").read_bytes()
            ).hexdigest()
            != training_manifest_hash
            or hashlib.sha256(
                evaluation_path.with_suffix(".manifest.json").read_bytes()
            ).hexdigest()
            != evaluation_manifest_hash
        ):
            raise SealedSplitError(f"{symbol} 的数据 manifest 哈希不匹配")
        for field in TRAINING_ITEM_KEYS:
            if field == "data_manifest_sha256":
                continue
            if training_manifest.get(field) != training[field]:
                raise SealedSplitError(
                    f"{symbol}.training.{field} 与物理数据不一致"
                )
        for field in TRAINING_ITEM_KEYS:
            if field == "data_manifest_sha256":
                continue
            if evaluation_manifest.get(field) != evaluation[field]:
                raise SealedSplitError(
                    f"{symbol}.sealed_evaluation.{field} 与物理数据不一致"
                )
        training_derivation = training_manifest.get("derivation")
        evaluation_derivation = evaluation_manifest.get("derivation")
        if (
            not isinstance(training_derivation, dict)
            or not isinstance(evaluation_derivation, dict)
            or training_derivation.get("purpose")
            != AKSHARE_SLICE_TRAINING
            or evaluation_derivation.get("purpose")
            != AKSHARE_SLICE_SEALED_EVALUATION
            or training_derivation.get("parent_data_sha256") != parent_hash
            or evaluation_derivation.get("parent_data_sha256") != parent_hash
            or training_derivation.get("universe_contract_sha256")
            != universe["contract_sha256"]
            or evaluation_derivation.get("universe_contract_sha256")
            != universe["contract_sha256"]
            or evaluation_derivation.get("score_start")
            != evaluation["score_start"]
            or evaluation_derivation.get("warmup_bars") != warmup_bars
            or training_manifest["data_rows"] != training_rows
        ):
            raise SealedSplitError(f"{symbol} 的切片派生身份不一致")

    if min(test_counts) != minimum_count or max(test_counts) != maximum_count:
        raise SealedSplitError("标的测试 K 线数量汇总与 items 不一致")
    return payload


def _load_parent(
    parent_dir: Path,
    symbol: str,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    path = parent_dir / f"{symbol}_D1.parquet"
    try:
        frame = pd.read_parquet(path)
        manifest = load_akshare_hfq_manifest(path, frame)
    except Exception as exc:
        raise SealedSplitError(
            f"{symbol} 父数据验证失败: {type(exc).__name__}: {exc}"
        ) from exc
    if manifest is None or manifest.get("derivation") is not None:
        raise SealedSplitError(f"{symbol} 必须使用供应商原始后复权冻结数据")
    if manifest.get("symbol") != symbol:
        raise SealedSplitError(f"{symbol} 父数据股票身份不匹配")
    return path, frame, manifest


def _existing_or_publish(
    *,
    parent_path: Path,
    output_dir: Path,
    start_index: int,
    end_index: int,
    purpose: str,
    universe_contract_sha256: str,
    score_start_index: int | None,
) -> dict[str, Any]:
    target = output_dir / parent_path.name
    manifest_path = target.with_suffix(".manifest.json")
    if target.exists() != manifest_path.exists():
        raise SealedSplitError(f"{target.name} 的切片数据与 manifest 不完整")
    if not target.exists():
        try:
            return publish_akshare_hfq_slice(
                parent_data_file=parent_path,
                output_dir=output_dir,
                start_index=start_index,
                end_index=end_index,
                purpose=purpose,
                universe_contract_sha256=universe_contract_sha256,
                score_start_index=score_start_index,
            )
        except AKShareDataError as exc:
            raise SealedSplitError(f"{target.name} 切片发布失败: {exc}") from exc

    try:
        frame = pd.read_parquet(target)
        manifest = load_akshare_hfq_manifest(target, frame)
    except Exception as exc:
        raise SealedSplitError(
            f"{target.name} 现有切片验证失败: {type(exc).__name__}: {exc}"
        ) from exc
    if manifest is None:
        raise SealedSplitError(f"{target.name} 现有切片合同不受支持")
    parent_frame = pd.read_parquet(parent_path)
    expected = parent_frame.iloc[start_index:end_index].reset_index(drop=True)
    if not frame.equals(expected):
        raise SealedSplitError(f"{target.name} 现有切片内容与父数据范围不一致")
    derivation = manifest.get("derivation")
    expected_score_start = (
        _utc_iso(int(parent_frame["time"].iloc[score_start_index]))
        if score_start_index is not None
        else None
    )
    if (
        not isinstance(derivation, dict)
        or derivation.get("purpose") != purpose
        or derivation.get("parent_data_sha256")
        != load_akshare_hfq_manifest(parent_path, parent_frame)["data_sha256"]
        or derivation.get("universe_contract_sha256")
        != universe_contract_sha256
        or derivation.get("score_start") != expected_score_start
    ):
        raise SealedSplitError(f"{target.name} 现有切片身份与本次合同不一致")
    return {
        **manifest,
        "data_file": str(target.resolve()),
        "manifest_file": str(manifest_path.resolve()),
    }


def build_a50_sealed_split(
    *,
    universe_json: str | Path,
    parent_dir: str | Path,
    training_dir: str | Path,
    evaluation_dir: str | Path,
    requested_test_start: str,
    output_contract: str | Path,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
) -> dict[str, Any]:
    output_path = Path(output_contract).resolve()
    if output_path.exists():
        raise SealedSplitError(f"切分合同已存在，拒绝覆盖: {output_path}")
    if (
        isinstance(warmup_bars, bool)
        or not isinstance(warmup_bars, int)
        or warmup_bars < DEFAULT_WARMUP_BARS
    ):
        raise SealedSplitError(
            f"warmup_bars 不得小于 {DEFAULT_WARMUP_BARS}"
        )
    requested_seconds = _requested_start_seconds(requested_test_start)
    try:
        universe = load_frozen_universe(universe_json)
    except UniverseContractError as exc:
        raise SealedSplitError(f"冻结股票池验证失败: {exc}") from exc

    parent_root = Path(parent_dir).resolve()
    training_root = Path(training_dir).resolve()
    evaluation_root = Path(evaluation_dir).resolve()
    loaded: dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]] = {}
    common_times: set[int] | None = None
    for constituent in universe["constituents"]:
        symbol = constituent["symbol"]
        parent = _load_parent(parent_root, symbol)
        loaded[symbol] = parent
        times = set(
            parent[1]["time"].to_numpy(dtype=np.int64, copy=False).tolist()
        )
        common_times = times if common_times is None else common_times & times
    if not common_times:
        raise SealedSplitError("50 股没有共同交易时间")
    common_test_times = sorted(
        value for value in common_times if value >= requested_seconds
    )
    if len(common_test_times) < 2:
        raise SealedSplitError("50 股没有可用的统一封存起点和终点")
    score_start_seconds = common_test_times[0]
    test_end_seconds = common_test_times[-1]

    plans: dict[str, dict[str, int]] = {}
    for symbol, (_path, frame, parent_manifest) in loaded.items():
        times = frame["time"].to_numpy(dtype=np.int64, copy=False)
        score_index = int(np.searchsorted(times, score_start_seconds))
        end_index = int(np.searchsorted(times, test_end_seconds)) + 1
        if (
            score_index >= len(times)
            or int(times[score_index]) != score_start_seconds
            or end_index <= score_index
            or int(times[end_index - 1]) != test_end_seconds
        ):
            raise SealedSplitError(f"{symbol} 不覆盖统一封存窗口")
        if end_index != len(times):
            raise SealedSplitError(f"{symbol} 父数据终点晚于统一封存终点")
        test_bar_count = end_index - score_index
        if test_bar_count < MIN_SYMBOL_TEST_BARS:
            raise SealedSplitError(
                f"{symbol} 封存期真实 K 线不足: "
                f"{test_bar_count} < {MIN_SYMBOL_TEST_BARS}"
            )
        evaluation_start = score_index - warmup_bars
        if evaluation_start < 0:
            raise SealedSplitError(f"{symbol} 没有足够的封存评估预热数据")
        if score_index < parent_manifest["minimum_bars"]:
            raise SealedSplitError(f"{symbol} 训练切片数据不足")
        if end_index - evaluation_start < parent_manifest["minimum_bars"]:
            raise SealedSplitError(f"{symbol} 封存评估切片数据不足")
        plans[symbol] = {
            "score_index": score_index,
            "evaluation_start": evaluation_start,
            "end_index": end_index,
            "test_bar_count": test_bar_count,
        }

    items: list[dict[str, Any]] = []
    for constituent in universe["constituents"]:
        symbol = constituent["symbol"]
        parent_path, _frame, parent_manifest = loaded[symbol]
        plan = plans[symbol]
        training = _existing_or_publish(
            parent_path=parent_path,
            output_dir=training_root,
            start_index=0,
            end_index=plan["score_index"],
            purpose=AKSHARE_SLICE_TRAINING,
            universe_contract_sha256=universe["contract_sha256"],
            score_start_index=None,
        )
        evaluation = _existing_or_publish(
            parent_path=parent_path,
            output_dir=evaluation_root,
            start_index=plan["evaluation_start"],
            end_index=plan["end_index"],
            purpose=AKSHARE_SLICE_SEALED_EVALUATION,
            universe_contract_sha256=universe["contract_sha256"],
            score_start_index=plan["score_index"],
        )
        items.append(
            {
                "symbol": symbol,
                "name": constituent["name"],
                "parent_data_sha256": parent_manifest["data_sha256"],
                "training": {
                    "data_filename": training["data_filename"],
                    "data_sha256": training["data_sha256"],
                    "data_manifest_sha256": hashlib.sha256(
                        Path(training["manifest_file"]).read_bytes()
                    ).hexdigest(),
                    "dataset_id": training["dataset_id"],
                    "data_rows": training["data_rows"],
                    "data_start": training["data_start"],
                    "data_end": training["data_end"],
                },
                "sealed_evaluation": {
                    "data_filename": evaluation["data_filename"],
                    "data_sha256": evaluation["data_sha256"],
                    "data_manifest_sha256": hashlib.sha256(
                        Path(evaluation["manifest_file"]).read_bytes()
                    ).hexdigest(),
                    "dataset_id": evaluation["dataset_id"],
                    "data_rows": evaluation["data_rows"],
                    "data_start": evaluation["data_start"],
                    "data_end": evaluation["data_end"],
                    "score_start": evaluation["derivation"]["score_start"],
                    "warmup_bars": evaluation["derivation"]["warmup_bars"],
                },
            }
        )

    body = {
        "format": SPLIT_FORMAT,
        "universe_file": Path(universe_json).as_posix(),
        "universe_contract_sha256": universe["contract_sha256"],
        "universe_snapshot_date": universe["snapshot_date"],
        "source_id": AKSHARE_HFQ_SOURCE_ID,
        "requested_test_start": requested_test_start,
        "test_start": _utc_iso(score_start_seconds),
        "test_end": _utc_iso(test_end_seconds),
        "common_test_bar_count": len(common_test_times),
        "minimum_symbol_test_bar_count": min(
            plan["test_bar_count"] for plan in plans.values()
        ),
        "maximum_symbol_test_bar_count": max(
            plan["test_bar_count"] for plan in plans.values()
        ),
        "warmup_bars": warmup_bars,
        "symbol_count": universe["constituent_count"],
        "training_data_dir": Path(training_dir).as_posix(),
        "evaluation_data_dir": Path(evaluation_dir).as_posix(),
        "items": items,
    }
    contract = {**body, "contract_sha256": _canonical_hash(body)}
    try:
        write_json_exclusive(output_path, contract)
    except UniverseContractError as exc:
        raise SealedSplitError(str(exc)) from exc
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建 50 股训练集与统一一次性封存评估集"
    )
    parser.add_argument("--universe-json", required=True)
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--requested-test-start", required=True)
    parser.add_argument("--output-contract", required=True)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=DEFAULT_WARMUP_BARS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        contract = build_a50_sealed_split(
            universe_json=args.universe_json,
            parent_dir=args.parent_dir,
            training_dir=args.training_dir,
            evaluation_dir=args.evaluation_dir,
            requested_test_start=args.requested_test_start,
            output_contract=args.output_contract,
            warmup_bars=args.warmup_bars,
        )
    except SealedSplitError as exc:
        print(f"切分失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
