"""50 只 A 股封存样本外评估合同与一次性揭盲门禁。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


CAMPAIGN_FORMAT = "alphamaster_sealed_oos_campaign_v2"
RESULT_FORMAT = "alphamaster_sealed_oos_campaign_result_v2"
REVEAL_LOCK_FORMAT = "alphamaster_sealed_oos_reveal_lock_v2"
SEALED_REPORT_FORMAT = "alphamaster_sealed_oos_report_v2"
SEALED_DATASET_FORMAT = "alphamaster_sealed_oos_dataset_v2"
EVALUATION_MODE = "sealed_oos"
REVEAL_STARTED = "REVEAL_STARTED"
REQUIRED_SYMBOL_COUNT = 50
SHARPE_THRESHOLD = 1.0
REVEAL_REGISTRY_DIR = (
    Path.home() / ".alphamaster" / "sealed_oos_reveal_registry"
)

_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ITEM_KEYS = {
    "symbol",
    "data_sha256",
    "strategy_sha256",
    "test_start",
    "test_end",
    "report_path",
}
_COST_KEYS = {
    "commission_pct",
    "slippage_pct",
    "cost_rate",
}
_CONTRACT_KEYS = {
    "format",
    "campaign_id",
    "evaluation_mode",
    "costs",
    "sharpe_gate",
    "symbol_count",
    "sealed_dataset_sha256",
    "result_path",
    "items",
    "contract_sha256",
}


class SealedOOSCampaignError(ValueError):
    """封存样本外评估合同错误。"""


class ContractValidationError(SealedOOSCampaignError):
    """合同内容或身份无效。"""


class ResultAlreadyExistsError(SealedOOSCampaignError, FileExistsError):
    """同一封存合同已经揭盲，禁止覆盖或重测。"""


class RevealAlreadyStartedError(SealedOOSCampaignError, FileExistsError):
    """同一密封数据集已经开始揭盲，即使换合同或策略也禁止重试。"""


@dataclass(frozen=True)
class SealedOOSItem:
    """一个标的在封存测试中的不可变身份。"""

    symbol: str
    data_sha256: str
    strategy_sha256: str
    test_start: str
    test_end: str
    report_path: str | Path


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def normalize_cost_policy(
    *,
    commission_pct: Any,
    slippage_pct: Any,
) -> dict[str, float]:
    """规范化并验证封存样本外评估使用的单边交易成本。"""

    normalized: dict[str, float] = {}
    for field, value in (
        ("commission_pct", commission_pct),
        ("slippage_pct", slippage_pct),
    ):
        if type(value) not in {int, float}:
            raise ContractValidationError(f"{field} 必须是有限数字")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ContractValidationError(f"{field} 必须是有限数字")
        if numeric < 0:
            raise ContractValidationError(f"{field} 不能为负")
        normalized[field] = numeric

    cost_rate = (
        normalized["commission_pct"] + normalized["slippage_pct"]
    ) / 100.0
    if cost_rate <= 0:
        raise ContractValidationError(
            "sealed_oos 必须使用严格大于 0 的手续费与滑点总成本"
        )
    return {
        **normalized,
        "cost_rate": cost_rate,
    }


def _validate_cost_policy_payload(value: Any) -> dict[str, float]:
    if type(value) is not dict or set(value) != _COST_KEYS:
        raise ContractValidationError(
            f"costs 字段必须严格等于 {sorted(_COST_KEYS)}"
        )
    normalized = normalize_cost_policy(
        commission_pct=value["commission_pct"],
        slippage_pct=value["slippage_pct"],
    )
    raw_cost_rate = value["cost_rate"]
    if type(raw_cost_rate) not in {int, float} or not math.isfinite(
        float(raw_cost_rate)
    ):
        raise ContractValidationError("costs.cost_rate 必须是有限数字")
    if float(raw_cost_rate) != normalized["cost_rate"]:
        raise ContractValidationError(
            "costs.cost_rate 必须严格等于 (commission_pct + slippage_pct) / 100"
        )
    return normalized


def _sealed_dataset_sha256(items: list[dict[str, str]]) -> str:
    """只绑定 50 份密封文件的字节身份，任何重新解释都不能再次揭盲。"""

    identity = {
        "format": SEALED_DATASET_FORMAT,
        "evaluation_mode": EVALUATION_MODE,
        "symbol_count": REQUIRED_SYMBOL_COUNT,
        # 排序后保留重复值：锁只认物理密封数据集合，不认可被重新编排的
        # symbol、评分窗口、策略和路径，避免同一批字节换说法后再次窥视。
        "sealed_data_sha256": sorted(item["data_sha256"] for item in items),
    }
    return _payload_sha256(identity)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 包含非法非有限数字: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _load_json_bytes(content: bytes) -> Any:
    text = content.decode("utf-8")
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _validate_hash(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ContractValidationError(f"{field} 必须是 64 位小写十六进制 SHA-256")
    return value


def _validate_timestamp(value: Any, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ContractValidationError(f"{field} 必须是 UTC RFC3339 时间并以 Z 结尾")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractValidationError(f"{field} 不是合法 UTC RFC3339 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractValidationError(f"{field} 必须使用 UTC")
    canonical = parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
    if canonical != value:
        raise ContractValidationError(
            f"{field} 必须使用 YYYY-MM-DDTHH:MM:SSZ 的规范格式"
        )
    return value


def _normalize_frozen_path(value: Any, field: str) -> str:
    if isinstance(value, Path):
        value = str(value)
    if type(value) is not str or not value or value != value.strip():
        raise ContractValidationError(f"{field} 必须是非空且无首尾空格的路径")
    if "\x00" in value:
        raise ContractValidationError(f"{field} 不能包含空字符")
    normalized = value.replace("\\", "/")
    if normalized.endswith("/"):
        raise ContractValidationError(f"{field} 必须指向文件")
    parts = normalized.split("/")
    if "." in parts or ".." in parts:
        raise ContractValidationError(f"{field} 不能包含 . 或 .. 路径段")
    return normalized


def _contract_relative_path(value: str | Path, contract_path: Path, field: str) -> str:
    frozen = _normalize_frozen_path(value, field)
    candidate = Path(frozen)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        relative = candidate.relative_to(contract_path.parent.resolve())
    except ValueError:
        return candidate.as_posix()
    return relative.as_posix()


def _resolve_frozen_path(contract_path: Path, frozen_path: str) -> Path:
    candidate = Path(frozen_path)
    if candidate.is_absolute():
        return candidate
    return contract_path.parent / candidate


def _coerce_item(
    item: SealedOOSItem | Mapping[str, Any],
    *,
    contract_path: Path,
    index: int,
) -> dict[str, str]:
    if isinstance(item, SealedOOSItem):
        raw: Mapping[str, Any] = asdict(item)
    elif isinstance(item, Mapping):
        raw = item
    else:
        raise ContractValidationError(f"items[{index}] 必须是 SealedOOSItem 或映射")
    if set(raw) != _ITEM_KEYS:
        raise ContractValidationError(
            f"items[{index}] 字段必须严格等于 {sorted(_ITEM_KEYS)}"
        )

    symbol = raw["symbol"]
    if type(symbol) is not str or _SYMBOL_RE.fullmatch(symbol) is None:
        raise ContractValidationError(f"items[{index}].symbol 必须是 6 位数字")
    test_start = _validate_timestamp(raw["test_start"], f"items[{index}].test_start")
    test_end = _validate_timestamp(raw["test_end"], f"items[{index}].test_end")
    if test_start >= test_end:
        raise ContractValidationError(
            f"items[{index}] 的 test_start 必须早于 test_end"
        )
    return {
        "symbol": symbol,
        "data_sha256": _validate_hash(
            raw["data_sha256"], f"items[{index}].data_sha256"
        ),
        "strategy_sha256": _validate_hash(
            raw["strategy_sha256"], f"items[{index}].strategy_sha256"
        ),
        "test_start": test_start,
        "test_end": test_end,
        "report_path": _contract_relative_path(
            raw["report_path"], contract_path, f"items[{index}].report_path"
        ),
    }


def _validate_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(items) != REQUIRED_SYMBOL_COUNT:
        raise ContractValidationError(
            f"campaign 必须恰好包含 {REQUIRED_SYMBOL_COUNT} 个标的"
        )
    symbols = [item["symbol"] for item in items]
    if len(set(symbols)) != REQUIRED_SYMBOL_COUNT:
        raise ContractValidationError("campaign 的 50 个 A 股代码必须唯一")
    report_paths = [item["report_path"] for item in items]
    if len(set(report_paths)) != REQUIRED_SYMBOL_COUNT:
        raise ContractValidationError("campaign 的 50 个报告路径必须唯一")
    test_windows = {
        (item["test_start"], item["test_end"])
        for item in items
    }
    if len(test_windows) != 1:
        raise ContractValidationError(
            "campaign 的 50 个标的必须使用完全一致的 test_start/test_end"
        )
    return sorted(items, key=lambda item: item["symbol"])


def _contract_body(
    *,
    campaign_id: str,
    costs: Mapping[str, float],
    sealed_dataset_sha256: str,
    result_path: str,
    items: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "format": CAMPAIGN_FORMAT,
        "campaign_id": campaign_id,
        "evaluation_mode": EVALUATION_MODE,
        "costs": dict(costs),
        "sharpe_gate": {
            "metric": "sharpe",
            "operator": ">",
            "threshold": SHARPE_THRESHOLD,
        },
        "symbol_count": REQUIRED_SYMBOL_COUNT,
        "sealed_dataset_sha256": sealed_dataset_sha256,
        "result_path": result_path,
        "items": items,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ResultAlreadyExistsError(f"文件已存在，禁止覆盖: {path}") from exc


def _reveal_lock_path(sealed_dataset_sha256: str) -> Path:
    dataset_hash = _validate_hash(
        sealed_dataset_sha256, "sealed_dataset_sha256"
    )
    return REVEAL_REGISTRY_DIR / f"{dataset_hash}.reveal.lock"


def _claim_reveal(
    *,
    lock_path: Path,
    campaign_id: str,
    contract_sha256: str,
    sealed_dataset_sha256: str,
) -> None:
    """先完整落盘，再用硬链接原子占位，锁出现时内容必定完整。"""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": REVEAL_LOCK_FORMAT,
        "campaign_id": campaign_id,
        "contract_sha256": contract_sha256,
        "sealed_dataset_sha256": sealed_dataset_sha256,
        "status": REVEAL_STARTED,
    }
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    temporary_path = lock_path.with_name(
        f".{lock_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, lock_path)
        except FileExistsError as exc:
            raise RevealAlreadyStartedError(
                "同一密封数据集已经开始揭盲；禁止再次读取任何报告，"
                "也禁止换合同、路径或策略重试: "
                f"{lock_path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def create_sealed_oos_campaign(
    contract_path: str | Path,
    *,
    campaign_id: str,
    items: Iterable[SealedOOSItem | Mapping[str, Any]],
    commission_pct: float,
    slippage_pct: float,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    """创建不可覆盖的 50 标的封存合同，不读取或生成回测报告。"""

    path = Path(contract_path)
    if type(campaign_id) is not str or _CAMPAIGN_ID_RE.fullmatch(campaign_id) is None:
        raise ContractValidationError(
            "campaign_id 必须是 1-128 位字母、数字、点、下划线或连字符"
        )
    if result_path is None:
        frozen_result_path = f"{path.stem}.result.json"
    else:
        frozen_result_path = _contract_relative_path(
            result_path, path, "result_path"
        )

    normalized_items = _validate_items(
        [
            _coerce_item(item, contract_path=path, index=index)
            for index, item in enumerate(items)
        ]
    )
    costs = normalize_cost_policy(
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )
    sealed_dataset_sha256 = _sealed_dataset_sha256(normalized_items)
    if frozen_result_path in {item["report_path"] for item in normalized_items}:
        raise ContractValidationError("result_path 不能与任何 report_path 相同")
    resolved_result = _resolve_frozen_path(path, frozen_result_path)
    if resolved_result.exists():
        raise ResultAlreadyExistsError(
            f"结果文件已存在，不能为同一路径创建新 seal: {resolved_result}"
        )
    reveal_lock_path = _reveal_lock_path(sealed_dataset_sha256)
    if reveal_lock_path.exists():
        raise RevealAlreadyStartedError(
            f"同一密封数据集已经揭盲，不能创建新 seal: {reveal_lock_path}"
        )

    body = _contract_body(
        campaign_id=campaign_id,
        costs=costs,
        sealed_dataset_sha256=sealed_dataset_sha256,
        result_path=frozen_result_path,
        items=normalized_items,
    )
    contract = {**body, "contract_sha256": _payload_sha256(body)}
    _write_json_exclusive(path, contract)
    return contract


def _validate_loaded_contract(payload: Any) -> dict[str, Any]:
    if type(payload) is not dict:
        raise ContractValidationError("campaign 合同必须是 JSON 对象")
    if set(payload) != _CONTRACT_KEYS:
        raise ContractValidationError(
            f"campaign 合同字段必须严格等于 {sorted(_CONTRACT_KEYS)}"
        )
    if payload["format"] != CAMPAIGN_FORMAT:
        raise ContractValidationError("campaign format 不受支持")
    campaign_id = payload["campaign_id"]
    if type(campaign_id) is not str or _CAMPAIGN_ID_RE.fullmatch(campaign_id) is None:
        raise ContractValidationError("campaign_id 无效")
    if payload["evaluation_mode"] != EVALUATION_MODE:
        raise ContractValidationError("campaign evaluation_mode 必须是 sealed_oos")
    costs = _validate_cost_policy_payload(payload["costs"])
    if payload["symbol_count"] != REQUIRED_SYMBOL_COUNT:
        raise ContractValidationError("campaign symbol_count 必须是 50")
    if payload["sharpe_gate"] != {
        "metric": "sharpe",
        "operator": ">",
        "threshold": SHARPE_THRESHOLD,
    }:
        raise ContractValidationError("campaign Sharpe 门禁必须严格为 > 1.0")
    result_path = _normalize_frozen_path(payload["result_path"], "result_path")
    raw_items = payload["items"]
    if type(raw_items) is not list:
        raise ContractValidationError("campaign items 必须是列表")
    items = _validate_items(
        [
            _coerce_item(item, contract_path=Path("."), index=index)
            for index, item in enumerate(raw_items)
        ]
    )
    if raw_items != items:
        raise ContractValidationError("campaign items 必须按 symbol 升序保存")
    if result_path in {item["report_path"] for item in items}:
        raise ContractValidationError("result_path 不能与任何 report_path 相同")
    expected_dataset_hash = _sealed_dataset_sha256(items)
    sealed_dataset_sha256 = _validate_hash(
        payload["sealed_dataset_sha256"], "sealed_dataset_sha256"
    )
    if sealed_dataset_sha256 != expected_dataset_hash:
        raise ContractValidationError(
            "sealed_dataset_sha256 与 50 只标的的密封数据身份不一致"
        )

    body = _contract_body(
        campaign_id=campaign_id,
        costs=costs,
        sealed_dataset_sha256=sealed_dataset_sha256,
        result_path=result_path,
        items=items,
    )
    expected_hash = _payload_sha256(body)
    actual_hash = _validate_hash(payload["contract_sha256"], "contract_sha256")
    if actual_hash != expected_hash:
        raise ContractValidationError("campaign contract_sha256 与合同内容不一致")
    return {**body, "contract_sha256": actual_hash}


def load_sealed_oos_campaign(contract_path: str | Path) -> dict[str, Any]:
    """读取并验证合同自身身份。"""

    path = Path(contract_path)
    try:
        payload = _load_json_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractValidationError(f"无法读取合法 campaign 合同: {path}") from exc
    return _validate_loaded_contract(payload)


def _read_and_evaluate_report(
    *,
    contract_path: Path,
    item: Mapping[str, str],
    costs: Mapping[str, float],
) -> dict[str, Any]:
    report_path = _resolve_frozen_path(contract_path, item["report_path"])
    result: dict[str, Any] = {
        "symbol": item["symbol"],
        "report_path": item["report_path"],
        "report_sha256": None,
        "sharpe": None,
        "status": "FAIL",
        "failure_codes": [],
    }
    failures: list[str] = result["failure_codes"]
    try:
        content = report_path.read_bytes()
    except FileNotFoundError:
        failures.append("report_missing")
        return result
    except OSError:
        failures.append("report_unreadable")
        return result

    result["report_sha256"] = hashlib.sha256(content).hexdigest()
    try:
        report = _load_json_bytes(content)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        failures.append("report_invalid_json")
        return result
    if type(report) is not dict:
        failures.append("report_not_object")
        return result

    identity_fields = (
        "format",
        "symbol",
        "data_sha256",
        "strategy_sha256",
        "evaluation_mode",
        "test_start",
        "test_end",
    )
    expected = {
        "format": SEALED_REPORT_FORMAT,
        "symbol": item["symbol"],
        "data_sha256": item["data_sha256"],
        "strategy_sha256": item["strategy_sha256"],
        "evaluation_mode": EVALUATION_MODE,
        "test_start": item["test_start"],
        "test_end": item["test_end"],
    }
    for field in identity_fields:
        if field not in report:
            failures.append(f"{field}_missing")
        elif type(report[field]) is not str or report[field] != expected[field]:
            failures.append(f"{field}_mismatch")

    for field in _COST_KEYS:
        if field not in report:
            failures.append(f"{field}_missing")
            continue
        value = report[field]
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            failures.append(f"{field}_not_finite")
        elif float(value) != costs[field]:
            failures.append(f"{field}_mismatch")

    if "sharpe" not in report:
        failures.append("sharpe_missing")
    else:
        sharpe = report["sharpe"]
        if type(sharpe) not in {int, float} or not math.isfinite(float(sharpe)):
            failures.append("sharpe_not_finite")
        else:
            numeric_sharpe = float(sharpe)
            result["sharpe"] = numeric_sharpe
            if numeric_sharpe <= SHARPE_THRESHOLD:
                failures.append("sharpe_not_above_1")

    if not failures:
        result["status"] = "PASS"
    return result


def evaluate_sealed_oos_campaign(
    contract_path: str | Path,
) -> dict[str, Any]:
    """一次性读取全 50 份报告并发布不可覆盖的整批门禁结果。"""

    path = Path(contract_path)
    contract = load_sealed_oos_campaign(path)
    result_path = _resolve_frozen_path(path, contract["result_path"])
    if result_path.exists():
        raise ResultAlreadyExistsError(
            f"campaign 已经揭盲，禁止覆盖或在同一 seal 上重测: {result_path}"
        )
    _claim_reveal(
        lock_path=_reveal_lock_path(contract["sealed_dataset_sha256"]),
        campaign_id=contract["campaign_id"],
        contract_sha256=contract["contract_sha256"],
        sealed_dataset_sha256=contract["sealed_dataset_sha256"],
    )

    # 不因单只失败提前返回，保证一次揭盲覆盖合同中的全部 50 份报告。
    symbol_results = [
        _read_and_evaluate_report(
            contract_path=path,
            item=item,
            costs=contract["costs"],
        )
        for item in contract["items"]
    ]
    pass_count = sum(row["status"] == "PASS" for row in symbol_results)
    finite_sharpes = [
        row["sharpe"] for row in symbol_results if row["sharpe"] is not None
    ]
    minimum_sharpe = (
        min(finite_sharpes)
        if len(finite_sharpes) == REQUIRED_SYMBOL_COUNT
        else None
    )
    batch_passed = pass_count == REQUIRED_SYMBOL_COUNT
    result = {
        "format": RESULT_FORMAT,
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "sealed_dataset_sha256": contract["sealed_dataset_sha256"],
        "evaluation_mode": EVALUATION_MODE,
        "costs": contract["costs"],
        "status": "PASS" if batch_passed else "FAIL",
        "symbol_count": REQUIRED_SYMBOL_COUNT,
        "pass_count": pass_count,
        "minimum_sharpe": minimum_sharpe,
        "sharpe_gate": contract["sharpe_gate"],
        "results": symbol_results,
    }
    _write_json_exclusive(result_path, result)
    return result


__all__ = [
    "CAMPAIGN_FORMAT",
    "EVALUATION_MODE",
    "REQUIRED_SYMBOL_COUNT",
    "REVEAL_LOCK_FORMAT",
    "REVEAL_STARTED",
    "RESULT_FORMAT",
    "SEALED_DATASET_FORMAT",
    "SEALED_REPORT_FORMAT",
    "SHARPE_THRESHOLD",
    "ContractValidationError",
    "RevealAlreadyStartedError",
    "ResultAlreadyExistsError",
    "SealedOOSCampaignError",
    "SealedOOSItem",
    "create_sealed_oos_campaign",
    "evaluate_sealed_oos_campaign",
    "load_sealed_oos_campaign",
    "normalize_cost_policy",
]
