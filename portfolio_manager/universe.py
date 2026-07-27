"""把冻结股票池合同转换为组合控制器可审计的身份。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.freeze_csi_a50_universe import load_frozen_universe

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CSI_A50_UNIVERSE_PATH = (
    _PROJECT_ROOT / "universes" / "csi_a50_20260723.json"
)
_CSI_A50_CONTRACT_SHA256 = (
    "987387945fba0cb778b648860bc7579a3cf49e9c3b788596464f714e968bb896"
)
_DATE_RE = re.compile(r"^[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_HISTORY_DATA_FILE_RE = re.compile(r"^[0-9]{8}\.parquet$")
_CSI300_HISTORY_FORMAT = "free_stockdb_csi300_weight_history_v1"
_CSI300_INDEX = "000300.XSHG"
_CSI300_HISTORY_TRUST_POLICY = "alphamaster_csi300_history_roots_v1"
_CSI300_HISTORY_TRUSTED_ROOTS: dict[
    str,
    dict[str, dict[str, object]],
] = {
    _CSI300_HISTORY_TRUST_POLICY: {
        "1a7f10aa803ab2e3656c449c4cb6dec5b6da0f77df5d76b886dac6a4504ffcc3": {
            "format": _CSI300_HISTORY_FORMAT,
            "status": "completed",
            "endpoint": "8.138.149.215:12328",
            "sdk_sha256": (
                "3535c7714218d749912a98286e6fcd6882cc70b3fb500b08cb32acd8ba8913cb"
            ),
            "index": _CSI300_INDEX,
            "request_count": 255,
            "successful_api_calls": 255,
            "first_requested_date": "2005-04-30",
            "last_requested_date": "2026-06-30",
            "first_actual_weight_date": "2005-04-29",
            "last_actual_weight_date": "2026-06-30",
            "unique_actual_weight_dates": 255,
            "total_rows": 76_498,
            "combined_file": "csi300_weight_history.parquet",
            "combined_bytes": 157_902,
            "combined_sha256": (
                "68e68f13755c45e7e2981db0ca6bc0536e8363929ff4880bdb1afb678092dd8e"
            ),
        }
    }
}
_CSI300_HISTORY_AVAILABILITY_POLICIES: dict[
    str,
    dict[str, object],
] = {
    _CSI300_HISTORY_TRUST_POLICY: {
        "incomplete_snapshots": {
            "2009-12-31": {
                "source_data_sha256": (
                    "8df75db03b6fbacb1bf49b7c75b32aadab4091265943035a949dd45d3e7fb718"
                ),
                "source_receipt_sha256": (
                    "8ed39cf44ef025e1c5031170cd234da43c350d0ae628c58193e1cbd6324887de"
                ),
                "constituent_count": 298,
                "valid_until_exclusive": "2010-01-29",
                "reason": "受信快照只有 298 只成分，不能作为沪深300股票池",
            }
        }
    }
}

UNIVERSE_QUERY_MODE_STRICT = "strict"
UNIVERSE_QUERY_MODE_RECONSTRUCTED = "reconstructed"
UNIVERSE_QUERY_MODE_STATIC = "static"
UNIVERSE_QUERY_MODE_UNTRUSTED = "untrusted"
UNIVERSE_CONTRACT_TYPE_HISTORICAL = "historical"
UNIVERSE_CONTRACT_TYPE_TRUSTED_STATIC = "trusted_static"
UNIVERSE_CONTRACT_TYPE_UNTRUSTED = "untrusted"
_UNIVERSE_QUERY_MODES = frozenset(
    {
        UNIVERSE_QUERY_MODE_STRICT,
        UNIVERSE_QUERY_MODE_RECONSTRUCTED,
    }
)
_HISTORICAL_SELECTION_FORMAT = "alphamaster_historical_universe_selection_v1"


class UniverseAvailabilityError(ValueError):
    """股票池缺少查询时点可知证据。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label}不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}顶层必须是对象")
    return payload


def _iso_date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是 YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label}必须是真实的 YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label}必须是规范 YYYY-MM-DD")
    return value


def _utc_timestamp(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}必须是带时区的 ISO 时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}必须是带时区的 ISO 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label}必须带时区")
    utc = parsed.astimezone(timezone.utc)
    canonical = utc.isoformat().replace("+00:00", "Z")
    return canonical, utc


def _receipt_effective_date(receipt: dict[str, Any]) -> str:
    actual = receipt.get("actual_weight_date")
    explicit = receipt.get("source_effective_date")
    if explicit is not None and actual is not None and explicit != actual:
        raise ValueError(
            "receipt 的 source_effective_date 与 actual_weight_date 不一致"
        )
    return _iso_date(
        explicit if explicit is not None else actual,
        "source_effective_date/actual_weight_date",
    )


def _receipt_observation_times(
    receipt: dict[str, Any],
) -> tuple[str, str, str | None]:
    captured_at = receipt.get("captured_at_utc")
    observed_raw = receipt.get("observed_at", captured_at)
    receipt_raw = receipt.get("receipt_at", captured_at)
    observed_at, _ = _utc_timestamp(observed_raw, "observed_at")
    receipt_at, _ = _utc_timestamp(receipt_raw, "receipt_at")
    strict_raw = receipt.get("strict_available_at")
    if strict_raw is None:
        return observed_at, receipt_at, None
    strict_available_at, _ = _utc_timestamp(
        strict_raw,
        "strict_available_at",
    )
    return observed_at, receipt_at, strict_available_at


def _load_trusted_history_manifest(
    root: Path,
    trust_policy: str | None,
) -> tuple[dict[str, Any], str]:
    if (
        not isinstance(trust_policy, str)
        or not trust_policy
        or trust_policy not in _CSI300_HISTORY_TRUSTED_ROOTS
    ):
        raise UniverseAvailabilityError("历史权重缺少代码内注册的受信根锚")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"历史权重 manifest 不存在: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        root_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("历史权重 manifest 不是合法 UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("历史权重 manifest 顶层必须是对象")
    expected = _CSI300_HISTORY_TRUSTED_ROOTS[trust_policy].get(root_sha256)
    if expected is None:
        raise UniverseAvailabilityError(
            "历史权重根 manifest SHA-256 不在代码受信锚中"
        )
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"历史权重根 manifest 的 {field} 与受信锚不匹配")

    items = manifest.get("items")
    request_count = manifest.get("request_count")
    successful_api_calls = manifest.get("successful_api_calls")
    if (
        not isinstance(items, list)
        or not items
        or isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count != len(items)
        or successful_api_calls != request_count
    ):
        raise ValueError("历史权重根 manifest 的请求数量与 items 不一致")
    requested_dates: list[str] = []
    effective_dates: list[str] = []
    total_rows = 0
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"历史权重 items[{position}] 必须是对象")
        requested_dates.append(
            _iso_date(
                item.get("requested_date"),
                f"items[{position}].requested_date",
            )
        )
        effective_dates.append(_receipt_effective_date(item))
        rows = item.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"历史权重 items[{position}].rows 非法")
        total_rows += rows
    if (
        len(set(requested_dates)) != request_count
        or len(set(effective_dates)) != manifest.get("unique_actual_weight_dates")
        or min(requested_dates) != manifest.get("first_requested_date")
        or max(requested_dates) != manifest.get("last_requested_date")
        or min(effective_dates) != manifest.get("first_actual_weight_date")
        or max(effective_dates) != manifest.get("last_actual_weight_date")
        or total_rows != manifest.get("total_rows")
    ):
        raise ValueError("历史权重根 manifest 的日期或总行数汇总不一致")

    combined_file = manifest.get("combined_file")
    if combined_file != "csi300_weight_history.parquet":
        raise ValueError("历史权重 combined_file 身份不匹配")
    combined_path = root / combined_file
    if (
        not combined_path.is_file()
        or combined_path.stat().st_size != manifest.get("combined_bytes")
        or _sha256_file(combined_path) != manifest.get("combined_sha256")
    ):
        raise ValueError("历史权重合并 Parquet 与受信根身份不匹配")
    return manifest, root_sha256


@dataclass(frozen=True)
class _HistoricalSelectionEvidence:
    history_root: Path
    root_manifest_sha256: str
    query_date: date
    requested_date: str
    effective_date: str
    receipt: dict[str, Any]
    receipt_path: Path
    data_path: Path
    data_sha256: str
    receipt_sha256: str
    observed_at: str
    receipt_at: str
    strict_available_at: str | None


def _raise_if_incomplete_snapshot(
    *,
    trust_policy: str,
    query_date: date,
    effective_date: str,
    data_sha256: str,
    receipt_sha256: str,
    constituent_count: object,
) -> None:
    policy = _CSI300_HISTORY_AVAILABILITY_POLICIES.get(
        trust_policy,
        {},
    )
    if not isinstance(policy, dict):
        raise ValueError("历史权重 availability policy 配置非法")
    incomplete_snapshots = policy.get("incomplete_snapshots", {})
    if not isinstance(incomplete_snapshots, dict):
        raise ValueError("历史权重 incomplete_snapshots 配置非法")
    snapshot_policy = incomplete_snapshots.get(effective_date)
    if snapshot_policy is None:
        return
    if not isinstance(snapshot_policy, dict):
        raise ValueError("历史权重不完整快照策略非法")
    valid_until_text = _iso_date(
        snapshot_policy.get("valid_until_exclusive"),
        "incomplete snapshot valid_until_exclusive",
    )
    valid_until = date.fromisoformat(valid_until_text)
    effective = date.fromisoformat(effective_date)
    reason = snapshot_policy.get("reason")
    expected_count = snapshot_policy.get("constituent_count")
    if (
        valid_until <= effective
        or not isinstance(reason, str)
        or not reason.strip()
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or constituent_count != expected_count
        or snapshot_policy.get("source_data_sha256") != data_sha256
        or snapshot_policy.get("source_receipt_sha256") != receipt_sha256
    ):
        raise ValueError("历史权重不完整快照策略与受信证据不一致")
    if query_date < valid_until:
        raise UniverseAvailabilityError(
            "受信历史快照被标记为不完整，"
            f"不可用区间为 [{effective_date}, {valid_until_text})：{reason}"
        )
    raise UniverseAvailabilityError(
        "受信历史快照不完整，且 valid_until_exclusive 后没有可用替代快照"
    )


def _derive_historical_selection_evidence(
    history_root: str | Path,
    *,
    as_of_date: str,
    trust_policy: str,
) -> _HistoricalSelectionEvidence:
    query_date = date.fromisoformat(_iso_date(as_of_date, "as_of_date"))
    root = Path(history_root).resolve()
    root_manifest, root_manifest_sha256 = _load_trusted_history_manifest(
        root,
        trust_policy,
    )
    if (
        root_manifest.get("format") != _CSI300_HISTORY_FORMAT
        or root_manifest.get("status") != "completed"
        or root_manifest.get("index") != _CSI300_INDEX
    ):
        raise ValueError("历史权重 manifest 的格式、状态或指数身份不匹配")
    raw_items = root_manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("历史权重 manifest 的 items 必须是非空列表")

    normalized: list[tuple[date, str, dict[str, Any]]] = []
    seen_effective_dates: set[date] = set()
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"历史权重 items[{position}] 必须是对象")
        if raw_item.get("format") != _CSI300_HISTORY_FORMAT:
            raise ValueError(f"历史权重 items[{position}] 的 format 不匹配")
        requested = _iso_date(
            raw_item.get("requested_date"),
            f"items[{position}].requested_date",
        )
        effective_text = _receipt_effective_date(raw_item)
        effective = date.fromisoformat(effective_text)
        if effective > date.fromisoformat(requested):
            raise ValueError("source_effective_date 不得晚于 requested_date")
        if effective in seen_effective_dates:
            raise ValueError("历史权重 source_effective_date 不能重复")
        seen_effective_dates.add(effective)
        normalized.append((effective, requested, raw_item))
    normalized.sort(key=lambda row: row[0])

    eligible = [row for row in normalized if row[0] <= query_date]
    if not eligible:
        raise UniverseAvailabilityError("查询日期早于首个历史权重时点")
    _, requested, root_receipt = eligible[-1]
    stem = requested.replace("-", "")
    data_file = root_receipt.get("data_file")
    if (
        not isinstance(data_file, str)
        or _HISTORY_DATA_FILE_RE.fullmatch(data_file) is None
        or data_file != f"{stem}.parquet"
    ):
        raise ValueError("历史权重 data_file 与 requested_date 不匹配")
    receipt_path = root / "manifests" / f"{stem}.json"
    receipt = _read_json_object(receipt_path, "历史权重 receipt")
    if receipt != root_receipt:
        raise ValueError("逐时点 receipt 与根 manifest 条目不一致")
    effective_text = _receipt_effective_date(receipt)
    observed_at, receipt_at, strict_available_at = (
        _receipt_observation_times(receipt)
    )
    data_sha256 = receipt.get("data_sha256")
    if (
        not isinstance(data_sha256, str)
        or _SHA256_RE.fullmatch(data_sha256) is None
    ):
        raise ValueError("历史权重 receipt 的 data_sha256 非法")
    receipt_sha256 = _sha256_file(receipt_path)
    _raise_if_incomplete_snapshot(
        trust_policy=trust_policy,
        query_date=query_date,
        effective_date=effective_text,
        data_sha256=data_sha256,
        receipt_sha256=receipt_sha256,
        constituent_count=receipt.get("rows"),
    )
    return _HistoricalSelectionEvidence(
        history_root=root,
        root_manifest_sha256=root_manifest_sha256,
        query_date=query_date,
        requested_date=requested,
        effective_date=effective_text,
        receipt=receipt,
        receipt_path=receipt_path,
        data_path=root / "data" / data_file,
        data_sha256=data_sha256,
        receipt_sha256=receipt_sha256,
        observed_at=observed_at,
        receipt_at=receipt_at,
        strict_available_at=strict_available_at,
    )


def _load_historical_constituents(
    evidence: _HistoricalSelectionEvidence,
) -> tuple[tuple[str, ...], tuple[WeightedUniverseConstituent, ...]]:
    data_path = evidence.data_path
    receipt = evidence.receipt
    if (
        not data_path.is_file()
        or _sha256_file(data_path) != evidence.data_sha256
    ):
        raise ValueError("历史权重 Parquet SHA-256 不匹配")
    expected_bytes = receipt.get("data_bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != data_path.stat().st_size
    ):
        raise ValueError("历史权重 Parquet 字节数不匹配")

    frame = pd.read_parquet(data_path)
    expected_columns = ["code", "date", "weight", "display_name"]
    if list(frame.columns) != expected_columns or frame.empty:
        raise ValueError("历史权重 Parquet 列合同不匹配或为空")
    expected_rows = receipt.get("rows")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows != len(frame)
    ):
        raise ValueError("历史权重 Parquet 行数与 receipt 不匹配")
    if not frame["code"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("历史权重代码必须保留为 6 位数字文本")
    symbols = tuple(frame["code"].tolist())
    if (
        any(_SYMBOL_RE.fullmatch(symbol) is None for symbol in symbols)
        or len(set(symbols)) != len(symbols)
    ):
        raise ValueError("历史权重代码格式非法或重复")
    actual_dates = pd.to_datetime(frame["date"], errors="raise").dt.date
    if (
        actual_dates.nunique() != 1
        or actual_dates.iloc[0].isoformat() != evidence.effective_date
    ):
        raise ValueError("历史权重 Parquet 日期与 source_effective_date 不匹配")
    weights = pd.to_numeric(frame["weight"], errors="raise").astype(float)
    if (
        not weights.map(math.isfinite).all()
        or not weights.gt(0).all()
        or not 99.0 <= float(weights.sum()) <= 101.0
    ):
        raise ValueError("历史权重必须为正有限数且合计位于 [99, 101]")
    names = frame["display_name"]
    if not names.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("历史权重 display_name 必须是非空文本")
    constituents = tuple(
        WeightedUniverseConstituent(
            symbol=symbol,
            weight=float(weight),
            display_name=display_name,
        )
        for symbol, weight, display_name in zip(
            symbols,
            weights.tolist(),
            names.tolist(),
            strict=True,
        )
    )
    return symbols, constituents


def _derive_historical_query_gates(
    *,
    mode: str,
    query_at: str | None,
    strict_available_at: str | None,
) -> tuple[str | None, bool, bool, bool]:
    if mode not in _UNIVERSE_QUERY_MODES:
        raise ValueError("mode 必须显式为 strict 或 reconstructed")
    if mode == UNIVERSE_QUERY_MODE_RECONSTRUCTED:
        return None, True, False, False
    if query_at is None:
        raise UniverseAvailabilityError("strict 查询必须显式提供 query_at")
    canonical_query_at, query_timestamp = _utc_timestamp(query_at, "query_at")
    if strict_available_at is None:
        raise UniverseAvailabilityError(
            "该历史时点没有 strict_available_at；observed_at/receipt_at "
            "只是事后抓取时间，不能证明当时可知"
        )
    _, strict_timestamp = _utc_timestamp(
        strict_available_at,
        "strict_available_at",
    )
    if strict_timestamp > query_timestamp:
        raise UniverseAvailabilityError(
            "该股票池在 query_at 时尚无严格可知证据"
        )
    return canonical_query_at, False, True, True


@dataclass(frozen=True, kw_only=True)
class UniverseContract:
    """组合决策必须绑定的冻结股票池身份。"""

    universe_id: str
    snapshot_date: str
    constituent_count: int
    universe_sha256: str
    symbols: tuple[str, ...]
    contract_type: str = UNIVERSE_CONTRACT_TYPE_UNTRUSTED
    query_mode: str = UNIVERSE_QUERY_MODE_UNTRUSTED
    point_in_time_safe: bool = False
    sealed_oos_eligible: bool = False
    provenance_identity: str = UNIVERSE_CONTRACT_TYPE_UNTRUSTED
    contract_sha256: str | None = None

    def __post_init__(self) -> None:
        self._validate_base_semantics()
        if type(self) is UniverseContract:
            self._validate_contract_sha256()

    def _validate_base_semantics(self) -> None:
        """无副作用重验基础字段、类型和资格门禁。"""
        if not isinstance(self.universe_id, str) or not self.universe_id.strip():
            raise ValueError("universe_id 必须是非空文本")
        if (
            not isinstance(self.snapshot_date, str)
            or _DATE_RE.fullmatch(self.snapshot_date) is None
        ):
            raise ValueError("snapshot_date 必须是 YYYYMMDD")
        if (
            isinstance(self.constituent_count, bool)
            or not isinstance(self.constituent_count, int)
            or self.constituent_count <= 0
        ):
            raise ValueError("constituent_count 必须是正整数")
        if (
            not isinstance(self.universe_sha256, str)
            or _SHA256_RE.fullmatch(self.universe_sha256) is None
        ):
            raise ValueError("universe_sha256 必须是小写 SHA-256")
        if not isinstance(self.symbols, tuple):
            raise ValueError("symbols 必须是不可变元组")
        if len(self.symbols) != self.constituent_count:
            raise ValueError("constituent_count 与 symbols 数量不一致")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("冻结股票池不能包含重复代码")
        if any(
            not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None
            for symbol in self.symbols
        ):
            raise ValueError("冻结股票池代码必须是 6 位数字文本")
        if not isinstance(self.point_in_time_safe, bool) or not isinstance(
            self.sealed_oos_eligible,
            bool,
        ):
            raise ValueError("PIT 与 sealed eligibility 必须是布尔值")
        if self.contract_type == UNIVERSE_CONTRACT_TYPE_UNTRUSTED:
            if (
                self.query_mode != UNIVERSE_QUERY_MODE_UNTRUSTED
                or self.point_in_time_safe
                or self.sealed_oos_eligible
                or self.provenance_identity != UNIVERSE_CONTRACT_TYPE_UNTRUSTED
            ):
                raise ValueError("untrusted 股票池不能声明 PIT、sealed 或受信来源")
        elif self.contract_type == UNIVERSE_CONTRACT_TYPE_TRUSTED_STATIC:
            trusted_payload = load_frozen_universe(_CSI_A50_UNIVERSE_PATH)
            trusted_symbols = tuple(
                row["symbol"] for row in trusted_payload["constituents"]
            )
            if (
                self.query_mode != UNIVERSE_QUERY_MODE_STATIC
                or not self.point_in_time_safe
                or not self.sealed_oos_eligible
                or self.universe_sha256 != _CSI_A50_CONTRACT_SHA256
                or self.provenance_identity != _CSI_A50_CONTRACT_SHA256
                or self.universe_id
                != (
                    f"{trusted_payload['format']}:"
                    f"{trusted_payload['index_code']}:"
                    f"{trusted_payload['snapshot_date']}"
                )
                or self.snapshot_date != trusted_payload["snapshot_date"]
                or self.constituent_count
                != trusted_payload["constituent_count"]
                or self.symbols != trusted_symbols
            ):
                raise ValueError("trusted_static 股票池合同门禁不一致")
        elif self.contract_type == UNIVERSE_CONTRACT_TYPE_HISTORICAL:
            if type(self) is UniverseContract:
                raise ValueError("历史股票池必须使用不可剥离的历史合同")
        else:
            raise ValueError("contract_type 非法")

    def _contract_identity_payload(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "snapshot_date": self.snapshot_date,
            "constituent_count": self.constituent_count,
            "universe_sha256": self.universe_sha256,
            "symbols": list(self.symbols),
            "contract_type": self.contract_type,
            "query_mode": self.query_mode,
            "point_in_time_safe": self.point_in_time_safe,
            "sealed_oos_eligible": self.sealed_oos_eligible,
            "provenance_identity": self.provenance_identity,
        }

    def _validate_contract_sha256(self) -> None:
        expected = _canonical_sha256(self._contract_identity_payload())
        if self.contract_sha256 is None:
            object.__setattr__(self, "contract_sha256", expected)
        elif (
            not isinstance(self.contract_sha256, str)
            or _SHA256_RE.fullmatch(self.contract_sha256) is None
            or self.contract_sha256 != expected
        ):
            raise ValueError("contract_sha256 与 canonical 合同内容不一致")

    def _validate_existing_contract_sha256(self) -> None:
        expected = _canonical_sha256(self._contract_identity_payload())
        if self.contract_sha256 != expected:
            raise ValueError("contract_sha256 与 canonical 合同内容不一致")

    def validate_contract_identity(self) -> None:
        """重新验证不可变合同，供 controller/ledger 在边界处失败关闭。"""
        self._validate_base_semantics()
        self._validate_existing_contract_sha256()

    def to_dict(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "snapshot_date": self.snapshot_date,
            "constituent_count": self.constituent_count,
            "universe_sha256": self.universe_sha256,
            "symbols": list(self.symbols),
            "contract_type": self.contract_type,
            "query_mode": self.query_mode,
            "point_in_time_safe": self.point_in_time_safe,
            "sealed_oos_eligible": self.sealed_oos_eligible,
            "provenance_identity": self.provenance_identity,
            "contract_sha256": self.contract_sha256,
        }


@dataclass(frozen=True)
class WeightedUniverseConstituent:
    """历史股票池中与来源字节绑定的一只成分。"""

    symbol: str
    weight: float
    display_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "weight": self.weight,
            "display_name": self.display_name,
        }


@dataclass(frozen=True, kw_only=True)
class HistoricalUniverseContract(UniverseContract):
    """可直接交给组合控制器、且不能剥离时点来源门禁的股票池合同。"""

    as_of_date: str
    source_trust_policy: str
    source_effective_date: str
    source_effective_until_exclusive: str | None
    observed_at: str
    receipt_at: str
    strict_available_at: str | None
    reconstructed: bool
    source_data_sha256: str
    source_receipt_sha256: str
    constituents: tuple[WeightedUniverseConstituent, ...]
    source_history_root: str | None = None
    query_at: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _iso_date(self.as_of_date, "as_of_date")
        _iso_date(self.source_effective_date, "source_effective_date")
        if self.contract_type != UNIVERSE_CONTRACT_TYPE_HISTORICAL:
            raise ValueError("历史合同 contract_type 非法")
        if self.query_mode not in _UNIVERSE_QUERY_MODES:
            raise ValueError("历史合同 mode 非法")
        if (
            not isinstance(self.source_trust_policy, str)
            or self.source_trust_policy not in _CSI300_HISTORY_TRUSTED_ROOTS
        ):
            raise ValueError("历史合同缺少代码内注册的 source_trust_policy")
        if (
            not isinstance(self.provenance_identity, str)
            or _SHA256_RE.fullmatch(self.provenance_identity) is None
        ):
            raise ValueError("历史合同 provenance_identity 必须是小写 SHA-256")
        expected_reconstructed = (
            self.query_mode == UNIVERSE_QUERY_MODE_RECONSTRUCTED
        )
        if (
            self.reconstructed is not expected_reconstructed
            or self.point_in_time_safe is expected_reconstructed
            or self.sealed_oos_eligible is expected_reconstructed
        ):
            raise ValueError("历史合同的 mode/PIT/sealed 门禁不一致")
        if self.query_mode == UNIVERSE_QUERY_MODE_STRICT:
            if self.strict_available_at is None:
                raise ValueError("strict 历史合同必须携带 strict_available_at")
            _utc_timestamp(self.strict_available_at, "strict_available_at")
        if self.source_effective_until_exclusive is not None:
            _iso_date(
                self.source_effective_until_exclusive,
                "source_effective_until_exclusive",
            )
        _utc_timestamp(self.observed_at, "observed_at")
        _utc_timestamp(self.receipt_at, "receipt_at")
        for label, digest in (
            ("source_data_sha256", self.source_data_sha256),
            ("source_receipt_sha256", self.source_receipt_sha256),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} 必须是小写 SHA-256")
        if (
            not isinstance(self.constituents, tuple)
            or tuple(row.symbol for row in self.constituents) != self.symbols
        ):
            raise ValueError("历史合同 constituents 与 symbols 不一致")
        self._validate_trusted_semantics()
        self._validate_contract_sha256()

    @property
    def mode(self) -> str:
        return self.query_mode

    def _validate_trusted_semantics(self) -> None:
        if (
            not isinstance(self.source_history_root, str)
            or not self.source_history_root
        ):
            raise ValueError("历史合同缺少可重验的受信历史根")
        canonical_root = str(Path(self.source_history_root).resolve())
        if self.source_history_root != canonical_root:
            raise ValueError("source_history_root 必须是规范绝对路径")
        evidence = _derive_historical_selection_evidence(
            canonical_root,
            as_of_date=self.as_of_date,
            trust_policy=self.source_trust_policy,
        )
        symbols, constituents = _load_historical_constituents(evidence)
        (
            canonical_query_at,
            reconstructed,
            point_in_time_safe,
            sealed_oos_eligible,
        ) = _derive_historical_query_gates(
            mode=self.query_mode,
            query_at=self.query_at,
            strict_available_at=evidence.strict_available_at,
        )
        provenance_identity = _canonical_sha256(
            {
                "contract_type": UNIVERSE_CONTRACT_TYPE_HISTORICAL,
                "source_trust_policy": self.source_trust_policy,
                "source_index": _CSI300_INDEX,
                "source_effective_date": evidence.effective_date,
                "source_data_sha256": evidence.data_sha256,
                "source_receipt_sha256": evidence.receipt_sha256,
            }
        )
        universe_sha256 = _canonical_sha256(
            {
                "format": _HISTORICAL_SELECTION_FORMAT,
                "index": _CSI300_INDEX,
                "source_trust_policy": self.source_trust_policy,
                "as_of_date": evidence.query_date.isoformat(),
                "mode": self.query_mode,
                "query_at": canonical_query_at,
                "source_effective_date": evidence.effective_date,
                "source_effective_until_exclusive": None,
                "observed_at": evidence.observed_at,
                "receipt_at": evidence.receipt_at,
                "strict_available_at": evidence.strict_available_at,
                "reconstructed": reconstructed,
                "point_in_time_safe": point_in_time_safe,
                "sealed_oos_eligible": sealed_oos_eligible,
                "source_data_sha256": evidence.data_sha256,
                "source_receipt_sha256": evidence.receipt_sha256,
                "constituents": [row.to_dict() for row in constituents],
            }
        )
        expected_universe_id = (
            f"{_CSI300_HISTORY_FORMAT}:{_CSI300_INDEX}:"
            f"{evidence.effective_date}:{self.query_mode}:"
            f"{evidence.data_sha256[:16]}"
        )
        if (
            self.query_at != canonical_query_at
            or self.as_of_date != evidence.query_date.isoformat()
            or self.source_effective_date != evidence.effective_date
            or self.source_effective_until_exclusive is not None
            or self.observed_at != evidence.observed_at
            or self.receipt_at != evidence.receipt_at
            or self.strict_available_at != evidence.strict_available_at
            or self.reconstructed is not reconstructed
            or self.point_in_time_safe is not point_in_time_safe
            or self.sealed_oos_eligible is not sealed_oos_eligible
            or self.source_data_sha256 != evidence.data_sha256
            or self.source_receipt_sha256 != evidence.receipt_sha256
            or self.provenance_identity != provenance_identity
            or self.symbols != symbols
            or self.constituents != constituents
            or self.constituent_count != len(symbols)
            or self.snapshot_date != evidence.effective_date.replace("-", "")
            or self.universe_id != expected_universe_id
            or self.universe_sha256 != universe_sha256
        ):
            raise ValueError("历史合同与受信 root/receipt/query_at 派生语义不一致")

    def _contract_identity_payload(self) -> dict[str, object]:
        payload = super()._contract_identity_payload()
        payload.update(
            {
                "source_trust_policy": self.source_trust_policy,
                "as_of_date": self.as_of_date,
                "source_effective_date": self.source_effective_date,
                "source_effective_until_exclusive": (
                    self.source_effective_until_exclusive
                ),
                "observed_at": self.observed_at,
                "receipt_at": self.receipt_at,
                "strict_available_at": self.strict_available_at,
                "query_at": self.query_at,
                "reconstructed": self.reconstructed,
                "source_data_sha256": self.source_data_sha256,
                "source_receipt_sha256": self.source_receipt_sha256,
                "constituents": [row.to_dict() for row in self.constituents],
            }
        )
        return payload

    def validate_contract_identity(self) -> None:
        self._validate_base_semantics()
        self._validate_trusted_semantics()
        self._validate_existing_contract_sha256()

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload.update(
            {
                "selection_format": _HISTORICAL_SELECTION_FORMAT,
                "source_format": _CSI300_HISTORY_FORMAT,
                "source_index": _CSI300_INDEX,
                "source_trust_policy": self.source_trust_policy,
                "as_of_date": self.as_of_date,
                "mode": self.query_mode,
                "source_effective_date": self.source_effective_date,
                "source_effective_until_exclusive": (
                    self.source_effective_until_exclusive
                ),
                "observed_at": self.observed_at,
                "receipt_at": self.receipt_at,
                "strict_available_at": self.strict_available_at,
                "source_history_root": self.source_history_root,
                "query_at": self.query_at,
                "reconstructed": self.reconstructed,
                "source_data_sha256": self.source_data_sha256,
                "source_receipt_sha256": self.source_receipt_sha256,
                "constituents": [row.to_dict() for row in self.constituents],
            }
        )
        return payload


def load_csi300_historical_universe_contract(
    history_root: str | Path,
    *,
    as_of_date: str,
    mode: str,
    query_at: str | None = None,
    trust_policy: str | None = _CSI300_HISTORY_TRUST_POLICY,
) -> HistoricalUniverseContract:
    """按历史生效日查询沪深300，并显式区分严格可知与事后重建。"""
    if not isinstance(trust_policy, str):
        raise UniverseAvailabilityError("历史权重缺少代码内注册的受信根锚")
    evidence = _derive_historical_selection_evidence(
        history_root,
        as_of_date=as_of_date,
        trust_policy=trust_policy,
    )
    symbols, constituents = _load_historical_constituents(evidence)
    (
        canonical_query_at,
        reconstructed,
        point_in_time_safe,
        sealed_oos_eligible,
    ) = _derive_historical_query_gates(
        mode=mode,
        query_at=query_at,
        strict_available_at=evidence.strict_available_at,
    )
    provenance_identity = _canonical_sha256(
        {
            "contract_type": UNIVERSE_CONTRACT_TYPE_HISTORICAL,
            "source_trust_policy": trust_policy,
            "source_index": _CSI300_INDEX,
            "source_effective_date": evidence.effective_date,
            "source_data_sha256": evidence.data_sha256,
            "source_receipt_sha256": evidence.receipt_sha256,
        }
    )
    universe_sha256 = _canonical_sha256(
        {
            "format": _HISTORICAL_SELECTION_FORMAT,
            "index": _CSI300_INDEX,
            "source_trust_policy": trust_policy,
            "as_of_date": evidence.query_date.isoformat(),
            "mode": mode,
            "query_at": canonical_query_at,
            "source_effective_date": evidence.effective_date,
            "source_effective_until_exclusive": None,
            "observed_at": evidence.observed_at,
            "receipt_at": evidence.receipt_at,
            "strict_available_at": evidence.strict_available_at,
            "reconstructed": reconstructed,
            "point_in_time_safe": point_in_time_safe,
            "sealed_oos_eligible": sealed_oos_eligible,
            "source_data_sha256": evidence.data_sha256,
            "source_receipt_sha256": evidence.receipt_sha256,
            "constituents": [row.to_dict() for row in constituents],
        }
    )
    return HistoricalUniverseContract(
        universe_id=(
            f"{_CSI300_HISTORY_FORMAT}:{_CSI300_INDEX}:"
            f"{evidence.effective_date}:{mode}:{evidence.data_sha256[:16]}"
        ),
        snapshot_date=evidence.effective_date.replace("-", ""),
        constituent_count=len(symbols),
        universe_sha256=universe_sha256,
        symbols=symbols,
        contract_type=UNIVERSE_CONTRACT_TYPE_HISTORICAL,
        query_mode=mode,
        point_in_time_safe=point_in_time_safe,
        sealed_oos_eligible=sealed_oos_eligible,
        provenance_identity=provenance_identity,
        as_of_date=evidence.query_date.isoformat(),
        source_trust_policy=trust_policy,
        source_effective_date=evidence.effective_date,
        # receipt 只证明该日快照，不猜测未来一次调样何时终止本快照。
        # 保持开放区间也确保追加未来月份不会改写既有查询结果。
        source_effective_until_exclusive=None,
        observed_at=evidence.observed_at,
        receipt_at=evidence.receipt_at,
        strict_available_at=evidence.strict_available_at,
        reconstructed=reconstructed,
        source_data_sha256=evidence.data_sha256,
        source_receipt_sha256=evidence.receipt_sha256,
        constituents=constituents,
        source_history_root=str(evidence.history_root),
        query_at=canonical_query_at,
    )


def load_csi_a50_universe_contract() -> UniverseContract:
    """读取仓库内唯一可信 A50 快照，并复核独立冻结哈希。"""
    payload = load_frozen_universe(_CSI_A50_UNIVERSE_PATH)
    if payload["contract_sha256"] != _CSI_A50_CONTRACT_SHA256:
        raise ValueError("中证 A50 冻结合同与内置可信 SHA-256 不一致")
    symbols = tuple(row["symbol"] for row in payload["constituents"])
    return UniverseContract(
        universe_id=(
            f"{payload['format']}:{payload['index_code']}:{payload['snapshot_date']}"
        ),
        snapshot_date=payload["snapshot_date"],
        constituent_count=payload["constituent_count"],
        universe_sha256=payload["contract_sha256"],
        symbols=symbols,
        contract_type=UNIVERSE_CONTRACT_TYPE_TRUSTED_STATIC,
        query_mode=UNIVERSE_QUERY_MODE_STATIC,
        point_in_time_safe=True,
        sealed_oos_eligible=True,
        provenance_identity=payload["contract_sha256"],
    )


__all__ = [
    "HistoricalUniverseContract",
    "UNIVERSE_CONTRACT_TYPE_HISTORICAL",
    "UNIVERSE_CONTRACT_TYPE_TRUSTED_STATIC",
    "UNIVERSE_CONTRACT_TYPE_UNTRUSTED",
    "UNIVERSE_QUERY_MODE_RECONSTRUCTED",
    "UNIVERSE_QUERY_MODE_STATIC",
    "UNIVERSE_QUERY_MODE_STRICT",
    "UNIVERSE_QUERY_MODE_UNTRUSTED",
    "UniverseAvailabilityError",
    "UniverseContract",
    "WeightedUniverseConstituent",
    "load_csi300_historical_universe_contract",
    "load_csi_a50_universe_contract",
]
