"""RQAlpha 冻结 bundle 的严格限域执行状态叠加层（execution-state overlay，只读）。

台账合法表述（唯一口径）：完整 RND-04A = BLOCKED 不变；本模块只实现
2026-07-26 独立裁决放行的严格限域 execution-state overlay 子集
（PARTIAL_READY），作为 RND-04C 日线 replay 的执行时钟输入层。
裁决全文：docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md。

语义边界（防洗白，逐条对应裁决禁止声明清单）：

- 本层只服务执行时钟 ``effective_at <= fill_ts``；输出永不进入决策时钟
  （选股、信号、权重、模型输入、校准历史）。不得称为 strict PIT。
- ``LIMIT_UP_LOCKED`` / ``LIMIT_DOWN_LOCKED`` 只是"保守日线触及规则"
  （conservative daily touch rule）：当日收盘价触及显式非零涨/跌停价即
  在执行层拒绝成交；不代表真实封单、盘口排队或不可成交证据。
- 停牌唯一来源是 suspended_days.h5；禁止由 bar 存在、价格不变或
  volume == 0 推断；停牌日的 RQAlpha 日线可能只是填充值，本层在停牌日
  不返回任何价格字段。
- ``limit_up == 0 and limit_down == 0`` 表示该日无普通日涨跌停边界
  （如科创板上市初期），禁止判 locked。
- RQAlpha 未复权 OHLC 只在本模块内部做同空间触及比较，永不充当估值、
  成交或信号价格源；禁止与 FreeStockDB 前复权（qfq）价格跨口径比较。
- 日期硬上限 2026-06-30；10 个截止日后的未决缺口永久失败关闭。
- 代码域只取 v3 的 741 个 available 交集；207 个 quarantine 与 1 个
  source_missing 不因 RQAlpha 有数据而升级。
- 990018.XSHG 的 prev_close 语义禁止使用。
- 本模块对 G 盘一律只读；不把任何状态写回 FreeStockDB Parquet，
  不混同 provenance。
- lot_size 来源：instruments.pk 的 round_lot 字段（经拒绝 find_class /
  persistent_load 的静态安全解码核实：5,553/5,553 条 CS 记录均携带，
  值域 {100, 200}，688xxx 全部 200；默认 100 对科创板是错的）。
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# 受信锚与语义常量
# ---------------------------------------------------------------------------

OVERLAY_CONTRACT_FORMAT = "alphamaster_rqalpha_execution_overlay_v1"

# 保守日线触及规则：当日未复权收盘价触及显式非零涨/跌停价即判 locked。
# 名称与 identity 一并绑定，禁止被改述为真实封板/盘口证据。
DERIVATION_RULE_VERSION = "conservative_daily_touch_rule_v1"
DERIVATION_RULE_DESCRIPTION = (
    "保守日线触及规则：仅当 close >= limit_up > 0 判 LIMIT_UP_LOCKED，"
    "仅当 0 < limit_down >= close 判 LIMIT_DOWN_LOCKED；"
    "比较只在 RQAlpha 未复权空间内进行，不证明真实封单或盘口排队。"
)

# 与 portfolio_manager.execution.ExecutionQuote 的四值状态枚举字面一致；
# 不直接 import，避免 data_pipeline 反向依赖组合层（一致性由单测锁定）。
EXECUTION_STATE_STATUSES = frozenset(
    {
        "OPEN",
        "SUSPENDED",
        "LIMIT_UP_LOCKED",
        "LIMIT_DOWN_LOCKED",
    }
)

# stocks.h5 每个 dataset 的结构化字段合同（与实证审计一致，顺序敏感）。
STOCKS_ALLOWED_FIELDS = (
    "datetime",
    "open",
    "close",
    "high",
    "low",
    "prev_close",
    "limit_up",
    "limit_down",
    "volume",
    "total_turnover",
)

PERMITTED_SEMANTICS = (
    "显式停牌状态（唯一来源 suspended_days.h5）",
    "显式逐日 ST 状态（st_stock_days.h5，存储为严格降序，读取时显式归一化）",
    "未复权 prev_close（排除 990018.XSHG）",
    "日线显式非零 limit_up/limit_down 价格边界（保守日线触及规则）",
)
FORBIDDEN_SEMANTICS = (
    "真实封板、开板、盘口队列或可成交量（LOCKED 只是保守日线触及规则）",
    "历史证券简称",
    "供应商公司行动 known_at 与修订链",
    "2026-07-01 及之后的执行状态",
    "决策时钟消费（选股、信号、权重、模型输入、校准历史）",
    "以 RQAlpha OHLC 充当估值、成交或信号价格源",
    "与前复权（qfq）价格的跨口径比较",
)

# dataset 缺失语义显式固定（裁决测试组 13）：
# suspended_days.h5 只有 3,440 个 dataset，缺失即"无停牌记录"，不是数据错误；
# st_stock_days.h5 覆盖全部 5,553 条 CS 主表记录，741 交集必须逐一存在，
# 空 dataset 表示"无 ST 记录"。
SUSPENDED_MISSING_DATASET_SEMANTICS = (
    "suspended_days.h5 无该证券 dataset == 该证券无停牌记录"
)
ST_DATASET_SEMANTICS = (
    "st_stock_days.h5 必须覆盖 741 交集内全部证券；空 dataset == 无 ST 记录"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_SESSION_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_NEVER_DELISTED = "0000-00-00"
_V3_MANIFEST_FORMAT = "free_stockdb_csi300_historical_am_inputs_v3"

# 默认冻结外部输入位置（G 盘只读；见 scratch/CLAUDE_HANDOFF_20260726.md 第 5、6 节）。
RQALPHA_EXTRACTED_ROOT = Path(r"G:\QuantData\rqalpha\extracted\rqbundle_202607")
V3_EXPORT_ROOT = Path(
    r"G:\QuantData\free-stockdb\am_exports\20260726_csi300_historical_am_inputs_v3"
)
FROZEN_CALENDAR_PATH = Path(
    r"G:\QuantData\free-stockdb\online_api\snapshots\20260724_bootstrap_v1"
    r"\trade_calendar.parquet"
)

# 代码内唯一受信锚：全部哈希来自 2026-07-26 只读实证审计
# scratch/rnd04a_rqalpha_bundle_20260726/data_inspection_result.json
# （审计 JSON 自身 SHA-256
#  157dbb37ed1eee8ae8947d6113ddd591abf1667eba277eb6397bc9bcf222f387）。
TRUSTED_OVERLAY_ANCHOR: dict[str, Any] = {
    "format": OVERLAY_CONTRACT_FORMAT,
    "bundle_name": "rqbundle_202607",
    "archive_sha256": (
        "f2d8a07e8791d93f4edeef0084c50ef3de03e176c42596f9e65a7bb13820d961"
    ),
    "member_inventory_sha256": (
        "3e5b24461307e6ab9177b6f353c669fb239d12c466a1b114c367561c20c71dd7"
    ),
    "member_sha256": {
        "stocks.h5": (
            "f94bde02683f9934c68f648896ace2a2c082d29269ba139949c5767c184b3202"
        ),
        "suspended_days.h5": (
            "9000c52bfcca5401006b30bd79f69c59c7ea2b1457483dfdbb15ef949a6f0f05"
        ),
        "st_stock_days.h5": (
            "c10a59827f9a324b0eb7a7c7aef638f314fa2edf043cd054aa50087e79ea5392"
        ),
        "dividends.h5": (
            "5a30dfec12a519901be6b2938a18dc3bb7fa0613b383f53714143f2e78d46d43"
        ),
        "split_factor.h5": (
            "7ecdc7f68015406af80d9741fa81c6eceed6f39d8fbefb77cb644135023cb863"
        ),
        "ex_cum_factor.h5": (
            "d9f1d62cec48fec49b964b7e7631686b69026ffd877c40f7ec3d9d1f6d7dd8f6"
        ),
        "trading_dates.npy": (
            "57dd7f539d1cdbefb528978a6c7bd79b118790f933ce63c37b10747b999f9585"
        ),
        "instruments.pk": (
            "472a7f7cc8ab8fe2798893dd83a6d2656328a40105ee2741d5a3bd470f0e3b46"
        ),
    },
    "session_first": 20050104,
    "session_last_inclusive": 20260630,
    "available_code_count": 741,
    "v3_manifest_sha256": (
        "e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c"
    ),
    "v3_coverage_matrix_sha256": (
        "0fc6914d1c467de9fead9df6a82c406d8b98cf713eb1a081da0b2ba39e7e4662"
    ),
    "frozen_calendar_sha256": (
        "49eb9814073441385b05bfb81f2bdffef7c0765c23b64e23a0acfcbd89d099c8"
    ),
    "calendar_intersection_rows": 5235,
    "prev_close_excluded": ("990018.XSHG",),
    # 10 个截止日后的未决缺口，永久失败关闭，直至冻结覆盖 >=2026-07-13 的
    # 同源新 bundle 并重复同一只读审计（裁决禁止声明第 8 条）。
    "unresolved_gaps": (
        ("600000.XSHG", 20260701),
        ("600000.XSHG", 20260702),
        ("688072.XSHG", 20260701),
        ("688072.XSHG", 20260702),
        ("688072.XSHG", 20260703),
        ("688072.XSHG", 20260706),
        ("688072.XSHG", 20260707),
        ("688072.XSHG", 20260708),
        ("688072.XSHG", 20260709),
        ("688072.XSHG", 20260710),
    ),
    # 静态安全解码实测 round_lot 值域：主板/创业板 100，科创板 200。
    "lot_size_allowed": (100, 200),
}


class RQAlphaOverlayError(ValueError):
    """execution-state overlay 的身份、域或数据一致性失败关闭。"""


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
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RQAlphaOverlayError(f"{label} 必须是小写 SHA-256")
    return value


def _require_yyyymmdd(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RQAlphaOverlayError(f"{label} 必须是 YYYYMMDD 整数")
    text = str(value)
    if len(text) != 8:
        raise RQAlphaOverlayError(f"{label} 必须是 YYYYMMDD 整数")
    try:
        date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise RQAlphaOverlayError(f"{label} 不是真实日历日期") from exc
    return value


class _SafeUnpickler(pickle.Unpickler):
    """拒绝任意类构造与持久化引用的只读解码器（复用审计 decode_policy）。"""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"instruments.pk 禁止 GLOBAL 解码: {module}.{name}"
        )

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("instruments.pk 禁止 persistent id")


def _safe_unpickle_instruments(path: Path) -> list[Any]:
    raw = path.read_bytes()
    try:
        payload = _SafeUnpickler(io.BytesIO(raw), fix_imports=False).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
        raise RQAlphaOverlayError(
            f"instruments.pk 安全解码失败（拒绝任意类/持久化引用）: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise RQAlphaOverlayError("instruments.pk 顶层必须是 list")
    return payload


@dataclass(frozen=True)
class _InstrumentRow:
    """CS 主表中本层消费的最小字段子集（其余字段一律不读语义）。"""

    order_book_id: str
    trading_code: str
    listed_date: int
    de_listed_date_exclusive: int | None
    round_lot: int


def _validate_anchor(anchor: Mapping[str, Any]) -> dict[str, Any]:
    """锚结构失败关闭校验；注入的测试锚也必须满足同一结构合同。"""
    if not isinstance(anchor, Mapping):
        raise RQAlphaOverlayError("trusted_anchor 必须是映射")
    expected_keys = set(TRUSTED_OVERLAY_ANCHOR)
    if set(anchor) != expected_keys:
        raise RQAlphaOverlayError(
            "trusted_anchor 字段集合与 overlay 合同不一致"
        )
    if anchor["format"] != OVERLAY_CONTRACT_FORMAT:
        raise RQAlphaOverlayError("trusted_anchor format 不受支持")
    if not isinstance(anchor["bundle_name"], str) or not anchor["bundle_name"]:
        raise RQAlphaOverlayError("bundle_name 必须是非空文本")
    _require_sha256(anchor["archive_sha256"], "archive_sha256")
    _require_sha256(anchor["member_inventory_sha256"], "member_inventory_sha256")
    members = anchor["member_sha256"]
    if (
        not isinstance(members, Mapping)
        or set(members) != set(TRUSTED_OVERLAY_ANCHOR["member_sha256"])
    ):
        raise RQAlphaOverlayError("member_sha256 成员文件清单不完整")
    for name, digest in members.items():
        _require_sha256(digest, f"member_sha256[{name}]")
    first = _require_yyyymmdd(anchor["session_first"], "session_first")
    last = _require_yyyymmdd(
        anchor["session_last_inclusive"], "session_last_inclusive"
    )
    if first > last:
        raise RQAlphaOverlayError("session_first 不得晚于 session_last_inclusive")
    count = anchor["available_code_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RQAlphaOverlayError("available_code_count 必须是正整数")
    _require_sha256(anchor["v3_manifest_sha256"], "v3_manifest_sha256")
    _require_sha256(
        anchor["v3_coverage_matrix_sha256"], "v3_coverage_matrix_sha256"
    )
    _require_sha256(anchor["frozen_calendar_sha256"], "frozen_calendar_sha256")
    rows = anchor["calendar_intersection_rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise RQAlphaOverlayError("calendar_intersection_rows 必须是正整数")
    excluded = anchor["prev_close_excluded"]
    if not isinstance(excluded, tuple) or not all(
        isinstance(item, str) and item for item in excluded
    ):
        raise RQAlphaOverlayError("prev_close_excluded 必须是非空文本元组")
    gaps = anchor["unresolved_gaps"]
    if not isinstance(gaps, tuple):
        raise RQAlphaOverlayError("unresolved_gaps 必须是元组")
    for item in gaps:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
        ):
            raise RQAlphaOverlayError("unresolved_gaps 项必须是 (order_book_id, 日期)")
        _require_yyyymmdd(item[1], "unresolved_gaps 日期")
    lots = anchor["lot_size_allowed"]
    if (
        not isinstance(lots, tuple)
        or not lots
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in lots
        )
    ):
        raise RQAlphaOverlayError("lot_size_allowed 必须是正整数元组")
    return {key: anchor[key] for key in expected_keys}


@dataclass(frozen=True)
class RQAlphaOverlayIdentity:
    """replay 运行层必须绑定的 overlay 来源身份（裁决硬条件 B）。

    ``ExecutionQuote`` 不携带来源字段；RND-04C 必须把本身份的
    ``identity_sha256`` 绑定进 replay 运行身份，使任何 overlay 状态或
    来源声明的篡改都能被身份链检出。
    """

    contract_format: str
    bundle_name: str
    archive_sha256: str
    member_inventory_sha256: str
    member_sha256: tuple[tuple[str, str], ...]
    session_first: int
    session_last_inclusive: int
    allowed_fields: tuple[str, ...]
    permitted_semantics: tuple[str, ...]
    forbidden_semantics: tuple[str, ...]
    suspended_missing_dataset_semantics: str
    st_dataset_semantics: str
    prev_close_excluded: tuple[str, ...]
    unresolved_gaps: tuple[tuple[str, int], ...]
    derivation_rule_version: str
    derivation_rule_description: str
    available_code_count: int
    v3_manifest_sha256: str
    v3_coverage_matrix_sha256: str
    frozen_calendar_sha256: str
    calendar_intersection_rows: int
    lot_size_source: str
    lot_size_allowed: tuple[int, ...]
    identity_sha256: str = ""

    def __post_init__(self) -> None:
        expected = _canonical_sha256(self._identity_payload())
        if self.identity_sha256 == "":
            object.__setattr__(self, "identity_sha256", expected)
        elif self.identity_sha256 != expected:
            raise RQAlphaOverlayError(
                "identity_sha256 与 canonical 身份内容不一致"
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "contract_format": self.contract_format,
            "bundle_name": self.bundle_name,
            "archive_sha256": self.archive_sha256,
            "member_inventory_sha256": self.member_inventory_sha256,
            "member_sha256": {name: digest for name, digest in self.member_sha256},
            "session_first": self.session_first,
            "session_last_inclusive": self.session_last_inclusive,
            "allowed_fields": list(self.allowed_fields),
            "permitted_semantics": list(self.permitted_semantics),
            "forbidden_semantics": list(self.forbidden_semantics),
            "suspended_missing_dataset_semantics": (
                self.suspended_missing_dataset_semantics
            ),
            "st_dataset_semantics": self.st_dataset_semantics,
            "prev_close_excluded": list(self.prev_close_excluded),
            "unresolved_gaps": [list(item) for item in self.unresolved_gaps],
            "derivation_rule_version": self.derivation_rule_version,
            "derivation_rule_description": self.derivation_rule_description,
            "available_code_count": self.available_code_count,
            "v3_manifest_sha256": self.v3_manifest_sha256,
            "v3_coverage_matrix_sha256": self.v3_coverage_matrix_sha256,
            "frozen_calendar_sha256": self.frozen_calendar_sha256,
            "calendar_intersection_rows": self.calendar_intersection_rows,
            "lot_size_source": self.lot_size_source,
            "lot_size_allowed": list(self.lot_size_allowed),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["identity_sha256"] = self.identity_sha256
        return payload


@dataclass(frozen=True)
class RQAlphaExecutionState:
    """单个 (symbol, session_date) 的执行时钟状态观测。

    价格字段全部位于 RQAlpha 未复权空间，仅供执行层触及核验；
    停牌日全部为 None（RQAlpha 停牌日 bar 可能是填充值，不外泄）。
    """

    symbol: str
    order_book_id: str
    session_date: str
    status: str
    is_st: bool
    lot_size: int
    close: float | None
    prev_close: float | None
    limit_up: float | None
    limit_down: float | None
    derivation_rule_version: str
    source_identity_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "order_book_id": self.order_book_id,
            "session_date": self.session_date,
            "status": self.status,
            "is_st": self.is_st,
            "lot_size": self.lot_size,
            "close": self.close,
            "prev_close": self.prev_close,
            "limit_up": self.limit_up,
            "limit_down": self.limit_down,
            "derivation_rule_version": self.derivation_rule_version,
            "source_identity_sha256": self.source_identity_sha256,
        }


def _load_v3_available_codes(
    v3_export_root: Path,
    anchor: dict[str, Any],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """按锚哈希校验 v3 manifest 与 coverage matrix，返回 949 状态表与 741 交集。"""
    import pandas as pd

    manifest_path = v3_export_root / "manifest.json"
    if not manifest_path.is_file():
        raise RQAlphaOverlayError(f"v3 manifest 不存在: {manifest_path}")
    if _sha256_file(manifest_path) != anchor["v3_manifest_sha256"]:
        raise RQAlphaOverlayError("v3 manifest SHA-256 与受信锚不一致")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RQAlphaOverlayError("v3 manifest 不是合法 UTF-8 JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != _V3_MANIFEST_FORMAT
        or manifest.get("status") != "completed"
    ):
        raise RQAlphaOverlayError("v3 manifest 的 format/status 不匹配")
    status_counts = manifest.get("status_counts")
    if (
        not isinstance(status_counts, dict)
        or status_counts.get("available") != anchor["available_code_count"]
    ):
        raise RQAlphaOverlayError(
            "v3 manifest 的 available 数量与受信锚不一致"
        )
    coverage_meta = manifest.get("coverage_matrix")
    if (
        not isinstance(coverage_meta, dict)
        or coverage_meta.get("sha256") != anchor["v3_coverage_matrix_sha256"]
    ):
        raise RQAlphaOverlayError(
            "v3 manifest 声明的 coverage matrix SHA-256 与受信锚不一致"
        )
    coverage_path = v3_export_root / str(coverage_meta.get("relative_path"))
    if not coverage_path.is_file():
        raise RQAlphaOverlayError(f"coverage matrix 不存在: {coverage_path}")
    if _sha256_file(coverage_path) != anchor["v3_coverage_matrix_sha256"]:
        raise RQAlphaOverlayError("coverage matrix SHA-256 与受信锚不一致")
    frame = pd.read_parquet(coverage_path, columns=["code", "status"])
    codes = frame["code"].tolist()
    statuses = frame["status"].tolist()
    status_by_code: dict[str, str] = {}
    for code, status in zip(codes, statuses, strict=True):
        if (
            not isinstance(code, str)
            or _SYMBOL_RE.fullmatch(code) is None
            or code in status_by_code
        ):
            raise RQAlphaOverlayError("coverage matrix 的 code 非法或重复")
        if status not in {"available", "quarantine", "source_missing"}:
            raise RQAlphaOverlayError(f"coverage matrix 出现未知状态: {status}")
        status_by_code[code] = status
    available = tuple(
        sorted(
            code
            for code, status in status_by_code.items()
            if status == "available"
        )
    )
    if len(available) != anchor["available_code_count"]:
        raise RQAlphaOverlayError("coverage matrix 的 available 数量与受信锚不一致")
    return status_by_code, available


def _load_calendar_sentinel(
    trading_dates_path: Path,
    frozen_calendar_path: Path,
    anchor: dict[str, Any],
) -> frozenset[int]:
    """日历一致性运行时哨兵：共同区间两源集合必须完全相等（裁决测试组 16）。"""
    import pandas as pd

    raw = np.load(trading_dates_path)
    if raw.ndim != 1 or raw.size == 0:
        raise RQAlphaOverlayError("trading_dates.npy 结构非法")
    rq_dates = [int(value) for value in raw.tolist()]
    for value in rq_dates:
        _require_yyyymmdd(value, "RQAlpha 交易日")
    if rq_dates != sorted(set(rq_dates)):
        raise RQAlphaOverlayError("RQAlpha 交易日历必须严格升序无重复")

    if not frozen_calendar_path.is_file():
        raise RQAlphaOverlayError(
            f"冻结交易日历不存在: {frozen_calendar_path}"
        )
    if _sha256_file(frozen_calendar_path) != anchor["frozen_calendar_sha256"]:
        raise RQAlphaOverlayError("冻结交易日历 SHA-256 与受信锚不一致")
    calendar = pd.read_parquet(frozen_calendar_path)
    if list(calendar.columns) != ["trade_date"] or calendar.empty:
        raise RQAlphaOverlayError("冻结交易日历列合同不匹配或为空")
    fs_dates = [
        int(stamp.strftime("%Y%m%d"))
        for stamp in pd.to_datetime(calendar["trade_date"], errors="raise")
    ]
    if fs_dates != sorted(set(fs_dates)):
        raise RQAlphaOverlayError("冻结交易日历必须严格升序无重复")

    common_first = max(rq_dates[0], fs_dates[0])
    common_last = min(rq_dates[-1], fs_dates[-1])
    rq_common = {d for d in rq_dates if common_first <= d <= common_last}
    fs_common = {d for d in fs_dates if common_first <= d <= common_last}
    if rq_common != fs_common:
        raise RQAlphaOverlayError(
            "RQAlpha 与 FreeStockDB 交易日历在共同区间不一致，"
            f"差异 {len(rq_common ^ fs_common)} 天"
        )
    if len(rq_common) != anchor["calendar_intersection_rows"]:
        raise RQAlphaOverlayError(
            "共同区间交易日数量与受信锚不一致: "
            f"{len(rq_common)} != {anchor['calendar_intersection_rows']}"
        )
    # 域内会话集合只保留 <= session_last_inclusive 的部分；
    # RQAlpha 日历 2026-06-30 之后的部分禁止参与任何执行状态判断。
    return frozenset(
        d for d in rq_dates if d <= anchor["session_last_inclusive"]
    )


def _load_instrument_rows(
    instruments_path: Path,
    anchor: dict[str, Any],
) -> dict[str, _InstrumentRow]:
    """静态安全解码 CS 主表，构建 trading_code -> 最小消费字段映射。"""
    rows: dict[str, _InstrumentRow] = {}
    allowed_lots = set(anchor["lot_size_allowed"])
    for item in _safe_unpickle_instruments(instruments_path):
        if not isinstance(item, dict) or item.get("type") != "CS":
            continue
        order_book_id = item.get("order_book_id")
        trading_code = item.get("trading_code")
        if (
            not isinstance(order_book_id, str)
            or not order_book_id
            or not isinstance(trading_code, str)
            or _SYMBOL_RE.fullmatch(trading_code) is None
        ):
            raise RQAlphaOverlayError(
                f"CS 主表记录的 order_book_id/trading_code 非法: {item!r:.120}"
            )
        if trading_code in rows:
            raise RQAlphaOverlayError(
                f"CS 主表 trading_code 出现重复映射: {trading_code}"
            )
        listed_raw = item.get("listed_date")
        if (
            not isinstance(listed_raw, str)
            or _ISO_DATE_RE.fullmatch(listed_raw) is None
        ):
            raise RQAlphaOverlayError(
                f"{order_book_id} 的 listed_date 非法: {listed_raw!r}"
            )
        listed = _require_yyyymmdd(
            int(listed_raw.replace("-", "")), f"{order_book_id}.listed_date"
        )
        de_listed_raw = item.get("de_listed_date")
        if de_listed_raw == _NEVER_DELISTED:
            de_listed: int | None = None
        elif (
            isinstance(de_listed_raw, str)
            and _ISO_DATE_RE.fullmatch(de_listed_raw) is not None
        ):
            de_listed = _require_yyyymmdd(
                int(de_listed_raw.replace("-", "")),
                f"{order_book_id}.de_listed_date",
            )
        else:
            raise RQAlphaOverlayError(
                f"{order_book_id} 的 de_listed_date 非法: {de_listed_raw!r}"
            )
        lot_raw = item.get("round_lot")
        if isinstance(lot_raw, bool):
            raise RQAlphaOverlayError(f"{order_book_id} 的 round_lot 非法")
        if isinstance(lot_raw, int):
            lot = lot_raw
        elif isinstance(lot_raw, float) and lot_raw.is_integer():
            lot = int(lot_raw)
        else:
            raise RQAlphaOverlayError(
                f"{order_book_id} 的 round_lot 非法: {lot_raw!r}"
            )
        if lot not in allowed_lots:
            raise RQAlphaOverlayError(
                f"{order_book_id} 的 round_lot={lot} 不在受信值域 "
                f"{sorted(allowed_lots)} 内"
            )
        rows[trading_code] = _InstrumentRow(
            order_book_id=order_book_id,
            trading_code=trading_code,
            listed_date=listed,
            de_listed_date_exclusive=de_listed,
            round_lot=lot,
        )
    if not rows:
        raise RQAlphaOverlayError("instruments.pk 中没有任何 CS 主表记录")
    return rows


def _load_day_sets(
    h5_path: Path,
    *,
    label: str,
    required_order: str,
) -> dict[str, frozenset[int]]:
    """把逐日状态 HDF5 全量载入内存集合，并显式校验存储方向。

    required_order:
      - "ascending_or_singleton"：suspended_days.h5 的审计事实；
      - "strictly_descending"：st_stock_days.h5 的审计事实，读取方必须
        显式处理降序，升序注入必须被检出（裁决测试组 5）。
    """
    import h5py

    result: dict[str, frozenset[int]] = {}
    with h5py.File(h5_path, "r") as handle:
        for key in handle.keys():
            arr = handle[key][:]
            if arr.size == 0:
                result[key] = frozenset()
                continue
            values = np.asarray(arr)
            if values.dtype.kind == "f":
                if not np.all(np.isfinite(values)) or not np.all(
                    values == np.floor(values)
                ):
                    raise RQAlphaOverlayError(
                        f"{label} 的 {key} 含非整数日期值"
                    )
                values = values.astype("int64")
            elif values.dtype.kind != "i":
                raise RQAlphaOverlayError(
                    f"{label} 的 {key} dtype 非法: {values.dtype}"
                )
            days = [int(v) for v in values.tolist()]
            for day in days:
                _require_yyyymmdd(day, f"{label}[{key}] 日期")
            if len(set(days)) != len(days):
                raise RQAlphaOverlayError(f"{label} 的 {key} 存在重复日期")
            if required_order == "strictly_descending":
                if len(days) > 1 and days != sorted(days, reverse=True):
                    raise RQAlphaOverlayError(
                        f"{label} 的 {key} 不是严格降序；"
                        "读取方必须显式处理降序存储，不得假定升序"
                    )
            elif required_order == "ascending_or_singleton":
                if len(days) > 1 and days != sorted(days):
                    raise RQAlphaOverlayError(
                        f"{label} 的 {key} 不是升序存储"
                    )
            else:  # pragma: no cover - 编码错误保护
                raise RQAlphaOverlayError("required_order 非法")
            result[key] = frozenset(days)
    return result


class RQAlphaExecutionOverlay:
    """RQAlpha bundle 的只读执行状态叠加层（严格限域，失败关闭）。

    实例只在加载时读取 G 盘一次性输入并完成全部身份校验；
    ``derive_execution_state`` 逐 (symbol, session_date) 派生执行状态。
    没有任何写路径；不提供价格填补、状态推断或跨口径比较接口。
    """

    def __init__(
        self,
        extracted_root: str | Path = RQALPHA_EXTRACTED_ROOT,
        *,
        v3_export_root: str | Path = V3_EXPORT_ROOT,
        frozen_calendar_path: str | Path = FROZEN_CALENDAR_PATH,
        trusted_anchor: Mapping[str, Any] | None = None,
    ) -> None:
        anchor = _validate_anchor(
            TRUSTED_OVERLAY_ANCHOR if trusted_anchor is None else trusted_anchor
        )
        root = Path(extracted_root)
        if not root.is_dir():
            raise RQAlphaOverlayError(f"RQAlpha 解压目录不存在: {root}")

        # 1) 供应链身份：8 个成员文件逐一真实哈希，任一不符拒载。
        for name, expected in sorted(anchor["member_sha256"].items()):
            member = root / name
            if not member.is_file():
                raise RQAlphaOverlayError(f"bundle 成员文件缺失: {member}")
            actual = _sha256_file(member)
            if actual != expected:
                raise RQAlphaOverlayError(
                    f"bundle 成员 {name} SHA-256 与受信锚不一致"
                )

        # 2) 代码域：v3 manifest + coverage matrix 哈希校验，取 741 available。
        status_by_code, available_codes = _load_v3_available_codes(
            Path(v3_export_root), anchor
        )

        # 3) 日历一致性运行时哨兵。
        session_days = _load_calendar_sentinel(
            root / "trading_dates.npy",
            Path(frozen_calendar_path),
            anchor,
        )

        # 4) CS 主表映射（静态安全解码；lot_size 一并派生并校验值域）。
        instrument_rows = _load_instrument_rows(
            root / "instruments.pk", anchor
        )
        available_rows: dict[str, _InstrumentRow] = {}
        for code in available_codes:
            row = instrument_rows.get(code)
            if row is None:
                raise RQAlphaOverlayError(
                    f"available 代码 {code} 在 CS 主表中没有唯一映射"
                )
            available_rows[code] = row

        # 5) 逐日停牌 / ST 状态全量载入，显式处理存储方向。
        suspended_days = _load_day_sets(
            root / "suspended_days.h5",
            label="suspended_days.h5",
            required_order="ascending_or_singleton",
        )
        st_days = _load_day_sets(
            root / "st_stock_days.h5",
            label="st_stock_days.h5",
            required_order="strictly_descending",
        )

        # 6) dataset 缺失语义显式固定（裁决测试组 13）：
        #    741 交集逐一确认 stocks/st 存在性；suspended 缺失 == 无停牌记录。
        import h5py

        stocks_handle = h5py.File(root / "stocks.h5", "r")
        try:
            presence = {"stocks": 0, "suspended": 0, "st": 0}
            for code, row in available_rows.items():
                obid = row.order_book_id
                if obid not in stocks_handle:
                    raise RQAlphaOverlayError(
                        f"available 代码 {code} 在 stocks.h5 中缺少 dataset"
                    )
                presence["stocks"] += 1
                if obid not in st_days:
                    raise RQAlphaOverlayError(
                        f"available 代码 {code} 在 st_stock_days.h5 中缺少 "
                        "dataset；空 dataset 才表示无 ST 记录"
                    )
                presence["st"] += 1
                if obid in suspended_days:
                    presence["suspended"] += 1
        except Exception:
            stocks_handle.close()
            raise

        self._anchor = anchor
        self._stocks = stocks_handle
        self._suspended_days = suspended_days
        self._st_days = st_days
        self._status_by_code = status_by_code
        self._available_rows = available_rows
        self._session_days = session_days
        self._prev_close_excluded = frozenset(anchor["prev_close_excluded"])
        self._unresolved_gaps = frozenset(
            (obid, day) for obid, day in anchor["unresolved_gaps"]
        )
        self.dataset_presence_summary = dict(presence)
        self.identity = RQAlphaOverlayIdentity(
            contract_format=anchor["format"],
            bundle_name=anchor["bundle_name"],
            archive_sha256=anchor["archive_sha256"],
            member_inventory_sha256=anchor["member_inventory_sha256"],
            member_sha256=tuple(sorted(anchor["member_sha256"].items())),
            session_first=anchor["session_first"],
            session_last_inclusive=anchor["session_last_inclusive"],
            allowed_fields=STOCKS_ALLOWED_FIELDS,
            permitted_semantics=PERMITTED_SEMANTICS,
            forbidden_semantics=FORBIDDEN_SEMANTICS,
            suspended_missing_dataset_semantics=(
                SUSPENDED_MISSING_DATASET_SEMANTICS
            ),
            st_dataset_semantics=ST_DATASET_SEMANTICS,
            prev_close_excluded=tuple(anchor["prev_close_excluded"]),
            unresolved_gaps=tuple(anchor["unresolved_gaps"]),
            derivation_rule_version=DERIVATION_RULE_VERSION,
            derivation_rule_description=DERIVATION_RULE_DESCRIPTION,
            available_code_count=anchor["available_code_count"],
            v3_manifest_sha256=anchor["v3_manifest_sha256"],
            v3_coverage_matrix_sha256=anchor["v3_coverage_matrix_sha256"],
            frozen_calendar_sha256=anchor["frozen_calendar_sha256"],
            calendar_intersection_rows=anchor["calendar_intersection_rows"],
            lot_size_source=(
                "instruments.pk round_lot（静态安全解码，5,553/5,553 CS 记录）"
            ),
            lot_size_allowed=tuple(anchor["lot_size_allowed"]),
        )

    # -- 生命周期 -----------------------------------------------------------

    def close(self) -> None:
        self._stocks.close()

    def __enter__(self) -> "RQAlphaExecutionOverlay":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- 状态派生 -----------------------------------------------------------

    def derive_execution_state(
        self,
        symbol: str,
        session_date: str,
    ) -> RQAlphaExecutionState:
        """派生 (symbol, session_date) 的执行时钟状态；一切异常失败关闭。"""
        if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
            raise RQAlphaOverlayError(
                f"symbol 必须是 6 位数字文本: {symbol!r}"
            )
        row = self._available_rows.get(symbol)
        if row is None:
            status = self._status_by_code.get(symbol)
            if status is None:
                raise RQAlphaOverlayError(
                    f"代码 {symbol} 不在冻结 949 覆盖矩阵内"
                )
            raise RQAlphaOverlayError(
                f"代码 {symbol} 的 v3 状态为 {status}，不在 741 available "
                "交集内；不得因 RQAlpha 有数据而升级"
            )
        obid = row.order_book_id
        if obid in self._prev_close_excluded:
            raise RQAlphaOverlayError(
                f"{obid} 的 prev_close 语义被裁决排除，禁止派生执行状态"
            )

        if (
            not isinstance(session_date, str)
            or _SESSION_DATE_RE.fullmatch(session_date) is None
        ):
            raise RQAlphaOverlayError(
                f"session_date 必须是 YYYY-MM-DD: {session_date!r}"
            )
        try:
            date.fromisoformat(session_date)
        except ValueError as exc:
            raise RQAlphaOverlayError(
                f"session_date 不是有效日历日期: {session_date}"
            ) from exc
        day = int(session_date.replace("-", ""))
        anchor = self._anchor
        if day > anchor["session_last_inclusive"]:
            raise RQAlphaOverlayError(
                f"{session_date} 晚于 bundle 截止日 "
                f"{anchor['session_last_inclusive']}；2026-07-01 及之后的"
                "执行状态被裁决禁止，需冻结新 bundle 并重复只读审计"
            )
        if day < anchor["session_first"]:
            raise RQAlphaOverlayError(
                f"{session_date} 早于 bundle 首个交易日 "
                f"{anchor['session_first']}"
            )
        if (obid, day) in self._unresolved_gaps:
            raise RQAlphaOverlayError(
                f"{obid}/{session_date} 属于 10 个未决缺口之一，永久失败关闭"
            )
        if day not in self._session_days:
            raise RQAlphaOverlayError(
                f"{session_date} 不是 RQAlpha 域内交易日"
            )
        if day < row.listed_date or (
            row.de_listed_date_exclusive is not None
            and day >= row.de_listed_date_exclusive
        ):
            raise RQAlphaOverlayError(
                f"{obid} 在 {session_date} 不在上市有效期内"
            )

        is_st = day in self._st_days[obid]

        # 停牌唯一来源 suspended_days.h5；停牌短路，填充 bar 不外泄。
        if day in self._suspended_days.get(obid, frozenset()):
            return RQAlphaExecutionState(
                symbol=symbol,
                order_book_id=obid,
                session_date=session_date,
                status="SUSPENDED",
                is_st=is_st,
                lot_size=row.round_lot,
                close=None,
                prev_close=None,
                limit_up=None,
                limit_down=None,
                derivation_rule_version=DERIVATION_RULE_VERSION,
                source_identity_sha256=self.identity.identity_sha256,
            )

        dataset = self._stocks[obid]
        names = dataset.dtype.names
        if names is None or tuple(names) != STOCKS_ALLOWED_FIELDS:
            raise RQAlphaOverlayError(
                f"stocks.h5 的 {obid} 字段与允许字段合同不一致: {names}"
            )
        datetimes = dataset["datetime"][:]
        target = day * 1_000_000
        index = int(np.searchsorted(datetimes, target))
        if index >= len(datetimes) or int(datetimes[index]) != target:
            # 硬条件 A：覆盖缺口失败关闭，不静默缩池、不用其他来源填补。
            raise RQAlphaOverlayError(
                f"{obid} 在 {session_date} 非停牌但无 RQAlpha 日线，"
                "覆盖缺口必须失败关闭"
            )
        bar = dataset[index]
        close = float(bar["close"])
        prev_close = float(bar["prev_close"])
        limit_up = float(bar["limit_up"])
        limit_down = float(bar["limit_down"])
        for label, value in (
            ("close", close),
            ("prev_close", prev_close),
            ("limit_up", limit_up),
            ("limit_down", limit_down),
        ):
            if not np.isfinite(value):
                raise RQAlphaOverlayError(
                    f"{obid}/{session_date} 的 {label} 非有限数，失败关闭"
                )
        if close <= 0.0 or prev_close <= 0.0:
            raise RQAlphaOverlayError(
                f"{obid}/{session_date} 的 close/prev_close 必须为正"
            )
        if limit_up < 0.0 or limit_down < 0.0:
            raise RQAlphaOverlayError(
                f"{obid}/{session_date} 的涨跌停价不得为负"
            )
        if limit_up > 0.0 and limit_down > 0.0 and limit_down >= limit_up:
            raise RQAlphaOverlayError(
                f"{obid}/{session_date} 的 limit_down >= limit_up，数据矛盾"
            )

        # 保守日线触及规则：只用显式非零限价；0 值日不判 locked。
        if limit_up > 0.0 and close >= limit_up:
            status = "LIMIT_UP_LOCKED"
        elif limit_down > 0.0 and close <= limit_down:
            status = "LIMIT_DOWN_LOCKED"
        else:
            status = "OPEN"

        return RQAlphaExecutionState(
            symbol=symbol,
            order_book_id=obid,
            session_date=session_date,
            status=status,
            is_st=is_st,
            lot_size=row.round_lot,
            close=close,
            prev_close=prev_close,
            limit_up=limit_up,
            limit_down=limit_down,
            derivation_rule_version=DERIVATION_RULE_VERSION,
            source_identity_sha256=self.identity.identity_sha256,
        )


def load_rqalpha_execution_overlay(
    extracted_root: str | Path = RQALPHA_EXTRACTED_ROOT,
    *,
    v3_export_root: str | Path = V3_EXPORT_ROOT,
    frozen_calendar_path: str | Path = FROZEN_CALENDAR_PATH,
    trusted_anchor: Mapping[str, Any] | None = None,
) -> RQAlphaExecutionOverlay:
    """加载只读 execution-state overlay；trusted_anchor 仅供测试注入。"""
    return RQAlphaExecutionOverlay(
        extracted_root,
        v3_export_root=v3_export_root,
        frozen_calendar_path=frozen_calendar_path,
        trusted_anchor=trusted_anchor,
    )


__all__ = [
    "DERIVATION_RULE_DESCRIPTION",
    "DERIVATION_RULE_VERSION",
    "EXECUTION_STATE_STATUSES",
    "FORBIDDEN_SEMANTICS",
    "FROZEN_CALENDAR_PATH",
    "OVERLAY_CONTRACT_FORMAT",
    "PERMITTED_SEMANTICS",
    "RQALPHA_EXTRACTED_ROOT",
    "RQAlphaExecutionOverlay",
    "RQAlphaExecutionState",
    "RQAlphaOverlayError",
    "RQAlphaOverlayIdentity",
    "STOCKS_ALLOWED_FIELDS",
    "ST_DATASET_SEMANTICS",
    "SUSPENDED_MISSING_DATASET_SEMANTICS",
    "TRUSTED_OVERLAY_ANCHOR",
    "V3_EXPORT_ROOT",
    "load_rqalpha_execution_overlay",
]
