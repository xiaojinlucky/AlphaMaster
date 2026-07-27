"""RND-04C：动态历史股票池的确定性日线组合 replay（工程证据层）。

本模块把四件既有组件串成一条确定性的逐日回放链，不新建任何平行实现：

- 决策时钟股票池：``portfolio_manager.universe`` 的受信历史沪深300合同
  （``load_csi300_historical_universe_contract``，本层固定 reconstructed 模式）。
- 决策门禁：``portfolio_manager.controller`` 的 signals 完整覆盖门
  （``_build_portfolio_decision``，本层只喂入确定性机制验证信号）。
- 虚拟执行：``portfolio_manager.execution`` 的 T+1/整手/费用/状态约束。
- 账本：``portfolio_manager.ledger`` 的不可篡改 SQLite 账本
  （本轮新增 replay 身份绑定表，见 ledger.record_replay_binding）。

语义边界（防洗白，逐条对应 2026-07-26 裁决与交接第 9 节第三步）：

1. 本层信号是 **engineering replay signal（机制验证信号）**：按股票代码
   升序并列等分，Top-N 即代码最小的 N 只。它不是策略、没有预测语义；
   replay 产物只作工程证据，禁止输出或引申任何策略绩效结论。
2. 两种价格身份严格分离：估值/成交价只用 FreeStockDB v3 qfq 冻结价格
   （FreeStockDB 身份）；执行状态（停牌/涨跌停/ST/整手）只用 RQAlpha
   execution-state overlay（RQAlpha 身份）。RQAlpha 价格永不进入估值、
   成交或信号；v3 价格永不用于限价触及判断。
3. RQAlpha 状态不进入决策时钟：信号只由股票池成员与 v3 可用性派生；
   决策日估值行情的 ``status``/``lot_size`` 是 ExecutionQuote 结构占位
   常量（``OPEN``/100），不承载任何执行状态声明。overlay 在决策日只被
   用作"v3 缺 bar 是否为显式停牌"的数据完整性闸门，不影响选股。
4. 失败关闭：时点成分含 quarantine/source_missing（含 990018）→ 当日
   失败关闭，不静默缩池；v3 缺 bar 且非显式停牌（169 个漏数日语义）→
   失败关闭，禁止用 RQAlpha 价格填补；[2009-12-31, 2010-01-29) 空窗由
   replay 域下限与 universe 层双重失败关闭。
5. LOCKED 拒单遵循 overlay 的"保守日线触及规则"
   （conservative_daily_touch_rule），不代表真实封单或盘口。
6. 裁决 04C 侧义务：生产 overlay 的 ``identity_sha256`` 固化为本模块
   常量 ``PRODUCTION_OVERLAY_IDENTITY_SHA256``，运行前必须比对一致，
   闭合 trusted_anchor 注入口的身份链（硬条件 B）。
7. 本层是日线 replay，不是真实交易，也不是 sealed OOS；使用 reconstructed
   历史合同（point_in_time_safe=False、sealed_oos_eligible=False），
   不得改述为 strict PIT。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from data_pipeline.rqalpha_execution_overlay import (
    DERIVATION_RULE_DESCRIPTION,
    DERIVATION_RULE_VERSION,
    FROZEN_CALENDAR_PATH,
    TRUSTED_OVERLAY_ANCHOR,
    V3_EXPORT_ROOT,
    RQAlphaExecutionOverlay,
    RQAlphaExecutionState,
    RQAlphaOverlayError,
)
from portfolio_manager.controller import (
    ModelSignalSnapshot,
    PortfolioPolicy,
    _build_portfolio_decision,
)
from portfolio_manager.execution import (
    AShareFeeSchedule,
    ExecutionQuote,
    VirtualAccount,
    account_snapshot_sha256,
    execute_portfolio_decision,
)
from portfolio_manager.ledger import (
    REPLAY_BINDING_VERSION,
    PortfolioDecisionLedger,
)
from portfolio_manager.universe import (
    UNIVERSE_QUERY_MODE_RECONSTRUCTED,
    HistoricalUniverseContract,
    UniverseAvailabilityError,
    load_csi300_historical_universe_contract,
)

# 与 universe.py 的代码内受信根策略保持同一字面（生产 CLI 用）。
from portfolio_manager.universe import (  # noqa: E501  # 私有常量复用，见下注释
    _CSI300_HISTORY_TRUST_POLICY as PRODUCTION_CSI300_TRUST_POLICY,
)

# ---------------------------------------------------------------------------
# 合同常量
# ---------------------------------------------------------------------------

REPLAY_CONTRACT_VERSION = "alphamaster_dynamic_daily_replay_v1"
REPLAY_RUN_MANIFEST_FORMAT = "alphamaster_dynamic_replay_run_manifest_v1"

# 裁决 04C 侧义务（硬条件 B 配套）：生产 bundle 的 overlay identity_sha256
# 固化为常量；运行前与真实加载的 overlay.identity.identity_sha256 比对，
# 任何 trusted_anchor 注入或成员文件替换都会在此失败关闭。
# 该值由 TRUSTED_OVERLAY_ANCHOR + overlay 模块语义常量确定性重算得出，
# 并由测试 test_production_overlay_identity_constant_recomputes 锁定。
PRODUCTION_OVERLAY_IDENTITY_SHA256 = (
    "eaf4d1459b5bb8bdd4e1839d5cad42b8f7de8601c5125d9173f64dbb8d878f7b"
)

# replay 可用域：受信历史股票池 2010-01-29 起可用（此前是 2009-12-31
# 空窗，见 universe.py 的 incomplete snapshot 策略）；上限为 RQAlpha
# bundle 截止日 2026-06-30。
REPLAY_DOMAIN_FIRST = "2010-01-29"
REPLAY_DOMAIN_LAST = "2026-06-30"

# 决策时钟股票池固定使用 reconstructed 模式：这些月度快照没有
# strict_available_at 证据，本层如实声明"事后重建"，不冒充 strict PIT。
REPLAY_UNIVERSE_QUERY_MODE = UNIVERSE_QUERY_MODE_RECONSTRUCTED

# engineering replay signal（机制验证信号）声明：写入每个绑定与运行清单。
ENGINEERING_SIGNAL_RULE_VERSION = "deterministic_code_sorted_topn_v1"
ENGINEERING_SIGNAL_DECLARATION = (
    "engineering replay signal（机制验证信号）：全体成员使用相同分数，"
    "横截面排名退化为股票代码升序，Top-N 即代码最小的 N 只；"
    "调样退出成员通过 model_exit 强制离场。该规则不是策略、无预测语义，"
    "replay 产物只作工程证据，不构成任何策略绩效证据。"
)

# 决策层行情来源标识：v3 qfq 冻结收盘价（FreeStockDB 身份）。
REPLAY_MARKET_SOURCE = "free_stockdb_v3_qfq_frozen_close"

# 决策日估值行情的结构占位常量：ExecutionQuote 要求 status/lot_size 字段，
# 但决策层禁止消费 RQAlpha 状态，也不需要整手语义（估值只用价格），
# 因此固定为占位值并在此显式声明，不承载任何执行状态判断。
DECISION_VALUATION_STATUS_PLACEHOLDER = "OPEN"
DECISION_VALUATION_LOT_SIZE_PLACEHOLDER = 100

# v3 D1 数据文件列合同（与 v3 manifest 的 data_contract.columns 一致）。
V3_DATA_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")

# 生产受信历史股票池根（交接第 5 节；G 盘只读）。
PRODUCTION_CSI300_HISTORY_ROOT = Path(
    r"G:\QuantData\free-stockdb\online_api\snapshots"
    r"\20260630_csi300_weight_history_v1"
)

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# 机制验证信号的固定分数：全体相同 → 校准分位恒为 1.0 → 排名由代码升序
# 决定（controller 的 tie-break 是股票代码）。
_ENGINEERING_RAW_SCORE = 1.0
_ENGINEERING_HISTORY_SCORES = (0.0, 0.5)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# 机制验证信号的常量身份：模型/校准历史都由声明文本确定性派生，
# 每个信号的 data_version 则绑定该股票 v3 数据文件的真实 SHA-256。
_ENGINEERING_MODEL_VERSION = _sha256_text(
    f"{ENGINEERING_SIGNAL_RULE_VERSION}:{ENGINEERING_SIGNAL_DECLARATION}"
)
_ENGINEERING_CALIBRATION_HISTORY_SHA256 = _sha256_text(
    _canonical_json(
        {
            "rule_version": ENGINEERING_SIGNAL_RULE_VERSION,
            "raw_score": _ENGINEERING_RAW_SCORE,
            "history_scores": list(_ENGINEERING_HISTORY_SCORES),
        }
    )
)


class DynamicReplayError(RuntimeError):
    """replay 链上的失败关闭（数据缺口、身份不一致、域越界等）。"""


def _iso_day(day: int) -> str:
    text = str(day)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _day_int(iso: str, label: str) -> int:
    if not isinstance(iso, str):
        raise DynamicReplayError(f"{label} 必须是 YYYY-MM-DD 文本")
    try:
        parsed = date.fromisoformat(iso)
    except ValueError as exc:
        raise DynamicReplayError(f"{label} 不是有效日期: {iso!r}") from exc
    if parsed.isoformat() != iso:
        raise DynamicReplayError(f"{label} 必须是规范 YYYY-MM-DD: {iso!r}")
    return int(iso.replace("-", ""))


def _bar_close_ts(day: int) -> int:
    """日线收盘时间戳：上海时区 15:00（controller 对 1d 周期强制校验）。"""
    text = str(day)
    closed = datetime(
        int(text[:4]), int(text[4:6]), int(text[6:8]), 15, 0, tzinfo=_SHANGHAI
    )
    return int(closed.timestamp())


@dataclass(frozen=True)
class DynamicReplayConfig:
    """一次确定性 replay 的完整可复算配置。"""

    start_date: str
    end_date: str
    top_k: int
    dropout_rank: int
    initial_cash: float
    run_label: str

    def validated(self) -> "DynamicReplayConfig":
        start = _day_int(self.start_date, "start_date")
        end = _day_int(self.end_date, "end_date")
        if start > end:
            raise DynamicReplayError("start_date 不得晚于 end_date")
        domain_first = _day_int(REPLAY_DOMAIN_FIRST, "REPLAY_DOMAIN_FIRST")
        domain_last = _day_int(REPLAY_DOMAIN_LAST, "REPLAY_DOMAIN_LAST")
        if start < domain_first:
            raise DynamicReplayError(
                f"start_date {self.start_date} 早于 replay 可用域下限 "
                f"{REPLAY_DOMAIN_FIRST}（[2009-12-31, 2010-01-29) 为受信"
                "历史股票池空窗，失败关闭，禁止补齐或回退）"
            )
        if end > domain_last:
            raise DynamicReplayError(
                f"end_date {self.end_date} 晚于 replay 可用域上限 "
                f"{REPLAY_DOMAIN_LAST}（RQAlpha bundle 截止日之后的执行"
                "状态被裁决禁止）"
            )
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise DynamicReplayError("top_k 必须是整数")
        if isinstance(self.dropout_rank, bool) or not isinstance(
            self.dropout_rank, int
        ):
            raise DynamicReplayError("dropout_rank 必须是整数")
        if self.top_k < 1 or self.dropout_rank < self.top_k:
            raise DynamicReplayError("必须满足 1 <= top_k <= dropout_rank")
        if (
            isinstance(self.initial_cash, bool)
            or not isinstance(self.initial_cash, (int, float))
            or not float(self.initial_cash) > 0.0
        ):
            raise DynamicReplayError("initial_cash 必须是正数")
        if not isinstance(self.run_label, str) or not self.run_label.strip():
            raise DynamicReplayError("run_label 必须是非空文本")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "top_k": self.top_k,
            "dropout_rank": self.dropout_rank,
            "initial_cash": float(self.initial_cash),
            "run_label": self.run_label,
        }


class FrozenV3PriceStore:
    """v3 qfq 冻结价格的只读存取（FreeStockDB 身份，逐文件哈希校验）。"""

    def __init__(
        self,
        export_root: str | Path,
        *,
        expected_manifest_sha256: str,
        expected_coverage_sha256: str,
    ) -> None:
        import pandas as pd

        root = Path(export_root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DynamicReplayError(f"v3 manifest 不存在: {manifest_path}")
        actual_manifest_sha = _sha256_file(manifest_path)
        if actual_manifest_sha != expected_manifest_sha256:
            raise DynamicReplayError("v3 manifest SHA-256 与受信身份不一致")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicReplayError("v3 manifest 不是合法 UTF-8 JSON") from exc
        coverage_meta = manifest.get("coverage_matrix")
        if (
            not isinstance(coverage_meta, dict)
            or coverage_meta.get("sha256") != expected_coverage_sha256
        ):
            raise DynamicReplayError(
                "v3 manifest 声明的 coverage matrix SHA-256 与受信身份不一致"
            )
        coverage_path = root / str(coverage_meta.get("relative_path"))
        if not coverage_path.is_file():
            raise DynamicReplayError(f"coverage matrix 不存在: {coverage_path}")
        if _sha256_file(coverage_path) != expected_coverage_sha256:
            raise DynamicReplayError("coverage matrix SHA-256 与受信身份不一致")
        frame = pd.read_parquet(
            coverage_path,
            columns=["code", "status", "data_relative_path", "data_sha256"],
        )
        rows: dict[str, dict[str, Any]] = {}
        for code, status, relative, sha in zip(
            frame["code"], frame["status"], frame["data_relative_path"],
            frame["data_sha256"], strict=True,
        ):
            if not isinstance(code, str) or code in rows:
                raise DynamicReplayError("coverage matrix 的 code 非法或重复")
            rows[code] = {
                "status": status,
                "data_relative_path": relative,
                "data_sha256": sha,
            }
        self._root = root
        self._rows = rows
        self._closes: dict[str, dict[int, float]] = {}
        self.manifest_sha256 = actual_manifest_sha
        self.coverage_matrix_sha256 = expected_coverage_sha256

    def status(self, symbol: str) -> str | None:
        row = self._rows.get(symbol)
        return None if row is None else str(row["status"])

    def data_identity(self, symbol: str) -> tuple[str, str]:
        """返回 (相对路径, SHA-256)；非 available 一律失败关闭。"""
        row = self._rows.get(symbol)
        if row is None or row["status"] != "available":
            raise DynamicReplayError(
                f"代码 {symbol} 不在 v3 available 交集内，价格层失败关闭"
            )
        relative = row["data_relative_path"]
        sha = row["data_sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(sha, str)
            or _SHA256_RE.fullmatch(sha) is None
        ):
            raise DynamicReplayError(f"代码 {symbol} 的 v3 数据身份字段非法")
        return relative, sha

    def _load_closes(self, symbol: str) -> dict[int, float]:
        import pandas as pd

        cached = self._closes.get(symbol)
        if cached is not None:
            return cached
        relative, expected_sha = self.data_identity(symbol)
        data_path = self._root / relative
        if not data_path.is_file():
            raise DynamicReplayError(f"v3 数据文件不存在: {data_path}")
        if _sha256_file(data_path) != expected_sha:
            raise DynamicReplayError(
                f"v3 数据文件 SHA-256 与 coverage matrix 不一致: {relative}"
            )
        frame = pd.read_parquet(data_path)
        if tuple(frame.columns) != V3_DATA_COLUMNS or frame.empty:
            raise DynamicReplayError(f"v3 数据文件列合同不匹配或为空: {relative}")
        stamps = pd.to_datetime(
            frame["time"], unit="s", utc=True
        ).dt.tz_convert("Asia/Shanghai")
        if not (stamps.dt.strftime("%H%M%S") == "150000").all():
            raise DynamicReplayError(
                f"v3 数据文件存在非 15:00 收盘时间戳: {relative}"
            )
        days = [int(text) for text in stamps.dt.strftime("%Y%m%d")]
        if days != sorted(set(days)):
            raise DynamicReplayError(
                f"v3 数据文件时间戳必须严格升序且无重复: {relative}"
            )
        closes = {
            day: float(value)
            for day, value in zip(days, frame["close"], strict=True)
        }
        self._closes[symbol] = closes
        return closes

    def session_close(self, symbol: str, day: int) -> float | None:
        """返回该交易日 qfq 收盘价；无 bar 返回 None（由调用方裁决语义）。"""
        closes = self._load_closes(symbol)
        value = closes.get(day)
        if value is None:
            return None
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise DynamicReplayError(
                f"{symbol}/{_iso_day(day)} 的 v3 close 非有限数，失败关闭"
            )
        if value <= 0.0:
            raise DynamicReplayError(
                f"{symbol}/{_iso_day(day)} 的 v3 close 非正（qfq 历史异常），"
                "失败关闭"
            )
        return value


def _load_frozen_calendar(
    path: str | Path,
    expected_sha256: str,
) -> tuple[int, ...]:
    """加载冻结 FreeStockDB 交易日历（主时钟），逐字节哈希校验。"""
    import pandas as pd

    calendar_path = Path(path)
    if not calendar_path.is_file():
        raise DynamicReplayError(f"冻结交易日历不存在: {calendar_path}")
    if _sha256_file(calendar_path) != expected_sha256:
        raise DynamicReplayError("冻结交易日历 SHA-256 与受信身份不一致")
    frame = pd.read_parquet(calendar_path)
    if list(frame.columns) != ["trade_date"] or frame.empty:
        raise DynamicReplayError("冻结交易日历列合同不匹配或为空")
    days = [
        int(stamp.strftime("%Y%m%d"))
        for stamp in pd.to_datetime(frame["trade_date"], errors="raise")
    ]
    if days != sorted(set(days)):
        raise DynamicReplayError("冻结交易日历必须严格升序无重复")
    return tuple(days)


class DynamicDailyReplay:
    """把历史股票池、执行状态 overlay 与四件既有组件串成确定性日线 replay。

    确定性与崩溃恢复：``run()`` 每次都从区间首日开始逐日重算；
    所有账本写入（决策/执行/绑定）都是内容寻址幂等的，因此中断后重跑
    同一配置会精确续跑——已存在的记录逐字节比对通过后跳过写入，
    不重复成交、不跳日；任何内容漂移都会失败关闭。
    """

    def __init__(
        self,
        config: DynamicReplayConfig,
        *,
        overlay: RQAlphaExecutionOverlay,
        ledger: PortfolioDecisionLedger,
        fee_schedule: AShareFeeSchedule,
        history_root: str | Path,
        trust_policy: str,
        v3_export_root: str | Path = V3_EXPORT_ROOT,
        calendar_path: str | Path = FROZEN_CALENDAR_PATH,
        expected_overlay_identity_sha256: str = (
            PRODUCTION_OVERLAY_IDENTITY_SHA256
        ),
        expected_v3_manifest_sha256: str = (
            TRUSTED_OVERLAY_ANCHOR["v3_manifest_sha256"]
        ),
        expected_v3_coverage_sha256: str = (
            TRUSTED_OVERLAY_ANCHOR["v3_coverage_matrix_sha256"]
        ),
        expected_calendar_sha256: str = (
            TRUSTED_OVERLAY_ANCHOR["frozen_calendar_sha256"]
        ),
    ) -> None:
        self.config = config.validated()
        if not isinstance(overlay, RQAlphaExecutionOverlay):
            raise DynamicReplayError("overlay 必须是 RQAlphaExecutionOverlay")
        if not isinstance(ledger, PortfolioDecisionLedger):
            raise DynamicReplayError("ledger 必须是 PortfolioDecisionLedger")
        if not isinstance(fee_schedule, AShareFeeSchedule):
            raise DynamicReplayError("fee_schedule 必须是 AShareFeeSchedule")
        fee_schedule.validated()

        # 裁决 04C 侧义务：运行前比对固化的 overlay identity 常量。
        if (
            not isinstance(expected_overlay_identity_sha256, str)
            or _SHA256_RE.fullmatch(expected_overlay_identity_sha256) is None
        ):
            raise DynamicReplayError(
                "expected_overlay_identity_sha256 必须是小写 SHA-256"
            )
        if overlay.identity.identity_sha256 != expected_overlay_identity_sha256:
            raise DynamicReplayError(
                "overlay identity_sha256 与固化常量不一致，拒绝 replay："
                f"{overlay.identity.identity_sha256} != "
                f"{expected_overlay_identity_sha256}"
            )
        # overlay 与价格层必须指向同一份 v3 导出身份，防止两套 v3 混用。
        if overlay.identity.v3_manifest_sha256 != expected_v3_manifest_sha256:
            raise DynamicReplayError(
                "overlay 绑定的 v3 manifest 身份与价格层期望不一致"
            )

        self.overlay = overlay
        self.ledger = ledger
        self.fee_schedule = fee_schedule
        self.price_store = FrozenV3PriceStore(
            v3_export_root,
            expected_manifest_sha256=expected_v3_manifest_sha256,
            expected_coverage_sha256=expected_v3_coverage_sha256,
        )
        self.calendar_days = _load_frozen_calendar(
            calendar_path, expected_calendar_sha256
        )
        self.calendar_sha256 = expected_calendar_sha256

        root = Path(history_root).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise DynamicReplayError(
                f"历史股票池 manifest 不存在: {manifest_path}"
            )
        self.history_root = root
        self.history_root_manifest_sha256 = _sha256_file(manifest_path)
        if not isinstance(trust_policy, str) or not trust_policy:
            raise DynamicReplayError("trust_policy 必须是非空文本")
        self.trust_policy = trust_policy

        self.policy = PortfolioPolicy(
            top_k=self.config.top_k,
            dropout_rank=self.config.dropout_rank,
            minimum_history=2,
        )
        # 信号 run_id 种子只绑定配置与机制规则，不含 overlay/历史根身份：
        # 未来状态追加（bundle/快照续档）不得改变历史决策身份。
        self._signal_seed = _sha256_text(
            _canonical_json(
                {
                    "replay_contract_version": REPLAY_CONTRACT_VERSION,
                    "config": self.config.to_dict(),
                    "engineering_signal_rule_version": (
                        ENGINEERING_SIGNAL_RULE_VERSION
                    ),
                }
            )
        )
        # replay_run_id 则绑定完整输入世界身份（含 overlay 与历史根），
        # 输入世界变化会得到新的 run 身份，但不改变既有决策/执行身份。
        self.replay_run_id = "AM-REPLAY-" + _sha256_text(
            _canonical_json(self._run_identity_payload())
        )[:24].upper()

    # -- 运行身份 -----------------------------------------------------------

    def _run_identity_payload(self) -> dict[str, Any]:
        return {
            "replay_contract_version": REPLAY_CONTRACT_VERSION,
            "config": self.config.to_dict(),
            "engineering_signal_rule_version": ENGINEERING_SIGNAL_RULE_VERSION,
            "universe_query_mode": REPLAY_UNIVERSE_QUERY_MODE,
            "trust_policy": self.trust_policy,
            "history_root_manifest_sha256": self.history_root_manifest_sha256,
            "overlay_identity_sha256": self.overlay.identity.identity_sha256,
            "v3_manifest_sha256": self.price_store.manifest_sha256,
            "v3_coverage_matrix_sha256": (
                self.price_store.coverage_matrix_sha256
            ),
            "frozen_calendar_sha256": self.calendar_sha256,
            "fee_schedule": self.fee_schedule.to_dict(),
        }

    # -- 时钟 ---------------------------------------------------------------

    def _window_days(self) -> tuple[int, ...]:
        start = _day_int(self.config.start_date, "start_date")
        end = _day_int(self.config.end_date, "end_date")
        days = tuple(d for d in self.calendar_days if start <= d <= end)
        if len(days) < 2:
            raise DynamicReplayError(
                "区间内不足两个交易日，无法构成决策日/执行日对"
            )
        return days

    # -- 股票池 -------------------------------------------------------------

    def _load_universe(self, as_of_day: int) -> HistoricalUniverseContract:
        try:
            contract = load_csi300_historical_universe_contract(
                self.history_root,
                as_of_date=_iso_day(as_of_day),
                mode=REPLAY_UNIVERSE_QUERY_MODE,
                trust_policy=self.trust_policy,
            )
        except UniverseAvailabilityError as exc:
            raise DynamicReplayError(
                f"决策日 {_iso_day(as_of_day)} 历史股票池失败关闭: {exc}"
            ) from exc
        # 禁止声明第 12 条：replay 不是 sealed OOS，也不是 strict PIT。
        if contract.point_in_time_safe or contract.sealed_oos_eligible:
            raise DynamicReplayError(
                "replay 只允许 reconstructed 历史合同，拒绝 strict/sealed 语义"
            )
        return contract

    def _coverage_gaps(self, symbols: tuple[str, ...]) -> dict[str, str]:
        """时点成分中不在 v3 available 交集内的覆盖缺口（硬条件 A）。"""
        gaps: dict[str, str] = {}
        for symbol in symbols:
            status = self.price_store.status(symbol)
            if status != "available":
                gaps[symbol] = (
                    "not_in_949_coverage" if status is None else status
                )
        return gaps

    # -- 价格解析（FreeStockDB 身份）----------------------------------------

    def _resolve_price(
        self,
        symbol: str,
        day: int,
        *,
        known_state: RQAlphaExecutionState | None = None,
    ) -> tuple[float, int]:
        """解析 (symbol, day) 的 v3 qfq 估值/成交价，返回 (价格, 基准日)。

        - 当日有 v3 bar：直接使用（基准日=当日）。
        - 当日无 bar：唯一允许的解释是 overlay 显式停牌；此时向前回溯，
          每个缺 bar 的交易日都必须被显式停牌解释，取最近有 bar 日的
          收盘价作估值基准（基准日如实记录，仍是 FreeStockDB 价格身份）。
        - 无 bar 且非停牌：即 FreeStockDB 漏数日语义（169 日组），
          失败关闭，禁止用 RQAlpha 价格填补。
        """
        close = self.price_store.session_close(symbol, day)
        if close is not None:
            return close, day
        state = known_state
        if state is None:
            state = self._derive_state(symbol, day)
        if state.status != "SUSPENDED":
            raise DynamicReplayError(
                f"{symbol}/{_iso_day(day)} 无 v3 qfq bar 且非显式停牌，"
                "属 FreeStockDB 漏数日语义，价格层失败关闭"
                "（禁止用 RQAlpha 价格填补）"
            )
        # 防御性显式检查（2026-07-27 审查 P2-⑤）：day 不在冻结日历时必须走
        # DynamicReplayError 失败关闭路径，不允许裸 ValueError 绕过清单语义。
        if day not in self.calendar_days:
            raise DynamicReplayError(
                f"{symbol}/{_iso_day(day)} 不在冻结交易日历内，价格层失败关闭"
            )
        position = self.calendar_days.index(day)
        for prev in reversed(self.calendar_days[:position]):
            close = self.price_store.session_close(symbol, prev)
            if close is not None:
                return close, prev
            prev_state = self._derive_state(symbol, prev)
            if prev_state.status != "SUSPENDED":
                raise DynamicReplayError(
                    f"{symbol}/{_iso_day(prev)} 无 v3 qfq bar 且非显式停牌，"
                    "回溯路径上的缺口不可解释，价格层失败关闭"
                )
        raise DynamicReplayError(
            f"{symbol}/{_iso_day(day)} 停牌回溯未找到任何 v3 qfq 基准价"
        )

    def _derive_state(self, symbol: str, day: int) -> RQAlphaExecutionState:
        try:
            return self.overlay.derive_execution_state(symbol, _iso_day(day))
        except RQAlphaOverlayError as exc:
            raise DynamicReplayError(
                f"overlay 执行状态失败关闭: {exc}"
            ) from exc

    # -- 信号（决策时钟，机制验证信号）--------------------------------------

    def _signal_run_id(self, day: int, symbol: str) -> str:
        digest = _sha256_text(f"{self._signal_seed}:{day}:{symbol}")[:8]
        return f"run_{day}T150000Z_{digest}"

    def _engineering_signals(
        self,
        universe: HistoricalUniverseContract,
        day: int,
        forced_exits: frozenset[str],
    ) -> tuple[tuple[ModelSignalSnapshot, ...], dict[str, str]]:
        """为股票池成员生成机制验证信号。

        v3 覆盖缺口成员不伪造信号（返回 skipped），由 controller 的
        signals 完整覆盖门失败关闭（裁决测试组 19）。
        """
        bar_ts = _bar_close_ts(day)
        session_date = _iso_day(day)
        signals: list[ModelSignalSnapshot] = []
        skipped: dict[str, str] = {}
        for symbol in universe.symbols:
            status = self.price_store.status(symbol)
            if status != "available":
                skipped[symbol] = (
                    "not_in_949_coverage" if status is None else status
                )
                continue
            _, data_sha = self.price_store.data_identity(symbol)
            signals.append(
                ModelSignalSnapshot(
                    run_id=self._signal_run_id(day, symbol),
                    symbol=symbol,
                    bar_ts=bar_ts,
                    session_date=session_date,
                    timeframe="1d",
                    market_source=REPLAY_MARKET_SOURCE,
                    raw_score=_ENGINEERING_RAW_SCORE,
                    requested_exposure=1.0,
                    confidence=1.0,
                    model_version=_ENGINEERING_MODEL_VERSION,
                    data_version=data_sha,
                    calibration_version=ENGINEERING_SIGNAL_RULE_VERSION,
                    calibration_history_sha256=(
                        _ENGINEERING_CALIBRATION_HISTORY_SHA256
                    ),
                    history_scores=_ENGINEERING_HISTORY_SCORES,
                    model_exit=symbol in forced_exits,
                )
            )
        return tuple(signals), skipped

    # -- 行情构造 -----------------------------------------------------------

    def _decision_quotes(
        self,
        account: VirtualAccount,
        day: int,
    ) -> tuple[tuple[ExecutionQuote, ...], dict[str, dict[str, Any]]]:
        """决策日估值行情：只覆盖当前持仓，价格为 v3 qfq（占位状态）。"""
        held = sorted({lot.symbol for lot in account.lots})
        quotes: list[ExecutionQuote] = []
        provenance: dict[str, dict[str, Any]] = {}
        for symbol in held:
            price, basis = self._resolve_price(symbol, day)
            relative, sha = self.price_store.data_identity(symbol)
            quotes.append(
                ExecutionQuote(
                    symbol=symbol,
                    session_date=_iso_day(day),
                    price=price,
                    status=DECISION_VALUATION_STATUS_PLACEHOLDER,
                    lot_size=DECISION_VALUATION_LOT_SIZE_PLACEHOLDER,
                )
            )
            provenance[symbol] = {
                "price": price,
                "price_basis_session": _iso_day(basis),
                "data_relative_path": relative,
                "data_sha256": sha,
                "status_is_placeholder": True,
            }
        return tuple(quotes), provenance

    def _execution_quotes(
        self,
        symbols: tuple[str, ...],
        day: int,
    ) -> tuple[tuple[ExecutionQuote, ...], dict[str, dict[str, Any]]]:
        """执行日行情：价格=v3 qfq（FreeStockDB 身份），状态/整手=overlay
        （RQAlpha 身份）；两种身份在 provenance 中分开记录，绝不混同。"""
        quotes: list[ExecutionQuote] = []
        provenance: dict[str, dict[str, Any]] = {}
        for symbol in sorted(symbols):
            state = self._derive_state(symbol, day)
            price, basis = self._resolve_price(
                symbol, day, known_state=state
            )
            relative, sha = self.price_store.data_identity(symbol)
            quotes.append(
                ExecutionQuote(
                    symbol=symbol,
                    session_date=_iso_day(day),
                    price=price,
                    # lot_size 必须显式用 overlay 的 round_lot（688 股=200，
                    # 绝不用 ExecutionQuote 默认 100）。
                    status=state.status,
                    lot_size=state.lot_size,
                )
            )
            provenance[symbol] = {
                "price": price,
                "price_basis_session": _iso_day(basis),
                "data_relative_path": relative,
                "data_sha256": sha,
                "execution_state": state.to_dict(),
            }
        return tuple(quotes), provenance

    # -- 账户度量 -----------------------------------------------------------

    @staticmethod
    def _held_shares(account: VirtualAccount) -> dict[str, int]:
        shares: dict[str, int] = {}
        for lot in account.lots:
            shares[lot.symbol] = shares.get(lot.symbol, 0) + lot.shares
        return shares

    def _current_weights(
        self,
        account: VirtualAccount,
        quotes: Mapping[str, ExecutionQuote],
    ) -> dict[str, float]:
        shares = self._held_shares(account)
        nav = float(account.cash) + sum(
            shares[symbol] * quotes[symbol].price for symbol in shares
        )
        if not nav > 0.0:
            raise DynamicReplayError("决策日净值必须大于 0")
        return {
            symbol: shares[symbol] * quotes[symbol].price / nav
            for symbol in sorted(shares)
        }

    # -- 主循环 -------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """执行（或幂等续跑）整个区间，返回运行清单 payload。"""
        days = self._window_days()
        self._check_ledger_run_affinity()

        account = VirtualAccount(cash=float(self.config.initial_cash))
        previous_bar_ts: int | None = None
        anchor_as_of: int | None = None
        day_records: list[dict[str, Any]] = []
        first_pair = True

        for decision_day, execution_day in zip(days[:-1], days[1:]):
            record = self._run_pair(
                decision_day=decision_day,
                execution_day=execution_day,
                account=account,
                previous_bar_ts=previous_bar_ts,
                anchor_as_of=anchor_as_of,
                bootstrap_account=first_pair,
            )
            account = record.pop("_account_after")
            previous_bar_ts = record.pop("_bar_ts")
            anchor_as_of = record.pop("_anchor_as_of")
            first_pair = False
            day_records.append(record)

        manifest = self._build_manifest(day_records, account)
        return manifest

    def _check_ledger_run_affinity(self) -> None:
        """一账本一 run：已有绑定必须属于同一 replay_run_id。

        崩溃窗口兼容：允许最后一笔执行暂缺绑定（写入执行与写入绑定之间
        中断），续跑时幂等补写；此外任何执行缺绑定都失败关闭。
        """
        records = self.ledger.list_executions()
        for position, record in enumerate(records):
            binding = self.ledger.get_replay_binding(record["execution_id"])
            if binding is None:
                if position != len(records) - 1:
                    raise DynamicReplayError(
                        f"执行 {record['execution_id']} 缺少 replay 绑定，"
                        "账本不完整，失败关闭"
                    )
                continue
            if binding["replay_run_id"] != self.replay_run_id:
                raise DynamicReplayError(
                    "账本已属于另一 replay run："
                    f"{binding['replay_run_id']} != {self.replay_run_id}"
                )

    def _run_pair(
        self,
        *,
        decision_day: int,
        execution_day: int,
        account: VirtualAccount,
        previous_bar_ts: int | None,
        anchor_as_of: int | None,
        bootstrap_account: bool,
    ) -> dict[str, Any]:
        holdings = frozenset(self._held_shares(account))

        # 1) 决策时钟股票池（as_of=决策日）。
        effective_universe = self._load_universe(decision_day)
        effective_symbols = frozenset(effective_universe.symbols)

        # 2) 时点覆盖缺口失败关闭（硬条件 A：不静默缩池）。
        effective_gaps = self._coverage_gaps(effective_universe.symbols)

        # 3) 调样过渡：持仓不再全部属于当日成分时，本日决策仍使用
        #    "持仓所属"的上一股票池（as_of 锚定日重查，结果确定），并对
        #    已退出成分强制 model_exit；持仓回到当日成分子集后恢复正常。
        transition_mode = not holdings.issubset(effective_symbols)
        if transition_mode:
            if anchor_as_of is None:
                raise DynamicReplayError(
                    "内部不变量被破坏：无历史锚却持有当日成分之外的股票"
                )
            universe = self._load_universe(anchor_as_of)
            if not holdings.issubset(frozenset(universe.symbols)):
                raise DynamicReplayError(
                    "内部不变量被破坏：持仓不属于锚定股票池"
                )
            forced_exits = frozenset(universe.symbols) - effective_symbols
            if effective_gaps:
                # 当日真实成分含覆盖缺口：即便过渡决策不直接消费这些
                # 成员，也必须失败关闭（时点成分覆盖缺口不因过渡豁免）。
                raise DynamicReplayError(
                    f"决策日 {_iso_day(decision_day)} 时点成分存在 v3 覆盖"
                    f"缺口，失败关闭（不静默缩池）: {effective_gaps}"
                )
            universe_as_of = anchor_as_of
        else:
            universe = effective_universe
            forced_exits = frozenset()
            universe_as_of = decision_day

        # 4) 机制验证信号：覆盖缺口成员不伪造信号，由 controller 的
        #    signals 完整覆盖门失败关闭（裁决测试组 19）。
        signals, skipped = self._engineering_signals(
            universe, decision_day, forced_exits
        )

        # 5) 决策日估值（FreeStockDB 身份；仅持仓股票）。
        decision_quotes, decision_provenance = self._decision_quotes(
            account, decision_day
        )
        quote_map = {quote.symbol: quote for quote in decision_quotes}
        current_weights = self._current_weights(account, quote_map)
        snapshot_sha = account_snapshot_sha256(account, decision_quotes)

        try:
            decision = _build_portfolio_decision(
                signals,
                universe=universe,
                current_weights=current_weights,
                account_snapshot_sha256=snapshot_sha,
                policy=self.policy,
                previous_decision_ts=previous_bar_ts,
            )
        except ValueError as exc:
            if skipped:
                raise DynamicReplayError(
                    f"决策日 {_iso_day(decision_day)} 时点成分存在 v3 覆盖"
                    f"缺口，signals 完整覆盖门失败关闭（不静默缩池）："
                    f"缺口={skipped}；controller: {exc}"
                ) from exc
            raise DynamicReplayError(
                f"决策日 {_iso_day(decision_day)} controller 门禁失败: {exc}"
            ) from exc

        self.ledger.record_decision(decision)

        # 6) 执行日（T+1）：状态用 overlay（RQAlpha 身份），价格用 v3 qfq。
        required = tuple(
            sorted(holdings | set(decision.selected_symbols))
        )
        execution_quotes, execution_provenance = self._execution_quotes(
            required, execution_day
        )
        try:
            result = execute_portfolio_decision(
                decision,
                execution_session=_iso_day(execution_day),
                account=account,
                decision_quotes=decision_quotes,
                quotes=execution_quotes,
                fee_schedule=self.fee_schedule,
            )
        except (RuntimeError, ValueError) as exc:
            raise DynamicReplayError(
                f"执行日 {_iso_day(execution_day)} 虚拟执行失败关闭: {exc}"
            ) from exc

        stored, _ = self.ledger.record_execution(
            result,
            decision_quotes=decision_quotes,
            execution_quotes=execution_quotes,
            fee_schedule=self.fee_schedule,
            bootstrap_account=bootstrap_account,
        )

        # 7) 全链身份绑定进账本（硬条件 B + 裁决测试组 17）。
        binding_payload = {
            "contract_version": REPLAY_BINDING_VERSION,
            "replay_contract_version": REPLAY_CONTRACT_VERSION,
            "replay_run_id": self.replay_run_id,
            "execution_id": stored["execution_id"],
            "decision_id": decision.decision_id,
            "decision_session": decision.session_date,
            "execution_session": _iso_day(execution_day),
            "universe": {
                "contract_sha256": decision.universe.contract_sha256,
                "universe_sha256": decision.universe.universe_sha256,
                "as_of_date": _iso_day(universe_as_of),
                "source_effective_date": (
                    decision.universe.source_effective_date
                ),
                "query_mode": decision.universe.query_mode,
                "constituent_count": decision.universe.constituent_count,
                "transition_mode": transition_mode,
                "forced_exit_symbols": sorted(forced_exits),
                "effective_universe_sha256": (
                    effective_universe.universe_sha256
                ),
            },
            "overlay_identity_sha256": (
                self.overlay.identity.identity_sha256
            ),
            "derivation_rule_version": DERIVATION_RULE_VERSION,
            "freestockdb_price_identity": {
                "v3_manifest_sha256": self.price_store.manifest_sha256,
                "v3_coverage_matrix_sha256": (
                    self.price_store.coverage_matrix_sha256
                ),
            },
            "decision_quote_provenance": decision_provenance,
            "execution_quote_provenance": execution_provenance,
            "engineering_signal": {
                "rule_version": ENGINEERING_SIGNAL_RULE_VERSION,
                "declaration": ENGINEERING_SIGNAL_DECLARATION,
                "market_source": REPLAY_MARKET_SOURCE,
            },
            "fee_schedule": self.fee_schedule.to_dict(),
            "execution_row_sha256": stored["row_sha256"],
        }
        self.ledger.record_replay_binding(binding_payload)

        order_events = [
            {
                "symbol": order.symbol,
                "side": order.side,
                "status": order.status,
                "reason": order.reason,
                "requested_shares": order.requested_shares,
                "filled_shares": order.filled_shares,
            }
            for order in result.orders
        ]
        return {
            "decision_session": decision.session_date,
            "execution_session": _iso_day(execution_day),
            "decision_id": decision.decision_id,
            "execution_id": stored["execution_id"],
            "universe_as_of": _iso_day(universe_as_of),
            "universe_effective_date": (
                decision.universe.source_effective_date
            ),
            "universe_sha256": decision.universe.universe_sha256,
            "transition_mode": transition_mode,
            "forced_exit_symbols": sorted(forced_exits),
            "selected_symbols": list(decision.selected_symbols),
            "orders": order_events,
            "_account_after": result.account_after,
            "_bar_ts": decision.bar_ts,
            "_anchor_as_of": universe_as_of,
        }

    # -- 运行清单 -----------------------------------------------------------

    def _build_manifest(
        self,
        day_records: list[dict[str, Any]],
        final_account: VirtualAccount,
    ) -> dict[str, Any]:
        order_total = sum(len(record["orders"]) for record in day_records)
        rejected = [
            {
                "execution_session": record["execution_session"],
                "symbol": order["symbol"],
                "side": order["side"],
                "reason": order["reason"],
            }
            for record in day_records
            for order in record["orders"]
            if order["status"] == "REJECTED"
        ]
        return {
            "format": REPLAY_RUN_MANIFEST_FORMAT,
            "replay_contract_version": REPLAY_CONTRACT_VERSION,
            "replay_run_id": self.replay_run_id,
            "run_label": self.config.run_label,
            "status": "COMPLETED",
            "declarations": {
                "engineering_signal": ENGINEERING_SIGNAL_DECLARATION,
                "not_real_trading": True,
                "not_sealed_oos": True,
                "universe_query_mode": REPLAY_UNIVERSE_QUERY_MODE,
                "universe_mode_note": (
                    "历史股票池为 reconstructed 事后重建合同，"
                    "point_in_time_safe=False，不得改述为 strict PIT"
                ),
                "locked_rule": DERIVATION_RULE_VERSION,
                "locked_rule_description": DERIVATION_RULE_DESCRIPTION,
                "performance_claim": (
                    "本清单只作工程证据，不构成任何策略绩效结论"
                ),
            },
            "config": self.config.to_dict(),
            "identities": self._run_identity_payload(),
            "days": day_records,
            "summary": {
                "pair_count": len(day_records),
                "order_count": order_total,
                "rejected_orders": rejected,
                "final_account": final_account.to_dict(),
            },
        }

    # -- 事后核验 -----------------------------------------------------------

    def verify(self) -> dict[str, int]:
        """重算核验整条 replay 链（裁决测试组 17 的全链版）。

        - 账本执行链自身由 ledger 的 12 入口失败关闭机制核验；
        - 每笔执行必须有绑定，且 overlay 身份等于固化常量；
        - 绑定中每个执行状态、价格与来源身份都用当前 overlay/v3 重算比对，
          任何 overlay 状态或来源声明的篡改都会在此检出。
        """
        records = self.ledger.list_executions()
        verified_states = 0
        for record in records:
            binding = self.ledger.get_replay_binding(record["execution_id"])
            if binding is None:
                raise DynamicReplayError(
                    f"执行 {record['execution_id']} 缺少 replay 绑定"
                )
            if binding["replay_run_id"] != self.replay_run_id:
                raise DynamicReplayError("绑定 replay_run_id 不一致")
            # 引擎构造时已强制 overlay.identity == 固化期望常量；
            # 这里再比对绑定记录，闭合"事后篡改绑定身份"的检出路径。
            if binding["overlay_identity_sha256"] != (
                self.overlay.identity.identity_sha256
            ):
                raise DynamicReplayError(
                    "绑定的 overlay identity 与运行前比对过的 overlay "
                    "身份不一致（来源声明篡改检出）"
                )
            price_identity = binding["freestockdb_price_identity"]
            if (
                price_identity["v3_manifest_sha256"]
                != self.price_store.manifest_sha256
                or price_identity["v3_coverage_matrix_sha256"]
                != self.price_store.coverage_matrix_sha256
            ):
                raise DynamicReplayError(
                    "绑定的 FreeStockDB 价格身份与当前 v3 导出不一致"
                )

            decision_payload = self.ledger.get_decision(
                record["decision_id"]
            )
            if decision_payload is None:
                raise DynamicReplayError("绑定引用的决策不存在")
            if (
                decision_payload["universe"]["contract_sha256"]
                != binding["universe"]["contract_sha256"]
            ):
                raise DynamicReplayError("绑定的 universe 合同身份不一致")

            # P1-2（2026-07-27 对抗审查修复）：绑定 payload 的副本字段逐项与
            # 权威源重算比对，堵住"改绑定字段 + 重算 payload 哈希"的自洽篡改。
            if binding["contract_version"] != REPLAY_BINDING_VERSION:
                raise DynamicReplayError("绑定 contract_version 篡改检出")
            if binding["replay_contract_version"] != REPLAY_CONTRACT_VERSION:
                raise DynamicReplayError(
                    "绑定 replay_contract_version 篡改检出"
                )
            if binding["derivation_rule_version"] != DERIVATION_RULE_VERSION:
                raise DynamicReplayError(
                    "绑定 derivation_rule_version 篡改检出"
                )
            if binding["engineering_signal"] != {
                "rule_version": ENGINEERING_SIGNAL_RULE_VERSION,
                "declaration": ENGINEERING_SIGNAL_DECLARATION,
                "market_source": REPLAY_MARKET_SOURCE,
            }:
                raise DynamicReplayError("绑定 engineering_signal 篡改检出")
            if binding["fee_schedule"] != self.fee_schedule.to_dict():
                raise DynamicReplayError("绑定 fee_schedule 篡改检出")

            bound_universe = binding["universe"]
            decision_universe = decision_payload["universe"]
            for key in (
                "universe_sha256",
                "as_of_date",
                "source_effective_date",
                "query_mode",
                "constituent_count",
            ):
                if key in decision_universe and (
                    bound_universe[key] != decision_universe[key]
                ):
                    raise DynamicReplayError(
                        f"绑定 universe.{key} 与决策账本记录不一致（篡改检出）"
                    )

            dec_day = _day_int(record["decision_session"], "decision_session")
            recomputed_effective = self._load_universe(dec_day)
            if bound_universe["effective_universe_sha256"] != (
                recomputed_effective.universe_sha256
            ):
                raise DynamicReplayError(
                    "绑定 effective_universe_sha256 与按决策日重算不一致"
                    "（篡改检出）"
                )
            decision_symbols = frozenset(decision_universe["symbols"])
            recomputed_effective_symbols = frozenset(
                recomputed_effective.symbols
            )
            if bound_universe["transition_mode"]:
                expected_forced = sorted(
                    decision_symbols - recomputed_effective_symbols
                )
            else:
                # 非过渡：决策股票池必须就是当日股票池，且无强制离场。
                if bound_universe["universe_sha256"] != (
                    recomputed_effective.universe_sha256
                ):
                    raise DynamicReplayError(
                        "绑定 transition_mode 与重算不一致（篡改检出）"
                    )
                expected_forced = []
            if (
                list(bound_universe["forced_exit_symbols"])
                != expected_forced
            ):
                raise DynamicReplayError(
                    "绑定 forced_exit_symbols 与重算不一致（篡改检出）"
                )

            execution_session = record["execution_session"]
            execution_day = _day_int(execution_session, "execution_session")
            input_quotes = {
                quote["symbol"]: quote
                for quote in record["input"]["execution_quotes"]
            }
            bound_provenance = binding["execution_quote_provenance"]
            if set(input_quotes) != set(bound_provenance):
                raise DynamicReplayError(
                    "绑定的执行行情覆盖集合与账本输入不一致"
                )
            for symbol, entry in bound_provenance.items():
                state = self._derive_state(symbol, execution_day)
                if state.to_dict() != entry["execution_state"]:
                    raise DynamicReplayError(
                        f"{symbol}/{execution_session} 的 overlay 执行状态"
                        "与绑定记录不一致（状态篡改检出）"
                    )
                price, basis = self._resolve_price(
                    symbol, execution_day, known_state=state
                )
                quote = input_quotes[symbol]
                if (
                    quote["price"] != price
                    or entry["price"] != price
                    or entry["price_basis_session"] != _iso_day(basis)
                    or quote["status"] != state.status
                    or quote["lot_size"] != state.lot_size
                ):
                    raise DynamicReplayError(
                        f"{symbol}/{execution_session} 的价格或状态身份"
                        "与重算不一致（价格/状态篡改检出）"
                    )
                relative, sha = self.price_store.data_identity(symbol)
                if (
                    entry["data_relative_path"] != relative
                    or entry["data_sha256"] != sha
                ):
                    raise DynamicReplayError(
                        f"{symbol} 的 v3 数据文件身份与绑定不一致"
                    )
                verified_states += 1

            decision_session = record["decision_session"]
            decision_day = _day_int(decision_session, "decision_session")
            decision_quotes = {
                quote["symbol"]: quote
                for quote in record["input"]["decision_quotes"]
            }
            bound_decision = binding["decision_quote_provenance"]
            if set(decision_quotes) != set(bound_decision):
                raise DynamicReplayError(
                    "绑定的决策估值覆盖集合与账本输入不一致"
                )
            for symbol, entry in bound_decision.items():
                price, basis = self._resolve_price(symbol, decision_day)
                quote = decision_quotes[symbol]
                if (
                    quote["price"] != price
                    or entry["price"] != price
                    or entry["price_basis_session"] != _iso_day(basis)
                    or quote["status"]
                    != DECISION_VALUATION_STATUS_PLACEHOLDER
                    or quote["lot_size"]
                    != DECISION_VALUATION_LOT_SIZE_PLACEHOLDER
                ):
                    raise DynamicReplayError(
                        f"{symbol}/{decision_session} 的决策估值身份"
                        "与重算不一致"
                    )
        return {
            "execution_count": len(records),
            "verified_state_count": verified_states,
        }


__all__ = [
    "DECISION_VALUATION_LOT_SIZE_PLACEHOLDER",
    "DECISION_VALUATION_STATUS_PLACEHOLDER",
    "DynamicDailyReplay",
    "DynamicReplayConfig",
    "DynamicReplayError",
    "ENGINEERING_SIGNAL_DECLARATION",
    "ENGINEERING_SIGNAL_RULE_VERSION",
    "FrozenV3PriceStore",
    "PRODUCTION_CSI300_HISTORY_ROOT",
    "PRODUCTION_CSI300_TRUST_POLICY",
    "PRODUCTION_OVERLAY_IDENTITY_SHA256",
    "REPLAY_CONTRACT_VERSION",
    "REPLAY_DOMAIN_FIRST",
    "REPLAY_DOMAIN_LAST",
    "REPLAY_MARKET_SOURCE",
    "REPLAY_RUN_MANIFEST_FORMAT",
    "REPLAY_UNIVERSE_QUERY_MODE",
]
