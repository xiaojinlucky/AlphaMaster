"""
strategy_manager/runner.py — MT5 策略主循环控制器（回测对标版）

核心改动（vs 旧版本）：
  1. 信号改为 tanh 连续仓位，与 backtest.py 完全一致（Config.SIGNAL_MODE）
  2. 入场/出场统一为「信号翻转驱动」(_reconcile_positions)
  3. 支持做空，多/空均可反手
  4. K 线收盘触发调仓（REBALANCE_ON_BAR_CLOSE=True），消除时间偏差
  5. EXIT_MODE 控制是否叠加风控层（signal / risk / hybrid）
  6. MAX_OPEN_POSITIONS=None 表示不限制，严格对标回测
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
import time
from numbers import Real
from pathlib import Path

import torch
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

    class _MT5Stub:
        def shutdown(self) -> None:
            pass

    mt5 = _MT5Stub()  # type: ignore[assignment]

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from execution.price_feed import MT5PriceFeed
try:
    from execution.trader import MT5Trader
except ImportError:
    # execution/trader.py（真正下单的模块）已被移除：本项目不需要自动交易功能。
    # 用占位类代替，保证 strategy_manager.runner 仍可被导入（训练/回测/测试依赖它），
    # 但任何试图真正连接/下单的调用都会明确报错，而不是静默假装成功。
    class MT5Trader:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._connected = False

        def _unavailable(self, action: str):
            raise RuntimeError(
                f"execution.trader 已被移除（本项目不提供自动交易功能）："
                f"无法执行 {action}。MT5StrategyRunner 仅可用于信号计算，不能实盘下单。"
            )

        def connect(self):
            self._unavailable("connect()")

        def close(self):
            pass

        def get_account_info(self):
            return None

        def get_positions(self, symbol=None, magic=None):
            return []

        def buy(self, symbol, lot):
            self._unavailable("buy()")

        def sell(self, symbol, lot):
            self._unavailable("sell()")

        def open_short(self, symbol, lot):
            self._unavailable("open_short()")

        def close_position(self, symbol, lot, direction, ticket=0):
            self._unavailable("close_position()")

        def close_all_positions(self, symbol, magic=None, *, filter_magic=True):
            self._unavailable("close_all_positions()")
from model_core.vm import StackVM
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import FORMULA_VOCAB, VocabVersionMismatchError
from strategy_manager.portfolio import MT5PortfolioManager
from strategy_manager.risk import MT5RiskEngine
from strategy_manager.signal import (
    compute_target_positions,
    target_to_direction,
    reconcile_action,
    HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, REVERSE_TO_LONG, REVERSE_TO_SHORT,
)

_LOOP_INTERVAL: int = 60
RUNNER_POSITION_CONTRACT_VERSION = "direction_entry_size_hold_v1"
_SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def validate_strategy_symbol(symbol: object) -> str:
    """校验券商品种名，禁止把它解释成策略文件路径。"""
    if not isinstance(symbol, str) or _SYMBOL_PATTERN.fullmatch(symbol) is None:
        raise ValueError(f"非法交易品种名: {symbol!r}")
    if ".." in symbol:
        raise ValueError(f"交易品种名不得包含 '..': {symbol!r}")
    return symbol


def _strategy_path_within(root: Path, *parts: str) -> Path:
    """解析策略路径，并确认普通路径或符号链接都没有逃出根目录。"""
    boundary = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"策略路径逃出根目录: {candidate}") from exc
    return candidate


def _validate_formula_tokens(value: object) -> list[int]:
    """严格校验公式 token 类型、范围与逆波兰表达式栈结构。"""
    if not isinstance(value, list) or not value:
        raise ValueError("公式必须是非空 list")

    formula: list[int] = []
    stack_depth = 0
    arity_map = StackVM().arity_map
    for index, token in enumerate(value):
        if type(token) is not int:
            raise ValueError(
                f"formula[{index}] 必须是 int，实际为 {type(token).__name__}"
            )
        if not 0 <= token < FORMULA_VOCAB.size:
            raise ValueError(
                f"formula[{index}]={token} 超出词表范围 [0, {FORMULA_VOCAB.size})"
            )
        if token < FORMULA_VOCAB.operator_offset:
            stack_depth += 1
        else:
            arity = arity_map[token]
            if stack_depth < arity:
                raise ValueError(
                    f"formula[{index}] 算子缺少操作数：需要 {arity}，"
                    f"当前栈深 {stack_depth}"
                )
            stack_depth = stack_depth - arity + 1
        formula.append(token)

    if stack_depth != 1:
        raise ValueError(f"公式结束时栈深必须为 1，实际为 {stack_depth}")
    return formula


def _load_contract_formula(path: Path) -> list[int] | None:
    """读取可用于真实信号的公式；版本、合同或分数不合格时拒绝。"""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("策略顶层不是对象")
        FORMULA_VOCAB.verify(data.get("vocab_version"))
        if data.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
            raise ValueError("评分合同不兼容")
        raw_formula = (
            data["formula"]
            if "formula" in data
            else data.get("formula_tokens")
        )
        formula = _validate_formula_tokens(raw_formula)
        score = data.get("best_score")
        if score is None:
            score = data.get("train_best_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
            or float(score) <= 0.0
        ):
            raise ValueError(f"best_score={score!r} 不是严格正有限数")
        return formula
    except (json.JSONDecodeError, OSError, TypeError, ValueError, VocabVersionMismatchError) as exc:
        logger.warning(f"[Runner] {path.name}: 加载失败 {exc}")
        return None


def load_symbol_formulas(
    symbols: list[str],
    *,
    strategies_dir: Path | None = None,
) -> dict[str, list[list[int]]]:
    """按 Runner 的真实优先级加载并去重每个品种的全部有效公式。"""
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("请求品种必须是非空 list")
    checked_symbols = [validate_strategy_symbol(symbol) for symbol in symbols]
    if len(set(checked_symbols)) != len(checked_symbols):
        raise ValueError("请求品种存在重复")

    root = strategies_dir or Path("strategies")
    forex_group = {"EURUSD", "USDJPY"}
    metals_comm_group = {"XAUUSD", "AAVUSD", "COCOA.c"}
    loaded: dict[str, list[list[int]]] = {}

    for symbol in checked_symbols:
        formulas: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        def add(path: Path, label: str) -> None:
            formula = _load_contract_formula(path)
            if formula is None:
                return
            key = tuple(formula)
            if key in seen:
                return
            seen.add(key)
            formulas.append(formula)
            logger.info(f"[Runner] {symbol}: 加载公式 [{label}] {formula}")

        add(
            _strategy_path_within(root, f"best_{symbol}.json"),
            f"best_{symbol}",
        )
        if symbol in forex_group:
            add(
                _strategy_path_within(root, "best_forex.json"),
                "best_forex(v2)",
            )
            add(
                _strategy_path_within(
                    root,
                    "archive",
                    "best_forex_20250705_pre_refactor.json",
                ),
                "archive_forex_v1",
            )
        if symbol in metals_comm_group:
            add(
                _strategy_path_within(root, "best_metals_comm.json"),
                "best_metals_comm(v2)",
            )

        if formulas:
            loaded[symbol] = formulas

    if loaded:
        missing = [symbol for symbol in checked_symbols if symbol not in loaded]
        if missing:
            raise ValueError(
                "以下请求品种没有有效公式，拒绝部分策略集合启动: "
                + ", ".join(missing)
            )
        return loaded

    raise FileNotFoundError(
        f"请求品种均无有效策略文件: {', '.join(checked_symbols)}"
    )


def formula_set_sha256(formulas: list[list[int]]) -> str:
    """计算 Runner 有序公式集合的稳定身份。"""
    validated = [_validate_formula_tokens(formula) for formula in formulas]
    payload = json.dumps(
        validated,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_formula_set_hashes(
    formulas_by_symbol: dict[str, list[list[int]]],
    expected_hashes: dict[str, str],
) -> None:
    """确认内存中即将执行的公式集合与已审核身份完全一致。"""
    actual_symbols = set(formulas_by_symbol)
    expected_symbols = set(expected_hashes)
    if actual_symbols != expected_symbols:
        missing = sorted(expected_symbols - actual_symbols)
        extra = sorted(actual_symbols - expected_symbols)
        raise ValueError(
            f"公式集合品种不一致: missing={missing}, extra={extra}"
        )
    for symbol, formulas in formulas_by_symbol.items():
        actual = formula_set_sha256(formulas)
        if actual != expected_hashes[symbol]:
            raise ValueError(
                f"{symbol} 当前公式集合与扫描报告不一致，必须重新扫描"
            )


def compute_formula_set_positions(
    vm: StackVM,
    formulas: list[list[int]],
    feature: torch.Tensor,
    *,
    symbol: str,
    min_exposure: float,
) -> torch.Tensor:
    """严格执行 Runner 的完整公式集合，返回全时间序列目标仓位。"""
    if not formulas:
        raise ValueError(f"{symbol}: 公式集合为空")
    if not isinstance(feature, torch.Tensor) or feature.ndim != 3:
        raise ValueError(f"{symbol}: feature 必须是 [N, F, T] 张量")
    if feature.shape[0] < 1 or feature.shape[-1] < 1:
        raise ValueError(f"{symbol}: feature 的 N 和 T 必须严格大于 0")
    if (
        isinstance(min_exposure, bool)
        or not isinstance(min_exposure, Real)
        or not math.isfinite(float(min_exposure))
        or not 0.0 <= float(min_exposure) <= 1.0
    ):
        raise ValueError(f"{symbol}: min_exposure 必须是 [0, 1] 有限数")

    signals: list[torch.Tensor] = []
    expected_shape = (int(feature.shape[0]), int(feature.shape[-1]))
    for index, formula in enumerate(formulas):
        validated = _validate_formula_tokens(formula)
        raw = vm.execute(validated, feature)
        if raw is None:
            raise RuntimeError(f"{symbol} formula[{index}]: StackVM 执行失败")
        if not isinstance(raw, torch.Tensor) or raw.ndim != 2:
            raise RuntimeError(
                f"{symbol} formula[{index}]: StackVM 输出必须是 [N, T] 张量"
            )
        if tuple(raw.shape) != expected_shape:
            raise RuntimeError(
                f"{symbol} formula[{index}]: StackVM 输出形状 "
                f"{tuple(raw.shape)} != {expected_shape}"
            )
        if not torch.isfinite(raw).all():
            raise RuntimeError(
                f"{symbol} formula[{index}]: StackVM 输出含非有限值"
            )
        signals.append(torch.tanh(raw))

    target = torch.stack(signals, dim=0).mean(dim=0)
    if min_exposure > 0:
        target = torch.where(
            target.abs() >= min_exposure,
            target,
            torch.zeros_like(target),
        )
    return target


def apply_runner_position_state(
    target: torch.Tensor,
    *,
    min_exposure: float,
    fixed_exposure: bool,
) -> torch.Tensor:
    """把信号转换为 Runner 实际的“方向变化才重开仓”持仓序列。"""
    if not isinstance(target, torch.Tensor) or target.ndim != 2:
        raise ValueError("target 必须是 [N, T] 张量")
    if not torch.isfinite(target).all():
        raise ValueError("target 含非有限值")
    if (
        isinstance(min_exposure, bool)
        or not isinstance(min_exposure, Real)
        or not math.isfinite(float(min_exposure))
        or not 0.0 <= float(min_exposure) <= 1.0
    ):
        raise ValueError("min_exposure 必须是 [0, 1] 有限数")

    executed = torch.zeros_like(target)
    for row in range(target.shape[0]):
        current_direction = 0
        current_position = 0.0
        for column in range(target.shape[1]):
            value = float(target[row, column].item())
            desired_direction = (
                1
                if value >= min_exposure
                else (-1 if value <= -min_exposure else 0)
            )
            if desired_direction != current_direction:
                current_direction = desired_direction
                if desired_direction == 0:
                    current_position = 0.0
                elif fixed_exposure:
                    current_position = float(desired_direction)
                else:
                    current_position = value
            executed[row, column] = current_position
    return executed


def compute_latest_formula_target(
    vm: StackVM,
    formulas: list[list[int]],
    feature: torch.Tensor,
    *,
    symbol: str,
    min_exposure: float,
) -> float:
    """执行并平均 Runner 实际采用的多公式最新信号。"""
    positions = compute_formula_set_positions(
        vm,
        formulas,
        feature,
        symbol=symbol,
        min_exposure=min_exposure,
    )
    return float(positions[0, -1].item())


class _DryRunTraderProxy:
    """只放行明确的只读/生命周期方法，其余调用默认拦截。"""

    _ALLOWED_CALLS = frozenset(
        {
            "connect",
            "close",
            "get_account_info",
            "get_positions",
        }
    )

    def __init__(self, trader: object) -> None:
        self._trader = trader

    def __getattr__(self, name: str):
        attr = getattr(self._trader, name)
        if not callable(attr) or name in self._ALLOWED_CALLS:
            return attr

        def blocked(*args, **kwargs):
            logger.warning(
                f"[Runner] DRY RUN：已拦截非只读交易调用 {name}"
            )
            return False

        return blocked


class MT5StrategyRunner:
    """同步策略主循环控制器（回测对标版）。

    与旧版本关键差异：
    - 使用 compute_target_positions() 替代 sigmoid+阈值
    - _reconcile_positions() 替代 _scan_for_entries()
    - K 线收盘触发，消除回测-实盘时间偏差
    - 支持做空与反手
    - EXIT_MODE 控制风控叠加
    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
        expected_formula_set_sha256: dict[str, str] | None = None,
        min_trade_exposure: float | None = None,
    ) -> None:
        # ── 加载策略：支持每品种多公式（信号取平均合并）──────────────
        try:
            self.symbol_formulas_multi = load_symbol_formulas(Config.SYMBOLS)
            if expected_formula_set_sha256 is not None:
                verify_formula_set_hashes(
                    self.symbol_formulas_multi,
                    expected_formula_set_sha256,
                )
        except (FileNotFoundError, ValueError) as exc:
            logger.critical(f"{exc}。请先用当前合同重新训练。")
            sys.exit(1)
        self.symbol_formulas = {
            symbol: formulas[0]
            for symbol, formulas in self.symbol_formulas_multi.items()
        }
        self.execution_symbols = tuple(Config.SYMBOLS)

        # ── 打印加载汇总 ─────────────────────────────────────────────
        for sym, fmls in self.symbol_formulas_multi.items():
            logger.success(f"[Runner] {sym}: {len(fmls)} 条公式已加载")

        # 向后兼容：取第一个品种的第一条公式
        self.formula = next(iter(self.symbol_formulas.values()))

        self.vm        = StackVM()
        self.portfolio = MT5PortfolioManager()
        self.risk      = MT5RiskEngine()
        trader = MT5Trader()
        self.trader = _DryRunTraderProxy(trader) if dry_run else trader
        self.dry_run = dry_run
        self.min_trade_exposure = (
            float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05))
            if min_trade_exposure is None
            else float(min_trade_exposure)
        )
        if (
            not math.isfinite(self.min_trade_exposure)
            or not 0.0 <= self.min_trade_exposure <= 1.0
        ):
            raise ValueError("MIN_TRADE_EXPOSURE 必须是 [0, 1] 有限数")

        self._fetcher: MT5DataFetcher | None       = None
        self._data_manager: MT5DataManager | None  = None
        self._last_refresh: float                   = 0.0
        self._last_bar_time: torch.Tensor | None    = None


    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """同步主循环。

        流程：
            1. 连接 MT5 终端
            2. while True:
               a. 检查停止信号
               b. 按需刷新数据
               c. 若 REBALANCE_ON_BAR_CLOSE=True，只在新 K 线收盘时调仓
               d. 同步 MT5 仓位
               e. 调仓（_reconcile_positions）
               f. 若 EXIT_MODE in ('risk','hybrid')，叠加风控监控
               g. 休眠
        """
        logger.info("[Runner] Starting MT5StrategyRunner (backtest-parity mode)...")
        logger.info(f"  SIGNAL_MODE={Config.SIGNAL_MODE}  EXIT_MODE={Config.EXIT_MODE}  "
                    f"MAX_OPEN_POSITIONS={Config.MAX_OPEN_POSITIONS}  "
                    f"REBALANCE_ON_BAR_CLOSE={Config.REBALANCE_ON_BAR_CLOSE}")

        try:
            self.trader.connect()
        except (ConnectionError, RuntimeError) as exc:
            logger.critical(f"[Runner] Cannot connect MT5 trader: {exc}")
            sys.exit(1)

        self._fetcher = MT5DataFetcher()
        try:
            self._fetcher.connect()
        except ConnectionError as exc:
            logger.critical(f"[Runner] Cannot connect MT5 fetcher: {exc}")
            sys.exit(1)

        self._data_manager = MT5DataManager(
            self._fetcher,
            require_all_symbols=True,
        )
        try:
            self._data_manager.load(list(self.execution_symbols))
            self._last_refresh = time.time()
        except Exception as exc:
            raise RuntimeError(
                f"[Runner] Initial data load failed: {exc}"
            ) from exc

        logger.info("[Runner] MT5 connections established. Entering main loop.")

        # ── 启动时立即执行一次调仓（不等下一根 K 线收盘）────────────────
        # 原因：程序刚启动时持仓状态未知，应立即同步信号与仓位，
        # 而不是等最多 1 小时才做第一次动作。
        # 同时初始化 _last_bar_time，避免第一个正常循环被误判为 new_bar。
        logger.info("[Runner] 启动后立即执行初始调仓...")
        self._run_initial_reconcile()

        while True:
            loop_start = time.time()

            # a. 停止信号
            if self._handle_stop_signal():
                logger.info("[Runner] Stop signal detected. Exiting.")
                break

            # b. 数据刷新
            refresh_ok = True
            if time.time() - self._last_refresh >= Config.DATA_REFRESH_INTERVAL:
                try:
                    self._data_manager.reload()
                    self._last_refresh = time.time()
                    logger.info("[Runner] Data refreshed.")
                except Exception as exc:
                    logger.error(f"[Runner] Data reload failed: {exc}")
                    refresh_ok = False

            # c. 检测新 K 线收盘
            new_bar = False
            if refresh_ok:
                if Config.REBALANCE_ON_BAR_CLOSE:
                    new_bar = self._has_new_closed_bar()
                else:
                    new_bar = True

            # d. 同步 MT5 仓位
            try:
                self.portfolio.sync_from_mt5()
            except Exception as exc:
                logger.warning(f"[Runner] Portfolio sync failed: {exc}")

            if new_bar:
                # e. 计算信号并对账调仓
                targets = self._compute_targets()
                if targets is not None:
                    try:
                        self._reconcile_positions(targets)
                    except Exception as exc:
                        logger.error(f"[Runner] _reconcile_positions raised: {exc}")
            else:
                logger.debug("[Runner] Same bar, skipping rebalance.")

            # f. 风控监控（可选叠加层）
            if Config.EXIT_MODE in ("risk", "hybrid"):
                try:
                    self._monitor_positions()
                except Exception as exc:
                    logger.error(f"[Runner] _monitor_positions raised: {exc}")

            # g. 休眠
            elapsed = time.time() - loop_start
            sleep_t = max(10, _LOOP_INTERVAL - elapsed)
            logger.info(f"[Runner] Cycle {elapsed:.2f}s. Sleep {sleep_t:.2f}s.")
            time.sleep(sleep_t)

    def shutdown(self) -> None:
        logger.info("[Runner] Shutting down...")
        try:
            if self._fetcher is not None:
                self._fetcher.shutdown()
        except Exception as exc:
            logger.warning(f"[Runner] fetcher.shutdown() raised: {exc}")
        mt5.shutdown()
        logger.info("[Runner] Stopped.")


    # ──────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────

    def _handle_stop_signal(self) -> bool:
        stop_path = Config.STOP_SIGNAL
        if not os.path.exists(stop_path):
            return False
        logger.warning(f"[Runner] STOP_SIGNAL detected at '{stop_path}'.")
        try:
            with open(stop_path, "w", encoding="utf-8") as f:
                f.write("STOPPED")
        except OSError as exc:
            logger.warning(f"[Runner] Failed to mark stop signal: {exc}")
        return True

    def _has_new_closed_bar(self) -> bool:
        """只有完整、同形且明确变化的已收盘时间戳才能触发调仓。"""
        if self._data_manager is None:
            return False
        try:
            current = self._data_manager.bar_time
            if (
                not isinstance(current, torch.Tensor)
                or current.ndim != 1
                or current.shape[0]
                != len(
                    getattr(
                        self,
                        "execution_symbols",
                        tuple(Config.SYMBOLS),
                    )
                )
                or current.dtype != torch.int64
                or bool((current <= 0).any().item())
            ):
                raise ValueError(
                    "bar_time 必须是完整交易品种集合的严格正 int64 向量"
                )
            if self._last_bar_time is None:
                self._last_bar_time = current.clone()
                return False
            if current.shape != self._last_bar_time.shape:
                raise ValueError("bar_time 与上次快照形状不一致")
            if bool((current == self._last_bar_time).all().item()):
                return False
            if not bool((current > self._last_bar_time).all().item()):
                raise ValueError("bar_time 必须对全部品种严格向前推进")
            changed = True
            if changed:
                self._last_bar_time = current.clone()
            return changed
        except Exception as exc:
            logger.error(f"[Runner] bar_time 校验失败，本轮禁止调仓: {exc}")
            return False

    def _run_initial_reconcile(self) -> bool:
        """建立可信已收盘时钟后才允许启动时第一次调仓。"""
        try:
            self.portfolio.sync_from_mt5()
        except Exception as exc:
            logger.error(f"[Runner] 初始 portfolio sync 失败: {exc}")
            return False
        self._last_bar_time = None
        self._has_new_closed_bar()
        if self._last_bar_time is None:
            logger.error("[Runner] 无法建立已收盘 K 线时钟，跳过初始调仓")
            return False
        init_targets = self._compute_targets()
        if init_targets is None:
            return False
        try:
            self._reconcile_positions(init_targets)
        except Exception as exc:
            logger.error(f"[Runner] 初始调仓失败: {exc}")
            return False
        logger.info("[Runner] 初始调仓完成。")
        return True

    def _compute_targets(self) -> torch.Tensor | None:
        """为每个品种计算合并后的目标仓位 [-1, +1]，形状 [N]。

        多公式合并逻辑（信号平均）：
          - 每个有效公式独立执行 StackVM，得到最新 bar 的因子值
          - 对所有公式的 tanh(factor) 取算术平均，作为最终仓位信号
          - 若两条公式方向相反，信号相互抵消 → 趋近于 0 → 不开仓
          - 若两条方向一致，信号叠加强化 → 更大仓位比例
        这是标准 alpha 合成方法，安全且符合回测逻辑。
        """
        if self._data_manager is None:
            return None
        try:
            from model_core.features import MT5FeatureEngineer
            raw_dict = self._data_manager.raw_dict
            symbols  = self._data_manager.symbols
            N        = len(symbols)
            feat_all = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]

            targets   = torch.zeros(N, dtype=torch.float32)
            prev_dirs = torch.zeros(N, dtype=torch.float32)
            for i, sym in enumerate(symbols):
                prev_dirs[i] = float(self.portfolio.get_direction(sym))

            for i, sym in enumerate(symbols):
                formulas = self.symbol_formulas_multi.get(sym)
                if not formulas:
                    logger.warning(f"[Runner] {sym}: 无策略公式，保持空仓")
                    continue

                feat_i = feat_all[i:i+1]   # [1, F, T]
                targets[i] = compute_latest_formula_target(
                    self.vm,
                    formulas,
                    feat_i,
                    symbol=sym,
                    min_exposure=self.min_trade_exposure,
                )

            logger.info(
                "[Runner] 最终目标仓位: " +
                " | ".join(
                    f"{sym}={targets[i].item():+.3f}"
                    for i, sym in enumerate(symbols)
                )
            )
            return targets.float()

        except Exception as exc:
            logger.error(f"[Runner] _compute_targets failed: {exc}")
            return None

    def _reconcile_positions(self, targets: torch.Tensor) -> None:
        """对每个品种对账并执行调仓（替代旧版 _scan_for_entries）。

        对账逻辑（严格对标回测）：
            current = portfolio.get_direction(symbol)  # +1 / -1 / 0
            target  = sign(targets[i]) with min exposure band
            action  = reconcile_action(current, target)

        根据 action 执行对应 MT5 订单。
        """
        if self._data_manager is None:
            return

        symbols = self._data_manager.symbols
        expected_symbols = list(
            getattr(self, "execution_symbols", tuple(symbols))
        )
        if symbols != expected_symbols:
            raise RuntimeError("数据品种集合与冻结交易品种集合不一致")
        if (
            not isinstance(targets, torch.Tensor)
            or targets.ndim != 1
            or len(targets) != len(symbols)
            or not torch.isfinite(targets).all()
        ):
            raise RuntimeError("目标仓位必须完整覆盖全部交易品种且为有限一维张量")
        n = len(symbols)
        live_by_symbol = self._preflight_live_position_contracts(symbols)

        for idx in range(n):
            symbol       = symbols[idx]
            target_value = float(targets[idx].item())
            target       = target_to_direction(target_value)
            exposure     = abs(target_value) if target != 0 else 0.0

            # 以 MT5 实盘为准；对冲账户下同品种可能多空并存，需先清理
            live_positions = live_by_symbol[symbol]
            has_buy = any(getattr(p, "type", 0) == 0 for p in live_positions)
            has_sell = any(getattr(p, "type", 0) == 1 for p in live_positions)
            if has_buy and has_sell:
                logger.warning(
                    f"[Reconcile] {symbol}: 检测到同品种多空并存，先全部平仓"
                )
                if self._close_symbol_positions(symbol):
                    current = 0
                else:
                    logger.error(f"[Reconcile] {symbol}: 清理多空并存失败，跳过本轮")
                    continue
            else:
                current = self._mt5_net_direction(symbol, live_positions)

            action = reconcile_action(current, target)

            if action == HOLD:
                logger.debug(
                    f"[Reconcile] {symbol}: HOLD "
                    f"(dir={current}, target={target_value:+.2f})"
                )
                continue

            # MAX_OPEN_POSITIONS 约束（None 表示不限）
            max_pos = Config.MAX_OPEN_POSITIONS
            if max_pos is not None and action in (OPEN_LONG, OPEN_SHORT):
                if self.portfolio.get_open_count() >= max_pos:
                    logger.info(
                        f"[Reconcile] {symbol}: skip {action} — "
                        f"max_positions={max_pos} reached"
                    )
                    continue

            logger.info(
                f"[Reconcile] {symbol}: {action}  current={current}→target={target} "
                f"raw={target_value:+.2f}"
            )

            # ── 执行动作 ────────────────────────────────────────────
            if action == OPEN_LONG:
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                if self.trader.buy(symbol, lot):
                    self._record_position_after_open(
                        symbol, "BUY", lot, exposure
                    )

            elif action == OPEN_SHORT:
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                if self.trader.open_short(symbol, lot):
                    self._record_position_after_open(
                        symbol, "SELL", lot, exposure
                    )

            elif action == CLOSE:
                if self._close_symbol_positions(symbol):
                    self.portfolio.close_position(symbol)

            elif action == REVERSE_TO_LONG:
                if not self._close_symbol_positions(symbol):
                    logger.error(f"[Reconcile] {symbol}: 反手平空失败，跳过开多")
                    continue
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                if self.trader.buy(symbol, lot):
                    self._record_position_after_open(
                        symbol, "BUY", lot, exposure
                    )

            elif action == REVERSE_TO_SHORT:
                if not self._close_symbol_positions(symbol):
                    logger.error(f"[Reconcile] {symbol}: 反手平多失败，跳过开空")
                    continue
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                if self.trader.open_short(symbol, lot):
                    self._record_position_after_open(
                        symbol, "SELL", lot, exposure
                    )

    def _preflight_live_position_contracts(
        self,
        symbols: list[str],
    ) -> dict[str, list]:
        """在任何订单前冻结实盘持仓，并拒绝未知入场 exposure。"""
        live_by_symbol: dict[str, list] = {}
        dry_run = bool(getattr(self, "dry_run", False))
        for symbol in symbols:
            fetched = self.trader.get_positions(
                symbol,
                Config.MAGIC_NUMBER,
            )
            if fetched is None:
                raise RuntimeError(
                    f"{symbol}: 无法确认实盘持仓，禁止自动调仓"
                )
            live = list(fetched)
            live_by_symbol[symbol] = live
            if dry_run or not live:
                continue
            recorded = self.portfolio.positions.get(symbol)
            exposure = (
                None if recorded is None else recorded.target_exposure
            )
            if (
                exposure is None
                or isinstance(exposure, bool)
                or not isinstance(exposure, Real)
                or not math.isfinite(float(exposure))
                or not 0.0 < float(exposure) <= 1.0
            ):
                raise RuntimeError(
                    f"{symbol}: 实盘持仓缺少可信 target_exposure，"
                    "禁止自动调仓，需人工确认"
                )
        return live_by_symbol

    def _monitor_positions(self) -> None:
        """可选风控层（EXIT_MODE='risk' 或 'hybrid'）。

        多头：profit = current/entry - 1
        空头：profit = entry/current - 1（方向相反）
        止损、部分止盈、追踪止损逻辑同旧版，但空头追踪最低价。

        hybrid 模式下仅做紧急熔断（止损），不做部分止盈/追踪止损。
        """
        for symbol, pos in list(self.portfolio.positions.items()):
            tick = MT5PriceFeed.get_tick(symbol)
            if tick is None:
                logger.warning(f"[Monitor] Cannot fetch price for {symbol}.")
                continue

            current_price: float = tick["mid"]
            self.portfolio.update_price(symbol, current_price)

            if pos.entry_price <= 0:
                continue

            if pos.direction == "BUY":
                profit = current_price / pos.entry_price - 1.0
            else:  # SELL（空头）
                profit = pos.entry_price / current_price - 1.0

            # ── 止损（所有模式）────────────────────────────────────
            if profit < Config.STOP_LOSS_PCT:
                logger.warning(
                    f"[Monitor] STOP LOSS: {symbol} {pos.direction} "
                    f"profit={profit:.2%}"
                )
                ok = self.trader.close_all_positions(symbol, Config.MAGIC_NUMBER)
                if ok:
                    self.portfolio.close_position(symbol)
                continue

            # hybrid 模式只做止损，跳过下面的止盈/追踪
            if Config.EXIT_MODE == "hybrid":
                continue

            # ── 部分止盈（risk 模式）────────────────────────────────
            if profit > Config.TAKE_PROFIT_PCT and not pos.is_partial_closed:
                half = round(pos.lot_size / 2, 2)
                if half > 0:
                    logger.info(f"[Monitor] Partial TP: {symbol} profit={profit:.2%}")
                    ok = self.trader.close_position(
                        symbol, half, pos.direction, pos.ticket
                    )
                    if ok:
                        pos.is_partial_closed = True
                        self.portfolio.save_state()
                continue

            # ── 追踪止损（risk 模式，多头用最高价，空头用最低价）──
            if profit > Config.TRAILING_ACTIVATION:
                if pos.direction == "BUY" and pos.highest_price > 0:
                    drawdown = (pos.highest_price - current_price) / pos.highest_price
                    if drawdown > Config.TRAILING_DROP:
                        logger.warning(
                            f"[Monitor] TRAILING STOP (long): {symbol} "
                            f"dd={drawdown:.2%}"
                        )
                        ok = self.trader.close_position(
                            symbol, pos.lot_size, pos.direction, pos.ticket
                        )
                        if ok:
                            self.portfolio.close_position(symbol)
                elif pos.direction == "SELL" and pos.lowest_price > 0:
                    # 空头：从最低价反弹超过 TRAILING_DROP 则止损
                    rebound = (current_price - pos.lowest_price) / pos.lowest_price
                    if rebound > Config.TRAILING_DROP:
                        logger.warning(
                            f"[Monitor] TRAILING STOP (short): {symbol} "
                            f"rebound={rebound:.2%}"
                        )
                        ok = self.trader.close_position(
                            symbol, pos.lot_size, pos.direction, pos.ticket
                        )
                        if ok:
                            self.portfolio.close_position(symbol)

    # ──────────────────────────────────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────────────────────────────────

    def _mt5_net_direction(self, symbol: str, live_positions: list | None = None) -> int:
        """根据 MT5 实盘持仓计算品种净方向。"""
        positions = (
            live_positions
            if live_positions is not None
            else self.trader.get_positions(symbol, Config.MAGIC_NUMBER)
        )
        net = 0.0
        for p in positions:
            vol = float(getattr(p, "volume", 0.0))
            if getattr(p, "type", 0) == 0:
                net += vol
            else:
                net -= vol
        if net > 0:
            return 1
        if net < 0:
            return -1
        return 0

    def _close_symbol_positions(self, symbol: str) -> bool:
        """平掉该品种下本策略全部持仓，并同步本地状态。"""
        ok = self.trader.close_all_positions(symbol, Config.MAGIC_NUMBER)
        if ok and symbol in self.portfolio.positions:
            self.portfolio.close_position(symbol)
        return ok

    def _record_position_after_open(
        self,
        symbol: str,
        direction: str,
        lot: float,
        target_exposure: float,
    ) -> None:
        """开仓后从 MT5 回读 position ticket，避免 ticket=0 导致反手误开新单。"""
        positions = self.trader.get_positions(symbol, Config.MAGIC_NUMBER)
        want_type = 0 if direction == "BUY" else 1
        matched = [p for p in positions if getattr(p, "type", -1) == want_type]
        if not matched and positions:
            matched = [positions[-1]]

        if matched:
            p = matched[-1]
            price = float(getattr(p, "price_open", 0.0))
            if price <= 0:
                price = self._get_price(symbol) or 0.0
            self.portfolio.add_position(
                symbol,
                int(getattr(p, "ticket", 0)),
                price,
                float(getattr(p, "volume", lot)),
                direction,
                target_exposure=target_exposure,
            )
            return

        price = self._get_price(symbol) or 0.0
        logger.warning(f"[Runner] {symbol}: 开仓后未读到 MT5 持仓，本地 ticket 暂记为 0")
        self.portfolio.add_position(
            symbol,
            0,
            price,
            lot,
            direction,
            target_exposure=target_exposure,
        )

    def _calc_lot(self, symbol: str, exposure: float = 1.0) -> float:
        """按 XAUUSD 0.01 手的 ATR 美元波动预算计算手数。"""
        exposure = max(0.0, min(1.0, float(exposure)))
        if exposure <= 0:
            return 0.0

        fixed_map = getattr(Config, "FIXED_LOT_BY_SYMBOL", {}) or {}
        sym_key = symbol.upper().split(".")[0] if "." not in symbol else symbol
        if symbol in fixed_map:
            return float(fixed_map[symbol])
        if sym_key in fixed_map:
            return float(fixed_map[sym_key])

        # 从当前数据中取该品种最近 14 根 K 线的 ATR
        atr_price = self._get_atr(symbol)
        if not isinstance(atr_price, Real) or atr_price <= 0:
            logger.warning(f"[_calc_lot] {symbol}: ATR 获取失败，跳过开仓")
            return 0.0

        ref_symbol = getattr(Config, "VOL_TARGET_REFERENCE_SYMBOL", "XAUUSD")
        ref_lot = float(getattr(Config, "VOL_TARGET_REFERENCE_LOT", 0.01))
        ref_atr = self._get_atr(ref_symbol)
        if not isinstance(ref_atr, Real) or ref_atr <= 0:
            logger.warning(f"[_calc_lot] {symbol}: reference ATR 获取失败 ({ref_symbol})")
            return 0.0

        ref_value_per_unit = self.risk.value_per_price_unit(ref_symbol)
        if ref_value_per_unit <= 0:
            logger.warning(f"[_calc_lot] {symbol}: reference tick value 获取失败 ({ref_symbol})")
            return 0.0

        target_usd = ref_lot * ref_atr * ref_value_per_unit

        max_lot = getattr(Config, "MAX_LOT_PER_TRADE", 0.1)
        lot = self.risk.calculate_lot_for_volatility_target(
            symbol=symbol,
            atr_price=atr_price,
            target_usd=target_usd,
            exposure=exposure,
            max_lot=max_lot,
            sharpe_weight=self._vol_target_weight(symbol),
        )
        return lot

    def _vol_target_weight(self, symbol: str) -> float:
        """Optional Sharpe-based multiplier around the XAUUSD volatility budget."""
        sharpe_map = getattr(Config, "VOL_TARGET_SHARPE_BY_SYMBOL", {}) or {}
        ref = float(getattr(Config, "VOL_TARGET_SHARPE_REFERENCE", 0.0) or 0.0)
        sym_sharpe = sharpe_map.get(symbol)
        if sym_sharpe is None:
            return 1.0
        try:
            sym_sharpe = float(sym_sharpe)
            exponent = float(getattr(Config, "VOL_TARGET_SHARPE_EXPONENT", 0.5))
            min_w = float(getattr(Config, "VOL_TARGET_MIN_SHARPE_WEIGHT", 0.5))
            max_w = float(getattr(Config, "VOL_TARGET_MAX_SHARPE_WEIGHT", 1.5))
        except Exception:
            return 1.0
        if ref <= 0 or sym_sharpe <= 0:
            return min_w
        weight = (sym_sharpe / ref) ** exponent
        return max(min_w, min(max_w, weight))

    def _get_atr(self, symbol: str, period: int = 14) -> float | None:
        """从已加载数据中读取该品种最近 period 根 K 线的 ATR。"""
        if self._data_manager is None:
            return None
        try:
            raw   = self._data_manager.raw_dict
            syms  = self._data_manager.symbols
            if symbol not in syms:
                return None
            idx   = syms.index(symbol)
            hi    = raw["high"][idx, -period:].float()
            lo    = raw["low"][idx,  -period:].float()
            cl    = raw["close"][idx, -period:].float()
            # 简化 ATR：high-low 均值（因果，不看前一根收盘）
            atr   = (hi - lo).mean().item()
            return atr
        except Exception as exc:
            logger.warning(f"[_get_atr] {symbol}: {exc}")
            return None

    def _get_price(self, symbol: str) -> float:
        """获取当前中间价，失败返回 0.0。"""
        tick = MT5PriceFeed.get_tick(symbol)
        return tick["mid"] if tick else 0.0
