"""
live_trade.py — 自动交易启动脚本

默认品种来自当前评分合同的 factor_scan.json，报告缺失或失配时拒绝启动。

已停用、永不自动交易：
  XAGUSD — 白银合约乘数过大，实盘盈亏波动远超指数，2026-07-08 起排除。
"""
import sys
import os
import json
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import Config
from model_core.target_contract import SCORING_CONTRACT_VERSION
from strategy_manager.runner import (
    MT5StrategyRunner,
    load_symbol_formulas,
    verify_formula_set_hashes,
    validate_strategy_symbol,
)
from scan_all_factors import factor_scan_execution_contract
from loguru import logger

def _apply_trade_filters(symbols: list[str]) -> list[str]:
    """严格校验、去重，并去掉明确禁用的品种。"""
    excluded = set(getattr(Config, "EXCLUDED_TRADE_SYMBOLS", []) or [])
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        try:
            symbol = validate_strategy_symbol(symbol)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if symbol in excluded or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _load_scan_contracts() -> tuple[dict[str, str], dict[str, dict]]:
    """读取当前扫描报告，返回过滤后的品种与公式集合身份。"""
    p = os.path.join(_dir, "backtest_output", "factor_scan.json")
    if not os.path.exists(p):
        raise RuntimeError("实盘品种扫描报告不存在，必须先重新扫描")
    try:
        data = json.load(open(p, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"实盘品种扫描报告无法读取: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("实盘品种扫描报告顶层必须是对象")
    if data.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
        raise RuntimeError("实盘品种扫描报告评分合同不兼容，必须重新生成")
    rows = data.get("valid")
    if not isinstance(rows, list):
        raise RuntimeError("实盘品种扫描报告缺少 valid 列表")
    if not rows:
        raise RuntimeError("实盘品种扫描报告没有合格品种")

    report_hashes: dict[str, str] = {}
    report_execution_contracts: dict[str, dict] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("valid") is not True:
            raise RuntimeError(f"实盘品种扫描报告 valid[{index}] 结构无效")
        symbol = row.get("symbol")
        digest = row.get("formula_set_sha256")
        execution_contract = row.get("execution_contract")
        if not isinstance(symbol, str) or not symbol:
            raise RuntimeError(f"实盘品种扫描报告 valid[{index}] 缺少 symbol")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"实盘品种扫描报告 {symbol} 缺少有效 formula_set_sha256"
            )
        if symbol in report_hashes:
            raise RuntimeError(f"实盘品种扫描报告重复品种: {symbol}")
        report_hashes[symbol] = digest
        if not isinstance(execution_contract, dict):
            raise RuntimeError(f"实盘品种扫描报告 {symbol} 缺少执行合同")
        report_execution_contracts[symbol] = execution_contract

    syms = _apply_trade_filters(list(report_hashes))
    if not syms:
        raise RuntimeError("实盘品种扫描报告经禁用过滤后没有合格品种")

    filtered_hashes = {symbol: report_hashes[symbol] for symbol in syms}
    filtered_contracts = {
        symbol: report_execution_contracts[symbol] for symbol in syms
    }
    for symbol, report_contract in filtered_contracts.items():
        current_contract = factor_scan_execution_contract(symbol)
        if report_contract != current_contract:
            raise RuntimeError(
                f"{symbol} 扫描执行合同与当前配置不一致，必须重新扫描"
            )
    return filtered_hashes, filtered_contracts


def _load_bound_formulas(
    expected_hashes: dict[str, str],
) -> dict[str, list[list[int]]]:
    formulas_by_symbol = load_symbol_formulas(
        list(expected_hashes),
        strategies_dir=Path(_dir) / "strategies",
    )
    try:
        verify_formula_set_hashes(formulas_by_symbol, expected_hashes)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return formulas_by_symbol


def _load_valid_from_scan() -> list[str]:
    expected_hashes, _contracts = _load_scan_contracts()
    _load_bound_formulas(expected_hashes)
    return list(expected_hashes)


@dataclass(frozen=True)
class LiveStrategyPlan:
    symbols: tuple[str, ...]
    expected_formula_set_sha256: dict[str, str] | None
    min_trade_exposure: float


def resolve_live_strategy_plan(
    symbol_override: list[str] | None = None,
) -> LiveStrategyPlan:
    """解析启动计划；扫描模式同时携带 Runner 最终重验所需身份。"""
    if symbol_override is not None:
        resolved = _apply_trade_filters(symbol_override)
        if not resolved:
            raise RuntimeError("显式交易品种经禁用过滤后为空")
        return LiveStrategyPlan(
            tuple(resolved),
            None,
            float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05)),
        )

    expected_hashes, contracts = _load_scan_contracts()
    _load_bound_formulas(expected_hashes)
    min_exposures = {
        float(contract["min_trade_exposure"])
        for contract in contracts.values()
    }
    if len(min_exposures) != 1:
        raise RuntimeError("扫描报告各品种的 min_trade_exposure 不一致")
    return LiveStrategyPlan(
        tuple(expected_hashes),
        expected_hashes,
        min_exposures.pop(),
    )


def resolve_live_symbols(symbol_override: list[str] | None = None) -> list[str]:
    """给交易 Runner 与风险监控返回同一套品种集合。"""
    return list(resolve_live_strategy_plan(symbol_override).symbols)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaMaster MT5 信号运行入口")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="读取真实数据与持仓，但在订单边界硬拦截全部写入",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="显式指定品种；不会自动插入其他品种",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    plan = resolve_live_strategy_plan(args.symbols)
    Config.SYMBOLS = list(plan.symbols)

    logger.info(f"[live_trade] 交易品种: {Config.SYMBOLS}")
    excluded = getattr(Config, "EXCLUDED_TRADE_SYMBOLS", [])
    if excluded:
        logger.info(f"[live_trade] 已禁用（永不自动交易）: {excluded}")

    if args.dry_run:
        logger.info("[live_trade] DRY RUN 模式：只打印信号，不下单")

    logger.info("=" * 60)
    logger.info("  AlphaGPT 自动交易 [XAUUSD + 指数有效因子]")
    logger.info(f"  品种:     {Config.SYMBOLS}")
    logger.info(f"  周期:     H1")
    logger.info(f"  XAUUSD:   best_XAUUSD.json (precious_metals_v1) 固定 0.01 手")
    logger.info(f"  XAGUSD:   已停用，不自动交易")
    logger.info(
        f"  仓位:     以 {Config.VOL_TARGET_REFERENCE_SYMBOL} "
        f"{Config.VOL_TARGET_REFERENCE_LOT} 手的一根 ATR 美元波动为基准"
    )
    logger.info(f"  信号模式: {Config.SIGNAL_MODE}")
    logger.info("=" * 60)

    runner = MT5StrategyRunner(
        dry_run=args.dry_run,
        expected_formula_set_sha256=plan.expected_formula_set_sha256,
        min_trade_exposure=plan.min_trade_exposure,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("[live_trade] 收到 Ctrl+C，正在停止...")
    finally:
        runner.shutdown()
        logger.info("[live_trade] 已停止。")


if __name__ == "__main__":
    main()
