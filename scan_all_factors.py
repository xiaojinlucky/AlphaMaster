"""
scan_all_factors.py — 扫描 Runner 真实公式集合，逐品种 solo 回测（只看收益）

判定有效：年化收益 > 0（忽略 MDD / Sharpe / WF）
数据：D:\\K线数据 离线 H1
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import VOCAB_VERSION
from model_core.vm import StackVM
from model_core.target_contract import (
    SCORING_CONTRACT_VERSION,
    align_target_return_window,
)
from strategy_manager.runner import (
    RUNNER_POSITION_CONTRACT_VERSION,
    apply_runner_position_state,
    compute_formula_set_positions,
    formula_set_sha256,
    load_symbol_formulas,
    validate_strategy_symbol,
)

PERIODS_PER_YEAR = 6240
COST = {
    "forex": 0.00015,
    "metals": 0.00020,
    "index": 0.00030,
}


def _cost(sym: str) -> float:
    if sym.startswith("XAU") or sym.startswith("XAG"):
        return COST["metals"]
    if ".cash" in sym or sym.startswith("US") or sym.startswith("JP"):
        return COST["index"]
    return COST["forex"]


def _uses_fixed_lot(symbol: str) -> bool:
    fixed_map = getattr(Config, "FIXED_LOT_BY_SYMBOL", {}) or {}
    sym_key = symbol.upper() if "." not in symbol else symbol
    return symbol in fixed_map or sym_key in fixed_map


def factor_scan_execution_contract(symbol: str) -> dict:
    """生成会改变扫描收益或 Runner 持仓的全部执行假设。"""
    return {
        "position_contract_version": RUNNER_POSITION_CONTRACT_VERSION,
        "min_trade_exposure": float(
            getattr(Config, "MIN_TRADE_EXPOSURE", 0.05)
        ),
        "fixed_exposure": _uses_fixed_lot(symbol),
        "cost_rate": _cost(symbol),
        "periods_per_year": PERIODS_PER_YEAR,
    }


def solo_backtest(formulas: list[list[int]], symbol: str) -> dict:
    with MT5DataFetcher(offline=True) as fetcher:
        orig = Config.SYMBOLS[:]
        Config.SYMBOLS = [symbol]
        try:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            if symbol not in mgr.symbols:
                return {"error": "no data"}
            vm = StackVM()
            contract = factor_scan_execution_contract(symbol)
            target_position = compute_formula_set_positions(
                vm,
                formulas,
                mgr.feat_tensor,
                symbol=symbol,
                min_exposure=contract["min_trade_exposure"],
            )
            position = apply_runner_position_state(
                target_position,
                min_exposure=contract["min_trade_exposure"],
                fixed_exposure=contract["fixed_exposure"],
            )
            position, target_ret = align_target_return_window(
                position,
                mgr.target_ret,
            )
            prev = torch.zeros_like(position)
            prev[:, 1:] = position[:, :-1]
            turnover = (position - prev).abs()
            cr = contract["cost_rate"]
            pnl = (position * target_ret - turnover * cr).squeeze(0)
            T = int(pnl.shape[0])
            ann = float(pnl.mean().item() * PERIODS_PER_YEAR)
            total = float(pnl.sum().item())
            m = pnl.mean().item()
            s = pnl.std(unbiased=False).item()
            sharpe = float(m / (s + 1e-8) * math.sqrt(PERIODS_PER_YEAR))
            cum = torch.cumsum(pnl, dim=0)
            mdd = float((torch.cummax(cum, 0).values - cum).max().item())
            return {
                "T": T,
                "years": T / PERIODS_PER_YEAR,
                "ann_ret": ann,
                "total_ret": total,
                "sharpe": sharpe,
                "mdd": mdd,
                "valid": ann > 0,
            }
        finally:
            Config.SYMBOLS = orig


def discover_formula_sets(
    strategies_dir: Path,
) -> dict[str, list[list[int]]]:
    """按 Runner 优先级解析扫描候选的完整公式集合。"""
    group_names = {"index", "precious_metals", "forex", "metals_comm"}
    symbols = sorted(
        {
            validate_strategy_symbol(path.stem.removeprefix("best_"))
            for path in strategies_dir.glob("best_*.json")
            if path.stem.removeprefix("best_") not in group_names
        }
    )
    if not symbols:
        return {}
    return load_symbol_formulas(
        symbols,
        strategies_dir=strategies_dir,
    )


def main():
    strategies_dir = Path("strategies")
    formula_sets = discover_formula_sets(strategies_dir)
    rows = []
    print(f"\nFactor Scan (returns-only) | vocab={VOCAB_VERSION} | offline\n")
    print(f"{'品种':<16} {'年化%':>8} {'Sharpe':>8} {'MDD%':>8} {'年数':>6} {'有效':>6}  文件")
    print("-" * 80)

    for sym, formulas in formula_sets.items():
        bt = solo_backtest(formulas, sym)
        if "error" in bt:
            print(f"{sym:<16} ERROR: {bt['error']}")
            continue
        tag = "YES" if bt["valid"] else "NO"
        print(
            f"{sym:<16} {bt['ann_ret']*100:>8.2f} {bt['sharpe']:>8.3f} "
            f"{bt['mdd']*100:>8.2f} {bt['years']:>6.2f} {tag:>6}  "
            f"{len(formulas)} formulas"
        )
        rows.append(
            {
                "symbol": sym,
                "formula_count": len(formulas),
                "formula_set_sha256": formula_set_sha256(formulas),
                "execution_contract": factor_scan_execution_contract(sym),
                **bt,
            }
        )

    valid = [r for r in rows if r.get("valid")]
    print(f"\n有效因子（年化>0）: {len(valid)}/{len(rows)}")
    for r in sorted(valid, key=lambda x: -x["ann_ret"]):
        print(
            f"  {r['symbol']:<16} ann={r['ann_ret']*100:+.2f}%  "
            f"{r['formula_count']} formulas"
        )

    out = Path("backtest_output/factor_scan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "all": rows,
                "valid": valid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON → {out}")


if __name__ == "__main__":
    main()
