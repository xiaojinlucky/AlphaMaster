"""
monitor_live_risk.py - read-only live position risk checker.

It compares current MT5 positions with the volatility-parity lot model:
the 1-ATR dollar move of XAUUSD 0.01 lot is the base risk budget.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore

from config import Config
import live_trade as live_trade_config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.runner import (
    compute_latest_formula_target,
    load_symbol_formulas,
    verify_formula_set_hashes,
)
from strategy_manager.risk import MT5RiskEngine
from strategy_manager.portfolio import MT5PortfolioManager, Position


def _symbols(symbol_override: list[str] | None = None) -> list[str]:
    return live_trade_config.resolve_live_symbols(symbol_override)


def _atr(raw: dict, symbols: list[str], symbol: str, period: int = 14) -> float:
    idx = symbols.index(symbol)
    hi = raw["high"][idx, -period:].float()
    lo = raw["low"][idx, -period:].float()
    return float((hi - lo).mean().item())


def _weight(symbol: str) -> float:
    sharpe_map = getattr(Config, "VOL_TARGET_SHARPE_BY_SYMBOL", {}) or {}
    ref = float(getattr(Config, "VOL_TARGET_SHARPE_REFERENCE", 0.0) or 0.0)
    sym_sharpe = sharpe_map.get(symbol)
    if sym_sharpe is None:
        return 1.0
    sym_sharpe = float(sym_sharpe)
    min_w = float(getattr(Config, "VOL_TARGET_MIN_SHARPE_WEIGHT", 0.5))
    max_w = float(getattr(Config, "VOL_TARGET_MAX_SHARPE_WEIGHT", 1.5))
    if ref <= 0 or sym_sharpe <= 0:
        return min_w
    exponent = float(getattr(Config, "VOL_TARGET_SHARPE_EXPONENT", 0.5))
    return max(min_w, min(max_w, (sym_sharpe / ref) ** exponent))


def _position_map() -> dict[str, list]:
    if mt5 is None:
        return {}
    positions = mt5.positions_get() or []
    by_symbol: dict[str, list] = {}
    for p in positions:
        by_symbol.setdefault(p.symbol, []).append(p)
    return by_symbol


def _monitor_exposure(
    signal: float,
    live_side: str,
    recorded: Position | None,
) -> float | None:
    """持仓时沿用 Runner 入场 exposure；空仓时展示当前拟入场 exposure。"""
    if live_side == "FLAT":
        return abs(signal)
    if (
        recorded is None
        or recorded.target_exposure is None
        or not 0.0 < float(recorded.target_exposure) <= 1.0
    ):
        return None
    return float(recorded.target_exposure)


def _current_targets(
    mgr: MT5DataManager,
    expected_formula_set_sha256: dict[str, str] | None = None,
    min_trade_exposure: float | None = None,
) -> dict[str, float]:
    vm = StackVM()
    feat_all = MT5FeatureEngineer.compute_features(mgr.raw_dict)
    formulas_by_symbol = load_symbol_formulas(mgr.symbols)
    if expected_formula_set_sha256 is not None:
        verify_formula_set_hashes(
            formulas_by_symbol,
            expected_formula_set_sha256,
        )
    targets: dict[str, float] = {}
    min_exp = (
        float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05))
        if min_trade_exposure is None
        else float(min_trade_exposure)
    )
    for idx, symbol in enumerate(mgr.symbols):
        targets[symbol] = compute_latest_formula_target(
            vm,
            formulas_by_symbol[symbol],
            feat_all[idx:idx + 1],
            symbol=symbol,
            min_exposure=min_exp,
        )
    return targets


def check_once(
    offline: bool = False,
    symbols_override: list[str] | None = None,
) -> int:
    if mt5 is None:
        print("ALERT MetaTrader5 package is unavailable", flush=True)
        return 2

    if not mt5.initialize():
        print(f"ALERT mt5.initialize failed: {mt5.last_error()}", flush=True)
        return 2

    plan = live_trade_config.resolve_live_strategy_plan(symbols_override)
    Config.SYMBOLS = list(plan.symbols)
    risk = MT5RiskEngine()
    portfolio = MT5PortfolioManager()

    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            if not offline:
                fetcher.connect()
            mgr = MT5DataManager(fetcher, require_all_symbols=True)
            mgr.load(list(plan.symbols))

        # MT5DataFetcher closes the terminal connection on context exit.
        # Reconnect before reading symbol specs and live positions.
        if not mt5.initialize():
            print(f"ALERT mt5 reconnect failed: {mt5.last_error()}", flush=True)
            return 2

        symbols = mgr.symbols
        raw = mgr.raw_dict
        targets = _current_targets(
            mgr,
            plan.expected_formula_set_sha256,
            plan.min_trade_exposure,
        )
        ref_symbol = getattr(Config, "VOL_TARGET_REFERENCE_SYMBOL", "XAUUSD")
        ref_lot = float(getattr(Config, "VOL_TARGET_REFERENCE_LOT", 0.01))
        ref_atr = _atr(raw, symbols, ref_symbol)
        ref_value = risk.value_per_price_unit(ref_symbol)
        target_usd = ref_lot * ref_atr * ref_value

        positions = _position_map()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] target_1atr_usd={target_usd:.2f} ref_atr={ref_atr:.4f}", flush=True)
        print("symbol       side    lot  signal expected exp_usd pos_usd ratio", flush=True)

        alerts = 0
        for symbol in symbols:
            atr = _atr(raw, symbols, symbol)
            value = risk.value_per_price_unit(symbol)
            weight = _weight(symbol)
            signal = targets.get(symbol, 0.0)
            live = positions.get(symbol, [])
            live_lot = sum(float(p.volume) for p in live)
            side = "FLAT"
            if live:
                net = sum(
                    float(p.volume) if p.type == 0 else -float(p.volume)
                    for p in live
                )
                side = "BUY" if net > 0 else ("SELL" if net < 0 else "MIX")
            recorded = portfolio.positions.get(symbol)
            exposure = _monitor_exposure(signal, side, recorded)
            fixed = (getattr(Config, "FIXED_LOT_BY_SYMBOL", {}) or {}).get(symbol)
            if exposure is None:
                expected = 0.0
            elif fixed is not None and exposure > 0:
                expected = float(fixed)
            elif exposure <= 0:
                expected = 0.0
            else:
                expected = risk.calculate_lot_for_volatility_target(
                    symbol=symbol,
                    atr_price=atr,
                    target_usd=target_usd,
                    exposure=exposure,
                    max_lot=float(getattr(Config, "MAX_LOT_PER_TRADE", 5.0)),
                    sharpe_weight=weight,
                )

            pos_usd = live_lot * atr * value
            exp_usd = expected * atr * value
            ratio = pos_usd / max(exp_usd, 1e-9) if exp_usd > 0 else 0.0

            print(
                f"{symbol:<12} {side:<5} {live_lot:>5.2f} {signal:>7.2f} {expected:>8.2f} "
                f"{exp_usd:>7.2f} {pos_usd:>7.2f} {ratio:>5.2f}",
                flush=True,
            )

            if expected > 0 and ratio > 1.35:
                alerts += 1
                print(f"ALERT {symbol} live risk exceeds expected: {ratio:.2f}", flush=True)
            if expected == 0 and live_lot > 0:
                alerts += 1
                if exposure is None:
                    print(
                        f"ALERT {symbol} live position lacks recorded entry exposure",
                        flush=True,
                    )
                else:
                    print(
                        f"ALERT {symbol} has position while target is flat",
                        flush=True,
                    )
            if symbol == "XAGUSD" and live:
                alerts += 1
                print("ALERT XAGUSD has live position but should be disabled", flush=True)

        return 1 if alerts else 0
    finally:
        mt5.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--symbols", nargs="+")
    args = parser.parse_args()

    if not args.watch:
        raise SystemExit(
            check_once(
                offline=args.offline,
                symbols_override=args.symbols,
            )
        )

    while True:
        try:
            check_once(
                offline=args.offline,
                symbols_override=args.symbols,
            )
        except Exception as exc:
            print(f"ALERT monitor exception: {exc}", flush=True)
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
