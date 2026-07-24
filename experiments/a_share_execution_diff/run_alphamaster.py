"""用 AlphaMaster 执行固定的一手买卖差分样本。"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_fixture, verify_repo_paths_clean, write_result

from portfolio_manager import controller
from portfolio_manager.controller import ModelSignalSnapshot, PortfolioPolicy
from portfolio_manager.execution import (
    AShareFeeSchedule,
    ExecutionQuote,
    VirtualAccount,
    account_snapshot_sha256,
    execute_portfolio_decision,
)
from portfolio_manager.universe import UniverseContract


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _bar_ts(session: str) -> int:
    return int(
        datetime.fromisoformat(f"{session}T15:00:00")
        .replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        .timestamp()
    )


def _signal(
    symbol: str,
    session: str,
    *,
    model_exit: bool,
) -> ModelSignalSnapshot:
    return ModelSignalSnapshot(
        run_id=f"run_{session.replace('-', '')}T070000Z_{_sha(session)[:8]}",
        symbol=symbol,
        bar_ts=_bar_ts(session),
        session_date=session,
        timeframe="1d",
        market_source="synthetic_e3_e4",
        raw_score=5.0,
        requested_exposure=1.0,
        confidence=1.0,
        model_version=_sha("model"),
        data_version=_sha("data"),
        calibration_version="e3-e4-v1",
        calibration_history_sha256=_sha("history"),
        history_scores=(0.0, 1.0, 2.0, 3.0, 4.0),
        model_exit=model_exit,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture, fixture_sha256 = load_fixture(args.fixture)
    verify_repo_paths_clean(
        REPO_ROOT,
        [
            "portfolio_manager/controller.py",
            "portfolio_manager/execution.py",
            "portfolio_manager/universe.py",
        ],
    )
    symbol = str(fixture["symbol"])
    price = Decimal(str(fixture["price"]))
    shares = int(fixture["shares"])
    initial_cash = Decimal(str(fixture["initial_cash"]))
    fees = fixture["fee_schedule"]
    gross_exposure = Decimal(shares) * price / initial_cash

    universe = UniverseContract(
        universe_id=f"third-party-diff:{fixture['case_id']}",
        snapshot_date=str(fixture["sell_session"]).replace("-", ""),
        constituent_count=1,
        universe_sha256=fixture_sha256,
        symbols=(symbol,),
    )
    policy = PortfolioPolicy(
        top_k=1,
        dropout_rank=1,
        minimum_history=5,
        minimum_model_exposure=0.0,
        minimum_confidence=0.0,
        minimum_calibrated_score=0.5,
        gross_exposure=float(gross_exposure),
    )
    schedule = AShareFeeSchedule(
        commission_rate=float(fees["commission_rate"]),
        minimum_commission=float(fees["minimum_commission"]),
        stamp_duty_rate=float(fees["stamp_duty_rate"]),
        transfer_fee_rate=float(fees["transfer_fee_rate"]),
        slippage_rate=float(fees["slippage_rate"]),
    )

    initial = VirtualAccount(cash=float(initial_cash))
    buy_decision = controller._build_portfolio_decision(
        [_signal(symbol, fixture["signal_session"], model_exit=False)],
        universe=universe,
        current_weights={},
        account_snapshot_sha256=account_snapshot_sha256(initial, ()),
        policy=policy,
    )
    buy_result = execute_portfolio_decision(
        buy_decision,
        execution_session=fixture["buy_session"],
        account=initial,
        decision_quotes=(),
        quotes=(
            ExecutionQuote(
                symbol,
                fixture["buy_session"],
                float(price),
                "OPEN",
                int(fixture["lot_size"]),
            ),
        ),
        fee_schedule=schedule,
    )

    bought = buy_result.account_after
    bought_shares = sum(lot.shares for lot in bought.lots)
    decision_quote = ExecutionQuote(
        symbol,
        fixture["buy_session"],
        float(price),
        "OPEN",
        int(fixture["lot_size"]),
    )
    nav = Decimal(str(bought.cash)) + Decimal(bought_shares) * price
    sell_decision = controller._build_portfolio_decision(
        [_signal(symbol, fixture["buy_session"], model_exit=True)],
        universe=universe,
        current_weights={
            symbol: float(Decimal(bought_shares) * price / nav),
        },
        account_snapshot_sha256=account_snapshot_sha256(
            bought,
            (decision_quote,),
        ),
        policy=policy,
    )
    sell_result = execute_portfolio_decision(
        sell_decision,
        execution_session=fixture["sell_session"],
        account=bought,
        decision_quotes=(decision_quote,),
        quotes=(
            ExecutionQuote(
                symbol,
                fixture["sell_session"],
                float(price),
                "OPEN",
                int(fixture["lot_size"]),
            ),
        ),
        fee_schedule=schedule,
    )
    buy_order = buy_result.orders[0]
    sell_order = sell_result.orders[0]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    write_result(
        args.output,
        {
            "engine": "alphamaster",
            "engine_version": head,
            "scope": "target_weight_to_terminal_fill",
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "buy": {
                "status": buy_order.status,
                "filled_shares": buy_order.filled_shares,
                "gross_amount": str(Decimal(str(buy_order.gross_amount))),
                "fees": str(Decimal(str(buy_order.fees))),
            },
            "sell": {
                "status": sell_order.status,
                "filled_shares": sell_order.filled_shares,
                "gross_amount": str(Decimal(str(sell_order.gross_amount))),
                "fees": str(Decimal(str(sell_order.fees))),
            },
            "ending_cash": str(Decimal(str(sell_result.account_after.cash))),
            "ending_shares": sum(
                lot.shares for lot in sell_result.account_after.lots
            ),
            "transfer_fee_included": True,
        },
    )


if __name__ == "__main__":
    main()
