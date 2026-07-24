"""复验 RQAlpha 默认股票费用器的一手买卖费用。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from common import load_fixture, verify_git_source, write_result
from rqalpha.const import INSTRUMENT_TYPE, POSITION_EFFECT, SIDE
from rqalpha.core.events import EventBus
from rqalpha.environment import Environment
from rqalpha.interface import TransactionCostArgs
from rqalpha.mod.rqalpha_mod_sys_transaction_cost.deciders import (
    StockTransactionCostDecider,
)

EXPECTED_VERSION = "6.3.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    version = importlib.metadata.version("rqalpha")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"RQAlpha 版本必须是 {EXPECTED_VERSION}，实际为 {version}"
        )
    fixture, fixture_sha256 = load_fixture(args.fixture)
    source_commit = verify_git_source(
        args.source_root,
        args.source_commit,
    )
    installed_decider_path = Path(
        inspect.getfile(StockTransactionCostDecider)
    ).resolve()
    source_decider_path = (
        args.source_root
        / "rqalpha"
        / "mod"
        / "rqalpha_mod_sys_transaction_cost"
        / "deciders.py"
    ).resolve()
    if not source_decider_path.is_file():
        raise RuntimeError(
            f"RQAlpha 费用器源码不存在: {source_decider_path}"
        )
    installed_source_sha256 = hashlib.sha256(
        installed_decider_path.read_bytes()
    ).hexdigest()
    snapshot_source_sha256 = hashlib.sha256(
        source_decider_path.read_bytes()
    ).hexdigest()
    if installed_source_sha256 != snapshot_source_sha256:
        raise RuntimeError(
            "RQAlpha 已安装费用器与固定源码快照不一致"
        )
    fees = fixture["fee_schedule"]
    price = Decimal(str(fixture["price"]))
    quantity = int(fixture["shares"])
    initial_cash = Decimal(str(fixture["initial_cash"]))

    # RQAlpha 股票费用器的基础佣金率是 0.0008；用 multiplier 对齐样本的
    # 0.0003。这里只隔离复验费用模块，不伪装成完整回测。
    commission_multiplier = (
        Decimal(str(fees["commission_rate"])) / Decimal("0.0008")
    )
    Environment._env = SimpleNamespace()
    decider = StockTransactionCostDecider(
        commission_multiplier=float(commission_multiplier),
        min_commission=float(fees["minimum_commission"]),
        tax_multiplier=1.0,
        pit_tax=False,
        event_bus=EventBus(),
    )
    instrument = SimpleNamespace(type=INSTRUMENT_TYPE.CS)
    buy = decider.calc(
        TransactionCostArgs(
            instrument=instrument,
            price=float(price),
            quantity=quantity,
            side=SIDE.BUY,
            position_effect=POSITION_EFFECT.OPEN,
            order_id=1,
        )
    )
    sell = decider.calc(
        TransactionCostArgs(
            instrument=instrument,
            price=float(price),
            quantity=quantity,
            side=SIDE.SELL,
            position_effect=POSITION_EFFECT.CLOSE,
            order_id=2,
        )
    )
    gross = price * quantity
    ending_cash = (
        initial_cash
        - gross
        - Decimal(str(buy.total))
        + gross
        - Decimal(str(sell.total))
    )
    write_result(
        args.output,
        {
            "engine": "rqalpha",
            "engine_version": version,
            "source_commit": source_commit,
            "installed_source_sha256": installed_source_sha256,
            "snapshot_source_sha256": snapshot_source_sha256,
            "scope": "default_stock_transaction_cost_decider",
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "buy": {
                "status": "CALCULATED",
                "filled_shares": quantity,
                "gross_amount": str(gross),
                "fees": str(Decimal(str(buy.total))),
            },
            "sell": {
                "status": "CALCULATED",
                "filled_shares": quantity,
                "gross_amount": str(gross),
                "fees": str(Decimal(str(sell.total))),
            },
            "ending_cash": str(ending_cash),
            "ending_shares": None,
            "transfer_fee_included": False,
        },
    )


if __name__ == "__main__":
    main()
