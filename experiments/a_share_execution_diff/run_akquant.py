"""用 AKQuant 固定版本执行同一手买卖费用样本。"""

from __future__ import annotations

import argparse
import importlib.metadata
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import akquant as aq
from akquant import AssetType, Bar, Engine, Instrument, Strategy
from common import load_fixture, verify_git_source, write_result

EXPECTED_VERSION = "0.3.20"


class OneLotRoundTrip(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.day = 0

    def on_bar(self, bar: aq.Bar) -> None:
        self.day += 1
        if self.day == 1:
            self.buy(bar.symbol, 100)
        elif self.day == 2:
            self.sell(bar.symbol, 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-test-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    version = importlib.metadata.version("akquant")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"AKQuant 版本必须是 {EXPECTED_VERSION}，实际为 {version}"
        )
    fixture, fixture_sha256 = load_fixture(args.fixture)
    source_test_commit = verify_git_source(
        args.source_root,
        args.source_test_commit,
    )
    symbol = str(fixture["symbol"])
    price = float(fixture["price"])
    quantity = int(fixture["shares"])
    fees = fixture["fee_schedule"]
    if quantity != 100:
        raise RuntimeError("当前固定策略只接受 100 股样本")
    if Decimal(str(fees["slippage_rate"])) != Decimal(0):
        raise RuntimeError("AKQuant 固定差分样本只接受零滑点")

    tz = timezone(timedelta(hours=8))
    sessions = [
        datetime.fromisoformat(f"{fixture['buy_session']}T15:00:00").replace(
            tzinfo=tz
        ),
        datetime.fromisoformat(f"{fixture['sell_session']}T15:00:00").replace(
            tzinfo=tz
        ),
    ]
    engine = Engine()
    engine.use_china_market()
    engine.set_cash(float(fixture["initial_cash"]))
    engine.set_stock_fee_rules(
        float(fees["commission_rate"]),
        float(fees["stamp_duty_rate"]),
        float(fees["transfer_fee_rate"]),
        float(fees["minimum_commission"]),
    )
    cast(Any, engine).set_fill_policy("close", 0, "same_cycle")
    engine.add_instrument(
        Instrument(
            symbol=symbol,
            asset_type=AssetType.Stock,
            multiplier=1.0,
            margin_ratio=1.0,
            tick_size=0.01,
            lot_size=float(fixture["lot_size"]),
        )
    )
    engine.add_bars(
        [
            Bar(
                int(session.timestamp() * 1e9),
                price,
                price,
                price,
                price,
                1_000_000.0,
                symbol,
            )
            for session in sessions
        ]
    )
    engine.run(OneLotRoundTrip(), show_progress=False)
    orders = engine.get_orders_dataframe().to_dict(orient="records")
    trades = list(engine.trades)
    result = engine.get_results()
    if len(orders) != 2 or len(trades) != 2:
        raise RuntimeError(
            f"AKQuant 预期 2 个订单和 2 笔成交，实际为 "
            f"{len(orders)} 个订单、{len(trades)} 笔成交"
        )
    expected_sides = ("BUY", "SELL")
    for index, expected_side in enumerate(expected_sides):
        order_side = str(orders[index]["side"]).split(".")[-1].upper()
        trade_side = str(trades[index].side).split(".")[-1].upper()
        if order_side != expected_side or trade_side != expected_side:
            raise RuntimeError(
                f"AKQuant 第 {index + 1} 笔方向错误: "
                f"order={order_side}, trade={trade_side}"
            )
        if (
            str(trades[index].order_id) != str(orders[index]["order_id"])
            or str(trades[index].symbol) != symbol
            or Decimal(str(trades[index].quantity)) != Decimal(quantity)
        ):
            raise RuntimeError(f"AKQuant 第 {index + 1} 笔成交身份不匹配")
    ending_cash = Decimal(str(list(result.cash_curve)[-1][1]))
    position_history = result.get_positions_dict()
    symbol_rows = [
        index
        for index, value in enumerate(position_history["symbol"])
        if str(value) == symbol
    ]
    if symbol_rows:
        last_index = max(
            symbol_rows,
            key=lambda index: position_history["date"][index],
        )
        ending_quantity = (
            Decimal(str(position_history["long_shares"][last_index]))
            - Decimal(str(position_history["short_shares"][last_index]))
        )
    else:
        ending_quantity = Decimal(0)
    if ending_quantity != ending_quantity.to_integral_value():
        raise RuntimeError(f"AKQuant 期末持仓不是整数股: {ending_quantity}")
    write_result(
        args.output,
        {
            "engine": "akquant",
            "engine_version": version,
            "source_test_commit": source_test_commit,
            "scope": "same_cycle_terminal_fill",
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "buy": {
                "status": str(orders[0]["status"]).split(".")[-1].upper(),
                "filled_shares": int(orders[0]["filled"]),
                "gross_amount": str(
                    Decimal(str(trades[0].price))
                    * Decimal(str(trades[0].quantity))
                ),
                "fees": str(Decimal(str(trades[0].commission))),
            },
            "sell": {
                "status": str(orders[1]["status"]).split(".")[-1].upper(),
                "filled_shares": int(orders[1]["filled"]),
                "gross_amount": str(
                    Decimal(str(trades[1].price))
                    * Decimal(str(trades[1].quantity))
                ),
                "fees": str(Decimal(str(trades[1].commission))),
            },
            "ending_cash": str(ending_cash),
            "ending_shares": int(ending_quantity),
            "transfer_fee_included": True,
        },
    )


if __name__ == "__main__":
    main()
