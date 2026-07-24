"""比较三个固定引擎输出并生成机器可读结论。"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from common import load_fixture, write_result


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} 不一致: actual={actual!r}, expected={expected!r}")


def _require_decimal(name: str, actual: object, expected: Decimal) -> None:
    number = Decimal(str(actual))
    if number != expected:
        raise RuntimeError(f"{name} 不一致: actual={number}, expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--alphamaster", required=True, type=Path)
    parser.add_argument("--akquant", required=True, type=Path)
    parser.add_argument("--rqalpha", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture, fixture_sha256 = load_fixture(args.fixture)
    results = {
        "alphamaster": _read(args.alphamaster),
        "akquant": _read(args.akquant),
        "rqalpha": _read(args.rqalpha),
    }
    for engine, result in results.items():
        _require_equal(f"{engine}.engine", result["engine"], engine)
        _require_equal(
            f"{engine}.fixture_sha256",
            result["fixture_sha256"],
            fixture_sha256,
        )
        _require_equal(
            f"{engine}.case_id",
            result["case_id"],
            fixture["case_id"],
        )
    _require_equal(
        "akquant.engine_version",
        results["akquant"]["engine_version"],
        "0.3.20",
    )
    _require_equal(
        "akquant.source_test_commit",
        results["akquant"]["source_test_commit"],
        "30054523fb905adb1c3f250749e1b5ff61cf8452",
    )
    _require_equal(
        "rqalpha.engine_version",
        results["rqalpha"]["engine_version"],
        "6.3.0",
    )
    _require_equal(
        "rqalpha.source_commit",
        results["rqalpha"]["source_commit"],
        "3503ab57932540cd36bf8375134e52c6923bf0d2",
    )
    _require_equal(
        "rqalpha.installed_source_sha256",
        results["rqalpha"]["installed_source_sha256"],
        results["rqalpha"]["snapshot_source_sha256"],
    )

    price = Decimal(str(fixture["price"]))
    shares = Decimal(str(fixture["shares"]))
    initial_cash = Decimal(str(fixture["initial_cash"]))
    fees = fixture["fee_schedule"]
    gross = price * shares
    commission = max(
        gross * Decimal(str(fees["commission_rate"])),
        Decimal(str(fees["minimum_commission"])),
    )
    transfer = gross * Decimal(str(fees["transfer_fee_rate"]))
    stamp = gross * Decimal(str(fees["stamp_duty_rate"]))
    buy_full = commission + transfer
    sell_full = commission + transfer + stamp
    expected_full_cash = initial_cash - buy_full - sell_full
    expected_without_transfer = initial_cash - commission - commission - stamp

    for engine in ("alphamaster", "akquant"):
        result = results[engine]
        _require_equal(
            f"{engine}.transfer_fee_included",
            result["transfer_fee_included"],
            True,
        )
        _require_equal(f"{engine}.buy.status", result["buy"]["status"], "FILLED")
        _require_equal(f"{engine}.sell.status", result["sell"]["status"], "FILLED")
        _require_decimal(f"{engine}.buy.fees", result["buy"]["fees"], buy_full)
        _require_decimal(f"{engine}.sell.fees", result["sell"]["fees"], sell_full)
        _require_decimal(
            f"{engine}.ending_cash",
            result["ending_cash"],
            expected_full_cash,
        )
        _require_equal(f"{engine}.ending_shares", result["ending_shares"], 0)
        _require_equal(
            f"{engine}.buy.filled_shares",
            result["buy"]["filled_shares"],
            int(shares),
        )
        _require_equal(
            f"{engine}.sell.filled_shares",
            result["sell"]["filled_shares"],
            int(shares),
        )
        _require_decimal(
            f"{engine}.buy.gross_amount",
            result["buy"]["gross_amount"],
            gross,
        )
        _require_decimal(
            f"{engine}.sell.gross_amount",
            result["sell"]["gross_amount"],
            gross,
        )

    rqalpha = results["rqalpha"]
    _require_equal(
        "rqalpha.transfer_fee_included",
        rqalpha["transfer_fee_included"],
        False,
    )
    for side in ("buy", "sell"):
        _require_equal(
            f"rqalpha.{side}.status",
            rqalpha[side]["status"],
            "CALCULATED",
        )
        _require_equal(
            f"rqalpha.{side}.filled_shares",
            rqalpha[side]["filled_shares"],
            int(shares),
        )
        _require_decimal(
            f"rqalpha.{side}.gross_amount",
            rqalpha[side]["gross_amount"],
            gross,
        )
    _require_decimal("rqalpha.buy.fees", rqalpha["buy"]["fees"], commission)
    _require_decimal(
        "rqalpha.sell.fees",
        rqalpha["sell"]["fees"],
        commission + stamp,
    )
    _require_decimal(
        "rqalpha.ending_cash",
        rqalpha["ending_cash"],
        expected_without_transfer,
    )

    write_result(
        args.output,
        {
            "contract_version": fixture["contract_version"],
            "case_id": fixture["case_id"],
            "fixture_sha256": fixture_sha256,
            "status": "PASS_WITH_DOCUMENTED_DIFFERENCE",
            "checks": {
                "alphamaster_matches_hand_calculation": True,
                "akquant_matches_alphamaster": True,
                "rqalpha_default_fee_gap_reproduced": True,
            },
            "expected": {
                "buy_fee_with_transfer": str(buy_full),
                "sell_fee_with_transfer": str(sell_full),
                "ending_cash_with_transfer": str(expected_full_cash),
                "rqalpha_ending_cash_without_transfer": str(
                    expected_without_transfer
                ),
            },
            "documented_differences": [
                {
                    "engine": "rqalpha",
                    "scope": "default_stock_transaction_cost_decider",
                    "difference": "默认股票费用器没有计算过户费",
                    "cash_difference": str(
                        expected_without_transfer - expected_full_cash
                    ),
                }
            ],
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
