"""RND-04C 动态组合日线 replay 的生产 CLI 入口（工程证据，不是策略回测）。

固定使用生产身份：真实沪深300受信历史根 + 生产 RQAlpha overlay
（identity_sha256 固化常量运行前比对）+ v3 qfq 冻结价格 + 冻结交易日历。
不提供任何身份覆盖参数；失败关闭是本工具的正常合法结果之一
（例如时点成分含 v3 quarantine 成员时，按硬条件 A 拒绝该时点）。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.rqalpha_execution_overlay import (
    RQAlphaOverlayError,
    load_rqalpha_execution_overlay,
)
from portfolio_manager.execution import AShareFeeSchedule
from portfolio_manager.ledger import PortfolioDecisionLedger
from portfolio_manager.replay import (
    PRODUCTION_CSI300_HISTORY_ROOT,
    PRODUCTION_CSI300_TRUST_POLICY,
    REPLAY_RUN_MANIFEST_FORMAT,
    DynamicDailyReplay,
    DynamicReplayConfig,
    DynamicReplayError,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RND-04C 确定性动态组合日线 replay（engineering replay，"
            "机制验证信号，产物只作工程证据）"
        )
    )
    parser.add_argument("--start", required=True, help="决策区间首日 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="区间末日 YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--dropout-rank", type=int, required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="运行产物目录（账本 SQLite 与运行清单 JSON）",
    )
    # 费用假设显式声明（工程假设，写入运行身份；无隐藏默认语义）。
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "replay_run_manifest.json"
    config = DynamicReplayConfig(
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        dropout_rank=args.dropout_rank,
        initial_cash=args.initial_cash,
        run_label=args.run_label,
    )
    fee_schedule = AShareFeeSchedule(
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        stamp_duty_rate=args.stamp_duty_rate,
        transfer_fee_rate=args.transfer_fee_rate,
        slippage_rate=args.slippage_rate,
    )

    overlay = None
    try:
        overlay = load_rqalpha_execution_overlay()
        ledger = PortfolioDecisionLedger(output_dir / "replay_ledger.sqlite3")
        engine = DynamicDailyReplay(
            config,
            overlay=overlay,
            ledger=ledger,
            fee_schedule=fee_schedule,
            history_root=PRODUCTION_CSI300_HISTORY_ROOT,
            trust_policy=PRODUCTION_CSI300_TRUST_POLICY,
        )
        manifest = engine.run()
        verification = engine.verify()
        manifest["verification"] = verification
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = manifest["summary"]
        print(
            f"COMPLETED replay_run_id={manifest['replay_run_id']} "
            f"pairs={summary['pair_count']} orders={summary['order_count']} "
            f"rejected={len(summary['rejected_orders'])} "
            f"verified_states={verification['verified_state_count']}"
        )
        print(f"manifest: {manifest_path}")
        return 0
    except (DynamicReplayError, RQAlphaOverlayError) as exc:
        # 失败关闭是合法结果：如实落盘失败清单，不静默缩池、不填补。
        fail_manifest = {
            "format": REPLAY_RUN_MANIFEST_FORMAT,
            "status": "FAIL_CLOSED",
            "run_label": args.run_label,
            "config": config.to_dict(),
            "fail_reason": str(exc),
            "declarations": {
                "note": (
                    "失败关闭产物：本次 replay 在命中失败关闭条件处停止，"
                    "未静默缩池、未用其他来源填补；本文件只作工程证据"
                ),
            },
        }
        manifest_path.write_text(
            json.dumps(fail_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"FAIL_CLOSED: {exc}")
        print(f"manifest: {manifest_path}")
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if overlay is not None:
            overlay.close()


if __name__ == "__main__":
    raise SystemExit(main())
