"""旧 MT5 Parquet 两阶段审计注册 CLI。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _default_artifact(kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("scratch") / f"legacy_mt5_{kind}_{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为用户确认来源的旧 MT5 Parquet 建立可审计 sidecar"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="只读扫描并生成注册计划")
    plan.add_argument("--input-dir", required=True)
    plan.add_argument("--recursive", action="store_true")
    plan.add_argument(
        "--source-report",
        action="append",
        default=[],
        help="可重复传入，用于绑定同一批数据的多份来源报告",
    )
    plan.add_argument("--feed-id", default="")
    plan.add_argument("--output-plan")

    apply = sub.add_parser("apply", help="按固定计划重新验证并发布 sidecar")
    apply.add_argument("--plan", required=True)
    apply.add_argument("--plan-sha256", required=True)
    apply.add_argument(
        "--acknowledge-source",
        required=True,
        choices=("MetaTrader5",),
        help="明确确认计划中的精确文件字节来自旧 MetaTrader5 数据",
    )
    apply.add_argument("--output-report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from data_pipeline.legacy_mt5_registry import (
        LegacyRegistrationError,
        apply_registration_plan,
        build_registration_plan,
        load_registration_plan,
        write_registration_plan,
        write_registration_report,
    )

    try:
        if args.command == "plan":
            plan = build_registration_plan(
                args.input_dir,
                recursive=bool(args.recursive),
                source_report=args.source_report,
                feed_id=args.feed_id,
            )
            target = Path(args.output_plan) if args.output_plan else _default_artifact("plan")
            path = write_registration_plan(target, plan)
            print(json.dumps(plan["summary"], ensure_ascii=False))
            print(f"计划: {path}")
            print(f"plan_sha256: {plan['plan_sha256']}")
            return 0 if plan["summary"]["rejected"] == 0 else 2

        plan = load_registration_plan(args.plan)
        report = apply_registration_plan(
            plan,
            expected_plan_sha256=args.plan_sha256,
            source_acknowledgement=args.acknowledge_source,
        )
        target = (
            Path(args.output_report)
            if args.output_report
            else _default_artifact("report")
        )
        path = write_registration_report(target, report)
        print(json.dumps(report["summary"], ensure_ascii=False))
        print(f"报告: {path}")
        print(f"report_sha256: {report['report_sha256']}")
        return 0 if report["summary"]["failed"] == 0 else 3
    except (LegacyRegistrationError, OSError, ValueError) as exc:
        print(f"注册失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
