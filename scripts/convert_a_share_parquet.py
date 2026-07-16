"""把旧 A 股个股 Parquet 显式转换为 AlphaMaster 训练副本。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.a_share_data import AShareDataError, convert_legacy_a_share_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="严格转换一个旧 A 股 stocks Parquet；不修改原文件、不自动去重"
    )
    parser.add_argument("--input-file", required=True, help="stocks 下的旧 Parquet 文件")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "local_data" / "a_share_training"),
        help="训练副本输出目录（默认: local_data/a_share_training）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = convert_legacy_a_share_file(args.input_file, args.output_dir)
    except (AShareDataError, OSError) as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
