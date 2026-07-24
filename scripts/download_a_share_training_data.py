"""按冻结的中证 A50 合同串行下载 AKShare 前复权日线。"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.a_share_akshare import (
    AKShareDataError,
    akshare_sina_provider_symbol,
    download_akshare_hfq_daily,
    load_akshare_hfq_manifest,
)
from scripts.freeze_csi_a50_universe import (
    UniverseContractError,
    load_frozen_universe,
    write_json_exclusive,
)


SUMMARY_FORMAT = "alphamaster_a_share_download_summary_v1"
MINIMUM_THROTTLE_SECONDS = 1.0
_DATE_RE = re.compile(r"^[0-9]{8}$")
_SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
DownloadFunction = Callable[..., dict[str, Any]]
SleepFunction = Callable[[float], None]


class BatchDownloadError(RuntimeError):
    """批量下载失败；已成功发布的快照保留供下次严格恢复。"""


def _parse_date(value: str, label: str) -> str:
    text = str(value)
    if _DATE_RE.fullmatch(text) is None:
        raise BatchDownloadError(f"{label} 必须是 YYYYMMDD")
    try:
        parsed = datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise BatchDownloadError(f"{label} 不是合法日期") from exc
    return parsed.strftime("%Y%m%d")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _verify_existing_snapshot(
    *,
    output_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any] | None:
    data_file = output_dir / f"{symbol}_D1.parquet"
    manifest_file = data_file.with_suffix(".manifest.json")
    data_exists = data_file.is_file()
    manifest_exists = manifest_file.is_file()
    if not data_exists and not manifest_exists:
        return None
    if data_exists != manifest_exists:
        raise BatchDownloadError(
            f"{symbol} 的 Parquet 与 manifest 不完整，拒绝跳过"
        )
    try:
        frame = pd.read_parquet(data_file)
        manifest = load_akshare_hfq_manifest(data_file, frame)
    except Exception as exc:
        raise BatchDownloadError(
            f"{symbol} 的现有快照完整性验证失败: {type(exc).__name__}: {exc}"
        ) from exc
    if manifest is None:
        raise BatchDownloadError(f"{symbol} 的现有 manifest 不是 AKShare hfq 合同")
    expected_request = {
        "canonical_symbol": symbol,
        "symbol": akshare_sina_provider_symbol(symbol),
        "start_date": start_date,
        "end_date": end_date,
        "adjust": "hfq",
    }
    if manifest.get("request") != expected_request:
        raise BatchDownloadError(f"{symbol} 的现有快照请求范围与本批次不一致")
    return {
        "data_file": str(data_file),
        "manifest_file": str(manifest_file),
        "data_sha256": manifest["data_sha256"],
        "dataset_id": manifest["dataset_id"],
        "data_rows": manifest["data_rows"],
        "data_start": manifest["data_start"],
        "data_end": manifest["data_end"],
    }


def _summary_payload(
    *,
    universe_path: Path,
    universe: dict[str, Any],
    output_dir: Path,
    start_date: str,
    end_date: str,
    timeout: float,
    throttle_seconds: float,
    started_at: str,
    items: list[dict[str, Any]],
    download_attempt_count: int,
    status: str,
) -> dict[str, Any]:
    downloaded_count = sum(item["status"] == "downloaded" for item in items)
    skipped_count = sum(item["status"] == "skipped_verified" for item in items)
    failed_count = sum(item["status"] == "failed" for item in items)
    return {
        "format": SUMMARY_FORMAT,
        "status": status,
        "universe_file": str(universe_path),
        "universe_contract_sha256": universe["contract_sha256"],
        "universe_snapshot_date": universe["snapshot_date"],
        "output_dir": str(output_dir),
        "request": {
            "start_date": start_date,
            "end_date": end_date,
            "adjust": "hfq",
            "provider_interface": "stock_zh_a_daily",
            "timeout_seconds": timeout,
            "throttle_seconds": throttle_seconds,
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
        "total_constituents": universe["constituent_count"],
        "processed_count": len(items),
        "download_attempt_count": download_attempt_count,
        "downloaded_count": downloaded_count,
        "skipped_verified_count": skipped_count,
        "failed_count": failed_count,
        "items": items,
    }


def download_universe_data(
    *,
    universe_json: str | Path,
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    summary_json: str | Path,
    timeout: float = 20.0,
    throttle_seconds: float = MINIMUM_THROTTLE_SECONDS,
    download_func: DownloadFunction = download_akshare_hfq_daily,
    sleep_func: SleepFunction = time.sleep,
) -> dict[str, Any]:
    """下载整个冻结池；一个标的失败即停止，不重试、不继续。"""
    universe_path = Path(universe_json).resolve()
    destination = Path(output_dir).resolve()
    summary_path = Path(summary_json).resolve()
    if summary_path.exists():
        raise BatchDownloadError(f"summary 已存在，拒绝覆盖: {summary_path}")

    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start >= end:
        raise BatchDownloadError("start_date 必须早于 end_date")
    if datetime.strptime(end, "%Y%m%d").date() >= datetime.now(
        _SHANGHAI_TIMEZONE
    ).date():
        raise BatchDownloadError("end_date 必须早于上海当前日期")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise BatchDownloadError("timeout 必须是正数")
    if (
        isinstance(throttle_seconds, bool)
        or not isinstance(throttle_seconds, (int, float))
        or not math.isfinite(throttle_seconds)
        or throttle_seconds < MINIMUM_THROTTLE_SECONDS
    ):
        raise BatchDownloadError("throttle_seconds 不得小于 1 秒")

    try:
        universe = load_frozen_universe(universe_path)
    except UniverseContractError as exc:
        raise BatchDownloadError(f"冻结成分合同验证失败: {exc}") from exc
    destination.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    items: list[dict[str, Any]] = []
    download_attempt_count = 0
    for constituent in universe["constituents"]:
        symbol = constituent["symbol"]
        try:
            existing = _verify_existing_snapshot(
                output_dir=destination,
                symbol=symbol,
                start_date=start,
                end_date=end,
            )
            if existing is not None:
                items.append(
                    {
                        "symbol": symbol,
                        "name": constituent["name"],
                        "status": "skipped_verified",
                        **existing,
                    }
                )
                continue

            if download_attempt_count:
                sleep_func(float(throttle_seconds))
            download_attempt_count += 1
            download_func(
                symbol=symbol,
                start_date=start,
                end_date=end,
                output_dir=destination,
                timeout=float(timeout),
            )
            published = _verify_existing_snapshot(
                output_dir=destination,
                symbol=symbol,
                start_date=start,
                end_date=end,
            )
            if published is None:
                raise BatchDownloadError(f"{symbol} 下载后未发布完整快照")
            items.append(
                {
                    "symbol": symbol,
                    "name": constituent["name"],
                    "status": "downloaded",
                    **published,
                }
            )
        except Exception as exc:
            items.append(
                {
                    "symbol": symbol,
                    "name": constituent["name"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            summary = _summary_payload(
                universe_path=universe_path,
                universe=universe,
                output_dir=destination,
                start_date=start,
                end_date=end,
                timeout=float(timeout),
                throttle_seconds=float(throttle_seconds),
                started_at=started_at,
                items=items,
                download_attempt_count=download_attempt_count,
                status="failed",
            )
            write_json_exclusive(summary_path, summary)
            raise BatchDownloadError(
                f"{symbol} 处理失败，批次已停止；摘要: {summary_path}"
            ) from exc

    summary = _summary_payload(
        universe_path=universe_path,
        universe=universe,
        output_dir=destination,
        start_date=start,
        end_date=end,
        timeout=float(timeout),
        throttle_seconds=float(throttle_seconds),
        started_at=started_at,
        items=items,
        download_attempt_count=download_attempt_count,
        status="completed",
    )
    write_json_exclusive(summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按冻结中证 A50 成分串行下载 AKShare 前复权日线"
    )
    parser.add_argument("--universe-json", required=True, help="冻结成分 JSON")
    parser.add_argument("--output-dir", required=True, help="Parquet 输出目录")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD，必须是已收盘日")
    parser.add_argument("--summary-json", required=True, help="不可覆盖的批次摘要 JSON")
    parser.add_argument("--timeout", type=float, default=20.0, help="单次请求超时秒数")
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=MINIMUM_THROTTLE_SECONDS,
        help="两次实际下载之间的最小等待秒数，不得小于 1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = download_universe_data(
            universe_json=args.universe_json,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            summary_json=args.summary_json,
            timeout=args.timeout,
            throttle_seconds=args.throttle_seconds,
        )
    except (
        BatchDownloadError,
        AKShareDataError,
        UniverseContractError,
        OSError,
    ) as exc:
        print(f"批量下载失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
