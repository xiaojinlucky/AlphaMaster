"""独立复核 WO-AM-07A 主线 v4 的 26 只数据与连续性。

本脚本不导入构建器或构建核心。它使用独立的 factor 对齐方式重算 qfq，
逐文件核对 105 件产物，并且全程只读。
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path(
    r"G:\QuantData\free-stockdb\am_exports"
    r"\20260727_csi300_qfq_repair_v4"
)
V3_ROOT = Path(
    r"G:\QuantData\free-stockdb\am_exports"
    r"\20260726_csi300_historical_am_inputs_v3"
)
EXPECTED_V3_SHA256 = (
    "e07fffd04c9d53a897ae688ad05897a03273acf14010f799e1aca85579a8404c"
)
EXPECTED_V4_MANIFEST_SHA256 = (
    "b4a582318e4c2f94bbd10feec99894cb82b4077ab0f691ce83450f3eeebf0628"
)
EXPECTED_REPAIR_CODES = {
    "000001",
    "000408",
    "000538",
    "000651",
    "000657",
    "000661",
    "000876",
    "002074",
    "002466",
    "600039",
    "600089",
    "600219",
    "600515",
    "600660",
    "600674",
    "600690",
    "600741",
    "600760",
    "600803",
    "600886",
    "600887",
    "601607",
    "688047",
    "688506",
    "688521",
    "688981",
}
D1_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")


class MainlineAuditError(RuntimeError):
    pass


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _strict_json_object(blob: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复键 {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"非法 JSON 数值 {value}")

    try:
        value = json.loads(
            blob.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MainlineAuditError(f"{label} 不是严格 UTF-8 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise MainlineAuditError(f"{label} JSON 根节点必须是对象")
    return value


def _parse_js_assignment(raw: bytes, label: str) -> tuple[str, str]:
    text = raw.decode("utf-8-sig")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("/*", "//", "*"))
    ]
    if not lines or "=" not in lines[0]:
        raise MainlineAuditError(f"{label} 首个有效行不是 JavaScript 赋值")
    lhs, rhs = lines[0].split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if rhs.endswith(";"):
        rhs = rhs[:-1].strip()
    if not lhs or not rhs:
        raise MainlineAuditError(f"{label} JavaScript 赋值为空")
    return lhs, rhs


def _market_symbol(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _decode_history(raw: bytes, code: str) -> pd.DataFrame:
    try:
        from akshare.stock.cons import hk_js_decode
        from py_mini_racer import MiniRacer
    except ImportError as exc:
        raise MainlineAuditError("AKShare 的新浪安全解码依赖不可用") from exc

    lhs, rhs = _parse_js_assignment(raw, f"{code} 新浪历史")
    if lhs != f"var KLC_K2_{_market_symbol(code)}":
        raise MainlineAuditError(f"{code} 新浪历史变量名不符")
    try:
        encoded = json.loads(rhs)
    except json.JSONDecodeError as exc:
        raise MainlineAuditError(f"{code} 新浪历史编码不是合法 JSON") from exc
    if not isinstance(encoded, str):
        raise MainlineAuditError(f"{code} 新浪历史右侧不是字符串")
    context = MiniRacer()
    context.eval(hk_js_decode)
    records = context.call("d", encoded)
    if not isinstance(records, list) or not records:
        raise MainlineAuditError(f"{code} 新浪历史没有记录")

    required = ("date", "open", "high", "low", "close", "volume")
    frame = pd.DataFrame(records)
    if not set(required).issubset(frame.columns):
        raise MainlineAuditError(f"{code} 新浪历史缺列")
    frame = frame.loc[:, list(required)].copy()
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="raise", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise MainlineAuditError(f"{code} 新浪历史含重复日")
    numeric = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric).all().all():
        raise MainlineAuditError(f"{code} 新浪历史含非有限数值")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise MainlineAuditError(f"{code} 新浪历史含非正价格")
    if (frame["volume"] < 0).any():
        raise MainlineAuditError(f"{code} 新浪历史含负成交量")
    return frame


def _decode_factors(raw: bytes, code: str) -> pd.DataFrame:
    lhs, rhs = _parse_js_assignment(raw, f"{code} 新浪 qfq-factor")
    if lhs != f"var {_market_symbol(code)}qfq":
        raise MainlineAuditError(f"{code} 新浪 qfq-factor 变量名不符")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复键 {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            rhs,
            object_pairs_hook=reject_duplicates,
            parse_float=Decimal,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"非法 JSON 数值 {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise MainlineAuditError(f"{code} qfq-factor JSON 非法") from exc
    if not isinstance(payload, dict):
        raise MainlineAuditError(f"{code} qfq-factor 根节点不是对象")
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise MainlineAuditError(f"{code} qfq-factor 没有记录")
    if payload.get("total") is not None and int(payload["total"]) != len(records):
        raise MainlineAuditError(f"{code} qfq-factor total 不符")

    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"d", "f"}:
            raise MainlineAuditError(f"{code} qfq-factor 记录字段必须精确为 d/f")
        if isinstance(record["f"], (bool, np.bool_)):
            raise MainlineAuditError(f"{code} qfq-factor 是布尔值")
        try:
            factor = Decimal(str(record["f"]))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise MainlineAuditError(f"{code} qfq-factor 不是十进制数") from exc
        if not factor.is_finite() or factor <= 0:
            raise MainlineAuditError(f"{code} qfq-factor 必须为有限正数")
        factor_float = float(factor)
        if not math.isfinite(factor_float) or factor_float <= 0:
            raise MainlineAuditError(f"{code} qfq-factor 无法转为 float64")
        normalized.append(
            {
                "date": pd.Timestamp(record["d"]).normalize(),
                "factor": factor_float,
                "factor_decimal": format(factor, "f"),
            }
        )
    frame = pd.DataFrame(normalized).sort_values(
        "date", kind="stable"
    ).reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise MainlineAuditError(f"{code} qfq-factor 含重复日期")
    return frame


def _independent_qfq(
    history: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    aligned = pd.merge_asof(
        history.sort_values("date", kind="stable"),
        factors.sort_values("date", kind="stable"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    if aligned[["factor", "factor_decimal"]].isna().any().any():
        raise MainlineAuditError("最早交易日没有可前向匹配的 qfq 因子")
    raw_close = aligned["close"].astype("float64").copy()
    for column in ("open", "high", "low", "close"):
        aligned[column] = aligned[column].astype("float64") / aligned["factor"]
    factor_changed = (
        aligned["factor_decimal"].ne(aligned["factor_decimal"].shift(1))
        & aligned.index.to_series().gt(0)
    )
    difference = aligned["close"].pct_change().sub(
        raw_close.pct_change()
    ).abs()
    unexplained = (
        difference.gt(0.10)
        & ~factor_changed
        & difference.notna()
    )
    violations = tuple(
        aligned.loc[unexplained, "date"].dt.strftime("%Y-%m-%d")
    )
    return aligned, violations


def _independent_d1(qfq: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(qfq["date"])
    localized = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    volume = qfq["volume"].to_numpy(dtype="float64")
    if (
        not np.isfinite(volume).all()
        or (volume < 0).any()
        or not np.equal(volume, np.floor(volume)).all()
    ):
        raise MainlineAuditError("成交量不能无损转换为 int64")
    return pd.DataFrame(
        {
            "time": (
                localized.as_unit("ns").asi8 // 1_000_000_000
            ).astype("int64"),
            "open": qfq["open"].to_numpy(dtype="float32"),
            "high": qfq["high"].to_numpy(dtype="float32"),
            "low": qfq["low"].to_numpy(dtype="float32"),
            "close": qfq["close"].to_numpy(dtype="float32"),
            "tick_volume": volume.astype("int64"),
        }
    )


def _validate_manifest(
    output_root: Path,
    v3_root: Path,
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    v3_manifest = v3_root / "manifest.json"
    if _sha256(v3_manifest) != EXPECTED_V3_SHA256:
        raise MainlineAuditError("FAIL v3 manifest 哈希漂移")
    manifest_path = output_root / "manifest.json"
    if _sha256(manifest_path) != EXPECTED_V4_MANIFEST_SHA256:
        raise MainlineAuditError("FAIL v4 manifest 哈希漂移")
    manifest = _strict_json_object(manifest_path.read_bytes(), "v4 manifest")
    if (
        manifest.get("format")
        != "free_stockdb_csi300_qfq_repair_v4_mainline"
        or manifest.get("status") != "completed"
        or manifest.get("parent_v3_manifest_sha256") != EXPECTED_V3_SHA256
        or manifest.get("research_semantics")
        != {
            "point_in_time_eligible": False,
            "retrospective_only": True,
            "sealed_evaluation_eligible": False,
        }
    ):
        raise MainlineAuditError("FAIL v4 manifest 身份或研究语义不符")
    scope = manifest.get("scope")
    codes = scope.get("repair_codes") if isinstance(scope, dict) else None
    if (
        not isinstance(codes, list)
        or codes != sorted(EXPECTED_REPAIR_CODES)
        or scope.get("repair_count") != 26
        or scope.get("non_available_union_count") != 29
    ):
        raise MainlineAuditError("FAIL v4 repair code 范围不符")
    contract = manifest.get("data_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("adjustment") != "qfq"
        or contract.get("timeframe") != "D1"
        or contract.get("replay_start") != "2006-10-31"
        or contract.get("source_as_of") != "2026-07-24"
        or contract.get("columns") != list(D1_COLUMNS)
    ):
        raise MainlineAuditError("FAIL v4 数据合同不符")
    inventory_list = manifest.get("inventory")
    if not isinstance(inventory_list, list):
        raise MainlineAuditError("FAIL v4 inventory 不是列表")
    inventory: dict[str, dict[str, Any]] = {}
    for item in inventory_list:
        if (
            not isinstance(item, dict)
            or set(item) != {"relative_path", "bytes", "sha256"}
            or not isinstance(item["relative_path"], str)
            or item["relative_path"] in inventory
        ):
            raise MainlineAuditError("FAIL v4 inventory 字段或路径唯一性不符")
        inventory[item["relative_path"]] = item
    expected_paths = {
        rel
        for code in codes
        for rel in (
            f"D1/{code}_D1.parquet",
            f"audits/{code}.json",
            f"source/{code}/sina_history.js",
            f"source/{code}/sina_qfq_factor.js",
        )
    }
    if set(inventory) != expected_paths:
        raise MainlineAuditError("FAIL v4 inventory 路径集合不符")
    actual_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_paths | {"manifest.json"}:
        raise MainlineAuditError("FAIL v4 实际文件集合不是精确 105 件")
    return manifest, codes, inventory


def audit(v3_root: Path, output_root: Path) -> None:
    manifest, codes, inventory = _validate_manifest(output_root, v3_root)
    total_violations = 0
    total_ohlc_violations = 0
    for code in codes:
        for rel in (
            f"D1/{code}_D1.parquet",
            f"audits/{code}.json",
            f"source/{code}/sina_history.js",
            f"source/{code}/sina_qfq_factor.js",
        ):
            path = output_root / rel
            item = inventory[rel]
            if (
                type(item["bytes"]) is not int
                or item["bytes"] < 0
                or path.stat().st_size != item["bytes"]
                or _sha256(path) != item["sha256"]
            ):
                raise MainlineAuditError(f"FAIL {code} inventory 不符：{rel}")

        history_blob = (
            output_root / f"source/{code}/sina_history.js"
        ).read_bytes()
        factor_blob = (
            output_root / f"source/{code}/sina_qfq_factor.js"
        ).read_bytes()
        history = _decode_history(history_blob, code)
        factors = _decode_factors(factor_blob, code)
        history = history[
            history["date"].between(
                manifest["data_contract"]["replay_start"],
                manifest["data_contract"]["source_as_of"],
            )
        ].reset_index(drop=True)
        qfq, violations = _independent_qfq(history, factors)
        ohlc_bad = (
            qfq["high"].lt(qfq[["open", "low", "close"]].max(axis=1))
            | qfq["low"].gt(qfq[["open", "high", "close"]].min(axis=1))
        )
        expected = _independent_d1(qfq)
        actual = pd.read_parquet(output_root / f"D1/{code}_D1.parquet")
        if tuple(actual.columns) != D1_COLUMNS:
            raise MainlineAuditError(f"FAIL {code} D1 列不符")
        for column in D1_COLUMNS:
            if actual[column].dtype != expected[column].dtype:
                raise MainlineAuditError(f"FAIL {code} {column} dtype 不符")
            if not np.array_equal(
                actual[column].to_numpy(),
                expected[column].to_numpy(),
            ):
                raise MainlineAuditError(f"FAIL {code} {column} 值不符")
        record = _strict_json_object(
            (output_root / f"audits/{code}.json").read_bytes(),
            f"{code} audit",
        )
        if (
            record.get("code") != code
            or record.get("rows") != len(actual)
            or record.get("continuity_violations") != 0
            or record.get("ohlc_violations") != 0
            or record.get("inputs")
            != {
                "sina_history_sha256": _sha256_bytes(history_blob),
                "sina_qfq_factor_sha256": _sha256_bytes(factor_blob),
            }
            or record.get("output", {}).get("sha256")
            != _sha256(output_root / f"D1/{code}_D1.parquet")
        ):
            raise MainlineAuditError(f"FAIL {code} 审计记录不符")
        total_violations += len(violations)
        total_ohlc_violations += int(ohlc_bad.sum())

    if total_violations or total_ohlc_violations:
        raise MainlineAuditError(
            "FAIL "
            f"continuity_violations={total_violations} "
            f"ohlc_violations={total_ohlc_violations}"
        )
    print(
        "AUDIT PASS "
        f"repair_codes={len(codes)} "
        "continuity_violations=0 ohlc_violations=0 "
        f"v3_manifest_sha256={EXPECTED_V3_SHA256} "
        f"v4_manifest_sha256={EXPECTED_V4_MANIFEST_SHA256}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="独立复核 WO-AM-07A 26 只 qfq v4"
    )
    parser.add_argument("--v3", type=Path, default=V3_ROOT)
    parser.add_argument("--v4", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    audit(args.v3.resolve(), args.v4.resolve())


if __name__ == "__main__":
    main()
