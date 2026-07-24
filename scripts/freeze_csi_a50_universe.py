"""把中证指数官网 XLS 冻结为不可覆盖的中证 A50 成分合同。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_XLS = PROJECT_ROOT / "scratch" / "930050cons.xls"
DEFAULT_SOURCE_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/"
    "file/autofile/cons/930050cons.xls"
)
UNIVERSE_FORMAT = "alphamaster_csi_a50_universe_v1"
INDEX_CODE = "930050"
INDEX_NAME = "中证A50"
INDEX_NAME_EN = "CSI A50"
EXPECTED_CONSTITUENT_COUNT = 50
EXPECTED_COLUMNS = (
    "日期Date",
    "指数代码 Index Code",
    "指数名称 Index Name",
    "指数英文名称Index Name(Eng)",
    "成份券代码Constituent Code",
    "成份券名称Constituent Name",
    "成份券英文名称Constituent Name(Eng)",
    "交易所Exchange",
    "交易所英文名称Exchange(Eng)",
)
CONSTITUENT_KEYS = (
    "symbol",
    "name",
    "name_en",
    "exchange",
    "exchange_name",
    "exchange_name_en",
)
UNIVERSE_KEYS = (
    "format",
    "index_code",
    "index_name",
    "index_name_en",
    "snapshot_date",
    "constituent_count",
    "source_url",
    "source_xls_sha256",
    "constituents",
    "contract_sha256",
)
EXCHANGE_IDENTITIES = {
    ("上海证券交易所", "Shanghai Stock Exchange"): "SSE",
    ("深圳证券交易所", "Shenzhen Stock Exchange"): "SZSE",
}
_DATE_RE = re.compile(r"^[0-9]{8}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UniverseContractError(RuntimeError):
    """官网成分文件或冻结 JSON 不满足固定合同。"""


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _contract_sha256(payload_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload_without_hash)).hexdigest()


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> Path:
    """同目录临时写入后用硬链接发布，目标已存在时绝不覆盖。"""
    target = Path(path).resolve()
    if target.exists():
        raise UniverseContractError(f"目标已存在，拒绝覆盖: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, target)
        except FileExistsError as exc:
            raise UniverseContractError(f"目标在发布期间被创建，拒绝覆盖: {target}") from exc
    finally:
        temp.unlink(missing_ok=True)
    return target


def _require_exact_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UniverseContractError(f"{label} 必须是非空文本")
    if value != value.strip():
        raise UniverseContractError(f"{label} 不得含首尾空白")
    return value


def _parse_snapshot_date(value: Any) -> str:
    text = _require_exact_text(value, "快照日期")
    if _DATE_RE.fullmatch(text) is None:
        raise UniverseContractError("快照日期必须是 YYYYMMDD")
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise UniverseContractError("快照日期不是合法日期") from exc
    return parsed.strftime("%Y%m%d")


def _validate_constituents(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != EXPECTED_CONSTITUENT_COUNT:
        raise UniverseContractError("成分列表必须恰好包含 50 只股票")

    validated: list[dict[str, str]] = []
    symbols: set[str] = set()
    for position, raw in enumerate(value, start=1):
        if (
            not isinstance(raw, dict)
            or len(raw) != len(CONSTITUENT_KEYS)
            or set(raw) != set(CONSTITUENT_KEYS)
        ):
            raise UniverseContractError(f"第 {position} 行成分字段合同发生变化")
        row = {
            key: _require_exact_text(raw.get(key), f"第 {position} 行 {key}")
            for key in CONSTITUENT_KEYS
        }
        symbol = row["symbol"]
        if _SYMBOL_RE.fullmatch(symbol) is None:
            raise UniverseContractError(f"第 {position} 行股票代码必须是 6 位数字")
        if symbol in symbols:
            raise UniverseContractError(f"股票代码重复: {symbol}")
        symbols.add(symbol)

        exchange_identity = (
            row["exchange_name"],
            row["exchange_name_en"],
        )
        expected_exchange = EXCHANGE_IDENTITIES.get(exchange_identity)
        if expected_exchange is None or row["exchange"] != expected_exchange:
            raise UniverseContractError(f"{symbol} 的交易所合同不合法")
        validated.append(row)
    return validated


def validate_universe_payload(payload: Any) -> dict[str, Any]:
    """严格复核冻结 JSON，并验证可重算合同哈希。"""
    if (
        not isinstance(payload, dict)
        or len(payload) != len(UNIVERSE_KEYS)
        or set(payload) != set(UNIVERSE_KEYS)
    ):
        raise UniverseContractError("冻结 JSON 顶层字段合同发生变化")
    expected_scalars = {
        "format": UNIVERSE_FORMAT,
        "index_code": INDEX_CODE,
        "index_name": INDEX_NAME,
        "index_name_en": INDEX_NAME_EN,
        "constituent_count": EXPECTED_CONSTITUENT_COUNT,
        "source_url": DEFAULT_SOURCE_URL,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            raise UniverseContractError(f"冻结 JSON 的 {field} 不匹配")

    snapshot_date = _parse_snapshot_date(payload.get("snapshot_date"))
    source_hash = payload.get("source_xls_sha256")
    if not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None:
        raise UniverseContractError("source_xls_sha256 必须是小写 SHA-256")
    constituents = _validate_constituents(payload.get("constituents"))

    contract_hash = payload.get("contract_sha256")
    if not isinstance(contract_hash, str) or _SHA256_RE.fullmatch(contract_hash) is None:
        raise UniverseContractError("contract_sha256 必须是小写 SHA-256")
    contract_body = {key: payload[key] for key in UNIVERSE_KEYS[:-1]}
    if _contract_sha256(contract_body) != contract_hash:
        raise UniverseContractError("contract_sha256 与冻结内容不一致")

    return {
        **payload,
        "snapshot_date": snapshot_date,
        "constituents": constituents,
    }


def load_frozen_universe(path: str | Path) -> dict[str, Any]:
    universe_path = Path(path).resolve()
    try:
        payload = json.loads(universe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UniverseContractError("冻结 JSON 不是合法 UTF-8 JSON") from exc
    return validate_universe_payload(payload)


def _build_payload(frame: pd.DataFrame, source_xls_sha256: str) -> dict[str, Any]:
    columns = tuple(str(column) for column in frame.columns)
    if columns != EXPECTED_COLUMNS:
        raise UniverseContractError(
            "官网 XLS 列合同发生变化，拒绝猜测映射: "
            f"期望 {list(EXPECTED_COLUMNS)}，实际 {list(columns)}"
        )
    if len(frame) != EXPECTED_CONSTITUENT_COUNT:
        raise UniverseContractError(
            f"官网 XLS 必须恰好有 50 行，实际 {len(frame)} 行"
        )
    if bool(frame.isna().any().any()):
        raise UniverseContractError("官网 XLS 含空值")

    dates: set[str] = set()
    index_codes: set[str] = set()
    index_names: set[str] = set()
    index_names_en: set[str] = set()
    constituents: list[dict[str, str]] = []
    for position, raw in enumerate(frame.to_dict(orient="records"), start=1):
        row = {
            str(key): _require_exact_text(value, f"官网 XLS 第 {position} 行 {key}")
            for key, value in raw.items()
        }
        dates.add(_parse_snapshot_date(row[EXPECTED_COLUMNS[0]]))
        index_codes.add(row[EXPECTED_COLUMNS[1]])
        index_names.add(row[EXPECTED_COLUMNS[2]])
        index_names_en.add(row[EXPECTED_COLUMNS[3]])

        exchange_identity = (
            row[EXPECTED_COLUMNS[7]],
            row[EXPECTED_COLUMNS[8]],
        )
        exchange = EXCHANGE_IDENTITIES.get(exchange_identity)
        if exchange is None:
            raise UniverseContractError(
                f"官网 XLS 第 {position} 行交易所不受支持: {exchange_identity}"
            )
        constituents.append(
            {
                "symbol": row[EXPECTED_COLUMNS[4]],
                "name": row[EXPECTED_COLUMNS[5]],
                "name_en": row[EXPECTED_COLUMNS[6]],
                "exchange": exchange,
                "exchange_name": exchange_identity[0],
                "exchange_name_en": exchange_identity[1],
            }
        )

    if len(dates) != 1:
        raise UniverseContractError("官网 XLS 的成分快照日期必须完全一致")
    if index_codes != {INDEX_CODE}:
        raise UniverseContractError("官网 XLS 的指数代码必须全部为 930050")
    if index_names != {INDEX_NAME} or index_names_en != {INDEX_NAME_EN}:
        raise UniverseContractError("官网 XLS 的中英文指数名称不匹配")
    validated_constituents = _validate_constituents(constituents)

    contract_body: dict[str, Any] = {
        "format": UNIVERSE_FORMAT,
        "index_code": INDEX_CODE,
        "index_name": INDEX_NAME,
        "index_name_en": INDEX_NAME_EN,
        "snapshot_date": next(iter(dates)),
        "constituent_count": EXPECTED_CONSTITUENT_COUNT,
        "source_url": DEFAULT_SOURCE_URL,
        "source_xls_sha256": source_xls_sha256,
        "constituents": validated_constituents,
    }
    payload = {
        **contract_body,
        "contract_sha256": _contract_sha256(contract_body),
    }
    return validate_universe_payload(payload)


def freeze_csi_a50_universe(
    input_xls: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    source = Path(input_xls).resolve()
    target = Path(output_json).resolve()
    if target.exists():
        raise UniverseContractError(f"目标已存在，拒绝覆盖: {target}")
    if not source.is_file():
        raise UniverseContractError(f"官网 XLS 不存在: {source}")
    try:
        source_bytes = source.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        frame = pd.read_excel(
            io.BytesIO(source_bytes),
            sheet_name=0,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise UniverseContractError(
            f"无法读取官网 XLS: {type(exc).__name__}: {exc}"
        ) from exc
    payload = _build_payload(frame, source_hash)
    write_json_exclusive(target, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="严格冻结中证指数官网当前中证 A50 的 50 只成分"
    )
    parser.add_argument(
        "--input-xls",
        default=str(DEFAULT_INPUT_XLS),
        help=f"官网成分 XLS（默认: {DEFAULT_INPUT_XLS}）",
    )
    parser.add_argument("--output-json", required=True, help="不可覆盖的冻结 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = freeze_csi_a50_universe(args.input_xls, args.output_json)
    except UniverseContractError as exc:
        print(f"冻结失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
