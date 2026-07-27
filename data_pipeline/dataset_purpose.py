"""Parquet 数据用途的文件内合同；只读 footer，不读取数据表。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_PURPOSE_TRAINING = "training"
DATASET_PURPOSE_SEALED_OOS = "sealed_oos_evaluation"
DATASET_PURPOSE_METADATA_KEY = b"alphamaster.dataset_purpose"
DATASET_PURPOSES = {
    DATASET_PURPOSE_TRAINING,
    DATASET_PURPOSE_SEALED_OOS,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_SEALED_SPLIT_CONTRACTS = (
    PROJECT_ROOT / "universes" / "csi_a50_20260723_sealed_20250724.json",
)


def _validate_purpose(value: str) -> str:
    if value not in DATASET_PURPOSES:
        raise ValueError("Parquet dataset purpose 不受支持")
    return value


def read_parquet_dataset_purpose(path: str | Path) -> str | None:
    """从 Parquet footer 读取用途，不物化任何数据列。"""
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(Path(path)).metadata or {}
    raw = metadata.get(DATASET_PURPOSE_METADATA_KEY)
    if raw is None:
        return None
    try:
        purpose = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Parquet dataset purpose 不是合法 UTF-8") from exc
    return _validate_purpose(purpose)


def sha256_file_bytes(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trusted_dataset_purpose_for_sha256(
    data_sha256: str,
    *,
    contract_paths: tuple[Path, ...] | None = None,
) -> str | None:
    """从 Git 固定 split 合同按数据字节哈希判定用途。"""
    if (
        not isinstance(data_sha256, str)
        or len(data_sha256) != 64
        or any(char not in "0123456789abcdef" for char in data_sha256)
    ):
        raise ValueError("data_sha256 不是小写 SHA-256")
    resolved_purpose: str | None = None
    paths = (
        TRUSTED_SEALED_SPLIT_CONTRACTS
        if contract_paths is None
        else contract_paths
    )
    for path in paths:
        if not path.is_file():
            raise ValueError(f"受信 sealed split 合同缺失: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"受信 sealed split 合同无法读取: {path}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("format")
            not in {
                "alphamaster_a50_sealed_split_v1",
                "alphamaster_a50_sealed_split_v2",
            }
            or payload.get("symbol_count") != 50
            or not isinstance(payload.get("items"), list)
            or len(payload["items"]) != 50
        ):
            raise ValueError(f"受信 sealed split 合同结构非法: {path}")
        expected_contract_sha256 = payload.get("contract_sha256")
        body = {
            key: value
            for key, value in payload.items()
            if key != "contract_sha256"
        }
        actual_contract_sha256 = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if expected_contract_sha256 != actual_contract_sha256:
            raise ValueError(f"受信 sealed split 合同哈希非法: {path}")
        for item in payload["items"]:
            if not isinstance(item, dict):
                raise ValueError(f"受信 sealed split item 非法: {path}")
            for field, purpose in (
                ("training", DATASET_PURPOSE_TRAINING),
                ("sealed_evaluation", DATASET_PURPOSE_SEALED_OOS),
            ):
                identity = item.get(field)
                if not isinstance(identity, dict):
                    raise ValueError(
                        f"受信 sealed split 缺少 {field}: {path}"
                    )
                if identity.get("data_sha256") != data_sha256:
                    continue
                if (
                    resolved_purpose is not None
                    and resolved_purpose != purpose
                ):
                    raise ValueError("同一数据哈希在受信 split 中用途冲突")
                resolved_purpose = purpose
    return resolved_purpose


def write_dataframe_with_dataset_purpose(
    frame: Any,
    path: str | Path,
    *,
    purpose: str,
) -> None:
    """写入带不可分离用途元数据的 Parquet 数据文件。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    resolved = _validate_purpose(purpose)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[DATASET_PURPOSE_METADATA_KEY] = resolved.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), Path(path))
