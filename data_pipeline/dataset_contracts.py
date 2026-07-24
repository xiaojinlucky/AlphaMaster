"""AlphaMaster 数据来源与可信等级合同。"""
from __future__ import annotations

MT5_SOURCE = "MetaTrader5"
MT5_FORMAT = "alphamaster_mt5_dataset_v1"
MT5_SOURCE_ID = "mt5"

MT5_LEGACY_SOURCE = "MetaTrader5Legacy"
MT5_LEGACY_FORMAT = "alphamaster_mt5_legacy_attestation_v1"
MT5_LEGACY_SOURCE_ID = "mt5_legacy_attested"

OKX_SOURCE = "OKX"
OKX_FORMAT = "alphamaster_okx_dataset_v1"
OKX_SOURCE_ID = "okx"
OKX_LEGACY_SOURCE_ID = "okx_legacy_attested"

AKSHARE_SOURCE = "AKShare"
AKSHARE_HFQ_FORMAT = "alphamaster_ashare_akshare_sina_hfq_dataset_v1"
AKSHARE_HFQ_SOURCE_ID = "ashare_akshare_sina_hfq"

TRAINING_SOURCE_IDS = frozenset(
    {
        MT5_SOURCE_ID,
        MT5_LEGACY_SOURCE_ID,
        OKX_SOURCE_ID,
        OKX_LEGACY_SOURCE_ID,
        "ashare_local",
        AKSHARE_HFQ_SOURCE_ID,
    }
)
DATA_SOURCE_IDS = TRAINING_SOURCE_IDS | {"local_file"}

GENERIC_SOURCE_CONTRACTS = {
    (MT5_SOURCE, MT5_FORMAT): MT5_SOURCE_ID,
    (MT5_LEGACY_SOURCE, MT5_LEGACY_FORMAT): MT5_LEGACY_SOURCE_ID,
    (OKX_SOURCE, OKX_FORMAT): OKX_SOURCE_ID,
}

REMOTE_SOURCE_CONTRACTS = {
    MT5_SOURCE: (MT5_SOURCE_ID, MT5_FORMAT),
    MT5_LEGACY_SOURCE: (MT5_LEGACY_SOURCE_ID, MT5_LEGACY_FORMAT),
    OKX_SOURCE: (OKX_SOURCE_ID, OKX_FORMAT),
}

SOURCE_FAMILIES = {
    MT5_SOURCE_ID: "mt5",
    MT5_LEGACY_SOURCE_ID: "mt5",
    OKX_SOURCE_ID: "okx",
    OKX_LEGACY_SOURCE_ID: "okx",
    "ashare_local": "ashare",
    AKSHARE_HFQ_SOURCE_ID: "ashare",
    "local_file": "local",
}

_SECONDS_PER_YEAR = 365.2425 * 24 * 60 * 60
_OKX_SOURCE_BARS = {
    "M5": "5m",
    "M15": "15m",
    "H1": "1H",
    "D1": "1D",
}


def infer_periods_per_year(
    *,
    rows: int,
    start_unix: int,
    end_unix: int,
) -> int:
    """按当前文件实际覆盖范围推断可复核的年化 K 线数。"""
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 2
        or isinstance(start_unix, bool)
        or not isinstance(start_unix, int)
        or isinstance(end_unix, bool)
        or not isinstance(end_unix, int)
        or end_unix <= start_unix
    ):
        raise ValueError("无法从无效行数或时间范围推断 periods_per_year")
    inferred = round((rows - 1) * _SECONDS_PER_YEAR / (end_unix - start_unix))
    if inferred <= 0:
        raise ValueError("推断得到的 periods_per_year 非法")
    return int(inferred)


def source_family(source_id: str) -> str:
    """返回回测兼容性使用的来源族；未知来源保持原值以失败关闭。"""
    return SOURCE_FAMILIES.get(source_id, source_id)


def resolve_okx_source_id(
    payload: dict,
    *,
    symbol: str,
    timeframe: str,
) -> str:
    """区分新版下载器验证数据和旧 OKX 归档声明，禁止互相冒充。"""
    provenance_status = payload.get("provenance_status")
    if provenance_status is None:
        if payload.get("source_family") != "OKX":
            raise ValueError("新版 OKX manifest 的 source_family 必须是 OKX")
        if payload.get("provenance_level") != "downloader_verified":
            raise ValueError("新版 OKX manifest 必须声明 downloader_verified")
        if payload.get("bar_completion") != "confirmed_only":
            raise ValueError("新版 OKX manifest 必须声明仅包含已完成 K 线")
        return OKX_SOURCE_ID

    if provenance_status != "legacy_archive_attestation":
        raise ValueError("OKX manifest 的 provenance_status 不受支持")
    if payload.get("closed_bars_only") is not True:
        raise ValueError("旧 OKX 归档必须声明 closed_bars_only=true")
    if payload.get("source_endpoint") != "/api/v5/market/history-candles":
        raise ValueError("旧 OKX 归档的 source_endpoint 不匹配")
    if payload.get("source_bar") != _OKX_SOURCE_BARS.get(timeframe):
        raise ValueError("旧 OKX 归档的 source_bar 与 timeframe 不匹配")
    instrument = payload.get("source_instrument")
    if not isinstance(instrument, str) or not instrument.endswith("-SWAP"):
        raise ValueError("旧 OKX 归档的 source_instrument 非法")
    normalized_symbol = instrument.removesuffix("-SWAP").replace("-", "")
    if normalized_symbol != symbol:
        raise ValueError("旧 OKX 归档的 source_instrument 与 symbol 不匹配")
    if payload.get("volume_semantics") != (
        "OKX contract volume mapped to tick_volume"
    ):
        raise ValueError("旧 OKX 归档的 volume_semantics 不匹配")
    provenance = payload.get("provenance")
    if not isinstance(provenance, str) or not provenance.startswith(
        "user_provided_archive:"
    ):
        raise ValueError("旧 OKX 归档的 provenance 非法")
    derived = payload.get("derived_from")
    if (
        not isinstance(derived, dict)
        or not isinstance(derived.get("archive_member"), str)
        or not derived["archive_member"]
        or not isinstance(derived.get("data_sha256"), str)
        or len(derived["data_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in derived["data_sha256"])
    ):
        raise ValueError("旧 OKX 归档的 derived_from 非法")
    transform = payload.get("transform")
    dropped = (
        transform.get("dropped_trailing_unclosed_bars")
        if isinstance(transform, dict)
        else None
    )
    if (
        isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or dropped < 0
        or transform.get("cutoff_reference") != "source_file_mtime"
    ):
        raise ValueError("旧 OKX 归档的 transform 非法")
    return OKX_LEGACY_SOURCE_ID
