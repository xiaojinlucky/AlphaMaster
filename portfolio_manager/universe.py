"""把冻结股票池合同转换为组合控制器可审计的身份。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scripts.freeze_csi_a50_universe import load_frozen_universe

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CSI_A50_UNIVERSE_PATH = (
    _PROJECT_ROOT / "universes" / "csi_a50_20260723.json"
)
_CSI_A50_CONTRACT_SHA256 = (
    "987387945fba0cb778b648860bc7579a3cf49e9c3b788596464f714e968bb896"
)
_DATE_RE = re.compile(r"^[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")


@dataclass(frozen=True)
class UniverseContract:
    """组合决策必须绑定的冻结股票池身份。"""

    universe_id: str
    snapshot_date: str
    constituent_count: int
    universe_sha256: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.universe_id, str) or not self.universe_id.strip():
            raise ValueError("universe_id 必须是非空文本")
        if (
            not isinstance(self.snapshot_date, str)
            or _DATE_RE.fullmatch(self.snapshot_date) is None
        ):
            raise ValueError("snapshot_date 必须是 YYYYMMDD")
        if (
            isinstance(self.constituent_count, bool)
            or not isinstance(self.constituent_count, int)
            or self.constituent_count <= 0
        ):
            raise ValueError("constituent_count 必须是正整数")
        if (
            not isinstance(self.universe_sha256, str)
            or _SHA256_RE.fullmatch(self.universe_sha256) is None
        ):
            raise ValueError("universe_sha256 必须是小写 SHA-256")
        if not isinstance(self.symbols, tuple):
            raise ValueError("symbols 必须是不可变元组")
        if len(self.symbols) != self.constituent_count:
            raise ValueError("constituent_count 与 symbols 数量不一致")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("冻结股票池不能包含重复代码")
        if any(
            not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None
            for symbol in self.symbols
        ):
            raise ValueError("冻结股票池代码必须是 6 位数字文本")

    def to_dict(self) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "snapshot_date": self.snapshot_date,
            "constituent_count": self.constituent_count,
            "universe_sha256": self.universe_sha256,
            "symbols": list(self.symbols),
        }


def load_csi_a50_universe_contract() -> UniverseContract:
    """读取仓库内唯一可信 A50 快照，并复核独立冻结哈希。"""
    payload = load_frozen_universe(_CSI_A50_UNIVERSE_PATH)
    if payload["contract_sha256"] != _CSI_A50_CONTRACT_SHA256:
        raise ValueError("中证 A50 冻结合同与内置可信 SHA-256 不一致")
    symbols = tuple(row["symbol"] for row in payload["constituents"])
    return UniverseContract(
        universe_id=(
            f"{payload['format']}:{payload['index_code']}:{payload['snapshot_date']}"
        ),
        snapshot_date=payload["snapshot_date"],
        constituent_count=payload["constituent_count"],
        universe_sha256=payload["contract_sha256"],
        symbols=symbols,
    )


__all__ = ["UniverseContract", "load_csi_a50_universe_contract"]
