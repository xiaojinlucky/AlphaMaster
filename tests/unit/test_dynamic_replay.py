# -*- coding: utf-8 -*-
"""RND-04C 动态组合日线 replay 的验收与失败关闭测试。

覆盖交接第 9 节第三步 10 项验收 + 裁决第 17/18/19 组（04C 侧）：
历史成分进入/退出、2009-12-31 空窗、停牌拒单与复牌边界、ST 进入/退出、
10%/20% 切换、上市初期 0 限价（0 限价不判 locked + 688 整手 200）、
触及显式限价保守拒单、崩溃恢复幂等、账本篡改失败关闭、未来状态追加
不改变历史 replay、全链身份篡改检出、FreeStockDB 漏数日失败关闭、
quarantine 成分 → signals 完整覆盖门失败。

合成 fixture 为主；真实 G 盘用例（skipif 保护）覆盖：真实历史调样日、
真实停牌日拒单、真实触及限价拒单、真实漏数日、真实时点 quarantine
失败关闭与真实 2009 空窗。
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import portfolio_manager.universe as universe_module
from data_pipeline.rqalpha_execution_overlay import (
    DERIVATION_RULE_DESCRIPTION,
    DERIVATION_RULE_VERSION,
    FORBIDDEN_SEMANTICS,
    FROZEN_CALENDAR_PATH,
    PERMITTED_SEMANTICS,
    RQALPHA_EXTRACTED_ROOT,
    ST_DATASET_SEMANTICS,
    STOCKS_ALLOWED_FIELDS,
    SUSPENDED_MISSING_DATASET_SEMANTICS,
    TRUSTED_OVERLAY_ANCHOR,
    V3_EXPORT_ROOT,
    RQAlphaOverlayIdentity,
    load_rqalpha_execution_overlay,
)
from portfolio_manager.execution import AShareFeeSchedule
from portfolio_manager.ledger import PortfolioDecisionLedger
from portfolio_manager.replay import (
    DECISION_VALUATION_LOT_SIZE_PLACEHOLDER,
    DECISION_VALUATION_STATUS_PLACEHOLDER,
    PRODUCTION_CSI300_HISTORY_ROOT,
    PRODUCTION_CSI300_TRUST_POLICY,
    PRODUCTION_OVERLAY_IDENTITY_SHA256,
    REPLAY_UNIVERSE_QUERY_MODE,
    DynamicDailyReplay,
    DynamicReplayConfig,
    DynamicReplayError,
)
from portfolio_manager.universe import (
    UniverseAvailabilityError,
    load_csi300_historical_universe_contract,
)

_G_DRIVE_READY = (
    RQALPHA_EXTRACTED_ROOT.is_dir()
    and V3_EXPORT_ROOT.is_dir()
    and FROZEN_CALENDAR_PATH.is_file()
    and PRODUCTION_CSI300_HISTORY_ROOT.is_dir()
)
requires_real_data = pytest.mark.skipif(
    not _G_DRIVE_READY,
    reason="G 盘 RQAlpha bundle / v3 导出 / 冻结日历 / 历史权重根不可读",
)

FEES = AShareFeeSchedule(
    commission_rate=0.0003,
    minimum_commission=5.0,
    stamp_duty_rate=0.0005,
    transfer_fee_rate=0.00001,
    slippage_rate=0.001,
)

_UNIVERSE_FORMAT = "free_stockdb_csi300_weight_history_v1"

_STOCK_DTYPE = np.dtype(
    [
        ("datetime", "<i8"),
        ("open", "<f8"),
        ("close", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("prev_close", "<f8"),
        ("limit_up", "<f8"),
        ("limit_down", "<f8"),
        ("volume", "<f8"),
        ("total_turnover", "<f8"),
    ]
)

# 合成交易日：2026-06-15 ~ 2026-06-26（10 个交易日，周末 06-20/21 缺席）。
SESSIONS = [
    20260615, 20260616, 20260617, 20260618, 20260619,
    20260622, 20260623, 20260624, 20260625, 20260626,
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso(day: int) -> str:
    text = str(day)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _bar(day, *, close, prev_close, limit_up, limit_down, volume=1_000_000.0):
    return (
        day * 1_000_000,
        close - 0.1,
        close,
        close + 0.1,
        close - 0.2,
        prev_close,
        limit_up,
        limit_down,
        volume,
        volume * close,
    )


def _rq_bars() -> dict[str, list[tuple]]:
    """合成 RQAlpha 未复权 bar：600001 触限/ST/制度切换，600002 停牌，
    688001 全程 0 限价，600005 全交易日有 bar（v3 侧故意缺 06-18）。"""
    bars: dict[str, list[tuple]] = {}
    seq_600001 = [
        (20260615, 10.0, 9.8, 10.78, 8.82),
        (20260616, 11.0, 10.0, 11.0, 9.0),      # 触及显式涨停价
        (20260617, 10.5, 11.0, 12.1, 9.9),
        (20260618, 10.6, 10.5, 11.55, 9.45),    # 10% 限价
        (20260619, 10.7, 10.6, 11.66, 9.54),    # 10% 限价
        (20260622, 10.8, 10.7, 12.84, 8.56),    # 切换为 20% 限价
        (20260623, 10.9, 10.8, 12.96, 8.64),
        (20260624, 11.0, 10.9, 13.08, 8.72),
        (20260625, 11.1, 11.0, 13.20, 8.80),
        (20260626, 11.2, 11.1, 13.32, 8.88),
    ]
    bars["600001.XSHG"] = [
        _bar(d, close=c, prev_close=p, limit_up=u, limit_down=l)
        for d, c, p, u, l in seq_600001
    ]

    def _plain(prefix_close: float, skip: set[int] = frozenset()) -> list:
        rows = []
        prev = prefix_close
        for index, day in enumerate(SESSIONS):
            close = round(prefix_close + 0.1 * index, 2)
            if day not in skip:
                rows.append(
                    _bar(
                        day,
                        close=close,
                        prev_close=prev,
                        limit_up=round(prev * 1.1, 2),
                        limit_down=round(prev * 0.9, 2),
                    )
                )
            prev = close
        return rows

    bars["600002.XSHE"] = _plain(20.0, skip={20260617, 20260618})
    bars["600003.XSHG"] = _plain(30.0)
    bars["600004.XSHG"] = _plain(15.0)
    bars["600005.XSHG"] = _plain(8.0)
    bars["600007.XSHG"] = _plain(12.0)
    bars["600008.XSHG"] = _plain(13.0)
    bars["688001.XSHG"] = [
        _bar(
            day,
            close=round(50.0 + 0.1 * index, 2),
            prev_close=round(50.0 + 0.1 * (index - 1), 2) if index else 49.9,
            limit_up=0.0,
            limit_down=0.0,
        )
        for index, day in enumerate(SESSIONS)
    ]
    return bars


def _build_world(tmp_path: Path, *, future_append: bool = False) -> dict:
    """构造合成 bundle / v3 导出 / 冻结日历与注入锚。

    v3 目录只写一次（冻结发布语义）；future_append=True 时只把"未来
    执行状态"追加进 RQAlpha bundle（bar/停牌/交易日），v3 与日历不变。
    """
    import h5py
    import pandas as pd

    bundle = tmp_path / "bundle"
    v3_root = tmp_path / "v3"
    bundle.mkdir(exist_ok=True)

    bars = _rq_bars()
    if future_append:
        bars["600001.XSHG"].append(
            _bar(
                20260701, close=11.3, prev_close=11.2,
                limit_up=12.32, limit_down=10.08,
            )
        )
        bars["600003.XSHG"].append(
            _bar(
                20260701, close=31.0, prev_close=30.9,
                limit_up=33.99, limit_down=27.81,
            )
        )
    with h5py.File(bundle / "stocks.h5", "w") as handle:
        for key, rows in bars.items():
            handle.create_dataset(key, data=np.array(rows, dtype=_STOCK_DTYPE))

    suspended_600002 = [20260617, 20260618]
    if future_append:
        suspended_600002.append(20260701)
    with h5py.File(bundle / "suspended_days.h5", "w") as handle:
        handle.create_dataset(
            "600002.XSHE", data=np.array(suspended_600002, dtype="int64")
        )

    with h5py.File(bundle / "st_stock_days.h5", "w") as handle:
        # 600001 于 06-18 进入 ST、06-19 之后退出（存储为严格降序）。
        handle.create_dataset(
            "600001.XSHG", data=np.array([20260619, 20260618], dtype="int64")
        )
        for key in bars:
            if key != "600001.XSHG":
                handle.create_dataset(key, data=np.array([], dtype="float64"))

    def _cs(code: str, obid: str, lot: int) -> dict:
        return {
            "type": "CS",
            "order_book_id": obid,
            "trading_code": code,
            "listed_date": "2020-01-02",
            "de_listed_date": "0000-00-00",
            "round_lot": lot,
        }

    instruments = [{"type": "INDX", "order_book_id": "000300.XSHG"}]
    for obid in bars:
        code = obid.split(".")[0]
        instruments.append(_cs(code, obid, 200 if code.startswith("688") else 100))
    (bundle / "instruments.pk").write_bytes(
        pickle.dumps(instruments, protocol=2)
    )

    trading_days = SESSIONS + [20260701]
    np.save(bundle / "trading_dates.npy", np.array(trading_days, dtype="int64"))
    for name in ("dividends.h5", "split_factor.h5", "ex_cum_factor.h5"):
        (bundle / name).write_bytes(f"placeholder:{name}".encode("utf-8"))

    calendar_path = tmp_path / "trade_calendar.parquet"
    if not calendar_path.is_file():
        pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [str(d) for d in SESSIONS], format="%Y%m%d"
                )
            }
        ).to_parquet(calendar_path)

    if not v3_root.is_dir():
        v3_root.mkdir()
        (v3_root / "D1").mkdir()
        statuses = {
            "600001": "available",
            "600002": "available",
            "600003": "available",
            "600004": "available",
            "600005": "available",
            "688001": "available",
            "600007": "quarantine",
            "600008": "source_missing",
        }
        cov_rows = []
        for code, status in statuses.items():
            relative = f"D1/{code}_D1.parquet"
            sha = "0" * 64
            if status == "available":
                rq_rows = bars[
                    next(k for k in bars if k.startswith(code))
                ]
                day_close = {
                    int(row[0] // 1_000_000): float(row[2]) for row in rq_rows
                }
                days = [d for d in SESSIONS if d in day_close]
                if code == "600005":
                    # FreeStockDB 漏数日语义：RQAlpha 有 bar、非停牌，
                    # 但 v3 冻结价格缺 06-18。
                    days = [d for d in days if d != 20260618]
                stamps = [
                    int(
                        pd.Timestamp(
                            f"{_iso(d)} 15:00:00", tz="Asia/Shanghai"
                        ).timestamp()
                    )
                    for d in days
                ]
                closes = np.array(
                    [round(day_close[d] * 0.8, 2) for d in days],
                    dtype="float32",
                )
                frame = pd.DataFrame(
                    {
                        "time": np.array(stamps, dtype="int64"),
                        "open": closes,
                        "high": closes,
                        "low": closes,
                        "close": closes,
                        "tick_volume": np.array([1000] * len(days), "int64"),
                    }
                )
                data_path = v3_root / relative
                frame.to_parquet(data_path, index=False)
                sha = _sha256_file(data_path)
            cov_rows.append(
                {
                    "code": code,
                    "status": status,
                    "data_relative_path": relative,
                    "data_sha256": sha,
                }
            )
        coverage_path = v3_root / "coverage_matrix.parquet"
        pd.DataFrame(cov_rows).to_parquet(coverage_path, index=False)
        manifest = {
            "format": "free_stockdb_csi300_historical_am_inputs_v3",
            "status": "completed",
            "status_counts": {
                "available": 6,
                "quarantine": 1,
                "source_missing": 1,
            },
            "coverage_matrix": {
                "relative_path": "coverage_matrix.parquet",
                "sha256": _sha256_file(coverage_path),
            },
        }
        (v3_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    coverage_sha = _sha256_file(v3_root / "coverage_matrix.parquet")
    anchor = {
        "format": TRUSTED_OVERLAY_ANCHOR["format"],
        "bundle_name": "synthetic_replay_bundle",
        "archive_sha256": "1" * 64,
        "member_inventory_sha256": "2" * 64,
        "member_sha256": {
            name: _sha256_file(bundle / name)
            for name in TRUSTED_OVERLAY_ANCHOR["member_sha256"]
        },
        "session_first": 20260615,
        "session_last_inclusive": 20260630,
        "available_code_count": 6,
        "v3_manifest_sha256": _sha256_file(v3_root / "manifest.json"),
        "v3_coverage_matrix_sha256": coverage_sha,
        "frozen_calendar_sha256": _sha256_file(calendar_path),
        "calendar_intersection_rows": len(SESSIONS),
        "prev_close_excluded": ("990018.XSHG",),
        "unresolved_gaps": (),
        "lot_size_allowed": (100, 200),
    }
    return {
        "bundle": bundle,
        "v3_root": v3_root,
        "calendar_path": calendar_path,
        "anchor": anchor,
    }


def _load_world_overlay(world: dict):
    return load_rqalpha_execution_overlay(
        world["bundle"],
        v3_export_root=world["v3_root"],
        frozen_calendar_path=world["calendar_path"],
        trusted_anchor=world["anchor"],
    )


# ---------------------------------------------------------------------------
# 历史股票池 fixture（append 语义：既有快照字节不重写）
# ---------------------------------------------------------------------------


def _publish_month(root: Path, *, requested_date: str, effective_date: str,
                   symbols: tuple[str, ...]) -> dict:
    import pandas as pd

    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    stem = requested_date.replace("-", "")
    data_path = root / "data" / f"{stem}.parquet"
    receipt_path = root / "manifests" / f"{stem}.json"
    if receipt_path.is_file():
        # append 语义：既有快照保持字节不变，直接复用其 receipt。
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    weight = 100.0 / len(symbols)
    frame = pd.DataFrame(
        {
            "code": list(symbols),
            "date": [effective_date] * len(symbols),
            "weight": [weight] * len(symbols),
            "display_name": [f"股票{symbol}" for symbol in symbols],
        }
    )
    frame.to_parquet(data_path, index=False)
    receipt = {
        "format": _UNIVERSE_FORMAT,
        "requested_date": requested_date,
        "actual_weight_date": effective_date,
        "rows": len(frame),
        "weight_sum": float(frame["weight"].sum()),
        "data_file": data_path.name,
        "data_bytes": data_path.stat().st_size,
        "data_sha256": _sha256_file(data_path),
        "captured_at_utc": "2026-07-25T04:00:00Z",
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def _publish_history(root: Path, months: list[dict]) -> None:
    import pandas as pd

    items = [
        _publish_month(
            root,
            requested_date=month["requested_date"],
            effective_date=month["effective_date"],
            symbols=month["symbols"],
        )
        for month in months
    ]
    combined_path = root / "csi300_weight_history.parquet"
    combined = pd.concat(
        [pd.read_parquet(root / "data" / str(item["data_file"])) for item in items],
        ignore_index=True,
    )
    combined.to_parquet(combined_path, index=False)
    requested = [str(item["requested_date"]) for item in items]
    effective = [str(item["actual_weight_date"]) for item in items]
    manifest = {
        "format": _UNIVERSE_FORMAT,
        "status": "completed",
        "endpoint": "test.invalid:1",
        "sdk_file": "test/stockdb.pyd",
        "sdk_sha256": "a" * 64,
        "index": "000300.XSHG",
        "request_count": len(items),
        "successful_api_calls": len(items),
        "first_requested_date": min(requested),
        "last_requested_date": max(requested),
        "first_actual_weight_date": min(effective),
        "last_actual_weight_date": max(effective),
        "unique_actual_weight_dates": len(set(effective)),
        "total_rows": len(combined),
        "combined_file": combined_path.name,
        "combined_bytes": combined_path.stat().st_size,
        "combined_sha256": _sha256_file(combined_path),
        "items": items,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _register_test_root(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    policy: str = "test_csi300_history_roots",
) -> str:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = (
        "format", "status", "endpoint", "sdk_sha256", "index",
        "request_count", "successful_api_calls", "first_requested_date",
        "last_requested_date", "first_actual_weight_date",
        "last_actual_weight_date", "unique_actual_weight_dates",
        "total_rows", "combined_file", "combined_bytes", "combined_sha256",
    )
    policies = deepcopy(universe_module._CSI300_HISTORY_TRUSTED_ROOTS)
    roots = policies.setdefault(policy, {})
    roots[_sha256_file(manifest_path)] = {
        key: manifest[key] for key in expected_keys
    }
    monkeypatch.setattr(
        universe_module, "_CSI300_HISTORY_TRUSTED_ROOTS", policies
    )
    return policy


MAIN_MONTHS = [
    {
        "requested_date": "2026-06-12",
        "effective_date": "2026-06-12",
        "symbols": ("600001", "600002", "600003", "688001"),
    },
    {
        "requested_date": "2026-06-23",
        "effective_date": "2026-06-23",
        "symbols": ("600001", "600002", "600004", "688001"),
    },
]
FUTURE_MONTH = {
    "requested_date": "2026-06-30",
    "effective_date": "2026-06-30",
    "symbols": ("600001", "600002", "600004", "600005"),
}


def _make_engine(world, overlay, ledger, config, universe_root, policy):
    return DynamicDailyReplay(
        config,
        overlay=overlay,
        ledger=ledger,
        fee_schedule=FEES,
        history_root=universe_root,
        trust_policy=policy,
        v3_export_root=world["v3_root"],
        calendar_path=world["calendar_path"],
        expected_overlay_identity_sha256=overlay.identity.identity_sha256,
        expected_v3_manifest_sha256=world["anchor"]["v3_manifest_sha256"],
        expected_v3_coverage_sha256=(
            world["anchor"]["v3_coverage_matrix_sha256"]
        ),
        expected_calendar_sha256=world["anchor"]["frozen_calendar_sha256"],
    )


def _main_config(**overrides) -> DynamicReplayConfig:
    payload = {
        "start_date": "2026-06-15",
        "end_date": "2026-06-26",
        "top_k": 3,
        "dropout_rank": 3,
        "initial_cash": 1_000_000.0,
        "run_label": "synthetic-main",
    }
    payload.update(overrides)
    return DynamicReplayConfig(**payload)


def _orders_of(manifest: dict, execution_session: str) -> list[dict]:
    day = next(
        record for record in manifest["days"]
        if record["execution_session"] == execution_session
    )
    return day["orders"]


def _order(manifest, execution_session, symbol, side) -> dict:
    matches = [
        order
        for order in _orders_of(manifest, execution_session)
        if order["symbol"] == symbol and order["side"] == side
    ]
    assert len(matches) == 1, (execution_session, symbol, side, matches)
    return matches[0]


def _binding_for(ledger, manifest, execution_session) -> dict:
    day = next(
        record for record in manifest["days"]
        if record["execution_session"] == execution_session
    )
    binding = ledger.get_replay_binding(day["execution_id"])
    assert binding is not None
    return binding


# ---------------------------------------------------------------------------
# 常量与门禁
# ---------------------------------------------------------------------------


def test_production_overlay_identity_constant_recomputes() -> None:
    anchor = TRUSTED_OVERLAY_ANCHOR
    identity = RQAlphaOverlayIdentity(
        contract_format=anchor["format"],
        bundle_name=anchor["bundle_name"],
        archive_sha256=anchor["archive_sha256"],
        member_inventory_sha256=anchor["member_inventory_sha256"],
        member_sha256=tuple(sorted(anchor["member_sha256"].items())),
        session_first=anchor["session_first"],
        session_last_inclusive=anchor["session_last_inclusive"],
        allowed_fields=STOCKS_ALLOWED_FIELDS,
        permitted_semantics=PERMITTED_SEMANTICS,
        forbidden_semantics=FORBIDDEN_SEMANTICS,
        suspended_missing_dataset_semantics=(
            SUSPENDED_MISSING_DATASET_SEMANTICS
        ),
        st_dataset_semantics=ST_DATASET_SEMANTICS,
        prev_close_excluded=tuple(anchor["prev_close_excluded"]),
        unresolved_gaps=tuple(anchor["unresolved_gaps"]),
        derivation_rule_version=DERIVATION_RULE_VERSION,
        derivation_rule_description=DERIVATION_RULE_DESCRIPTION,
        available_code_count=anchor["available_code_count"],
        v3_manifest_sha256=anchor["v3_manifest_sha256"],
        v3_coverage_matrix_sha256=anchor["v3_coverage_matrix_sha256"],
        frozen_calendar_sha256=anchor["frozen_calendar_sha256"],
        calendar_intersection_rows=anchor["calendar_intersection_rows"],
        lot_size_source=(
            "instruments.pk round_lot（静态安全解码，5,553/5,553 CS 记录）"
        ),
        lot_size_allowed=tuple(anchor["lot_size_allowed"]),
    )
    assert identity.identity_sha256 == PRODUCTION_OVERLAY_IDENTITY_SHA256


def test_engine_refuses_non_production_overlay_by_default(
    tmp_path, monkeypatch
) -> None:
    """裁决 04C 侧义务：不显式注入期望身份时，非生产 overlay 必须被拒。"""
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        with pytest.raises(DynamicReplayError, match="固化常量不一致"):
            DynamicDailyReplay(
                _main_config(),
                overlay=overlay,
                ledger=PortfolioDecisionLedger(tmp_path / "ledger.sqlite3"),
                fee_schedule=FEES,
                history_root=root,
                trust_policy=policy,
                v3_export_root=world["v3_root"],
                calendar_path=world["calendar_path"],
                expected_v3_manifest_sha256=(
                    world["anchor"]["v3_manifest_sha256"]
                ),
                expected_v3_coverage_sha256=(
                    world["anchor"]["v3_coverage_matrix_sha256"]
                ),
                expected_calendar_sha256=(
                    world["anchor"]["frozen_calendar_sha256"]
                ),
            )
    finally:
        overlay.close()


def test_replay_domain_gates() -> None:
    with pytest.raises(DynamicReplayError, match="空窗"):
        _main_config(start_date="2010-01-15", end_date="2010-02-27").validated()
    with pytest.raises(DynamicReplayError, match="上限"):
        _main_config(start_date="2026-06-15", end_date="2026-07-01").validated()
    with pytest.raises(DynamicReplayError, match="不得晚于"):
        _main_config(start_date="2026-06-20", end_date="2026-06-15").validated()


def test_universe_2009_window_fail_closed_synthetic(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "universe2009"
    _publish_history(
        root,
        [
            {
                "requested_date": "2009-12-31",
                "effective_date": "2009-12-31",
                "symbols": ("600001", "600002", "600003"),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root, policy="test_2009_roots")
    receipt = json.loads(
        (root / "manifests" / "20091231.json").read_text(encoding="utf-8")
    )
    policies = deepcopy(universe_module._CSI300_HISTORY_AVAILABILITY_POLICIES)
    policies[policy] = {
        "incomplete_snapshots": {
            "2009-12-31": {
                "source_data_sha256": receipt["data_sha256"],
                "source_receipt_sha256": _sha256_file(
                    root / "manifests" / "20091231.json"
                ),
                "constituent_count": 3,
                "valid_until_exclusive": "2010-01-29",
                "reason": "受信快照不完整，不能作为沪深300股票池",
            }
        }
    }
    monkeypatch.setattr(
        universe_module, "_CSI300_HISTORY_AVAILABILITY_POLICIES", policies
    )
    with pytest.raises(UniverseAvailabilityError, match="不完整"):
        load_csi300_historical_universe_contract(
            root,
            as_of_date="2010-01-10",
            mode=REPLAY_UNIVERSE_QUERY_MODE,
            trust_policy=policy,
        )
    with pytest.raises(UniverseAvailabilityError, match="替代快照"):
        load_csi300_historical_universe_contract(
            root,
            as_of_date="2010-02-05",
            mode=REPLAY_UNIVERSE_QUERY_MODE,
            trust_policy=policy,
        )


# ---------------------------------------------------------------------------
# 主流程：进入/退出、触限拒单、停牌估值、ST、10%/20%、0 限价、T+1
# ---------------------------------------------------------------------------


def test_main_flow_events_and_identity_chain(tmp_path, monkeypatch) -> None:
    import pandas as pd

    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        engine = _make_engine(
            world, overlay, ledger, _main_config(), root, policy
        )
        manifest = engine.run()

        assert manifest["status"] == "COMPLETED"
        assert manifest["summary"]["pair_count"] == 9
        assert manifest["declarations"]["not_sealed_oos"] is True
        assert "机制验证信号" in manifest["declarations"]["engineering_signal"]
        assert (
            manifest["declarations"]["locked_rule"] == DERIVATION_RULE_VERSION
        )

        # 触及显式涨停价：保守日线触及规则拒单，次日恢复成交。
        rejected = _order(manifest, "2026-06-16", "600001", "BUY")
        assert rejected["status"] == "REJECTED"
        assert rejected["reason"] == "LIMIT_UP_LOCKED"
        assert _order(manifest, "2026-06-16", "600002", "BUY")["status"] == (
            "FILLED"
        )
        assert _order(manifest, "2026-06-16", "600003", "BUY")["status"] == (
            "FILLED"
        )
        assert _order(manifest, "2026-06-17", "600001", "BUY")["status"] == (
            "FILLED"
        )

        # 停牌期间任何 600002 订单只能被拒（唯一来源 suspended_days）。
        for session in ("2026-06-17", "2026-06-18"):
            for order in _orders_of(manifest, session):
                if order["symbol"] == "600002":
                    assert order["status"] == "REJECTED"
                    assert order["reason"] == "SUSPENDED"

        # 调样过渡：600003 退出 → 强制离场；688001 补入 Top-3。
        transition = next(
            record for record in manifest["days"]
            if record["decision_session"] == "2026-06-23"
        )
        assert transition["transition_mode"] is True
        assert transition["forced_exit_symbols"] == ["600003"]
        assert transition["universe_as_of"] == "2026-06-22"
        assert _order(manifest, "2026-06-24", "600003", "SELL")["status"] == (
            "FILLED"
        )
        star_buy = _order(manifest, "2026-06-24", "688001", "BUY")
        assert star_buy["status"] == "FILLED"
        assert star_buy["filled_shares"] > 0
        assert star_buy["filled_shares"] % 200 == 0  # 科创板整手 200

        # 成分进入：600004 于新快照生效后被买入；688001 按 dropout 离场，
        # T+1 语义下于买入次日卖出。
        assert _order(manifest, "2026-06-25", "600004", "BUY")["status"] == (
            "FILLED"
        )
        assert _order(manifest, "2026-06-25", "688001", "SELL")["status"] == (
            "FILLED"
        )
        normal_days = [
            record for record in manifest["days"]
            if record["decision_session"] != "2026-06-23"
        ]
        assert all(not record["transition_mode"] for record in normal_days)

        # 执行行情价格身份 = v3 qfq 冻结价格（不是 RQAlpha 未复权价）。
        v3_600001 = pd.read_parquet(
            world["v3_root"] / "D1" / "600001_D1.parquet"
        )
        v3_close_0617 = float(v3_600001["close"].iloc[2])
        record_0617 = ledger.get_execution(
            next(
                day for day in manifest["days"]
                if day["execution_session"] == "2026-06-17"
            )["execution_id"]
        )
        quote_0617 = next(
            quote for quote in record_0617["input"]["execution_quotes"]
            if quote["symbol"] == "600001"
        )
        assert quote_0617["price"] == v3_close_0617
        assert quote_0617["price"] != 10.5  # RQAlpha 未复权 close

        # 决策日估值：停牌用最近 v3 收盘价 carry-forward，基准日如实记录；
        # 决策估值行情的 status/lot_size 是占位常量。
        binding_0618 = _binding_for(ledger, manifest, "2026-06-18")
        valuation = binding_0618["decision_quote_provenance"]["600002"]
        assert valuation["price_basis_session"] == "2026-06-16"
        assert valuation["status_is_placeholder"] is True
        decision_quote = next(
            quote for quote in ledger.get_execution(
                binding_0618["execution_id"]
            )["input"]["decision_quotes"]
            if quote["symbol"] == "600002"
        )
        assert decision_quote["status"] == DECISION_VALUATION_STATUS_PLACEHOLDER
        assert decision_quote["lot_size"] == (
            DECISION_VALUATION_LOT_SIZE_PLACEHOLDER
        )

        # ST 进入/退出：06-18 进入、06-19 保持、06-22 退出。
        def _state(session: str, symbol: str) -> dict:
            binding = _binding_for(ledger, manifest, session)
            return binding["execution_quote_provenance"][symbol][
                "execution_state"
            ]

        assert _state("2026-06-17", "600001")["is_st"] is False
        assert _state("2026-06-18", "600001")["is_st"] is True
        assert _state("2026-06-19", "600001")["is_st"] is True
        assert _state("2026-06-22", "600001")["is_st"] is False

        # 10% -> 20% 涨跌停切换：显式价格边界如实进入绑定。
        state_10 = _state("2026-06-19", "600001")
        state_20 = _state("2026-06-22", "600001")
        assert state_10["limit_up"] == pytest.approx(11.66)
        assert (
            state_10["limit_up"] / state_10["prev_close"] - 1.0
        ) == pytest.approx(0.10, abs=1e-3)
        assert state_20["limit_up"] == pytest.approx(12.84)
        assert (
            state_20["limit_up"] / state_20["prev_close"] - 1.0
        ) == pytest.approx(0.20, abs=1e-3)

        # 上市初期式 0 限价：显式 0 值不判 locked，整手 200 进入绑定。
        star_state = _state("2026-06-24", "688001")
        assert star_state["status"] == "OPEN"
        assert star_state["limit_up"] == 0.0
        assert star_state["limit_down"] == 0.0
        assert star_state["lot_size"] == 200

        # 全链身份核验：绑定的 overlay/universe/价格身份可重算一致。
        verification = engine.verify()
        assert verification["execution_count"] == 9
        binding_0616 = _binding_for(ledger, manifest, "2026-06-16")
        assert binding_0616["overlay_identity_sha256"] == (
            overlay.identity.identity_sha256
        )
        assert binding_0616["universe"]["query_mode"] == (
            REPLAY_UNIVERSE_QUERY_MODE
        )
        assert binding_0616["engineering_signal"]["rule_version"] == (
            "deterministic_code_sorted_topn_v1"
        )
    finally:
        overlay.close()


def test_suspension_buy_rejection_and_resumption_boundary(
    tmp_path, monkeypatch
) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2026-06-12",
                "effective_date": "2026-06-12",
                "symbols": ("600002", "600003"),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        config = _main_config(
            start_date="2026-06-16",
            end_date="2026-06-19",
            top_k=1,
            dropout_rank=1,
            run_label="synthetic-suspension",
        )
        engine = _make_engine(world, overlay, ledger, config, root, policy)
        manifest = engine.run()
        assert _order(manifest, "2026-06-17", "600002", "BUY")["reason"] == (
            "SUSPENDED"
        )
        assert _order(manifest, "2026-06-18", "600002", "BUY")["reason"] == (
            "SUSPENDED"
        )
        # 复牌边界：不再是停牌拒单，恢复真实成交（top_k=1 满仓目标下，
        # 费用会使最后一手买不起，PARTIAL/INSUFFICIENT_CASH 是执行器
        # 正常语义，不是停牌残留）。
        resumed = _order(manifest, "2026-06-19", "600002", "BUY")
        assert resumed["status"] in ("FILLED", "PARTIAL")
        assert resumed["reason"] != "SUSPENDED"
        assert resumed["filled_shares"] > 0
        binding = _binding_for(ledger, manifest, "2026-06-18")
        provenance = binding["execution_quote_provenance"]["600002"]
        assert provenance["execution_state"]["status"] == "SUSPENDED"
        assert provenance["execution_state"]["close"] is None
        assert provenance["price_basis_session"] == "2026-06-16"
        assert engine.verify()["execution_count"] == 3
    finally:
        overlay.close()


def test_missing_v3_day_fail_closed(tmp_path, monkeypatch) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2026-06-12",
                "effective_date": "2026-06-12",
                "symbols": ("600005",),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        config = _main_config(
            start_date="2026-06-17",
            end_date="2026-06-18",
            top_k=1,
            dropout_rank=1,
            run_label="synthetic-missing-day",
        )
        engine = _make_engine(world, overlay, ledger, config, root, policy)
        with pytest.raises(DynamicReplayError, match="漏数日"):
            engine.run()
        with sqlite3.connect(ledger.path) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM portfolio_executions"
            ).fetchone()[0] == 0
    finally:
        overlay.close()


def test_quarantine_constituent_fails_signals_gate(
    tmp_path, monkeypatch
) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2026-06-12",
                "effective_date": "2026-06-12",
                "symbols": ("600001", "600007"),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        config = _main_config(
            start_date="2026-06-15",
            end_date="2026-06-16",
            top_k=1,
            dropout_rank=1,
            run_label="synthetic-quarantine",
        )
        engine = _make_engine(world, overlay, ledger, config, root, policy)
        with pytest.raises(
            DynamicReplayError,
            match="覆盖缺口.*不静默缩池",
        ) as excinfo:
            engine.run()
        assert "600007" in str(excinfo.value)
        assert "quarantine" in str(excinfo.value)
        with sqlite3.connect(ledger.path) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM portfolio_decisions"
            ).fetchone()[0] == 0
    finally:
        overlay.close()


# ---------------------------------------------------------------------------
# 崩溃恢复幂等 / 账本篡改 / 全链身份伪造 / 未来状态追加
# ---------------------------------------------------------------------------


def test_crash_recovery_is_idempotent(tmp_path, monkeypatch) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        reference_ledger = PortfolioDecisionLedger(tmp_path / "ref.sqlite3")
        reference = _make_engine(
            world, overlay, reference_ledger, _main_config(), root, policy
        ).run()

        crash_ledger = PortfolioDecisionLedger(tmp_path / "crash.sqlite3")
        engine = _make_engine(
            world, overlay, crash_ledger, _main_config(), root, policy
        )
        original_binding = crash_ledger.record_replay_binding
        state = {"count": 0}

        def failing_binding(payload):
            if state["count"] >= 2:
                raise RuntimeError("模拟崩溃：绑定写入前断电")
            state["count"] += 1
            return original_binding(payload)

        crash_ledger.record_replay_binding = failing_binding
        with pytest.raises(RuntimeError, match="模拟崩溃"):
            engine.run()
        del crash_ledger.record_replay_binding

        resumed = engine.run()
        assert [d["decision_id"] for d in resumed["days"]] == [
            d["decision_id"] for d in reference["days"]
        ]
        assert [d["execution_id"] for d in resumed["days"]] == [
            d["execution_id"] for d in reference["days"]
        ]
        assert [d["execution_session"] for d in resumed["days"]] == [
            _iso(day) for day in SESSIONS[1:]
        ]
        with sqlite3.connect(crash_ledger.path) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM portfolio_executions"
            ).fetchone()[0] == 9
            assert conn.execute(
                "SELECT COUNT(*) FROM portfolio_replay_bindings"
            ).fetchone()[0] == 9
        assert engine.verify()["execution_count"] == 9
    finally:
        overlay.close()


def test_ledger_tamper_fails_closed(tmp_path, monkeypatch) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        engine = _make_engine(
            world, overlay, ledger, _main_config(), root, policy
        )
        manifest = engine.run()
        execution_id = manifest["days"][0]["execution_id"]

        with sqlite3.connect(ledger.path) as conn:
            conn.execute(
                """
                UPDATE portfolio_replay_bindings
                SET payload_json = payload_json || ' '
                WHERE execution_id = ?
                """,
                (execution_id,),
            )
        with pytest.raises(RuntimeError, match="哈希不一致"):
            ledger.get_replay_binding(execution_id)
        with pytest.raises(RuntimeError, match="哈希不一致"):
            engine.verify()

        # 执行行审计时间戳篡改：既有 12 入口失败关闭在 replay 读取路径同样生效。
        fresh_ledger = PortfolioDecisionLedger(tmp_path / "ledger2.sqlite3")
        engine2 = _make_engine(
            world, overlay, fresh_ledger, _main_config(), root, policy
        )
        engine2.run()
        with sqlite3.connect(fresh_ledger.path) as conn:
            conn.execute("UPDATE portfolio_executions SET created_at = 0")
        with pytest.raises(RuntimeError, match="审计身份"):
            engine2.verify()
    finally:
        overlay.close()


def test_binding_semantic_forgery_detected(tmp_path, monkeypatch) -> None:
    """裁决第 17 组全链版：篡改 overlay 状态或来源声明 → 身份链检出。"""
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        engine = _make_engine(
            world, overlay, ledger, _main_config(), root, policy
        )
        manifest = engine.run()
        execution_id = next(
            day for day in manifest["days"]
            if day["execution_session"] == "2026-06-18"
        )["execution_id"]

        def _rewrite(mutator) -> None:
            with sqlite3.connect(ledger.path) as conn:
                raw = conn.execute(
                    """
                    SELECT payload_json FROM portfolio_replay_bindings
                    WHERE execution_id = ?
                    """,
                    (execution_id,),
                ).fetchone()[0]
                payload = json.loads(raw)
                mutator(payload)
                forged = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                conn.execute(
                    """
                    UPDATE portfolio_replay_bindings
                    SET payload_json = ?, payload_sha256 = ?
                    WHERE execution_id = ?
                    """,
                    (
                        forged,
                        hashlib.sha256(forged.encode("utf-8")).hexdigest(),
                        execution_id,
                    ),
                )

        original = ledger.get_replay_binding(execution_id)

        # 伪造 overlay 执行状态（把 ST 标记翻转）：ledger 层通过，
        # replay 语义重算必须检出。
        def _flip_state(payload):
            payload["execution_quote_provenance"]["600001"][
                "execution_state"
            ]["is_st"] = False

        _rewrite(_flip_state)
        assert ledger.get_replay_binding(execution_id) is not None
        with pytest.raises(DynamicReplayError, match="状态篡改检出"):
            engine.verify()

        # 恢复后伪造来源声明（overlay identity）：同样必须检出。
        _rewrite(
            lambda payload: payload.update(
                original
            )
        )
        engine.verify()

        _rewrite(
            lambda payload: payload.update(
                {"overlay_identity_sha256": "0" * 64}
            )
        )
        with pytest.raises(DynamicReplayError, match="来源声明篡改检出"):
            engine.verify()
    finally:
        overlay.close()


def test_binding_copy_field_forgery_detected(tmp_path, monkeypatch) -> None:
    """P1-2（2026-07-27 对抗审查修复）：绑定副本字段的自洽篡改必须检出。

    针对"改绑定字段 + 重算 payload_sha256"的伪造路径，逐字段验证
    verify() 的权威源重算比对。已知无害残留：非过渡记录把
    transition_mode 翻成 true 且 forced 为空时语义上是无效谎言
    （股票池相同、无强制离场），不在检出范围。
    """
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        engine = _make_engine(
            world, overlay, ledger, _main_config(), root, policy
        )
        manifest = engine.run()

        def _rewrite(execution_id, mutator) -> None:
            with sqlite3.connect(ledger.path) as conn:
                raw = conn.execute(
                    """
                    SELECT payload_json FROM portfolio_replay_bindings
                    WHERE execution_id = ?
                    """,
                    (execution_id,),
                ).fetchone()[0]
                payload = json.loads(raw)
                mutator(payload)
                forged = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                conn.execute(
                    """
                    UPDATE portfolio_replay_bindings
                    SET payload_json = ?, payload_sha256 = ?
                    WHERE execution_id = ?
                    """,
                    (
                        forged,
                        hashlib.sha256(forged.encode("utf-8")).hexdigest(),
                        execution_id,
                    ),
                )

        all_ids = [day["execution_id"] for day in manifest["days"]]
        transition_ids = [
            eid
            for eid in all_ids
            if ledger.get_replay_binding(eid)["universe"]["transition_mode"]
        ]
        assert transition_ids, "MAIN 场景必须含调样过渡日（600003 离场）"
        plain_id = next(eid for eid in all_ids if eid not in transition_ids)
        originals = {
            eid: ledger.get_replay_binding(eid)
            for eid in (plain_id, transition_ids[0])
        }

        def _restore(eid) -> None:
            _rewrite(eid, lambda payload: payload.update(originals[eid]))
            engine.verify()

        cases = [
            (
                plain_id,
                lambda p: p.update(
                    {"replay_contract_version": "forged_contract_v9"}
                ),
                "replay_contract_version 篡改检出",
            ),
            (
                plain_id,
                lambda p: p["fee_schedule"].update(
                    {"commission_rate": "0.9"}
                ),
                "fee_schedule 篡改检出",
            ),
            (
                plain_id,
                lambda p: p["engineering_signal"].update(
                    {"rule_version": "forged_rule_v9"}
                ),
                "engineering_signal 篡改检出",
            ),
            (
                plain_id,
                lambda p: p["universe"].update(
                    {"universe_sha256": "f" * 64}
                ),
                "universe.universe_sha256 与决策账本记录不一致",
            ),
            (
                plain_id,
                lambda p: p["universe"].update(
                    {"effective_universe_sha256": "e" * 64}
                ),
                "effective_universe_sha256 与按决策日重算不一致",
            ),
            (
                plain_id,
                lambda p: p["universe"].update(
                    {"forced_exit_symbols": ["600001"]}
                ),
                "forced_exit_symbols 与重算不一致",
            ),
            (
                transition_ids[0],
                lambda p: p["universe"].update({"transition_mode": False}),
                "transition_mode 与重算不一致",
            ),
            (
                transition_ids[0],
                lambda p: p["universe"].update(
                    {"forced_exit_symbols": []}
                ),
                "forced_exit_symbols 与重算不一致",
            ),
        ]
        for eid, mutator, match in cases:
            _rewrite(eid, mutator)
            with pytest.raises(DynamicReplayError, match=match):
                engine.verify()
            _restore(eid)
    finally:
        overlay.close()


def test_replay_binding_rejects_unknown_execution(
    tmp_path, monkeypatch
) -> None:
    """P2-②（2026-07-27 审查建议）：绑定引用不存在的执行必须被拒绝。"""
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        engine = _make_engine(
            world, overlay, ledger, _main_config(), root, policy
        )
        manifest = engine.run()
        payload = dict(
            ledger.get_replay_binding(manifest["days"][0]["execution_id"])
        )
        payload["execution_id"] = "EXEC-DOES-NOT-EXIST"
        with pytest.raises(RuntimeError):
            ledger.record_replay_binding(payload)
    finally:
        overlay.close()


def test_future_state_append_keeps_history_replay(
    tmp_path, monkeypatch
) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        baseline_ledger = PortfolioDecisionLedger(tmp_path / "base.sqlite3")
        baseline = _make_engine(
            world, overlay, baseline_ledger, _main_config(), root, policy
        ).run()
    finally:
        overlay.close()

    # 未来状态追加：同一路径追加 RQAlpha bundle 未来 bar/停牌/交易日，
    # 同一 universe 根追加未来月份快照；v3 与冻结日历保持字节不变。
    appended_world = _build_world(tmp_path, future_append=True)
    _publish_history(root, MAIN_MONTHS + [FUTURE_MONTH])
    policy = _register_test_root(monkeypatch, root)
    appended_overlay = _load_world_overlay(appended_world)
    try:
        appended_ledger = PortfolioDecisionLedger(tmp_path / "appended.sqlite3")
        appended = _make_engine(
            appended_world,
            appended_overlay,
            appended_ledger,
            _main_config(),
            root,
            policy,
        ).run()
        assert [d["decision_id"] for d in appended["days"]] == [
            d["decision_id"] for d in baseline["days"]
        ]
        assert [d["execution_id"] for d in appended["days"]] == [
            d["execution_id"] for d in baseline["days"]
        ]
        # overlay 身份必然随成员字节变化；决策/执行身份逐字节不变才是
        # "未来状态追加不改变历史 replay" 的本体。
        assert (
            appended_overlay.identity.identity_sha256
            != baseline["identities"]["overlay_identity_sha256"]
        )
    finally:
        appended_overlay.close()


def test_ledger_bound_to_single_replay_run(tmp_path, monkeypatch) -> None:
    world = _build_world(tmp_path)
    root = tmp_path / "universe"
    _publish_history(root, MAIN_MONTHS)
    policy = _register_test_root(monkeypatch, root)
    overlay = _load_world_overlay(world)
    try:
        ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
        _make_engine(
            world, overlay, ledger, _main_config(), root, policy
        ).run()
        other = _make_engine(
            world,
            overlay,
            ledger,
            _main_config(run_label="another-run"),
            root,
            policy,
        )
        with pytest.raises(DynamicReplayError, match="另一 replay run"):
            other.run()
    finally:
        overlay.close()


# ---------------------------------------------------------------------------
# 真实 G 盘用例（真实调样日 / 真实停牌日拒单 / 真实触及限价拒单 / 漏数日 /
# 真实时点 quarantine / 真实 2009 空窗）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_overlay():
    if not _G_DRIVE_READY:
        pytest.skip("G 盘数据不可读")
    overlay = load_rqalpha_execution_overlay()
    yield overlay
    overlay.close()


def _real_engine(ledger, config, universe_root, policy, real_overlay):
    return DynamicDailyReplay(
        config,
        overlay=real_overlay,
        ledger=ledger,
        fee_schedule=FEES,
        history_root=universe_root,
        trust_policy=policy,
    )


@requires_real_data
def test_real_suspension_rejection_and_carry_forward(
    tmp_path, monkeypatch, real_overlay
) -> None:
    """真实停牌日拒单：002049 于 2026-01-06 显式停牌（suspended_days）。"""
    assert real_overlay.identity.identity_sha256 == (
        PRODUCTION_OVERLAY_IDENTITY_SHA256
    )
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2025-11-28",
                "effective_date": "2025-11-28",
                "symbols": ("002049", "600030"),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root, policy="test_real_susp")
    ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
    config = DynamicReplayConfig(
        start_date="2026-01-05",
        end_date="2026-01-06",
        top_k=1,
        dropout_rank=1,
        initial_cash=1_000_000.0,
        run_label="real-suspension",
    )
    engine = _real_engine(ledger, config, root, policy, real_overlay)
    manifest = engine.run()
    order = _order(manifest, "2026-01-06", "002049", "BUY")
    assert order["status"] == "REJECTED"
    assert order["reason"] == "SUSPENDED"
    binding = _binding_for(ledger, manifest, "2026-01-06")
    provenance = binding["execution_quote_provenance"]["002049"]
    assert provenance["execution_state"]["status"] == "SUSPENDED"
    # 停牌期估值 carry-forward：基准日为停牌前最后一个 v3 交易日。
    assert provenance["price_basis_session"] == "2025-12-29"
    assert engine.verify()["execution_count"] == 1


@requires_real_data
def test_real_limit_touch_rejection(
    tmp_path, monkeypatch, real_overlay
) -> None:
    """真实触及限价拒单：002050 于 2025-12-30 收盘触及显式涨停价。"""
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2025-11-28",
                "effective_date": "2025-11-28",
                "symbols": ("002050", "600030"),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root, policy="test_real_touch")
    ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
    config = DynamicReplayConfig(
        start_date="2025-12-29",
        end_date="2025-12-30",
        top_k=1,
        dropout_rank=1,
        initial_cash=1_000_000.0,
        run_label="real-limit-touch",
    )
    engine = _real_engine(ledger, config, root, policy, real_overlay)
    manifest = engine.run()
    order = _order(manifest, "2025-12-30", "002050", "BUY")
    assert order["status"] == "REJECTED"
    assert order["reason"] == "LIMIT_UP_LOCKED"
    binding = _binding_for(ledger, manifest, "2025-12-30")
    state = binding["execution_quote_provenance"]["002050"][
        "execution_state"
    ]
    assert state["status"] == "LIMIT_UP_LOCKED"
    assert state["close"] == pytest.approx(52.5)
    assert state["close"] == pytest.approx(state["limit_up"])
    assert engine.verify()["execution_count"] == 1


@requires_real_data
def test_real_reconstitution_entry_exit(
    tmp_path, monkeypatch, real_overlay
) -> None:
    """真实历史调样日：000800 于 2025-12-31 快照真实调出沪深300，
    002384 真实调入（fixture 只是把真实快照限制到这三只标的）。"""
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2025-11-28",
                "effective_date": "2025-11-28",
                "symbols": ("000800", "600030"),
            },
            {
                "requested_date": "2025-12-31",
                "effective_date": "2025-12-31",
                "symbols": ("002384", "600030"),
            },
        ],
    )
    policy = _register_test_root(monkeypatch, root, policy="test_real_rebal")
    ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
    config = DynamicReplayConfig(
        start_date="2025-12-30",
        end_date="2026-01-06",
        top_k=1,
        dropout_rank=1,
        initial_cash=1_000_000.0,
        run_label="real-reconstitution",
    )
    engine = _real_engine(ledger, config, root, policy, real_overlay)
    manifest = engine.run()

    entry_buy = _order(manifest, "2025-12-31", "000800", "BUY")
    assert entry_buy["status"] in ("FILLED", "PARTIAL")
    assert entry_buy["filled_shares"] > 0
    transition = next(
        record for record in manifest["days"]
        if record["decision_session"] == "2025-12-31"
    )
    assert transition["transition_mode"] is True
    assert transition["forced_exit_symbols"] == ["000800"]
    assert _order(manifest, "2026-01-05", "000800", "SELL")["status"] == (
        "FILLED"
    )
    new_member_buy = _order(manifest, "2026-01-06", "002384", "BUY")
    assert new_member_buy["status"] in ("FILLED", "PARTIAL")
    assert new_member_buy["filled_shares"] > 0
    assert engine.verify()["execution_count"] == 3


@requires_real_data
def test_real_freestockdb_missing_day_fail_closed(
    tmp_path, monkeypatch, real_overlay
) -> None:
    """真实漏数日（裁决第 18 组）：000338 在 2018-07-02 为当时真实成分，
    RQAlpha 有正常 bar 且非停牌，v3 冻结价格缺 bar → 价格层失败关闭。"""
    root = tmp_path / "universe"
    _publish_history(
        root,
        [
            {
                "requested_date": "2018-06-29",
                "effective_date": "2018-06-29",
                "symbols": ("000338",),
            }
        ],
    )
    policy = _register_test_root(monkeypatch, root, policy="test_real_missing")
    ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
    config = DynamicReplayConfig(
        start_date="2018-06-29",
        end_date="2018-07-02",
        top_k=1,
        dropout_rank=1,
        initial_cash=1_000_000.0,
        run_label="real-missing-day",
    )
    engine = _real_engine(ledger, config, root, policy, real_overlay)
    with pytest.raises(DynamicReplayError, match="漏数日"):
        engine.run()
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM portfolio_executions"
        ).fetchone()[0] == 0


@requires_real_data
def test_real_csi300_timepoint_quarantine_fail_closed(
    tmp_path, real_overlay
) -> None:
    """真实沪深300时点含 v3 quarantine 成分（如 000001）：硬条件 A，
    整个时点失败关闭，不静默缩池。"""
    ledger = PortfolioDecisionLedger(tmp_path / "ledger.sqlite3")
    config = DynamicReplayConfig(
        start_date="2025-12-29",
        end_date="2025-12-30",
        top_k=10,
        dropout_rank=10,
        initial_cash=1_000_000.0,
        run_label="real-csi300-failclosed",
    )
    engine = _real_engine(
        ledger,
        config,
        PRODUCTION_CSI300_HISTORY_ROOT,
        PRODUCTION_CSI300_TRUST_POLICY,
        real_overlay,
    )
    with pytest.raises(
        DynamicReplayError, match="覆盖缺口.*不静默缩池"
    ) as excinfo:
        engine.run()
    assert "quarantine" in str(excinfo.value)
    assert "000001" in str(excinfo.value)


@requires_real_data
def test_real_2009_window_universe_fail_closed() -> None:
    with pytest.raises(UniverseAvailabilityError, match="不完整"):
        load_csi300_historical_universe_contract(
            PRODUCTION_CSI300_HISTORY_ROOT,
            as_of_date="2010-01-15",
            mode=REPLAY_UNIVERSE_QUERY_MODE,
            trust_policy=PRODUCTION_CSI300_TRUST_POLICY,
        )
