# -*- coding: utf-8 -*-
"""RQAlpha execution-state overlay 的失败关闭测试。

覆盖 docs/evidence/rnd04a_execution_overlay_adjudication_20260726.md
第五节 19 组测试清单中属于适配器层的第 1-18 组；第 19 组（决策
universe 覆盖缺口 -> controller signals 完整性门）属 RND-04C 集成测试。

真实 G 盘 bundle 可读时执行审计样本断言；无 G 盘环境使用 tmp 合成
HDF5 fixture（身份校验用重算哈希的注入锚）。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
import re
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.rqalpha_execution_overlay import (
    DERIVATION_RULE_VERSION,
    EXECUTION_STATE_STATUSES,
    FROZEN_CALENDAR_PATH,
    RQALPHA_EXTRACTED_ROOT,
    RQAlphaOverlayError,
    STOCKS_ALLOWED_FIELDS,
    TRUSTED_OVERLAY_ANCHOR,
    V3_EXPORT_ROOT,
    load_rqalpha_execution_overlay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UNSET = object()  # 区分"未设置"与显式传入 None 的哨兵

_G_DRIVE_READY = (
    RQALPHA_EXTRACTED_ROOT.is_dir()
    and V3_EXPORT_ROOT.is_dir()
    and FROZEN_CALENDAR_PATH.is_file()
)
requires_real_bundle = pytest.mark.skipif(
    not _G_DRIVE_READY,
    reason="G 盘 RQAlpha bundle / v3 导出 / 冻结日历不可读",
)

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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bar(
    day: int,
    *,
    close: float,
    prev_close: float,
    limit_up: float,
    limit_down: float,
    volume: float = 1_000_000.0,
) -> tuple:
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


def _build_fixture(
    tmp_path: Path,
    *,
    st_ascending: bool = False,
    st_missing_dataset: bool = False,
    malicious_pickle: bool = False,
    future_append: bool = False,
    bad_round_lot: object = _UNSET,
    calendar_drop_day: bool = False,
    prev_close_nan: bool = False,
    gap_no_bar: bool = False,
    tamper_stocks_after_anchor: bool = False,
    anchor_overrides: dict | None = None,
) -> dict:
    """构造带重算哈希注入锚的合成 bundle / v3 导出 / 冻结日历。"""
    import h5py
    import pandas as pd

    root = tmp_path / "bundle"
    v3_root = tmp_path / "v3"
    root.mkdir()
    v3_root.mkdir()

    sessions = [20260622, 20260623, 20260624, 20260625, 20260626, 20260629, 20260630]

    # --- stocks.h5 ---
    bars_600001 = [
        _bar(20260622, close=10.5, prev_close=10.0, limit_up=11.0, limit_down=9.0),
        # 触及显式涨停价（保守日线触及规则）
        _bar(20260623, close=11.55, prev_close=10.5, limit_up=11.55, limit_down=9.45),
        # 触及显式跌停价
        _bar(20260624, close=10.4, prev_close=11.55, limit_up=12.71, limit_down=10.4),
        # 20260625 停牌且无 bar；20260626 停牌但保留填充 bar
        _bar(
            20260626,
            close=10.4,
            prev_close=10.4,
            limit_up=0.0,
            limit_down=0.0,
            volume=0.0,
        ),
        # 非停牌但 volume==0：禁止由成交量推断停牌
        _bar(
            20260629,
            close=10.6,
            prev_close=10.4,
            limit_up=11.44,
            limit_down=9.36,
            volume=0.0,
        ),
        _bar(20260630, close=10.8, prev_close=10.6, limit_up=11.66, limit_down=9.54),
    ]
    if future_append:
        bars_600001.append(
            _bar(20260701, close=11.0, prev_close=10.8, limit_up=11.88, limit_down=9.72)
        )
    bars_600002 = []
    for day in sessions:
        if gap_no_bar and day == 20260624:
            continue  # 非停牌覆盖缺口：必须失败关闭
        prev = float("nan") if (prev_close_nan and day == 20260622) else 20.0
        bars_600002.append(
            _bar(day, close=20.5, prev_close=prev, limit_up=22.0, limit_down=18.0)
        )
    bars_688001 = [
        # 科创板上市初期：显式 0 限价，禁止判 locked
        _bar(day, close=30.5, prev_close=30.0, limit_up=0.0, limit_down=0.0)
        for day in sessions
    ]
    bars_600005 = [
        _bar(day, close=5.5, prev_close=5.4, limit_up=5.94, limit_down=4.86)
        for day in (20260622, 20260623, 20260624)
    ]
    with h5py.File(root / "stocks.h5", "w") as handle:
        for key, rows in (
            ("600001.XSHG", bars_600001),
            ("600002.XSHG", bars_600002),
            ("688001.XSHG", bars_688001),
            ("600005.XSHG", bars_600005),
        ):
            handle.create_dataset(key, data=np.array(rows, dtype=_STOCK_DTYPE))

    # --- suspended_days.h5（600002 故意无 dataset：无停牌记录语义）---
    suspended_600001 = [20260625, 20260626]
    if future_append:
        suspended_600001.append(20260701)
    with h5py.File(root / "suspended_days.h5", "w") as handle:
        handle.create_dataset(
            "600001.XSHG", data=np.array(suspended_600001, dtype="int64")
        )

    # --- st_stock_days.h5（真实存储为严格降序）---
    st_600001 = [20260623, 20260624] if st_ascending else [20260624, 20260623]
    with h5py.File(root / "st_stock_days.h5", "w") as handle:
        handle.create_dataset("600001.XSHG", data=np.array(st_600001, dtype="int64"))
        if not st_missing_dataset:
            handle.create_dataset("600002.XSHG", data=np.array([], dtype="float64"))
        handle.create_dataset("688001.XSHG", data=np.array([], dtype="float64"))
        handle.create_dataset("600005.XSHG", data=np.array([], dtype="float64"))

    # --- instruments.pk（纯 dict/str/int，安全解码可读）---
    def _cs(code: str, obid: str, lot: object, listed: str = "1999-11-10",
            de_listed: str = "0000-00-00") -> dict:
        return {
            "type": "CS",
            "order_book_id": obid,
            "trading_code": code,
            "listed_date": listed,
            "de_listed_date": de_listed,
            "round_lot": lot,
        }

    instruments = [
        {"type": "INDX", "order_book_id": "000300.XSHG"},
        _cs("600001", "600001.XSHG", 100),
        _cs("600002", "600002.XSHG", 100),
        _cs("688001", "688001.XSHG", 200 if bad_round_lot is _UNSET else bad_round_lot,
            listed="2026-06-20"),
        _cs("600005", "600005.XSHG", 100, de_listed="2026-06-25"),
        _cs("600003", "600003.XSHG", 100),
        _cs("600004", "600004.XSHG", 100),
    ]
    if malicious_pickle:
        # 含 GLOBAL opcode 的 protocol 2 pickle：安全解码必须结构性拒绝
        (root / "instruments.pk").write_bytes(b"\x80\x02cbuiltins\nprint\nq\x00.")
    else:
        (root / "instruments.pk").write_bytes(
            pickle.dumps(instruments, protocol=2)
        )

    # --- trading_dates.npy（RQAlpha 日历长于冻结日历，截止日后部分禁用）---
    np.save(
        root / "trading_dates.npy",
        np.array(sessions + [20260701], dtype="int64"),
    )

    # --- 其余 bundle 成员（本层只哈希不读取语义）---
    for name in ("dividends.h5", "split_factor.h5", "ex_cum_factor.h5"):
        (root / name).write_bytes(f"placeholder:{name}".encode("utf-8"))

    # --- 冻结交易日历 ---
    calendar_days = [d for d in sessions if not (calendar_drop_day and d == 20260629)]
    calendar_path = tmp_path / "trade_calendar.parquet"
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [str(d) for d in calendar_days], format="%Y%m%d"
            )
        }
    ).to_parquet(calendar_path)

    # --- v3 coverage matrix 与 manifest ---
    coverage_path = v3_root / "coverage_matrix.parquet"
    pd.DataFrame(
        {
            "code": ["600001", "600002", "688001", "600005", "600003", "600004"],
            "status": [
                "available",
                "available",
                "available",
                "available",
                "quarantine",
                "source_missing",
            ],
        }
    ).to_parquet(coverage_path)
    coverage_sha = _sha256_file(coverage_path)
    manifest = {
        "format": "free_stockdb_csi300_historical_am_inputs_v3",
        "status": "completed",
        "status_counts": {
            "available": 4,
            "quarantine": 1,
            "source_missing": 1,
        },
        "coverage_matrix": {
            "relative_path": "coverage_matrix.parquet",
            "sha256": coverage_sha,
        },
    }
    manifest_path = v3_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    anchor = {
        "format": TRUSTED_OVERLAY_ANCHOR["format"],
        "bundle_name": "synthetic_fixture_bundle",
        "archive_sha256": "1" * 64,
        "member_inventory_sha256": "2" * 64,
        "member_sha256": {
            name: _sha256_file(root / name)
            for name in TRUSTED_OVERLAY_ANCHOR["member_sha256"]
        },
        "session_first": 20260622,
        "session_last_inclusive": 20260630,
        "available_code_count": 4,
        "v3_manifest_sha256": _sha256_file(manifest_path),
        "v3_coverage_matrix_sha256": coverage_sha,
        "frozen_calendar_sha256": _sha256_file(calendar_path),
        "calendar_intersection_rows": len(calendar_days),
        "prev_close_excluded": ("990018.XSHG",),
        "unresolved_gaps": (("600002.XSHG", 20260629),),
        "lot_size_allowed": (100, 200),
    }
    if tamper_stocks_after_anchor:
        # 锚定型后篡改成员字节：供应链身份必须拒载
        with (root / "stocks.h5").open("r+b") as handle:
            handle.seek(-1, 2)
            last = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([last[0] ^ 0xFF]))
    if anchor_overrides:
        anchor.update(anchor_overrides)
    return {
        "root": root,
        "v3_root": v3_root,
        "calendar_path": calendar_path,
        "anchor": anchor,
    }


def _load_fixture(fixture: dict):
    return load_rqalpha_execution_overlay(
        fixture["root"],
        v3_export_root=fixture["v3_root"],
        frozen_calendar_path=fixture["calendar_path"],
        trusted_anchor=fixture["anchor"],
    )


@pytest.fixture()
def synthetic_overlay(tmp_path):
    overlay = _load_fixture(_build_fixture(tmp_path))
    yield overlay
    overlay.close()


# ---------------------------------------------------------------------------
# 组 1：供应链身份
# ---------------------------------------------------------------------------


def test_member_bytes_tampered_after_anchor_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path, tamper_stocks_after_anchor=True)
    with pytest.raises(RQAlphaOverlayError, match="stocks.h5 SHA-256"):
        _load_fixture(fixture)


def test_anchor_member_sha_mismatch_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    fixture["anchor"]["member_sha256"] = dict(fixture["anchor"]["member_sha256"])
    fixture["anchor"]["member_sha256"]["suspended_days.h5"] = "a" * 64
    with pytest.raises(RQAlphaOverlayError, match="suspended_days.h5 SHA-256"):
        _load_fixture(fixture)


def test_anchor_missing_member_entry_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    members = dict(fixture["anchor"]["member_sha256"])
    members.pop("instruments.pk")
    fixture["anchor"]["member_sha256"] = members
    with pytest.raises(RQAlphaOverlayError, match="成员文件清单不完整"):
        _load_fixture(fixture)


def test_anchor_invalid_archive_sha_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path, anchor_overrides={"archive_sha256": "XYZ"})
    with pytest.raises(RQAlphaOverlayError, match="archive_sha256"):
        _load_fixture(fixture)


def test_anchor_missing_field_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    fixture["anchor"].pop("unresolved_gaps")
    with pytest.raises(RQAlphaOverlayError, match="字段集合"):
        _load_fixture(fixture)


def test_default_trusted_anchor_rejects_synthetic_directory(tmp_path) -> None:
    fixture = _build_fixture(tmp_path)
    with pytest.raises(RQAlphaOverlayError, match="SHA-256|不存在"):
        load_rqalpha_execution_overlay(
            fixture["root"],
            v3_export_root=fixture["v3_root"],
            frozen_calendar_path=fixture["calendar_path"],
            trusted_anchor=None,
        )


def test_v3_manifest_hash_mismatch_rejected(tmp_path) -> None:
    fixture = _build_fixture(tmp_path, anchor_overrides={"v3_manifest_sha256": "b" * 64})
    with pytest.raises(RQAlphaOverlayError, match="v3 manifest SHA-256"):
        _load_fixture(fixture)


# ---------------------------------------------------------------------------
# 组 2：日期域上限与未决缺口
# ---------------------------------------------------------------------------


def test_session_after_bundle_cutoff_rejected(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="晚于 bundle 截止日"):
        synthetic_overlay.derive_execution_state("600001", "2026-07-01")


def test_session_before_bundle_first_rejected(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="早于 bundle 首个交易日"):
        synthetic_overlay.derive_execution_state("600001", "2026-06-19")


def test_non_trading_day_rejected(synthetic_overlay) -> None:
    # 2026-06-27 是周六，不在域内交易日集合
    with pytest.raises(RQAlphaOverlayError, match="不是 RQAlpha 域内交易日"):
        synthetic_overlay.derive_execution_state("600001", "2026-06-27")


def test_unresolved_gap_fail_closed(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="未决缺口"):
        synthetic_overlay.derive_execution_state("600002", "2026-06-29")


@requires_real_bundle
def test_real_unresolved_gap_dates_fail_closed(real_overlay) -> None:
    for symbol, day in (
        ("600000", "2026-07-01"),
        ("600000", "2026-07-02"),
        ("688072", "2026-07-01"),
        ("688072", "2026-07-02"),
        ("688072", "2026-07-03"),
        ("688072", "2026-07-06"),
        ("688072", "2026-07-07"),
        ("688072", "2026-07-08"),
        ("688072", "2026-07-09"),
        ("688072", "2026-07-10"),
    ):
        with pytest.raises(RQAlphaOverlayError, match="晚于 bundle 截止日"):
            real_overlay.derive_execution_state(symbol, day)


# ---------------------------------------------------------------------------
# 组 3：代码域＝741 available 交集
# ---------------------------------------------------------------------------


def test_quarantine_code_rejected(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="quarantine"):
        synthetic_overlay.derive_execution_state("600003", "2026-06-22")


def test_source_missing_code_rejected(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="source_missing"):
        synthetic_overlay.derive_execution_state("600004", "2026-06-22")


def test_unknown_code_rejected(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="不在冻结 949 覆盖矩阵内"):
        synthetic_overlay.derive_execution_state("999999", "2026-06-22")


@requires_real_bundle
def test_real_quarantine_and_source_missing_not_upgraded(real_overlay) -> None:
    status_by_code = real_overlay._status_by_code
    quarantine = sorted(
        code for code, status in status_by_code.items() if status == "quarantine"
    )
    source_missing = sorted(
        code for code, status in status_by_code.items() if status == "source_missing"
    )
    assert len(quarantine) == 207
    assert len(source_missing) == 1
    with pytest.raises(RQAlphaOverlayError, match="quarantine"):
        real_overlay.derive_execution_state(quarantine[0], "2026-06-30")
    with pytest.raises(RQAlphaOverlayError, match="source_missing"):
        real_overlay.derive_execution_state(source_missing[0], "2026-06-30")


# ---------------------------------------------------------------------------
# 组 4：停牌只读 suspended_days.h5
# ---------------------------------------------------------------------------


def test_suspended_without_bar_is_suspended(synthetic_overlay) -> None:
    state = synthetic_overlay.derive_execution_state("600001", "2026-06-25")
    assert state.status == "SUSPENDED"
    assert state.close is None
    assert state.prev_close is None
    assert state.limit_up is None
    assert state.limit_down is None


def test_suspended_with_filler_bar_is_suspended(synthetic_overlay) -> None:
    # 停牌日存在填充 bar：结论仍只来自 suspended_days.h5，价格不外泄
    state = synthetic_overlay.derive_execution_state("600001", "2026-06-26")
    assert state.status == "SUSPENDED"
    assert state.close is None


def test_zero_volume_without_suspension_is_open(synthetic_overlay) -> None:
    # 禁止由 volume==0 推断停牌
    state = synthetic_overlay.derive_execution_state("600001", "2026-06-29")
    assert state.status == "OPEN"
    assert state.close == pytest.approx(10.6)


def test_non_suspended_missing_bar_fail_closed(tmp_path) -> None:
    overlay = _load_fixture(_build_fixture(tmp_path, gap_no_bar=True))
    try:
        with pytest.raises(RQAlphaOverlayError, match="覆盖缺口必须失败关闭"):
            overlay.derive_execution_state("600002", "2026-06-24")
    finally:
        overlay.close()


@requires_real_bundle
def test_real_000002_suspension_and_resumption(real_overlay) -> None:
    # 审计反例：2006-05-30 停牌日 bar 是填充值，停牌只来自 suspended_days.h5
    suspended = real_overlay.derive_execution_state("000002", "2006-05-30")
    assert suspended.status == "SUSPENDED"
    assert suspended.close is None

    long_stop = real_overlay.derive_execution_state("000002", "2017-07-17")
    assert long_stop.status == "SUSPENDED"

    resumed = real_overlay.derive_execution_state("000002", "2017-07-18")
    assert resumed.status == "OPEN"
    assert resumed.close == pytest.approx(25.3)
    assert resumed.prev_close == pytest.approx(24.59)
    assert resumed.limit_up == pytest.approx(27.05)
    assert resumed.limit_down == pytest.approx(22.13)


# ---------------------------------------------------------------------------
# 组 5：st_stock_days 降序显式处理与 ST 边界
# ---------------------------------------------------------------------------


def test_st_ascending_injection_detected(tmp_path) -> None:
    with pytest.raises(RQAlphaOverlayError, match="不是严格降序"):
        _load_fixture(_build_fixture(tmp_path, st_ascending=True))


def test_synthetic_st_days_resolved(synthetic_overlay) -> None:
    assert synthetic_overlay.derive_execution_state("600001", "2026-06-22").is_st is False
    assert synthetic_overlay.derive_execution_state("600001", "2026-06-23").is_st is True
    assert synthetic_overlay.derive_execution_state("600001", "2026-06-24").is_st is True


@requires_real_bundle
def test_real_st_entry_exit_boundaries_002602(real_overlay) -> None:
    # 审计 st_transition_samples 中唯一 v3 available 的样本股；
    # 000518/600228 属 quarantine，由下方防升级用例覆盖。
    assert real_overlay.derive_execution_state("002602", "2024-11-07").is_st is False
    assert real_overlay.derive_execution_state("002602", "2024-11-08").is_st is True
    assert real_overlay.derive_execution_state("002602", "2025-11-11").is_st is True
    assert real_overlay.derive_execution_state("002602", "2025-11-12").is_st is False


@requires_real_bundle
def test_real_002602_st_limit_prices(real_overlay) -> None:
    first = real_overlay.derive_execution_state("002602", "2024-11-08")
    assert first.limit_up == pytest.approx(5.39)
    assert first.limit_down == pytest.approx(4.87)
    exited = real_overlay.derive_execution_state("002602", "2025-11-12")
    assert exited.limit_up == pytest.approx(19.37)
    assert exited.limit_down == pytest.approx(15.85)


@requires_real_bundle
def test_real_quarantine_audit_st_samples_not_upgraded(real_overlay) -> None:
    # 000518/600228 在 v3 中为 quarantine：即便 RQAlpha 有完整 ST 边界数据，
    # 也必须维持拒绝，不得因此升级（硬条件、防洗白声明第 7 条）。
    for symbol, first_st_day in (("000518", "2025-04-30"), ("600228", "2025-04-28")):
        with pytest.raises(RQAlphaOverlayError, match="quarantine"):
            real_overlay.derive_execution_state(symbol, first_st_day)


# ---------------------------------------------------------------------------
# 组 6：10% -> 20% 涨跌停制度切换（prev_close 复算）
# ---------------------------------------------------------------------------


@requires_real_bundle
def test_real_300750_limit_regime_transition(real_overlay) -> None:
    before = real_overlay.derive_execution_state("300750", "2020-08-21")
    assert before.prev_close == pytest.approx(188.42)
    assert before.limit_up == pytest.approx(207.26)
    ratio_before = (before.limit_up / before.prev_close - 1.0) * 100.0
    assert abs(ratio_before - 10.0) < 0.01

    after = real_overlay.derive_execution_state("300750", "2020-08-24")
    assert after.prev_close == pytest.approx(192.23)
    assert after.limit_up == pytest.approx(230.68)
    ratio_after = (after.limit_up / after.prev_close - 1.0) * 100.0
    assert abs(ratio_after - 20.0) < 0.01


@requires_real_bundle
def test_real_600519_ten_percent_negative_control(real_overlay) -> None:
    state = real_overlay.derive_execution_state("600519", "2020-08-24")
    assert state.status == "OPEN"
    assert state.prev_close == pytest.approx(1676.0)
    assert state.limit_up == pytest.approx(1843.6)
    assert state.limit_down == pytest.approx(1508.4)


# ---------------------------------------------------------------------------
# 组 7：科创板上市初期显式 0 限价不判 locked
# ---------------------------------------------------------------------------


def test_synthetic_zero_limit_never_locked(synthetic_overlay) -> None:
    for day in ("2026-06-22", "2026-06-23", "2026-06-30"):
        state = synthetic_overlay.derive_execution_state("688001", day)
        assert state.status == "OPEN"
        assert state.limit_up == 0.0
        assert state.limit_down == 0.0


@requires_real_bundle
def test_real_star_market_listing_zero_limit_not_locked(real_overlay) -> None:
    # 审计样本股 688981 在 v3 中为 quarantine，必须维持拒绝、不得升级；
    # 同语义改用 v3 available 的科创板股 688072（上市日 2022-04-20）验证。
    with pytest.raises(RQAlphaOverlayError, match="quarantine"):
        real_overlay.derive_execution_state("688981", "2020-07-16")
    for day in (
        "2022-04-20",
        "2022-04-21",
        "2022-04-22",
        "2022-04-25",
        "2022-04-26",
    ):
        state = real_overlay.derive_execution_state("688072", day)
        assert state.status == "OPEN"
        assert state.limit_up == 0.0
        assert state.limit_down == 0.0
    # 上市第 6 个交易日恢复显式 20% 限价，且当日精确触及涨停价
    sixth = real_overlay.derive_execution_state("688072", "2022-04-27")
    assert sixth.status == "LIMIT_UP_LOCKED"
    assert sixth.close == pytest.approx(110.59)
    assert sixth.limit_up == pytest.approx(110.59)
    ratio = (sixth.limit_up / sixth.prev_close - 1.0) * 100.0
    assert abs(ratio - 20.0) < 0.01


# ---------------------------------------------------------------------------
# 组 8：未来状态追加不改变历史输出
# ---------------------------------------------------------------------------


def test_future_state_append_keeps_history_bytes(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline"
    appended_dir = tmp_path / "appended"
    baseline_dir.mkdir()
    appended_dir.mkdir()
    queries = [
        ("600001", "2026-06-22"),
        ("600001", "2026-06-23"),
        ("600001", "2026-06-25"),
        ("600001", "2026-06-30"),
        ("600002", "2026-06-30"),
        ("688001", "2026-06-30"),
    ]

    def _observe(fixture: dict) -> str:
        overlay = _load_fixture(fixture)
        try:
            rows = []
            for symbol, day in queries:
                payload = overlay.derive_execution_state(symbol, day).to_dict()
                # 来源身份哈希必然随成员字节变化，历史状态语义必须逐字节不变
                payload.pop("source_identity_sha256")
                rows.append(payload)
            return json.dumps(rows, ensure_ascii=False, sort_keys=True)
        finally:
            overlay.close()

    baseline = _observe(_build_fixture(baseline_dir))
    appended = _observe(_build_fixture(appended_dir, future_append=True))
    assert baseline == appended


def test_future_session_still_rejected_after_append(tmp_path) -> None:
    overlay = _load_fixture(_build_fixture(tmp_path, future_append=True))
    try:
        with pytest.raises(RQAlphaOverlayError, match="晚于 bundle 截止日"):
            overlay.derive_execution_state("600001", "2026-07-01")
    finally:
        overlay.close()


# ---------------------------------------------------------------------------
# 组 9：990018.XSHG prev_close 失败关闭
# ---------------------------------------------------------------------------


def test_prev_close_excluded_symbol_rejected(tmp_path) -> None:
    fixture = _build_fixture(
        tmp_path,
        anchor_overrides={"prev_close_excluded": ("990018.XSHG", "600002.XSHG")},
    )
    overlay = _load_fixture(fixture)
    try:
        with pytest.raises(RQAlphaOverlayError, match="prev_close 语义被裁决排除"):
            overlay.derive_execution_state("600002", "2026-06-22")
    finally:
        overlay.close()


def test_prev_close_nan_fail_closed(tmp_path) -> None:
    overlay = _load_fixture(_build_fixture(tmp_path, prev_close_nan=True))
    try:
        with pytest.raises(RQAlphaOverlayError, match="prev_close 非有限数"):
            overlay.derive_execution_state("600002", "2026-06-22")
    finally:
        overlay.close()


@requires_real_bundle
def test_real_990018_not_reachable(real_overlay) -> None:
    # 990018 是 949 矩阵中唯一的 source_missing；同时 990018.XSHG 的
    # prev_close 语义被裁决显式排除，双重失败关闭。
    assert "990018.XSHG" in real_overlay.identity.prev_close_excluded
    assert real_overlay._status_by_code.get("990018") == "source_missing"
    with pytest.raises(RQAlphaOverlayError, match="source_missing"):
        real_overlay.derive_execution_state("990018", "2026-06-30")


# ---------------------------------------------------------------------------
# 组 10：FreeStockDB Parquet 只读约束
# ---------------------------------------------------------------------------


@requires_real_bundle
def test_real_inputs_stay_untouched(real_overlay) -> None:
    manifest_path = V3_EXPORT_ROOT / "manifest.json"
    coverage_path = V3_EXPORT_ROOT / "coverage_matrix.parquet"
    before = {
        "manifest": _sha256_file(manifest_path),
        "coverage": _sha256_file(coverage_path),
        "calendar": _sha256_file(FROZEN_CALENDAR_PATH),
    }
    real_overlay.derive_execution_state("600519", "2020-08-24")
    after = {
        "manifest": _sha256_file(manifest_path),
        "coverage": _sha256_file(coverage_path),
        "calendar": _sha256_file(FROZEN_CALENDAR_PATH),
    }
    assert before == after
    assert before["manifest"] == TRUSTED_OVERLAY_ANCHOR["v3_manifest_sha256"]
    assert before["coverage"] == TRUSTED_OVERLAY_ANCHOR["v3_coverage_matrix_sha256"]
    assert before["calendar"] == TRUSTED_OVERLAY_ANCHOR["frozen_calendar_sha256"]


# ---------------------------------------------------------------------------
# 组 11：pickle 安全解码策略复用
# ---------------------------------------------------------------------------


def test_malicious_pickle_rejected_by_safe_unpickler(tmp_path) -> None:
    # 锚哈希与恶意字节一致：拒绝必须来自安全解码器，而非哈希差异
    with pytest.raises(RQAlphaOverlayError, match="安全解码失败"):
        _load_fixture(_build_fixture(tmp_path, malicious_pickle=True))


# ---------------------------------------------------------------------------
# 组 12：h5py 通过项目 requirements/锁文件体系进入
# ---------------------------------------------------------------------------


def test_h5py_pinned_in_project_lock_files() -> None:
    source = (PROJECT_ROOT / "requirements-windows.in").read_text(encoding="utf-8")
    assert re.search(r"^h5py==3\.16\.0$", source, re.M)
    for lock_name in ("requirements-windows.lock", "requirements-dev.lock"):
        lock = (PROJECT_ROOT / lock_name).read_text(encoding="utf-8")
        assert re.search(r"^h5py==3\.16\.0 \\$", lock, re.M), lock_name


# ---------------------------------------------------------------------------
# 组 13：dataset 缺失语义显式固定
# ---------------------------------------------------------------------------


def test_missing_suspended_dataset_means_no_record(synthetic_overlay) -> None:
    # 600002 在 suspended_days.h5 无 dataset：显式声明为"无停牌记录"
    state = synthetic_overlay.derive_execution_state("600002", "2026-06-22")
    assert state.status == "OPEN"
    identity = synthetic_overlay.identity
    assert "无停牌记录" in identity.suspended_missing_dataset_semantics


def test_missing_st_dataset_fail_closed(tmp_path) -> None:
    with pytest.raises(RQAlphaOverlayError, match="st_stock_days.h5 中缺少"):
        _load_fixture(_build_fixture(tmp_path, st_missing_dataset=True))


@requires_real_bundle
def test_real_741_dataset_presence_mapping(real_overlay) -> None:
    summary = real_overlay.dataset_presence_summary
    assert summary["stocks"] == 741
    assert summary["st"] == 741
    assert 0 < summary["suspended"] <= 741


# ---------------------------------------------------------------------------
# 组 14：派生空间隔离（qfq×未复权跨口径比较必须被禁止）
# ---------------------------------------------------------------------------


def test_synthetic_touch_derivation_in_raw_space(synthetic_overlay) -> None:
    locked_up = synthetic_overlay.derive_execution_state("600001", "2026-06-23")
    assert locked_up.status == "LIMIT_UP_LOCKED"
    assert locked_up.close == pytest.approx(11.55)
    assert locked_up.limit_up == pytest.approx(11.55)
    locked_down = synthetic_overlay.derive_execution_state("600001", "2026-06-24")
    assert locked_down.status == "LIMIT_DOWN_LOCKED"
    assert locked_down.close == pytest.approx(10.4)
    assert locked_down.derivation_rule_version == DERIVATION_RULE_VERSION


@requires_real_bundle
def test_real_cross_space_comparison_would_misjudge(real_overlay) -> None:
    import pandas as pd

    # RQAlpha 未复权空间：2006-11-30 万科 close 精确触及显式涨停价 11.98
    state = real_overlay.derive_execution_state("000002", "2006-11-30")
    assert state.status == "LIMIT_UP_LOCKED"
    assert state.close == pytest.approx(11.98)
    assert state.limit_up == pytest.approx(11.98)

    # FreeStockDB v3 是前复权（qfq）价格：同日 close 与未复权空间不同口径
    coverage = pd.read_parquet(
        V3_EXPORT_ROOT / "coverage_matrix.parquet",
        columns=["code", "data_relative_path"],
    )
    relative = coverage.loc[coverage["code"] == "000002", "data_relative_path"]
    assert len(relative) == 1
    qfq = pd.read_parquet(V3_EXPORT_ROOT / str(relative.iloc[0]))
    time_column = qfq["time"]
    if time_column.dtype.kind in "iu":
        stamps = pd.to_datetime(time_column, unit="s", utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
    else:
        stamps = pd.to_datetime(time_column, errors="raise")
    day_rows = qfq.loc[stamps.dt.strftime("%Y-%m-%d") == "2006-11-30"]
    assert len(day_rows) == 1
    qfq_close = float(day_rows["close"].iloc[0])
    # 若把 qfq close 与未复权 limit_up 跨口径比较，会把真实触及日误判为未触及
    assert qfq_close != pytest.approx(state.close)
    assert (qfq_close >= state.limit_up) != (state.close >= state.limit_up)


# ---------------------------------------------------------------------------
# 组 15：lot_size 来源（instruments.pk round_lot）
# ---------------------------------------------------------------------------


def test_synthetic_lot_size_from_round_lot(synthetic_overlay) -> None:
    assert synthetic_overlay.derive_execution_state("600001", "2026-06-22").lot_size == 100
    assert synthetic_overlay.derive_execution_state("688001", "2026-06-22").lot_size == 200


@pytest.mark.parametrize("bad_lot", [0, -100, 7, 100.5, None, "100"])
def test_invalid_round_lot_fail_closed(tmp_path, bad_lot) -> None:
    with pytest.raises(RQAlphaOverlayError, match="round_lot"):
        _load_fixture(_build_fixture(tmp_path, bad_round_lot=bad_lot))


@requires_real_bundle
def test_real_star_market_lot_size_is_200(real_overlay) -> None:
    star = real_overlay.derive_execution_state("688072", "2022-04-27")
    assert star.lot_size == 200
    main_board = real_overlay.derive_execution_state("600519", "2020-08-24")
    assert main_board.lot_size == 100
    assert "round_lot" in real_overlay.identity.lot_size_source


# ---------------------------------------------------------------------------
# 组 16：日历一致性运行时哨兵
# ---------------------------------------------------------------------------


def test_calendar_sentinel_detects_divergence(tmp_path) -> None:
    with pytest.raises(RQAlphaOverlayError, match="交易日历在共同区间不一致"):
        _load_fixture(_build_fixture(tmp_path, calendar_drop_day=True))


def test_calendar_row_count_bound_to_anchor(tmp_path) -> None:
    fixture = _build_fixture(
        tmp_path, anchor_overrides={"calendar_intersection_rows": 9}
    )
    with pytest.raises(RQAlphaOverlayError, match="共同区间交易日数量与受信锚不一致"):
        _load_fixture(fixture)


@requires_real_bundle
def test_real_calendar_sentinel_matches_audit(real_overlay) -> None:
    assert real_overlay.identity.calendar_intersection_rows == 5235


# ---------------------------------------------------------------------------
# 组 17：来源身份可绑定、不可变、篡改可检出
# ---------------------------------------------------------------------------


def test_identity_sha256_is_deterministic_and_tamper_evident(
    synthetic_overlay,
) -> None:
    identity = synthetic_overlay.identity
    assert identity.identity_sha256
    payload = identity.to_dict()
    assert payload["identity_sha256"] == identity.identity_sha256

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.archive_sha256 = "0" * 64  # type: ignore[misc]

    with pytest.raises(RQAlphaOverlayError, match="identity_sha256"):
        dataclasses.replace(identity, archive_sha256="0" * 64)

    rebuilt = dataclasses.replace(
        identity, archive_sha256="0" * 64, identity_sha256=""
    )
    assert rebuilt.identity_sha256 != identity.identity_sha256


def test_states_carry_source_identity(synthetic_overlay) -> None:
    state = synthetic_overlay.derive_execution_state("600001", "2026-06-22")
    assert state.source_identity_sha256 == (
        synthetic_overlay.identity.identity_sha256
    )
    assert state.derivation_rule_version == DERIVATION_RULE_VERSION


def test_identity_declares_adjudication_semantics(synthetic_overlay) -> None:
    identity = synthetic_overlay.identity
    assert identity.allowed_fields == STOCKS_ALLOWED_FIELDS
    assert "conservative_daily_touch_rule" in identity.derivation_rule_version
    joined_forbidden = "\n".join(identity.forbidden_semantics)
    assert "盘口" in joined_forbidden
    assert "2026-07-01" in joined_forbidden
    assert "决策时钟" in joined_forbidden
    assert len(identity.unresolved_gaps) == len(
        synthetic_overlay._anchor["unresolved_gaps"]
    )


@requires_real_bundle
def test_real_identity_matches_audit_anchor(real_overlay) -> None:
    identity = real_overlay.identity
    assert identity.archive_sha256 == TRUSTED_OVERLAY_ANCHOR["archive_sha256"]
    assert dict(identity.member_sha256) == TRUSTED_OVERLAY_ANCHOR["member_sha256"]
    assert identity.session_last_inclusive == 20260630
    assert identity.available_code_count == 741
    assert len(identity.unresolved_gaps) == 10


# ---------------------------------------------------------------------------
# 组 18：169 个 FreeStockDB 漏数日不得填补、不得归类为停牌
# ---------------------------------------------------------------------------


@requires_real_bundle
def test_real_freestockdb_missing_days_not_reclassified(real_overlay) -> None:
    # RQAlpha 有正常 bar 且非停牌；FreeStockDB 漏数由 04C 价格层失败关闭，
    # overlay 不得把这些日子归类为 SUSPENDED，也不提供价格填补接口。
    for symbol, day in (("000338", "2018-07-02"), ("000333", "2015-08-05")):
        state = real_overlay.derive_execution_state(symbol, day)
        assert state.status != "SUSPENDED"
    assert not hasattr(real_overlay, "fill_missing_prices")


# ---------------------------------------------------------------------------
# 其他适配器层合同
# ---------------------------------------------------------------------------


def test_status_enum_matches_execution_quote_contract() -> None:
    from portfolio_manager import execution as execution_module

    assert EXECUTION_STATE_STATUSES == execution_module._QUOTE_STATUSES


def test_symbol_and_session_validation(synthetic_overlay) -> None:
    with pytest.raises(RQAlphaOverlayError, match="6 位数字文本"):
        synthetic_overlay.derive_execution_state("60001", "2026-06-22")
    with pytest.raises(RQAlphaOverlayError, match="YYYY-MM-DD"):
        synthetic_overlay.derive_execution_state("600001", "20260622")
    with pytest.raises(RQAlphaOverlayError, match="有效日历日期"):
        synthetic_overlay.derive_execution_state("600001", "2026-02-30")


def test_delisted_symbol_outside_listing_validity(synthetic_overlay) -> None:
    assert (
        synthetic_overlay.derive_execution_state("600005", "2026-06-24").status
        == "OPEN"
    )
    with pytest.raises(RQAlphaOverlayError, match="不在上市有效期内"):
        synthetic_overlay.derive_execution_state("600005", "2026-06-26")


@requires_real_bundle
def test_real_open_state_shape(real_overlay) -> None:
    state = real_overlay.derive_execution_state("600519", "2020-08-24")
    payload = state.to_dict()
    assert payload["status"] in EXECUTION_STATE_STATUSES
    assert payload["order_book_id"] == "600519.XSHG"
    assert payload["lot_size"] == 100
    assert payload["source_identity_sha256"] == (
        real_overlay.identity.identity_sha256
    )


@pytest.fixture(scope="module")
def real_overlay():
    if not _G_DRIVE_READY:
        pytest.skip("G 盘 RQAlpha bundle / v3 导出 / 冻结日历不可读")
    overlay = load_rqalpha_execution_overlay()
    yield overlay
    overlay.close()
