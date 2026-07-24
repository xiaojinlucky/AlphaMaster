"""MT5 正式数据只能持久化已收盘 K 线。"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import data_pipeline.kline_cache as cache_module
from data_pipeline.kline_cache import (
    KlineCache,
    _has_current_cache_contract,
)


def _bar(moment: int, close: float) -> dict:
    return {
        "time": moment,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "tick_volume": 1,
    }


def test_incremental_cache_upserts_same_closed_bar_timestamp(
    tmp_path,
    monkeypatch,
) -> None:
    fake_mt5 = SimpleNamespace(
        copy_rates_from_pos=lambda *_args: [
            _bar(100, 2.0),
            _bar(200, 3.0),
        ]
    )
    monkeypatch.setattr(cache_module, "_MT5_AVAILABLE", True)
    monkeypatch.setattr(cache_module, "mt5", fake_mt5)
    cache = KlineCache(cache_dir=tmp_path)
    local = pd.DataFrame([_bar(100, 1.0)])

    updated = cache._incremental_update("XAUUSD", local, last_time=100)

    assert updated["time"].tolist() == [100, 200]
    assert updated["close"].tolist() == [2.0, 3.0]


def test_cache_requests_previous_closed_bar_not_current_bar(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    def copy_rates(symbol, timeframe, start_pos, count):
        calls.append((symbol, timeframe, start_pos, count))
        return [_bar(100, 1.0)]

    fake_mt5 = SimpleNamespace(copy_rates_from_pos=copy_rates)
    monkeypatch.setattr(cache_module, "_MT5_AVAILABLE", True)
    monkeypatch.setattr(cache_module, "mt5", fake_mt5)
    cache = KlineCache(cache_dir=tmp_path, timeframe=16385, bars_count=50)

    cache._full_download("XAUUSD")

    assert calls == [("XAUUSD", 16385, 1, 50)]
    assert _has_current_cache_contract(cache._cache_path("XAUUSD"))


def test_incremental_cache_drops_legacy_unclosed_tail(
    tmp_path,
    monkeypatch,
) -> None:
    fake_mt5 = SimpleNamespace(
        copy_rates_from_pos=lambda *_args: [
            _bar(100, 1.0),
            _bar(200, 2.0),
        ]
    )
    monkeypatch.setattr(cache_module, "_MT5_AVAILABLE", True)
    monkeypatch.setattr(cache_module, "mt5", fake_mt5)
    cache = KlineCache(cache_dir=tmp_path)
    local = pd.DataFrame(
        [_bar(100, 1.0), _bar(200, 1.5), _bar(300, 9.0)]
    )

    updated = cache._incremental_update("XAUUSD", local, last_time=300)

    assert updated["time"].tolist() == [100, 200]
    assert updated["close"].tolist() == [1.0, 2.0]


def test_legacy_cache_contract_forces_full_online_rebuild(
    tmp_path,
    monkeypatch,
) -> None:
    cache = KlineCache(cache_dir=tmp_path)
    path = cache._cache_path("XAUUSD")
    pd.DataFrame([_bar(100, 9.0)]).to_parquet(path, index=False)
    rebuilt = pd.DataFrame([_bar(100, 1.0)])
    calls = []

    def full_download(symbol):
        calls.append(symbol)
        return rebuilt

    monkeypatch.setattr(cache, "_full_download", full_download)

    result = cache.get("XAUUSD", mt5_connected=True)

    assert calls == ["XAUUSD"]
    assert result.equals(rebuilt)


def test_legacy_cache_contract_is_rejected_offline(tmp_path) -> None:
    cache = KlineCache(cache_dir=tmp_path)
    path = cache._cache_path("XAUUSD")
    pd.DataFrame([_bar(100, 9.0)]).to_parquet(path, index=False)

    assert cache.get("XAUUSD", mt5_connected=False) is None
