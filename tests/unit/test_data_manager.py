"""
tests/unit/test_data_manager.py — MT5DataManager 单元测试

验证需求：
  - Req 3.5: 少于 MIN_BARS 的品种应被排除并记录 WARNING

注意：测试使用较小的数据量（100/2000 bars），需同时 patch Config.MIN_BARS=100
以避免受全局配置（3000）影响。
"""

import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# 测试用的 MIN_BARS 值（与测试数据大小匹配）
_TEST_MIN_BARS = 100


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_ohlcv_df(n_rows: int, start_time: int = 1_000_000) -> pd.DataFrame:
    """构造含 n_rows 行的标准 OHLCV DataFrame。

    列：time, open, high, low, close, tick_volume
    time 为唯一递增整数（Unix 时间戳风格）。
    """
    times = np.arange(start_time, start_time + n_rows, dtype=np.int64)
    opens = np.random.uniform(100.0, 200.0, size=n_rows)
    highs = opens + np.random.uniform(0.1, 5.0, size=n_rows)
    lows  = opens - np.random.uniform(0.1, 5.0, size=n_rows)
    closes = opens + np.random.uniform(-2.0, 2.0, size=n_rows)
    volumes = np.random.randint(100, 10_000, size=n_rows).astype(np.int64)

    return pd.DataFrame({
        "time":        times,
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "tick_volume": volumes,
    })


def _make_mock_fetcher(return_map: dict) -> MagicMock:
    """构造 MT5DataFetcher mock，根据 symbol 返回不同 DataFrame。

    Args:
        return_map: {symbol: pd.DataFrame}
    """
    fetcher = MagicMock()

    def _fetch_side_effect(symbol, timeframe, count):
        if symbol in return_map:
            return return_map[symbol]
        # 默认返回空 DataFrame
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "tick_volume"]
        )

    fetcher.fetch.side_effect = _fetch_side_effect
    return fetcher


def _make_deterministic_ohlcv_df(
    times: list[int], base_open: float
) -> pd.DataFrame:
    """构造价格可预测的 OHLCV 数据，便于验证时间轴语义。"""
    opens = base_open + np.arange(len(times), dtype=np.float64)
    return pd.DataFrame({
        "time":        np.asarray(times, dtype=np.int64),
        "open":        opens,
        "high":        opens + 1.0,
        "low":         opens - 1.0,
        "close":       opens + 0.5,
        "tick_volume": np.full(len(times), 100, dtype=np.int64),
    })


# ── 测试 1：少于 100 bars 的品种被排除 ────────────────────────────────────────

class TestSymbolExcludedWhenBelowMinBars:
    """Req 3.5: 数据不足 MIN_BARS(100) 的品种必须被排除。"""

    def test_symbol_with_fewer_than_100_bars_is_excluded(self):
        """US500 只有 50 行，应被排除；XAUUSD 和 EURUSD 各有 2000 行，应保留。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(2000, start_time=1_000_000),
            "US500":  _make_ohlcv_df(50,   start_time=2_000_000),   # < 100
            "EURUSD": _make_ohlcv_df(2000, start_time=1_000_000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        assert "US500"  not in manager.symbols, "US500 应因 bars < 100 被排除"
        assert "XAUUSD" in manager.symbols,     "XAUUSD 有 2000 bars，应保留"
        assert "EURUSD" in manager.symbols,     "EURUSD 有 2000 bars，应保留"

    def test_excluded_symbol_fetch_was_called(self):
        """即使 US500 被排除，fetcher.fetch() 也应被调用过（先获取后过滤）。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(2000),
            "US500":  _make_ohlcv_df(50),
            "EURUSD": _make_ohlcv_df(2000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        # fetch 应被调用 3 次（每个品种一次）
        assert mock_fetcher.fetch.call_count == 3

    def test_valid_symbols_count_after_exclusion(self):
        """排除 US500 后，manager.symbols 应只有 2 个有效品种。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(2000),
            "US500":  _make_ohlcv_df(10),   # 远低于 100
            "EURUSD": _make_ohlcv_df(500),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        assert len(manager.symbols) == 2

    def test_strict_trade_manager_rejects_one_missing_symbol(self):
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(200),
            "US500": _make_ohlcv_df(50),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(
            mock_fetcher,
            require_all_symbols=True,
        )
        with (
            patch.object(Config, "MIN_BARS", _TEST_MIN_BARS),
            patch.object(Config, "SYMBOLS", ["XAUUSD", "US500"]),
            pytest.raises(ValueError, match="缺少请求品种.*US500"),
        ):
            manager.load()

        assert manager.symbols == []


# ── 测试 2：所有品种都不足 100 bars 时抛出 ValueError ─────────────────────────

# ── 多品种时间轴回退必须保持因果 ─────────────────────────────────────────────

class TestCausalTimelineFallback:
    """交集不足时只能使用已有报价，不能把未来首报价填到过去。"""

    def test_union_fallback_drops_head_before_late_symbol_first_quote(self):
        """交集不足时，结果必须从晚开始品种的首个真实报价开始。"""
        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        # 两个品种各有 10 根；交集只有 9 根，触发回退。
        # 修复前 B 的第一根报价会被 bfill 到 time=0。
        fetch_map = {
            "A": _make_deterministic_ohlcv_df(list(range(0, 10)), 10.0),
            "B": _make_deterministic_ohlcv_df(list(range(1, 11)), 100.0),
        }
        manager = MT5DataManager(_make_mock_fetcher(fetch_map))

        with patch.object(Config, "MIN_BARS", 10), patch.object(Config, "SYMBOLS", ["A", "B"]):
            manager.load()

        raw = manager.raw_dict
        expected_times = torch.arange(1, 11, dtype=torch.int64)
        assert torch.equal(raw["time"][0], expected_times)
        assert torch.equal(raw["time"][1], expected_times)
        assert raw["open"].shape == (2, 10), "因果裁剪后恰好 MIN_BARS 也应可用"
        for field in ["open", "high", "low", "close", "volume"]:
            assert bool(torch.isfinite(raw[field]).all()), f"{field} 不应含 NaN 或 Inf"
        assert bool(torch.isfinite(manager.target_ret).all())

    def test_union_fallback_uses_only_prior_quote_for_internal_gap(self):
        """回退中的内部缺口只能沿用上一根真实报价。"""
        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        fetch_map = {
            "A": _make_deterministic_ohlcv_df(list(range(0, 11)), 10.0),
            "B": _make_deterministic_ohlcv_df(
                [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12], 100.0
            ),
        }
        manager = MT5DataManager(_make_mock_fetcher(fetch_map))

        with patch.object(Config, "MIN_BARS", 11), patch.object(Config, "SYMBOLS", ["A", "B"]):
            manager.load()

        raw = manager.raw_dict
        # 共同时间轴从 1 开始；time=4 缺 B 的真实报价，必须沿用 time=3 的 102。
        assert raw["time"][1, 3].item() == 4
        assert raw["open"][1, 3].item() == pytest.approx(102.0)
        assert raw["open"][1, 4].item() == pytest.approx(103.0)

    def test_strict_intersection_path_is_unchanged(self):
        """交集足够时必须只保留真实共享时间戳，不进入回退填充。"""
        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        fetch_map = {
            "A": _make_deterministic_ohlcv_df([0, 1, 3, 4], 10.0),
            "B": _make_deterministic_ohlcv_df([1, 2, 3, 5], 100.0),
        }
        manager = MT5DataManager(_make_mock_fetcher(fetch_map))

        with patch.object(Config, "MIN_BARS", 2), patch.object(Config, "SYMBOLS", ["A", "B"]):
            manager.load()

        raw = manager.raw_dict
        expected_times = torch.tensor([1, 3], dtype=torch.int64)
        assert torch.equal(raw["time"][0], expected_times)
        assert torch.equal(raw["time"][1], expected_times)
        assert torch.equal(raw["open"][0], torch.tensor([11.0, 12.0]))
        assert torch.equal(raw["open"][1], torch.tensor([100.0, 102.0]))

    def test_insufficient_causal_timeline_raises_without_partial_state(self):
        """原始行数够但去重后公共因果时间轴不足时，必须失败关闭。"""
        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        # B 原始行数为 10，但全是一个重复时间戳；去重后仅有一根真实报价。
        fetch_map = {
            "A": _make_deterministic_ohlcv_df(list(range(0, 10)), 10.0),
            "B": _make_deterministic_ohlcv_df([100] * 10, 100.0),
        }
        manager = MT5DataManager(_make_mock_fetcher(fetch_map))

        with patch.object(Config, "MIN_BARS", 10), patch.object(Config, "SYMBOLS", ["A", "B"]):
            with pytest.raises(ValueError, match="MIN_BARS"):
                manager.load()

        assert manager.symbols == []
        with pytest.raises(RuntimeError):
            _ = manager.raw_dict


class TestAllSymbolsBelowMinBarsRaisesError:
    """Req 3.5: 若所有品种均不满足 MIN_BARS，应抛出 ValueError。"""

    def test_raises_value_error_when_all_symbols_below_min_bars(self):
        """所有品种返回 < 100 行数据时，load() 必须抛出 ValueError。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(50),
            "US500":  _make_ohlcv_df(30),
            "EURUSD": _make_ohlcv_df(1),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            with pytest.raises(ValueError) as exc_info:
                manager.load()

        # 错误消息应提示无可用品种
        assert "No valid symbols" in str(exc_info.value) or \
               "fewer than" in str(exc_info.value) or \
               "MIN_BARS" in str(exc_info.value)

    def test_raises_value_error_with_empty_dataframes(self):
        """所有品种返回空 DataFrame（0 行）时，load() 也应抛出 ValueError。"""
        empty_df = pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "tick_volume"]
        )
        fetch_map = {
            "XAUUSD": empty_df,
            "EURUSD": empty_df,
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "EURUSD"]):
            with pytest.raises(ValueError):
                manager.load()


# ── 测试 3：恰好 100 bars 的品种应被保留（边界值）────────────────────────────

class TestExactlyMinBarsIsAccepted:
    """Req 3.5: MIN_BARS = 100，恰好 100 bars 的品种不应被排除。"""

    def test_exactly_100_bars_is_included(self):
        """品种返回恰好 100 行（= MIN_BARS）时，应被包含在 manager.symbols 中。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(100),  # 恰好等于 MIN_BARS
            "EURUSD": _make_ohlcv_df(2000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "EURUSD"]):
            manager.load()

        assert "XAUUSD" in manager.symbols, \
            "恰好 100 bars（= MIN_BARS）的品种应被保留，不应被排除"
        assert "EURUSD" in manager.symbols

    def test_99_bars_is_excluded_but_100_is_included(self):
        """99 bars（< MIN_BARS）应被排除，100 bars（= MIN_BARS）应保留——边界严格区分。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(99),   # 比 MIN_BARS 少 1
            "US500":  _make_ohlcv_df(100),  # 恰好等于 MIN_BARS
            "EURUSD": _make_ohlcv_df(2000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        assert "XAUUSD" not in manager.symbols, "99 bars 应被排除"
        assert "US500"  in manager.symbols,     "100 bars 应被保留"
        assert "EURUSD" in manager.symbols,     "2000 bars 应被保留"
