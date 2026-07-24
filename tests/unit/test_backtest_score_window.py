from __future__ import annotations

import numpy as np
import pytest
import torch

from backtest_viz.chart import BacktestChart
from backtest_viz.engine import BacktestEngine
from backtest_viz.chart import (
    _entry_execution_bar,
    _exit_execution_bar,
    _market_series,
)


class _Vm:
    def execute(self, _formula, features):
        return torch.ones((features.shape[0], features.shape[-1]))


def _inputs(bars: int = 20):
    base = torch.linspace(100.0, 120.0, bars).reshape(1, bars)
    raw = {
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base + 0.5,
        "volume": torch.ones_like(base),
        "time": torch.arange(bars, dtype=torch.int64).reshape(1, bars),
    }
    features = torch.zeros((1, 1, bars))
    return raw, features


def test_score_window_keeps_feature_warmup_but_scores_only_tail() -> None:
    raw, features = _inputs()
    engine = BacktestEngine([1], cost_rate=0)
    engine.vm = _Vm()

    result = engine.run(
        raw,
        features,
        ["X"],
        score_start_index=7,
    )[0]

    assert len(result.pnl) == 11
    assert result.times[0] == 7
    assert result.times[-1] == 17
    assert np.isclose(result.position[0], np.tanh(1.0))
    assert result.trades[-1].exit_price == pytest.approx(120.0)

    trade = result.trades[-1]
    market_times = _market_series(result, "times")
    market_open = _market_series(result, "open")
    entry_exec = _entry_execution_bar(trade, len(market_times))
    exit_exec = _exit_execution_bar(trade, len(market_times))
    assert entry_exec == trade.entry_exec_bar == 1
    assert exit_exec == trade.exit_exec_bar == 12
    assert market_times[exit_exec] == trade.exit_time == 19
    assert market_open[exit_exec] == pytest.approx(trade.exit_price)


def test_trade_zoom_exit_marker_uses_recorded_exit_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, features = _inputs()
    engine = BacktestEngine([1], cost_rate=0)
    engine.vm = _Vm()
    result = engine.run(raw, features, ["X"], score_start_index=7)[0]

    import matplotlib.axes

    observed: list[tuple[float, float]] = []
    original_plot = matplotlib.axes.Axes.plot

    def capture_plot(self, *args, **kwargs):
        if kwargs.get("marker") == "D":
            observed.append((float(args[0]), float(args[1])))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture_plot)
    BacktestChart().plot_trade_zoom(result, 0)

    trade = result.trades[0]
    win_start = max(0, trade.entry_exec_bar - 20)
    assert len(observed) == 1
    assert observed[0][0] == pytest.approx(float(trade.exit_exec_bar - win_start))
    assert observed[0][1] == pytest.approx(trade.exit_price)


@pytest.mark.parametrize("index", [-1, 18, True])
def test_score_window_rejects_invalid_start(index) -> None:
    raw, features = _inputs()
    engine = BacktestEngine([1])
    engine.vm = _Vm()

    with pytest.raises(ValueError, match="至少 3 根"):
        engine.run(raw, features, ["X"], score_start_index=index)
