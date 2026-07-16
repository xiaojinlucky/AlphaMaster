from __future__ import annotations

import numpy as np
import pytest
import torch

from backtest_viz.engine import BacktestEngine


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

    assert len(result.pnl) == 13
    assert result.times[0] == 7
    assert np.isclose(result.position[0], np.tanh(1.0))


@pytest.mark.parametrize("index", [-1, 18, True])
def test_score_window_rejects_invalid_start(index) -> None:
    raw, features = _inputs()
    engine = BacktestEngine([1])
    engine.vm = _Vm()

    with pytest.raises(ValueError, match="至少 3 根"):
        engine.run(raw, features, ["X"], score_start_index=index)
