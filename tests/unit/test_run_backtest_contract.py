from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from run_backtest import _validate_strategy_data_contract


_DIGEST = "a" * 64


def _manager() -> SimpleNamespace:
    times = torch.tensor(
        [[1_700_000_000 + index * 3600 for index in range(4000)]],
        dtype=torch.int64,
    )
    return SimpleNamespace(
        symbol="BTCUSDT",
        timeframe="H1",
        source="okx",
        periods_per_year=6240,
        minimum_bars=3000,
        data_sha256=_DIGEST,
        dataset_id=f"sha256:{_DIGEST}",
        data_rows=4000,
        data_start="2023-11-14T22:13:20Z",
        data_end="2024-04-29T13:13:20Z",
        columns=["time", "open", "high", "low", "close", "tick_volume"],
        raw_dict={"time": times},
    )


def _strategy() -> dict:
    return {
        "symbol": "BTCUSDT",
        "timeframe": "H1",
        "local_source": "okx",
        "periods_per_year": 6240,
        "minimum_bars": 3000,
        "data_sha256": _DIGEST,
        "dataset_id": f"sha256:{_DIGEST}",
        "data_rows": 4000,
        "data_start": "2023-11-14T22:13:20Z",
        "data_end": "2024-04-29T13:13:20Z",
        "columns": ["time", "open", "high", "low", "close", "tick_volume"],
    }


def test_matching_strategy_data_contract_is_accepted() -> None:
    result = _validate_strategy_data_contract(_strategy(), _manager())
    assert result["evaluation_mode"] == "replay"
    assert result["score_start_index"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "ETHUSDT"),
        ("timeframe", "M15"),
        ("local_source", "mt5"),
    ],
)
def test_strategy_data_identity_mismatch_is_rejected(field: str, value) -> None:
    strategy = _strategy()
    strategy[field] = value
    match = "来源族" if field == "local_source" else field
    with pytest.raises(ValueError, match=match):
        _validate_strategy_data_contract(strategy, _manager())


def test_backtest_uses_evaluation_annualization_without_rejecting_old_strategy_basis() -> None:
    strategy = _strategy()
    strategy["periods_per_year"] = 968
    strategy["data_end"] = "2023-12-01T00:00:00Z"
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    result = _validate_strategy_data_contract(strategy, manager)

    assert result["annualization"] == {
        "basis": "evaluation_data",
        "training_periods_per_year": 968,
        "evaluation_periods_per_year": 6240,
        "same_periods_per_year": False,
    }


@pytest.mark.parametrize(
    "field",
    [
        "symbol",
        "timeframe",
        "local_source",
        "periods_per_year",
        "minimum_bars",
        "data_sha256",
        "dataset_id",
        "data_rows",
        "data_start",
        "data_end",
    ],
)
def test_legacy_strategy_missing_identity_is_rejected(field: str) -> None:
    strategy = _strategy()
    strategy.pop(field)
    with pytest.raises(ValueError, match=field):
        _validate_strategy_data_contract(strategy, _manager())


@pytest.mark.parametrize("field", ["periods_per_year", "minimum_bars"])
def test_integer_identity_rejects_bool(field: str) -> None:
    strategy = _strategy()
    strategy[field] = True
    with pytest.raises(ValueError, match="必须是整数"):
        _validate_strategy_data_contract(strategy, _manager())


def test_strategy_missing_columns_is_rejected_before_backtest() -> None:
    strategy = _strategy()
    strategy.pop("columns")

    with pytest.raises(ValueError, match="columns"):
        _validate_strategy_data_contract(strategy, _manager())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_rows", 3999),
        ("data_start", "2023-11-15T22:13:20Z"),
        ("data_end", "2024-04-28T13:13:20Z"),
        ("periods_per_year", 968),
        ("minimum_bars", 2999),
        ("local_source", "okx_legacy_attested"),
    ],
)
def test_same_hash_requires_exact_loader_metadata(field: str, value) -> None:
    strategy = _strategy()
    strategy[field] = value

    with pytest.raises(ValueError, match=field):
        _validate_strategy_data_contract(strategy, _manager())


def test_strategy_training_range_must_be_ordered() -> None:
    strategy = _strategy()
    strategy["data_start"] = strategy["data_end"]

    with pytest.raises(ValueError, match="data_start < data_end"):
        _validate_strategy_data_contract(strategy, _manager())


def test_different_hash_uses_post_training_bars_as_oos_score_window() -> None:
    strategy = _strategy()
    strategy["data_end"] = "2023-12-01T00:00:00Z"
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    result = _validate_strategy_data_contract(strategy, manager)

    assert result["evaluation_mode"] == "out_of_sample"
    assert result["same_dataset"] is False
    assert result["score_start_index"] > 0
    assert result["score_start"] > strategy["data_end"]


def test_oos_requires_semantically_compatible_columns() -> None:
    strategy = _strategy()
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"
    manager.columns = ["time", "open", "high", "low", "close", "volume"]

    with pytest.raises(ValueError, match="columns"):
        _validate_strategy_data_contract(strategy, manager)


def test_replay_rejects_different_evaluation_hash() -> None:
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    with pytest.raises(ValueError, match="重放"):
        _validate_strategy_data_contract(
            _strategy(),
            manager,
            evaluation_mode="replay",
        )


def test_diagnostic_overlap_is_explicitly_labeled() -> None:
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    result = _validate_strategy_data_contract(
        _strategy(),
        manager,
        evaluation_mode="diagnostic_overlap",
    )

    assert result["evaluation_mode"] == "diagnostic_overlap"
    assert result["score_start_index"] == 0
