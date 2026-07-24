from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from run_backtest import (
    _build_sealed_report_payload,
    _validate_sealed_report_cli,
    _validate_strategy_data_contract,
    _write_sealed_report_atomic,
)


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


def test_sealed_oos_requires_explicit_post_training_score_start() -> None:
    strategy = _strategy()
    strategy["data_end"] = "2023-12-01T00:00:00Z"
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    with pytest.raises(ValueError, match="显式提供 score_start"):
        _validate_strategy_data_contract(
            strategy,
            manager,
            evaluation_mode="sealed_oos",
        )


def test_sealed_oos_rejects_training_data_hash() -> None:
    with pytest.raises(ValueError, match="hash 与训练数据不同"):
        _validate_strategy_data_contract(
            _strategy(),
            _manager(),
            evaluation_mode="sealed_oos",
            score_start="2024-05-01T00:00:00Z",
        )


def test_sealed_oos_score_start_must_be_strictly_after_training_end() -> None:
    strategy = _strategy()
    strategy["data_end"] = "2023-12-01T00:00:00Z"
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    with pytest.raises(ValueError, match="必须晚于训练数据结束时间"):
        _validate_strategy_data_contract(
            strategy,
            manager,
            evaluation_mode="sealed_oos",
            score_start=strategy["data_end"],
        )


def test_sealed_oos_preserves_mode_and_actual_score_window() -> None:
    strategy = _strategy()
    strategy["data_end"] = "2023-12-01T00:00:00Z"
    manager = _manager()
    manager.data_sha256 = "b" * 64
    manager.dataset_id = f"sha256:{manager.data_sha256}"

    result = _validate_strategy_data_contract(
        strategy,
        manager,
        evaluation_mode="sealed_oos",
        score_start="2024-01-01T00:15:00Z",
    )

    assert result["evaluation_mode"] == "sealed_oos"
    assert result["same_dataset"] is False
    assert result["score_start"] == "2024-01-01T01:13:20Z"
    assert result["score_end"] == manager.data_end


@pytest.mark.parametrize(
    ("evaluation_mode", "strategy_file", "sealed_report", "single_mode", "match"),
    [
        ("sealed_oos", "strategy.json", None, False, "--sealed-report"),
        ("out_of_sample", "strategy.json", "sealed.json", False, "仅允许"),
        ("sealed_oos", None, "sealed.json", False, "--strategy-file"),
        ("sealed_oos", "strategy.json", "sealed.json", True, "不接受 --single"),
    ],
)
def test_sealed_report_cli_rejects_invalid_combinations(
    evaluation_mode: str,
    strategy_file: str | None,
    sealed_report: str | None,
    single_mode: bool,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_sealed_report_cli(
            evaluation_mode=evaluation_mode,
            strategy_file=strategy_file,
            sealed_report=sealed_report,
            single_mode=single_mode,
        )


def test_sealed_report_cli_accepts_explicit_single_strategy_mode() -> None:
    result = _validate_sealed_report_cli(
        evaluation_mode="sealed_oos",
        strategy_file="strategy.json",
        sealed_report="sealed.json",
        single_mode=False,
    )

    assert result == Path("sealed.json")


def test_sealed_report_requires_exactly_one_symbol_result() -> None:
    with pytest.raises(ValueError, match="正好一个品种结果"):
        _build_sealed_report_payload(
            results_map={
                "BTCUSDT": {"sharpe": 1.1, "cost_rate": 0.0003},
                "ETHUSDT": {"sharpe": 1.2, "cost_rate": 0.0003},
            },
            evaluation_contract={
                "evaluation_mode": "sealed_oos",
                "score_start": "2024-01-01T00:00:00Z",
                "score_end": "2024-12-31T00:00:00Z",
            },
            data_sha256="b" * 64,
            strategy_bytes=b"strategy",
            commission_pct=0.02,
            slippage_pct=0.01,
        )


def test_sealed_report_has_exact_fields_and_raw_strategy_hash() -> None:
    strategy_bytes = b'{\r\n  "formula": [1, 2, 3]\r\n}\r\n'

    payload = _build_sealed_report_payload(
        results_map={
            "BTCUSDT": {"sharpe": 1.23456789, "cost_rate": 0.0003}
        },
        evaluation_contract={
            "evaluation_mode": "sealed_oos",
            "score_start": "2024-01-01T00:00:00Z",
            "score_end": "2024-12-31T00:00:00Z",
        },
        data_sha256="b" * 64,
        strategy_bytes=strategy_bytes,
        commission_pct=0.02,
        slippage_pct=0.01,
    )

    assert set(payload) == {
        "format",
        "symbol",
        "data_sha256",
        "strategy_sha256",
        "evaluation_mode",
        "test_start",
        "test_end",
        "commission_pct",
        "slippage_pct",
        "cost_rate",
        "sharpe",
    }
    assert payload == {
        "format": "alphamaster_sealed_oos_report_v2",
        "symbol": "BTCUSDT",
        "data_sha256": "b" * 64,
        "strategy_sha256": hashlib.sha256(strategy_bytes).hexdigest(),
        "evaluation_mode": "sealed_oos",
        "test_start": "2024-01-01T00:00:00Z",
        "test_end": "2024-12-31T00:00:00Z",
        "commission_pct": 0.02,
        "slippage_pct": 0.01,
        "cost_rate": 0.0003,
        "sharpe": 1.23456789,
    }


def test_sealed_report_write_is_atomic_and_never_overwrites(
    tmp_path: Path,
) -> None:
    target = tmp_path / "reports" / "sealed.json"
    payload = {
        "format": "alphamaster_sealed_oos_report_v2",
        "symbol": "BTCUSDT",
        "data_sha256": "b" * 64,
        "strategy_sha256": "c" * 64,
        "evaluation_mode": "sealed_oos",
        "test_start": "2024-01-01T00:00:00Z",
        "test_end": "2024-12-31T00:00:00Z",
        "commission_pct": 0.02,
        "slippage_pct": 0.01,
        "cost_rate": 0.0003,
        "sharpe": 1.25,
    }

    _write_sealed_report_atomic(target, payload)
    original_bytes = target.read_bytes()

    assert json.loads(original_bytes) == payload
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        _write_sealed_report_atomic(target, {**payload, "sharpe": 9.9})
    assert target.read_bytes() == original_bytes
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize("sharpe", [float("nan"), float("inf"), float("-inf")])
def test_sealed_report_rejects_non_finite_sharpe(
    sharpe: float,
) -> None:
    with pytest.raises(ValueError, match="有限浮点数"):
        _build_sealed_report_payload(
            results_map={
                "BTCUSDT": {"sharpe": sharpe, "cost_rate": 0.0003}
            },
            evaluation_contract={
                "evaluation_mode": "sealed_oos",
                "score_start": "2024-01-01T00:00:00Z",
                "score_end": "2024-12-31T00:00:00Z",
            },
            data_sha256="b" * 64,
            strategy_bytes=b"strategy",
            commission_pct=0.02,
            slippage_pct=0.01,
        )


def test_sealed_report_rejects_zero_total_cost() -> None:
    with pytest.raises(ValueError, match="严格大于 0"):
        _build_sealed_report_payload(
            results_map={"BTCUSDT": {"sharpe": 1.2, "cost_rate": 0.0}},
            evaluation_contract={
                "evaluation_mode": "sealed_oos",
                "score_start": "2024-01-01T00:00:00Z",
                "score_end": "2024-12-31T00:00:00Z",
            },
            data_sha256="b" * 64,
            strategy_bytes=b"strategy",
            commission_pct=0.0,
            slippage_pct=0.0,
        )


def test_sealed_report_rejects_declared_cost_mismatch() -> None:
    with pytest.raises(ValueError, match="实际 cost_rate 不一致"):
        _build_sealed_report_payload(
            results_map={"BTCUSDT": {"sharpe": 1.2, "cost_rate": 0.0}},
            evaluation_contract={
                "evaluation_mode": "sealed_oos",
                "score_start": "2024-01-01T00:00:00Z",
                "score_end": "2024-12-31T00:00:00Z",
            },
            data_sha256="b" * 64,
            strategy_bytes=b"strategy",
            commission_pct=0.02,
            slippage_pct=0.01,
        )
