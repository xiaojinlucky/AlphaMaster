from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

import web.app as app_module
from data_pipeline.legacy_mt5_registry import build_single_file_registration_plan
from model_core.target_contract import SCORING_CONTRACT_VERSION
from model_core.vocab import VOCAB_VERSION


BASE_URL = "http://127.0.0.1:8765"
ORIGIN_HEADERS = {"Origin": BASE_URL}


def _control_headers(client: TestClient) -> dict[str, str]:
    response = client.get("/api/session", headers=ORIGIN_HEADERS)
    token = response.json()["control_token"]
    return {**ORIGIN_HEADERS, "X-AlphaMaster-Control": token}


def _write_parquet(path: Path, *, start: int = 1_700_000_000, rows: int = 3000) -> None:
    frame = pd.DataFrame(
        {
            "time": [start + index * 3600 for index in range(rows)],
            "open": [100.0 + index / 1000 for index in range(rows)],
            "high": [101.0 + index / 1000 for index in range(rows)],
            "low": [99.0 + index / 1000 for index in range(rows)],
            "close": [100.5 + index / 1000 for index in range(rows)],
            "tick_volume": [100 + index for index in range(rows)],
        }
    )
    frame.to_parquet(path, index=False)


def test_strategy_browse_rejects_old_scoring_contract_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "best_XAUUSD.json"
    path.write_text(
        json.dumps(
            {
                "vocab_version": VOCAB_VERSION,
                "scoring_contract_version": "previous_contract",
                "symbol": "XAUUSD",
                "formula": [1, 2, 3],
                "best_score": 9.99,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "pick_strategy_file", lambda: str(path))
    monkeypatch.setattr(
        app_module,
        "save_settings",
        lambda *_args, **_kwargs: pytest.fail("旧评分合同不得保存为最近策略"),
    )

    with pytest.raises(HTTPException) as exc_info:
        app_module._browse_strategy_file()

    assert exc_info.value.status_code == 400
    assert "不能正式回测" in str(exc_info.value.detail)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app_module.app, base_url=BASE_URL, raise_server_exceptions=False)


def test_web_hides_previous_contract_backtest_outputs(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "BACKTEST_OUTPUT_DIR", tmp_path)
    (tmp_path / "multi_factor_report.json").write_text(
        json.dumps(
            {
                "scoring_contract_version": "previous_contract",
                "symbols": {"XAUUSD": {"sharpe": 9.99}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "equity_curve.json").write_text(
        json.dumps(
            {
                "scoring_contract_version": "previous_contract",
                "symbols": {"XAUUSD": {"equity": [9.99]}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "portfolio_equity.png").write_bytes(b"legacy")

    headers = _control_headers(client)
    report = client.get(
        "/api/backtest/report?symbol=XAUUSD",
        headers=headers,
    ).json()
    equity = client.get(
        "/api/backtest/equity?symbol=XAUUSD",
        headers=headers,
    ).json()
    chart = client.get(
        "/api/backtest/chart/portfolio_equity.png",
        headers=headers,
    )

    assert report["available"] is False
    assert report["charts"] == []
    assert equity["available"] is False
    assert chart.status_code == 404


def test_web_serves_only_current_contract_backtest_outputs(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "BACKTEST_OUTPUT_DIR", tmp_path)
    chart_content = b"current"
    equity_content = json.dumps(
        {
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "symbols": {"XAUUSD": {"equity": [0.0, 0.1]}},
        }
    ).encode()
    (tmp_path / "multi_factor_report.json").write_text(
        json.dumps(
            {
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "chart_artifacts": [
                    {
                        "name": "portfolio_equity.png",
                        "sha256": hashlib.sha256(chart_content).hexdigest(),
                        "label": "组合资金曲线",
                        "kind": "portfolio",
                    }
                ],
                "equity_artifact": {
                    "name": "equity_curve.json",
                    "sha256": hashlib.sha256(equity_content).hexdigest(),
                },
                "symbols": {"XAUUSD": {"sharpe": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "equity_curve.json").write_bytes(equity_content)
    (tmp_path / "portfolio_equity.png").write_bytes(chart_content)
    (tmp_path / "equity_old_contract.png").write_bytes(b"legacy")

    headers = _control_headers(client)
    report = client.get(
        "/api/backtest/report?symbol=XAUUSD",
        headers=headers,
    ).json()
    equity = client.get(
        "/api/backtest/equity?symbol=XAUUSD",
        headers=headers,
    ).json()
    chart = client.get(
        "/api/backtest/chart/portfolio_equity.png",
        headers=headers,
    )
    stale_chart = client.get(
        "/api/backtest/chart/equity_old_contract.png",
        headers=headers,
    )

    assert report["available"] is True
    assert [row["name"] for row in report["charts"]] == ["portfolio_equity.png"]
    assert equity["available"] is True
    assert chart.status_code == 200
    assert chart.content == chart_content
    assert stale_chart.status_code == 404

    (tmp_path / "equity_curve.json").write_bytes(b'{"stale": true}')
    (tmp_path / "portfolio_equity.png").write_bytes(b"tampered")
    tampered_equity = client.get(
        "/api/backtest/equity?symbol=XAUUSD",
        headers=headers,
    ).json()
    tampered_chart = client.get(
        "/api/backtest/chart/portfolio_equity.png",
        headers=headers,
    )

    assert tampered_equity["available"] is False
    assert tampered_chart.status_code == 404


def test_slurm_preflight_marks_bare_file_untrainable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: (_ for _ in ()).throw(
            AssertionError("/api/config 不得等待远端训练状态")
        ),
    )

    info = app_module._inspect_or_http(
        str(data),
        include_registration_plan=True,
    )

    assert info["valid"] is True
    assert info["registration"] == "bare_legacy"
    assert info["capabilities"]["training"] is False
    assert info["reason_code"] == "manifest_required_for_slurm"
    assert len(info["registration_plan_sha256"]) == 64


def test_local_preflight_allows_same_bare_file_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    monkeypatch.setenv("TRAINING_BACKEND", "local")

    info = app_module._inspect_or_http(str(data))

    assert info["registration"] == "bare_legacy"
    assert info["capabilities"]["training"] is True
    assert "registration_plan_sha256" not in info


def test_single_file_registration_endpoint_refreshes_training_capability(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    plan = build_single_file_registration_plan(data)
    app_module._REGISTRATION_PLANS[plan["plan_sha256"]] = plan
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: {"backend": "slurm", "active": False, "job": None},
    )
    monkeypatch.setattr(app_module, "save_settings", lambda _payload: {})
    monkeypatch.setattr(
        app_module,
        "write_registration_report",
        lambda path, _report: Path(path),
    )

    response = client.post(
        "/api/data-file/register-legacy-mt5",
        headers={**_control_headers(client), "Content-Type": "application/json"},
        json={
            "data_file": str(data),
            "plan_sha256": plan["plan_sha256"],
            "source_acknowledgement": "MetaTrader5",
        },
    )

    assert response.status_code == 200
    payload = response.json()["data_file"]
    assert payload["source"] == "mt5_legacy_attested"
    assert payload["capabilities"]["training"] is True
    assert data.with_suffix(".manifest.json").is_file()


def test_config_exposes_registration_plan_for_saved_bare_mt5_file(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "NVDA_M5.parquet"
    _write_parquet(data)
    monkeypatch.setattr(
        app_module,
        "load_settings",
        lambda: {
            "last_data_file": str(data),
            "last_backtest_data_file": "",
            "last_strategy_file": "",
            "debug_mode": False,
            "ai_provider": "deepseek",
            "ai_api_key": "",
            "bt_commission_pct": 0.02,
            "bt_slippage_pct": 0.01,
        },
    )
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: {"backend": "slurm", "active": False, "job": None},
    )

    response = client.get("/api/config", headers=_control_headers(client))

    assert response.status_code == 200
    data_file = response.json()["data_file"]
    assert data_file["capabilities"]["training"] is False
    assert len(data_file["registration_plan_sha256"]) == 64


def test_read_only_training_apis_use_cached_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = {
        "backend": "slurm",
        "active": True,
        "job": {
            "run_id": "run_20260724T010000Z_1234abcd",
            "symbol": "000333",
            "remote_state": "RUNNING",
            "training_parameters": {"train_steps": 200},
        },
    }
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: (_ for _ in ()).throw(
            AssertionError("只读 API 不得等待远端训练状态")
        ),
    )
    monkeypatch.setattr(
        app_module.training_manager,
        "tail_log",
        lambda _lines=200, **_kwargs: (_ for _ in ()).throw(
            AssertionError("只读 API 不得拉取远端日志")
        ),
    )
    monkeypatch.setattr(
        app_module.training_manager,
        "snapshot",
        lambda: copy.deepcopy(cached),
    )
    cached_log_calls: list[str | None] = []

    def cached_log_tail(
        _lines: int = 200,
        *,
        run_id: str | None = None,
    ) -> list[str]:
        cached_log_calls.append(run_id)
        return ["[10/200] cached"]

    monkeypatch.setattr(
        app_module.training_manager,
        "cached_log_tail",
        cached_log_tail,
    )
    monkeypatch.setattr(
        app_module.training_manager,
        "parse_step_from_log",
        lambda *, refresh_remote=True, run_id=None: 10
        if refresh_remote is False
        else (_ for _ in ()).throw(
            AssertionError("概览不得刷新远端日志")
        ),
    )
    monkeypatch.setattr(
        app_module,
        "load_settings",
        lambda: {"last_data_file": ""},
    )
    pipeline_snapshot_calls: list[str | None] = []

    def pipeline_snapshot(run_id: str | None) -> dict[str, str]:
        pipeline_snapshot_calls.append(run_id)
        return {"status": "WAITING_TRAINING"}

    monkeypatch.setattr(
        app_module.a_share_pipeline_manager,
        "snapshot",
        pipeline_snapshot,
    )
    monkeypatch.setattr(
        app_module.a_share_pipeline_manager,
        "observe",
        lambda _training: (_ for _ in ()).throw(
            AssertionError("只读 API 不得推进流水线状态")
        ),
    )

    overview = app_module.api_overview()
    training = app_module.api_training_status()
    pipeline = app_module.api_pipeline_status()

    assert overview["training"]["job"]["symbol"] == "000333"
    assert training["log_tail"] == ["[10/200] cached"]
    assert pipeline["pipeline"]["status"] == "WAITING_TRAINING"
    assert cached_log_calls == ["run_20260724T010000Z_1234abcd"]
    assert pipeline_snapshot_calls == [
        "run_20260724T010000Z_1234abcd",
        "run_20260724T010000Z_1234abcd",
        "run_20260724T010000Z_1234abcd",
    ]


def test_backtest_endpoint_accepts_independent_future_dataset(
    tmp_path: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = tmp_path / "NVDA_H1.parquet"
    _write_parquet(evaluation, start=1_704_067_200)
    strategy_path = tmp_path / "best_NVDA.json"
    strategy = {
        "vocab_version": VOCAB_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "symbol": "NVDA",
        "timeframe": "H1",
        "formula": [0],
        "best_score": 1.0,
        "local_source": "local_file",
        "periods_per_year": 6240,
        "minimum_bars": 3000,
        "dataset_id": f"sha256:{'a' * 64}",
        "data_sha256": "a" * 64,
        "data_rows": 3000,
        "data_start": "2023-01-01T00:00:00Z",
        "data_end": "2023-12-31T23:00:00Z",
        "columns": ["time", "open", "high", "low", "close", "tick_volume"],
    }
    strategy_path.write_text(
        json.dumps(strategy, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app_module,
        "load_settings",
        lambda: {
            "last_backtest_data_file": "",
            "last_data_file": "",
            "bt_commission_pct": 0.02,
            "bt_slippage_pct": 0.01,
        },
    )
    monkeypatch.setattr(app_module, "save_settings", lambda _payload: {})
    monkeypatch.setattr(
        app_module.training_manager,
        "status",
        lambda: {"backend": "local", "active": False, "job": None},
    )

    captured: dict = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"state": "running", **kwargs})

    monkeypatch.setattr(app_module.backtest_manager, "start", fake_start)

    response = client.post(
        "/api/backtest/start",
        headers={**_control_headers(client), "Content-Type": "application/json"},
        json={
            "strategy_file": str(strategy_path),
            "data_file": str(evaluation),
            "evaluation_mode": "auto",
            "commission_pct": 0.02,
            "slippage_pct": 0.01,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluation"]["evaluation_mode"] == "out_of_sample"
    assert payload["data_resolution"] == "explicit_evaluation"
    assert captured["data_file"] == str(evaluation.resolve())
    assert captured["evaluation_mode"] == "auto"
    assert hashlib.sha256(evaluation.read_bytes()).hexdigest() == (
        payload["evaluation"]["evaluation_data"]["data_sha256"]
    )
