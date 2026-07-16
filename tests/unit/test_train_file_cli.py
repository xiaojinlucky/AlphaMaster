"""train_file CLI 参数测试，不启动真实训练。"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest



def _stub_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def train_file_module(monkeypatch: pytest.MonkeyPatch):
    """隔离 CLI 解析测试，避免把训练依赖当成本测试前置条件。"""
    class FakeModelConfig:
        TRAIN_STEPS = 9000
        REWARD_MODE = "ftmo"

    packages = {
        "utils": _stub_module("utils"),
        "data_pipeline": _stub_module("data_pipeline"),
        "model_core": _stub_module("model_core"),
    }
    for package in packages.values():
        package.__path__ = []  # type: ignore[attr-defined]
    stubs = {
        **packages,
        "utils.train_logging": _stub_module(
            "utils.train_logging",
            configure_train_stdio=lambda: None,
        ),
        "config": _stub_module("config", Config=object),
        "data_pipeline.parquet_manager": _stub_module(
            "data_pipeline.parquet_manager",
            ParquetDataManager=object,
            inspect_parquet_file=lambda _path: {},
        ),
        "model_core.config": _stub_module("model_core.config", ModelConfig=FakeModelConfig),
        "model_core.engine": _stub_module("model_core.engine", AlphaEngine=object),
        "model_core.vocab": _stub_module("model_core.vocab", VOCAB_VERSION="test"),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[2] / "train_file.py"
    spec = importlib.util.spec_from_file_location("train_file_cli_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_steps_default_remains_project_default(train_file_module) -> None:
    args = train_file_module.build_arg_parser().parse_args(["--data-file", "XAUUSD_H1.parquet"])
    assert args.train_steps == train_file_module.DEFAULT_TRAIN_STEPS
    assert args.periods_per_year is None
    assert args.minimum_bars is None
    assert args.data_source is None


@pytest.mark.parametrize("value", ["0", "-1", "+1", "1.5", "１２", "abc", ""])
def test_train_steps_rejects_non_strict_positive_integer(train_file_module, value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        train_file_module._positive_int(value)


def test_main_applies_train_steps_before_training(
    train_file_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_train(
        data_file: str,
        *,
        from_scratch: bool,
        data_source: str | None,
        periods_per_year: int | None,
        minimum_bars: int | None,
    ):
        observed["data_file"] = data_file
        observed["from_scratch"] = from_scratch
        observed["periods_per_year"] = periods_per_year
        observed["minimum_bars"] = minimum_bars
        observed["data_source"] = data_source
        observed["train_steps"] = train_file_module.ModelConfig.TRAIN_STEPS
        return SimpleNamespace(
            target_symbol="XAUUSD",
            best_score=1.25,
            best_formula=None,
        )

    monkeypatch.setattr(train_file_module, "train_from_file", fake_train)

    exit_code = train_file_module.main(
        [
            "--data-file",
            "XAUUSD_H1.parquet",
            "--from-scratch",
            "--train-steps",
            "20",
            "--periods-per-year",
            "968",
            "--minimum-bars",
            "1936",
            "--data-source",
            "ashare_local",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "data_file": "XAUUSD_H1.parquet",
        "from_scratch": True,
        "periods_per_year": 968,
        "minimum_bars": 1936,
        "data_source": "ashare_local",
        "train_steps": 20,
    }


def test_main_returns_failure_when_training_fails(
    train_file_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_file_module, "train_from_file", lambda *_args, **_kwargs: None)
    assert train_file_module.main(["--data-file", "XAUUSD_H1.parquet"]) == 1
