"""一次性 Slurm checkpoint 导入器的纯合同测试。"""
from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from scripts import slurm_checkpoint_import as importer


def test_checkpoint_path_binds_symbol_timeframe_data_and_step() -> None:
    data_sha256 = "6" * 64
    value = (
        f"checkpoints/D1/{data_sha256}/run_01785293593994423452/"
        "ckpt_000617_step_2760.pt"
    )

    assert importer._validate_checkpoint_path(
        value,
        symbol="000617",
        timeframe="D1",
        data_sha256=data_sha256,
        step=2760,
    ) == PurePosixPath(value)

    with pytest.raises(importer.ImportError, match="训练身份"):
        importer._validate_checkpoint_path(
            value,
            symbol="000617",
            timeframe="D1",
            data_sha256=data_sha256,
            step=2761,
        )


@pytest.mark.parametrize(
    "value",
    (
        "/checkpoints/D1/file.pt",
        "../checkpoints/D1/file.pt",
        "checkpoints/D1/../file.pt",
        r"checkpoints\D1\file.pt",
    ),
)
def test_checkpoint_path_rejects_noncanonical_or_escaping_paths(
    value: str,
) -> None:
    with pytest.raises(importer.ImportError):
        importer._validate_checkpoint_path(
            value,
            symbol="000617",
            timeframe="D1",
            data_sha256="6" * 64,
            step=2760,
        )


def test_copy_checkpoint_closes_source_directory_when_target_open_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = 0
    closed: list[int] = []

    def fake_open_chain(*_args, **_kwargs):
        nonlocal opened
        opened += 1
        if opened == 1:
            return 41
        raise OSError("target unavailable")

    monkeypatch.setattr(importer, "_open_directory_chain", fake_open_chain)
    monkeypatch.setattr(importer.os, "close", closed.append)

    with pytest.raises(importer.ImportError, match="无法安全导入"):
        importer._copy_checkpoint(
            tmp_path / "source",
            tmp_path / "target",
            PurePosixPath(
                "checkpoints/D1/"
                + "6" * 64
                + "/run_01785293593994423452/"
                "ckpt_000617_step_2760.pt"
            ),
            expected_size=123,
            expected_sha256="5" * 64,
        )

    assert closed == [41]


def test_main_rejects_wrong_argument_count(capsys) -> None:
    assert importer.main([]) == 2
    assert '"ok": false' in capsys.readouterr().out


def test_import_rejects_parent_job_without_remote_node_fail(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_run_id = "run_20260723T235959Z_867bfc69"
    fake_control = SimpleNamespace(
        _load_binding=lambda run_id, job_id: (
            tmp_path / run_id,
            {"run_id": run_id, "job_id": job_id},
        ),
        SLURM_BIN=tmp_path / "slurm",
        _job_name=lambda run_id: f"alphamaster_{run_id}",
        _base_state=lambda state: state,
        _run_slurm=lambda _args: SimpleNamespace(
            stdout=(
                "581389|test-user|"
                f"alphamaster_{parent_run_id}|COMPLETED|0:0|"
                "2026-07-29T10:52:50|2026-07-29T12:50:09|"
                "01:57:19|12|cu19|\n"
            )
        ),
    )
    monkeypatch.setattr(importer, "_load_control", lambda _root: fake_control)
    monkeypatch.setattr(importer.getpass, "getuser", lambda: "test-user")

    with pytest.raises(importer.ImportError, match="NODE_FAIL"):
        importer.import_checkpoint(
            str(tmp_path.resolve()),
            "run_20260729T130000Z_1234abcd",
            parent_run_id,
            "581389",
            (
                "checkpoints/D1/"
                + "6" * 64
                + "/run_01785293593994423452/"
                "ckpt_000617_step_2760.pt"
            ),
            "5" * 64,
            "5965894",
            "2760",
        )


def test_parent_node_fail_requires_unique_matching_sacct_row(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_run_id = "run_20260723T235959Z_867bfc69"
    observed: list[list[str]] = []

    def fake_run(args):
        observed.append(list(args))
        return SimpleNamespace(
            stdout=(
                "581389.batch|test-user|batch|CANCELLED|0:15|"
                "2026-07-29T10:52:50|2026-07-29T12:50:09|"
                "01:57:19|12|cu19|\n"
                "581389|test-user|"
                f"alphamaster_{parent_run_id}|NODE_FAIL|0:0|"
                "2026-07-29T10:52:50|2026-07-29T12:50:09|"
                "01:57:19|12|cu19|\n"
            )
        )

    control = SimpleNamespace(
        SLURM_BIN=tmp_path / "slurm",
        _job_name=lambda run_id: f"alphamaster_{run_id}",
        _base_state=lambda state: state.split()[0].rstrip("+").upper(),
        _run_slurm=fake_run,
    )
    monkeypatch.setattr(importer.getpass, "getuser", lambda: "test-user")

    importer._require_parent_node_fail(
        control,
        parent_run_id,
        "581389",
    )
    assert observed == [
        [
            str(tmp_path / "slurm" / "sacct"),
            "-n",
            "-P",
            "-X",
            "-j",
            "581389",
            (
                "--format=JobIDRaw,User,JobName,State,ExitCode,Start,End,"
                "Elapsed,AllocCPUS,NodeList,MaxRSS"
            ),
        ]
    ]


@pytest.mark.parametrize(
    ("stdout", "message"),
    (
        (
            (
                "581389|test-user|alphamaster_run_20260723T235959Z_867bfc69|"
                "NODE_FAIL|0:0|start|end|01:57:19|12|cu19|\n"
            )
            * 2,
            "唯一完整终态",
        ),
        (
            (
                "581389.batch|test-user|batch|CANCELLED|0:15|"
                "start|end|01:57:19|12|cu19|\n"
            ),
            "唯一完整终态",
        ),
        (
            (
                "581389|other-user|alphamaster_run_20260723T235959Z_867bfc69|"
                "NODE_FAIL|0:0|start|end|01:57:19|12|cu19|\n"
            ),
            "明确的 NODE_FAIL",
        ),
        (
            (
                "581389|test-user|alphamaster_wrong|NODE_FAIL|0:0|"
                "start|end|01:57:19|12|cu19|\n"
            ),
            "明确的 NODE_FAIL",
        ),
        (
            (
                "581389|test-user|alphamaster_run_20260723T235959Z_867bfc69|"
                "NODE_FAIL|0:0|start|end|01:57:19|12|cu19|extra|\n"
            ),
            "唯一完整终态",
        ),
    ),
)
def test_parent_node_fail_rejects_ambiguous_or_mismatched_sacct(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    message: str,
) -> None:
    parent_run_id = "run_20260723T235959Z_867bfc69"
    control = SimpleNamespace(
        SLURM_BIN=tmp_path / "slurm",
        _job_name=lambda run_id: f"alphamaster_{run_id}",
        _base_state=lambda state: state.split()[0].rstrip("+").upper(),
        _run_slurm=lambda _args: SimpleNamespace(stdout=stdout),
    )
    monkeypatch.setattr(importer.getpass, "getuser", lambda: "test-user")

    with pytest.raises(importer.ImportError, match=message):
        importer._require_parent_node_fail(
            control,
            parent_run_id,
            "581389",
        )
