from __future__ import annotations

import train_file
from data_pipeline.dataset_contracts import DATA_SOURCE_IDS, TRAINING_SOURCE_IDS
from scripts import slurm_control, train_slurm_worker
from web import training_package


def test_all_training_entrypoints_share_one_source_id_contract() -> None:
    parser = train_file.build_arg_parser()
    source_action = next(
        action for action in parser._actions if action.dest == "data_source"
    )

    assert set(source_action.choices) == set(TRAINING_SOURCE_IDS)
    assert slurm_control.ALLOWED_LOCAL_SOURCES == TRAINING_SOURCE_IDS
    assert train_slurm_worker.ALLOWED_LOCAL_SOURCES == TRAINING_SOURCE_IDS
    assert training_package._LOCAL_SOURCES == DATA_SOURCE_IDS
    assert "data_pipeline/dataset_contracts.py" in (
        slurm_control.REQUIRED_SOURCE_FILES
    )
    assert "data_pipeline/dataset_contracts.py" in (
        train_slurm_worker.REQUIRED_SOURCE_FILES
    )
