# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training builds the task's model on the robot that a recorded dataset describes."""

import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

from newton_neural.data import TrajectoryDataset
from newton_neural.models import NeRDBase

from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/test/neural"))
from nerd_test_utils import FRANKA_REACH, cuda_device, record_dataset

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small recording of the reach task, taken once for all tests of this module."""
    dataset = tmp_path_factory.mktemp("reach_franka") / "dataset"
    record_dataset(FRANKA_REACH, dataset, 8, 128, cuda_device())
    return dataset


def _fit(dataset: Path, checkpoint: Path, *overrides: str) -> NeRDBase:
    """Train the model of the reach task on ``dataset`` for two steps with ``train.py`` and load ``checkpoint``."""
    argv = ["--task", FRANKA_REACH, "--dataset", str(dataset), "--checkpoint", str(checkpoint)]
    argv += ["--device", cuda_device(), "--max_steps", "2", "agent.batch_size=4", "agent.horizons=[1]", *overrides]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))["main"](argv)
    return NeRDBase.load(checkpoint, device="cpu")


def test_one_dataset_fits_any_architecture(dataset, tmp_path, capsys):
    """One recording trains the task's architecture and another one, and refuses other recorded inputs."""
    recorded = TrajectoryDataset(dataset).generation["isaaclab"]

    def fit(name: str, *overrides: str) -> NeRDBase:
        return _fit(dataset, tmp_path / "output" / name, *overrides)

    declared = fit("declared.pt")
    assert declared.fitted
    assert declared.backbone.spans == tuple(load_cfg_from_registry(FRANKA_REACH, "nerd_cfg_entry_point").backbone_spans)
    # Deployment checks the live robot against the contract that the recording measured.
    assert declared.metadata["isaaclab"]["physical_model"] == recorded["physical_model"]

    narrow = fit("narrow.pt", "model.backbone_spans=[2,2]")
    assert narrow.fitted
    assert narrow.backbone.spans == (2, 2)
    assert narrow.metadata["isaaclab"]["model_cfg"]["backbone_spans"] == [2, 2]

    # The contact input selects recorded signals, so data without it cannot train a model that reads it.
    with pytest.raises(SystemExit):
        fit("contact.pt", "model.contact_bodies=panda_hand")
    assert "record the dataset again" in capsys.readouterr().err
    assert not (tmp_path / "output" / "contact.pt").exists()


def test_training_without_triton(dataset, tmp_path, monkeypatch, caplog):
    """Without Triton, which PyTorch installs on Linux only, the script trains and reports without ``torch.compile``."""
    monkeypatch.setitem(sys.modules, "triton", None)  # Importing Triton fails, as where it is not installed
    assert _fit(dataset, tmp_path / "eager.pt").fitted
    assert "Triton is not installed" in caplog.text
