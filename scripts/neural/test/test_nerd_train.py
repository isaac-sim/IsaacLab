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


def test_one_dataset_fits_any_architecture(tmp_path, capsys):
    """One recording trains the task's architecture and another one, and refuses other recorded inputs."""
    dataset, device = tmp_path / "dataset", cuda_device()
    record_dataset(FRANKA_REACH, dataset, 8, 128, device)
    recorded = TrajectoryDataset(dataset).generation["isaaclab"]

    def fit(name: str, *overrides: str) -> NeRDBase:
        checkpoint = tmp_path / "output" / name
        argv = ["--task", FRANKA_REACH, "--dataset", str(dataset), "--checkpoint", str(checkpoint)]
        argv += ["--device", device, "--max_steps", "2", "agent.batch_size=4", "agent.horizons=[1]", *overrides]
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))["main"](argv)
        return NeRDBase.load(checkpoint, device="cpu")

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
