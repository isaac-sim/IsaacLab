# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training consumes the recorded model without reconstructing a deployment preset."""

import runpy
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

from newton_neural.models import NeRDBase

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/test/neural"))
from nerd_test_utils import FRANKA_REACH, cuda_device, record_dataset

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def test_recorded_model_and_explicit_output(tmp_path):
    dataset, checkpoint = tmp_path / "dataset", tmp_path / "output/model.pt"
    device = cuda_device()
    record_dataset(FRANKA_REACH, dataset, 8, 128, device)
    before = NeRDBase.load(dataset / "model.pt", device="cpu")
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))["main"](
        [
            "--task",
            FRANKA_REACH,
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--max_steps",
            "2",
            "agent.batch_size=4",
            "agent.horizons=[1]",
        ]
    )
    after = NeRDBase.load(checkpoint, device="cpu")
    assert after.fitted
    assert after.metadata == before.metadata
    assert any(not torch.equal(a, b) for a, b in zip(after.parameters(), before.parameters(), strict=True))
