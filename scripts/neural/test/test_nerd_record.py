# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end capture contracts for integrating and free-acceleration models."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

from newton_neural.data import TrajectoryDataset
from newton_neural.models import NeRDBase

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/test/neural"))
from nerd_test_utils import TASKS, cuda_device, record_dataset

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@pytest.mark.parametrize("task", TASKS)
def test_dataset(task, tmp_path):
    dataset_path = tmp_path / "dataset"
    record_dataset(task, dataset_path, 8, 128, cuda_device())
    dataset = TrajectoryDataset(dataset_path)
    nerd = NeRDBase.load(dataset_path / "model.pt", device="cpu")
    # Existing RecorderManager exports independent episodes; native worlds identify those episodes.
    assert dataset.world_count >= 8
    assert dataset.signals == nerd.signals
    assert not nerd.fitted
    assert "physical_model" in nerd.metadata["isaaclab"]
    assert dataset.frame_dt > 0
    usable_frames = 0
    for shard in dataset.shards:
        valid = np.load(dataset_path / shard["path"] / "valid.npy")
        starts = np.load(dataset_path / shard["path"] / "episode_start.npy")
        assert starts[:, 0].all()
        assert not starts[:, 1:].any()
        usable_frames += valid.sum()
    # Native training windows independently validate field layouts and episode boundaries.
    from newton_neural.data import TrajectoryWindows

    windows = TrajectoryWindows(dataset_path, 0 if not nerd.prediction.integrates else 16, device="cpu")
    assert len(windows) > 0
    assert usable_frames >= 0.9 * 8 * 128
