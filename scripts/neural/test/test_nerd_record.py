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

from isaaclab_neural.nerd import build_model
from newton_neural.data import TrajectoryDataset, TrajectoryWindows
from newton_neural.systems import SystemBase

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/test/neural"))
from nerd_test_utils import ANYMAL_C_FLAT, FRANKA_REACH, TASKS, cuda_device, record_dataset

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

WORLD_COUNT = 8


@pytest.mark.parametrize("task", TASKS)
def test_dataset(task, tmp_path):
    """Each environment records one contiguous stream whose episodes last their declared number of frames."""
    dataset_path, frame_count = tmp_path / "dataset", 128
    record_dataset(task, dataset_path, WORLD_COUNT, frame_count, cuda_device())
    dataset = TrajectoryDataset(dataset_path)
    assert (dataset.world_count, dataset.frame_count) == (WORLD_COUNT, frame_count)
    assert dataset.generation["complete"]
    assert dataset.frame_dt > 0
    # The dataset describes its robot, and holds the signals of the model that the task defines on it.
    nerd = build_model(dataset, load_cfg_from_registry(task, "nerd_cfg_entry_point"))
    assert dataset.signals == nerd.signals
    assert nerd.metadata["isaaclab"]["physical_model"] == dataset.generation["isaaclab"]["physical_model"]

    arrays = dataset.arrays(0)
    starts, valid = arrays["episode_start"], arrays["valid"]
    assert starts[:, 0].all()
    assert valid.mean() >= 0.9
    exploration = load_cfg_from_registry(task, "nerd_recording_cfg_entry_point").actions.joint_targets
    shortest, longest = exploration.episode_frames
    for world in range(WORLD_COUNT):
        first_frames = np.flatnonzero(starts[world])
        for first, following in zip(first_frames[:-1], first_frames[1:], strict=True):
            # A divergent state ends an episode early and invalidates its last frame.
            timed_out = valid[world, following - 1]
            assert following - first <= longest
            assert following - first >= shortest or not timed_out
    # Native training windows independently validate field layouts and episode boundaries.
    windows = TrajectoryWindows(dataset_path, 16 if nerd.prediction.integrates else 0, device="cpu")
    assert len(windows) > 0


@pytest.mark.parametrize("task", [FRANKA_REACH, ANYMAL_C_FLAT])
def test_frames_follow_environment_steps(task, tmp_path, monkeypatch):
    """Frame t of an environment holds its state before step t and the joint targets applied during that step.

    Shards of 20 frames and episodes of 5 to 9 frames put shard boundaries and restarts inside the 48 recorded
    steps. Isaac Lab's own view of the robot and of the action term is the reference.
    """
    live = {"joint_pos": [], "targets": [], "starting": [], "terminated": []}
    step = ManagerBasedRLEnv.step

    def observed_step(env, action):
        robot, exploration = env.scene["robot"], env.action_manager.get_term("joint_targets")
        live["joint_names"] = robot.joint_names
        live["starting"].append((env.episode_length_buf == 0).cpu().numpy())
        live["joint_pos"].append(robot.data.joint_pos.torch.cpu().numpy().copy())
        result = step(env, action)
        live["targets"].append(exploration.processed_actions.cpu().numpy().copy())
        live["terminated"].append(result[2].cpu().numpy().copy())
        return result

    monkeypatch.setattr(ManagerBasedRLEnv, "step", observed_step)
    dataset_path = tmp_path / "dataset"
    overrides = ("env.recorders.dynamics.shard_frames=20", "env.actions.joint_targets.episode_frames=[5,9]")
    record_dataset(task, dataset_path, WORLD_COUNT, 48, cuda_device(), *overrides)
    dataset = TrajectoryDataset(dataset_path)
    # Every shard continues each environment's stream, and closing writes the frames that remain.
    assert [(shard["frame_count"], shard["continues"]) for shard in dataset.shards] == [
        (20, False),
        (20, True),
        (8, True),
    ]
    shards = [dataset.arrays(index) for index in range(len(dataset.shards))]
    recorded = {name: np.concatenate([shard[name] for shard in shards], axis=1) for name in shards[0]}
    expected = {name: np.stack(values, axis=1) for name, values in live.items() if name != "joint_names"}

    # The recorded signals follow the joint order of the recorded system, Isaac Lab's the order of the asset.
    labels = SystemBase.create(**dataset.generation["isaaclab"]["system"]).spec.actuated_joint_labels
    order = [labels.index(name) for name in live["joint_names"]]
    joint_q = recorded["joint_q"][..., -len(labels) :]  # A floating base precedes the joint coordinates.
    assert np.array_equal(joint_q[..., order], expected["joint_pos"])
    assert np.array_equal(recorded["joint_target_q"][..., order], expected["targets"])
    assert np.array_equal(recorded["episode_start"], expected["starting"])
    assert recorded["episode_start"][:, 1:].any()
    assert np.array_equal(recorded["valid"], ~expected["terminated"])
