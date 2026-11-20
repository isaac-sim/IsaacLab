# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The world-model backend: chunked prediction, joint state and resets."""

import numpy as np
import pytest

from .conftest import FakeModel, J, N, T, world_model_sim


def bound_manager(sim, num_envs=N):
    manager = sim.backend_manager
    manager.bind(num_envs, J)
    return manager


def step(manager, target):
    manager.set_joint_target(np.full((manager.num_envs, J), target, np.float32))
    manager.step()


def test_reset_starts_from_the_record(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(np.arange(N))
        assert manager.joint_state.shape == (N, J)
        assert manager.joint_state[0, 0] in (np.float32(0.1), np.float32(0.2))
        assert (manager.frame_history[1] == 201).all() and not manager.new_frames


def test_the_model_runs_once_per_chunk_and_the_joint_state_follows_the_last_target(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(np.arange(N))
        start = manager.joint_state.clone()
        for k in range(T):
            step(manager, k + 1)
            assert manager.new_frames == (k == T - 1)
            assert (manager.joint_state == k + 1).all()
        assert FakeModel.chunks == 1
        targets, chunk_start = FakeModel.steps[0]
        assert targets.shape == (N, T, J) and targets[0, :, 0].tolist() == [1, 2, 3, 4]
        assert np.array_equal(chunk_start, start)
        assert manager.frame_history.shape[:2] == (N, T)


def test_environments_can_only_reset_between_chunks(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(np.arange(N))
        step(manager, 0.0)
        with pytest.raises(RuntimeError, match="chunk boundaries"):
            manager.reset_envs(np.array([0]))
        for _ in range(T - 1):
            step(manager, 0.0)
        manager.reset_envs(np.array([0]))


def test_a_group_shares_a_record_and_other_splits_visit_records_in_order(tmp_path):
    states = [np.full(J, float(i)) for i in range(3)]
    with world_model_sim(tmp_path, num_envs=4, envs_per_reset_record=2, states=states, split="validation") as sim:
        manager = bound_manager(sim, num_envs=4)
        manager.reset_envs(np.arange(4))
        assert manager.joint_state[:, 0].tolist() == [0, 0, 1, 1]
        manager.reset_envs(np.arange(4))
        assert manager.joint_state[:, 0].tolist() == [2, 2, 0, 0]
