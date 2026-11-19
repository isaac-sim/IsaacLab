# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The world-model backend: chunked prediction, joint state and resets."""

import numpy as np
import pytest
import torch

from isaaclab_neural.physics import WorldModelCfg, WorldModelManager

from .conftest import FakeModel, FakeModelCfg, J, N, T, world_model_sim, write_manifest


def bound_manager(sim, num_envs=N):
    manager = sim.backend_manager
    manager.bind(num_envs, J)
    return manager


def step(manager, target):
    manager.joint_target.fill_(target)
    manager.step()


def envs(*ids, num_envs=N):
    """A mask of the given environments; all of them when none are given."""
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[list(ids) if ids else slice(None)] = True
    return mask


def test_reset_starts_from_the_record_and_the_model_restarts_before_the_next_chunk(tmp_path):
    states = [np.full(J, float(i)) for i in range(3)]
    with world_model_sim(tmp_path, states=states, split="validation") as sim:
        manager = bound_manager(sim)
        manager.reset_envs(envs(1))
        # The episode starts at once from record 0: its state and its start image.
        assert manager.joint_state[:, 0].tolist() == [0, 0] and manager.joint_target[1, 0] == 0
        assert (manager.frame_history[1] == 200).all() and manager.frame_serial.tolist() == [0, 1]
        assert not FakeModel.resets
        manager.reset_envs(envs(0))
        for _ in range(T):
            step(manager, 5.0)
        # The step that ends the chunk restarts the model's environments first, with their records.
        ((env_ids, images, prompts, seeds),) = FakeModel.resets
        assert env_ids == [0, 1] and images == [201, 200] and prompts == ["prompt 1", "prompt 0"]
        assert len(set(seeds)) == 2
        assert FakeModel.chunks == 1


def test_the_model_runs_once_per_chunk_and_the_joint_state_follows_the_last_target(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(envs())
        start = manager.joint_state.clone()
        for k in range(T):
            step(manager, k + 1)
            assert manager.new_frames == (k == T - 1)
            assert (manager.joint_state == k + 1).all()
        assert FakeModel.chunks == 1
        targets, chunk_start = FakeModel.steps[0]
        assert targets.shape == (N, T, J) and targets[0, :, 0].tolist() == [1, 2, 3, 4]
        assert np.array_equal(chunk_start, start.numpy())
        assert manager.frame_history.shape[:2] == (N, T)


def test_a_partial_reset_keeps_the_chunk_boundary_of_the_step(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(envs())
        for _ in range(T):
            step(manager, 0.0)
        manager.reset_envs(envs(0))
        assert manager.new_frames and manager.chunk_boundary.all()


def test_a_new_initialization_replaces_the_model_and_one_articulation_binds(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        with pytest.raises(ValueError, match="one articulation"):
            manager.bind(N, J)
        manager.reset()  # a full simulation reset, after which the articulation initializes again
        manager.bind(N, J)
        assert FakeModel.closed == 1


def test_the_reset_records_must_match_the_split_and_the_joints(tmp_path):
    with pytest.raises(ValueError, match="no records in split 'validation'"):
        WorldModelManager(
            WorldModelCfg(model=FakeModelCfg(), reset_manifest=write_manifest(tmp_path), split="validation")
        )
    with world_model_sim(tmp_path, states=[np.zeros(J + 1)]) as sim:
        with pytest.raises(ValueError, match="joint positions"):
            bound_manager(sim)


def test_environments_can_only_reset_between_chunks(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(envs())
        for _ in range(T):
            step(manager, 0.0)
        manager.reset_envs(envs(0))
        step(manager, 0.0)
        # A reset inside the chunk is reported where the chunk ends, before the model predicts it.
        manager.reset_envs(envs(1))
        for _ in range(T - 2):
            step(manager, 0.0)
        with pytest.raises(RuntimeError, match=r"Environments \[1\] reset inside a chunk"):
            step(manager, 0.0)


def test_a_group_shares_a_record_and_other_splits_visit_records_in_order(tmp_path):
    states = [np.full(J, float(i)) for i in range(3)]
    with world_model_sim(tmp_path, num_envs=4, envs_per_reset_record=2, states=states, split="validation") as sim:
        manager = bound_manager(sim, num_envs=4)
        manager.reset_envs(envs(num_envs=4))
        assert manager.joint_state[:, 0].tolist() == [0, 0, 1, 1]
        manager.reset_envs(envs(2, 3, num_envs=4))
        assert manager.joint_state[:, 0].tolist() == [0, 0, 2, 2]
        manager.reset_envs(envs(num_envs=4))
        assert manager.joint_state[:, 0].tolist() == [0, 0, 1, 1]
