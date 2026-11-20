# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The world-model backend: chunked prediction, joint state and resets."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_neural.mdp import reset_from_records
from isaaclab_neural.physics import WorldModelCfg, WorldModelManager

from .conftest import DT, FakeModel, FakeModelCfg, H, J, N, T, W, world_model_sim, write_manifest


def bound_manager(sim, num_envs=N):
    manager = sim.backend_manager
    manager.bind(num_envs, J)
    return manager


def step(manager, target):
    manager.set_joint_target(np.full((manager.num_envs, J), target, np.float32))
    manager.step()


def test_reset_publishes_the_record_and_restarts_the_model_before_the_next_chunk(tmp_path):
    with world_model_sim(tmp_path, envs_per_reset_record=1, split="validation") as sim:
        manager = bound_manager(sim)
        manager.reset_envs(torch.tensor([False, True], device=sim.device))
        assert manager.joint_state[:, 0].tolist() == [0.0, pytest.approx(0.1)]
        assert (manager.frames().frames[1] == 200).all()
        manager.reset_envs(torch.tensor([True, False], device=sim.device))
        assert not FakeModel.resets
        assert (manager.frames().frames[0] == 201).all() and not manager.new_frames
        for _ in range(T - 1):
            step(manager, 5.0)
            assert not FakeModel.calls
        step(manager, 5.0)
        assert FakeModel.calls == ["reset", "step"]
        ((env_ids, images, prompts, seeds),) = FakeModel.resets
        assert env_ids == [0, 1]
        assert images["camera"][:, 0, 0, 0].tolist() == [201, 200]
        assert prompts == ["prompt 1", "prompt 0"] and len(set(seeds)) == 2

        for _ in range(T):
            step(manager, 6.0)
        assert FakeModel.calls == ["reset", "step", "step"]
        previous = manager.frames().frames[1].clone()
        manager.reset_envs(torch.tensor([True, False], device=sim.device))
        assert (manager.frames().frames[0] == 200).all()
        manager.reset_envs(torch.tensor([True, False], device=sim.device))
        assert (manager.frames().frames[0] == 201).all()
        assert np.array_equal(manager.frames().frames[1], previous)
        assert len(FakeModel.resets) == 1
        for _ in range(T):
            step(manager, 7.0)
        assert FakeModel.calls == ["reset", "step", "step", "reset", "step"]
        assert FakeModel.resets[-1][0] == [0]
        assert FakeModel.resets[-1][2] == ["prompt 1"]


def test_record_images_must_match_the_model_size(tmp_path):
    with world_model_sim(tmp_path) as sim:
        Image.new("RGB", (W + 1, H)).save(tmp_path / "start_0.png")
        with pytest.raises(ValueError, match="world model expects"):
            bound_manager(sim)


def test_a_missing_reset_split_reports_the_split_before_binding(tmp_path):
    with pytest.raises(ValueError, match="no records in split 'validation'"):
        WorldModelManager(
            WorldModelCfg(model=FakeModelCfg(), reset_manifest=write_manifest(tmp_path), split="validation")
        )


@pytest.mark.parametrize("states", [[np.zeros(1)], [np.zeros(J + 1)], [np.zeros(J), np.zeros(J - 1)]])
def test_reset_record_joint_counts_must_match_the_robot(tmp_path, states):
    with world_model_sim(tmp_path, states=states) as sim:
        with pytest.raises(ValueError, match=f"articulation has {J} joints"):
            bound_manager(sim)


def test_duplicate_binding_is_rejected_and_full_reset_replaces_the_model(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        first = manager.model
        with pytest.raises(ValueError, match="one articulation"):
            manager.bind(N, J)
        assert manager.model is first and not first.closed
        manager.reset(soft=True)
        with pytest.raises(ValueError, match="one articulation"):
            manager.bind(N, J)
        manager.reset()
        manager.bind(N, J)
        assert first.closed and manager.model is not first
        assert not manager.model.closed
        manager.reset_envs(torch.ones(N, dtype=torch.bool, device=sim.device))
        for _ in range(T):
            step(manager, 2.0)
        assert FakeModel.calls == ["reset", "step"]
        assert (manager.frames().frames == torch.arange(10, 10 + T)[None, :, None, None, None]).all()


def test_the_model_runs_once_per_chunk_and_the_joint_state_follows_the_last_target(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(torch.ones(N, dtype=torch.bool, device=sim.device))
        start = manager.joint_state.clone()
        for k in range(T):
            step(manager, k + 1)
            assert manager.new_frames == (k == T - 1)
            assert (manager.joint_state == k + 1).all()
        assert FakeModel.chunks == 1
        targets, chunk_start = FakeModel.steps[0]
        assert targets.shape == (N, T, J) and targets[0, :, 0].tolist() == [1, 2, 3, 4]
        assert np.array_equal(chunk_start, start)
        assert manager.frames().frames.shape[:2] == (N, T)


def test_partial_reset_keeps_the_last_steps_chunk_boundary_until_the_next_step(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(torch.ones(N, dtype=torch.bool, device=sim.device))
        for _ in range(T):
            step(manager, 1.0)
        previous = manager.frames().frames[1].clone()
        manager.reset_envs(torch.tensor([True, False], device=sim.device))
        assert manager.new_frames and manager.chunk_boundary.all()
        torch.testing.assert_close(manager.frames().frames[1], previous)
        step(manager, 2.0)
        assert not manager.new_frames and not manager.chunk_boundary.any()
        for _ in range(T - 1):
            step(manager, 2.0)
        assert manager.new_frames and manager.chunk_boundary.all()


def test_environments_can_only_reset_between_chunks(tmp_path):
    with world_model_sim(tmp_path) as sim:
        manager = bound_manager(sim)
        manager.reset_envs(torch.ones(N, dtype=torch.bool, device=sim.device))
        step(manager, 0.0)
        with pytest.raises(RuntimeError, match="chunk boundaries"):
            manager.reset_envs(torch.tensor([True, False], device=sim.device))
        for _ in range(T - 1):
            step(manager, 0.0)
        manager.reset_envs(torch.tensor([True, False], device=sim.device))


def test_a_group_shares_a_record_and_other_splits_visit_records_in_order(tmp_path):
    states = [np.full(J, float(i)) for i in range(3)]
    with world_model_sim(tmp_path, num_envs=4, envs_per_reset_record=2, states=states, split="validation") as sim:
        manager = bound_manager(sim, num_envs=4)
        manager.reset_envs(torch.ones(4, dtype=torch.bool, device=sim.device))
        assert manager.joint_state[:, 0].tolist() == [0, 0, 1, 1]
        manager.reset_envs(torch.ones(4, dtype=torch.bool, device=sim.device))
        assert manager.joint_state[:, 0].tolist() == [2, 2, 0, 0]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("split", ["train", "validation"])
def test_reset_event_samples_on_device_without_host_sync_and_reproduces_draws(tmp_path, device, split):
    states = [np.full(J, float(i)) for i in range(8)]
    runs = []
    for _ in range(2):
        with world_model_sim(
            tmp_path, num_envs=4, envs_per_reset_record=2, states=states, split=split, device=device
        ) as sim:
            manager = bound_manager(sim, num_envs=4)
            env = SimpleNamespace(
                sim=sim,
                cfg=SimpleNamespace(decimation=1),
                scene={"robot": SimpleNamespace(num_instances=4, device=device)},
                step_dt=DT,
                max_episode_length=3 * T,
            )
            event = reset_from_records(None, env)
            partial = torch.tensor([0], device=device)
            empty = torch.empty(0, dtype=torch.long, device=device)
            previous_mode = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                event(env, slice(None))
                initial = manager.joint_state.clone()
                event(env, partial)
                updated = manager.joint_state.clone()
                event(env, empty)
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
            torch.testing.assert_close(manager.joint_state, updated)
            torch.testing.assert_close(manager.joint_target, updated)
            torch.testing.assert_close(updated[1:], initial[1:])
            assert initial[:, 0].tolist()[::2] == initial[:, 0].tolist()[1::2]
            assert ((updated >= 0) & (updated < 8)).all()
            assert manager.frames().frames[:, 0, 0, 0, 0].tolist() == (updated[:, 0] + 200).tolist()
            if split == "validation":
                assert initial[:, 0].tolist() == [0, 0, 1, 1]
                assert updated[:, 0].tolist() == [2, 0, 1, 1]
            runs.append((initial.cpu(), updated.cpu()))
            for _ in range(T):
                step(manager, 1.0)
            assert FakeModel.resets[0][0] == [0, 1, 2, 3]
            assert FakeModel.resets[0][2] == [f"prompt {int(x)}" for x in updated[:, 0].tolist()]
            # Empty resets do not overwrite the last actual reset's model seeds.
            assert FakeModel.resets[0][3] == [1994, 998, 999, 1000]
    for first, second in zip(*runs):
        torch.testing.assert_close(first, second)
