# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical receipt counting across target handoffs and independent environment resets."""

from contextlib import closing

import gymnasium as gym
import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_partial_batch_reset_only_rewards_new_physical_receipts(device):
    """Pre-sorted inventory is physical, earns no receipt credit, and is excluded from full-start metrics."""
    cfg = parse_env_cfg("IsaacContrib-Conveyor-Suction-Sorting", device=device, num_envs=2)
    cfg.commands.sort.deliveries_per_episode = 12
    cfg.episode_length_s = 240
    cfg.sim.visualizer_cfgs = []
    cfg.sim.save_logs_to_file = False
    cfg.terminations.success = None
    with closing(gym.make("IsaacContrib-Conveyor-Suction-Sorting", cfg=cfg).unwrapped) as env:
        reset = env.event_manager.get_term_cfg("reset_parcels").func
        rows = torch.tensor([0, 38], device=device)  # Last-parcel release and an unsorted full belt start.
        reset.reset_sampler.sampler.sample = lambda probabilities, count: rows[:count]
        env.reset(seed=41)
        command = env.command_manager.get_term("sort")
        command.evaluate()
        assert command.delivered.sum(dim=1).tolist() == [11, 0]
        assert not command.delivered[env.scene._ALL_INDICES, env.active_ids].any()
        assert not env.episode_success.any()
        assert not command.new_success.any()
        assert command.metrics["correct_deliveries"].tolist() == [0, 0]
        positions = torch.stack([parcel.data.root_pos_w.torch[0] for parcel in env.parcels])
        offsets = positions - env.scene.env_origins[0] - env.bin_positions[env.parcel_classes]
        seeded = command.delivered[0]
        # Bounding circles establish containment independently of the reset's bin-frame transform.
        assert (
            torch.linalg.vector_norm(offsets[seeded, :2], dim=-1)
            + torch.linalg.vector_norm(env.parcel_sizes[seeded, :2] / 2, dim=-1)
            < 0.28
        ).all()
        assert (offsets[seeded, 2].abs() + env.parcel_sizes[seeded, 2] / 2 < 0.28).all()
        actions = torch.zeros((2, 4), device=device)
        actions[:, 3] = 1.0
        for _ in range(720):
            env.step(actions)
            if env.episode_success[0]:
                break
        assert env.episode_success[0]
        assert command.delivered[0].all()
        assert command.metrics["correct_deliveries"][0] == 1
        extras = command.reset(torch.tensor([0], device=device))
        assert extras["full_start_episodes"] == 0
        assert "full_start_success_rate" not in extras and "full_batch_success_rate" not in extras
        full = command.reset(torch.tensor([1], device=device))
        assert full["full_start_episodes"] == 1
        assert full["full_start_success_rate"].tolist() == full["full_batch_success_rate"].tolist() == [0]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_partial_reset_cannot_credit_the_next_selected_parcel(device):
    """A handoff is not another receipt; two receipts complete an episode, and an inactive loss fails it."""
    task = "IsaacContrib-Conveyor-Suction-Sorting"
    cfg = parse_env_cfg(task, device=device, num_envs=2)
    cfg.sim.visualizer_cfgs = []
    cfg.sim.save_logs_to_file = False
    cfg.seed = 41
    cfg.terminations.success = None
    cfg.terminations.failure = None
    cfg.terminations.invalid_action = None
    with closing(gym.make(task, cfg=cfg).unwrapped) as env:
        reset = env.event_manager.get_term_cfg("reset_parcels").func
        rows = torch.stack(
            [((reset.parcel_ids == parcel) & (reset.phase_ids == 0)).nonzero()[0, 0] for parcel in (0, 4)]
        )
        reset.reset_sampler.sampler.sample = lambda probabilities, count: rows[:count]
        env.reset()
        command = env.command_manager.get_term("sort")
        actions = torch.zeros((2, 4), device=device)
        actions[:, 3] = 1.0
        # A recorded outfeed impact must settle in the bin without crossing its floor.
        impact_ids = torch.tensor([0], device=device)
        pose = torch.tensor(
            [[2.07563591, 0.28751278, -0.72892869, -0.97748345, -0.07764439, -0.19566739, -0.01455657]],
            device=device,
        )
        pose[:, :3] += env.scene.env_origins[impact_ids]
        velocity = torch.tensor(
            [[0.31988335, 0.05606911, -3.89049959, -1.84165192, 10.48158455, 0.01690909]], device=device
        )
        env.parcels[0].write_root_pose_to_sim_index(root_pose=pose, env_ids=impact_ids)
        env.parcels[0].write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=impact_ids)
        env.sim.forward()
        env.scene.update(env.step_dt)
        for _ in range(720):
            env.step(actions)
            if command.delivered[0, 0] and command.delivered[1, 4]:
                break
        else:
            pytest.fail("The seeded release never reached its physical bin.")
        assert not command.failed.any()
        assert env.active_ids[1] != 4, "The release fixture must also reach the next-target handoff."
        expected = torch.zeros(len(env.parcels), dtype=torch.bool, device=device)
        expected[4] = True
        torch.testing.assert_close(command.delivered[1], expected)

        # Resetting the other world invalidates the shared evaluation cache without advancing physics.
        env.reset(env_ids=impact_ids)
        command.evaluate()
        torch.testing.assert_close(command.delivered[1], expected)
        assert command.metrics["correct_deliveries"][1] == 1
        assert not env.episode_success[1]

        def place_in_bin(env_id, parcel_id, bin_id):
            ids = torch.tensor([env_id], device=device)
            parcel = env.parcels[parcel_id]
            pose = parcel.data.default_root_pose.torch[ids].clone()
            pose[:, :3] = env.bin_positions[bin_id] + env.scene.env_origins[ids]
            pose[:, 2] -= 0.28 - env.parcel_sizes[parcel_id, 2] / 2 - 0.006
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=ids)
            env.sim.forward()
            env.scene.update(env.step_dt)

        # Seed a previously lifted and released second parcel; the native scene establishes settlement.
        second = env.active_ids[1].item()
        command.lifted[1] = True
        place_in_bin(1, second, env.parcel_classes[second])
        for _ in range(15):
            env.step(actions)
        assert env.episode_success[1]
        assert command.delivered[1].sum() == command.metrics["correct_deliveries"][1] == 2

        # An unselected parcel cannot disappear into a wrong bin without failing its world.
        inactive = next(
            i for i in range(len(env.parcels)) if i != env.active_ids[0].item() and not command.delivered[0, i]
        )
        place_in_bin(0, inactive, (env.parcel_classes[inactive].item() + 1) % 5)
        for _ in range(15):
            env.step(actions)
        assert command.failed.tolist() == [True, False]
        assert env.episode_success.tolist() == [False, True]

        # A rejected policy action must not become a successful reset-sampling outcome.
        actions[1, 0] = torch.nan
        _, rewards, _, _, _ = env.step(actions)
        assert command.failed[1]
        assert not env.episode_success[1]
        assert command.metrics["success_rate"][1] == 0
        assert torch.isfinite(rewards).all()
