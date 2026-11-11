# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Essential native release qualification, receipt, handoff, and reset isolation."""

import math
from contextlib import closing

import gymnasium as gym
import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_suction.actions import solve_tool_ik
from isaaclab_tasks.contrib.conveyor_suction.scene import BELT_TOP_Z, BIN_INTERIOR_SIZE, OutfeedLayout, OutfeedsCfg
from isaaclab_tasks.utils import parse_env_cfg

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_qualified_receipt_rewards_once_and_preserves_other_world(device):
    """Only a qualified physical receipt earns +10; handoffs and unrelated resets preserve it."""
    cfg = parse_env_cfg("IsaacContrib-Conveyor-Suction-Sorting", device=device, num_envs=2)
    cfg.sim.visualizer_cfgs = []
    cfg.sim.save_logs_to_file = False
    cfg.seed = 41
    cfg.events.reset_parcels.params["samples_per_path"] = 2
    cfg.commands.sort.goal_radial_range = cfg.commands.sort.goal_lateral_range = (0.0, 0.0)
    angles = tuple(math.radians(value) for value in (10, 35, 60, 85, 110))
    cfg.scene.outfeeds.spawn.assets_cfg = [OutfeedsCfg(layout=OutfeedLayout(angles, (0.0,) * 5, (0.0,) * 5))]
    for index in range(cfg.parcel_count):
        setattr(cfg.events, f"parcel_{index}_mass", None)
        setattr(cfg.events, f"parcel_{index}_material", None)
    # Keep the rejected world observable instead of immediately resetting it.
    cfg.terminations.failure = None
    with closing(gym.make("IsaacContrib-Conveyor-Suction-Sorting", cfg=cfg).unwrapped) as env:
        command = env.command_manager.get_term("sort")
        reset = env.event_manager.get_term_cfg("reset_parcels").func
        reset.reset_sampler.sampler.sample = lambda probabilities, count: torch.full(
            (count,), 2, dtype=torch.long, device=device
        )
        env.reset()
        parcels = env.active_ids.clone()
        robot = env.scene["robot"]
        assert env.gripper.engaged.all() and not command.lifted.any()

        # Establish lift qualification through real robot actions, without assigning command flags.
        actions = torch.zeros((2, 4), device=device)
        actions[:, 2], actions[:, 3] = 1.0, -1.0
        for _ in range(90):
            env.step(actions)
            if command.lifted.all():
                break
        assert env.gripper.engaged.all() and command.lifted.all()
        actions[:, 2] = 0.0

        def set_parcel(env_id, parcel_id, position):
            ids = torch.tensor([env_id], device=device)
            parcel = env.parcels[parcel_id]
            pose = parcel.data.default_root_pose.torch[ids].clone()
            pose[:, :3] = position + env.scene.env_origins[ids]
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=ids)

        # Seed two measured release fixtures: supported placement versus an elevated drop.
        # The fixture relocates bodies; native seals and qualification still determine the outcome.
        ids = torch.arange(2, device=device)
        tool = command.goal_positions.clone()
        tool[:, 2] += env.parcel_sizes[parcels, 2] / 2.0 + 0.010
        tool[1, 2] += 0.05
        joints = robot.data.default_joint_pos.torch.clone()
        arm_ids = robot.find_joints("panda_joint[1-7]")[0]
        joints[:, arm_ids] = solve_tool_ik(tool)
        robot.write_joint_position_to_sim_index(position=joints, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), env_ids=ids)
        robot.actuators.target_command.set_position_index(value=joints, env_ids=ids)
        for world, parcel in enumerate(parcels.tolist()):
            center = tool[world].clone()
            center[2] -= env.parcel_sizes[parcel, 2] / 2.0 + 0.002
            set_parcel(world, parcel, center)
        env.action_manager.reset(ids)
        env.gripper.reset(ids)
        env.sim.forward()
        env.scene.update(env.step_dt)
        env.gripper.set_command(torch.ones(2, dtype=torch.bool, device=device), parcels)
        env.gripper.refresh()
        for _ in range(10):
            env.step(actions)
        assert env.gripper.engaged.all()
        actions[:, 3] = 1.0
        env.step(actions)
        assert command.metrics["controlled_placements"].tolist() == [1, 0]
        assert command.metrics["invalid_releases"].tolist() == [0, 1]
        assert command.failed.tolist() == [False, True]

        # Both actual bodies settle into their correct bins. The unqualified drop still earns nothing.
        for world, parcel in enumerate(parcels.tolist()):
            center = env.bin_positions[world, command.destination_ids[world]].clone()
            center[2] -= BIN_INTERIOR_SIZE[2] / 2.0 - env.parcel_sizes[parcel, 2] / 2.0 - 0.006
            set_parcel(world, parcel, center)
        env.sim.forward()
        env.scene.update(env.step_dt)
        total_reward = torch.zeros(2, device=device)
        pulses = torch.zeros(2, dtype=torch.long, device=device)
        for _ in range(15):
            before = env.episode_length_buf.clone()
            previous = env.active_ids[0].item()
            position = robot.data.joint_pos.torch[0].clone()
            observations, rewards, terminated, timed_out, extras = env.step(actions)
            assert not terminated.any() and not timed_out.any()
            torch.testing.assert_close(env.episode_length_buf, before + 1)
            if env.active_ids[0] != previous:
                assert (robot.data.joint_pos.torch[0] - position).abs().max() < 0.05
            assert torch.isfinite(observations["policy"]).all() and torch.isfinite(rewards).all()
            total_reward += rewards
            pulses += extras["successes"].long()
        assert total_reward.tolist() == pytest.approx([10.0, 0.0])
        assert pulses.tolist() == [1, 0]
        assert command.metrics["correct_deliveries"].tolist() == [1, 0]
        assert command.delivered[0, parcels[0]] and not command.delivered[1].any()

        # Supply one next arrival if the racetrack has not already provided it.
        if env.active_ids[0] == parcels[0]:
            candidate = (parcels[0].item() + 1) % len(env.parcels)
            center = torch.tensor((0.30, -0.72, 0.0), device=device)
            center[2] = BELT_TOP_Z + env.parcel_sizes[candidate, 2] / 2.0 + 0.006
            set_parcel(0, candidate, center)
            env.sim.forward()
            env.scene.update(env.step_dt)
            before = env.episode_length_buf.clone()
            position = robot.data.joint_pos.torch[0].clone()
            _, rewards, terminated, timed_out, extras = env.step(actions)
            torch.testing.assert_close(env.episode_length_buf, before + 1)
            assert not terminated.any() and not timed_out.any()
            assert (robot.data.joint_pos.torch[0] - position).abs().max() < 0.05
            assert rewards[0] == 0 and not extras["successes"][0]
        assert env.active_ids[0] != parcels[0]
        assert command.metrics["correct_deliveries"][0] == 1

        # Resetting only the rejected world must not move, relabel, or erase the successful world.
        positions = robot.data.joint_pos.torch[0].clone()
        velocities = robot.data.joint_vel.torch[0].clone()
        states = [
            (parcel.data.root_link_pose_w.torch[0].clone(), parcel.data.root_link_vel_w.torch[0].clone())
            for parcel in env.parcels
        ]
        active = env.active_ids[0].clone()
        lengths = env.episode_length_buf.clone()
        delivered = command.delivered[0].clone()
        env.reset(env_ids=torch.tensor([1], device=device))
        torch.testing.assert_close(robot.data.joint_pos.torch[0], positions, rtol=0, atol=0)
        torch.testing.assert_close(robot.data.joint_vel.torch[0], velocities, rtol=0, atol=0)
        for parcel, (pose, velocity) in zip(env.parcels, states, strict=True):
            torch.testing.assert_close(parcel.data.root_link_pose_w.torch[0], pose, rtol=0, atol=0)
            torch.testing.assert_close(parcel.data.root_link_vel_w.torch[0], velocity, rtol=0, atol=0)
        assert env.active_ids[0] == active and env.episode_length_buf[0] == lengths[0]
        torch.testing.assert_close(command.delivered[0], delivered)
        assert command.metrics["correct_deliveries"][0] == 1
        assert not command.delivered[1].any() and env.episode_length_buf[1] == 0
