# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Captured reset replay and episode attribution through the existing Lift environment."""

import gymnasium as gym
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_lift_native_reset_replay_and_completed_outcomes(device, monkeypatch):
    cfg = load_cfg_from_registry("Isaac-Lift-Franka", "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=("newton_mjwarp",))
    cfg.sim.device = device
    cfg.scene.num_envs = 4
    cfg.seed = 20261116
    cfg.events.conditional_reset.params["buffer_size_per_group"] = 4
    cfg.events.conditional_reset.params["max_prefill_iters"] = 256
    with launch_simulation(cfg):
        env = gym.make("Isaac-Lift-Franka", cfg=cfg).unwrapped
        try:
            env.reset()
            term = env.event_manager.get_term_cfg("conditional_reset").func
            mask = wp.ones(env.num_envs, dtype=wp.bool, device=device)
            completed = wp.zeros(env.num_envs, dtype=wp.bool, device=device)
            requested = wp.clone(term._sampler.state_ids.warp)
            term.reset_mask(mask, state_ids=requested, completed=completed)
            with wp.ScopedCapture(device=device) as capture:
                term.reset_mask(mask, state_ids=requested, completed=completed)
            assets = {**env.scene.articulations, **env.scene.rigid_objects}
            previous = {name: asset.data.root_link_pose_w.torch.clone() for name, asset in assets.items()}
            robot = env.scene["robot"]
            previous_joints = robot.data.joint_pos.torch.clone()
            previous_targets = robot.actuators.target_command.position.torch.clone()
            selected = torch.tensor([True, False, True, False], device=device)
            ids = term._sampler.state_ids.torch.clone()
            ids[selected] = (ids[selected] // 4) * 4 + (ids[selected] + 1) % 4
            requested.assign(ids.cpu().numpy())
            mask.assign(selected.cpu().numpy())
            wp.capture_launch(capture.graph)
            torch.testing.assert_close(term._sampler.state_ids.torch, ids)
            expected = term._bank.get_state(term._sampler.indices)
            for name, asset in assets.items():
                pose = expected[name]["root_pose"].torch.clone()
                pose[:, :3] += env.scene.env_origins
                torch.testing.assert_close(asset.data.root_link_pose_w.torch[selected], pose[selected])
                torch.testing.assert_close(asset.data.root_link_pose_w.torch[~selected], previous[name][~selected])
            torch.testing.assert_close(
                robot.data.joint_pos.torch[selected], expected["robot"]["joint_position"].torch[selected]
            )
            torch.testing.assert_close(robot.data.joint_pos.torch[~selected], previous_joints[~selected])
            torch.testing.assert_close(
                robot.actuators.target_command.position.torch[selected], robot.data.joint_pos.torch[selected]
            )
            torch.testing.assert_close(
                robot.actuators.target_command.position.torch[~selected], previous_targets[~selected]
            )
            torch.testing.assert_close(
                robot.actuators.target_command.velocity.torch[selected], torch.zeros_like(previous_joints[selected])
            )

            # Captured completion reads the live success flag and credits the previous assignment.
            previous_row = int(term._sampler.indices.torch[0])
            env.episode_length_buf[0] = 1
            env.reward_manager.get_term_cfg("success").func.succeeded[0] = True
            completed.assign([True, False, False, False])
            wp.capture_launch(capture.graph)
            assert term._sampler.episode_counts.torch.sum() == 1
            assert term._sampler.episode_counts.torch[previous_row] == 1
            assert term._sampler.success_rate.torch[previous_row] == 1.0
            completed.zero_()

            # Explicit environment resets do not count as completed episodes.
            recorded = term._sampler.episode_counts.torch.clone()
            env.reset()
            torch.testing.assert_close(term._sampler.episode_counts.torch, recorded)

            # RSL-RL can replace the counter tensor while initializing randomized episode lengths.
            env.episode_length_buf = env.episode_length_buf.clone()
            # A viewer request ends the wrapper episode but is not a completed natural attempt.
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
            monkeypatch.setattr(env.sim, "consume_reset_request", lambda: True)
            observations, reward, _, _, _ = env.step(actions)
            assert not env.reset_buf.any()
            torch.testing.assert_close(term._sampler.episode_counts.torch, recorded)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in observations.values())

            # One timeout plus a viewer request in the same frame credits only the timeout's old row.
            monkeypatch.setattr(env.sim, "consume_reset_request", lambda: False)
            env.step(actions)
            env.episode_length_buf[0] = env.max_episode_length - 1
            env.reward_manager.get_term_cfg("success").func.succeeded[0] = True
            previous_row = int(term._sampler.indices.torch[0])
            recorded[previous_row] += 1
            monkeypatch.setattr(env.sim, "consume_reset_request", lambda: True)
            env.step(actions)
            assert env.reset_buf.tolist() == [True, False, False, False]
            torch.testing.assert_close(term._sampler.episode_counts.torch, recorded)
            assert term._sampler.success_rate.torch[previous_row] == 1.0
        finally:
            env.close()
