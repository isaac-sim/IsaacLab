# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resets of the manager-based Warp environment."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import torch
from isaaclab_experimental.envs.frontend import WarpFrontend

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 8


def _make_cartpole():
    env_cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    sim_utils.create_new_stage()
    return WarpFrontend.build_env(env_cfg, "Isaac-Cartpole").unwrapped


def test_reset_with_mask_resets_only_selected_environments():
    env = _make_cartpole()
    try:
        env.reset()
        actions = torch.ones((_NUM_ENVS, env.action_space.shape[-1]), device=env.device)
        for _ in range(5):
            env.step(actions)
        robot = env.scene["robot"]
        joint_pos = robot.data.joint_pos.torch.clone()
        selected = torch.arange(_NUM_ENVS, device=env.device) % 2 == 0

        env.reset(env_mask=selected)

        assert torch.equal(env.episode_length_buf[selected], torch.zeros_like(env.episode_length_buf[selected]))
        assert torch.equal(env.episode_length_buf[~selected], torch.full_like(env.episode_length_buf[~selected], 5))
        assert torch.equal(robot.data.joint_pos.torch[~selected], joint_pos[~selected])
        # the reset events sample new joint positions for every selected environment
        assert not torch.isclose(robot.data.joint_pos.torch[selected], joint_pos[selected]).all(dim=1).any()
    finally:
        env.close()


def test_set_term_cfg_applies_to_recorded_reset_events():
    """A replaced event term configuration must not replay the scalars recorded from the old one."""
    env = _make_cartpole()
    try:
        env.reset()
        env.step(torch.zeros((_NUM_ENVS, env.action_space.shape[-1]), device=env.device))
        selected = torch.arange(_NUM_ENVS, device=env.device) % 2 == 0
        # records the reset events with the configured random cart offsets
        env.reset(env_mask=selected)
        assert "EventManager_apply_reset" in env._warp_graph_cache.captured_stages

        term_cfg = env.event_manager.get_term_cfg("reset_cart_position")
        term_cfg.params["position_range"] = (0.5, 0.5)
        term_cfg.params["velocity_range"] = (0.0, 0.0)
        env.event_manager.set_term_cfg("reset_cart_position", term_cfg)
        env.reset(env_mask=selected)

        robot = env.scene["robot"]
        cart = robot.find_joints("slider_to_cart")[0][0]
        cart_pos = robot.data.joint_pos.torch[selected, cart]
        default_pos = robot.data.default_joint_pos.torch[selected, cart]
        assert torch.allclose(cart_pos, default_pos + 0.5)
    finally:
        env.close()
