# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mask-first resets of the manager-based Warp environment."""

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


def test_reset_with_mask_resets_only_selected_environments():
    env_cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    sim_utils.create_new_stage()
    env = WarpFrontend.build_env(env_cfg, "Isaac-Cartpole").unwrapped
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
