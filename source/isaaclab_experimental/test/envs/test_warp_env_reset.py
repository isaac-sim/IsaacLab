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
from isaaclab.managers import CurriculumTermCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 8

_curriculum_env_ids = []


def _record_curriculum_env_ids(env, env_ids):
    """Curriculum term that records the indices of the environments it updates."""
    _curriculum_env_ids.append(env_ids)


def _make_cartpole(curriculum: dict | None = None):
    env_cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    env_cfg.curriculum = curriculum
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


def test_reset_with_ids_or_mask_passes_the_same_ids_to_curriculum_terms():
    # the stable curriculum manager takes indices, so a masked reset hands it the indices of its mask
    env = _make_cartpole(curriculum={"record_env_ids": CurriculumTermCfg(func=_record_curriculum_env_ids)})
    try:
        env.reset()
        _curriculum_env_ids.clear()

        env.reset(env_ids=[0, 2, 4, 6])
        env.reset(env_mask=torch.arange(_NUM_ENVS, device=env.device) % 2 == 0)

        expected = torch.tensor([0, 2, 4, 6], device=env.device)
        assert len(_curriculum_env_ids) == 2
        assert all(torch.equal(env_ids, expected) for env_ids in _curriculum_env_ids)
    finally:
        env.close()
