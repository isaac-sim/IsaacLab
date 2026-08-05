# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless Newton runtime smoke tests for UR10 particle push."""

from __future__ import annotations

import math
import os

import pytest

_RUNTIME_UNAVAILABLE_REASON = "Isaac Sim runtime is unavailable because EXP_PATH is not set."
_RUNTIME_AVAILABLE = bool(os.environ.get("EXP_PATH"))

if _RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import newton
    import torch

    import isaaclab.sim as sim_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.ur10_particle_push.ur10_particle_push_env_cfg import (
        MPM_ENTRY,
        PILE_LATTICE_RESOLUTION,
        PUSH_ACTION_DIM,
        PUSH_CRITIC_OBSERVATION_DIM,
        PUSH_POLICY_OBSERVATION_DIM,
    )
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytestmark = pytest.mark.isaacsim_ci

_TASK_ID = "IsaacContrib-UR10-Particle-Push"


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_coupled_model_uses_the_physical_paddle_as_its_proxy():
    """Ensure the coupled model resolves only the physical paddle body as its proxy."""
    sim_utils.create_new_stage()
    env = None
    try:
        cfg = parse_env_cfg(_TASK_ID, device="cuda:0", num_envs=1)
        env = gym.make(_TASK_ID, cfg=cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        model = task.sim.physics_manager.get_model()
        proxy_body_ids = {int(body_id) for body_id in cfg.sim.physics.solver_cfg.proxies[0].bodies}
        assert len(proxy_body_ids) == 1
        paddle_body_id = next(iter(proxy_body_ids))
        assert model.body_label[paddle_body_id].endswith("/Robot/ee_link/Paddle")
        assert any(
            int(body_id) == paddle_body_id and label is not None and "/Paddle/geometry/mesh" in label
            for body_id, label in zip(model.shape_body.numpy(), model.shape_label, strict=True)
        )
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_randomized_reset_preserves_fixed_payload_and_unselected_world():
    """Smoke-test fixed payloads, split-pile resets, CUDA graphs, and selective exact reset."""
    sim_utils.create_new_stage()
    env = None
    try:
        cfg = parse_env_cfg(_TASK_ID, device="cuda:0", num_envs=2)
        cfg.seed = 23
        cfg.reset_cycle = True
        cfg.heightmap_depth_noise_std = 0.0
        cfg.heightmap_xy_noise_std = 0.0
        cfg.heightmap_dropout_probability = 0.0
        env = gym.make(_TASK_ID, cfg=cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        observations, _ = env.reset()

        assert observations["policy"].shape == (2, PUSH_POLICY_OBSERVATION_DIM)
        assert observations["heightmap"].shape == (2, 3, *cfg.heightmap_shape)
        assert observations["critic"].shape == (2, PUSH_CRITIC_OBSERVATION_DIM)
        torch.testing.assert_close(observations["heightmap"][:, 2], task._heightmap_goal_mask)

        assert cfg.sim.physics.use_cuda_graph
        assert task._media.particles_per_object == math.prod(PILE_LATTICE_RESOLUTION)
        assert task._particle_position_e().shape[1] == task._media.particles_per_object
        model = task.sim.physics_manager.get_model()
        particle_flags = model.particle_flags.numpy()[: model.particle_count]
        assert ((particle_flags & int(newton.ParticleFlags.ACTIVE)) != 0).all()
        assert torch.all(task._bin_fraction < cfg.success_fraction)
        assert torch.all(task._spill_fraction == 0.0)
        torch.testing.assert_close(
            task._episode_success_fraction,
            torch.full((task.num_envs,), cfg.success_fraction, device=task.device),
        )

        zero_action = torch.zeros((task.num_envs, PUSH_ACTION_DIM), device=task.device)
        observations, reward, terminated, truncated, _ = env.step(zero_action)
        assert torch.all(torch.isfinite(reward))
        assert not torch.any(terminated | truncated)
        assert all(torch.all(torch.isfinite(value)) for value in observations.values())
        assert task.sim.physics_manager._graph is not None
        task.sim.physics_manager._solver.solver(MPM_ENTRY).check_status()

        split_level = cfg.curriculum_source_pile_count.index(2)
        original_level_override = task.cfg.curriculum_level_override
        task.cfg.curriculum_level_override = split_level
        try:
            task._reset_idx(torch.tensor([0], dtype=torch.long, device=task.device))
        finally:
            task.cfg.curriculum_level_override = original_level_override
        assert task._curriculum_level[0] == split_level
        particle_position_e = task._particle_position_e()[0]
        source_mask = particle_position_e[:, 0] < cfg.bin_inner_x_bounds[0]
        focused_mask = task._particle_focused_source_mask[0]
        other_source_mask = source_mask & ~focused_mask
        assert focused_mask.any() and other_source_mask.any()
        assert particle_position_e[focused_mask, 1].mean() * particle_position_e[other_source_mask, 1].mean() < 0.0

        protected_joint_position = task._robot.data.joint_pos.torch[1].clone()
        protected_particle_position = task._media.data.particle_pos_w.torch[1].clone()
        protected_particle_velocity = task._media.data.particle_vel_w.torch[1].clone()
        protected_episode_length = task.episode_length_buf[1].clone()

        final_level = len(cfg.curriculum_pile_center_x) - 1
        task.cfg.curriculum_level_override = final_level
        try:
            task._reset_idx(torch.tensor([0], dtype=torch.long, device=task.device))
        finally:
            task.cfg.curriculum_level_override = original_level_override

        assert task.episode_length_buf[0] == 0
        torch.testing.assert_close(task._robot.data.joint_pos.torch[1], protected_joint_position, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            task._media.data.particle_pos_w.torch[1], protected_particle_position, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            task._media.data.particle_vel_w.torch[1], protected_particle_velocity, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(task.episode_length_buf[1], protected_episode_length)

        observations, reward, terminated, truncated, _ = env.step(zero_action)
        assert torch.all(torch.isfinite(reward))
        assert not torch.any(terminated | truncated)
        assert all(torch.all(torch.isfinite(value)) for value in observations.values())
        task.sim.physics_manager._solver.solver(MPM_ENTRY).check_status()
    finally:
        if env is not None:
            env.close()
