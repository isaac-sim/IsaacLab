# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-level smoke tests for OVPhysX volume and surface deformables."""

from __future__ import annotations

import gymnasium as gym
import ovphysx.types  # noqa: F401
import pytest
import torch
import warp as wp
from isaaclab_ovphysx import tensor_types as TT  # noqa: E402

from isaaclab.sim import SimulationContext  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg  # noqa: E402
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

wp.init()

_NUM_ENVS = 2


def _configure_deformable_lift_ovphysx_smoke(
    cfg_cls: type[FrankaSoftEnvCfg],
) -> FrankaSoftEnvCfg:
    """Build a minimal multi-environment OvPhysX deformable-lift task."""
    cfg = resolve_presets(cfg_cls(), ("ovphysx",))
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = _NUM_ENVS

    # Keep these smokes focused on the stock task deformable and shared MDP data
    # path while avoiding unrelated external props.
    cfg.scene.table = None
    cfg.scene.sky_light = None
    cfg.scene.ground = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.ui_window_class_type = None
    return cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_lift_franka_soft_task_reads_and_steps_volume_deformable():
    """Reset and step finite soft-lift observations and deformable state."""
    env = None
    try:
        cfg = _configure_deformable_lift_ovphysx_smoke(FrankaSoftEnvCfg)
        env = gym.make("Isaac-Lift-Soft-Franka", cfg=cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        obs, _ = env.reset()
        policy_obs = obs["policy"]
        assert policy_obs.shape[0] == _NUM_ENVS
        assert torch.isfinite(policy_obs).all()

        deformable = env.unwrapped.scene["deformable"]
        assert deformable.is_initialized
        assert deformable.num_instances == _NUM_ENVS
        assert deformable.max_sim_vertices_per_body > 0
        assert torch.isfinite(deformable.data.nodal_state_w.torch).all()

        ee_frame = env.unwrapped.scene["ee_frame"]
        assert torch.isfinite(ee_frame.data.target_pos_w.torch).all()

        targets = deformable.data.nodal_kinematic_target
        assert targets is not None
        expected_targets = targets.torch.clone()
        updated_targets = expected_targets[:1].clone()
        updated_targets[..., 3] = 1.0
        updated_targets[:, :, :3] = deformable.data.nodal_pos_w.torch[:1] + torch.tensor(
            [0.0, 0.0, 0.03], device=env.unwrapped.device
        )
        updated_targets[:, :, 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(
            updated_targets, env_ids=torch.tensor([0], device=env.unwrapped.device)
        )
        expected_targets[0] = updated_targets[0]
        readback_targets = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET))
        torch.testing.assert_close(readback_targets, expected_targets, rtol=1e-5, atol=1e-5)

        arm_action = env.unwrapped.action_manager.get_term("arm_action")
        ee_pos_curr, ee_quat_curr = arm_action._compute_frame_pose()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        actions[:, :3] = ee_pos_curr
        actions[:, 3:7] = ee_quat_curr
        for _ in range(3):
            obs, reward, terminated, time_out, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(reward).all()
            assert torch.isfinite(deformable.data.nodal_state_w.torch).all()
            assert torch.isfinite(deformable.data.root_pos_w.torch).all()
            assert torch.isfinite(ee_frame.data.target_pos_w.torch).all()
            assert not terminated.any()
            assert not time_out.any()
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            SimulationContext.clear_instance()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_lift_franka_cloth_task_reads_and_steps_surface_deformable():
    """Reset and step finite cloth-lift observations and deformable state."""
    env = None
    try:
        cfg = _configure_deformable_lift_ovphysx_smoke(FrankaClothEnvCfg)
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None

        obs, _ = env.reset()
        assert obs["policy"].shape[0] == _NUM_ENVS
        assert torch.isfinite(obs["policy"]).all()

        deformable = env.unwrapped.scene["deformable"]
        assert deformable.is_initialized
        assert deformable.num_instances == _NUM_ENVS
        assert deformable.max_sim_vertices_per_body > 0
        assert deformable.data.nodal_kinematic_target is None
        assert torch.isfinite(deformable.data.nodal_state_w.torch).all()
        assert torch.isfinite(deformable.data.root_pos_w.torch).all()

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(3):
            obs, reward, terminated, time_out, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(reward).all()
            assert torch.isfinite(deformable.data.nodal_state_w.torch).all()
            assert torch.isfinite(deformable.data.root_pos_w.torch).all()
            assert not terminated.any()
            assert not time_out.any()
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            SimulationContext.clear_instance()
