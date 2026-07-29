# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-level smoke tests for OVPhysX volume deformables."""

from __future__ import annotations

import gymnasium as gym
import ovphysx.types  # noqa: F401
import pytest
import torch
import warp as wp
from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

from isaaclab.sim import SimulationContext  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from ..deformable_utils import pre_tetrahedralized_deformable_spawn_cfg  # noqa: E402

wp.init()


def _configure_soft_lift_ovphysx_smoke():
    """Build a minimal OVPhysX variant of the Franka soft-lift task."""
    cfg = parse_env_cfg("Isaac-Lift-Soft-Franka", device="cuda:0", num_envs=1)
    cfg.sim.physics = OvPhysxCfg()
    cfg.sim.device = "cuda:0"
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.scene.num_envs = 1
    cfg.scene.replicate_physics = False
    cfg.scene.deformable.spawn = pre_tetrahedralized_deformable_spawn_cfg()

    # Keep this smoke focused on the deformable task data path. The stock task
    # still depends on unrelated external assets and articulation Jacobians.
    cfg.scene.table = None
    cfg.scene.sky_light = None
    cfg.scene.ground = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.actions.arm_action = None
    cfg.ui_window_class_type = None
    return cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OVPhysX deformables require CUDA")
@pytest.mark.isaacsim_ci
def test_lift_franka_soft_task_reads_and_steps_volume_deformable():
    """Reset and step finite soft-lift observations and deformable state."""
    env = None
    try:
        env = gym.make("Isaac-Lift-Soft-Franka", cfg=_configure_soft_lift_ovphysx_smoke())
        env.unwrapped.sim._app_control_on_stop_handle = None

        obs, _ = env.reset()
        policy_obs = obs["policy"]
        assert policy_obs.shape == (1, 86)
        assert torch.isfinite(policy_obs).all()

        deformable = env.unwrapped.scene["deformable"]
        assert deformable.is_initialized
        assert deformable.num_instances == 1
        assert deformable.max_sim_vertices_per_body == 5
        assert torch.isfinite(deformable.data.nodal_state_w.torch).all()

        ee_frame = env.unwrapped.scene["ee_frame"]
        assert torch.isfinite(ee_frame.data.target_pos_w.torch).all()

        targets = deformable.data.nodal_kinematic_target
        assert targets is not None
        updated_targets = targets.torch.clone()
        updated_targets[..., 3] = 1.0
        updated_targets[0, :, :3] = deformable.data.nodal_pos_w.torch[0] + torch.tensor(
            [0.0, 0.0, 0.03], device=env.unwrapped.device
        )
        updated_targets[0, :, 3] = 0.0
        deformable.write_nodal_kinematic_target_to_sim_index(
            updated_targets, env_ids=torch.tensor([0], device=env.unwrapped.device)
        )
        readback_targets = wp.to_torch(deformable.root_view.get_attribute(TT.DEFORMABLE_SIM_KINEMATIC_TARGET))
        torch.testing.assert_close(readback_targets, updated_targets, rtol=1e-5, atol=1e-5)

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
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
