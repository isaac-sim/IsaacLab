# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 terrain resets honor current ranges and preserve unselected robots."""

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.event_cfg import G1EventCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env_cfg import G1MovingPatchEnvCfg_PLAY
from isaaclab_tasks.utils.hydra import resolve_presets


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_g1_terrain_reset_uses_current_ranges(device):
    """Full and partial resets sample current bounds before a finite simulation step."""
    cfg = resolve_presets(G1MovingPatchEnvCfg_PLAY(), selected={"mjwarp_mpm_proxy"})
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.copy()
    cfg.sim.device = device
    cfg.sim.visualizer_cfgs = []
    cfg.sim.default_visualizer_cfg = None
    cfg.sim.physics.use_cuda_graph = True
    cfg.video_recorders = []
    cfg.events.mpm_material = G1EventCfg().mpm_material.copy()
    cfg.events.mpm_material.params["parameter_ranges"] = {"density": (2000.0, 2000.0)}
    cfg.events.reset_base.params["pose_range"] = {"x": (0.1, 0.1)}
    cfg.events.reset_base.params["velocity_range"] = {}

    with launch_simulation(cfg.sim, {"headless": True, "device": device}):
        env = gym.make("IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-Play", cfg=cfg)
        try:
            env.reset()
            raw = env.unwrapped
            robot = raw.scene["robot"]
            expected = robot.data.default_root_pose.torch[:, :3].clone() + raw.scene.env_origins
            expected[:, 0] += 0.1
            torch.testing.assert_close(robot.data.root_pos_w.torch, expected)

            term = raw.event_manager.get_term_cfg("reset_base")
            term.params["pose_range"] = {"x": (0.3, 0.3)}
            term.params["velocity_range"] = {"x": (0.2, 0.2)}
            before = robot.data.root_state_w.torch.clone()
            raw.reset(env_ids=torch.tensor([1], device=device))
            torch.testing.assert_close(robot.data.root_state_w.torch[0], before[0])
            expected[1, 0] += 0.2
            torch.testing.assert_close(robot.data.root_pos_w.torch, expected)
            torch.testing.assert_close(robot.data.root_lin_vel_w.torch[1], torch.tensor([0.2, 0.0, 0.0], device=device))

            before = robot.data.root_state_w.torch.clone()
            raw.reset(env_ids=slice(0, 1))
            torch.testing.assert_close(robot.data.root_state_w.torch[1], before[1])
            expected[0, 0] += 0.2
            torch.testing.assert_close(robot.data.root_pos_w.torch, expected)
            actions = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=device)
            for _ in range(3):
                observations, rewards, _, _, _ = env.step(actions)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in observations.values())
            assert torch.isfinite(robot.data.root_state_w.torch).all()

            # Shift one patch after graph replay has started, then replay again.
            patch = raw.moving_patch_particle
            before_centers = wp.to_torch(patch.patch_centers).clone()
            graph = NewtonManager._graph
            assert graph is not None
            term.params["pose_range"] = {"x": (1.1, 1.1)}
            particle_env_ids = wp.to_torch(patch.model.particle_world)
            dynamic_mass_before = wp.to_torch(patch.particle_dynamic_mass).clone()
            raw.event_manager.get_term_cfg("mpm_material").params["parameter_ranges"] = {
                "density": (2400.0, 2400.0),
                "friction": (0.45, 0.45),
            }
            raw.reset(env_ids=torch.tensor([1], device=device))
            observations, rewards, _, _, _ = env.step(actions)
            assert NewtonManager._graph is graph
            dynamic_mass = wp.to_torch(patch.particle_dynamic_mass)
            torch.testing.assert_close(dynamic_mass[particle_env_ids == 0], dynamic_mass_before[particle_env_ids == 0])
            radius = wp.to_torch(patch.model.particle_radius)[particle_env_ids == 1]
            torch.testing.assert_close(dynamic_mass[particle_env_ids == 1], 2400.0 * (8.0 * radius**3))
            assert wp.to_torch(patch.patch_centers)[1, 0] > before_centers[1, 0] + 0.4
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(value).all() for value in observations.values())
            torch.testing.assert_close(
                wp.to_torch(patch.model.particle_mass), wp.to_torch(patch.solver.model.particle_mass)
            )
            torch.testing.assert_close(
                wp.to_torch(patch.solver._mpm_model.particle_density),
                wp.to_torch(patch.model.particle_mass) / wp.to_torch(patch.solver._mpm_model.particle_volume),
            )

            # Enabling viscosity changes a captured solver path and must trigger recapture.
            NewtonMPMManager.set_particle_material_parameters({"viscosity": 1.0})
            observations, rewards, _, _, _ = env.step(actions)
            assert NewtonManager._graph is not graph
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(value).all() for value in observations.values())

            # Graph replay must not discard a terrain-query error between substeps.
            patch.terrain_query_miss_count.fill_(1)
            with pytest.raises(ValueError, match="Recycled particles left"):
                env.step(actions)
        finally:
            env.close()
