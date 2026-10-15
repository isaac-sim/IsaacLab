# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 terrain resets honor current ranges and preserve unselected robots."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.app import launch_simulation
from isaaclab.terrains import FlatPatchSamplingCfg, HfPyramidSlopedTerrainCfg, TerrainGeneratorCfg
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.event_cfg import G1EventCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env_cfg import (
    G1MixedTerrainEnvCfg,
    G1MixedTerrainEnvCfg_PLAY,
)
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env_cfg import G1MovingPatchEnvCfg_PLAY
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.terrain import PairedTerrainGenerator, split_contact_surfaces
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_mixed_terrain_fixed_foot_margins(device):
    """Both solver models use constant per-terrain foot margins across resets."""
    cfg = resolve_presets(G1MixedTerrainEnvCfg_PLAY(), [])
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.replace(num_rows=1, num_cols=3, border_width=1.0)
    cfg.mpm_contact_margin = 0.02
    cfg.rigid_contact_margin = 0.0
    cfg.sim.device = device
    cfg.sim.visualizer_cfgs = []
    cfg.sim.default_visualizer_cfg = None
    cfg.video_recorders = []
    cfg.events.mpm_material = G1MixedTerrainEnvCfg().events.mpm_material.copy()
    cfg.events.mpm_material.params["parameter_ranges"] = {"density": (2000.0, 2000.0)}
    with launch_simulation(cfg, {"headless": True, "device": device}):
        env = gym.make("IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain-Play", cfg=cfg)
        try:
            env.reset()
            models = [NewtonManager.get_model()]
            models.extend(entry.solver.model for entry in NewtonManager._solver._entries.values())
            expected = torch.tensor([0.02, 0.02, 0.0, 0.0], device=device)
            saved_margins = []
            for model in models:
                shape_bodies = model.shape_body.numpy()
                foot_shapes = [
                    i
                    for i, body in enumerate(shape_bodies)
                    if body >= 0 and model.body_label[body].endswith("_ankle_roll_link")
                ]
                world_ids = wp.to_torch(model.body_world)[shape_bodies[foot_shapes]].long()
                assert set(world_ids.tolist()) == {0, 1, 2, 3}
                margins = wp.to_torch(model.shape_margin)
                torch.testing.assert_close(margins[foot_shapes], expected[world_ids])
                saved_margins.append(margins.clone())

            actions = torch.zeros((4, env.unwrapped.action_manager.total_action_dim), device=device)
            raw = env.unwrapped
            assert set(wp.to_torch(NewtonManager.get_model().particle_world).tolist()) == {0, 1}
            material = raw.event_manager.get_term_cfg("mpm_material").func
            torch.testing.assert_close(material.material_parameters["density"], torch.full((2,), 2000.0, device=device))
            peak_force = torch.zeros(4, device=device)
            peak_mpm_force = torch.zeros(4, device=device)
            for _ in range(24):
                observations, rewards, _, _, _ = env.step(actions)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in observations.values())
                peak_force = torch.maximum(peak_force, raw.foot_contact_force[..., 2].sum(dim=1))
                mpm_force = wp.to_torch(raw._coupling_forces)[raw._coupling_force_body_ids, 2].sum(dim=1)
                peak_mpm_force = torch.maximum(peak_mpm_force, mpm_force)
            assert (peak_force > 20.0).all()
            assert (peak_mpm_force[:2] > 20.0).all()
            torch.testing.assert_close(peak_mpm_force[2:], torch.zeros(2, device=device))
            raw.event_manager.get_term_cfg("mpm_material").params["parameter_ranges"] = {"density": (2400.0, 2400.0)}
            raw.reset(env_ids=torch.tensor([0, 2], device=device))
            torch.testing.assert_close(
                material.material_parameters["density"], torch.tensor([2400.0, 2000.0], device=device)
            )
            for model, previous in zip(models, saved_margins):
                torch.testing.assert_close(wp.to_torch(model.shape_margin), previous)
        finally:
            env.close()


def test_paired_terrain_preserves_geometry_and_spawn_locations():
    """Both contact groups receive identical sampled heights and spawn points, even with three columns."""
    cfg = TerrainGeneratorCfg(
        seed=42,
        size=(4.0, 4.0),
        num_rows=2,
        num_cols=3,
        border_width=0.0,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "flat": HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.1, 0.5),
                platform_width=1.0,
                flat_patch_sampling={
                    "spawn": FlatPatchSamplingCfg(
                        num_patches=2, patch_radius=0.1, max_height_diff=0.01, x_range=(-0.3, 0.3), y_range=(-0.3, 0.3)
                    )
                },
            ),
        },
    )
    terrain = PairedTerrainGenerator(cfg, device="cpu")
    assert cfg.num_cols == 3
    assert terrain.terrain_origins.shape == (2, 6, 3)
    translation = np.array([0.0, 12.0, 0.0])
    np.testing.assert_allclose(terrain.terrain_origins[:, 3:] - translation, terrain.terrain_origins[:, :3])
    assert np.ptp(terrain.terrain_origins[..., 2]) > 0.1
    patches = terrain.flat_patches["spawn"]
    torch.testing.assert_close(patches[:, 3:] - torch.tensor(translation, dtype=patches.dtype), patches[:, :3])
    sand_floor, rigid_surface = split_contact_surfaces(terrain.terrain_mesh, particle_depth=0.25)
    # Compare world-space geometry after accounting for only the group translation and sand depth.
    sand_vertices = np.unique(np.round(sand_floor.vertices, 6), axis=0)
    rigid_vertices = np.unique(np.round(rigid_surface.vertices - translation - [0.0, 0.0, 0.25], 6), axis=0)
    np.testing.assert_allclose(sand_vertices, rigid_vertices, atol=1e-6)
