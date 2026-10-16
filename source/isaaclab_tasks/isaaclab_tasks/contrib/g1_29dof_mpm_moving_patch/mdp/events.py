# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms specific to the granular-bed locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.envs.mdp.events import randomize_mpm_material as randomize_newton_mpm_material

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg

from ..util.terrain import BackgroundTerrainImporterCfg

if TYPE_CHECKING:
    from ..mpm_env import G1MovingPatchEnv


class randomize_mpm_material(randomize_newton_mpm_material):
    """Keep moving-patch recycling masses consistent with generic MPM material randomization."""

    def __call__(
        self,
        env: G1MovingPatchEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        parameter_ranges: dict[str, tuple[float, float]],
        distribution: str = "uniform",
    ) -> None:
        super().__call__(env, env_ids, asset_cfg, parameter_ranges, distribution)
        if "density" in parameter_ranges:
            ids = slice(None) if env_ids is None else env_ids
            env.moving_patch_particle.set_dynamic_particle_density(self.material_parameters["density"][ids], env_ids)


class reset_root_state_on_terrain(reset_root_state_uniform):
    """Reset robots above the terrain height sampled at their randomized world XY.

    Pose and velocity are sampled from the current configured ranges. The sampled terrain height
    replaces the environment origin's Z offset, preserving the default root clearance [m].
    Regular terrain importers use the environment origin's Z offset instead.
    """

    def __call__(
        self,
        env: G1MovingPatchEnv,
        env_ids: torch.Tensor | slice,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Sample root pose and velocity, then write the terrain-adjusted state once.

        Args:
            env: Moving-patch environment with a shared terrain surface.
            env_ids: Environment indices or a slice selecting robots to reset.
            pose_range: Pose offsets [m or rad, depending on axis], cached at construction.
            velocity_range: Velocity offsets [m/s or rad/s], cached at construction.
            asset_cfg: Robot selection.
        """
        asset: Articulation = env.scene[asset_cfg.name]
        default_root_pose = asset.data.default_root_pose.torch[env_ids]
        default_root_vel = asset.data.default_root_vel.torch[env_ids]
        if default_root_pose.shape[0] == 0:
            return

        rand_samples = math_utils.sample_uniform_from_ranges(
            pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), default_root_pose.shape[0], device=asset.device
        )
        positions = default_root_pose[:, :3] + rand_samples[:, :3]
        positions[:, :2] += env.scene.env_origins[env_ids][:, :2]
        if isinstance(env.scene.terrain.cfg, BackgroundTerrainImporterCfg):
            points = wp.from_torch(positions.clone().contiguous(), dtype=wp.vec3)
            env.scene.terrain.background_mesh.sample_surface_heights(points)
            positions[:, 2] += wp.to_torch(points)[:, 2]
        else:
            positions[:, 2] += env.scene.env_origins[env_ids][:, 2]

        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
        rand_samples = math_utils.sample_uniform_from_ranges(
            velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), default_root_pose.shape[0], device=asset.device
        )
        velocities = default_root_vel + rand_samples
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
