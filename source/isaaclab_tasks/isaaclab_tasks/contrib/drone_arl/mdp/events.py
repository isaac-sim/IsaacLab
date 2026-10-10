# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions specific to the drone ARL environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .curriculums import get_obstacle_curriculum_term

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv


def reset_obstacles_with_individual_ranges(
    env: ManagerBasedRLEnv,
    env_mask: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    obstacle_configs: dict,
    wall_configs: dict,
    env_size: tuple[float, float, float],
    use_curriculum: bool = True,
    min_num_obstacles: int = 1,
    max_num_obstacles: int = 10,
    ground_offset: float = 0.1,
) -> None:
    """Reset obstacle and wall positions for specified environments without collision checking.

    This function repositions all walls and a curriculum-determined subset of obstacles
    within the specified environment bounds.

    Walls are positioned at fixed locations based on their configuration ratios. Obstacles
    are randomly placed within their designated zones, with the number of active obstacles
    determined by the curriculum difficulty level. Inactive obstacles are moved far below
    the scene (-1000m in Z) to effectively remove them from the environment.

    The curriculum scaling works as:
        num_obstacles = min + (difficulty / max_difficulty) * (max - min)

    Args:
        env: The manager-based RL environment instance.
        env_mask: Boolean mask of the environments to reset. Shape is (num_envs,).
        asset_cfg: Scene entity configuration identifying the obstacle collection.
        obstacle_configs: Dictionary mapping obstacle type names to their BoxCfg
            configurations, specifying size and placement ranges.
        wall_configs: Dictionary mapping wall names to their BoxCfg configurations.
        env_size: Tuple of (length, width, height) defining the environment bounds in meters.
        use_curriculum: If True, number of obstacles scales with curriculum difficulty.
            If False, spawns max_num_obstacles in every environment. Defaults to True.
        min_num_obstacles: Minimum number of obstacles to spawn per environment.
            Defaults to 1.
        max_num_obstacles: Maximum number of obstacles to spawn per environment.
            Defaults to 10.
        ground_offset: Z-axis offset to prevent obstacles from spawning at z=0.
            Defaults to 0.1 meters.

    Note:
        This function expects the environment to have `_obstacle_difficulty_levels` and
        `_max_obstacle_difficulty` attributes when `use_curriculum=True`. These are
        typically set by :func:`obstacle_density_curriculum`.
    """
    obstacles: RigidObjectCollection = env.scene[asset_cfg.name]

    num_objects = obstacles.num_objects
    num_envs = env.num_envs
    object_names = obstacles.object_names
    env_origins = env.scene.env_origins
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    hidden_offset = torch.tensor([0.0, 0.0, -1000.0], device=env.device)

    # Get difficulty levels per environment
    if use_curriculum:
        curriculum_term = get_obstacle_curriculum_term(env)
        if curriculum_term is not None:
            difficulty_levels = curriculum_term.difficulty_levels
            max_difficulty = curriculum_term.max_difficulty
        else:
            # Fallback: use max obstacles if curriculum not found
            difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
            max_difficulty = max_num_obstacles
    else:
        difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
        max_difficulty = max_num_obstacles

    # Calculate active obstacles per env based on difficulty
    obstacles_per_env = (
        min_num_obstacles + (difficulty_levels / max_difficulty) * (max_num_obstacles - min_num_obstacles)
    ).long()

    # Prepare tensors for all environments; only the selected ones are written
    all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
    all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)

    wall_names = list(wall_configs.keys())
    obstacle_types = list(obstacle_configs.values())
    env_size_t = torch.tensor(env_size, device=env.device)

    # place walls
    for wall_name, wall_cfg in wall_configs.items():
        if wall_name in object_names:
            wall_idx = object_names.index(wall_name)

            min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
            max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)

            if all(
                math.isclose(lo, hi, rel_tol=1e-5, abs_tol=1e-8)
                for lo, hi in zip(wall_cfg.center_ratio_min, wall_cfg.center_ratio_max)
            ):
                center_ratios = min_ratio.unsqueeze(0).repeat(num_envs, 1)
            else:
                ratios = torch.rand(num_envs, 3, device=env.device)
                center_ratios = ratios * (max_ratio - min_ratio) + min_ratio

            positions = (center_ratios - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            positions += env_origins

            all_poses[:, wall_idx, 0:3] = positions
            all_poses[:, wall_idx, 3:7] = identity_quat

    # Get obstacle indices
    obstacle_indices = [idx for idx, name in enumerate(object_names) if name not in wall_names]

    if len(obstacle_indices) > 0:
        # Activate a uniformly random subset of ``obstacles_per_env`` obstacles in each environment: the ranks of
        # i.i.d. uniform keys form a random permutation per environment.
        ranks = torch.rand(num_envs, len(obstacle_indices), device=env.device).argsort(dim=1).argsort(dim=1)
        active_masks = ranks < obstacles_per_env.unsqueeze(1)

        # place obstacles
        for obj_list_idx, obj_idx in enumerate(obstacle_indices):
            # Which envs need this obstacle?
            envs_need_obstacle = active_masks[:, obj_list_idx].unsqueeze(1)

            # Get obstacle config
            obs_cfg = obstacle_types[obj_list_idx % len(obstacle_types)]
            min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
            max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)

            # sample object positions and orientations
            ratios = torch.rand(num_envs, 3, device=env.device)
            positions = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            quats = math_utils.random_orientation(num_envs, device=env.device)

            # Move inactive obstacles far away
            all_poses[:, obj_idx, 0:3] = env_origins + torch.where(envs_need_obstacle, positions, hidden_offset)
            all_poses[:, obj_idx, 3:7] = torch.where(envs_need_obstacle, quats, identity_quat)

    # Write to sim
    obstacles.write_body_pose_to_sim_mask(body_poses=all_poses, env_mask=env_mask)
    obstacles.write_body_velocity_to_sim_mask(body_velocities=all_velocities, env_mask=env_mask)
