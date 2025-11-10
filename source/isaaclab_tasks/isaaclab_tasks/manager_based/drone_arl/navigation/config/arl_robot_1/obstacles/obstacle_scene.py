# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .obstacle_scene_cfg import ObstaclesSceneCfg

"""Obstacle scene generation and reset functionality for drone navigation environments.

This module provides utilities for generating dynamic 3D obstacle courses with walls and
floating obstacles. The obstacle configurations support curriculum learning where difficulty
can be progressively increased by adjusting the number of active obstacles.
"""

OBSTACLE_SCENE_CFG = ObstaclesSceneCfg(
    env_size=(12.0, 8.0, 6.0),
    min_num_obstacles=20,
    max_num_obstacles=40,
    ground_offset=3.0,
)


def generate_obstacle_collection(cfg: ObstaclesSceneCfg) -> RigidObjectCollectionCfg:
    """Generate a rigid object collection configuration for walls and obstacles.

    Creates a complete scene with boundary walls and a variety of floating obstacles
    (panels, cubes, rods, etc.) based on the provided configuration. Each obstacle is
    assigned random colors and configured with appropriate physics properties.

    Wall objects are configured with very high mass (10^7 kg) and high damping to remain
    stationary during collisions. Obstacle objects have moderate mass (100 kg) to move in the right position if reset
    in collision.

    Args:
        cfg: Configuration object specifying obstacle types, sizes, quantities, and
            positioning constraints.

    Returns:
        A RigidObjectCollectionCfg containing all wall and obstacle configurations,
        ready to be added to a scene.

    Note:
        All obstacles are initially placed at origin [0, 0, 0]. Actual positions are
        set during environment reset via :func:`reset_obstacles_with_individual_ranges`.
    """
    max_num_obstacles = cfg.max_num_obstacles

    rigid_objects = {}

    for wall_name, wall_cfg in cfg.wall_cfgs.items():
        # Walls get their specific size and default center
        default_center = [0.0, 0.0, 0.0]  # Will be set properly at reset
        color = float(np.random.randint(0, 256, size=1, dtype=np.uint8)) / 255.0

        rigid_objects[wall_name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obstacle_{wall_name}",
            spawn=sim_utils.CuboidCfg(
                size=wall_cfg.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, color), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    disable_gravity=True,
                    kinematic_enabled=False,
                    linear_damping=9999.0,
                    angular_damping=9999.0,
                    max_linear_velocity=0.0,
                    max_angular_velocity=0.0,
                ),
                # mass of walls needs to be way larger than weight of obstacles to make them not move during reset
                mass_props=sim_utils.MassPropertiesCfg(mass=10000000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=0,
        )

    obstacle_types = list(cfg.obstacle_cfgs.values())
    for i in range(max_num_obstacles):
        obj_name = f"obstacle_{i}"
        obs_cfg = obstacle_types[i % len(obstacle_types)]

        default_center = [0.0, 0.0, 0.0]
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        color_normalized = tuple(float(c) / 255.0 for c in color)

        rigid_objects[obj_name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{obj_name}",
            spawn=sim_utils.CuboidCfg(
                size=obs_cfg.size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_normalized, metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    disable_gravity=True,
                    kinematic_enabled=False,
                    linear_damping=1.0,
                    angular_damping=1.0,
                    max_linear_velocity=0.0,
                    max_angular_velocity=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(default_center)),
            collision_group=0,
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def reset_obstacles_with_individual_ranges(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
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
        env_ids: Tensor of environment indices to reset.
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
    num_envs = len(env_ids)
    object_names = obstacles.object_names

    # Get difficulty levels per environment
    if use_curriculum and hasattr(env, "_obstacle_difficulty_levels"):
        difficulty_levels = env._obstacle_difficulty_levels[env_ids]
        max_difficulty = env._max_obstacle_difficulty
    else:
        difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
        max_difficulty = max_num_obstacles

    # Calculate active obstacles per env based on difficulty
    obstacles_per_env = (
        min_num_obstacles + (difficulty_levels / max_difficulty) * (max_num_obstacles - min_num_obstacles)
    ).long()

    # Prepare tensors
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

            if torch.allclose(min_ratio, max_ratio):
                center_ratios = min_ratio.unsqueeze(0).repeat(num_envs, 1)
            else:
                ratios = torch.rand(num_envs, 3, device=env.device)
                center_ratios = ratios * (max_ratio - min_ratio) + min_ratio

            positions = (center_ratios - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            positions += env.scene.env_origins[env_ids]

            all_poses[:, wall_idx, 0:3] = positions
            all_poses[:, wall_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)

    # Get obstacle indices
    obstacle_indices = [idx for idx, name in enumerate(object_names) if name not in wall_names]

    if len(obstacle_indices) == 0:
        obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
        obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
        return

    # Determine which obstacles are active per env
    active_masks = torch.zeros(num_envs, len(obstacle_indices), dtype=torch.bool, device=env.device)
    for env_idx in range(num_envs):
        num_active = obstacles_per_env[env_idx].item()
        perm = torch.randperm(len(obstacle_indices), device=env.device)[:num_active]
        active_masks[env_idx, perm] = True

    # place obstacles
    for obj_list_idx in range(len(obstacle_indices)):
        obj_idx = obstacle_indices[obj_list_idx]

        # Which envs need this obstacle?
        envs_need_obstacle = active_masks[:, obj_list_idx]

        if not envs_need_obstacle.any():
            # Move all to -1000
            all_poses[:, obj_idx, 0:3] = env.scene.env_origins[env_ids] + torch.tensor(
                [0.0, 0.0, -1000.0], device=env.device
            )
            all_poses[:, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            continue

        # Get obstacle config
        config_idx = obj_list_idx % len(obstacle_types)
        obs_cfg = obstacle_types[config_idx]

        min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
        max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)

        # sample object positions
        num_active_envs = envs_need_obstacle.sum().item()
        ratios = torch.rand(num_active_envs, 3, device=env.device)
        positions = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
        positions[:, 2] += ground_offset

        # Add env origins
        active_env_indices = torch.where(envs_need_obstacle)[0]
        positions += env.scene.env_origins[env_ids[active_env_indices]]

        # Generate quaternions
        quats = math_utils.random_orientation(num_envs, device=env.device)

        # Write poses
        all_poses[envs_need_obstacle, obj_idx, 0:3] = positions
        all_poses[envs_need_obstacle, obj_idx, 3:7] = quats[envs_need_obstacle]

        # Move inactive obstacles far away
        inactive = ~envs_need_obstacle
        all_poses[inactive, obj_idx, 0:3] = env.scene.env_origins[env_ids[inactive]] + torch.tensor(
            [0.0, 0.0, -1000.0], device=env.device
        )
        all_poses[inactive, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
