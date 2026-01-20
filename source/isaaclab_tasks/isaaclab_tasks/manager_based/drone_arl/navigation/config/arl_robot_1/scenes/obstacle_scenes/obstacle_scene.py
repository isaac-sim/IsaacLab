# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

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
