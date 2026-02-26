# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions specific to locomanipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pick_place_table_frame(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = -0.20,
    max_y: float = 0.05,
    max_height: float = 1.10,
    min_vel: float = 0.20,
    debug: bool = False,
) -> torch.Tensor:
    """Determine if the object placement task is complete for locomanipulation environments.

    Unlike the base pick-place termination, this version checks object position relative
    to the destination table frame rather than the environment origin. This is necessary
    for locomanipulation tasks where the destination table can be placed at arbitrary
    world positions.

    This function checks whether all success conditions for the task have been met:
    1. Object is within the target x/y range relative to the destination table
    2. Object is below a maximum height relative to the destination table
    3. Object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x threshold in table frame)

    Args:
        env: The RL environment instance.
        task_link_name: Name of the right wrist link on the robot.
        object_cfg: Configuration for the object entity.
        table_cfg: Configuration for the destination table entity (must be an XformPrimView).
        right_wrist_max_x: Maximum x position of the right wrist in table frame for task completion.
        min_x: Minimum x position of the object relative to the table for task completion.
        max_x: Maximum x position of the object relative to the table for task completion.
        min_y: Minimum y position of the object relative to the table for task completion.
        max_y: Maximum y position of the object relative to the table for task completion.
        max_height: Maximum height (z) of the object relative to the table for task completion.
        min_vel: Maximum velocity magnitude of the object for task completion.
        debug: If True, print debug info for the first environment each step.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    object: RigidObject = env.scene[object_cfg.name]
    table = env.scene[table_cfg.name]
    if not isinstance(table, XformPrimView):
        raise TypeError(f"Expected table '{table_cfg.name}' to be an XformPrimView, got {type(table)}")

    # Get table world pose
    table_pos_w, table_quat_w = table.get_world_poses()

    # Broadcast table pose if a single table is shared across all envs
    object_root_pos_w = wp.to_torch(object.data.root_pos_w)  # [num_envs, 3]
    if table_pos_w.shape[0] == 1 and object_root_pos_w.shape[0] > 1:
        table_pos_w = table_pos_w.expand(object_root_pos_w.shape[0], -1)
        table_quat_w = table_quat_w.expand(object_root_pos_w.shape[0], -1)

    # Object position in table frame
    object_to_table_pos = quat_apply_inverse(table_quat_w, object_root_pos_w - table_pos_w)
    object_x = object_to_table_pos[:, 0]
    object_y = object_to_table_pos[:, 1]
    object_height = object_to_table_pos[:, 2]
    object_vel = torch.abs(wp.to_torch(object.data.root_vel_w))

    # Right wrist position in table frame
    robot_body_pos_w = wp.to_torch(env.scene["robot"].data.body_pos_w)
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_pos_w = robot_body_pos_w[:, right_eef_idx, :] - table_pos_w
    right_wrist_x = quat_apply_inverse(table_quat_w, right_wrist_pos_w)[:, 0]

    # Check all success conditions
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    if debug:
        env_id = 0
        obj_vel_env = object_vel[env_id]
        print(
            "[task_done_pick_place] "
            f"obj_pos_t=({object_x[env_id]:.3f}, {object_y[env_id]:.3f}, {object_height[env_id]:.3f}), "
            f"obj_vel=({obj_vel_env[0]:.3f}, {obj_vel_env[1]:.3f}, {obj_vel_env[2]:.3f}), "
            f"right_wrist_x={right_wrist_x[env_id]:.3f} | "
            f"x[{min_x:.3f},{max_x:.3f}] y[{min_y:.3f},{max_y:.3f}] "
            f"z<{max_height:.3f} wrist_x<{right_wrist_max_x:.3f} vel<{min_vel:.3f} "
            f"=> done={bool(done[env_id])}",
            flush=True,
        )

    return done


def object_too_far_from_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Terminate when the object is too far from the robot (failed to pick up).

    This checks the distance between the robot's root position and the object's position.
    If the distance exceeds max_distance, the episode is terminated as a failure.

    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot entity.
        object_cfg: Configuration for the object entity.
        max_distance: Maximum allowed distance between robot and object.

    Returns:
        Boolean tensor indicating which environments have exceeded the distance threshold.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    robot_pos = wp.to_torch(robot.data.root_pos_w)[:, :3]
    object_pos = wp.to_torch(object.data.root_pos_w)[:, :3]

    distance = torch.norm(robot_pos - object_pos, dim=1)

    return distance > max_distance
