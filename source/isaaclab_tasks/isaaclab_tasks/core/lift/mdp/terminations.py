# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the rigid and deformable lift tasks.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

from .curriculums import lift_difficulty_fraction


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    # Convert to torch for combine_frame_transforms (robot data may be Warp arrays under Newton)
    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    object_pos_w = object.data.root_pos_w.torch
    distance = torch.linalg.norm(des_pos_w - object_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


class ObjectPoseHeld(ManagerTermBase):
    """Terminate after the object continuously holds the commanded pose."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        hold_time = cfg.params.get("hold_time", 1.0)
        if hold_time <= 0.0:
            raise ValueError(f"hold_time must be positive, got {hold_time}.")
        self.hold_steps = max(1, math.ceil(hold_time / env.step_dt))
        self.consecutive_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.position_error = torch.full((env.num_envs,), float("inf"), device=env.device)
        self.orientation_error = torch.full((env.num_envs,), float("inf"), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear held-pose progress for the reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.consecutive_steps[env_ids] = 0
        self.position_error[env_ids] = float("inf")
        self.orientation_error[env_ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        position_threshold: float,
        orientation_threshold: float,
        hold_time: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Update held-pose progress and return the sustained success signal."""
        del hold_time
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        goal_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            command[:, :3],
        )
        goal_quat_w = quat_mul(robot.data.root_quat_w.torch, command[:, 3:7])
        self.position_error = torch.linalg.norm(goal_pos_w - object.data.root_pos_w.torch, dim=1)
        self.orientation_error = quat_error_magnitude(object.data.root_quat_w.torch, goal_quat_w)
        within_tolerance = (self.position_error <= position_threshold) & (
            self.orientation_error <= orientation_threshold
        )
        self.consecutive_steps = torch.where(
            within_tolerance,
            self.consecutive_steps + 1,
            torch.zeros_like(self.consecutive_steps),
        )
        return self.consecutive_steps >= self.hold_steps


def curriculum_object_below_reset_height(
    env: ManagerBasedRLEnv,
    high_object_height: float,
    low_object_height: float,
    transition_start: float,
    transition_end: float,
    height_margin: float,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate failed pre-grasped episodes before they exploit table-level shaping."""
    object: RigidObject = env.scene[object_cfg.name]
    difficulty = lift_difficulty_fraction(env, slice(None))
    blend = ((difficulty - transition_start) / (transition_end - transition_start)).clamp(0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    expected_height = high_object_height + blend * (low_object_height - high_object_height)
    threshold = (expected_height - height_margin).clamp(min=minimum_height)
    object_height = object.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return object_height < threshold


def deformable_com_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Termination signal when the deformable's COM falls below ``minimum_height`` [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = wp.to_torch(asset.data.root_pos_w)[:, 2]
    return com_z < minimum_height


def deformable_outside_table_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate if any deformable nodal point leaves the table footprint.

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = wp.to_torch(asset.data.nodal_pos_w) - env.scene.env_origins.unsqueeze(1)
    outside_x = (nodal_pos[..., 0] < x_bounds[0]) | (nodal_pos[..., 0] > x_bounds[1])
    outside_y = (nodal_pos[..., 1] < y_bounds[0]) | (nodal_pos[..., 1] > y_bounds[1])
    return torch.any(outside_x | outside_y, dim=1)


def ee_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Termination signal when the end-effector falls below ``minimum_height`` [m].

    Height is measured in the environment frame (``z`` of the EE position with the env
    origin subtracted), so the threshold is independent of the environment's xy offset.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed EE height in the environment frame [m].
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    return ee_z < minimum_height
