# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_to_box_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    box_name: str,
    grasp_offset: torch.Tensor,  # shape: [num_envs, 3]
) -> torch.Tensor:
    """Compute distance between gripper and desired grasp point on box.

    Args:
        env: The environment instance
        robot_cfg: Configuration for the robot (specifies which gripper to use)
        box_name: Name of the box object
        grasp_offset: Offset from box center for desired grasp point [x,y,z]

    Returns:
        Distance between gripper and grasp point
    """
    # Get box pose
    box: RigidObject = env.scene[box_name]
    box_pos = box.data.root_state_w[:, :3]
    box_quat = box.data.root_state_w[:, 3:7]

    # Transform grasp offset to world frame
    grasp_pos_w, _ = combine_frame_transforms(box_pos, box_quat, grasp_offset)

    # Get gripper position
    robot: RigidObject = env.scene[robot_cfg.name]
    gripper_pos = robot.data.body_state_w[:, robot_cfg.body_ids[0], :3]  # type: ignore

    # TODO: Only compute box distance in X and Y, compare Z to height above table/robot base

    # Compute distance
    return torch.norm(gripper_pos - grasp_pos_w, dim=1)


class WaypointProgress(ManagerTermBase):
    """
    Tracks progression through a list of waypoint grasp offsets for each environment.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Expect lists of waypoints for each arm, which are [x, y, z] offsets from box center
        self.left_waypoints: list[list[float]] = cfg.params["left_waypoints"]
        self.right_waypoints: list[list[float]] = cfg.params["right_waypoints"]

        # Initialize current waypoint index tensor for each environment
        self.current_waypoints = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

        self.completion_threshold = cfg.params.get("completion_threshold", 0.05)
        self.left_robot_cfg = cfg.params["left_robot_cfg"]
        self.right_robot_cfg = cfg.params["right_robot_cfg"]
        self.box_name = cfg.params["box_name"]

        cfg.params = {}

    def reset(self, env_ids: torch.Tensor):
        """Reset waypoint progress for specified environments."""
        self.current_waypoints[env_ids] = 0

    def __call__(self, env, valid_env_ids: torch.Tensor):
        # Get current waypoints for each environment
        left_offsets = torch.tensor(self.left_waypoints, device=env.device)[
            self.current_waypoints
        ]
        right_offsets = torch.tensor(self.right_waypoints, device=env.device)[
            self.current_waypoints
        ]

        # Compute distances for each environment
        left_distances = gripper_to_box_distance(
            env,
            robot_cfg=self.left_robot_cfg,
            box_name=self.box_name,
            grasp_offset=left_offsets,  # Now a tensor of shape [num_envs, 3]
        )
        right_distances = gripper_to_box_distance(
            env,
            robot_cfg=self.right_robot_cfg,
            box_name=self.box_name,
            grasp_offset=right_offsets,  # Now a tensor of shape [num_envs, 3]
        )

        # Find environments where both grippers are at their waypoints
        both_at_waypoint = (left_distances < self.completion_threshold) & (
            right_distances < self.completion_threshold
        )

        # Increment waypoint index for those environments, but don't exceed max waypoints
        max_waypoint = len(self.left_waypoints) - 1
        self.current_waypoints[both_at_waypoint] = torch.clamp(
            self.current_waypoints[both_at_waypoint] + 1, max=max_waypoint
        )

        return self.current_waypoints


def gripper_to_dynamic_waypoint(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    box_name: str,
    is_left_arm: bool,
) -> torch.Tensor:
    """
    Compute distances to current waypoints for each environment.

    Args:
        env: The environment instance
        robot_cfg: Configuration for the robot arm
        box_name: Name of the box object
        is_left_arm: Whether this is for the left arm (True) or right arm (False)
    """
    # Get the waypoint progress manager from the event manager
    progress_manager = env.event_manager.get_term_cfg("waypoint_progress").func

    # Get waypoints list based on which arm we're computing for
    waypoints = (
        progress_manager.left_waypoints
        if is_left_arm
        else progress_manager.right_waypoints
    )

    # Convert waypoints to tensor and index using current waypoint for each env
    current_offsets = torch.tensor(waypoints, device=env.device)[
        progress_manager.current_waypoints
    ]

    return gripper_to_box_distance(env, robot_cfg, box_name, current_offsets)


def waypoint_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    progress_manager = env.event_manager.get_term_cfg("waypoint_progress").func
    return progress_manager.current_waypoints


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def object_is_lifted(
    env: ManagerBasedRLEnv, box_name: str, minimal_height: float
) -> torch.Tensor:
    """Reward the agent for lifting the box above the minimal height.

    Args:
        env: The environment instance
        box_name: Name of the box object
        minimal_height: Minimum height threshold for considering box as lifted

    Returns:
        Binary reward tensor (1.0 if lifted, 0.0 otherwise)
    """
    box: RigidObject = env.scene[box_name]
    return torch.where(box.data.root_state_w[:, 2] > minimal_height, 1.0, 0.0)


def box_height(
    env: ManagerBasedRLEnv, box_name: str, min_height: float
) -> torch.Tensor:
    """Continuous reward based on box height relative to minimum height.

    Args:
        env: The environment instance
        box_name: Name of the box object
        min_height: Minimum height threshold

    Returns:
        Height-based reward that is positive above min_height and negative below
    """
    box: RigidObject = env.scene[box_name]
    box_height = box.data.root_state_w[:, 2]  # z-coordinate
    height_diff = box_height - min_height
    # Positive reward above min_height, negative below
    return height_diff


def box_height_threshold(
    env: ManagerBasedRLEnv, box_name: str, min_height: float
) -> torch.Tensor:
    """Check if box has fallen below minimum height (for termination).

    Args:
        env: The environment instance
        box_name: Name of the box object
        min_height: Minimum allowed height

    Returns:
        Boolean tensor indicating if box is below minimum height
    """
    box: RigidObject = env.scene[box_name]
    box_height = box.data.root_state_w[:, 2]  # z-coordinate
    return box_height < min_height
