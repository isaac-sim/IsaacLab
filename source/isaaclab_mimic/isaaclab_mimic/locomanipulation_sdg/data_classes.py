# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass


@dataclass
class LocomanipulationSDGInputData:
    """Data container for in-place manipulation recording state.  Used during locomanipulation replay."""

    left_hand_pose_target: torch.Tensor
    """The pose of the left hand in world coordinates."""

    right_hand_pose_target: torch.Tensor
    """The pose of the right hand in world coordinates."""

    left_hand_joint_positions_target: torch.Tensor
    """The left hand joint positions."""

    right_hand_joint_positions_target: torch.Tensor
    """The right hand joint positions."""

    base_pose: torch.Tensor
    """The robot base pose in world coordinates."""

    object_pose: torch.Tensor
    """The target object pose in world coordinates."""

    fixture_pose: torch.Tensor
    """The fixture (ie: table) pose in world coordinates."""


@dataclass
class LocomanipulationSDGOutputData:
    """A container for data that is recorded during locomanipulation replay.  This is the final output of the pipeline"""

    left_hand_pose_target: torch.Tensor | None = None
    """The left hand's target pose."""

    right_hand_pose_target: torch.Tensor | None = None
    """The right hand's target pose."""

    left_hand_joint_positions_target: torch.Tensor | None = None
    """The left hand's target joint positions"""

    right_hand_joint_positions_target: torch.Tensor | None = None
    """The right hand's target joint positions"""

    base_velocity_target: torch.Tensor | None = None
    """The target velocity of the robot base.  This value is provided to the underlying base controller or policy."""

    start_fixture_pose: torch.Tensor | None = None
    """The pose of the start fixture (ie: pick-up table)."""

    end_fixture_pose: torch.Tensor | None = None
    """The pose of the end / destination fixture (ie: drop-off table)"""

    object_pose: torch.Tensor | None = None
    """The pose of the target object."""

    base_pose: torch.Tensor | None = None
    """The pose of the robot base."""

    data_generation_state: int | None = None
    """The state of the the locomanipulation SDG replay script's state machine."""

    base_goal_pose: torch.Tensor | None = None
    """The goal pose of the robot base (ie: the final destination before dropping off the object)"""

    base_goal_approach_pose: torch.Tensor | None = None
    """The goal pose provided to the path planner (this may be offset from the final destination to enable approach.)"""

    base_path: torch.Tensor | None = None
    """The robot base path as determined by the path planner."""

    recording_step: int | None = None
    """The current recording step used for upper body replay."""

    obstacle_fixture_poses: torch.Tensor | None = None
    """The pose of all obstacle fixtures in the scene."""
