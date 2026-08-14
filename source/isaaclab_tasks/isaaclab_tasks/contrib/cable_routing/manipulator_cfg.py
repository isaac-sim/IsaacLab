# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-facing contract for the bimanual cable-routing task."""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class CableRoutingManipulatorCfg:
    """Describe one manipulator without coupling task logic to a robot model."""

    asset_name: str = MISSING
    """Scene entity containing the robot articulation."""

    arm_joint_names: list[str] = MISSING
    """Ordered names or expressions for arm joints controlled during reset."""

    gripper_joint_names: list[str] = MISSING
    """Ordered gripper joints written from one scalar opening command."""

    gripper_joint_multipliers: list[float] = MISSING
    """Multiplier from the scalar opening command to each gripper joint."""

    end_effector_body_name: str = MISSING
    """Body carrying the task contact frame."""

    contact_frame_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Contact-frame translation from the end-effector body frame [m]."""

    contact_frame_offset_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Contact-frame rotation from the end-effector body frame as ``(x, y, z, w)``."""

    arm_action_name: str = MISSING
    """Action-manager term controlling the arm."""

    gripper_action_name: str = MISSING
    """Action-manager term controlling the gripper."""

    cage_gripper_joint_position: float = MISSING
    """Scalar gripper command used to cage the cable [m or rad, depending on joint type]."""

    def __post_init__(self) -> None:
        """Validate names, dimensions, and frame values."""
        for name in (
            "asset_name",
            "end_effector_body_name",
            "arm_action_name",
            "gripper_action_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not self.arm_joint_names or not all(isinstance(name, str) and name for name in self.arm_joint_names):
            raise ValueError("arm_joint_names must contain at least one non-empty name or expression.")
        if not self.gripper_joint_names or not all(isinstance(name, str) and name for name in self.gripper_joint_names):
            raise ValueError("gripper_joint_names must contain at least one non-empty name or expression.")
        if len(self.gripper_joint_names) != len(self.gripper_joint_multipliers):
            raise ValueError("gripper_joint_names and gripper_joint_multipliers must have equal length.")
        if not all(math.isfinite(multiplier) for multiplier in self.gripper_joint_multipliers):
            raise ValueError("gripper_joint_multipliers must contain only finite values.")
        if self.gripper_joint_multipliers[0] == 0.0:
            raise ValueError("The first gripper joint multiplier must be non-zero.")
        if len(self.contact_frame_offset_pos) != 3 or not all(
            math.isfinite(value) for value in self.contact_frame_offset_pos
        ):
            raise ValueError("contact_frame_offset_pos must contain three finite values.")
        if len(self.contact_frame_offset_quat) != 4 or not all(
            math.isfinite(value) for value in self.contact_frame_offset_quat
        ):
            raise ValueError("contact_frame_offset_quat must contain four finite values.")
        quaternion_norm = math.sqrt(sum(value * value for value in self.contact_frame_offset_quat))
        if quaternion_norm <= 1.0e-8:
            raise ValueError("contact_frame_offset_quat must have non-zero norm.")
        if not math.isfinite(self.cage_gripper_joint_position):
            raise ValueError("cage_gripper_joint_position must be finite.")


def validate_bimanual_manipulators(
    manipulators: tuple[CableRoutingManipulatorCfg, ...],
) -> tuple[CableRoutingManipulatorCfg, CableRoutingManipulatorCfg]:
    """Validate and return the ordered two-manipulator task interface."""
    if len(manipulators) != 2:
        raise ValueError(f"Cable routing requires exactly two manipulators; got {len(manipulators)}.")
    if len({manipulator.asset_name for manipulator in manipulators}) != 2:
        raise ValueError("Cable-routing manipulator asset names must be unique.")
    action_names = [
        action_name
        for manipulator in manipulators
        for action_name in (manipulator.arm_action_name, manipulator.gripper_action_name)
    ]
    if len(set(action_names)) != len(action_names):
        raise ValueError("Cable-routing manipulator action names must be unique.")
    return manipulators[0], manipulators[1]
