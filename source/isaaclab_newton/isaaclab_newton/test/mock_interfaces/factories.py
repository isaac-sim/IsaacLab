# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory functions for creating mock Newton views."""

from __future__ import annotations

from .views import MockNewtonArticulationView


def create_mock_articulation_view(
    count: int = 1,
    num_joints: int = 1,
    num_bodies: int = 2,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    is_fixed_base: bool = False,
    device: str = "cpu",
) -> MockNewtonArticulationView:
    """Create a mock Newton articulation view.

    Args:
        count: Number of articulation instances.
        num_joints: Number of degrees of freedom (joints).
        num_bodies: Number of bodies (links).
        joint_names: Names of the joints. Defaults to auto-generated names.
        body_names: Names of the bodies. Defaults to auto-generated names.
        is_fixed_base: Whether the articulation has a fixed base.
        device: Device for array allocation.

    Returns:
        A MockNewtonArticulationView instance.
    """
    return MockNewtonArticulationView(
        num_instances=count,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
    )


# -- Pre-configured factories --


def create_mock_quadruped_view(
    count: int = 1,
    device: str = "cpu",
) -> MockNewtonArticulationView:
    """Create a mock articulation view configured for a quadruped robot.

    Configuration:
        - 12 DOFs (3 per leg x 4 legs: hip, thigh, calf)
        - 13 links (base + 3 per leg)
        - Floating base

    Args:
        count: Number of articulation instances.
        device: Device for array allocation.

    Returns:
        A MockNewtonArticulationView configured for quadruped.
    """
    joint_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    body_names = [
        "base",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    ]
    return MockNewtonArticulationView(
        num_instances=count,
        num_bodies=13,
        num_joints=12,
        device=device,
        is_fixed_base=False,
        joint_names=joint_names,
        body_names=body_names,
    )


def create_mock_humanoid_view(
    count: int = 1,
    device: str = "cpu",
) -> MockNewtonArticulationView:
    """Create a mock articulation view configured for a humanoid robot.

    Configuration:
        - 21 DOFs (typical humanoid configuration)
        - 22 links
        - Floating base

    Args:
        count: Number of articulation instances.
        device: Device for array allocation.

    Returns:
        A MockNewtonArticulationView configured for humanoid.
    """
    joint_names = [
        # Torso
        "torso_joint",
        # Left arm
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        # Right arm
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        # Left leg
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        # Right leg
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
    ]
    body_names = [
        "pelvis",
        "torso",
        # Left arm
        "left_shoulder",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        # Right arm
        "right_shoulder",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        # Left leg
        "left_hip",
        "left_upper_leg",
        "left_lower_leg",
        "left_ankle",
        "left_foot",
        # Right leg
        "right_hip",
        "right_upper_leg",
        "right_lower_leg",
        "right_ankle",
        "right_foot",
        # Head
        "neck",
        "head",
    ]
    return MockNewtonArticulationView(
        num_instances=count,
        num_bodies=22,
        num_joints=21,
        device=device,
        is_fixed_base=False,
        joint_names=joint_names,
        body_names=body_names,
    )
