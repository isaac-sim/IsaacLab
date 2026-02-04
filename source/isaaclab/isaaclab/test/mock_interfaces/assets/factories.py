# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory functions for creating pre-configured mock assets."""

from __future__ import annotations

import torch

from .mock_articulation import MockArticulation
from .mock_rigid_object import MockRigidObject
from .mock_rigid_object_collection import MockRigidObjectCollection


def create_mock_articulation(
    num_instances: int = 1,
    num_joints: int = 1,
    num_bodies: int = 2,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    is_fixed_base: bool = False,
    device: str = "cpu",
) -> MockArticulation:
    """Create a mock articulation with default configuration.

    Args:
        num_instances: Number of articulation instances.
        num_joints: Number of joints.
        num_bodies: Number of bodies.
        joint_names: Names of joints.
        body_names: Names of bodies.
        is_fixed_base: Whether the articulation has a fixed base.
        device: Device for tensor allocation.

    Returns:
        Configured MockArticulation instance.
    """
    return MockArticulation(
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=num_bodies,
        joint_names=joint_names,
        body_names=body_names,
        is_fixed_base=is_fixed_base,
        device=device,
    )


def create_mock_quadruped(
    num_instances: int = 1,
    device: str = "cpu",
) -> MockArticulation:
    """Create a mock quadruped robot articulation.

    Creates a quadruped with 12 joints (3 per leg) and 13 bodies.

    Args:
        num_instances: Number of robot instances.
        device: Device for tensor allocation.

    Returns:
        Configured MockArticulation instance for a quadruped.
    """
    leg_names = ["FL", "FR", "RL", "RR"]
    joint_suffixes = ["hip", "thigh", "calf"]

    joint_names = [f"{leg}_{suffix}" for leg in leg_names for suffix in joint_suffixes]
    body_names = ["base"] + [f"{leg}_{part}" for leg in leg_names for part in ["hip", "thigh", "calf"]]

    robot = MockArticulation(
        num_instances=num_instances,
        num_joints=12,
        num_bodies=13,
        joint_names=joint_names,
        body_names=body_names,
        is_fixed_base=False,
        device=device,
    )

    # Set reasonable default joint limits for a quadruped
    joint_pos_limits = torch.zeros(num_instances, 12, 2, device=device)
    joint_pos_limits[..., 0] = -1.57  # Lower limit
    joint_pos_limits[..., 1] = 1.57  # Upper limit
    robot.data.set_joint_pos_limits(joint_pos_limits)

    return robot


def create_mock_humanoid(
    num_instances: int = 1,
    device: str = "cpu",
) -> MockArticulation:
    """Create a mock humanoid robot articulation.

    Creates a humanoid with 21 joints and 22 bodies.

    Args:
        num_instances: Number of robot instances.
        device: Device for tensor allocation.

    Returns:
        Configured MockArticulation instance for a humanoid.
    """
    # Simplified humanoid joint structure
    joint_names = [
        # Torso
        "torso_yaw",
        "torso_pitch",
        "torso_roll",
        # Left arm
        "L_shoulder_pitch",
        "L_shoulder_roll",
        "L_shoulder_yaw",
        "L_elbow",
        # Right arm
        "R_shoulder_pitch",
        "R_shoulder_roll",
        "R_shoulder_yaw",
        "R_elbow",
        # Left leg
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee",
        "L_ankle_pitch",
        # Right leg
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee",
        "R_ankle_pitch",
    ]

    body_names = [
        "pelvis",
        "torso",
        "L_upper_arm",
        "L_lower_arm",
        "L_hand",
        "R_upper_arm",
        "R_lower_arm",
        "R_hand",
        "L_thigh",
        "L_shin",
        "L_foot",
        "R_thigh",
        "R_shin",
        "R_foot",
        "head",
    ]

    return MockArticulation(
        num_instances=num_instances,
        num_joints=21,
        num_bodies=len(body_names),
        joint_names=joint_names,
        body_names=body_names,
        is_fixed_base=False,
        device=device,
    )


def create_mock_rigid_object(
    num_instances: int = 1,
    body_names: list[str] | None = None,
    device: str = "cpu",
) -> MockRigidObject:
    """Create a mock rigid object with default configuration.

    Args:
        num_instances: Number of rigid object instances.
        body_names: Names of bodies.
        device: Device for tensor allocation.

    Returns:
        Configured MockRigidObject instance.
    """
    return MockRigidObject(
        num_instances=num_instances,
        body_names=body_names,
        device=device,
    )


def create_mock_rigid_object_collection(
    num_instances: int = 1,
    num_bodies: int = 1,
    body_names: list[str] | None = None,
    device: str = "cpu",
) -> MockRigidObjectCollection:
    """Create a mock rigid object collection with default configuration.

    Args:
        num_instances: Number of environment instances.
        num_bodies: Number of bodies in the collection.
        body_names: Names of bodies.
        device: Device for tensor allocation.

    Returns:
        Configured MockRigidObjectCollection instance.
    """
    return MockRigidObjectCollection(
        num_instances=num_instances,
        num_bodies=num_bodies,
        body_names=body_names,
        device=device,
    )
