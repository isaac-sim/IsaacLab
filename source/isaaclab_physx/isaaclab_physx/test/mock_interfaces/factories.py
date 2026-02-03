# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory functions for creating mock PhysX views."""

from __future__ import annotations

from .views import MockArticulationView, MockRigidBodyView, MockRigidContactView


def create_mock_rigid_body_view(
    count: int = 1,
    prim_paths: list[str] | None = None,
    device: str = "cpu",
) -> MockRigidBodyView:
    """Create a mock rigid body view.

    Args:
        count: Number of rigid body instances.
        prim_paths: USD prim paths for each instance.
        device: Device for tensor allocation.

    Returns:
        A MockRigidBodyView instance.
    """
    return MockRigidBodyView(count=count, prim_paths=prim_paths, device=device)


def create_mock_articulation_view(
    count: int = 1,
    num_dofs: int = 1,
    num_links: int = 2,
    dof_names: list[str] | None = None,
    link_names: list[str] | None = None,
    fixed_base: bool = False,
    prim_paths: list[str] | None = None,
    device: str = "cpu",
) -> MockArticulationView:
    """Create a mock articulation view.

    Args:
        count: Number of articulation instances.
        num_dofs: Number of degrees of freedom (joints).
        num_links: Number of links (bodies).
        dof_names: Names of the DOFs.
        link_names: Names of the links.
        fixed_base: Whether the articulation has a fixed base.
        prim_paths: USD prim paths for each instance.
        device: Device for tensor allocation.

    Returns:
        A MockArticulationView instance.
    """
    return MockArticulationView(
        count=count,
        num_dofs=num_dofs,
        num_links=num_links,
        dof_names=dof_names,
        link_names=link_names,
        fixed_base=fixed_base,
        prim_paths=prim_paths,
        device=device,
    )


def create_mock_rigid_contact_view(
    count: int = 1,
    num_bodies: int = 1,
    filter_count: int = 0,
    max_contact_data_count: int = 16,
    device: str = "cpu",
) -> MockRigidContactView:
    """Create a mock rigid contact view.

    Args:
        count: Number of instances.
        num_bodies: Number of bodies per instance.
        filter_count: Number of filter bodies for contact filtering.
        max_contact_data_count: Maximum number of contact data points.
        device: Device for tensor allocation.

    Returns:
        A MockRigidContactView instance.
    """
    return MockRigidContactView(
        count=count,
        num_bodies=num_bodies,
        filter_count=filter_count,
        max_contact_data_count=max_contact_data_count,
        device=device,
    )


# -- Pre-configured factories --


def create_mock_quadruped_view(
    count: int = 1,
    device: str = "cpu",
) -> MockArticulationView:
    """Create a mock articulation view configured for a quadruped robot.

    Configuration:
        - 12 DOFs (3 per leg x 4 legs: hip, thigh, calf)
        - 13 links (base + 3 per leg)
        - Floating base

    Args:
        count: Number of articulation instances.
        device: Device for tensor allocation.

    Returns:
        A MockArticulationView configured for quadruped.
    """
    dof_names = [
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
    link_names = [
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
    return MockArticulationView(
        count=count,
        num_dofs=12,
        num_links=13,
        dof_names=dof_names,
        link_names=link_names,
        fixed_base=False,
        device=device,
    )


def create_mock_humanoid_view(
    count: int = 1,
    device: str = "cpu",
) -> MockArticulationView:
    """Create a mock articulation view configured for a humanoid robot.

    Configuration:
        - 21 DOFs (typical humanoid configuration)
        - 22 links
        - Floating base

    Args:
        count: Number of articulation instances.
        device: Device for tensor allocation.

    Returns:
        A MockArticulationView configured for humanoid.
    """
    dof_names = [
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
    link_names = [
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
    return MockArticulationView(
        count=count,
        num_dofs=21,
        num_links=22,
        dof_names=dof_names,
        link_names=link_names,
        fixed_base=False,
        device=device,
    )
