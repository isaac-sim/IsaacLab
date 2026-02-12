# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX-specific benchmark utilities.

This module provides helper functions for creating tensor indices and warp masks
used in PhysX benchmark input generators.
"""

from __future__ import annotations

import torch
import warp as wp


def make_tensor_env_ids(num_instances: int, device: str) -> torch.Tensor:
    """Create a tensor of environment IDs.

    Args:
        num_instances: Number of environment instances.
        device: Device to create the tensor on.

    Returns:
        Tensor of environment IDs [0, 1, ..., num_instances-1].
    """
    return torch.arange(num_instances, dtype=torch.long, device=device)


def make_tensor_joint_ids(num_joints: int, device: str) -> torch.Tensor:
    """Create a tensor of joint IDs.

    Args:
        num_joints: Number of joints.
        device: Device to create the tensor on.

    Returns:
        Tensor of joint IDs [0, 1, ..., num_joints-1].
    """
    return torch.arange(num_joints, dtype=torch.long, device=device)


def make_tensor_body_ids(num_bodies: int, device: str) -> torch.Tensor:
    """Create a tensor of body IDs.

    Args:
        num_bodies: Number of bodies.
        device: Device to create the tensor on.

    Returns:
        Tensor of body IDs [0, 1, ..., num_bodies-1].
    """
    return torch.arange(num_bodies, dtype=torch.long, device=device)


def make_warp_env_mask(num_instances: int, device: str) -> wp.array:
    """Create an all-true environment mask.

    Args:
        num_instances: Number of environment instances.
        device: Device to create the mask on.

    Returns:
        Warp array of booleans, all set to True.
    """
    return wp.ones((num_instances,), dtype=wp.bool, device=device)


def make_warp_joint_mask(num_joints: int, device: str) -> wp.array:
    """Create an all-true joint mask.

    Args:
        num_joints: Number of joints.
        device: Device to create the mask on.

    Returns:
        Warp array of booleans, all set to True.
    """
    return wp.ones((num_joints,), dtype=wp.bool, device=device)


def make_warp_body_mask(num_bodies: int, device: str) -> wp.array:
    """Create an all-true body mask.

    Args:
        num_bodies: Number of bodies.
        device: Device to create the mask on.

    Returns:
        Warp array of booleans, all set to True.
    """
    return wp.ones((num_bodies,), dtype=wp.bool, device=device)
