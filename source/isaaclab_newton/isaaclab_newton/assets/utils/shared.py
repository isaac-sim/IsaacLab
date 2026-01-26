# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from collections.abc import Sequence

import warp as wp

import isaaclab.utils.string as string_utils


def find_bodies(
    body_names: list[str],
    name_keys: str | Sequence[str],
    body_subset: list[str] | None = None,
    preserve_order: bool = False,
    device: str = "cuda:0",
) -> tuple[wp.array, list[str], list[int]]:
    """Find bodies in the articulation based on the name keys.

    Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
    information on the name matching.

    Args:
        body_names: The names of all the bodies in the articulation / assets.
        name_keys: A regular expression or a list of regular expressions to match the body names.
        body_subset: A subset of bodies to search for. Defaults to None, which means all bodies
            in the articulation are searched.
        preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.
        device: The device to use for the output mask. Defaults to "cuda:0".
    Returns:
        A tuple of lists containing the body mask, names, and indices.
    """
    # Use subset if provided, otherwise default to full list
    search_target = body_subset if body_subset is not None else body_names

    # Search in the target (subset or full)
    indices, names = string_utils.resolve_matching_names(name_keys, search_target, preserve_order)

    # If a subset was explicitly provided, map names back to global indices.
    # This block is skipped if body_subset is None, preserving zero overhead for default usage.
    if body_subset is not None:
        indices = [body_names.index(n) for n in names]

    # Create global mask
    mask = np.zeros(len(body_names), dtype=bool)
    mask[indices] = True
    mask = wp.array(mask, dtype=wp.bool, device=device)
    return mask, names, indices


def find_joints(
    joint_names: list[str],
    name_keys: str | Sequence[str],
    joint_subset: list[str] | None = None,
    preserve_order: bool = False,
    device: str = "cuda:0",
) -> tuple[wp.array, list[str], list[int]]:
    """Find joints in the articulation based on the name keys.

    Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
    on the name matching.

    Args:
        joint_names: The names of all the joints in the articulation / assets.
        name_keys: A regular expression or a list of regular expressions to match the joint names.
        joint_subset: A subset of joints to search for. Defaults to None, which means all joints
            in the articulation are searched.
        preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.
        device: The device to use for the output mask. Defaults to "cuda:0".
    Returns:
        A tuple of lists containing the joint mask, names, and indices.
    """
    # Use subset if provided, otherwise default to full list
    search_target = joint_subset if joint_subset is not None else joint_names

    # Search in the target (subset or full)
    indices, names = string_utils.resolve_matching_names(name_keys, search_target, preserve_order)

    # If a subset was explicitly provided, map names back to global indices.
    # This block is skipped if joint_subset is None, preserving zero overhead for default usage.
    if joint_subset is not None:
        indices = [joint_names.index(n) for n in names]

    # Create global mask
    mask = np.zeros(len(joint_names), dtype=bool)
    mask[indices] = True
    mask = wp.array(mask, dtype=wp.bool, device=device)
    return mask, names, indices
