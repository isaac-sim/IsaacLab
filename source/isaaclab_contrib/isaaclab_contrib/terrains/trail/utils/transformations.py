# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Utilities to build 4x4 homogeneous transforms using the trimesh helpers.

For more information see:
    https://trimesh.org/trimesh.transformations.html
"""

import numpy as np
import trimesh


def identity() -> np.ndarray:
    """Returns a transform that maps an object into itself."""
    return trimesh.transformations.identity_matrix()


def translation(vec: tuple[float, float, float]) -> np.ndarray:
    """Return a transform that translates an object.

    Args:
        vec: Translation vector (x, y, z).
    """
    return trimesh.transformations.translation_matrix(vec)


def roll(angle: float) -> np.ndarray:
    """Return a transform that rotates (rolls) the object about the X axis.

    Args:
        angle: Roll angle in radians.
    """
    return trimesh.transformations.rotation_matrix(angle, [1.0, 0.0, 0.0])


def yaw(angle: float) -> np.ndarray:
    """Return a transform that rotates (yaws) the object about the Z axis.

    Args:
        angle: Yaw angle in radians.
    """
    return trimesh.transformations.rotation_matrix(angle, [0.0, 0.0, 1.0])


def translate_and_roll(vec: tuple[float, float, float], angle: float) -> np.ndarray:
    """Return a transform that translates then rolls the object.

    Args:
        vec: Translation vector (x, y, z).
        angle: Roll angle in radians.
    """
    T1 = translation(vec)
    T2 = roll(angle=angle)
    return trimesh.transformations.concatenate_matrices(T1, T2)


def scale(vec: tuple[float, float, float]) -> np.ndarray:
    """Return a non-uniform scaling transform.

    Args:
        vec: Scale factors along (x, y, z) axes.

    Returns:
        A 4x4 homogeneous transform that scales coordinates by the given
        factors.
    """
    T = identity()
    T[0, 0] = vec[0]
    T[1, 1] = vec[1]
    T[2, 2] = vec[2]
    return T
