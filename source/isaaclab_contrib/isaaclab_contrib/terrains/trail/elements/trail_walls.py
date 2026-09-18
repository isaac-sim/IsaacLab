# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
This module contains functions to generate 2D wall profiles. Walls are used
to separate the trail from the surrounding environment.

When creating new wall functions, consider the following:
    * Walls are defined in the yz plane.
    * The y axis points to the right and the z axis points upward.
    * The wall's anchor point is at (y, z) = (0, 0).
    * The wall is parameterized as a NumPy array with shape (N, 2).
    * The function signature for each wall type should be::

        def function_name(wall_width: float, wall_height: float, num_segments: int):
            Args:
                wall_width: Width of the wall in the y direction.
                wall_height: Height of the wall in the z direction.
                num_segments: Number of segments used to parameterize the wall.

            Returns:
                y coordinates of the wall with respect to the anchor point,
                z coordinates of the wall with respect to the anchor point,
                a bool indicating if the wall is smooth (i.e., can be driven onto).

    Example wall shape produced by the functions below::

            z
            ^
            |            +----- wall_height
            |          +      |
            |        +        |
            +-------+---------+------------> y
                    y=0      wall_width
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def no_wall(wall_width: float, wall_height: float, num_segments: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a tiny, flat wall representation (effectively no wall).

    Uses the minimum number of segments needed to preserve sharp color transitions when the mesh is generated.
    """
    num_segments = 2  # minimum number required for a sharp color transition
    wall_height = 0.0
    wall_width = 0.1  # keep non-zero to avoid accidental vertex merging
    yi = np.linspace(0.0, 1.0, num_segments)
    zi = np.linspace(0.0, 1.0, num_segments)
    return (yi * wall_width, zi * wall_height, True)


def linear_wall(wall_width: float, wall_height: float, num_segments: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a linearly sloped wall (constant slope)."""
    yi = np.linspace(0.0, 1.0, num_segments)
    zi = np.linspace(0.0, 1.0, num_segments)
    return (yi * wall_width, zi * wall_height, True)


def half_cos_wall(wall_width: float, wall_height: float, num_segments: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a wall modeled as a half-cosine that smoothly integrates with the trail."""
    yi = np.linspace(0.0, 1.0, num_segments)
    zi = 1.0 - np.cos(0.5 * np.pi * yi)
    return (yi * wall_width, zi * wall_height, True)


def cos_wall(wall_width: float, wall_height: float, num_segments: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a wall modeled as a cosine wave that smoothly integrates with the trail and border."""
    yi = np.linspace(0.0, 1.0, num_segments)
    zi = 0.5 * (1.0 - np.cos(np.pi * yi))
    return (yi * wall_width, zi * wall_height, True)


def circular_wall(wall_width: float, wall_height: float, num_segments: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a wall modeled as a quarter-circle segment."""
    si = np.linspace(0.0, 1.0, num_segments)
    yi = np.sin(si * np.pi * 0.5)
    zi = 1.0 - np.cos(si * np.pi * 0.5)
    return (yi * wall_width, zi * wall_height, True)


def gaussian_wall(wall_width: float, wall_height: float, num_segments: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return a wall with a globally linear slope and local Gaussian roughness.

    The returned wall is not considered "smooth" for driving onto.
    """
    # Noise parameters
    max_n = 0.2
    sigma = 0.5
    # Baseline linear ramp from trail edge to floor edge.
    yi = np.linspace(0.0, 1.0, num_segments)
    # Add zero-mean noise before smoothing.
    noise = np.random.normal(0.0, max_n, size=num_segments)
    zi = yi + noise
    # Repeated smoothing produces gentle undulations rather than sharp spikes.
    for _ in range(num_segments):
        zi = gaussian_filter(zi, sigma=sigma)
    # Normalize so the profile starts at 0 and ends at 1.
    zi -= zi[0]
    if zi[-1] == 0.0:
        return gaussian_wall(wall_width, wall_height, num_segments)
    zi /= zi[-1]
    return (yi * wall_width, zi * wall_height, False)
