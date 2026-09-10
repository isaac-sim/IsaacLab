# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
This module contains functions to generate 2D profiles that represent the
shape of trail objects. Trail objects are placed inside the trail to simulate
obstacles.

When creating new objects, consider the following:
     * Objects are defined in either the xz or yz plane.
     * The x (or y) axis points to the right and the z axis points downward
        (z coordinates are flipped).
     * The object's anchor point is at (x/y, z) = (0, 0).
     * The object is parameterized as a NumPy array with shape (N, 2).
     * The outline of the object is not closed (i.e., the first and last point
        are not identical).

                    x/y=0
         +--------+----------------------> x or y
         |        |
         |        |
      z=0+--------+ (beginning of object)
         |
         |
         z
"""

import math
import random
from typing import Literal

import numpy as np

from ..utils.math import sample


def curved_ramp_profile(
    length: float,
    height: float,
    num_segments: int,
    type: Literal["cos", "half-cos", "rand"],
) -> np.ndarray:
    """Helper to generate a cosine-shaped ramp profile.

    Args:
        length: Length of the ramp, from entry point to exit point [m].
        height: Height of the element, reached at the peak point [m].
        num_segments: Number of segments used to parametrize the surface.
        type: Type of the ramp. Can be:
            * ``"cos"``: smooth connection to floor and top
            * ``"half-cos"``: smooth connection to floor only
            * ``"rand"``: randomly picks one of the above options

    Returns:
        A polygon specifying the profile outline, parameterized as a NumPy array
        with shape (N, 2).
    """
    s = np.linspace(0.0, 1.0, (num_segments if length > 0.0 else 2))
    xi = s * length
    if type == "rand":
        type = random.choice(["cos", "half-cos"])
    if type == "cos":
        zi = 0.5 * (np.cos(np.pi * s) - 1.0) * height
    elif type == "half-cos":
        zi = (np.cos(0.5 * np.pi * s) - 1.0) * height
    else:
        raise RuntimeError("Unknown type argument.")
    return np.stack([xi, zi], axis=1)


def wave_profile(
    length: float,
    height: float,
    num_segments: int,
    platform_length: float = 0.0,
    gap: bool | tuple[bool, bool] = False,
    exponent: int | tuple[int, int] = 2,
    type: Literal["cos", "half-cos", "rand"] = "cos",
) -> np.ndarray:
    """Helper to generate a cosine-shaped wave profile consisting of two ramps connected by an optional middle platform.

    Args:
        length: Length of each ramp, from entry point to peak point [m].
        height: Height of the element at the platform [m].
        num_segments: Number of segments used to parametrize the surface.
        platform_length (optional): Platform length inserted between the two
            ramps [m]. Default is 0.
        gap (optional): If true, the platform is not filled (there is a gap
            between the two ramps). Default is False.
        exponent: Larger exponents make the gap profile approach a square;
            a value of 1 yields a sinusoidal shape.
        type (optional): Type of the ramp. Default is "cos". Can be:
            * ``"cos"``: smooth connection to floor and top
            * ``"half-cos"``: smooth connection to floor only
            * ``"rand"``: randomly picks one of the above options

    Returns:
        A polygon specifying the profile outline, parameterized as a NumPy
        array with shape (N, 2).
    """
    # Generate profile for ramp
    ramp_up = curved_ramp_profile(length=length, height=height, num_segments=num_segments, type=type)

    # Extend the ramp to a wave by appending an inverted ramp on the other side
    ramp_down = ramp_up.copy()
    ramp_down[:, 1] = np.flip(ramp_up[:, 1])
    ramp_down[:, 0] += length + platform_length  # insert platform

    if platform_length == 0.0:
        s = np.linspace(1.0, 0.0, 2 * num_segments)
        floor = np.stack([s * (2.0 * length), s * 0.0], axis=1)
        return np.vstack([ramp_up, ramp_down, floor[1:-1, :]])

    # Floor
    s = np.linspace(1.0, 0.0, 3 * num_segments)
    floor = np.stack([s * (2.0 * length + platform_length), s * 0.0], axis=1)

    # Add gap between the two ramps
    s = np.linspace(0.0, 1.0, 2 * num_segments)
    xi = s * (ramp_down[0, 0] - ramp_up[-1, 0]) + ramp_up[-1, 0]
    if sample(gap):
        rel_platform_h = sample((0.05, 0.2))
        zi = (
            -((0.5 * (np.cos(2.0 * np.pi * s) + 1.0)) ** sample(exponent)) * (1.0 - rel_platform_h) * height
            - rel_platform_h * height
        )
    else:
        zi = s * 0.0 - height
    gap = np.stack([xi, zi], axis=1)
    return np.vstack([ramp_up, gap[1:-1, :], ramp_down, floor[1:-1, :]])


def root_profile(length: float, height: float, num_segments: int, exponent: int | tuple[int, int]) -> np.ndarray:
    """Helper function to generate a root profile.

    Args:
        length: length of the root profile.
        height: the height of the root profile.
        num_segments: number of segments used to parametrize profile shape.
        exponent: the larger the exponent is, the more the root profile approaches a square.
            A value of 1 means a sinusoidal shape.

    Returns:
        a polygon specifying the profile outline, parameterized as a NumPy array.
    """
    s = np.linspace(0.0, 1.0, num_segments)
    xi = s * length
    zi = (((1.0 - np.sin(np.pi * s)) ** sample(exponent)) - 1.0) * height
    return np.stack([xi, zi], axis=1)


def ramp_profile(length: float, height: float, num_segments: int, elevation: float = 0.0) -> np.ndarray:
    """Helper function to generate a ramp profile.

    Args:
        length: length of the ramp profile.
        height: the height of the ramp profile.
        num_segments: number of segments used to parametrize profile shape. Must > 1.
        elevation: additional support height below the ramp [m]. If > 0,
            the ramp is shifted upward and a box-like support is added below it.

    Returns:
        a polygon specifying the profile outline, parameterized as a NumPy array.
    """
    # ramp
    s = np.linspace(0.0, 1.0, num_segments)
    xi = s * length
    zi = -height + s * height - elevation
    ramp = np.stack([xi, zi], axis=1)
    # floor
    s = np.linspace(1.0, 0.0, num_segments)
    xi = s * length
    zi = s * 0.0
    floor = np.stack([xi, zi], axis=1)
    if elevation > 0.0:
        # Add the top-right corner of the support box explicitly so the profile
        # contains a vertical face at x=length.
        support_corner = np.array([[length, 0.0]])
        return np.vstack([ramp, support_corner, floor[1:, :]])
    # combine
    return np.vstack([ramp, floor[1:, :]])


def box_profile(length: float, height: float) -> np.ndarray:
    """Helper function to generate a box profile.

    Args:
        length: length of the box profile.
        height: the height of the box profile.

    Returns:
        a polygon specifying the profile outline, parameterized as a NumPy array.
    """
    xi = [0.0, 0.0, length, length]
    zi = [0.0, -height, -height, 0.0]
    return np.stack([xi, zi], axis=1)


def stair_profile(length: float, height: float, step_width: float) -> np.ndarray:
    """Helper function to generate a stair profile.

    Args:
        length: length of the stair profile [m].
        height: the height of the stair profile [m].
        step_width: step width [m].

    Returns:
        a polygon specifying the profile outline, parameterized as a NumPy array.
    """
    # init
    xi = [0.0]
    zi = [0.0]
    x = 0.0
    z = 0.0

    # compute number of steps
    num_steps = math.floor(length / step_width)
    if num_steps == 0:
        num_steps = 1
    # recompute stair parameters to fit number of steps
    step_width = length / num_steps
    step_height = height / num_steps

    for _ in range(num_steps):
        # add current step
        xi = np.concatenate([xi, [x, x + step_width]], axis=0)
        zi = np.concatenate([zi, [z - step_height, z - step_height]], axis=0)
        # prepare next step
        x += step_width
        z -= step_height

    # close the stair profile
    xi = np.concatenate([xi, [xi[-1]]], axis=0)
    zi = np.concatenate([zi, [0.0]], axis=0)
    return np.stack([xi, zi], axis=1)
