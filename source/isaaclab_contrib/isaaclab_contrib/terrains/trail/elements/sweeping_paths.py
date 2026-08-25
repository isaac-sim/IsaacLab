# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""This file contains sweeping paths for trail curves and trail objects.

When creating new curves, consider the following:
    * The trail profile polygon is constructed in yz plane, and then swept along an arbitrary axis.
    * For sweeping along y axis, the following coordinate system is utilized
                 x
                 ^
                 |
                 |
        y <------+
    * For sweeping along z axis, the following coordinate system is utilized
                 z
                 ^
                 |
                 |
        y <------+
"""

import random

import numpy as np

from ..utils.math import sample


def linear_path_y(width: float, num_segments) -> np.ndarray:
    """Sweeping happens from left to right, i.e. along y axis.

    Args:
        width: the width in y direction [m].
        num_segments: number of segments used to approximate the path.

    Returns:
        sweeping path parametrized as numpay array.
    """
    si = np.linspace(0.0, 1.0, num_segments)
    xi = si * 0.0
    yi = si * width - 0.5 * width
    zi = si * 0.0
    return np.stack([xi, yi, zi], axis=1)


def sinusoidal_z_curve_x(
    length: float,
    amplitude: float | tuple[float, float],
    num_curves: int | tuple[int, int],
    num_segments: int,
) -> np.ndarray:
    """Sinusoidal curve in xz plane.

    Args:
        length: Total length of curve segment in heading direction of trail [m].
        amplitude: Amplitude of the sinusoidal wave [m]. Positive means the curve is facing up wrt. the trail direction.
        num_curves: Number of curves joint together. 1 means half of a sinusoidal curve on a (0,pi) interval.
        num_segments: number of segments used to approximate half of a sinusoidal curve.

    Returns:
        Sweeping path parametrized as numpay array with dimensions Nx3.
    """
    s = np.linspace(0.0, 1.0, num_segments * num_curves)
    xi = s * length
    yi = s * 0.0
    zi = 0.5 * (1.0 - np.cos(sample(num_curves) * np.pi * s)) * sample(amplitude)
    return np.stack([xi, yi, zi], axis=1)


def sinusoidal_y_curve_x(
    length: float,
    amplitude: float | tuple[float, float],
    num_curves: int | tuple[int, int],
    num_segments: int,
) -> np.ndarray:
    """Sinusoidal curve in xy plane.

    Args:
        length: Total length of curve segment in heading direction of trail [m].
        amplitude: Amplitude of the sinusoidal wave [m].
            Positive means the curve is facing left wrt. the trail direction.
        num_curves: Number of curves joint together. 1 means half of a sinusoidal curve on a (0,pi) interval.
        num_segments: number of segments used to approximate half of a sinusoidal curve.

    Returns:
        Sweeping path parametrized as numpay array with dimensions Nx3.
    """
    curve = sinusoidal_z_curve_x(
        length=length,
        amplitude=amplitude,
        num_curves=2 * sample(num_curves),
        num_segments=num_segments,
    )
    return np.stack([curve[:, 0], curve[:, 2], curve[:, 1]], axis=1)  # switch z with y axis


def circular_xy_curve_x(
    length: float,
    radius: float | tuple[float, float],
    rel_angle: float | tuple[float, float],
    slope: float | tuple[float, float],
    max_wing_length: float,
    num_segments: int,
) -> np.ndarray:
    """This function creates three circular curves in xy plane, creating a wing like structure.

        * The first curve is a left turn if rel_angle > 0 (otherwise it is a right turn).
        * A segment of zero curvature follows, called the wing. The length of this segment is computed from the length.
        * The second curve is twice as large as the first one.
        * Another wing segment follows, that connects the second with the third curve.
        * A final curve aligns the trail back to the x axis.
        * If the wings are larger than max_wing_length, the entire curve is adapted.
            The total length will then be smaller then specified.

                                     ^ x
                                     |
                                +    |
                                |    |
                         +-----+     |
                       +             |
                      |              |
                       +             |
                         +-----+     |
                                +    |
                                |    |
             y <---------------------+

    Args:
        length: Total length of curve segment in heading direction of trail [m].
        radius: The radius of each curve [m].
        rel_angle: Relative angle wrt to 90 degrees in (0,1). 1 means 90 degrees and is the maximum, 0 means 0 degrees.
            If positive, the segments is curved towards the left (+y), otherwise towards the right (-y).
        slope: The range of slopes of the curve [m].
        max_wing_length: If the wing length is larger then this value,
            the curve will be adapted to satisfy this constraint.
        num_segments: number of segments used to approximate each 90 degrees segment.

    Returns:
        Sweeping path parametrized as numpay array with dimensions Nx3.
    """
    # sample
    radius = sample(radius)
    rel_angle = sample(rel_angle)
    slope = sample(slope)
    max_wing_length = sample(max_wing_length)

    # extract sign from angle.
    sing_angle = np.sign(rel_angle)
    rel_angle = abs(rel_angle)

    # check validity of arguments
    if rel_angle > 1.0:
        raise RuntimeError(
            "rel_angle has to be in the interval [0,1]. A value larger then this would yield backwards oriented trails."
        )
    if 2.0 * radius > max_wing_length:
        raise RuntimeError("Cannot satisfy max_wing_length constraints. Choose a smaller radius.")

    # compute the curve angle.
    angle = 0.5 * rel_angle * np.pi
    s = np.linspace(0.0, 1.0, num_segments)

    # first circle
    center = [0.0, radius * sing_angle]  # center of the circle
    phase = 0.0
    x1 = center[0] + np.sin(angle * s + phase) * radius
    y1 = center[1] - np.cos(angle * s + phase) * radius * sing_angle
    z1 = s * 0.0

    # compute position of second circle
    center[0] = x1[-1] + np.sin(angle) * radius
    center[1] = y1[-1] - np.cos(angle) * radius * sing_angle

    # add constant segment
    wing_length = 0.5 * length - 2.0 * x1[-1]
    if wing_length < 0.0:
        raise RuntimeError("The wing length is smaller then zero. Reduce the radius or increase the length.")
    center[0] += wing_length
    center[1] += np.tan(angle) * wing_length * sing_angle

    # If constraints are violated, move the center of second circle
    overshoot_wing = abs(center[1]) - (max_wing_length - radius)
    if overshoot_wing > 0.0:
        center[1] = np.clip(center[1], -max_wing_length + radius, max_wing_length - radius)
        if rel_angle < 1.0:
            center[0] -= overshoot_wing / np.tan(angle)
        else:
            center[0] = 2.0 * radius

    # second circle
    phase += 0.5 * np.pi - angle
    x2 = center[0] - np.cos(angle * s + phase) * radius
    y2 = center[1] + np.sin(angle * s + phase) * radius * sing_angle
    z2 = s * 0.0

    # combine the two curves
    xi = np.concatenate([x1, x2], axis=0)
    yi = np.concatenate([y1, y2], axis=0)
    zi = np.concatenate([z1, z2], axis=0)

    # mirror the curve along y axis (make sure the curve stops where it starts in y direction)
    xi = np.concatenate([xi, xi + xi[-1]], axis=0)
    yi = np.concatenate([yi, np.flip(yi)], axis=0)
    zi = np.concatenate([zi, zi], axis=0)

    # add slope
    zi += (xi - xi[0]) * slope

    return np.stack([xi, yi, zi], axis=1)


def zig_zag_yz_path_x(
    length: float,
    amplitude_y: float | tuple[float, float],
    amplitude_z: float | tuple[float, float],
    rel_knot_point: float | tuple[float, float],
) -> np.ndarray:
    """Sweeping follows the direction of the trail, with two lateral/vertical peaks.

    Args:
        length: the length of path measured in heading direction of the trail [m].
        amplitude_y: max overshoot in lateral direction [m].
        amplitude_z: max height above nominal ground at turning locations [m].
        rel_knot_point: location of the first knot, in (0, 0.5)

    Returns:
        Sweeping path parametrized as numpay array.
    """
    r = sample(rel_knot_point)
    if r > 0.5:
        raise RuntimeError("rel_knot_point needs to be a number between 0 and 0.5.")
    return np.vstack(
        [
            [length * 0.0, 0.0, 0.0],
            [length * 0.05, 0.0, 0.0],
            [length * r, sample(amplitude_y), sample(amplitude_z)],
            [length * (1.0 - r), sample(amplitude_y), sample(amplitude_z)],
            [length * 0.95, 0.0, 0.0],
            [length * 1.0, 0.0, 0.0],
        ]
    )


def sinusoidal_xz_curve_y(
    width: float,
    amplitude_x: float | tuple[float, float],
    amplitude_z: float | tuple[float, float],
    T_xz: float | tuple[float, float],
    num_segments: int,
) -> np.ndarray:
    """Sweeping path is defined by sinusoidal curves in x and z.

    The y direction is swept linearly.
        Args:
            width: max width in y direction [m].
            amplitude_x: the amplitude of the x-sine wave is sampled from this range [m].
            amplitude_z: the amplitude of the z-sine wave is sampled from this range [m].
            T_xz: wave length of the sinusoidal wave is sampled from this range [m].

        Returns:
            Sweeping path parametrized as numpay array.
    """

    def sample_sine_wave(
        amplitude: float | tuple[float, float],
        T_xz: float | tuple[float, float],
        s: np.ndarray,
    ) -> np.ndarray:
        A = sample(amplitude)  # sampled amplitude
        N = 1.0 / sample(T_xz)  # sampled number of waves
        phase = random.uniform(0.0, np.pi)  # sampled phase
        return np.sin(N * 2.0 * np.pi * s + phase) * A

    s = np.linspace(0.0, 1.0, num_segments)
    xi = sample_sine_wave(amplitude=amplitude_x, T_xz=T_xz, s=s)
    yi = (s - 0.5) * width
    zi = sample_sine_wave(amplitude=amplitude_z, T_xz=T_xz, s=s)
    return np.stack([xi, yi, zi], axis=1)


def sinusoidal_exp_z_curve_y(
    width: float,
    stone_length: float | tuple[float, float],
    stone_height: float | tuple[float, float],
    distance_between_stones: float | tuple[float, float],
    exponent: int | tuple[int, int],
    num_segments: int,
) -> np.ndarray:
    """Sweeping path is a repetition of sinusoidal curves in z.

    The y direction is swept linearly, x is hold constant.
        This sweeping function may be useful to generate gravel roads.

    Args:
        width: The width in y direction [m].
        stone_length: The length of the stone profile [m].
        stone_height: The height of the stone profile [m].
        distance_between_stones: Distance between two adjacent stones [m].
        exponent: the larger the exponent is, the more the stone profile approaches a square.
            A value of 1 means a sinusoidal shape.
        num_segments: number of segments used to approximate one wave (stone).

    Returns:
        Sweeping path parametrized as numpay array.
    """

    def sample_segment(
        length: float | tuple[float, float],
        height: float | tuple[float, float],
        exponent: tuple[int, int],
        s: np.ndarray,
    ) -> np.ndarray:
        A = sample(height)  # sampled amplitude
        L = sample(length)  # sampled length
        exp = sample(exponent)  # sampled exponent
        zi = -(((1.0 - np.sin(np.pi * s)) ** exp) - 1.0) * sample(height) - A
        xi = s * 0.0
        yi = s * L
        return np.stack([xi, yi, zi], axis=1)

    # create first segment
    s = np.linspace(0.0, 1.0, num_segments)
    path = sample_segment(length=stone_length, height=stone_height, exponent=exponent, s=s)

    # add segments until the path is full
    total_width = 0.0
    max_stone_length = stone_length[1] if isinstance(stone_length, tuple) else stone_length
    while total_width < width - max_stone_length * 0.5:
        segment = sample_segment(length=stone_length, height=stone_height, exponent=exponent, s=s)
        segment[:, 1] += path[-1, 1] + sample(distance_between_stones)
        path = np.vstack([path, segment])
        total_width = path[-1, 1]
    # center the path in the middle of the trail
    path[:, 1] -= 0.5 * total_width
    return path


def loop_curve_x(
    displacement_z: float,
    radius: float,
    angle: float,
    num_segments: int,
) -> np.ndarray:
    """Sweeping path creates a curve in the xy plane.

    The curve is swept in negative x-axis.
        Args:
            displacement_z: Height difference at the exit point [m].
            radius: The radius of the curve [m].
            angle: The circle is swept up to this angle [rad].
            num_segments: number of segments used to approximate one wave stone.

        Returns:
            Sweeping path parametrized as numpay array.
    """
    s = np.linspace(0.0, 1.0, num_segments)
    center = [0.0, radius]
    xi = center[0] - np.sin(angle * s) * radius
    yi = center[1] - np.cos(angle * s) * radius
    zi = s * displacement_z - displacement_z * 0.5
    return np.stack([xi, yi, zi], axis=1)
