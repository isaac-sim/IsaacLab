# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
Utilities to compute trail roll angles along a sweeping path.

Example usage::

    angles = roll_function(path, params)

where ``angles`` are the relative roll angles along the sweeping path.

The function signature for a roll function should be::

    def roll_function(path: np.ndarray, params: dict[str, float]) -> np.ndarray:
        Args:
            path: The sweeping path of the trail as an (N, M) array.
            params: Additional parameters used to compute the roll angles.

        Returns:
            A NumPy array of roll angles (radians) along the sweeping path.
"""

import random

import numpy as np


def lin_derivative_y(path: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return roll angles proportional to the y-derivative of the path.

    The angle at each knot is computed from the local difference in the y coordinate between consecutive path points and
    clipped to the configured maximum angle.
    """
    num_knots = path.shape[0]
    angles = np.zeros(num_knots)
    for id in range(1, num_knots, 1):
        angles[id - 1] -= (
            params["gain_der_y"] * (path[id, 1] - path[id - 1, 1]) / np.linalg.norm(path[id, :] - path[id - 1, :])
        )
    return np.clip(angles, a_min=-params["max_angle"], a_max=params["max_angle"])


def lin_y(path: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return roll angles proportional to the path y coordinate.

    Angles are scaled by ``params['gain_y']`` and clipped to ``[-params['max_angle'], params['max_angle']]``.
    """
    return np.clip(
        -params["gain_y"] * path[:, 1].squeeze(),
        a_min=-params["max_angle"],
        a_max=params["max_angle"],
    )


def sin_x(path: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return sinusoidal roll angles as a function of the x coordinate.

    The sinusoid amplitude and period are specified by ``params['Ax']`` and ``params['Tx']``. The phase is sampled
    randomly per invocation.
    """
    return params["Ax"] * np.sin(2.0 * np.pi * path[:, 0] / params["Tx"] + random.uniform(0.0, np.pi))


def sin_s(path: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return sinusoidal roll angles as a function of path length (s).

    The cumulative path length ``s`` is computed along the knot points and used to evaluate a sinusoid with amplitude
    ``params['As']`` and period ``params['Ts']``. The phase is randomized.
    """
    num_knots = path.shape[0]
    s = np.zeros(num_knots)
    for id in range(1, num_knots, 1):
        s[id] = s[id - 1] + np.linalg.norm(path[id, :] - path[id - 1, :])
    return params["As"] * np.sin(2.0 * np.pi * s / params["Ts"] + random.uniform(0.0, np.pi))


def const(path: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return a constant roll angle along the full path.

    The constant angle magnitude is specified by ``params['A']`` in radians, and the sign is sampled randomly per
    invocation.
    """
    sign = random.choice([-1.0, 1.0])
    return np.full(path.shape[0], sign * params["A"], dtype=float)
