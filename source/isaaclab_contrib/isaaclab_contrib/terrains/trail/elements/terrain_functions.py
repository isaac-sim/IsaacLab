# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Terrain deformation primitives used by trail generation.

Each function in this module returns a vertex displacement array ("delta")
with the same shape as the input ``vertices`` (``N x 3``). The caller applies
the deformation as:

    new_vertices = vertices + delta

Most parameters support either a fixed value or a ``(min, max)`` tuple.
Tuple-valued parameters are sampled via :func:`sample` when the function runs.
"""

from typing import Literal

import numpy as np

from ..utils.math import sample


def delta_z_slope_x(vertices: np.ndarray, S: float | tuple[float, float], difficulty: float) -> np.ndarray:
    """Create a linear height ramp along the x-axis.

    The z displacement is computed as ``dz = (x - min_x) * S``.

    Args:
        vertices: Terrain vertices with shape ``(N, 3)``.
        S: Slope in ``m/m`` (float or sampled range).
        difficulty: Difficulty in ``[0, 1]``. Present for API consistency.

    Returns:
        Vertex displacement with shape ``(N, 3)``.
    """
    min_x = np.min(vertices[:, 0])
    x = vertices[:, 0] - min_x
    zeros = np.zeros(len(x))
    dz = x * sample(S)
    return np.stack([zeros, zeros, dz], axis=1)


def delta_xyz_sin_x(
    vertices: np.ndarray,
    A: float | tuple[float, float],
    T: float | tuple[float, float],
    N: int | tuple[int, int],
    difficulty: float,
) -> np.ndarray:
    """Apply sinusoidal displacement in all three coordinates as a function of x.

    For each iteration and each axis ``i in {x, y, z}``, the function adds:
    ``delta_i += A * sin(2*pi*(x - min_x) / T)``.

    Args:
        vertices: Terrain vertices with shape ``(N, 3)``.
        A: Wave amplitude in meters (float or sampled range).
        T: Wave period in meters (float or sampled range).
        N: Number of additive wave passes.
        difficulty: Difficulty in ``[0, 1]``. Present for API consistency.

    Returns:
        Vertex displacement with shape ``(N, 3)``.
    """
    min_x = np.min(vertices[:, 0])
    x = 2.0 * np.pi * (vertices[:, 0] - min_x)
    xyz = np.zeros_like(vertices)
    num_iter = sample(N)
    for _ in range(num_iter):
        for dim in range(3):
            xyz[:, dim] += sample(A) * np.sin(x / sample(T))
    return xyz


def delta_i_sin_x(
    vertices: np.ndarray,
    A: float | tuple[float, float],
    T: float | tuple[float, float],
    dim: Literal[0, 1, 2],
    N: int | tuple[int, int],
    difficulty: float,
) -> np.ndarray:
    """Apply sinusoidal displacement in a single selected coordinate.

    For each iteration, the function adds:
    ``delta[dim] += A * sin(2*pi*(x - min_x) / T + phase)``,
    where ``phase`` is sampled in ``[0, pi]``.

    Args:
        vertices: Terrain vertices with shape ``(N, 3)``.
        A: Wave amplitude in meters (float or sampled range).
        T: Wave period in meters (float or sampled range).
        dim: Target coordinate index: ``0`` (x), ``1`` (y), or ``2`` (z).
        N: Number of additive wave passes.
        difficulty: Difficulty in ``[0, 1]``. Present for API consistency.

    Returns:
        Vertex displacement with shape ``(N, 3)``.
    """
    min_x = np.min(vertices[:, 0])
    x = 2.0 * np.pi * (vertices[:, 0] - min_x)
    xyz = np.zeros_like(vertices)
    num_iter = sample(N)
    for _ in range(num_iter):
        xyz[:, dim] += sample(A) * np.sin(x / sample(T) + sample((0.0, np.pi)))
    return xyz


def delta_z_sin_xy(
    vertices: np.ndarray,
    A: float | tuple[float, float],
    Tx: float | tuple[float, float],
    Ty: float | tuple[float, float],
    N: int | tuple[int, int],
    difficulty: float,
) -> np.ndarray:
    """Apply a 2D sinusoidal height pattern over x and y.

    The z displacement is formed from an x-varying amplitude and a y-wave:

        ``Az(x) = A * sin(2*pi*x / Tx)``
        ``dz(x, y) = Az(x) * sin(2*pi*y / Ty + phase)``

    Args:
        vertices: Terrain vertices with shape ``(N, 3)``.
        A: Maximum z-wave amplitude in meters (float or sampled range).
        Tx: Period in meters for amplitude modulation along x.
        Ty: Period in meters for sinusoid along y.
        N: Number of additive wave passes.
        difficulty: Difficulty in ``[0, 1]``. Present for API consistency.

    Returns:
        Vertex displacement with shape ``(N, 3)`` where only z is modified.
    """
    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    length_x = max_x - min_x
    x = 2.0 * np.pi * (vertices[:, 0] - min_x) / length_x
    xyz = np.zeros_like(vertices)
    Az = sample(A) * np.sin(x * round(length_x / sample(Tx)))
    num_iter = sample(N)
    for _ in range(num_iter):
        xyz[:, 2] += Az * np.sin(2.0 * np.pi * vertices[:, 1] / sample(Ty) + sample((0.0, np.pi)))
    return xyz


def delta_xyz_noise(
    vertices: np.ndarray,
    U: float | tuple[float, float],
    N: int | tuple[int, int],
    difficulty: float,
) -> np.ndarray:
    """Apply additive uniform noise in all three coordinates.

    Each iteration adds a random offset sampled from ``[-U, +U]`` per
    vertex coordinate.

    Args:
        vertices: Terrain vertices with shape ``(N, 3)``.
        U: Noise magnitude in meters (float or sampled range).
        N: Number of additive noise passes.
        difficulty: Difficulty in ``[0, 1]``. Present for API consistency.

    Returns:
        Vertex displacement with shape ``(N, 3)``.
    """
    xyz = np.zeros_like(vertices)
    num_iter = sample(N)
    for _ in range(num_iter):
        xyz += np.random.uniform(-1.0, 1.0, vertices.shape) * sample(U)
    return xyz


def delta_z_noise(
    vertices: np.ndarray,
    U: float | tuple[float, float],
    N: int | tuple[int, int],
    difficulty: float,
) -> np.ndarray:
    """Apply additive uniform noise only in z.

    Each iteration adds a random z offset sampled from ``[-U, +U]`` per
    vertex.

    Args:
        vertices: Terrain vertices with shape ``(N, 3)``.
        U: Noise magnitude in meters (float or sampled range).
        N: Number of additive noise passes.
        difficulty: Difficulty in ``[0, 1]``. Not used by this function.

    Returns:
        Vertex displacement with shape ``(N, 3)`` where only z is modified.
    """
    xyz = np.zeros_like(vertices)
    num_iter = sample(N)
    for _ in range(num_iter):
        xyz[:, 2] += np.random.uniform(-1.0, 1.0, vertices.shape[0]) * sample(U)
    return xyz
