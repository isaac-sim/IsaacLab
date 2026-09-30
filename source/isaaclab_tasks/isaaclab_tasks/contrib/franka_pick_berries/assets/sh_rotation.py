# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline SH3 transport in the Graphdeco real-SH basis used by the MDL shader."""

import numpy as np


def sh_basis(directions: np.ndarray) -> np.ndarray:
    """Evaluate all 16 real SH basis functions at unit directions, ordered by l,m."""
    x, y, z = np.moveaxis(directions, -1, 0)
    return np.stack(
        [
            x * 0 + 0.28209479177387814,
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * z * z - x * x - y * y),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
            -0.5900435899266435 * y * (3 * x * x - y * y),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
            0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
            -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
            1.445305721320277 * z * (x * x - y * y),
            -0.5900435899266435 * x * (x * x - 3 * y * y),
        ],
        axis=-1,
    )


def rotate_sh(coefficients: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply active material rotations: new radiance(d) = old radiance(R.T @ d).

    Rotation is a single 3x3 matrix or one per Gaussian. A 4x8 spherical
    quadrature integrates degree-six products exactly; no fitting or SH loss.
    """
    z, weights = np.polynomial.legendre.leggauss(4)
    azimuth = np.arange(8) * np.pi / 4
    directions = np.stack(
        [
            np.repeat(np.sqrt(1 - z * z), 8) * np.tile(np.cos(azimuth), 4),
            np.repeat(np.sqrt(1 - z * z), 8) * np.tile(np.sin(azimuth), 4),
            np.repeat(z, 8),
        ],
        axis=-1,
    )
    projection = sh_basis(directions).T * np.repeat(weights * np.pi / 4, 8)
    result = np.empty_like(coefficients)
    for start in range(0, len(coefficients), 2048):
        stop = start + 2048
        r = rotation if rotation.ndim == 2 else rotation[start:stop]
        local = directions @ r if r.ndim == 2 else np.einsum("dj,njk->ndk", directions, r)
        transform = projection @ sh_basis(local)
        result[start:stop] = transform @ coefficients[start:stop]
    return result


def proper_rotation(deformation: np.ndarray) -> np.ndarray:
    """Extract the material rotation, not the Gaussian covariance eigenvectors."""
    u, _, vt = np.linalg.svd(deformation)
    u[:, :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)[:, None]
    return u @ vt
