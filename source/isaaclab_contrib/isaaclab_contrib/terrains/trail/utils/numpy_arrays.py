# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""NumPy helper functions for trail generation.

These helpers operate on NumPy arrays describing 2D/N-D geometric profiles used by the trail generator.
"""

import numpy as np


def mirror_and_join(object: np.ndarray, dim: int = 0, dim_flip: int = -1, offset: float = 0.0) -> np.ndarray:
    """Mirror an array along one axis and join it with an offset.

    Args:
        object: Array of shape (N, m) to be mirrored and extended.
        dim: Dimension index to be extended (default 0). Use -1 for the last
            dimension.
        dim_flip: Dimension index to flip/mirror (default -1 for last dim).
        offset: Offset to add between the original array and its mirrored copy.

    Note:
        It is possible to set ``dim == dim_flip`` in principle, but this is
        not the intended use case. Only a single axis is mirrored, and the
        flip is performed on the component specified by ``dim_flip``.

    Returns:
        A new NumPy array containing the original array stacked with its
        mirrored counterpart.

    Raises:
        RuntimeError: if ``object`` does not have shape (N, m) or if ``dim`` is
            not a valid column index for the array.
    """
    if len(object.shape) != 2 or dim >= object.shape[1]:
        raise RuntimeError("Array has the wrong shape. Expect an array of shape (N, m) with dim < m.")
    mirror = object.copy()
    mirror[:, dim_flip] = np.flip(mirror[:, dim_flip])
    mirror[:, dim] += abs(object[0, dim] - object[-1, dim]) + offset
    return np.vstack([object, mirror])


def get_bounding_box(object: np.ndarray) -> np.ndarray:
    """Compute an axis-aligned bounding box for an array of points.

    Args:
        object: Array of shape (N, m) representing N points in m dimensions.

    Returns:
        Array of shape (2, m) where the first row contains minima and the
        second row contains maxima for each column.

    Raises:
        RuntimeError: if ``object`` is not a 2D array of shape (N, m).
    """
    if len(object.shape) != 2:
        raise RuntimeError("Array has the wrong shape. Expect an array of shape (N, m).")
    object_linearized = np.zeros((2, object.shape[1]))
    object_linearized[0, :] = np.min(object, axis=0)
    object_linearized[1, :] = np.max(object, axis=0)
    return object_linearized


def decay_at_boundaries(object: np.ndarray, vec: np.ndarray | None, dim: int = 0, threshold: float = 1.0):
    """Decay values toward zero at the boundaries along a specified dimension.

    The function scales the values in ``object[:, dim]`` near the lower and
    upper boundaries (as defined by ``threshold`` on ``vec``) so they smoothly
    approach zero. The operation is performed in-place on ``object``.

    Args:
        object: 1D array of length N or 2D array with shape (N, m). If a 1D
            array is provided it will be treated as shape (N, 1).
        vec: Reference vector used to determine boundary regions. If ``None``,
            ``object[:, 0]`` is used.
        dim: Column index to decay (default 0).
        threshold: Boundary width in the same units as ``vec``. Values with
            ``vec < threshold`` or ``vec > max(vec) - threshold`` will be
            decayed. Default is 1.0.

    Returns:
        The modified ``object`` (same array, modified in-place).

    Raises:
        RuntimeError: if ``object`` is neither 1D nor 2D.
    """
    if len(object.shape) == 1:
        object = np.expand_dims(object, axis=1)
        dim = 0
    if len(object.shape) != 2:
        raise RuntimeError("Array has the wrong shape. Expect an array of shape (N, m).")

    if vec is None:
        vec = object[:, 0]

    # Lower boundary: scale values where vec < threshold (range [0 .. threshold])
    ids = np.nonzero(vec < threshold)
    object[ids, dim] *= vec[ids] / threshold

    # Upper boundary: scale values where vec > max(vec) - threshold
    max_vec = np.max(vec)
    ids = np.nonzero(vec > (max_vec - threshold))
    decay = (vec[ids] - (max_vec - threshold)) / threshold
    object[ids, dim] *= 1.0 - decay
