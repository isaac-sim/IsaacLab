# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed transform data representations for the Scene Data Provider.

This module defines the concrete types used to pass transform data between
simulators, renderers, and visualizers without untyped dictionaries.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import warp as wp


class QuaternionConvention(enum.Enum):
    """Quaternion component ordering convention.

    Different subsystems in Isaac Lab use different quaternion orderings.
    PhysX and Isaac Lab's public API use XYZW; Warp's ``wp.quatf`` stores
    components as XYZW internally but constructs as ``wp.quatf(x, y, z, w)``.
    """

    XYZW = "xyzw"
    """Components ordered ``(x, y, z, w)``. Default for Isaac Lab and PhysX."""

    WXYZ = "wxyz"
    """Components ordered ``(w, x, y, z)``. Used by some external tools."""


class MatrixLayout(enum.Enum):
    """Matrix memory layout convention."""

    ROW_MAJOR = "row_major"
    """Rows are contiguous in memory. Default for most CPU/GPU math libraries."""

    COLUMN_MAJOR = "column_major"
    """Columns are contiguous in memory. Required by OVRTX."""


class TransformFormat(enum.Enum):
    """Supported rigid-body transform representations.

    Each variant corresponds to a concrete :class:`TransformData` subclass
    that holds typed Warp arrays on GPU.
    """

    VEC3_QUAT = "vec3_quat"
    """Separate position (vec3) and quaternion (quatf) arrays. Native to PhysX."""

    VEC3_MAT33 = "vec3_mat33"
    """Separate position (vec3) and 3x3 rotation matrix (mat33f) arrays."""

    TRANSFORM = "transform"
    """Packed 7-float transforms (transformf). Native to Newton ``state.body_q``."""

    MAT44 = "mat44"
    """4x4 homogeneous transformation matrices. Native to OVRTX."""


@dataclass
class TransformData:
    """Base class for typed transform buffer containers.

    All concrete subclasses hold Warp arrays on a specific device and carry
    metadata describing the data layout (quaternion convention, matrix layout).
    """

    format: TransformFormat = field(init=False)
    """The transform representation this container uses."""

    count: int = 0
    """Number of transforms in the container."""

    device: str = "cuda:0"
    """Device where the Warp arrays reside."""


@dataclass
class Vec3QuatTransforms(TransformData):
    """Positions and quaternion orientations as separate arrays.

    Attributes:
        positions: Body positions [m], shape ``(count,)``, dtype ``wp.vec3``.
        orientations: Body orientations, shape ``(count,)``, dtype ``wp.quatf``.
        quat_convention: Component ordering of the quaternion data.
    """

    positions: wp.array | None = None
    orientations: wp.array | None = None
    quat_convention: QuaternionConvention = QuaternionConvention.XYZW

    def __post_init__(self):
        self.format = TransformFormat.VEC3_QUAT


@dataclass
class Vec3Mat33Transforms(TransformData):
    """Positions and 3x3 rotation matrices as separate arrays.

    Attributes:
        positions: Body positions [m], shape ``(count,)``, dtype ``wp.vec3``.
        rotations: Body rotation matrices, shape ``(count,)``, dtype ``wp.mat33f``.
        layout: Memory layout of the rotation matrices.
    """

    positions: wp.array | None = None
    rotations: wp.array | None = None
    layout: MatrixLayout = MatrixLayout.ROW_MAJOR

    def __post_init__(self):
        self.format = TransformFormat.VEC3_MAT33


@dataclass
class TransformArrayData(TransformData):
    """Packed transforms as a single array.

    Each element is a ``wp.transformf`` containing a 3D position and a
    quaternion orientation (7 floats total).

    Attributes:
        transforms: Packed transforms, shape ``(count,)``, dtype ``wp.transformf``.
    """

    transforms: wp.array | None = None

    def __post_init__(self):
        self.format = TransformFormat.TRANSFORM


@dataclass
class Mat44Transforms(TransformData):
    """4x4 homogeneous transformation matrices.

    Attributes:
        matrices: Transform matrices, shape ``(count,)``,
            dtype ``wp.mat44f`` or ``wp.mat44d``.
        layout: Memory layout of the matrices.
        double_precision: Whether matrices use 64-bit floats.
    """

    matrices: wp.array | None = None
    layout: MatrixLayout = MatrixLayout.COLUMN_MAJOR
    double_precision: bool = False

    def __post_init__(self):
        self.format = TransformFormat.MAT44
