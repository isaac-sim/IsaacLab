# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for different data types."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass

SDF_type_to_Gf = {
    "matrix3d": "Gf.Matrix3d",
    "matrix3f": "Gf.Matrix3f",
    "matrix4d": "Gf.Matrix4d",
    "matrix4f": "Gf.Matrix4f",
    "range1d": "Gf.Range1d",
    "range1f": "Gf.Range1f",
    "range2d": "Gf.Range2d",
    "range2f": "Gf.Range2f",
    "range3d": "Gf.Range3d",
    "range3f": "Gf.Range3f",
    "rect2i": "Gf.Rect2i",
    "vec2d": "Gf.Vec2d",
    "vec2f": "Gf.Vec2f",
    "vec2h": "Gf.Vec2h",
    "vec2i": "Gf.Vec2i",
    "vec3d": "Gf.Vec3d",
    "double3": "Gf.Vec3d",
    "vec3f": "Gf.Vec3f",
    "vec3h": "Gf.Vec3h",
    "vec3i": "Gf.Vec3i",
    "vec4d": "Gf.Vec4d",
    "vec4f": "Gf.Vec4f",
    "vec4h": "Gf.Vec4h",
    "vec4i": "Gf.Vec4i",
}


@dataclass
class ArticulationActions:
    """Data container to store articulation's joints actions.

    This class is used to store the actions of the joints of an articulation.
    It is used to store the joint positions, velocities, efforts, and indices.

    If the actions are not provided, the values are set to None.
    """

    joint_positions: torch.Tensor | None = None
    """The joint positions of the articulation. Defaults to None."""

    joint_velocities: torch.Tensor | None = None
    """The joint velocities of the articulation. Defaults to None."""

    joint_efforts: torch.Tensor | None = None
    """The joint efforts of the articulation. Defaults to None."""

    joint_indices: torch.Tensor | Sequence[int] | slice | None = None
    """The joint indices of the articulation. Defaults to None.

    If the joint indices are a slice, this indicates that the indices are continuous and correspond
    to all the joints of the articulation. We use a slice to make the indexing more efficient.
    """
