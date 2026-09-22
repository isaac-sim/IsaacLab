# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend interface and data formats for the scene data provider.

These types live in :mod:`isaaclab.scene_data` rather than
:mod:`isaaclab.scene` so that physics backends (``isaaclab_physx``,
``isaaclab_newton``) can subclass :class:`SceneDataBackend` without pulling
:mod:`isaaclab.scene` into the ``AppLauncher`` pre-launch import chain.
``AppLauncher._create_app`` pops ``*lab*`` modules from ``sys.modules``
during Kit init and any submodule imported during that window ends up
orphaned from its parent's ``__dict__`` after restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

# Under Sphinx ``autodoc_mock_imports``, ``wp.struct`` is a ``_MockObject``
# that replaces the decorated class with another mock, hiding its docstring
# and fields from autodoc. Fall back to an identity decorator when warp is
# mocked so the documentation builds from the source classes directly.
if getattr(wp, "__sphinx_mock__", False):

    def wp_struct(cls):
        return cls
else:
    wp_struct = wp.struct


class SceneDataFormat:
    """Warp struct variants describing the transform layouts that a
    :class:`SceneDataBackend` may publish to consumers.
    """

    @wp_struct
    class Vec3_Quat:
        """Separate position and quaternion arrays."""

        positions: wp.array(dtype=wp.vec3f) = None
        """Per-transform positions [m]."""

        orientations: wp.array(dtype=wp.quatf) = None
        """Per-transform orientations as quaternions."""

    @wp_struct
    class Vec3_Matrix33:
        """Separate position and rotation-matrix arrays."""

        positions: wp.array(dtype=wp.vec3f) = None
        """Per-transform positions [m]."""

        orientations: wp.array(dtype=wp.mat33f) = None
        """Per-transform orientations as 3x3 rotation matrices."""

    @wp_struct
    class Transform:
        """Packed warp transforms (position + quaternion)."""

        transforms: wp.array(dtype=wp.transformf) = None
        """Per-transform packed position + orientation transforms [m, -]."""

    @wp_struct
    class Matrix44:
        """Packed 4x4 homogeneous transform matrices."""

        matrices: wp.array(dtype=wp.mat44f) = None
        """Per-transform 4x4 homogeneous transform matrices [m]."""

    @wp_struct
    class TransposedMatrix44d:
        """Double-precision row-vector transforms, as consumed by USD renderers."""

        matrices: wp.array(dtype=wp.mat44d) = None
        """World transforms [m], shape [transform_count]."""

    @dataclass(slots=True)
    class FabricMatrix44:
        """Indexed Fabric world matrices and their native-to-output mapping."""

        matrices: Any = None
        """Transposed double-precision ``omni:fabric:worldMatrix`` values [m]."""

        mapping: wp.array | None = None
        """Native-to-output indices; solver-only bodies without rigid destinations map to -1."""

    @wp_struct
    class Points:
        """Flat world-space nodal or particle positions."""

        points: wp.array(dtype=wp.vec3f) = None
        """World-space positions [m], shape [point_count]."""


@dataclass(slots=True)
class SceneDataPublication:
    """A producer-owned native-format pointer and its dirty latch.

    Producers mark the publication dirty after state writes or pointer swaps. SDP consumes the
    latch and owns format conversions; consumers must not modify the published arrays.
    """

    data: Any
    dirty: bool = True


class SceneDataBackend:
    @property
    def fabric_publication(self) -> SceneDataPublication | None:
        """Return an engine-owned Fabric interface and dirty latch, or None for SDP conversion."""
        return None

    @property
    def transform_publication(self) -> SceneDataPublication:
        """Return current native transforms and their dirty latch."""
        raise NotImplementedError

    @property
    def transforms(
        self,
    ) -> (
        SceneDataFormat.Vec3_Quat | SceneDataFormat.Transform | SceneDataFormat.Matrix44 | SceneDataFormat.Vec3_Matrix33
    ):
        """Return the native transform publication without copying its arrays."""
        return self.transform_publication.data

    @property
    def transform_count(self) -> int:
        """Return the number of transforms in the sim backend."""
        raise NotImplementedError

    @property
    def transform_paths(self) -> list[str]:
        """Return the paths for each transform."""
        raise NotImplementedError

    @property
    def points(self) -> SceneDataFormat.Points:
        """Return deformable or particle geometry as flat world-space positions."""
        return SceneDataFormat.Points()

    @property
    def point_count(self) -> int:
        """Return the number of points in :attr:`points`."""
        return 0

    @property
    def geometry_paths(self) -> list[str]:
        """Return one USD prim path per geometry entity (deformable body instance)."""
        return []

    @property
    def geometry_counts(self) -> list[int]:
        """Return the unpadded point count for each geometry entity."""
        return []
