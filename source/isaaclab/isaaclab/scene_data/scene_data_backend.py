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

    @wp_struct
    class FabricMatrix44:
        """Double-precision row-vector matrices in native Fabric storage."""

        matrices: wp.fabricarray(dtype=wp.mat44d) = None
        """Transforms [m], shape [transform_count]."""

    @wp_struct
    class Points:
        """Flat world-space nodal or particle positions."""

        points: wp.array(dtype=wp.vec3f) = None
        """World-space positions [m], shape [point_count]."""

    @wp_struct
    class WeightedPoints:
        """Visual vertices interpolated from four native simulation nodes."""

        points: wp.array(dtype=wp.vec3f)
        """Native world-space simulation nodes [m]."""
        indices: wp.array2d(dtype=wp.int32)
        """Four native node indices per visual vertex."""
        weights: wp.array2d(dtype=wp.float32)
        """Four barycentric weights per visual vertex."""

    @wp_struct
    class CapsuleEndpoints:
        """Polyline vertices derived from native capsule poses."""

        transforms: wp.array(dtype=wp.transformf)
        """World-space body poses [m, quaternion]."""
        shape_body: wp.array(dtype=wp.int32)
        shape_transform: wp.array(dtype=wp.transformf)
        """Capsule poses relative to their bodies [m, quaternion]."""
        shape_scale: wp.array(dtype=wp.vec3f)
        """Capsule scales [m]; the second component is the half-length."""
        endpoints: wp.array(dtype=wp.vec4i)
        """Two (capsule index, endpoint sign) pairs per output vertex; their positions are averaged."""

    @wp_struct
    class FabricPoints:
        """Native Fabric point arrays, already owned and updated by physics."""

        points: wp.fabricarrayarray(dtype=wp.vec3f)
        """Per-prim native point storage [m]."""


class SceneDataBackend:
    geometry_version: int = 0
    """Monotonic geometry publication version, including native buffer swaps and same-step writes."""

    @property
    def native_geometry_formats(self) -> tuple[Any, ...]:
        """Geometry formats the producer can publish without conversion."""
        return (SceneDataFormat.Points,)

    def get_geometry_batches(
        self, output_format: Any = SceneDataFormat.Points
    ) -> list[tuple[Any, dict[str, tuple[int, int]]]] | SceneDataFormat.FabricPoints:
        """Publish native arrays and exact visual-prim ranges established during construction.

        Each batch pairs a native format with ``path: (offset, count)`` ranges. Offsets index
        native points for ``Points`` or output vertices for an interpolated format. A requested
        native ``FabricPoints`` publication may be returned directly instead of host ranges.
        """
        return []

    transforms_version: int
    """Monotonic producer version, incremented after native writes or buffer swaps; never reset by readers."""

    @property
    def native_transform_formats(self) -> tuple[Any, ...]:
        """Formats available without conversion, used when binding consumer destinations."""
        return (self.transforms._cls,)

    def get_transforms(self, output_format: Any) -> Any:
        """Publish the requested native format when available, otherwise the primary format."""
        return self.transforms

    @property
    def transforms(
        self,
    ) -> (
        SceneDataFormat.Vec3_Quat | SceneDataFormat.Transform | SceneDataFormat.Matrix44 | SceneDataFormat.Vec3_Matrix33
    ):
        """Return native transforms without copying; pointer changes must increment ``transforms_version``."""
        raise NotImplementedError

    @property
    def transform_count(self) -> int:
        """Return the number of transforms in the sim backend."""
        raise NotImplementedError

    @property
    def transform_paths(self) -> list[str]:
        """Return the paths for each transform."""
        raise NotImplementedError
