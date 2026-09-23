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

import inspect
import textwrap
from typing import Any

import warp as wp
import warp._src.codegen as warp_codegen


def _patch_fabric_structs() -> None:
    """Backport factory-annotated Fabric struct kernel arguments for older Warp (NVIDIA/warp#1818).

    Patch descriptors and kernel code generation in memory, never installed files.
    Fabric storage cannot move across devices; NumPy struct serialization is not backported.
    """
    if "fabricarray" in warp_codegen._make_struct_field_constructor.__code__.co_names:
        return
    patches = (
        (
            warp_codegen.Struct,
            "__init__",
            "        elif isinstance(var.type, Struct):",
            "        elif isinstance(var.type, (warp.fabricarray, warp.indexedfabricarray)):\n"
            "            fields.append((label, type(var.type.__ctype__())))\n",
        ),
        (
            warp_codegen,
            "_make_struct_field_constructor",
            "    elif _is_texture_type(var_type):",
            "    elif isinstance(var_type, (warp.fabricarray, warp.indexedfabricarray)):\n"
            "        return lambda ctype: None\n",
        ),
        (
            warp_codegen,
            "_make_struct_field_setter",
            "    elif _is_texture_type(var_type):",
            "    elif isinstance(var_type, (warp.fabricarray, warp.indexedfabricarray)):\n"
            "        def set_fabric_value(inst, value):\n"
            "            if value is not None and (not isinstance(value, type(var_type))\n"
            "                    or not types_equal(value.dtype, var_type.dtype) or value.ndim != var_type.ndim):\n"
            "                raise TypeError(f'Invalid Fabric array for struct field {field!r}.')\n"
            "            setattr(inst._ctype, field, var_type.__ctype__() if value is None else value.__ctype__())\n"
            "            cls.__setattr__(inst, field, value)\n"
            "        return set_fabric_value\n",
        ),
        (
            warp_codegen,
            "codegen_struct",
            "atomic_add_body.append(",
            "if not isinstance(var.type, (warp.fabricarray, warp.indexedfabricarray)):\n            ",
        ),
        (
            warp_codegen.StructInstance,
            "to",
            "        elif isinstance(var.type, Struct):",
            "        elif isinstance(var.type, (warp.fabricarray, warp.indexedfabricarray)):\n"
            "            if value is not None and value.device is not None and value.device != warp.get_device(device):\n"
            "                raise ValueError(f'Cannot move Fabric struct field {name!r} across devices.')\n"
            "            setattr(dst, name, value)\n",
        ),
    )
    replacements = []
    for owner, name, anchor, insertion in patches:
        source = textwrap.dedent(inspect.getsource(getattr(owner, name)))
        if source.count(anchor) != 1:
            raise RuntimeError(f"Unsupported Warp {wp.__version__}: cannot backport {name} Fabric fields.")
        namespace = {}
        exec(compile(source.replace(anchor, insertion + anchor), warp_codegen.__file__, "exec"), vars(warp_codegen), namespace)
        replacements.append((owner, name, namespace[name]))
    for owner, name, replacement in replacements:
        setattr(owner, name, replacement)

# Under Sphinx ``autodoc_mock_imports``, ``wp.struct`` is a ``_MockObject``
# that replaces the decorated class with another mock, hiding its docstring
# and fields from autodoc. Fall back to an identity decorator when warp is
# mocked so the documentation builds from the source classes directly.
if getattr(wp, "__sphinx_mock__", False):

    def wp_struct(cls):
        return cls
else:
    _patch_fabric_structs()
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
        """Native Fabric world matrices, with SDP-owned bindings for foreign physics."""

        matrices: wp.fabricarray(dtype=wp.mat44d) = None
        """Transposed double-precision ``omni:fabric:worldMatrix`` values [m]."""

        local_matrices: wp.fabricarray(dtype=wp.mat44d) = None
        """Writable local matrices [m] for conversion; native Fabric needs no conversion destinations."""

        indices: wp.fabricarray(dtype=wp.int32) = None
        """Native source index per Fabric destination; solver-only bodies have no destination."""

        scales: wp.array(dtype=wp.vec3f) = None
        """Authored world scales captured once, indexed by native source, shape [transform_count]."""

    @wp_struct
    class Points:
        """Flat world-space nodal or particle positions."""

        points: wp.array(dtype=wp.vec3f) = None
        """World-space positions [m], shape [point_count]."""


class SceneDataBackend:
    transforms_dirty: bool
    """Set by producers after native writes or buffer swaps; cleared by SDP after reading ``transforms``."""

    fabric_dirty: bool
    """Independent dirty flag for native Fabric, when available; cleared by SDP after refreshing it."""

    @property
    def fabric(self) -> Any:
        """Return an engine-owned Fabric interface, or None for SDP conversion."""
        return None

    @property
    def transforms(
        self,
    ) -> (
        SceneDataFormat.Vec3_Quat | SceneDataFormat.Transform | SceneDataFormat.Matrix44 | SceneDataFormat.Vec3_Matrix33
    ):
        """Return native transforms without copying; pointer changes must set ``transforms_dirty``."""
        raise NotImplementedError

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
