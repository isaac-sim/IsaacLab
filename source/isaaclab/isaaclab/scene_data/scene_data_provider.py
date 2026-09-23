# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

import isaaclab.sim as sim_utils

from .scene_data_backend import SceneDataBackend, SceneDataFormat

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pxr import Usd

REQUIRES_STAGE_AND_MODEL: dict[str, tuple[bool, bool]] = {
    "kit": (True, False),
    "newton_gl": (False, True),
    "newton": (False, True),
    "newton_rtx": (False, True),
    "rerun": (False, True),
    "viser": (False, True),
    "isaac_rtx": (True, False),
    "newton_warp": (False, True),
    "ovrtx": (True, True),
}


def _publication_device(data: Any) -> wp.Device:
    """Return the common device of a populated scene-data publication."""
    arrays = tuple(array for name in data._cls.vars if (array := getattr(data, name)) is not None)
    if not arrays:
        raise ValueError(f"{data._cls.__name__} contains no published arrays.")
    device = arrays[0].device
    if any(array.device != device for array in arrays[1:]):
        raise ValueError(f"{data._cls.__name__} arrays must share one device.")
    return device


def _init_output(output: Any, count: int, device: wp.Device) -> None:
    """Allocate missing output fields on the source publication device."""
    for field_name, field_value in output._cls.vars.items():
        if getattr(output, field_name) is None:
            setattr(output, field_name, wp.empty(count, dtype=field_value.type.dtype, device=device))


class SceneDataProvider:
    def __init__(self, backend: SceneDataBackend):
        """Initialize the scene data provider.

        Args:
            backend: The simulation backend that supplies raw transform data.
        """
        self.backend = backend
        self._num_envs_cache: int | None = None
        self._interactive_scene: Any | None = None
        self._transform_generation = 0
        self._transform_cache: dict[tuple, tuple[int, Any]] = {}
        self._fabric_output: SceneDataFormat.FabricMatrix44 | None = None

    @property
    def transform_generation(self) -> int:
        """Generation of the last consumed transform publication."""
        return self._transform_generation

    def request_transforms(
        self,
        output_format: Any,
        mapping: wp.array | None = None,
        count: int | None = None,
        *,
        scales: wp.array | None = None,
    ) -> Any | None:
        """Request shared transforms, converting at most once per dirty generation and layout.

        A matching native format and ordering returns the producer's pointer without a copy.
        Converted outputs belong to SDP and are reused across consumers and clean requests.
        Fabric consumers bind their stage during initialization with ``_prepare_fabric``.

        Args:
            output_format: Requested :class:`SceneDataFormat` type.
            mapping: Native-to-output indices from :meth:`create_mapping`, or identity ordering.
            count: Destination count when remapping, or the native transform count.
            scales: Static output scales for ``TransposedMatrix44d``, shape [count].

        Returns:
            The requested format, or None when no transforms are published. Treat its arrays as read-only.
        """
        fabric = output_format is SceneDataFormat.FabricMatrix44
        if fabric:
            if mapping is not None or count is not None or scales is not None:
                raise ValueError("Fabric destinations already specify native ordering, count, and authored scale.")
            native_fabric = self.backend.fabric
            if native_fabric is not None:
                if self.backend.fabric_dirty:
                    native_fabric.force_update(0.0, 0.0)
                    self.backend.fabric_dirty = False
                return self._prepare_fabric_output()
        fabric_output = self._prepare_fabric_output() if fabric else None
        source = self.backend.transforms
        if self.backend.transforms_dirty:
            self._transform_generation += 1
            self.backend.transforms_dirty = False
        native_count = self.transform_count
        if native_count == 0:
            return None
        count = native_count if count is None else count
        if mapping is None and count != native_count:
            raise ValueError("A different destination count requires an explicit transform mapping.")
        if scales is not None and output_format is not SceneDataFormat.TransposedMatrix44d:
            raise ValueError("Static scales are supported only for TransposedMatrix44d destinations.")
        if source._cls is output_format and mapping is None and scales is None:
            return source
        key = (output_format, mapping, count, scales)
        cached = self._transform_cache.get(key)
        if (
            cached is not None
            and cached[0] == self._transform_generation
            and (fabric_output is None or cached[1] is fabric_output)
        ):
            return cached[1]
        device = _publication_device(source)
        if fabric:
            self._fabric_write_selection.PrepareForReuse()
            output = fabric_output
        else:
            output = cached[1] if cached is not None else output_format()
            _init_output(output, count, device)
        inputs = [source] if fabric else [source, mapping]
        if output_format is SceneDataFormat.TransposedMatrix44d:
            inputs.append(scales)
        kernel = getattr(ConversionKernels, f"convert_{source._cls.__name__}_to_{output_format.__name__}")
        wp.launch(
            kernel, dim=len(output.indices) if fabric else native_count, inputs=inputs, outputs=[output], device=device
        )
        if fabric:
            wp.synchronize_stream(device)
            # PrepareForReuse rebuilds the output on any Fabric structural change, not just rigid changes.
            if not self._fabric_hierarchy.update_world_xforms_gpu(cached is not None and cached[1] is output):
                raise RuntimeError("Fabric GPU transform hierarchy update failed.")
            wp.synchronize_device(device)
        self._transform_cache[key] = (self._transform_generation, output)
        return output

    def _prepare_fabric(self, stage: Usd.Stage, device: str) -> None:
        """Bind shared Fabric matrices once, preserving engine-owned poses when available."""
        if self._fabric_output is not None:
            return
        # Fabric is supplied by the running Kit application, not the standalone USD wheel.
        import usdrt  # noqa: PLC0415
        import usdrt.hierarchy  # noqa: PLC0415
        from pxr import UsdUtils  # noqa: PLC0415

        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
        native = self.backend.fabric is not None
        if not native:
            fabric_stage.SynchronizeToFabric()
            self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
            )
            self._fabric_hierarchy.update_world_xforms()
            for index, path in enumerate(self.backend.transform_paths):
                prim = fabric_stage.GetPrimAtPath(path)
                if not prim or not prim.HasAPI("PhysicsRigidBodyAPI"):
                    continue
                prim.CreateAttribute("isaaclab:transformIndex", usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(index)
                # Physics publishes absolute poses, including nested bodies. Only visual descendants inherit them.
                self._fabric_hierarchy.set_reset_xform_stack(prim.GetPath().fabricPath, True)
        attrs = [(usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read)]
        if not native:
            attrs.append((usdrt.Sdf.ValueTypeNames.Int, "isaaclab:transformIndex", usdrt.Usd.Access.Read))
            attrs.append((usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:localMatrix", usdrt.Usd.Access.Read))
        self._fabric_selection = fabric_stage.SelectPrims(
            require_applied_schemas=["PhysicsRigidBodyAPI"],
            require_attrs=attrs,
            device=device,
        )
        if not native:
            self._fabric_write_selection = fabric_stage.SelectPrims(
                require_applied_schemas=["PhysicsRigidBodyAPI"],
                require_attrs=[*attrs[:-1], (*attrs[-1][:2], usdrt.Usd.Access.ReadWrite)],
                device=device,
            )
        self._fabric_output = SceneDataFormat.FabricMatrix44()
        if not native:
            self._fabric_output.scales = wp.empty(self.transform_count, dtype=wp.vec3f, device=device)

    def _prepare_fabric_output(self) -> SceneDataFormat.FabricMatrix44:
        """Refresh the shared Fabric selection after topology changes."""
        changed = self._fabric_selection.PrepareForReuse()
        if changed or self._fabric_output.matrices is None:
            output = SceneDataFormat.FabricMatrix44()
            output.matrices = wp.fabricarray(self._fabric_selection, "omni:fabric:worldMatrix")
            if self.backend.fabric is None:
                self._fabric_write_selection.PrepareForReuse()
                output.local_matrices = wp.fabricarray(self._fabric_write_selection, "omni:fabric:localMatrix")
                output.indices = wp.fabricarray(self._fabric_selection, "isaaclab:transformIndex")
                output.scales = self._fabric_output.scales
                if self._fabric_output.matrices is None:
                    wp.launch(
                        ConversionKernels.capture_fabric_scales,
                        dim=len(output.indices),
                        outputs=[output],
                        device=output.scales.device,
                    )
            self._fabric_output = output
        return self._fabric_output

    def set_interactive_scene(self, scene: Any) -> None:
        """Attach the active interactive scene for scene-owned sensor discovery."""
        self._interactive_scene = scene

    def get_interactive_scene(self) -> Any | None:
        """Return the registered interactive scene, if available."""
        return self._interactive_scene

    def get_camera_sensors(self) -> dict[str, Any]:
        """Return Isaac Lab camera sensors keyed by scene sensor name."""
        if self._interactive_scene is None:
            return {}
        try:
            from isaaclab.sensors.camera import Camera
        except ImportError:
            return {}
        return {
            name: sensor
            for name, sensor in getattr(self._interactive_scene, "sensors", {}).items()
            if isinstance(sensor, Camera)
        }

    def get_contact_sensors(self) -> dict[str, Any]:
        """Return Isaac Lab contact sensors keyed by scene sensor name."""
        if self._interactive_scene is None:
            return {}
        from isaaclab.sensors.contact_sensor import BaseContactSensor

        return {
            name: sensor
            for name, sensor in getattr(self._interactive_scene, "sensors", {}).items()
            if isinstance(sensor, BaseContactSensor)
        }

    @property
    def transform_count(self) -> int:
        """Number of transforms available from the sim backend."""
        return self.backend.transform_count

    @property
    def usd_stage(self) -> Usd.Stage | None:
        """Pixar :class:`Usd.Stage` for visualizers and renderers that walk USD.

        Resolves to :attr:`isaaclab.sim.SimulationContext.stage`, falling back to
        ``omni.usd.get_context().get_stage()`` when the simulation context has no
        cached stage. Returns ``None`` on Newton-only headless runs without a USD
        stage.
        """
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        stage = getattr(sim, "stage", None) if sim is not None else None
        if stage is not None:
            return stage
        try:
            import omni.usd

            return omni.usd.get_context().get_stage()
        except Exception:
            return None

    def get_usd_stage(self) -> Usd.Stage | None:
        """Return the USD stage for callers using the older method-style API."""
        return self.usd_stage

    @property
    def num_envs(self) -> int:
        """Number of environments discovered from ``/World/envs/env_<id>`` prims.

        Cached on first call. Returns ``0`` when no USD stage is available or when
        no ``/World/envs/env_<id>`` prims exist.
        """
        if self._num_envs_cache is not None:
            return self._num_envs_cache
        self._num_envs_cache = _discover_num_envs(self.usd_stage)
        return self._num_envs_cache

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Per-camera, per-environment world transforms discovered from USD.

        Returns:
            Dictionary with keys ``order`` (list of template prim paths using
            ``env_%d``), ``positions`` and ``orientations`` (per-camera, per-env
            lists, with ``None`` for absent envs), and ``num_envs``. Returns
            ``None`` when no USD stage is available.
        """
        return _walk_camera_prims(self.usd_stage)

    def get_transforms(
        self,
        output: SceneDataFormat.Vec3_Quat
        | SceneDataFormat.Transform
        | SceneDataFormat.Matrix44
        | SceneDataFormat.Vec3_Matrix33,
        mapping: wp.array(dtype=wp.int32) | None = None,
        allow_passthrough: bool = True,
    ) -> bool:
        """Convert sim backend transforms into the requested output format.

        When the backend's native format matches ``output``, data is either passed
        through by reference (``allow_passthrough=True``) or deep-copied. Otherwise a
        Warp conversion kernel is launched to transform the data, applying ``mapping``
        to reorder the output if provided.

        Args:
            output: A pre-allocated :class:`SceneDataFormat` struct that determines the
                target format. Uninitialized (``None``) fields are allocated automatically
                when a conversion kernel is needed.
            mapping: Optional index remapping array produced by
                :meth:`create_mapping`. When ``None``, input and output indices are
                identical.
            allow_passthrough: If ``True`` and the formats already match, the output
                struct's fields are set to reference the input arrays directly
                (zero-copy). If ``False``, the data is always copied.

        Returns:
            ``True`` if the conversion succeeded, ``False`` if no suitable conversion
            kernel exists for the input/output format pair.
        """
        input = self.backend.transforms

        if mapping is None and type(input) is type(output):
            if allow_passthrough:
                for field_name in input._cls.vars:
                    setattr(output, field_name, getattr(input, field_name))
            else:
                _init_output(output, self.transform_count, _publication_device(input))
                for field_name in input._cls.vars:
                    wp.copy(getattr(output, field_name), getattr(input, field_name))
            return True

        conversion_kernel_name = f"convert_{input._cls.__name__}_to_{output._cls.__name__}"

        if conversion_kernel := getattr(ConversionKernels, conversion_kernel_name, None):
            device = _publication_device(input)
            _init_output(output, self.transform_count, device)
            wp.launch(
                kernel=conversion_kernel,
                dim=self.transform_count,
                inputs=[input, mapping],
                outputs=[output],
                device=device,
            )
            return True

        return False

    def init_output(
        self,
        output: SceneDataFormat.Vec3_Quat
        | SceneDataFormat.Transform
        | SceneDataFormat.Matrix44
        | SceneDataFormat.Vec3_Matrix33,
    ):
        """Allocate any uninitialized fields in ``output`` with empty Warp arrays.

        Only fields that are currently ``None`` are allocated; already-initialized
        fields are left untouched.

        Args:
            output: A :class:`SceneDataFormat` struct whose ``None``-valued fields
                will be replaced with empty arrays of length :attr:`transform_count`.
        """
        input = self.backend.transforms
        _init_output(output, self.transform_count, _publication_device(input))

    def create_mapping(self, paths: list[str | None]) -> wp.array(dtype=wp.int32) | None:
        """Create an index mapping from sim backend transforms to desired output ordering.

        For each transform in the sim backend, the resulting array stores the index into
        ``paths`` where that transform should be written. Transforms whose path does not
        appear in ``paths`` (or maps to ``None``) receive an index of ``-1`` and are
        skipped during conversion.

        Args:
            paths: Desired output ordering expressed as prim paths. Use ``None`` for
                slots that should not receive any transform.

        Returns:
            A Warp int32 array of length :attr:`transform_count` containing the
            remapped indices, or ``None`` if the sim backend provides no transform
            paths or if no mapping is needed.
        """
        if input_paths := self.backend.transform_paths:
            # The map keeps resolution linear in the number of paths. For duplicate
            # paths the first occurrence wins, matching ``list.index``.
            path_to_out: dict[str | None, int] = {}
            for out_idx, out_path in enumerate(paths):
                if out_path not in path_to_out:
                    path_to_out[out_path] = out_idx
            mapping = [path_to_out.get(path, -1) for path in input_paths]
            if not np.array_equal(mapping, np.arange(len(input_paths))):
                input = self.backend.transforms
                return wp.array(mapping, dtype=wp.int32, device=_publication_device(input))
        return None

    def create_geometry_mapping(
        self,
        paths: list[str | None],
        particle_offsets: list[int],
    ) -> wp.array(dtype=wp.int32) | None:
        """Create a mapping from backend geometry entities to consumer particle offsets.

        For each geometry entity in the sim backend, the resulting array stores the
        destination particle offset in the consumer buffer. Entities whose path does not
        appear in ``paths`` receive ``-1`` and are skipped during copy.

        Args:
            paths: Desired consumer entity paths in particle-offset order.
            particle_offsets: Particle offset in the consumer buffer for each ``paths`` entry.

        Returns:
            A Warp int32 array of length ``len(geometry_paths)`` containing destination
            particle offsets, or ``None`` when no geometry is available or every entity
            maps identically in order.
        """
        input_paths = self.backend.geometry_paths
        input_counts = self.backend.geometry_counts
        if not input_paths or not input_counts:
            return None

        path_to_offset = {
            path: offset for path, offset in zip(paths, particle_offsets, strict=True) if path is not None
        }
        mapping = [-1] * len(input_paths)
        identity = True
        flat_offset = 0
        for index, path in enumerate(input_paths):
            dest_offset = path_to_offset.get(path, -1)
            mapping[index] = dest_offset
            if dest_offset != flat_offset:
                identity = False
            flat_offset += int(input_counts[index])

        if identity and all(value >= 0 for value in mapping):
            return None
        points = self.backend.points
        return wp.array(mapping, dtype=wp.int32, device=_publication_device(points))

    def get_points(
        self,
        output: SceneDataFormat.Points,
        mapping: wp.array(dtype=wp.int32) | None = None,
        allow_passthrough: bool = True,
    ) -> bool:
        """Copy sim backend geometry points into ``output``.

        Args:
            output: Pre-allocated :class:`SceneDataFormat.Points` buffer (typically aliased
                to shadow ``particle_q``).
            mapping: Optional destination particle-offset array from
                :meth:`create_geometry_mapping`.
            allow_passthrough: When ``True`` and no mapping is needed, alias ``output.points``
                directly to the backend buffer.

        Returns:
            ``True`` when points were copied or passed through, ``False`` when the backend
            exposes no geometry.
        """
        if self.point_count == 0:
            return False

        input_points = self.backend.points
        if input_points.points is None:
            return False

        if mapping is None and allow_passthrough:
            output.points = input_points.points
            return True

        if output.points is None:
            output.points = wp.empty(self.point_count, dtype=wp.vec3f, device=input_points.points.device)

        entity_counts = self.backend.geometry_counts
        if not entity_counts:
            wp.copy(output.points, input_points.points)
            return True

        from isaaclab.scene_data.geometry_points import scatter_geometry_points

        scatter_geometry_points(
            input_points.points,
            output.points,
            entity_counts,
            mapping,
            device=str(output.points.device),
        )
        return True

    @property
    def point_count(self) -> int:
        """Number of geometry points available from the sim backend."""
        return self.backend.point_count


class ConversionKernels:
    @wp.kernel(enable_backward=False)
    def capture_fabric_scales(output: SceneDataFormat.FabricMatrix44):
        """Capture authored scales before pose updates introduce rotation round-off."""
        index = wp.tid()
        matrix = wp.mat44f(output.matrices[index])
        output.scales[output.indices[index]] = wp.vec3f(
            wp.length(wp.vec3f(matrix[0, 0], matrix[0, 1], matrix[0, 2])),
            wp.length(wp.vec3f(matrix[1, 0], matrix[1, 1], matrix[1, 2])),
            wp.length(wp.vec3f(matrix[2, 0], matrix[2, 1], matrix[2, 2])),
        )

    @wp.func
    def fabric_transform(pose: wp.transformf, scale: wp.vec3f) -> wp.mat44d:
        """Preserve the destination's captured authored scale while replacing its pose."""
        return wp.mat44d(
            wp.transpose(
                wp.transform_compose(wp.transform_get_translation(pose), wp.transform_get_rotation(pose), scale)
            )
        )

    @wp.kernel(enable_backward=False)
    def convert_Transform_to_FabricMatrix44(input: SceneDataFormat.Transform, output: SceneDataFormat.FabricMatrix44):
        i = wp.tid()
        index = output.indices[i]
        output.local_matrices[i] = ConversionKernels.fabric_transform(input.transforms[index], output.scales[index])

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Quat_to_FabricMatrix44(input: SceneDataFormat.Vec3_Quat, output: SceneDataFormat.FabricMatrix44):
        i = wp.tid()
        index = output.indices[i]
        pose = wp.transformf(input.positions[index], input.orientations[index])
        output.local_matrices[i] = ConversionKernels.fabric_transform(pose, output.scales[index])

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Matrix33_to_FabricMatrix44(
        input: SceneDataFormat.Vec3_Matrix33, output: SceneDataFormat.FabricMatrix44
    ):
        i = wp.tid()
        index = output.indices[i]
        pose = wp.transformf(input.positions[index], wp.quat_from_matrix(input.orientations[index]))
        output.local_matrices[i] = ConversionKernels.fabric_transform(pose, output.scales[index])

    @wp.kernel(enable_backward=False)
    def convert_Matrix44_to_FabricMatrix44(input: SceneDataFormat.Matrix44, output: SceneDataFormat.FabricMatrix44):
        i = wp.tid()
        index = output.indices[i]
        output.local_matrices[i] = ConversionKernels.fabric_transform(
            wp.transform_from_matrix(input.matrices[index]), output.scales[index]
        )

    @wp.func
    def get_output_index(tid: wp.int32, mapping: wp.array(dtype=wp.int32)) -> wp.int32:
        if not mapping.shape[0]:
            return tid
        if tid < mapping.shape[0]:
            return mapping[tid]
        return wp.int32(-1)

    @wp.func
    def transposed_matrix(matrix: wp.mat44f, scales: wp.array(dtype=wp.vec3f), index: int) -> wp.mat44d:
        result = wp.mat44d(wp.transpose(matrix))
        if scales.shape[0]:
            scale = scales[index]
            for row in range(3):
                for column in range(3):
                    result[row, column] = result[row, column] * wp.float64(scale[row])
        return result

    @wp.kernel(enable_backward=False)
    def convert_Transform_to_TransposedMatrix44d(
        input: SceneDataFormat.Transform,
        mapping: wp.array(dtype=wp.int32),
        scales: wp.array(dtype=wp.vec3f),
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        tid = wp.tid()
        index = ConversionKernels.get_output_index(tid, mapping)
        if index > -1:
            output.matrices[index] = ConversionKernels.transposed_matrix(
                wp.transform_to_matrix(input.transforms[tid]), scales, index
            )

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Quat_to_TransposedMatrix44d(
        input: SceneDataFormat.Vec3_Quat,
        mapping: wp.array(dtype=wp.int32),
        scales: wp.array(dtype=wp.vec3f),
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        tid = wp.tid()
        index = ConversionKernels.get_output_index(tid, mapping)
        if index > -1:
            pose = wp.transformf(input.positions[tid], input.orientations[tid])
            output.matrices[index] = ConversionKernels.transposed_matrix(wp.transform_to_matrix(pose), scales, index)

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Matrix33_to_TransposedMatrix44d(
        input: SceneDataFormat.Vec3_Matrix33,
        mapping: wp.array(dtype=wp.int32),
        scales: wp.array(dtype=wp.vec3f),
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        tid = wp.tid()
        index = ConversionKernels.get_output_index(tid, mapping)
        if index > -1:
            pose = wp.transformf(input.positions[tid], wp.quat_from_matrix(input.orientations[tid]))
            output.matrices[index] = ConversionKernels.transposed_matrix(wp.transform_to_matrix(pose), scales, index)

    @wp.kernel(enable_backward=False)
    def convert_Matrix44_to_TransposedMatrix44d(
        input: SceneDataFormat.Matrix44,
        mapping: wp.array(dtype=wp.int32),
        scales: wp.array(dtype=wp.vec3f),
        output: SceneDataFormat.TransposedMatrix44d,
    ):
        tid = wp.tid()
        index = ConversionKernels.get_output_index(tid, mapping)
        if index > -1:
            output.matrices[index] = ConversionKernels.transposed_matrix(input.matrices[tid], scales, index)

    @wp.kernel
    def convert_Vec3_Quat_to_Vec3_Quat(
        input: SceneDataFormat.Vec3_Quat, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Quat
    ):
        """Pass-through Vec3/Quat"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.positions[idx] = input.positions[tid]
            output.orientations[idx] = input.orientations[tid]

    @wp.kernel
    def convert_Vec3_Quat_to_Vec3_Matrix33(
        input: SceneDataFormat.Vec3_Quat, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Matrix33
    ):
        """Convert Vec3/Quat to Vec3/Matrix33"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.positions[idx] = input.positions[tid]
            output.orientations[idx] = wp.quat_to_matrix(input.orientations[tid])

    @wp.kernel
    def convert_Vec3_Quat_to_Transform(
        input: SceneDataFormat.Vec3_Quat, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Transform
    ):
        """Convert Vec3/Quat to Transform"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.transforms[idx] = wp.transformf(input.positions[tid], input.orientations[tid])

    @wp.kernel
    def convert_Vec3_Quat_to_Matrix44(
        input: SceneDataFormat.Vec3_Quat, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Matrix44
    ):
        """Convert Vec3/Quat to Matrix44"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.matrices[idx] = wp.transform_to_matrix(wp.transformf(input.positions[tid], input.orientations[tid]))

    @wp.kernel
    def convert_Vec3_Matrix33_to_Vec3_Quat(
        input: SceneDataFormat.Vec3_Matrix33, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Quat
    ):
        """Convert Vec3/Matrix33 to Vec3/Quat"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.positions[idx] = input.positions[tid]
            output.orientations[idx] = wp.quat_from_matrix(input.orientations[tid])

    @wp.kernel
    def convert_Vec3_Matrix33_to_Vec3_Matrix33(
        input: SceneDataFormat.Vec3_Matrix33, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Matrix33
    ):
        """Pass-through Vec3/Matrix33"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.positions[idx] = input.positions[tid]
            output.orientations[idx] = input.orientations[tid]

    @wp.kernel
    def convert_Vec3_Matrix33_to_Transform(
        input: SceneDataFormat.Vec3_Matrix33, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Transform
    ):
        """Convert Vec3/Matrix33 to Transform"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.transforms[idx] = wp.transformf(input.positions[tid], wp.quat_from_matrix(input.orientations[tid]))

    @wp.kernel
    def convert_Vec3_Matrix33_to_Matrix44(
        input: SceneDataFormat.Vec3_Matrix33, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Matrix44
    ):
        """Convert Vec3/Matrix33 to Matrix44"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            transform = wp.transformf(input.positions[tid], wp.quat_from_matrix(input.orientations[tid]))
            output.matrices[idx] = wp.transform_to_matrix(transform)

    @wp.kernel
    def convert_Transform_to_Vec3_Quat(
        input: SceneDataFormat.Transform, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Quat
    ):
        """Convert Transform to Vec3/Quat"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.positions[idx] = wp.transform_get_translation(input.transforms[tid])
            output.orientations[idx] = wp.transform_get_rotation(input.transforms[tid])

    @wp.kernel
    def convert_Transform_to_Vec3_Matrix33(
        input: SceneDataFormat.Transform, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Matrix33
    ):
        """Convert Transform to Vec3/Matrix33"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.positions[idx] = wp.transform_get_translation(input.transforms[tid])
            output.orientations[idx] = wp.quat_to_matrix(wp.transform_get_rotation(input.transforms[tid]))

    @wp.kernel
    def convert_Transform_to_Transform(
        input: SceneDataFormat.Transform, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Transform
    ):
        """Pass-through Transform"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.transforms[idx] = input.transforms[tid]

    @wp.kernel
    def convert_Transform_to_Matrix44(
        input: SceneDataFormat.Transform, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Matrix44
    ):
        """Convert Transform to Matrix44"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.matrices[idx] = wp.transform_to_matrix(input.transforms[tid])

    @wp.kernel
    def convert_Matrix44_to_Vec3_Quat(
        input: SceneDataFormat.Matrix44, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Quat
    ):
        """Convert Matrix44 to Vec3/Quat"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            transform = wp.transform_from_matrix(input.matrices[tid])
            output.positions[idx] = wp.transform_get_translation(transform)
            output.orientations[idx] = wp.transform_get_rotation(transform)

    @wp.kernel
    def convert_Matrix44_to_Vec3_Matrix33(
        input: SceneDataFormat.Matrix44, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Vec3_Matrix33
    ):
        """Convert Matrix44 to Vec3/Matrix33"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            transform = wp.transform_from_matrix(input.matrices[tid])
            output.positions[idx] = wp.transform_get_translation(transform)
            output.orientations[idx] = wp.quat_to_matrix(wp.transform_get_rotation(transform))

    @wp.kernel
    def convert_Matrix44_to_Transform(
        input: SceneDataFormat.Matrix44, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Transform
    ):
        """Convert Matrix44 to Transform"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.transforms[idx] = wp.transform_from_matrix(input.matrices[tid])

    @wp.kernel
    def convert_Matrix44_to_Matrix44(
        input: SceneDataFormat.Matrix44, mapping: wp.array(dtype=wp.int32), output: SceneDataFormat.Matrix44
    ):
        """Pass-through Matrix44"""
        tid = wp.tid()
        idx = ConversionKernels.get_output_index(tid, mapping)
        if idx > -1:
            output.matrices[idx] = input.matrices[tid]


_ENV_NAME_RE = re.compile(r"^env_(\d+)$")
_ENV_PATH_RE = re.compile(r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)")


def _discover_num_envs(stage: Usd.Stage | None) -> int:
    """Infer environment count from ``/World/envs/env_<id>`` prim names on ``stage``.

    Args:
        stage: USD stage to inspect, or ``None``.

    Returns:
        Number of environments discovered, or ``0`` when ``stage`` is ``None`` or no
        ``/World/envs/env_<id>`` prims exist.
    """
    if stage is None:
        return 0
    max_env_id = -1
    envs_root = stage.GetPrimAtPath("/World/envs")
    if envs_root.IsValid():
        for child in envs_root.GetChildren():
            if match := _ENV_NAME_RE.match(child.GetName()):
                max_env_id = max(max_env_id, int(match.group(1)))
    return max_env_id + 1 if max_env_id >= 0 else 0


def _walk_camera_prims(stage: Usd.Stage | None) -> dict[str, Any] | None:
    """Walk ``stage`` and collect per-environment camera transforms.

    Args:
        stage: USD stage to traverse, or ``None``.

    Returns:
        Dictionary with keys ``order`` (template prim paths using ``env_%d``),
        ``positions``, ``orientations`` (per-camera, per-env, with ``None`` for
        absent envs), and ``num_envs``. Returns ``None`` when ``stage`` is ``None``.
    """
    if stage is None:
        return None

    from pxr import UsdGeom  # noqa: PLC0415

    shared_paths: list[str] = []
    instances: dict[str, list[tuple[int, str]]] = {}
    num_envs = -1

    stage_prims = deque([stage.GetPseudoRoot()])
    while stage_prims:
        prim = stage_prims.popleft()
        prim_path = prim.GetPath().pathString

        world_id = 0
        template_path = prim_path
        if match := _ENV_PATH_RE.match(prim_path):
            world_id = int(match.group("id"))
            template_path = match.group("root") + "%d" + match.group("path")
            if world_id > num_envs:
                num_envs = world_id

        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        if prim.IsA(UsdGeom.Camera):
            instances.setdefault(template_path, []).append((world_id, prim_path))
            if template_path not in shared_paths:
                shared_paths.append(template_path)

        if hasattr(UsdGeom, "TraverseInstanceProxies"):
            child_prims = prim.GetFilteredChildren(UsdGeom.TraverseInstanceProxies())
        else:
            child_prims = prim.GetChildren()
        if child_prims:
            stage_prims.extend(child_prims)

    num_envs += 1
    positions: list[list[list[float] | None]] = []
    orientations: list[list[list[float] | None]] = []

    for template_path in shared_paths:
        per_world_pos: list[list[float] | None] = [None] * num_envs
        per_world_ori: list[list[float] | None] = [None] * num_envs
        for world_id, prim_path in instances.get(template_path, []):
            if world_id < 0 or world_id >= num_envs:
                continue
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            pos, ori = sim_utils.resolve_prim_pose(prim)
            per_world_pos[world_id] = [float(pos[0]), float(pos[1]), float(pos[2])]
            per_world_ori[world_id] = [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
        positions.append(per_world_pos)
        orientations.append(per_world_ori)

    return {"order": shared_paths, "positions": positions, "orientations": orientations, "num_envs": num_envs}


if __name__ == "__main__":

    class ExampleSceneDataBackend(SceneDataBackend):
        def __init__(self):
            self._transforms = SceneDataFormat.Transform()
            self._transforms.transforms = wp.array([[x, 0, 0, 0, 0, 0, 1] for x in range(10)], dtype=wp.transformf)
            self.transforms_dirty = True

        @property
        def transforms(self) -> SceneDataFormat.Transform:
            return self._transforms

        @property
        def transform_count(self) -> int:
            return len(self._transforms.transforms)

        @property
        def transform_paths(self) -> list[str]:
            return [f"/world/shape_{index}" for index in range(self.transform_count)]

    sim = ExampleSceneDataBackend()
    sdp = SceneDataProvider(sim)
    mapping = sdp.create_mapping(sim.transform_paths[::-1])
    output_data = sdp.request_transforms(SceneDataFormat.Vec3_Matrix33, mapping)
    print(output_data.positions.numpy())
