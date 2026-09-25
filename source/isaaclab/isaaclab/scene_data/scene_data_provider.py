# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections import deque
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

import numpy as np
import warp as wp

from .. import sim as sim_utils
from .geometry_points import convert_geometry_fabric_kernel, convert_geometry_points_kernel
from .scene_data_backend import SceneDataBackend, SceneDataFormat

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
    "ovrtx": (True, False),
}


def _publication_device(data: Any) -> wp.Device:
    """Return the common device of a populated scene-data publication."""
    data_format = data._cls
    arrays = tuple(array for name in data_format.vars if (array := getattr(data, name)) is not None)
    if not arrays:
        raise ValueError(f"{data_format.__name__} contains no published arrays.")
    device = arrays[0].device
    if any(array.device != device for array in arrays[1:]):
        raise ValueError(f"{data_format.__name__} arrays must share one device.")
    return device


def _init_output(output: Any, count: int, device: wp.Device) -> None:
    """Allocate missing output fields on the source publication device."""
    for field_name, field_value in output._cls.vars.items():
        if getattr(output, field_name) is None:
            setattr(output, field_name, wp.empty(count, dtype=field_value.type.dtype, device=device))


class SceneDataProvider:
    """Borrow or convert published arrays; producers own native refresh, renderers own destination lifecycle."""

    def __init__(self, backend: SceneDataBackend):
        """Initialize the scene data provider.

        Args:
            backend: The simulation backend that supplies raw transform data.
        """
        self.backend = backend
        self._num_envs_cache: int | None = None
        self._interactive_scene: Any | None = None
        self._transform_cache: dict[tuple, tuple[int, Any]] = {}
        self._geometry_view_cache: tuple | None = None
        self._geometry_destination_cache = WeakKeyDictionary()

    def get_transforms(
        self,
        output: SceneDataFormat.Vec3_Quat
        | SceneDataFormat.Transform
        | SceneDataFormat.Matrix44
        | SceneDataFormat.Vec3_Matrix33
        | SceneDataFormat.TransposedMatrix44d
        | SceneDataFormat.FabricMatrix44,
        mapping: wp.array | wp.fabricarray | None = None,
        allow_passthrough: bool = True,
        *,
        count: int | None = None,
        scales: wp.array | None = None,
    ) -> bool:
        """Bind shared transforms or write them directly into caller-owned output arrays.

        With passthrough enabled, matching native arrays are borrowed without a copy; other
        layouts share SDP-owned buffers converted once per producer version. Treat these arrays
        as read-only. With passthrough disabled, conversion writes directly into ``output``.
        Fabric destinations must already be bound by their rendering owner.

        Args:
            output: A :class:`SceneDataFormat` struct instance specifying the requested format.
                Missing non-Fabric arrays are allocated when passthrough is disabled.
            mapping: Native-to-output indices from :meth:`create_mapping`, or identity ordering.
                Fabric destinations use their native output-to-source index attribute.
            allow_passthrough: Whether to bind shared arrays instead of writing caller-owned arrays.
            count: Destination count when remapping, or the native transform count.
            scales: Static output scales for ``TransposedMatrix44d``, shape [count], or
                source-indexed authored scales for Fabric. Mapping and scales are immutable
                for a binding's lifetime; replace their arrays when the layout changes.

        Returns:
            True if transforms are available in ``output``, False if no transforms are published
            or the format conversion is unsupported.
        """
        # Warp exposes the struct's field/type descriptor as _cls, not its Python type.
        output_format = output._cls
        fabric = output_format is SceneDataFormat.FabricMatrix44
        source = self.backend.get_transforms(output_format)
        source_format = source._cls
        version = self.backend.transforms_version
        native_count = next(
            (len(array) for name in source_format.vars if (array := getattr(source, name)) is not None), 0
        )
        if native_count == 0:
            return False
        count = native_count if count is None else count
        if mapping is None and count != native_count:
            raise ValueError("A different destination count requires an explicit transform mapping.")
        if scales is not None and output_format not in (
            SceneDataFormat.TransposedMatrix44d,
            SceneDataFormat.FabricMatrix44,
        ):
            raise ValueError("Static scales require double-precision row-vector matrix destinations.")
        if source_format is output_format and mapping is None and scales is None:
            result = source
            if not allow_passthrough:
                _init_output(output, count, _publication_device(source))
                for name in output_format.vars:
                    wp.copy(getattr(output, name), getattr(source, name))
                return True
        else:
            # A Fabric binding keeps its authored scales across selection reallocations.
            key = (output_format, scales) if fabric else (output_format, mapping, count, scales)
            cached = self._transform_cache.get(key) if allow_passthrough else None
            if not allow_passthrough or fabric:
                result = output
            else:
                result = cached[1] if cached is not None else output_format()
            if cached is None or cached[0] != version or cached[1] is not result:
                # Fabric changes storage and indexing, not the matrix conversion.
                format_name = "TransposedMatrix44d" if fabric else output_format.__name__
                kernel = getattr(ConversionKernels, f"convert_{source_format.__name__}_to_{format_name}", None)
                if kernel is None:
                    return False
                device = _publication_device(source)
                _init_output(result, count, device)
                inputs = [source, mapping if mapping is not None else wp.array(dtype=wp.int32)]
                if output_format is SceneDataFormat.TransposedMatrix44d or fabric:
                    inputs.append(scales)
                wp.launch(
                    kernel,
                    dim=len(result.matrices) if fabric else native_count,
                    inputs=inputs,
                    outputs=[result],
                    device=device,
                )
                if allow_passthrough:
                    self._transform_cache[key] = (version, result)
        for name in output_format.vars:
            setattr(output, name, getattr(result, name))
        return True

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
            if len(paths) != len(input_paths) or not np.array_equal(mapping, np.arange(len(input_paths))):
                input = self.backend.transforms
                return wp.array(mapping, dtype=wp.int32, device=_publication_device(input))
        return None

    def get_geometry_points(
        self,
        *,
        output: wp.array | SceneDataFormat.FabricPoints | None = None,
        offsets: dict[str, int] | None = None,
    ) -> dict[str, wp.array] | wp.array | SceneDataFormat.FabricPoints:
        """Borrow visual point views or convert directly into the requested native destination.

        Producers supply exact visual prim paths, native pointers and immutable interpolation
        metadata. SDP performs interpolation and destination reordering together, once per
        publication version and output layout. Only cross-device destinations require staging.

        Args:
            output: Consumer-owned world-space point buffer [m] or native Fabric destination.
                Omit to borrow shared point views. ``FabricPoints`` without offsets borrows
                native Fabric storage when the producer publishes it.
            offsets: Visual prim paths mapped to flat-buffer offsets or Fabric array indices.
                Keep this mapping immutable for the destination's lifetime.

        Returns:
            Read-only world-space views by visual prim path when no output is supplied;
            otherwise the supplied destination, populated directly on the same device.
        """
        fabric = output is not None and not isinstance(output, wp.array)
        requested_format = SceneDataFormat.FabricPoints if fabric and offsets is None else SceneDataFormat.Points
        batches = self.backend.get_geometry_batches(requested_format)
        if fabric and len(batches) == 1 and batches[0][0]._cls is SceneDataFormat.FabricPoints:
            output.points = batches[0][0].points
            return output
        version = self.backend.geometry_version
        if output is not None:
            if offsets is None:
                raise ValueError("A geometry destination requires its visual-path offsets.")
            destination = output.points if fabric else output
            cached = self._geometry_destination_cache.get(output)
            if cached is None or cached[2] is not offsets:
                jobs, bound = [], set()
                for source, ranges in batches:
                    selected = {path: bounds for path, bounds in ranges.items() if path in offsets}
                    count = sum(count for _, count in selected.values())
                    indices = np.empty((3 if fabric else 2, count), dtype=np.int32)
                    cursor = 0
                    for path, (start, count) in selected.items():
                        vertices = np.arange(count)
                        indices[0, cursor : cursor + count] = start + vertices
                        indices[1, cursor : cursor + count] = offsets[path] if fabric else offsets[path] + vertices
                        if fabric:
                            indices[2, cursor : cursor + count] = vertices
                        elif offsets[path] < 0 or offsets[path] + count > len(output):
                            raise ValueError("Geometry destination range exceeds its output buffer.")
                        cursor += count
                    device = _publication_device(source)
                    source_indices = wp.array(indices[0], device=device)
                    destination_indices = wp.array(
                        indices[1:].T if fabric else indices[1],
                        dtype=wp.vec2i if fabric else wp.int32,
                        device=destination.device,
                    )
                    transfer = None
                    if cursor and device != destination.device:
                        staging = SceneDataFormat.Points()
                        staging.points = wp.empty(
                            cursor, wp.vec3f, device=destination.device, pinned=destination.device.is_cpu
                        )
                        transfer = (wp.empty(cursor, wp.vec3f, device=device), staging, wp.array(dtype=wp.int32))
                    jobs.append((source_indices, destination_indices, transfer))
                    bound.update(selected)
                if bound != offsets.keys():
                    raise KeyError(f"Geometry destinations have no native publication: {offsets.keys() - bound}")
            else:
                previous_version, jobs, _ = cached
                if previous_version == version:
                    return output
            for (source, _), (source_indices, destination_indices, transfer) in zip(batches, jobs, strict=True):
                if len(source_indices):
                    device = _publication_device(source)
                    count = len(source_indices)
                    if transfer is not None:
                        packed, staging, identity = transfer
                        wp.launch(
                            convert_geometry_points_kernel,
                            dim=count,
                            inputs=[source, source_indices, identity, packed],
                            device=device,
                        )
                        wp.copy(staging.points, packed)
                        if destination.device.is_cpu:
                            wp.synchronize_stream(device)
                        source, source_indices = staging, identity
                    wp.launch(
                        convert_geometry_fabric_kernel if fabric else convert_geometry_points_kernel,
                        dim=count,
                        inputs=[source, source_indices, destination_indices, destination],
                        device=destination.device,
                    )
            # Retain conversion buffers, never the consumer's destination or slices of it.
            self._geometry_destination_cache[output] = (version, jobs, offsets)
            return output
        if offsets is not None:
            raise ValueError("Geometry offsets require a destination.")
        cached = self._geometry_view_cache
        if cached is None:
            views, jobs = {}, []
            for source, ranges in batches:
                device = _publication_device(source)
                count = max((start + count for start, count in ranges.values()), default=0)
                buffer = (
                    source.points if source._cls is SceneDataFormat.Points else wp.empty(count, wp.vec3f, device=device)
                )
                indices = wp.array(dtype=wp.int32, device=device)
                jobs.append((indices, buffer, ranges))
                views.update((path, buffer[start : start + count]) for path, (start, count) in ranges.items())
        else:
            previous_version, views, jobs = cached
            if previous_version == version:
                return views

        for index, ((source, _), (indices, buffer, ranges)) in enumerate(zip(batches, jobs, strict=True)):
            if source._cls is SceneDataFormat.Points:
                if buffer is not source.points:
                    buffer = source.points
                    jobs[index] = (indices, buffer, ranges)
                    views.update((path, buffer[start : start + count]) for path, (start, count) in ranges.items())
                continue
            if len(buffer):
                wp.launch(
                    convert_geometry_points_kernel,
                    dim=len(buffer),
                    inputs=[source, indices, indices, buffer],
                    device=buffer.device,
                )
        self._geometry_view_cache = (version, views, jobs)
        return views


class ConversionKernels:
    @wp.func
    def get_output_index(tid: wp.int32, mapping: wp.array(dtype=wp.int32)) -> wp.int32:
        if not mapping.shape[0]:
            return tid
        if tid < mapping.shape[0]:
            return mapping[tid]
        return wp.int32(-1)

    @wp.func
    def matrix_indices(tid: int, mapping: wp.array(dtype=wp.int32)):
        """Return source, destination, and authored-scale indices."""
        index = ConversionKernels.get_output_index(tid, mapping)
        return tid, index, index

    @wp.func
    def matrix_indices(tid: int, mapping: wp.fabricarray(dtype=wp.int32)):  # noqa: F811 - Warp overload
        return mapping[tid], tid, mapping[tid]

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
        input: SceneDataFormat.Transform, mapping: Any, scales: wp.array(dtype=wp.vec3f), output: Any
    ):
        source, index, scale_index = ConversionKernels.matrix_indices(wp.tid(), mapping)
        if index > -1:
            output.matrices[index] = ConversionKernels.transposed_matrix(
                wp.transform_to_matrix(input.transforms[source]), scales, scale_index
            )

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Quat_to_TransposedMatrix44d(
        input: SceneDataFormat.Vec3_Quat, mapping: Any, scales: wp.array(dtype=wp.vec3f), output: Any
    ):
        source, index, scale_index = ConversionKernels.matrix_indices(wp.tid(), mapping)
        if index > -1:
            pose = wp.transformf(input.positions[source], input.orientations[source])
            output.matrices[index] = ConversionKernels.transposed_matrix(
                wp.transform_to_matrix(pose), scales, scale_index
            )

    @wp.kernel(enable_backward=False)
    def convert_Vec3_Matrix33_to_TransposedMatrix44d(
        input: SceneDataFormat.Vec3_Matrix33, mapping: Any, scales: wp.array(dtype=wp.vec3f), output: Any
    ):
        source, index, scale_index = ConversionKernels.matrix_indices(wp.tid(), mapping)
        if index > -1:
            pose = wp.transformf(input.positions[source], wp.quat_from_matrix(input.orientations[source]))
            output.matrices[index] = ConversionKernels.transposed_matrix(
                wp.transform_to_matrix(pose), scales, scale_index
            )

    @wp.kernel(enable_backward=False)
    def convert_Matrix44_to_TransposedMatrix44d(
        input: SceneDataFormat.Matrix44, mapping: Any, scales: wp.array(dtype=wp.vec3f), output: Any
    ):
        source, index, scale_index = ConversionKernels.matrix_indices(wp.tid(), mapping)
        if index > -1:
            output.matrices[index] = ConversionKernels.transposed_matrix(input.matrices[source], scales, scale_index)

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
