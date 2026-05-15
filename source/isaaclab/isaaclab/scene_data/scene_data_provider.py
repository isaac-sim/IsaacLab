# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import re
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .scene_data_backend import SceneDataBackend, SceneDataFormat

if TYPE_CHECKING:
    from pxr import Usd


class SceneDataProvider:
    def __init__(self, backend: SceneDataBackend):
        """Initialize the scene data provider.

        Args:
            backend: The simulation backend that supplies raw transform data.
        """
        self.backend = backend
        self._num_envs_cache: int | None = None

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
                self.init_output(output)
                for field_name in input._cls.vars:
                    wp.copy(getattr(output, field_name), getattr(input, field_name))
            return True

        conversion_kernel_name = f"convert_{input._cls.__name__}_to_{output._cls.__name__}"

        if conversion_kernel := getattr(ConversionKernels, conversion_kernel_name, None):
            self.init_output(output)
            wp.launch(kernel=conversion_kernel, dim=self.transform_count, inputs=[input, mapping], outputs=[output])
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
        for field_name, field_value in output._cls.vars.items():
            if getattr(output, field_name) is None:
                setattr(output, field_name, wp.empty(self.transform_count, dtype=field_value.type.dtype))

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
            mapping = [-1] * len(input_paths)
            for i, path in enumerate(input_paths):
                with contextlib.suppress(ValueError):
                    mapping[i] = paths.index(path)
            if not np.array_equal(mapping, np.arange(len(input_paths))):
                return wp.array(mapping, dtype=wp.int32)
        return None


class ConversionKernels:
    @wp.func
    def get_output_index(tid: wp.int32, mapping: wp.array(dtype=wp.int32)) -> wp.int32:
        if not mapping.shape[0]:
            return tid
        if tid < mapping.shape[0]:
            return mapping[tid]
        return wp.int32(-1)

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

    from pxr import UsdGeom

    import isaaclab.sim as isaaclab_sim

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
            pos, ori = isaaclab_sim.resolve_prim_pose(prim)
            per_world_pos[world_id] = [float(pos[0]), float(pos[1]), float(pos[2])]
            per_world_ori[world_id] = [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
        positions.append(per_world_pos)
        orientations.append(per_world_ori)

    return {"order": shared_paths, "positions": positions, "orientations": orientations, "num_envs": num_envs}


############################
## Example

if __name__ == "__main__":

    class ExampleSceneDataBackend(SceneDataBackend):
        def __init__(self):
            self.__transforms = SceneDataFormat.Transform()
            self.__transforms.transforms = wp.array(np.hstack([np.arange(10).reshape(10, 1)] * 7), dtype=wp.transformf)

        @property
        def transforms(self) -> SceneDataFormat.Transform:
            return self.__transforms

        @property
        def transform_count(self) -> int:
            return self.__transforms.transforms.shape[0]

        @property
        def transform_paths(self):
            return [
                "/world/shape_01",
                "/world/shape_02",
                "/world/shape_03",
                "/world/shape_04",
                "/world/shape_05",
                "/world/shape_06",
                "/world/shape_07",
                "/world/shape_08",
                "/world/shape_09",
                "/world/shape_10",
            ]

    sim = ExampleSceneDataBackend()
    sdp = SceneDataProvider(sim)

    output_data = SceneDataFormat.Vec3_Matrix33()
    output_data.positions = wp.empty(sdp.transform_count, dtype=wp.vec3f)
    output_data.orientations = wp.empty(sdp.transform_count, dtype=wp.mat33f)

    print(sim.transforms.transforms)
    mapping = sdp.create_mapping(
        [
            "/world/shape_02",
            "/world/shape_01",
            "/world/shape_03",
            "/world/shape_04",
            "/world/shape_05",
            None,
            None,
            "/world/shape_10",
            None,
            None,
        ]
    )
    print(mapping)
    if sdp.get_transforms(output_data, mapping):
        print(output_data.positions)
    else:
        print("Failed to get transforms!")

    wp.synchronize()
