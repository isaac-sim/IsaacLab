# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.utils.warp.kernels import raycast_mesh_masked_kernel

from ..sensor_base import SensorBase
from .kernels import apply_z_drift_kernel, fill_vec3_inf_kernel, update_ray_caster_kernel
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg

logger = logging.getLogger(__name__)


class BaseRayCaster(SensorBase):
    """Backend-agnostic ray-casting sensor.

    Holds the warp mesh cache, ray-pattern + drift, and raycast-kernel
    orchestration. Backends supply only the per-step body-pose source via
    :meth:`_get_sensor_transforms_wp`; all init/cleanup happens in standard
    sensor-lifecycle overrides (``__init__``, ``_initialize_impl``,
    ``_invalidate_initialize_callback``). Single static mesh only — see
    :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster` for multi-mesh
    / dynamic tracking.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the ray-caster sensor."""

    meshes: ClassVar[dict[tuple[str, str], wp.Mesh]] = {}
    """Shared warp-mesh cache, keyed ``(prim_path, device)`` so a CPU mesh isn't reused on CUDA."""
    _instance_count: ClassVar[int] = 0
    """Live RayCaster instance count — drives class-var lifecycle (cleared on the last drop)."""

    def __init__(self, cfg: RayCasterCfg):
        BaseRayCaster._instance_count += 1
        super().__init__(cfg)
        self._resolve_and_spawn("raycaster")
        self._data = RayCasterData()

    def __str__(self) -> str:
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tbackend              : {self.__backend_name__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(BaseRayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._num_envs}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._num_envs}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._num_envs

    @property
    def data(self) -> RayCasterData:
        self._update_outdated_buffers()
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        super().reset(env_ids, env_mask)
        # Resolve env selector for torch indexing into drift buffers.
        if env_ids is not None:
            num_envs_ids = len(env_ids)
        elif env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
            num_envs_ids = len(env_ids)
        else:
            env_ids, num_envs_ids = slice(None), self._num_envs
        # Resample sensor-pose drift and ray-cast drift from configured ranges.
        self.drift[env_ids] = torch.empty(num_envs_ids, 3, device=self.device).uniform_(*self.cfg.drift_range)
        ranges = torch.tensor(
            [self.cfg.ray_cast_drift_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=self.device
        )
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        # Identity offsets — kernel does ``sensor_pose = view_pose * offset``. Camera
        # subclass bakes ``cfg.offset`` during ``_initialize_rays_impl``; backend
        # subclasses extend this method (after super) for their per-step pose source.
        self._offset_pos_wp = wp.zeros(self._num_envs, dtype=wp.vec3f, device=self._device)
        identity_quat = torch.zeros(self._num_envs, 4, device=self._device)
        identity_quat[:, 3] = 1.0
        self._offset_quat_contiguous = identity_quat.contiguous()
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

        alignment_map = {"world": 0, "yaw": 1, "base": 2}
        if self.cfg.ray_alignment not in alignment_map:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")
        self._alignment_mode = alignment_map[self.cfg.ray_alignment]

        self._initialize_warp_meshes()
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            mesh_key = (mesh_prim_path, self._device)
            if mesh_key in BaseRayCaster.meshes:
                continue

            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            if mesh_prim is None:
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                world_transform: Gf.Matrix4d = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                transform_matrix = np.array(world_transform).T
                points = np.matmul(points, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self._device)
                logger.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self._device)
                logger.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            BaseRayCaster.meshes[mesh_key] = wp_mesh

        if all((p, self._device) not in BaseRayCaster.meshes for p in self.cfg.mesh_prim_paths):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):
        # Local ray pattern with cfg.offset baked in.
        ray_starts, ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(ray_directions)
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        ray_directions = math_utils.quat_apply(offset_quat.repeat(len(ray_directions), 1), ray_directions)
        ray_starts = (ray_starts + offset_pos).repeat(self._num_envs, 1, 1)
        ray_directions = ray_directions.repeat(self._num_envs, 1, 1)

        # Warp owns memory; paired torch views allow indexed reset/access.
        self._ray_starts_local = wp.from_torch(ray_starts.contiguous(), dtype=wp.vec3f)
        self._ray_directions_local = wp.from_torch(ray_directions.contiguous(), dtype=wp.vec3f)
        self.ray_starts = wp.to_torch(self._ray_starts_local)
        self.ray_directions = wp.to_torch(self._ray_directions_local)

        self._drift = wp.zeros(self._num_envs, dtype=wp.vec3f, device=self._device)
        self._ray_cast_drift = wp.zeros(self._num_envs, dtype=wp.vec3f, device=self._device)
        self.drift = wp.to_torch(self._drift)
        self.ray_cast_drift = wp.to_torch(self._ray_cast_drift)

        self._ray_starts_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_starts_w_torch = wp.to_torch(self._ray_starts_w)
        self._ray_directions_w_torch = wp.to_torch(self._ray_directions_w)

        self._data.create_buffers(self._num_envs, self.num_rays, self._device)

        # Placeholders for the merged kernel signature — both flags are always 0 so the
        # kernel never writes here. If either is flipped on, resize to (num_envs, num_rays).
        self._dummy_ray_distance = wp.empty((1, 1), dtype=wp.float32, device=self._device)
        self._dummy_ray_normal = wp.empty((1, 1), dtype=wp.vec3f, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        # Compose backend-tracked sensor pose with local ray pattern → world rays + sensor pose.
        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._get_sensor_transforms_wp(),
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                self._alignment_mode,
            ],
            outputs=[self._data._pos_w, self._data._quat_w, self._ray_starts_w, self._ray_directions_w],
            device=self._device,
        )
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._data._ray_hits_w],
            device=self._device,
        )
        wp.launch(
            raycast_mesh_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                BaseRayCaster.meshes[(self.cfg.mesh_prim_paths[0], self._device)].id,
                env_mask,
                self._ray_starts_w,
                self._ray_directions_w,
                float(self.cfg.max_distance),
                int(False),  # return_distance — unused
                int(False),  # return_normal — unused
                self._data._ray_hits_w,
                self._dummy_ray_distance,
                self._dummy_ray_normal,
            ],
            device=self._device,
        )
        wp.launch(
            apply_z_drift_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, self._ray_cast_drift, self._data._ray_hits_w],
            device=self._device,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.ray_visualizer.set_visibility(True)
        elif hasattr(self, "ray_visualizer"):
            self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self._data._ray_hits_w is None:
            return
        viz_points = wp.to_torch(self._data._ray_hits_w).reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points.shape[0] == 0:
            return
        self.ray_visualizer.visualize(viz_points)

    """
    Backend-specific hooks.
    """

    @abstractmethod
    def _get_sensor_transforms_wp(self) -> wp.array:
        """Per-step sensor-body world transforms — ``wp.array(dtype=wp.transformf)`` of
        shape ``(num_envs,)`` packed ``(tx, ty, tz, qx, qy, qz, qw)``."""
        raise NotImplementedError

    def _get_sensor_world_poses(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(pos, quat)`` torch views of the per-step sensor Xform pose.

        Default derived from :meth:`_get_sensor_transforms_wp`; backends override only
        for a faster path. Used by :meth:`BaseRayCasterCamera.reset` /
        :meth:`BaseRayCasterCamera.set_world_poses` for indexed access.
        """
        transforms = wp.to_torch(self._get_sensor_transforms_wp()).view(-1, 7)
        pos, quat = transforms[:, :3], transforms[:, 3:]
        return (pos[env_ids], quat[env_ids]) if env_ids is not None else (pos, quat)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Zero public data on STOP. Backends override to drop tracker state, call super()."""
        super()._invalidate_initialize_callback(event)
        self._zero_data_buffers()

    def _zero_data_buffers(self) -> None:
        """Zero :class:`RayCasterData`. Overridden by :class:`BaseRayCasterCamera` —
        :class:`~isaaclab.sensors.camera.CameraData` has a different field layout."""
        self._data._pos_w.zero_()
        self._data._quat_w.zero_()
        self._data._ray_hits_w.zero_()

    def __del__(self):
        BaseRayCaster._instance_count -= 1
        if BaseRayCaster._instance_count == 0:
            BaseRayCaster.meshes.clear()
