# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.views import FrameView
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.utils.warp.kernels import raycast_mesh_masked_kernel

from ..sensor_base import SensorBase
from .kernels import (
    apply_z_drift_kernel,
    fill_vec3_inf_kernel,
    update_ray_caster_kernel,
)
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg

logger = logging.getLogger(__name__)


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the :attr:`meshes` dictionary. The ray-caster then ray-casts
    against these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    meshes: ClassVar[dict[tuple[str, str], wp.Mesh]] = {}
    """A dictionary to store warp meshes for raycasting, shared across all instances.

    The keys are ``(prim_path, device)`` tuples and values are the corresponding warp Mesh objects.
    Including the device in the key prevents a mesh created on one device (e.g. CPU) from being
    reused by a kernel running on a different device (e.g. CUDA)."""
    _instance_count: ClassVar[int] = 0
    """A counter to track the number of RayCaster instances, used to manage class variable lifecycle."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        RayCaster._instance_count += 1
        super().__init__(cfg)
        # Resolve physics-body paths and spawn the sensor Xform child if needed.
        self._resolve_and_spawn("raycaster")
        self._data = RayCasterData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(RayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # reset the timers and counters
        super().reset(env_ids, env_mask)
        # resolve to indices for torch indexing
        if env_ids is not None:
            num_envs_ids = len(env_ids)
        elif env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
            num_envs_ids = len(env_ids)
        else:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        # resample drift (uses torch views for indexing)
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)
        # resample the ray cast drift
        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # Build a FrameView over the sensor prim paths. The FrameView tracks the spawned
        # (non-physics) Xform directly, so no physics-body redirect or offset resolution
        # is needed at runtime — the world pose returned already includes any offset
        # baked into the prim's local transform.
        self._view = FrameView(self.cfg.prim_path, device=self._device, stage=self.stage)

        # Per-env identity offsets (kept for kernel ABI compatibility): the sensor frame is
        # already the FrameView's tracked prim, so no additional view-to-sensor offset applies.
        self._offset_pos_wp = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        identity_quat = torch.zeros(self._view.count, 4, device=self._device)
        identity_quat[:, 3] = 1.0
        self._offset_quat_contiguous = identity_quat.contiguous()
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

        # Resolve alignment mode to integer constant for kernel dispatch
        alignment_map = {"world": 0, "yaw": 1, "base": 2}
        if self.cfg.ray_alignment not in alignment_map:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")
        self._alignment_mode = alignment_map[self.cfg.ray_alignment]

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        # check number of mesh prims provided
        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )

        # read prims to ray-cast
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            mesh_key = (mesh_prim_path, self._device)
            if mesh_key in RayCaster.meshes:
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
                xformable = UsdGeom.Xformable(mesh_prim)
                world_transform: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                transform_matrix = np.array(world_transform).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self._device)
                logger.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self._device)
                logger.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            RayCaster.meshes[mesh_key] = wp_mesh

        if all((mesh_prim_path, self._device) not in RayCaster.meshes for mesh_prim_path in self.cfg.mesh_prim_paths):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):
        # Compute ray starts and directions from pattern (torch, init-time only)
        ray_starts_torch, ray_directions_torch = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(ray_directions_torch)

        # Apply sensor offset rotation/position to local ray pattern
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        ray_directions_torch = math_utils.quat_apply(
            offset_quat.repeat(len(ray_directions_torch), 1), ray_directions_torch
        )
        ray_starts_torch += offset_pos

        # Repeat for each environment
        ray_starts_torch = ray_starts_torch.repeat(self._view.count, 1, 1)
        ray_directions_torch = ray_directions_torch.repeat(self._view.count, 1, 1)

        # Create warp arrays from the init-time torch data
        # The warp arrays own the memory; torch views provide backward-compat indexing
        self._ray_starts_local = wp.from_torch(ray_starts_torch.contiguous(), dtype=wp.vec3f)
        self._ray_directions_local = wp.from_torch(ray_directions_torch.contiguous(), dtype=wp.vec3f)

        # Torch views (same attribute names as before for subclass compatibility)
        self.ray_starts = wp.to_torch(self._ray_starts_local)
        self.ray_directions = wp.to_torch(self._ray_directions_local)

        # Drift buffers (warp-owned, torch views for reset indexing)
        self._drift = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        self._ray_cast_drift = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        self.drift = wp.to_torch(self._drift)
        self.ray_cast_drift = wp.to_torch(self._ray_cast_drift)

        # World-frame ray buffers
        self._ray_starts_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Torch views for subclass compatibility
        self._ray_starts_w_torch = wp.to_torch(self._ray_starts_w)
        self._ray_directions_w_torch = wp.to_torch(self._ray_directions_w)

        # Data buffers
        self._data.create_buffers(self._view.count, self.num_rays, self._device)

        # Dummy distance/normal buffers required by the merged raycast_mesh_masked_kernel signature.
        # Sized (1, 1) even though the kernel is launched at (num_envs, num_rays): the kernel only
        # writes to these buffers when return_distance==1 or return_normal==1 respectively, and
        # RayCaster always passes 0 for both flags. If those flags are ever enabled here, these
        # buffers must be resized to (num_envs, num_rays) to avoid an out-of-bounds write.
        self._dummy_ray_distance = wp.empty((1, 1), dtype=wp.float32, device=self._device)
        self._dummy_ray_normal = wp.empty((1, 1), dtype=wp.vec3f, device=self._device)

    def _get_view_transforms_wp(self) -> wp.array:
        """Get world transforms from the frame view as a warp array of ``wp.transformf``.

        Returns:
            Warp array of ``wp.transformf`` with shape ``(num_envs,)``. Layout is
            ``(tx, ty, tz, qx, qy, qz, qw)`` per element, matching the quaternion
            convention returned by :class:`~isaaclab.sim.views.FrameView`.
        """
        pos_wp, quat_wp = self._view.get_world_poses()
        pos_torch = wp.to_torch(pos_wp).reshape(-1, 3)
        quat_torch = wp.to_torch(quat_wp).reshape(-1, 4)
        poses = torch.cat([pos_torch, quat_torch], dim=-1).contiguous()
        return wp.from_torch(poses).view(wp.transformf)

    def _update_ray_infos(self, env_mask: wp.array):
        """Updates sensor poses and ray world-frame buffers via a single warp kernel."""
        transforms = self._get_view_transforms_wp()

        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                transforms,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                self._alignment_mode,
            ],
            outputs=[
                self._data._pos_w,
                self._data._quat_w,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)

        # Fill ray hits with inf before raycasting
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._data._ray_hits_w],
            device=self._device,
        )

        # Ray-cast against the mesh
        wp.launch(
            raycast_mesh_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                RayCaster.meshes[(self.cfg.mesh_prim_paths[0], self._device)].id,
                env_mask,
                self._ray_starts_w,
                self._ray_directions_w,
                float(self.cfg.max_distance),
                int(False),  # return_distance: not needed by RayCaster
                int(False),  # return_normal: not needed by RayCaster
                self._data._ray_hits_w,
                self._dummy_ray_distance,
                self._dummy_ray_normal,
            ],
            device=self._device,
        )

        # Apply vertical drift to ray hits
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
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self._data._ray_hits_w is None:
            return
        ray_hits_torch = wp.to_torch(self._data._ray_hits_w)
        # remove possible inf values
        viz_points = ray_hits_torch.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]

        # if no points to visualize, skip
        if viz_points.shape[0] == 0:
            return

        self.ray_visualizer.visualize(viz_points)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        self._view = None

    def __del__(self):
        RayCaster._instance_count -= 1
        if RayCaster._instance_count == 0:
            RayCaster.meshes.clear()
