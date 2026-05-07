# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils.warp import kernels as warp_kernels

from .base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from .base_ray_caster_camera import BaseRayCasterCamera
from .kernels import (
    CAMERA_RAYCAST_MAX_DIST,
    compute_distance_to_image_plane_masked_kernel,
    fill_float2d_masked_kernel,
    fill_vec3_inf_kernel,
)
from .multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg


class BaseMultiMeshRayCasterCamera(BaseRayCasterCamera, BaseMultiMeshRayCaster):
    """Backend-agnostic multi-mesh ray-casting camera sensor.

    The ray-caster camera uses a set of rays to get the distances to meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor has the same interface as the
    :class:`isaaclab.sensors.Camera` that implements the camera class through USD camera prims.
    However, this class provides a faster image generation. The sensor converts meshes from the list of
    primitive paths provided in the configuration to Warp meshes. The camera then ray-casts against these
    Warp meshes only.

    Currently, only the following annotators are supported:

    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    """

    cfg: MultiMeshRayCasterCameraCfg
    """The configuration parameters."""

    def __init__(self, cfg: MultiMeshRayCasterCameraCfg):
        self._check_supported_data_types(cfg)
        # Skip BaseRayCasterCamera.__init__ — the camera-data swap and check are done above /
        # below; BaseMultiMeshRayCaster's __init__ handles the multi-mesh init.
        BaseMultiMeshRayCaster.__init__(self, cfg)
        self._data = MultiMeshRayCasterCameraData()

    def __str__(self) -> str:
        return (
            f"Multi-Mesh Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tbackend              : {self.__backend_name__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(BaseMultiMeshRayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._num_envs}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._num_envs}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Implementation.
    """

    def _initialize_warp_meshes(self):
        BaseMultiMeshRayCaster._initialize_warp_meshes(self)

    def _create_buffers(self):
        super()._create_buffers()
        self._data.image_mesh_ids = torch.zeros(
            self._num_envs, *self.image_shape, 1, device=self.device, dtype=torch.int16
        )

    def _initialize_rays_impl(self):
        # NOTE: This method intentionally does NOT call super()._initialize_rays_impl() through the MRO
        # chain. The intermediate classes (RayCasterCamera, MultiMeshRayCaster) use different internal
        # buffer names and orderings that are incompatible with the camera's full init path:
        #   - RayCasterCamera creates single-mesh ray buffers (_ray_distance, _ray_normal_w, etc.)
        #   - MultiMeshRayCaster creates _ray_distance_w / _ray_mesh_id_w for multi-mesh use
        # The camera replaces all of these with its own camera-named equivalents below.
        # If either parent class gains new shared buffers, they must be added here explicitly.

        # Camera-specific bookkeeping buffers
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Build camera output buffers (intrinsics, image data, etc.)
        self._create_buffers()
        self._compute_intrinsic_matrices()

        # Compute local ray starts/directions from the camera pattern (torch, init-time only)
        ray_starts_local, ray_directions_local = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        self.num_rays = ray_directions_local.shape[1]

        # Store local (sensor-frame) ray arrays as torch tensors for per-env camera-convention rotation
        self.ray_starts = ray_starts_local
        self.ray_directions = ray_directions_local

        # Camera-frame offset: convert from cfg convention to world convention
        quat_offset = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention,
            target="world",
        )
        self._offset_quat = quat_offset.repeat(self._num_envs, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._num_envs, 1)

        # Camera pose buffers (torch, part of CameraData)
        self._data.pos_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._data.quat_w_world = torch.zeros(self._num_envs, 4, device=self._device)
        # Warp-backed camera orientation buffer for warp kernel calls;
        # updated from self._data.quat_w_world in _update_buffers_impl.
        self._quat_w_wp = wp.zeros(self._num_envs, dtype=wp.quatf, device=self._device)
        self._quat_w_wp_torch = wp.to_torch(self._quat_w_wp)

        # Warp buffer for distance_to_image_plane output (if requested)
        if "distance_to_image_plane" in self.cfg.data_types:
            self._distance_to_image_plane_wp = wp.zeros(
                (self._num_envs, self.num_rays), dtype=wp.float32, device=self._device
            )

        # World-frame ray buffers: allocate as warp arrays first, then create zero-copy torch views.
        # Keeping warp arrays as primary storage avoids lifetime issues when passing to kernels.
        self._ray_starts_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.vec3f, device=self._device)
        # Zero-copy torch views used for indexing and post-processing
        self._ray_starts_w_torch = wp.to_torch(self._ray_starts_w)
        self._ray_directions_w_torch = wp.to_torch(self._ray_directions_w)

        # Ray hit positions as a warp array; expose a torch view for debug visualisation
        self._ray_hits_w_cam = wp.zeros((self._num_envs, self.num_rays), dtype=wp.vec3f, device=self._device)
        self.ray_hits_w = wp.to_torch(self._ray_hits_w_cam)

        # Per-ray closest-hit distance for atomic_min across meshes
        self._ray_distance_cam_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.float32, device=self._device)

        # Optional normal buffer (always allocated; filled only when "normals" is requested)
        self._ray_normal_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Mesh-id buffers from MultiMeshRayCaster._initialize_rays_impl
        if self.cfg.update_mesh_ids:
            self._ray_mesh_id_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.int16, device=self._device)
            self._data.ray_mesh_ids = wp.to_torch(self._ray_mesh_id_w).unsqueeze(-1)
        else:
            self._ray_mesh_id_w = wp.empty((1, 1), dtype=wp.int16, device=self._device)

        # Dummy face-id buffer (not used by camera but required by kernel signature)
        self._ray_face_id_w = wp.empty((1, 1), dtype=wp.int32, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return

        # Camera world pose = sensor_xform_pose * cfg.offset, in torch — multi-mesh's
        # per-env camera-frame offsets don't fit update_ray_caster_kernel's single-offset
        # ABI. The sensor xform pose comes from the backend body tracker.
        pos_w, quat_w = self._get_sensor_world_poses(env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        self._quat_w_wp_torch[env_ids] = quat_w

        # Rotate local ray starts/directions into world frame using full camera orientation.
        quat_repeat = quat_w.repeat(1, self.num_rays).reshape(-1, 4)
        ray_starts_w = math_utils.quat_apply(quat_repeat, self.ray_starts[env_ids].reshape(-1, 3)).reshape(
            len(env_ids), self.num_rays, 3
        ) + pos_w.unsqueeze(1)
        ray_dirs_w = math_utils.quat_apply(quat_repeat, self.ray_directions[env_ids].reshape(-1, 3)).reshape(
            len(env_ids), self.num_rays, 3
        )
        self._ray_starts_w_torch[env_ids] = ray_starts_w
        self._ray_directions_w_torch[env_ids] = ray_dirs_w

        self._frame[env_ids] += 1
        self._update_target_mesh_transforms()

        n_meshes = self._mesh_ids_wp.shape[1]
        return_normal = "normals" in self.cfg.data_types

        # Fill ray hit and distance buffers with inf for masked environments
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._ray_hits_w_cam],
            device=self._device,
        )
        wp.launch(
            fill_float2d_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._ray_distance_cam_w],
            device=self._device,
        )
        if return_normal:
            wp.launch(
                fill_vec3_inf_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, float("inf"), self._ray_normal_w],
                device=self._device,
            )

        # Ray-cast against all meshes; closest hit wins via atomic_min on ray_distance
        wp.launch(
            warp_kernels.raycast_dynamic_meshes_kernel,
            dim=(n_meshes, self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._ray_hits_w_cam,
                self._ray_distance_cam_w,
                self._ray_normal_w,
                self._ray_face_id_w,
                self._ray_mesh_id_w,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(CAMERA_RAYCAST_MAX_DIST),
                int(return_normal),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )

        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                compute_distance_to_image_plane_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._quat_w_wp, self._ray_distance_cam_w, self._ray_directions_w],
                outputs=[self._distance_to_image_plane_wp],
                device=self._device,
            )
            # Apply depth clipping on the intermediate buffer (leaves _ray_distance_cam_w unmodified)
            self._apply_depth_clipping(env_mask, self._distance_to_image_plane_wp)
            d2ip_torch = wp.to_torch(self._distance_to_image_plane_wp)
            self._data.output["distance_to_image_plane"][env_ids] = d2ip_torch[env_ids].view(-1, *self.image_shape, 1)

        if "distance_to_camera" in self.cfg.data_types:
            # d2ip (if requested) was computed before this block so _ray_distance_cam_w is still unclipped.
            self._apply_depth_clipping(env_mask, self._ray_distance_cam_w)
            ray_dist_torch = wp.to_torch(self._ray_distance_cam_w)
            self._data.output["distance_to_camera"][env_ids] = ray_dist_torch[env_ids].view(-1, *self.image_shape, 1)

        if return_normal:
            ray_normal_torch = wp.to_torch(self._ray_normal_w)
            self._data.output["normals"][env_ids] = ray_normal_torch[env_ids].view(-1, *self.image_shape, 3)

        if self.cfg.update_mesh_ids:
            self._data.image_mesh_ids[env_ids] = wp.to_torch(self._ray_mesh_id_w)[env_ids].view(
                -1, *self.image_shape, 1
            )
