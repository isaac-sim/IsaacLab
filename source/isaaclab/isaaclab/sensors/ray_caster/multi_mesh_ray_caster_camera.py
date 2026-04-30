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

from .kernels import (
    CAMERA_RAYCAST_MAX_DIST,
    compute_distance_to_image_plane_masked_kernel,
    fill_float2d_masked_kernel,
    fill_vec3_inf_kernel,
)
from .multi_mesh_ray_caster import MultiMeshRayCaster
from .multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData
from .ray_caster_camera import RayCasterCamera

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg


class MultiMeshRayCasterCamera(RayCasterCamera, MultiMeshRayCaster):
    """A multi-mesh ray-casting camera sensor.

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
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the ray-caster camera.
        """
        self._check_supported_data_types(cfg)
        # initialize base class
        MultiMeshRayCaster.__init__(self, cfg)
        # create empty variables for storing output data
        self._data = MultiMeshRayCasterCameraData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Multi-Mesh Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(MultiMeshRayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Implementation.
    """

    def _initialize_warp_meshes(self):
        MultiMeshRayCaster._initialize_warp_meshes(self)

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
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

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
        self._offset_quat = quat_offset.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        # Camera pose buffers (torch, part of CameraData)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w_world = torch.zeros(self._view.count, 4, device=self._device)
        # Warp-backed camera orientation buffer for warp kernel calls;
        # updated from self._data.quat_w_world in _update_ray_infos.
        self._quat_w_wp = wp.zeros(self._view.count, dtype=wp.quatf, device=self._device)
        self._quat_w_wp_torch = wp.to_torch(self._quat_w_wp)

        # Warp buffer for distance_to_image_plane output (if requested)
        if "distance_to_image_plane" in self.cfg.data_types:
            self._distance_to_image_plane_wp = wp.zeros(
                (self._view.count, self.num_rays), dtype=wp.float32, device=self._device
            )

        # World-frame ray buffers: allocate as warp arrays first, then create zero-copy torch views.
        # Keeping warp arrays as primary storage avoids lifetime issues when passing to kernels.
        self._ray_starts_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        # Zero-copy torch views used for indexing and post-processing
        self._ray_starts_w_torch = wp.to_torch(self._ray_starts_w)
        self._ray_directions_w_torch = wp.to_torch(self._ray_directions_w)

        # Ray hit positions as a warp array; expose a torch view for debug visualisation
        self._ray_hits_w_cam = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self.ray_hits_w = wp.to_torch(self._ray_hits_w_cam)

        # Per-ray closest-hit distance for atomic_min across meshes
        self._ray_distance_cam_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.float32, device=self._device)

        # Optional normal buffer (always allocated; filled only when "normals" is requested)
        self._ray_normal_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Mesh-id buffers from MultiMeshRayCaster._initialize_rays_impl
        if self.cfg.update_mesh_ids:
            self._ray_mesh_id_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.int16, device=self._device)
            self._data.ray_mesh_ids = wp.to_torch(self._ray_mesh_id_w).unsqueeze(-1)
        else:
            self._ray_mesh_id_w = wp.empty((1, 1), dtype=wp.int16, device=self._device)

        # Dummy face-id buffer (not used by camera but required by kernel signature)
        self._ray_face_id_w = wp.empty((1, 1), dtype=wp.int32, device=self._device)

    def _update_ray_infos(self, env_mask: wp.array):
        """Updates camera poses and world-frame ray buffers for masked environments.

        Args:
            env_mask: Boolean mask selecting which environments to update. Shape is (num_envs,).
        """
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return

        # Compute camera world poses by composing view pose with sensor offset
        indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32)
        pos_w, quat_w = self._view.get_world_poses(indices)
        pos_w, quat_w = pos_w.torch, quat_w.torch
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        # Store camera pose in CameraData (torch tensors) and warp-backed orientation buffer
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        self._quat_w_wp_torch[env_ids] = quat_w

        # Rotate local ray starts and directions into world frame using full camera orientation
        quat_w_repeated = quat_w.repeat(1, self.num_rays).reshape(-1, 4)
        ray_starts_local = self.ray_starts[env_ids].reshape(-1, 3)
        ray_dirs_local = self.ray_directions[env_ids].reshape(-1, 3)

        ray_starts_world = math_utils.quat_apply(quat_w_repeated, ray_starts_local).reshape(
            len(env_ids), self.num_rays, 3
        )
        ray_starts_world += pos_w.unsqueeze(1)
        ray_dirs_world = math_utils.quat_apply(quat_w_repeated, ray_dirs_local).reshape(len(env_ids), self.num_rays, 3)

        # Write back into the warp-backed buffers via zero-copy torch views
        self._ray_starts_w_torch[env_ids] = ray_starts_world
        self._ray_directions_w_torch[env_ids] = ray_dirs_world

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return

        self._update_ray_infos(env_mask)

        # Increment frame count for updated environments
        self._frame[env_ids] += 1

        self._update_mesh_transforms()

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
