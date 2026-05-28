# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp import kernels as warp_kernels

from . import kernels as ray_caster_kernels
from .base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from .base_ray_caster_camera import BaseRayCasterCamera
from .multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg


class BaseMultiMeshRayCasterCamera(BaseRayCasterCamera, BaseMultiMeshRayCaster):
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
        BaseMultiMeshRayCaster.__init__(self, cfg)
        # create empty variables for storing output data
        self._data = MultiMeshRayCasterCameraData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Multi-Mesh Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    def _initialize_warp_meshes(self):
        # The camera MRO would pick the single-mesh camera path; use the multi-mesh setup.
        BaseMultiMeshRayCaster._initialize_warp_meshes(self)

    def _create_buffers(self):
        super()._create_buffers()
        self._data.image_mesh_ids = ProxyArray(
            wp.zeros((self._num_envs, *self.image_shape, 1), dtype=wp.int16, device=self.device)
        )

    def _initialize_rays_impl(self):
        # NOTE: This method intentionally does NOT call super()._initialize_rays_impl() through the MRO
        # chain. The intermediate classes (RayCasterCamera, MultiMeshRayCaster) use different internal
        # buffer names and orderings that are incompatible with the camera's full init path:
        #   - RayCasterCamera creates single-mesh ray buffers (_ray_distance, _ray_normal_w, etc.)
        #   - MultiMeshRayCaster creates _ray_distance_wp / _ray_mesh_id_wp for multi-mesh use
        # The camera replaces all of these with its own camera-named equivalents below.
        # If either parent class gains new shared buffers, they must be added here explicitly.

        # Camera-specific bookkeeping buffers
        self._frame_wp = wp.zeros(self._view.count, dtype=wp.int64, device=self._device)
        self._frame = wp.to_torch(self._frame_wp)

        # Build camera output buffers (intrinsics, image data, etc.)
        self._create_buffers()
        self._compute_intrinsic_matrices()

        # Compute local ray starts/directions from the camera pattern (torch, init-time only)
        ray_starts_local, ray_directions_local = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices.torch, self._device
        )
        self.num_rays = ray_directions_local.shape[1]

        # Store local (sensor-frame) ray arrays as ProxyArrays for Warp kernel dispatch.
        self.ray_starts = ProxyArray(wp.from_torch(ray_starts_local.contiguous(), dtype=wp.vec3f))
        self.ray_directions = ProxyArray(wp.from_torch(ray_directions_local.contiguous(), dtype=wp.vec3f))

        # Camera-frame offset and drift buffers.
        self._create_offset_buffers(self._view.count)

        # World-frame ray buffers.
        self._ray_starts_w = wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Ray hit positions as a warp array; expose a ProxyArray for debug visualisation.
        self.ray_hits_w = ProxyArray(wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device))

        # Per-ray closest-hit distance for atomic_min across meshes
        self._ray_distance_cam_wp = wp.empty((self._view.count, self.num_rays), dtype=wp.float32, device=self._device)

        # Optional normal buffer (always allocated; filled only when "normals" is requested)
        self._ray_normal_w = wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Mesh-id buffers from MultiMeshRayCaster._initialize_rays_impl
        if self.cfg.update_mesh_ids:
            self._ray_mesh_id_wp = wp.zeros((self._view.count, self.num_rays), dtype=wp.int16, device=self._device)
        else:
            self._ray_mesh_id_wp = wp.empty((1, 1), dtype=wp.int16, device=self._device)

        # Dummy face-id buffer (not used by camera but required by kernel signature)
        self._ray_face_id_wp = wp.empty((1, 1), dtype=wp.int32, device=self._device)

    def _update_ray_infos(self, env_mask: wp.array):
        """Updates camera poses and world-frame ray buffers for masked environments.

        Args:
            env_mask: Boolean mask selecting which environments to update. Shape is (num_envs,).
        """
        transforms = self._get_view_transforms_wp()
        wp.launch(
            ray_caster_kernels.update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                transforms,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self.drift.warp,
                self.ray_cast_drift.warp,
                self.ray_starts.warp,
                self.ray_directions.warp,
                int(ray_caster_kernels.ALIGNMENT_BASE),
            ],
            outputs=[
                self._data.pos_w.warp,
                self._data.quat_w_world.warp,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)

        # Increment frame count for updated environments
        self._update_frame(env_mask, frame_op=1)

        self._update_mesh_transforms()

        return_normal = "normals" in self.cfg.data_types

        # Fill ray hit, distance, and optional normal buffers with inf for masked environments.
        wp.launch(
            ray_caster_kernels.fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, return_normal],
            outputs=[self.ray_hits_w.warp, self._ray_distance_cam_wp, self._ray_normal_w],
            device=self._device,
        )

        n_meshes = self._mesh_ids_wp.shape[1]

        # Ray-cast against all meshes; closest hit wins via atomic_min on ray_distance.
        wp.launch(
            warp_kernels.raycast_dynamic_meshes_kernel,
            dim=(n_meshes, self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self.ray_hits_w.warp,
                self._ray_distance_cam_wp,
                self._ray_normal_w,
                self._ray_face_id_wp,
                self._ray_mesh_id_wp,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(ray_caster_kernels.CAMERA_RAYCAST_MAX_DIST),
                int(return_normal),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )

        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._data.quat_w_world.warp,
                    self._ray_distance_cam_wp,
                    self._ray_directions_w,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[
                    self._data.output["distance_to_image_plane"].warp,
                ],
                device=self._device,
            )

        if "distance_to_camera" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.copy_float2d_to_image1_depth_clipped_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._ray_distance_cam_wp,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[
                    self._data.output["distance_to_camera"].warp,
                ],
                device=self._device,
            )

        if return_normal:
            wp.launch(
                ray_caster_kernels.copy_vec3_2d_to_image3_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_normal_w, int(self.image_shape[1]), self._data.output["normals"].warp],
                device=self._device,
            )

        if self.cfg.update_mesh_ids:
            wp.launch(
                ray_caster_kernels.copy_int16_2d_to_image1_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_mesh_id_wp, int(self.image_shape[1]), self._data.image_mesh_ids.warp],
                device=self._device,
            )
