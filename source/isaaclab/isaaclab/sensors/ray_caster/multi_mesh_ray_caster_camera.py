# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils.warp import raycast_dynamic_meshes

from .multi_mesh_ray_caster import MultiMeshRayCaster
from .multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData
from .ray_cast_utils import obtain_world_pose_from_view
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
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        self.num_rays = self.ray_directions.shape[1]
        # create buffer to store ray hits
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        self._data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self.device)

        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

    def _update_ray_infos(self, env_ids: Sequence[int]):
        """Updates the ray information buffers."""

        # compute poses from current view
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        # update the data
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        self._data.quat_w_ros[env_ids] = quat_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return
        self._update_ray_infos(env_ids)

        # increment frame count
        self._frame[env_ids] += 1

        # Update the mesh positions and rotations
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # update position of the target meshes
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:  # Mesh is not global, i.e. we have different meshes for each env
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        # ray cast and store the hits
        self.ray_hits_w[env_ids], ray_depth, ray_normal, _, ray_mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=any(
                [name in self.cfg.data_types for name in ["distance_to_image_plane", "distance_to_camera"]]
            ),
            return_normal="normals" in self.cfg.data_types,
            return_mesh_id=self.cfg.update_mesh_ids,
        )

        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (z-axis of camera frame)
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(self._data.quat_w_world[env_ids]).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * self._ray_directions_w[env_ids]),
                )
            )[:, :, 0]
            # apply the maximum distance after the transformation
            if self.cfg.depth_clipping_behavior == "max":
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth = torch.clip(ray_depth, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth[ray_depth > self.cfg.max_distance] = 0.0
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)

        if self.cfg.update_mesh_ids:
            self._data.image_mesh_ids[env_ids] = ray_mesh_ids.view(-1, *self.image_shape, 1)
