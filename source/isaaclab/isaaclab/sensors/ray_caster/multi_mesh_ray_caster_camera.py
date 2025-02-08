# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.sensors.camera import CameraData
from omni.isaac.lab.utils.warp import raycast_dynamic_meshes

from ..utils import compute_world_poses
from .multi_mesh_ray_caster import MultiMeshRayCaster
from .ray_caster_camera import RayCasterCamera

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg


class MultiMeshRayCasterCamera(RayCasterCamera, MultiMeshRayCaster):
    """A ray-casting camera sensor.

    The ray-caster camera uses a set of rays to get the distances to meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor has the same interface as the
    :class:`omni.isaac.lab.sensors.Camera` that implements the camera class through USD camera prims.
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
        self._data = CameraData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(MultiMeshRayCasterCamera.meshes)}\n"
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

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # increment frame count
        self._frame[env_ids] += 1

        # compute poses from current view
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        # update the data
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        if self.cfg.track_mesh_transforms:
            # Update the mesh positions and rotations
            mesh_idx = 0
            for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
                # update position of the target meshes
                pos_w, ori_w = compute_world_poses(view, None)
                pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
                ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

                count = view.count
                if not target_cfg.is_global:
                    count = count // self._num_envs
                    pos_w = pos_w.view(self._num_envs, count, 3)
                    ori_w = ori_w.view(self._num_envs, count, 4)

                self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
                self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
                mesh_idx += count

        # ray cast and store the hits
        self.ray_hits_w, ray_depth, ray_normal = raycast_dynamic_meshes(
            ray_starts_w,
            ray_directions_w,
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=1e6,
            mesh_positions_w=self._mesh_positions_w[env_ids] if self.cfg.track_mesh_transforms else None,
            mesh_orientations_w=self._mesh_orientations_w[env_ids] if self.cfg.track_mesh_transforms else None,
            return_distance=any(
                [name in self.cfg.data_types for name in ["distance_to_image_plane", "distance_to_camera"]]
            ),
            return_normal="normals" in self.cfg.data_types,
        )[:3]
        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (z-axis of camera frame)
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(quat_w).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * ray_directions_w),
                )
            )[:, :, 0]
            # apply the maximum distance after the transformation
            distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )
        if "distance_to_camera" in self.cfg.data_types:
            self._data.output["distance_to_camera"][env_ids] = torch.clip(
                ray_depth.view(-1, *self.image_shape, 1), max=self.cfg.max_distance
            )
        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)
