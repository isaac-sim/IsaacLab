# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sim.views import XformPrimView

from .occupancy_map_utils import OccupancyMap, intersect_occupancy_maps
from .transform_utils import transform_mul


class HasOccupancyMap:
    """An abstract base class for entities that have an associated occupancy map."""

    def get_occupancy_map(self) -> OccupancyMap:
        raise NotImplementedError


class HasPose2d:
    """An abstract base class for entities that have an associated 2D pose."""

    def get_pose_2d(self) -> torch.Tensor:
        """Get the 2D pose of the entity."""
        raise NotImplementedError

    def get_transform_2d(self):
        """Get the 2D transformation matrix of the entity."""

        pose = self.get_pose_2d()

        x = pose[..., 0]
        y = pose[..., 1]
        theta = pose[..., 2]
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)

        dims = tuple(list(pose.shape)[:-1] + [3, 3])
        transform = torch.zeros(dims)

        transform[..., 0, 0] = ctheta
        transform[..., 0, 1] = -stheta
        transform[..., 1, 0] = stheta
        transform[..., 1, 1] = ctheta
        transform[..., 0, 2] = x
        transform[..., 1, 2] = y
        transform[..., 2, 2] = 1.0

        return transform


class HasPose(HasPose2d):
    """An abstract base class for entities that have an associated 3D pose."""

    def get_pose(self):
        """Get the 3D pose of the entity."""
        raise NotImplementedError

    def get_pose_2d(self):
        """Get the 2D pose of the entity."""
        pose = self.get_pose()
        axis_angle = math_utils.axis_angle_from_quat(pose[..., 3:])

        yaw = axis_angle[..., 2:3]
        xy = pose[..., :2]

        pose_2d = torch.cat([xy, yaw], dim=-1)

        return pose_2d


class SceneBody(HasPose):
    """A helper class for working with rigid body objects in a scene."""

    def __init__(self, scene, entity_name: str, body_name: str):
        self.scene = scene
        self.entity_name = entity_name
        self.body_name = body_name

    def get_pose(self):
        """Get the 3D pose of the entity."""
        body_link_state_w = wp.to_torch(self.scene[self.entity_name].data.body_link_state_w)
        pose = body_link_state_w[
            :,
            self.scene[self.entity_name].data.body_names.index(self.body_name),
            :7,
        ]
        return pose


class SceneAsset(HasPose):
    """A helper class for working with assets in a scene."""

    def __init__(self, scene, entity_name: str):
        self.scene = scene
        self.entity_name = entity_name

    def _get_xform_view(self) -> XformPrimView:
        """Return the XformPrimView for this asset, refreshing it if prims were not yet cloned."""
        xform_prim = self.scene[self.entity_name]
        if xform_prim.count == 0:
            # The view was created before environment cloning; rebuild it now that prims exist.
            xform_prim = XformPrimView(xform_prim._prim_path, device=xform_prim.device)
            self.scene.extras[self.entity_name] = xform_prim
        return xform_prim

    def get_pose(self):
        """Get the 3D pose of the entity."""
        xform_prim = self._get_xform_view()
        position, orientation = xform_prim.get_world_poses()
        pose = torch.cat([position, orientation], dim=-1)
        return pose

    def set_pose(self, pose: torch.Tensor):
        """Set the 3D pose of the entity."""
        xform_prim = self._get_xform_view()
        position = pose[..., :3]
        orientation = pose[..., 3:]
        xform_prim.set_world_poses(position, orientation, None)


class RelativePose(HasPose):
    """A helper class for computing the pose of an entity given it's relative pose to a parent."""

    def __init__(self, relative_pose: torch.Tensor, parent: HasPose):
        self.relative_pose = relative_pose
        self.parent = parent

    def get_pose(self):
        """Get the 3D pose of the entity."""

        parent_pose = self.parent.get_pose()
        relative_pose = self.relative_pose.to(parent_pose.device)

        pose = transform_mul(parent_pose, relative_pose)

        return pose


class SceneFixture(SceneAsset, HasOccupancyMap):
    """A helper class for working with assets in a scene that have an associated occupancy map."""

    def __init__(self, scene, entity_name: str, local_occupancy_map: OccupancyMap):
        """Initialize a SceneFixture from a local occupancy map

        Args:
            scene: The scene
            entity_name: The name of the entity
            local_occupancy_map: The local occupancy map
        """
        SceneAsset.__init__(self, scene, entity_name)
        self.local_occupancy_map = local_occupancy_map

    @classmethod
    def from_boundary(
        cls, scene, entity_name: str, occupancy_map_boundary: np.ndarray, occupancy_map_resolution: float = 0.05
    ) -> "SceneFixture":
        """Create a SceneFixture from a known boundary/resolution pair

        Args:
            scene: The scene
            entity_name: The name of the entity
            occupancy_map_boundary: The boundary of the occupancy map
            occupancy_map_resolution: The resolution of the occupancy map

        Returns:
            SceneFixture: The SceneFixture
        """
        occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=occupancy_map_boundary, resolution=occupancy_map_resolution
        )
        return cls(scene, entity_name, occupancy_map)

    def get_occupancy_map(self):
        """Get the occupancy map of the SceneFixture

        Returns:
            OccupancyMap: The occupancy map
        """
        if self.local_occupancy_map is None:
            raise RuntimeError("SceneFixture requires an occupancy map before querying it.")

        transform = self.get_transform_2d().detach().cpu().numpy()
        # get_world_poses() may return a batched (num_envs, 3, 3) or empty (0, 3, 3) tensor.
        # For a fixed background asset placed at the world origin, fall back to identity when empty.
        if transform.ndim == 3:
            transform = transform[0] if transform.shape[0] > 0 else np.eye(3)

        return self.local_occupancy_map.transformed(transform)


def place_randomly(
    fixture: SceneFixture, background_occupancy_map: OccupancyMap, num_iter: int = 100, area_threshold: float = 1e-5
):
    """Place a scene fixture randomly in an unoccupied region of an occupancy."""

    # sample random xy in bounds
    bottom_left = background_occupancy_map.bottom_left_pixel_world_coords()
    top_right = background_occupancy_map.top_right_pixel_world_coords()

    initial_pose = fixture.get_pose()

    for i in range(num_iter):
        x = random.uniform(bottom_left[0], top_right[0])
        y = random.uniform(bottom_left[1], top_right[1])

        yaw = torch.tensor([random.uniform(-torch.pi, torch.pi)])
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        new_pose = initial_pose.clone()
        new_pose[0, 0] = x
        new_pose[0, 1] = y
        new_pose[0, 3:] = quat

        fixture.set_pose(new_pose)

        intersection_map = intersect_occupancy_maps([fixture.get_occupancy_map(), background_occupancy_map])

        intersection_area = np.count_nonzero(intersection_map.occupied_mask()) * (intersection_map.resolution**2)

        if intersection_area < area_threshold:
            return True

    return False
