# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import torch
import random
import numpy as np

import isaaclab.utils.math as math_utils

from dataclasses import dataclass
from occupancy_map import OccupancyMap, intersect_occupancy_maps

from isaacsim.replicator.mobility_gen.impl.path_planner import compress_path, generate_paths


def transform_to_matrix(transform: torch.Tensor):
    pose_matrix = math_utils.make_pose(
        transform[..., :3], 
        math_utils.matrix_from_quat(transform[..., 3:])
    )
    return pose_matrix


def transform_from_matrix(matrix: torch.Tensor):
    pos, rot = math_utils.unmake_pose(matrix)
    quat = math_utils.quat_from_matrix(rot)
    return torch.cat([pos, quat], dim=-1)


def transform_inv(transform: torch.Tensor):
    matrix = transform_to_matrix(transform)
    matrix = math_utils.pose_inv(matrix)
    return transform_from_matrix(matrix)


def transform_mul(transform_a, transform_b):
    return transform_from_matrix(
        torch.matmul(
            transform_to_matrix(transform_a),
            transform_to_matrix(transform_b)
        )
    )

def transform_relative_pose(
        world_pose: torch.Tensor,
        src_frame_pose: torch.Tensor,
        dst_frame_pose: torch.Tensor
    ):

    pose = transform_mul(
        dst_frame_pose,
        transform_mul(
            transform_inv(src_frame_pose),
            world_pose
        )
    )

    return pose


@dataclass
class DisjointNavRecordingItem:
    left_hand_pose_target: torch.Tensor
    right_hand_pose_target: torch.Tensor
    left_hand_joint_positions_target: torch.Tensor
    right_hand_joint_positions_target: torch.Tensor
    base_pose: torch.Tensor
    object_pose: torch.Tensor
    fixture_pose: torch.Tensor


class DisjointNavRecording:

    def get_initial_state(self):
        raise NotImplementedError

    def get_item(self, step: int) -> DisjointNavRecordingItem:
        raise NotImplementedError


class HasOccupancyMap:

    def get_occupancy_map(self) -> OccupancyMap:
        raise NotImplementedError
    

class HasPose2d:

    def get_pose_2d(self) -> torch.Tensor:
        raise NotImplementedError
    
    def get_transform_2d(self):

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
        transform[..., 2, 2] = 1.

        return transform


class HasPose(HasPose2d):

    def get_pose(self):
        raise NotImplementedError
    
    def get_pose_2d(self):
        pose = self.get_pose()
        axis_angle = math_utils.axis_angle_from_quat(pose[..., 3:])

        yaw = axis_angle[..., 2:3]
        xy = pose[..., :2]

        pose_2d = torch.cat([xy, yaw], dim=-1)

        return pose_2d



class SceneBody(HasPose):

    def __init__(self, scene, entity_name: str, body_name: str):
        self.scene = scene
        self.entity_name = entity_name
        self.body_name = body_name

    def get_pose(self):
        pose = self.scene[self.entity_name].data.body_link_state_w[
            :,
            self.scene[self.entity_name].data.body_names.index(self.body_name),
            :7,
        ]
        return pose
    

class SceneAsset(HasPose):

    def __init__(self, scene, entity_name: str):
        self.scene = scene
        self.entity_name = entity_name

    def get_pose(self):
        xform_prim = self.scene[self.entity_name]
        position, orientation = xform_prim.get_world_poses()
        position = position
        orientation = orientation
        pose = torch.cat([position, orientation], dim=-1)
        return pose

    def set_pose(self, pose: torch.Tensor):
        xform_prim = self.scene[self.entity_name]
        position = pose[..., :3]
        orientation = pose[..., 3:]
        xform_prim.set_world_poses(position, orientation, None)


class AbsolutePose(HasPose):

    def __init__(self, pose: torch.Tensor):
        self.pose = pose

    def get_pose(self):
        return self.pose
    

class RelativePose(HasPose):

    def __init__(self, relative_pose: torch.Tensor, parent: HasPose):
        self.relative_pose = relative_pose
        self.parent = parent

    def get_pose(self):

        parent_pose = self.parent.get_pose()

        pose = transform_mul(
            parent_pose,
            self.relative_pose
        )

        return pose


class SceneFixture(SceneAsset, HasOccupancyMap):
    pass


def plan_path(
        start: HasPose2d,
        end: HasPose2d,
        occupancy_map: OccupancyMap
    ):

    start_pose = start.get_pose_2d()[:, :2].numpy()
    end_pose = end.get_pose_2d()[:, :2].numpy()
    
    start_xy_px = occupancy_map.world_to_pixel_numpy(start_pose)
    end_xy_px = occupancy_map.world_to_pixel_numpy(end_pose)
    
    # xy -> yx
    start_yx_px = start_xy_px[..., 0, ::-1]
    end_yx_px = end_xy_px[..., 0, ::-1]

    path_planner_output = generate_paths(
        start=start_yx_px,
        freespace=occupancy_map.freespace_mask()
    )

    path_yx_px = path_planner_output.unroll_path(end_yx_px)
    path_yx_px, _ = compress_path(path_yx_px)

    # yx -> xy
    path_xy_px = path_yx_px[:, ::-1]

    path = occupancy_map.pixel_to_world_numpy(path_xy_px)

    path = torch.from_numpy(path)

    return path


def place_randomly(
        fixture: SceneFixture, 
        background_occupancy_map: OccupancyMap, 
        num_iter: int = 100,
        area_threshold: float = 1e-5
        ):

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

        intersection_map = intersect_occupancy_maps([
            fixture.get_occupancy_map(),
            background_occupancy_map
        ])

        intersection_area = np.count_nonzero(intersection_map.occupied_mask()) * (intersection_map.resolution**2)

        if intersection_area < area_threshold:
            return True
    
    return False


class DisjointNavScenario:

    def set_left_hand_pose_target(self, pose: torch.Tensor):
        """Set the left hand pose target in world coordinates."""
        raise NotImplementedError
    
    def set_right_hand_pose_target(self, pose: torch.Tensor):
        """Set the right hand pose target in world coordinates."""
        raise NotImplementedError
    
    def set_left_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        """Set the left hand joint position target."""
        raise NotImplementedError

    def set_right_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        """Set the right hand joint position target."""
        raise NotImplementedError
    
    def set_base_velocity_target(self, velocity: torch.Tensor):
        """Set the base velocity in local robot frame."""
        raise NotImplementedError
    
    def get_base(self) -> HasPose:
        """Get the robot base body."""
        raise NotImplementedError
    
    def get_left_hand(self) -> HasPose:
        """Get the robot left hand body."""
        raise NotImplementedError
    
    def get_right_hand(self) -> HasPose:
        """Get the robot right hand body."""
        raise NotImplementedError
    
    def get_object(self) -> HasPose:
        """Get the target object body."""
        raise NotImplementedError
    
    def get_start_fixture(self) -> SceneFixture:
        """Get the start fixture body."""
        raise NotImplementedError
    
    def get_end_fixture(self) -> SceneFixture:
        """Get the end fixture body."""
        raise NotImplementedError
    
    def get_obstacle_fixtures(self) -> list[SceneFixture]:
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def reset(self, intial_state = None):
        raise NotImplementedError
    
    