# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch

from isaacsim.replicator.mobility_gen.impl.path_planner import compress_path, generate_paths

from .occupancy_map_utils import OccupancyMap
from .scene_utils import HasPose2d


def nearest_point_on_segment(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest point on line segment AB to point C.

    This function computes the closest point on the line segment from A to B
    to a given point C, along with the distance from A to that point along the segment.

    Args:
        a (torch.Tensor): Start point of the line segment.
        b (torch.Tensor): End point of the line segment.
        c (torch.Tensor): Query point to find the nearest point to.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The nearest point on the segment AB to point C
            - The distance along the segment from A to the nearest point
    """
    a2b = b - a
    a2c = c - a
    a2b_mag = torch.sqrt(torch.sum(a2b**2))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = torch.dot(a2c, a2b_norm)
    if dist < 0:
        return a, dist
    elif dist > a2b_mag:
        return b, dist
    else:
        return a + a2b_norm * dist, dist


class ParameterizedPath:
    """Path parameterized by arc length for distance-based queries and interpolation."""

    def __init__(self, points: torch.Tensor) -> None:
        """Initialize parameterized path with waypoints.

        Args:
            points (torch.Tensor): Sequential waypoints of shape (N, 2).
        """
        self.points = points
        self._init_point_distances()

    def _init_point_distances(self) -> None:
        """Initialize arc length parameterization."""
        self._point_distances = torch.zeros(len(self.points))
        length = 0.0
        for i in range(0, len(self.points) - 1):
            self._point_distances[i] = length
            a = self.points[i]
            b = self.points[i + 1]
            dist = torch.sqrt(torch.sum((a - b) ** 2))
            length += dist
        self._point_distances[-1] = length

    def point_distances(self) -> torch.Tensor:
        """Get arc length parameters for each waypoint.

        Returns:
            torch.Tensor: Arc length parameter values.
        """
        return self._point_distances

    def get_path_length(self) -> float:
        """Calculate total path length.

        Returns:
            float: Total euclidean distance from start to end.
        """
        length = 0.0
        for i in range(1, len(self.points)):
            a = self.points[i - 1]
            b = self.points[i]
            dist = torch.sqrt(torch.sum((a - b) ** 2))
            length += dist
        return length

    def points_x(self) -> torch.Tensor:
        """Get x-coordinates of all path points.

        Returns:
            torch.Tensor: X-coordinates of all points.
        """
        return self.points[:, 0]

    def points_y(self) -> torch.Tensor:
        """Get y-coordinates of all path points.

        Returns:
            torch.Tensor: Y-coordinates of all points.
        """
        return self.points[:, 1]

    def get_segment_by_distance(self, distance: float) -> tuple[int, int]:
        """Find path segment containing given distance.

        Args:
            distance (float): Distance along path from start.

        Returns:
            Tuple[int, int]: Indices of segment endpoints.
        """
        for i in range(0, len(self.points) - 1):
            d_b = self._point_distances[i + 1]

            if distance < d_b:
                return (i, i + 1)

        i = len(self.points) - 2
        return (i, i + 1)

    def get_point_by_distance(self, distance: float) -> torch.Tensor:
        """Sample point at specified arc length parameter.

        Args:
            distance (float): Arc length parameter from start.

        Returns:
            torch.Tensor: Interpolated 2D coordinates.
        """
        a_idx, b_idx = self.get_segment_by_distance(distance)
        a, b = self.points[a_idx], self.points[b_idx]
        a_dist, b_dist = self._point_distances[a_idx], self._point_distances[b_idx]
        u = (distance - a_dist) / ((b_dist - a_dist) + 1e-6)
        u = torch.clip(u, 0.0, 1.0)
        return a + u * (b - a)

    def find_nearest(self, point: torch.Tensor) -> tuple[torch.Tensor, float, tuple[int, int], float]:
        """Find nearest point on path to query point.

        Args:
            point (torch.Tensor): The query point as a 2D tensor.

        Returns:
            Tuple containing:
            - torch.Tensor: The nearest point on the path to the query point
            - float: Distance along the path from the start to the nearest point
            - Tuple[int, int]: Indices of the segment containing the nearest point
            - float: Euclidean distance from the query point to the nearest point on path
        """
        min_pt_dist_to_seg = 1e9
        min_pt_seg = None
        min_pt = None
        min_pt_dist_along_path = None

        for a_idx in range(0, len(self.points) - 1):
            b_idx = a_idx + 1
            a = self.points[a_idx]
            b = self.points[b_idx]
            nearest_pt, dist_along_seg = nearest_point_on_segment(a, b, point)
            dist_to_seg = torch.sqrt(torch.sum((point - nearest_pt) ** 2))

            if dist_to_seg < min_pt_dist_to_seg:
                min_pt_seg = (a_idx, b_idx)
                min_pt_dist_to_seg = dist_to_seg
                min_pt = nearest_pt
                min_pt_dist_along_path = self._point_distances[a_idx] + dist_along_seg

        return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg


def plan_path(start: HasPose2d, end: HasPose2d, occupancy_map: OccupancyMap) -> torch.Tensor | None:
    """Plan collision-free path between start and end positions.

    Args:
        start (HasPose2d): Start entity with 2D pose.
        end (HasPose2d): Target entity with 2D pose.
        occupancy_map (OccupancyMap): Occupancy map defining obstacles.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) representing the planned path as a
                     sequence of 2D waypoints from start to end.
    """

    # Extract 2D positions from poses
    start_world_pos = start.get_pose_2d()[:, :2].numpy()
    end_world_pos = end.get_pose_2d()[:, :2].numpy()

    # Convert world coordinates to pixel coordinates
    start_xy_pixels = occupancy_map.world_to_pixel_numpy(start_world_pos)
    end_xy_pixels = occupancy_map.world_to_pixel_numpy(end_world_pos)

    # Convert from (x, y) to (y, x) format required by path planner
    start_yx_pixels = start_xy_pixels[..., 0, ::-1]
    end_yx_pixels = end_xy_pixels[..., 0, ::-1]

    # Check if end_yx_pixels are inside the occupancy map bounds
    map_height, map_width = occupancy_map.freespace_mask().shape
    start_y, start_x = int(start_yx_pixels[0]), int(start_yx_pixels[1])
    end_y, end_x = int(end_yx_pixels[0]), int(end_yx_pixels[1])

    if not occupancy_map.check_pixel_in_bounds(start_x, start_y):
        print(
            f"Warning: start_yx_pixels ({start_y}, {start_x}) is outside occupancy map bounds "
            f"(height={map_height}, width={map_width})"
        )
        return None

    if not occupancy_map.check_pixel_in_bounds(end_x, end_y):
        print(
            f"Warning: end_yx_pixels ({end_y}, {end_x}) is outside occupancy map bounds "
            f"(height={map_height}, width={map_width})"
        )
        return None

    # Generate path using the mobility path planner
    path_planner_output = generate_paths(start=start_yx_pixels, freespace=occupancy_map.freespace_mask())

    # Extract and compress the path
    path_yx_pixels = path_planner_output.unroll_path(end_yx_pixels)
    path_yx_pixels, _ = compress_path(path_yx_pixels)

    # Convert back from (y, x) to (x, y) format
    path_xy_pixels = path_yx_pixels[:, ::-1]

    # Convert pixel coordinates back to world coordinates
    path_world = occupancy_map.pixel_to_world_numpy(path_xy_pixels)

    # Convert to torch tensor and return
    path_tensor = torch.from_numpy(path_world)

    return path_tensor
