# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared geometry for Franka Pour runtime checks and reset generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils


@dataclass(frozen=True)
class CupGeometry:
    """Dimensions of one base-origin rectangular cup [m]."""

    inner_width: float
    inner_depth: float
    cavity_depth: float
    wall_thickness: float
    bottom_thickness: float

    @property
    def rim_height(self) -> float:
        """Cup rim height above its base [m]."""
        return self.bottom_thickness + self.cavity_depth

    @property
    def outer_half_extents(self) -> tuple[float, float, float]:
        """Half extents of the complete outer envelope [m]."""
        return (
            0.5 * self.inner_width + self.wall_thickness,
            0.5 * self.inner_depth + self.wall_thickness,
            0.5 * self.rim_height,
        )

    @property
    def inner_bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Lower and upper cavity bounds in the cup frame [m]."""
        return (
            (-0.5 * self.inner_width, -0.5 * self.inner_depth, self.bottom_thickness),
            (0.5 * self.inner_width, 0.5 * self.inner_depth, self.rim_height),
        )

    @property
    def collision_boxes(
        self,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
        """Bottom and four wall boxes as ``(offset, half_extents)`` pairs [m]."""
        outer_x, outer_y, _ = self.outer_half_extents
        half_wall = 0.5 * self.wall_thickness
        half_height = 0.5 * self.rim_height
        return (
            ((0.0, 0.0, 0.5 * self.bottom_thickness), (outer_x, outer_y, 0.5 * self.bottom_thickness)),
            ((0.5 * self.inner_width + half_wall, 0.0, half_height), (half_wall, outer_y, half_height)),
            ((-0.5 * self.inner_width - half_wall, 0.0, half_height), (half_wall, outer_y, half_height)),
            ((0.0, 0.5 * self.inner_depth + half_wall, half_height), (0.5 * self.inner_width, half_wall, half_height)),
            ((0.0, -0.5 * self.inner_depth - half_wall, half_height), (0.5 * self.inner_width, half_wall, half_height)),
        )


SOURCE_CUP_GEOMETRY = CupGeometry(0.042, 0.042, 0.110, 0.007, 0.009)
TARGET_CUP_GEOMETRY = CupGeometry(0.140, 0.140, 0.065, 0.009, 0.009)
CUP_GRASP_HEIGHT = 0.083
CUP_GRASP_TCP_QUAT_C = (0.0, 2.0**-0.5, 0.0, 2.0**-0.5)
MPM_COLLIDER_MARGIN = 0.004


def box_center_from_base_pose(pose: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """Return world box centers for poses whose origins lie at the outer base [m]."""
    offset = torch.zeros_like(pose[:, :3])
    offset[:, 2] = half_extents[2]
    return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], offset)


def points_inside_box(
    points: torch.Tensor,
    pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Return whether batched points lie in an oriented box [m]."""
    relative = points - pose[:, None, :3]
    quaternion = pose[:, None, 3:7].expand(-1, points.shape[1], -1)
    local = math_utils.quat_apply_inverse(quaternion, relative)
    return ((local >= lower - margin) & (local <= upper + margin)).all(dim=-1)


def oriented_boxes_overlap(
    center_a: torch.Tensor,
    quaternion_a: torch.Tensor,
    half_a: torch.Tensor,
    center_b: torch.Tensor,
    quaternion_b: torch.Tensor,
    half_b: torch.Tensor,
    clearance: float = 0.0,
) -> torch.Tensor:
    """Return batched oriented-box overlap using the complete 15-axis SAT."""
    rotation_a = math_utils.matrix_from_quat(quaternion_a)
    rotation_b = math_utils.matrix_from_quat(quaternion_b)
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    absolute = relative.abs() + 1.0e-6
    translation = (rotation_a.transpose(-1, -2) @ (center_b - center_a).unsqueeze(-1)).squeeze(-1)
    a = half_a + 0.5 * clearance
    b = half_b + 0.5 * clearance
    separated = torch.zeros(center_a.shape[0], device=center_a.device, dtype=torch.bool)
    for axis in range(3):
        separated |= translation[:, axis].abs() > a[axis] + (absolute[:, axis, :] * b).sum(-1)
        projection = (translation * relative[:, :, axis]).sum(-1).abs()
        separated |= projection > (absolute[:, :, axis] * a).sum(-1) + b[axis]
    for axis_a in range(3):
        for axis_b in range(3):
            other_a, last_a = (axis_a + 1) % 3, (axis_a + 2) % 3
            other_b, last_b = (axis_b + 1) % 3, (axis_b + 2) % 3
            projection = (
                translation[:, last_a] * relative[:, other_a, axis_b]
                - translation[:, other_a] * relative[:, last_a, axis_b]
            ).abs()
            radius_a = a[other_a] * absolute[:, last_a, axis_b] + a[last_a] * absolute[:, other_a, axis_b]
            radius_b = b[other_b] * absolute[:, axis_a, last_b] + b[last_b] * absolute[:, axis_a, other_b]
            separated |= projection > radius_a + radius_b
    return ~separated


def box_supported(
    pose: torch.Tensor,
    half_extents: torch.Tensor,
    support_lower: torch.Tensor,
    support_upper: torch.Tensor,
    clearance: float,
) -> torch.Tensor:
    """Return whether each oriented box lies inside planar support bounds [m]."""
    center = box_center_from_base_pose(pose, half_extents)
    radius = (math_utils.matrix_from_quat(pose[:, 3:7]).abs()[:, :2, :] * half_extents).sum(-1) + clearance
    return ((center[:, :2] - radius >= support_lower) & (center[:, :2] + radius <= support_upper)).all(-1)


def box_above_plane(
    pose: torch.Tensor,
    half_extents: torch.Tensor,
    *,
    height: float = 0.0,
    penetration_tolerance: float = 0.0,
) -> torch.Tensor:
    """Return whether each oriented box stays above a horizontal plane [m]."""
    center = box_center_from_base_pose(pose, half_extents)
    radius = (math_utils.matrix_from_quat(pose[:, 3:7]).abs()[:, 2, :] * half_extents).sum(-1)
    return center[:, 2] - radius >= height - penetration_tolerance
