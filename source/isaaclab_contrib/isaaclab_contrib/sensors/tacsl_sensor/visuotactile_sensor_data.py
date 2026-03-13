# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VisuoTactileSensorData:
    """Data container for the visuo-tactile sensor.

    This class contains the tactile sensor data that includes:

    - Camera-based tactile sensing (RGB and depth images)
    - Force field tactile sensing (normal and shear forces)
    - Tactile point positions and contact information

    """

    # Camera-based tactile data
    tactile_depth_image: torch.Tensor | None = None
    """Tactile depth images. Shape is (num_instances, height, width, 1)."""

    tactile_rgb_image: torch.Tensor | None = None
    """Tactile RGB images rendered using the Taxim approach from :cite:t:`si2022taxim`.
    Shape is (num_instances, height, width, 3).
    """

    # Force field tactile data
    tactile_points_pos_w: torch.Tensor | None = None
    """Positions of tactile points in world frame. Shape is (num_instances, num_tactile_points, 3)."""

    tactile_points_quat_w: torch.Tensor | None = None
    """Orientations of tactile points in world frame. Shape is (num_instances, num_tactile_points, 4)."""

    penetration_depth: torch.Tensor | None = None
    """Penetration depth at each tactile point. Shape is (num_instances, num_tactile_points)."""

    tactile_normal_force: torch.Tensor | None = None
    """Normal forces at each tactile point in sensor frame. Shape is (num_instances, num_tactile_points)."""

    tactile_shear_force: torch.Tensor | None = None
    """Shear forces at each tactile point in sensor frame. Shape is (num_instances, num_tactile_points, 2)."""
