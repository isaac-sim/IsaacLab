# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class VisuoTactileSensorData:
    """Data container for the visuo-tactile sensor.

    This class contains the tactile sensor data that includes:
    - Camera-based tactile sensing (RGB and depth images)
    - Force field tactile sensing (normal and shear forces)
    - Tactile point positions and contact information
    """

    # Camera-based tactile data
    tactile_camera_depth: torch.Tensor | None = None
    """Tactile depth images. Shape: (num_instances, height, width)."""

    taxim_tactile: torch.Tensor | None = None
    """Nominal (reference) tactile images with no contact. Shape: (num_instances, height, width, 3)."""

    # Force field tactile data
    tactile_points_pos_w: torch.Tensor | None = None
    """Positions of tactile points in world frame. Shape: (num_instances, num_tactile_points, 3)."""

    tactile_points_quat_w: torch.Tensor | None = None
    """Orientations of tactile points in world frame. Shape: (num_instances, num_tactile_points, 4)."""

    penetration_depth: torch.Tensor | None = None
    """Penetration depth at each tactile point. Shape: (num_instances, num_tactile_points)."""

    tactile_normal_force: torch.Tensor | None = None
    """Normal forces at each tactile point. Shape: (num_instances, num_tactile_points)."""

    tactile_shear_force: torch.Tensor | None = None
    """Shear forces at each tactile point. Shape: (num_instances, num_tactile_points, 2)."""
