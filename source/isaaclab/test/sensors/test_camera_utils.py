# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for invalid depth filtering in pointcloud helpers."""

import numpy as np
import pytest
import torch

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, create_pointcloud_from_rgbd

pytestmark = pytest.mark.unit

DEPTH = np.array([[1.0, 0.0, np.nan], [2.0, np.inf, 3.0]], dtype=np.float32)


def test_depth_filters_invalid_pixels_with_camera_pose():
    """Filtering uses input depth even when zero depth becomes the camera position."""
    points = create_pointcloud_from_depth(torch.eye(3), torch.from_numpy(DEPTH), position=(1.0, 2.0, 3.0))
    assert points.shape == (3, 3)
    assert {tuple(point) for point in points.tolist()} == {(1, 2, 4), (1, 4, 5), (7, 5, 6)}


def test_rgbd_filters_invalid_pixels_with_their_colors():
    """Only valid pixels survive, paired with their original RGB values."""
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    points, colors = create_pointcloud_from_rgbd(np.eye(3), DEPTH, rgb=rgb)
    assert isinstance(points, np.ndarray) and isinstance(colors, np.ndarray)
    assert points.shape == colors.shape == (3, 3)
    pairs = {(*point, *color) for point, color in zip(points, colors, strict=True)}
    assert pairs == {(0, 0, 1, 0, 1, 2), (0, 2, 2, 9, 10, 11), (6, 3, 3, 15, 16, 17)}
