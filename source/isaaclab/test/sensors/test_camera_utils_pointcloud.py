# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the depth-to-pointcloud helpers in :mod:`isaaclab.sensors.camera.utils`."""

import math

import numpy as np
import pytest
import torch

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, create_pointcloud_from_rgbd

pytestmark = pytest.mark.unit

# pixel (u, v) unprojects to (u * d, v * d, d) with an identity calibration matrix
INTRINSICS = torch.eye(3)
# (v, u) pixels with a missing (zero), NaN, and infinite depth measurement
INVALID_PIXELS = {(0, 1): 0.0, (1, 2): math.nan, (2, 3): math.inf}


def _depth_image() -> torch.Tensor:
    """A non-square (H=3, W=4) depth image with a distinct positive depth per pixel and three invalid pixels."""
    depth = 1.0 + 0.5 * torch.arange(12, dtype=torch.float32).reshape(3, 4)
    for pixel, value in INVALID_PIXELS.items():
        depth[pixel] = value
    return depth


def _sort_rows(points: torch.Tensor) -> torch.Tensor:
    """Sort points lexicographically so clouds can be compared independently of their order."""
    return torch.tensor(sorted(points.tolist()))


def test_pointcloud_from_depth_drops_invalid_pixels():
    """Zero, NaN, and infinite depths produce no points, including zero depth at a camera pose."""
    depth = _depth_image()
    position = torch.tensor([1.0, 2.0, 3.0])
    # 90 degrees about the z-axis, (x, y, z, w): maps (x, y, z) to (-y, x, z)
    orientation = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]

    points = create_pointcloud_from_depth(INTRINSICS, depth, position=position, orientation=orientation)

    expected = [
        [-v * d + 1.0, u * d + 2.0, d + 3.0]
        for v in range(3)
        for u in range(4)
        if (v, u) not in INVALID_PIXELS
        for d in [depth[v, u].item()]
    ]
    assert points.shape == (9, 3)
    torch.testing.assert_close(_sort_rows(points), _sort_rows(torch.tensor(expected)))


def test_pointcloud_from_depth_keep_invalid_keeps_every_pixel():
    """With ``keep_invalid``, every pixel keeps its point, so a zero depth stays at the camera origin."""
    points = create_pointcloud_from_depth(INTRINSICS, _depth_image(), keep_invalid=True)

    assert points.shape == (12, 3)
    assert (points == 0).all(dim=1).sum() == 1
    assert (~torch.isfinite(points).all(dim=1)).sum() == 2


def test_pointcloud_from_rgbd_drops_invalid_pixels_with_their_colors():
    """Invalid pixels are removed from both outputs, and each kept point keeps the color of its pixel."""
    depth = _depth_image().numpy()
    # color each pixel (u, v) as (u, v, 7) to check which pixel a kept color came from
    v_grid, u_grid = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    rgb = np.stack([u_grid, v_grid, np.full_like(u_grid, 7)], axis=-1).astype(np.uint8)

    points_xyz, points_rgb = create_pointcloud_from_rgbd(INTRINSICS, depth, rgb=rgb)

    assert isinstance(points_xyz, np.ndarray) and isinstance(points_rgb, np.ndarray)
    assert points_xyz.shape == (9, 3) and points_rgb.shape == (9, 3)
    kept_pixels = {(int(v), int(u)) for u, v, _ in points_rgb}
    assert kept_pixels == {(v, u) for v in range(3) for u in range(4)} - INVALID_PIXELS.keys()
    # the point of pixel (u, v) is (u * d, v * d, d)
    np.testing.assert_allclose(points_xyz[:, 0], points_rgb[:, 0] * points_xyz[:, 2])
    np.testing.assert_allclose(points_xyz[:, 1], points_rgb[:, 1] * points_xyz[:, 2])
    np.testing.assert_allclose(points_xyz[:, 2], depth[points_rgb[:, 1].astype(int), points_rgb[:, 0].astype(int)])
