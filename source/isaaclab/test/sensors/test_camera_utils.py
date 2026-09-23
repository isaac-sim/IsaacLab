# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the camera point cloud utilities."""

import numpy as np
import pytest
import torch

from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd

pytestmark = pytest.mark.unit

_DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
_INTRINSICS = [[20.0, 0.0, 2.5], [0.0, 20.0, 2.0], [0.0, 0.0, 1.0]]
_HEIGHT, _WIDTH = 4, 5


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("rgb, expected_color", [(None, (0, 0, 0)), ((255, 0, 128), (255, 0, 128))])
def test_pointcloud_from_rgbd_uniform_color(device, rgb, expected_color):
    """A color tuple, or no color, gives every point the same color on the depth image's device."""
    depth = torch.ones(_HEIGHT, _WIDTH, device=device)
    intrinsics = torch.tensor(_INTRINSICS, device=device)

    points_xyz, points_rgb = create_pointcloud_from_rgbd(intrinsics, depth, rgb=rgb)

    assert points_rgb.device == points_xyz.device
    assert points_rgb.dtype == torch.uint8
    expected = np.tile(np.array(expected_color, dtype=np.uint8), (points_xyz.shape[0], 1))
    np.testing.assert_array_equal(points_rgb.cpu().numpy(), expected)


def test_pointcloud_from_rgbd_uniform_color_numpy_normalized():
    """Numpy depth returns numpy outputs, and a normalized color tuple is scaled to [0, 1]."""
    depth = np.ones((_HEIGHT, _WIDTH), dtype=np.float32)

    points_xyz, points_rgb = create_pointcloud_from_rgbd(
        np.array(_INTRINSICS), depth, rgb=(255, 0, 51), normalize_rgb=True
    )

    assert isinstance(points_rgb, np.ndarray)
    expected = np.tile(np.array([1.0, 0.0, 0.2], dtype=np.float32), (points_xyz.shape[0], 1))
    np.testing.assert_allclose(points_rgb, expected, rtol=1e-6)
