# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rgb", [(255, 0, 0), [0, 255, 0]])
def test_create_pointcloud_from_rgbd_constant_color(rgb):
    """Constant tuple/list colors should be expanded across every point."""
    depth = torch.ones((2, 2), dtype=torch.float32)
    intrinsic_matrix = torch.eye(3, dtype=torch.float32)

    points, colors = create_pointcloud_from_rgbd(intrinsic_matrix, depth, rgb=rgb, device="cpu")

    assert points.shape == (4, 3)
    expected = torch.tensor(rgb, dtype=torch.uint8).expand(4, -1)
    assert torch.equal(colors, expected)


def test_create_pointcloud_from_rgbd_rejects_invalid_constant_color_width():
    """Constant colors must contain exactly RGB components."""
    depth = torch.ones((2, 2), dtype=torch.float32)
    intrinsic_matrix = torch.eye(3, dtype=torch.float32)

    with pytest.raises(ValueError, match="exactly three components"):
        create_pointcloud_from_rgbd(intrinsic_matrix, depth, rgb=[255, 0], device="cpu")


def test_create_pointcloud_from_rgbd_default_color():
    """Missing RGB data should use the documented black fallback without raising."""
    depth = torch.ones((2, 2), dtype=torch.float32)
    intrinsic_matrix = torch.eye(3, dtype=torch.float32)

    points, colors = create_pointcloud_from_rgbd(intrinsic_matrix, depth, rgb=None, device="cpu")

    assert points.shape == (4, 3)
    assert torch.equal(colors, torch.zeros((4, 3), dtype=torch.uint8))
