# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lidar pose composition and return-layout contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.sensors import LidarCfg
from isaaclab.sensors.lidar.lidar import Lidar
from isaaclab.utils.warp import ProxyArray


def _lidar_cfg(**overrides) -> LidarCfg:
    """Build a one-ray lidar config. ``overrides`` replace the valid defaults."""
    values = {
        "prim_path": "/World/envs/env_.*/Lidar",
        "azimuth_deg": [0.0],
        "elevation_deg": [0.0],
        "fire_time_ns": [0],
        "channel_id": [0],
        "renderer_cfg": SimpleNamespace(renderer_type="ovrtx"),
    }
    values.update(overrides)
    return LidarCfg(**values)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("coords_type", "SPHERICAL", "CARTESIAN"),
        ("include_invalid_points", False, "include_invalid_points"),
    ],
)
def test_lidar_cfg_rejects_returns_that_lose_ray_identity(field, value, match):
    """Spherical coordinates and compacted misses cannot be published as Cartesian ray slots."""
    with pytest.raises(ValueError, match=match):
        _lidar_cfg(**{field: value})


def test_lidar_world_pose_composes_a_rotated_parent():
    """A parent yaw moves the offset off the world X axis instead of adding it to the origin."""
    lidar = Lidar.__new__(Lidar)
    lidar._initialize_handle = None
    lidar._invalidate_initialize_handle = None
    lidar._prim_deletion_handle = None
    lidar._debug_vis_handle = None
    lidar._renderer = None
    lidar._render_data = None
    lidar._num_envs = 1
    lidar._local_pos = torch.tensor([1.0, 0.0, 0.0])
    lidar._local_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    lidar._pos_w = ProxyArray(wp.zeros(1, dtype=wp.vec3f, device="cpu"))
    lidar._quat_w = ProxyArray(wp.zeros(1, dtype=wp.quatf, device="cpu"))
    half = math.sqrt(0.5)
    parent_pos = torch.tensor([[2.0, 3.0, 4.0]])
    parent_quat = torch.tensor([[0.0, 0.0, half, half]])
    lidar._parent_view = SimpleNamespace(
        get_world_poses=lambda: (SimpleNamespace(torch=parent_pos), SimpleNamespace(torch=parent_quat)),
        close=lambda: None,
    )

    lidar._write_world_poses()

    torch.testing.assert_close(lidar._pos_w.torch, torch.tensor([[2.0, 4.0, 4.0]]))
    torch.testing.assert_close(lidar._quat_w.torch, parent_quat)
