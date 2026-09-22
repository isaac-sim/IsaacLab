# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ray caster pattern generators."""

from __future__ import annotations

import math

import pytest
import torch

from isaaclab.sensors.ray_caster.patterns import patterns, patterns_cfg

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", params=["cuda", "cpu"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


def _assert_unit_rays_from_origin(ray_starts: torch.Tensor, ray_directions: torch.Tensor):
    norms = torch.linalg.norm(ray_directions, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ray_starts, torch.zeros_like(ray_starts))


"""
Grid pattern
"""


@pytest.mark.parametrize(
    "size, resolution, ordering, expected_num_rays",
    [
        ((2.0, 2.0), 1.0, "xy", 9),
        ((4.0, 2.0), 1.0, "xy", 15),
        ((2.0, 4.0), 1.0, "yx", 15),
        ((1.0, 1.0), 0.25, "xy", 25),
    ],
)
def test_grid_pattern(device, size, resolution, ordering, expected_num_rays):
    """Ray count, bounds, z=0 starts, ordering, and direction of the grid pattern."""
    direction = (0.0, 0.0, -1.0)
    cfg = patterns_cfg.GridPatternCfg(size=size, resolution=resolution, ordering=ordering, direction=direction)
    ray_starts, ray_directions = patterns.grid_pattern(cfg, device)

    assert ray_starts.shape == ray_directions.shape == (expected_num_rays, 3)
    assert ray_starts[:, 0].abs().max() <= size[0] / 2 + 1e-6
    assert ray_starts[:, 1].abs().max() <= size[1] / 2 + 1e-6
    torch.testing.assert_close(ray_starts[:, 2], torch.zeros_like(ray_starts[:, 2]))
    torch.testing.assert_close(ray_directions, torch.tensor(direction, device=device).expand_as(ray_directions))
    # the first axis in the ordering varies fastest
    fast, slow = (0, 1) if ordering == "xy" else (1, 0)
    assert ray_starts[0, slow] == ray_starts[1, slow]
    assert ray_starts[0, fast] != ray_starts[1, fast]


def test_grid_pattern_invalid_cfg(device):
    with pytest.raises(ValueError, match="Ordering must be 'xy' or 'yx'"):
        patterns.grid_pattern(patterns_cfg.GridPatternCfg(size=(2.0, 2.0), resolution=1.0, ordering="invalid"), device)
    with pytest.raises(ValueError, match="Resolution must be greater than 0"):
        patterns.grid_pattern(patterns_cfg.GridPatternCfg(size=(2.0, 2.0), resolution=-1.0), device)


"""
Lidar pattern
"""


@pytest.mark.parametrize(
    "horizontal_fov_range, horizontal_res, channels, vertical_fov_range",
    [
        ((-180.0, 180.0), 90.0, 1, (-10.0, -10.0)),  # 360 deg: last sample excluded to avoid overlap
        ((-180.0, 180.0), 45.0, 16, (-15.0, 15.0)),
        ((-90.0, 90.0), 30.0, 1, (-10.0, -10.0)),  # partial FOV keeps both end points
        ((0.0, 180.0), 45.0, 5, (-30.0, 10.0)),
    ],
)
def test_lidar_pattern(device, horizontal_fov_range, horizontal_res, channels, vertical_fov_range):
    """Ray count, unit directions, and the vertical/horizontal angle coverage of the lidar pattern."""
    cfg = patterns_cfg.LidarPatternCfg(
        horizontal_fov_range=horizontal_fov_range,
        horizontal_res=horizontal_res,
        channels=channels,
        vertical_fov_range=vertical_fov_range,
    )
    ray_starts, ray_directions = patterns.lidar_pattern(cfg, device)

    fov = horizontal_fov_range[1] - horizontal_fov_range[0]
    expected_num_horizontal = math.ceil(fov / horizontal_res) + 1 - (abs(fov - 360.0) < 1e-6)
    assert ray_starts.shape == ray_directions.shape == (channels * expected_num_horizontal, 3)
    _assert_unit_rays_from_origin(ray_starts, ray_directions)

    # elevation spans the vertical range; azimuth samples are spaced by the horizontal resolution
    elevation = torch.rad2deg(torch.asin(ray_directions[:, 2]))
    torch.testing.assert_close(elevation.min().item(), vertical_fov_range[0], atol=1e-3, rtol=0)
    torch.testing.assert_close(elevation.max().item(), vertical_fov_range[1], atol=1e-3, rtol=0)
    azimuth = torch.rad2deg(torch.atan2(ray_directions[:, 1], ray_directions[:, 0])) % 360.0
    expected_azimuth = horizontal_fov_range[0] + horizontal_res * torch.arange(expected_num_horizontal, device=device)
    torch.testing.assert_close(
        azimuth.round(decimals=2).unique(sorted=True), (expected_azimuth % 360.0).round(decimals=2).unique(sorted=True)
    )


"""
Bpearl pattern
"""


@pytest.mark.parametrize("horizontal_fov, horizontal_res", [(360.0, 10.0), (180.0, 10.0), (90.0, 5.0)])
def test_bpearl_pattern(device, horizontal_fov, horizontal_res):
    """Ray count follows the horizontal parameters and the configured vertical angles."""
    vertical_ray_angles = [10.0, 20.0, 30.0]
    cfg = patterns_cfg.BpearlPatternCfg(
        horizontal_fov=horizontal_fov, horizontal_res=horizontal_res, vertical_ray_angles=vertical_ray_angles
    )
    ray_starts, ray_directions = patterns.bpearl_pattern(cfg, device)

    assert ray_starts.shape == (len(vertical_ray_angles) * int(horizontal_fov / horizontal_res), 3)
    _assert_unit_rays_from_origin(ray_starts, ray_directions)


"""
Pinhole camera pattern
"""


def test_pinhole_camera_pattern(device):
    """Rays are per-pixel unit vectors from the origin and depend on the per-camera intrinsics."""
    width, height = 32, 16
    cfg = patterns_cfg.PinholeCameraPatternCfg(width=width, height=height)
    intrinsics = torch.tensor([[[500.0, 0.0, 16.0], [0.0, 500.0, 8.0], [0.0, 0.0, 1.0]]] * 2, device=device)
    intrinsics[1, 0, 0] = intrinsics[1, 1, 1] = 800.0

    ray_starts, ray_directions = patterns.pinhole_camera_pattern(cfg, intrinsics, device)

    assert ray_starts.shape == ray_directions.shape == (2, width * height, 3)
    _assert_unit_rays_from_origin(ray_starts, ray_directions)
    # the center pixel looks along the sensor +x axis; the batches differ because the focal lengths differ
    center = (height // 2) * width + width // 2
    torch.testing.assert_close(
        ray_directions[0, center], torch.tensor([1.0, 0.0, 0.0], device=device), atol=0.05, rtol=0
    )
    assert not torch.allclose(ray_directions[0], ray_directions[1])


def test_pinhole_camera_cfg_from_intrinsic_matrix():
    """Apertures derived from an intrinsic matrix reproduce the given focal lengths."""
    width, height, fx, fy = 640, 480, 500.0, 400.0
    cfg = patterns_cfg.PinholeCameraPatternCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx, 0, width / 2, 0, fy, height / 2, 0, 0, 1], width=width, height=height
    )
    assert (cfg.width, cfg.height) == (width, height)
    assert width * cfg.focal_length / cfg.horizontal_aperture == pytest.approx(fx)
    assert height * cfg.focal_length / cfg.vertical_aperture == pytest.approx(fy)
