# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The analytic lens textures use the software sampler's directions and invert the native grid."""

import math

import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import angular_directions
from isaaclab_fovi.sensors.lens_lut import build_native_grid_textures
from isaaclab_fovi.sensors.native_grid import bounded_angular_radius, native_grid_directions, native_grid_exit_uv

pytestmark = pytest.mark.kitless


def texel_centers(resolution: int) -> torch.Tensor:
    """Native texture UV of each canonical sample: columns follow native x, rows native y."""
    axis = (torch.arange(resolution) + 0.5) / resolution
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((col, row), -1).reshape(-1, 2)


def test_rolled_grid_matches_fovi_and_inverts_every_texel():
    coords = SamplingCoords(170.0, 6.0, 65, style="warped_cartesian_as_grid", field_geometry="spherical")
    roll = math.radians(37.0)
    rendered = native_grid_directions(coords, image_rotation_deg=37.0)
    valid = coords.valid_mask
    cv_rays = angular_directions(coords.cartesian[valid], coords.fov)
    x, y, z = cv_rays.unbind(-1)
    reference = torch.stack((math.cos(roll) * x + math.sin(roll) * y, math.sin(roll) * x - math.cos(roll) * y, -z), -1)
    torch.testing.assert_close(rendered[valid], reference, atol=3e-7, rtol=0)

    uv, visible = native_grid_exit_uv(rendered, coords, image_rotation_deg=37.0)
    assert visible.all()
    torch.testing.assert_close(uv, texel_centers(coords.resolution), atol=2e-6, rtol=0)

    directions, exit_uv = build_native_grid_textures(coords, rendered, image_rotation_deg=37.0, exit_size=129)
    torch.testing.assert_close(directions.reshape(-1, 3), rendered)
    torch.testing.assert_close(exit_uv[64, 64], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(exit_uv[0, 0], torch.tensor([-1.0, -1.0]))


def test_spherical_padding_matches_rim_slope_and_does_not_wrap():
    radius = torch.tensor([0.99999, 1.0, 1.00001, 2.0, 10.0], dtype=torch.float64)
    half_fov = math.radians(170) / 2
    angles = bounded_angular_radius(radius, half_fov)
    assert (angles[1:] > angles[:-1]).all() and (angles < math.pi).all()
    left = (angles[1] - angles[0]) / (radius[1] - radius[0])
    right = (angles[2] - angles[1]) / (radius[2] - radius[1])
    torch.testing.assert_close(left, right, atol=2e-5, rtol=0)
