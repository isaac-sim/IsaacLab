# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests of sensor ray layout and the fixed optical center."""

import math

import pytest
import torch
import warp as wp

from isaaclab_fovi.sensors.foveated_rays import (
    compute_explicit_ray_array,
    compute_foveated_ray_array,
)


@pytest.fixture(scope="module", autouse=True)
def initialize_warp() -> None:
    wp.init()


@pytest.mark.parametrize("projection", ["pinhole", "fisheye_equidistant"])
def test_ray_layout_and_projection(projection: str) -> None:
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, -0.5], [-0.4, 0.3]])
    field_of_view = math.radians(90.0)
    rays = compute_foveated_ray_array(coordinates, field_of_view, "cpu", projection)
    actual = wp.to_torch(rays)[0, :, 0]

    assert rays.shape == (1, 4, 1, 2)
    assert rays.dtype == wp.vec3f
    torch.testing.assert_close(actual[:, 0], torch.zeros(4, 3))
    torch.testing.assert_close(actual[:, 1].norm(dim=-1), torch.ones(4))
    torch.testing.assert_close(actual[0, 1], torch.tensor([0.0, 0.0, -1.0]))
    if projection == "pinhole":
        expected = torch.cat((coordinates, -torch.ones(4, 1)), dim=-1)
        expected = torch.nn.functional.normalize(expected, dim=-1)
        torch.testing.assert_close(actual[:, 1], expected)
    else:
        # Equidistant projection maps eccentricity linearly to optical-axis angle.
        actual_angle = torch.acos(-actual[:, 1, 2])
        torch.testing.assert_close(actual_angle, coordinates.norm(dim=-1) * field_of_view / 2)


def test_explicit_rays_preserve_sample_order_and_optical_center() -> None:
    directions = torch.nn.functional.normalize(
        torch.tensor([[0.7, 0.3, -1.0], [-0.2, 0.6, -1.0], [0.0, 0.0, -1.0]]), dim=-1
    )
    rays = wp.to_torch(compute_explicit_ray_array(directions, device="cpu"))[0, :, 0]
    torch.testing.assert_close(rays[:, 0], torch.zeros_like(directions))
    torch.testing.assert_close(rays[:, 1], directions)


def test_explicit_rays_reject_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_explicit_ray_array(torch.zeros(3, 2), device="cpu")


def test_unknown_projection_fails() -> None:
    with pytest.raises(ValueError, match="Unknown projection"):
        compute_foveated_ray_array(torch.zeros(1, 2), 1.0, "cpu", "unknown")
