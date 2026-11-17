# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests of sensor ray layout and the fixed optical center."""

import pytest
import torch
import warp as wp
from isaaclab_fovi.sensors.foveated_rays import compute_explicit_ray_array


@pytest.fixture(scope="module", autouse=True)
def initialize_warp() -> None:
    wp.init()


def test_explicit_rays_preserve_sample_order_and_optical_center() -> None:
    directions = torch.nn.functional.normalize(
        torch.tensor([[0.7, 0.3, -1.0], [-0.2, 0.6, -1.0], [0.0, 0.0, -1.0]]), dim=-1
    )
    rays = compute_explicit_ray_array(directions, device="cpu")
    assert rays.shape == (1, 3, 1, 2)
    assert rays.dtype == wp.vec3f
    rays = wp.to_torch(rays)[0, :, 0]
    torch.testing.assert_close(rays[:, 0], torch.zeros_like(directions))
    torch.testing.assert_close(rays[:, 1], directions)


def test_explicit_rays_reject_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_explicit_ray_array(torch.zeros(3, 2), device="cpu")
