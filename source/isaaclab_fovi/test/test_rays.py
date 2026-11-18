# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests of Newton's foveated ray array: sample order, the fixed optical center, and tile padding."""

import pytest
import torch
import warp as wp
from isaaclab_fovi.sensors.foveated_newton_camera import explicit_rays

pytestmark = pytest.mark.kitless


@pytest.fixture(scope="module", autouse=True)
def initialize_warp() -> None:
    wp.init()


def test_explicit_rays_preserve_sample_order_and_optical_center() -> None:
    directions = torch.nn.functional.normalize(
        torch.tensor([[0.7, 0.3, -1.0], [-0.2, 0.6, -1.0], [0.0, 0.0, -1.0]]), dim=-1
    )
    rays = explicit_rays(directions, 3, 1)
    assert rays.shape == (3, 1, 1, 2)
    assert rays.dtype == wp.vec3f
    rays = wp.to_torch(rays)[:, 0, 0]
    torch.testing.assert_close(rays[:, 0], torch.zeros_like(directions))
    torch.testing.assert_close(rays[:, 1], directions)


def test_explicit_rays_pad_whole_tiles_with_the_last_ray() -> None:
    directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    rays = wp.to_torch(explicit_rays(directions, 2, 4)).reshape(8, 2, 3)
    torch.testing.assert_close(rays[:5, 1], directions)
    torch.testing.assert_close(rays[5:, 1], directions[-1].expand(3, -1))
    assert not rays[:, 0].any()


def test_explicit_rays_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        explicit_rays(torch.zeros(3, 2), 3, 1)
    with pytest.raises(ValueError, match="cannot hold"):
        explicit_rays(torch.zeros(5, 3), 2, 2)
