# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the warp ray-casting wrappers in :mod:`isaaclab.utils.warp.ops`."""

import numpy as np
import pytest
import torch

from isaaclab.utils.warp.ops import convert_to_warp_mesh, raycast_mesh

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def plane_mesh():
    """A unit square in the z=0 plane, built from two triangles."""
    points = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    indices = np.array([[0, 1, 2], [0, 2, 3]])
    return convert_to_warp_mesh(points, indices, "cpu")


@pytest.mark.parametrize("ray_shape", [(2,), (1, 2), (3, 1, 2), (2, 3, 1)])
def test_raycast_mesh_preserves_ray_dimensions(plane_mesh, ray_shape):
    """Every output keeps the ray dimensions of the input, for any number of them."""
    num_rays = int(np.prod(ray_shape))
    # Give every ray a different start height so the outputs also pin down the ordering.
    depths = torch.arange(1, num_rays + 1, dtype=torch.float32)
    ray_starts = torch.zeros(num_rays, 3)
    # Aim inside one triangle rather than at the diagonal the two triangles share.
    ray_starts[:, 0] = 0.5
    ray_starts[:, 1] = -0.25
    ray_starts[:, 2] = -depths
    ray_starts = ray_starts.view(*ray_shape, 3)
    ray_directions = torch.tensor([0.0, 0.0, 1.0]).repeat(num_rays, 1).view(*ray_shape, 3)

    ray_hits, ray_distance, ray_normal, ray_face_id = raycast_mesh(
        ray_starts,
        ray_directions,
        plane_mesh,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )

    assert ray_hits.shape == (*ray_shape, 3)
    assert ray_normal.shape == (*ray_shape, 3)
    assert ray_distance.shape == ray_shape
    assert ray_face_id.shape == ray_shape
    torch.testing.assert_close(ray_distance, depths.view(ray_shape))


@pytest.mark.parametrize("direction_shape", [(2, 3), (5, 3), (4, 2), (2, 2, 3)])
def test_raycast_mesh_rejects_direction_shape_mismatch(plane_mesh, direction_shape):
    """Directions that do not match the ray starts are rejected rather than reinterpreted."""
    ray_starts = torch.zeros(4, 3)
    ray_directions = torch.zeros(direction_shape)

    with pytest.raises(ValueError, match="same shape as ray_starts"):
        raycast_mesh(ray_starts, ray_directions, plane_mesh)
