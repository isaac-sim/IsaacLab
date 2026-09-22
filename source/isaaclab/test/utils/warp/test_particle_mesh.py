# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ParticleMeshCounter particle-in-mesh utility."""

import math

import pytest
import torch
import warp as wp

from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import ParticleMeshCounter, make_box_region_mesh, make_frustum_region_mesh

pytestmark = pytest.mark.unit

_UNIT_BOX = (0.1, 0.1, 0.1)


@pytest.fixture(params=test_devices())
def device(request):
    return request.param


def _box_counter(device, num_envs=1, half=_UNIT_BOX):
    return ParticleMeshCounter([make_box_region_mesh(half)], num_envs=num_envs, device=device)


def test_box_counts_match_analytic_containment(device):
    """Random points against an offset box match the analytic ground truth exactly, for both position layouts."""
    torch.manual_seed(0)
    num_envs, num_particles = 4, 256
    half = torch.tensor([0.1, 0.15, 0.08], device=device)
    region_pos = torch.tensor([0.2, -0.1, 0.05], device=device)
    points = (torch.rand(num_envs, num_particles, 3, device=device) - 0.5) * 0.8 + region_pos
    expected_mask = ((points - region_pos).abs() < half).all(dim=-1)
    assert 0 < expected_mask.sum() < expected_mask.numel()

    counter = ParticleMeshCounter([make_box_region_mesh(half.tolist())], num_envs=num_envs, device=device)
    counts, mask = counter.count(points, region_pos.expand(1, num_envs, 3), return_mask=True)
    assert mask.dtype == torch.bool and mask.shape == (num_envs, num_particles, 1)
    assert torch.equal(mask[..., 0], expected_mask)
    assert torch.equal(counts[:, 0], expected_mask.sum(dim=1).float())
    # a (num_regions, 3) position broadcasts to every environment
    assert torch.equal(counter.count(points, region_pos.expand(1, 3)), counts)


def test_frustum_matches_analytic_away_from_surface(device):
    """Points off the discretized frustum surface match the analytic cup volume; targeted points are classified."""
    torch.manual_seed(1)
    r_b, r_t, z_b, z_t = 0.02, 0.05, -0.03, 0.04
    counter = ParticleMeshCounter(
        [make_frustum_region_mesh(r_b, r_t, z_b, z_t, num_segments=64)], num_envs=1, device=device
    )
    targeted = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, z_b + 1e-3], [0.04, 0.0, 0.03], [0.06, 0.0, 0.0], [0.0, 0.0, z_t + 0.02]]],
        device=device,
    )
    _, mask = counter.count(targeted, torch.zeros(1, 1, 3, device=device), return_mask=True)
    assert mask[0, :, 0].int().tolist() == [1, 1, 1, 0, 0]

    pts = (torch.rand(1, 512, 3, device=device) - 0.5) * 0.16
    z = pts[0, :, 2]
    radius = r_b + ((z - z_b) / (z_t - z_b)).clamp(0.0, 1.0) * (r_t - r_b)
    radial = torch.linalg.norm(pts[0, :, :2], dim=-1)
    expected = (z > z_b) & (z < z_t) & (radial < radius)
    margin = 0.004  # the 64-gon approximates the circle; skip a thin shell around every surface
    near_surface = ((radial - radius).abs() < margin) | ((z - z_b).abs() < margin) | ((z - z_t).abs() < margin)
    _, mask = counter.count(pts, torch.zeros(1, 1, 3, device=device), return_mask=True)
    assert torch.equal(mask[0, ~near_surface, 0], expected[~near_surface])


def test_per_env_transforms_and_multiple_regions(device):
    """Each environment applies its own region pose; each region is counted independently."""
    counter = ParticleMeshCounter([make_box_region_mesh(_UNIT_BOX)] * 2, num_envs=2, device=device)
    assert counter.num_regions == 2 and counter.num_envs == 2 and counter.device == device
    # region 0 at x=0 (env 0) / x=1 (env 1); region 1 at x=5 in both envs
    region_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], device=device)
    points = torch.tensor(
        [[[0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [5.0, 0.0, 0.0]], [[0.05, 0.0, 0.0], [1.05, 0.0, 0.0], [1.5, 0.0, 0.0]]],
        device=device,
    )
    assert counter.count(points, region_pos).tolist() == [[2.0, 1.0], [1.0, 0.0]]

    # a thin box rotated 90 degrees about Z no longer contains a point on its long axis
    thin = ParticleMeshCounter([make_box_region_mesh((0.3, 0.02, 0.02))], num_envs=1, device=device)
    point = torch.tensor([[[0.2, 0.0, 0.0]]], device=device)
    origin = torch.zeros(1, 1, 3, device=device)
    q_identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)
    q_z90 = torch.tensor([[[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]]], device=device)
    assert thin.count(point, origin, q_identity).item() == 1.0
    assert thin.count(point, origin, q_z90).item() == 0.0


def test_buffer_resize_and_prebuilt_mesh(device):
    """The containment buffer follows the particle count and pre-built warp meshes are accepted."""
    verts, faces = make_box_region_mesh(_UNIT_BOX)
    wp_device = wp.device_from_torch(torch.device(device))
    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=wp_device),
        indices=wp.array(faces.flatten(), dtype=wp.int32, device=wp_device),
        support_winding_number=True,
    )
    counter = ParticleMeshCounter([mesh], num_envs=1, device=device)
    origin = torch.zeros(1, 1, 3, device=device)
    assert counter.count(torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], device=device), origin).item() == 1.0
    bigger = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.5, 0.0, 0.0]]], device=device)
    assert counter.count(bigger, origin).item() == 3.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ParticleMeshCounter([], num_envs=1, device="cpu")
    counter = _box_counter("cpu", num_envs=2)
    for particles, regions in [
        (torch.zeros(2, 4), torch.zeros(1, 2, 3)),  # particles not 3D
        (torch.zeros(3, 4, 3), torch.zeros(1, 3, 3)),  # wrong num_envs
        (torch.zeros(2, 4, 3), torch.zeros(1, 5, 3)),  # bad region shape
        (torch.zeros(2, 4, 3), torch.zeros(5, 3)),  # malformed 2-D region shape
    ]:
        with pytest.raises(ValueError):
            counter.count(particles, regions)


def test_region_mesh_factories():
    verts, faces = make_box_region_mesh((0.1, 0.2, 0.3), center=(1.0, 0.0, 0.0))
    assert verts.shape == (8, 3) and faces.shape == (12, 3)
    assert verts.min(axis=0).tolist() == pytest.approx([0.9, -0.2, -0.3])
    assert verts.max(axis=0).tolist() == pytest.approx([1.1, 0.2, 0.3])
    n = 16
    verts, faces = make_frustum_region_mesh(0.02, 0.04, -0.01, 0.03, num_segments=n)
    assert verts.shape == (2 * n + 2, 3) and faces.shape == (4 * n, 3)
    for bad_call in (
        lambda: make_box_region_mesh((0.1, 0.0, 0.1)),
        lambda: make_box_region_mesh((-0.1, 0.1, 0.1)),
        lambda: make_frustum_region_mesh(0.02, 0.04, -0.01, 0.03, num_segments=2),
        lambda: make_frustum_region_mesh(0.0, 0.04, -0.01, 0.03),
        lambda: make_frustum_region_mesh(0.02, 0.04, 0.03, -0.01),
    ):
        with pytest.raises(ValueError):
            bad_call()
