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


@pytest.fixture(params=test_devices())
def device(request):
    """Parametrize tests across CPU and CUDA devices."""
    return request.param


def _box_inside_analytic(points: torch.Tensor, region_pos: torch.Tensor, half) -> torch.Tensor:
    """Ground-truth containment for an axis-aligned box, shape (num_envs, num_particles)."""
    half_t = torch.tensor(half, device=points.device, dtype=torch.float32)
    # points: (E, P, 3); region_pos: (E, 3)
    local = points - region_pos.unsqueeze(1)
    return (local.abs() < half_t).all(dim=-1)


def _frustum_inside_analytic(points: torch.Tensor, r_b, r_t, z_b, z_t) -> torch.Tensor:
    """Ground-truth containment for a +Z frustum centered on the local axis."""
    z = points[..., 2]
    t = ((z - z_b) / (z_t - z_b)).clamp(0.0, 1.0)
    radius = r_b + t * (r_t - r_b)
    radial = torch.linalg.norm(points[..., :2], dim=-1)
    return (z > z_b) & (z < z_t) & (radial < radius)


class TestParticleMeshCounterBox:
    """Containment against an exact (non-discretized) box region mesh."""

    def test_box_counts_and_mask_match_analytic(self, device):
        """Random points against an offset box match the analytic ground truth exactly."""
        torch.manual_seed(0)
        num_envs, num_particles = 4, 512
        half = (0.1, 0.15, 0.08)
        region_pos_e = torch.tensor([0.2, -0.1, 0.05], device=device)
        # spread points well beyond the box so both inside and outside are represented
        points = (torch.rand(num_envs, num_particles, 3, device=device) - 0.5) * 0.8 + region_pos_e

        counter = ParticleMeshCounter([make_box_region_mesh(half)], num_envs=num_envs, device=device)
        region_pos = region_pos_e.expand(1, num_envs, 3)  # (num_regions=1, num_envs, 3)
        counts, mask = counter.count(points, region_pos, return_mask=True)

        expected_mask = _box_inside_analytic(points, region_pos_e.expand(num_envs, 3), half)
        assert mask.shape == (num_envs, num_particles, 1)
        assert mask.dtype == torch.bool
        assert torch.equal(mask[..., 0], expected_mask)
        assert torch.equal(counts[:, 0], expected_mask.sum(dim=1).float())
        # sanity: the box covers a non-trivial fraction of the points
        assert (counts[:, 0] > 0).all() and (counts[:, 0] < num_particles).all()

    def test_region_positions_broadcast_matches_explicit(self, device):
        """A (num_regions, 3) region position broadcasts identically to the per-env form."""
        num_envs, num_particles = 3, 64
        points = (torch.rand(num_envs, num_particles, 3, device=device) - 0.5) * 0.6
        counter = ParticleMeshCounter([make_box_region_mesh((0.1, 0.1, 0.1))], num_envs=num_envs, device=device)
        broadcast = counter.count(points, torch.zeros(1, 3, device=device)).clone()
        explicit = counter.count(points, torch.zeros(1, num_envs, 3, device=device)).clone()
        assert torch.equal(broadcast, explicit)


class TestParticleMeshCounterFrustum:
    """Containment against a capped circular frustum (cup cavity)."""

    def test_frustum_targeted_points(self, device):
        """Hand-picked points inside / outside a frustum are classified correctly."""
        r_b, r_t, z_b, z_t = 0.02, 0.04, -0.02, 0.03
        verts_faces = make_frustum_region_mesh(r_b, r_t, z_b, z_t, num_segments=48)
        counter = ParticleMeshCounter([verts_faces], num_envs=1, device=device)
        points = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],  # on-axis mid -> inside
                    [0.0, 0.0, z_b + 1e-3],  # just above floor -> inside
                    [0.03, 0.0, 0.02],  # within top radius -> inside
                    [0.05, 0.0, 0.0],  # beyond radius -> outside
                    [0.0, 0.0, z_t + 0.02],  # above top cap -> outside
                    [0.0, 0.0, z_b - 0.02],  # below bottom cap -> outside
                ]
            ],
            device=device,
        )
        _, mask = counter.count(points, torch.zeros(1, 1, 3, device=device), return_mask=True)
        assert mask[0, :, 0].int().tolist() == [1, 1, 1, 0, 0, 0]

    def test_frustum_matches_analytic_away_from_surface(self, device):
        """Random points (excluding a thin shell near the surface) match the analytic frustum."""
        torch.manual_seed(1)
        r_b, r_t, z_b, z_t = 0.02, 0.05, -0.03, 0.04
        counter = ParticleMeshCounter(
            [make_frustum_region_mesh(r_b, r_t, z_b, z_t, num_segments=64)], num_envs=1, device=device
        )
        pts = torch.zeros(1, 2000, 3, device=device)
        pts[0, :, 0] = (torch.rand(2000, device=device) - 0.5) * 0.16
        pts[0, :, 1] = (torch.rand(2000, device=device) - 0.5) * 0.16
        pts[0, :, 2] = (torch.rand(2000, device=device) - 0.5) * 0.16

        expected = _frustum_inside_analytic(pts, r_b, r_t, z_b, z_t)
        # exclude points within a small band of the lateral/cap surfaces (mesh is a 64-gon approx)
        z = pts[0, :, 2]
        t = ((z - z_b) / (z_t - z_b)).clamp(0.0, 1.0)
        radius = r_b + t * (r_t - r_b)
        radial = torch.linalg.norm(pts[0, :, :2], dim=-1)
        margin = 0.004
        near_surface = (
            (radial > radius - margin) & (radial < radius + margin)
            | (z > z_b - margin) & (z < z_b + margin)
            | (z > z_t - margin) & (z < z_t + margin)
        )
        keep = ~near_surface
        _, mask = counter.count(pts, torch.zeros(1, 1, 3, device=device), return_mask=True)
        assert torch.equal(mask[0, keep, 0], expected[0, keep])


class TestParticleMeshCounterTransforms:
    """Per-environment and rotated region transforms."""

    def test_multi_env_independent_transforms(self, device):
        """Each environment uses its own region transform."""
        counter = ParticleMeshCounter([make_box_region_mesh((0.1, 0.1, 0.1))], num_envs=2, device=device)
        # region at x=0 for env0, x=1 for env1
        region_pos = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0]]], device=device)
        points = torch.tensor(
            [
                [[0.05, 0, 0], [0.05, 0, 0], [0.5, 0, 0]],  # env0: in, in, out
                [[0.05, 0, 0], [1.05, 0, 0], [1.5, 0, 0]],  # env1: out, in, out
            ],
            device=device,
        )
        counts = counter.count(points, region_pos)
        assert counts[:, 0].tolist() == [2.0, 1.0]

    def test_rotated_region(self, device):
        """A thin box rotated 90 deg about Z excludes a point that was inside when axis-aligned."""
        counter = ParticleMeshCounter([make_box_region_mesh((0.3, 0.02, 0.02))], num_envs=1, device=device)
        point = torch.tensor([[[0.2, 0.0, 0.0]]], device=device)
        region_pos = torch.zeros(1, 1, 3, device=device)
        q_identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)
        q_z90 = torch.tensor([[[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]]], device=device)
        assert counter.count(point, region_pos, q_identity)[0, 0].item() == 1.0
        assert counter.count(point, region_pos, q_z90)[0, 0].item() == 0.0


class TestParticleMeshCounterMultiRegion:
    """Multiple region meshes per counter."""

    def test_disjoint_regions(self, device):
        """A point inside one region is not counted in a far-away region."""
        counter = ParticleMeshCounter(
            [make_box_region_mesh((0.1, 0.1, 0.1)), make_box_region_mesh((0.1, 0.1, 0.1))],
            num_envs=1,
            device=device,
        )
        region_pos = torch.tensor([[[0.0, 0, 0]], [[1.0, 0, 0]]], device=device)  # (2 regions, 1 env, 3)
        points = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]]], device=device)
        counts = counter.count(points, region_pos)
        assert counts.shape == (1, 2)
        assert counts[0].tolist() == [1.0, 1.0]
        assert counter.num_regions == 2


class TestParticleMeshCounterRobustness:
    """Buffer reuse, prebuilt meshes, and input validation."""

    def test_buffer_reuse_changing_particle_count(self, device):
        """The internal buffer resizes correctly when the particle count changes between calls."""
        counter = ParticleMeshCounter([make_box_region_mesh((0.1, 0.1, 0.1))], num_envs=1, device=device)
        region_pos = torch.zeros(1, 1, 3, device=device)
        small = torch.tensor([[[0.0, 0, 0], [0.5, 0, 0]]], device=device)
        assert counter.count(small, region_pos)[0, 0].item() == 1.0
        big = torch.tensor([[[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.5, 0, 0]]], device=device)
        assert counter.count(big, region_pos)[0, 0].item() == 3.0

    def test_prebuilt_warp_mesh_accepted(self, device):
        """A pre-built warp mesh can be passed directly."""
        verts, faces = make_box_region_mesh((0.1, 0.1, 0.1))
        wp_device = wp.device_from_torch(torch.device(device))
        mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=wp_device),
            indices=wp.array(faces.flatten(), dtype=wp.int32, device=wp_device),
            support_winding_number=True,
        )
        counter = ParticleMeshCounter([mesh], num_envs=1, device=device)
        points = torch.tensor([[[0.0, 0, 0], [0.5, 0, 0]]], device=device)
        assert counter.count(points, torch.zeros(1, 1, 3, device=device))[0, 0].item() == 1.0

    def test_invalid_inputs_raise(self):
        """Empty mesh list and malformed input shapes raise ValueError."""
        with pytest.raises(ValueError):
            ParticleMeshCounter([], num_envs=1, device="cpu")
        counter = ParticleMeshCounter([make_box_region_mesh((0.1, 0.1, 0.1))], num_envs=2, device="cpu")
        with pytest.raises(ValueError):
            counter.count(torch.zeros(2, 4), torch.zeros(1, 2, 3))  # particles not 3D
        with pytest.raises(ValueError):
            counter.count(torch.zeros(3, 4, 3), torch.zeros(1, 3, 3))  # wrong num_envs
        with pytest.raises(ValueError):
            counter.count(torch.zeros(2, 4, 3), torch.zeros(1, 5, 3))  # bad region shape
        with pytest.raises(ValueError):
            counter.count(torch.zeros(2, 4, 3), torch.zeros(5, 3))  # malformed 2-D region shape


class TestRegionMeshFactories:
    """Argument validation for the region-mesh factories."""

    @pytest.mark.parametrize(
        "factory, args, kwargs",
        [
            (make_frustum_region_mesh, (0.02, 0.04, -0.01, 0.03), {"num_segments": 2}),
            (make_box_region_mesh, ((0.1, 0.0, 0.1),), {}),
            (make_box_region_mesh, ((-0.1, 0.1, 0.1),), {}),
            (make_frustum_region_mesh, (0.0, 0.04, -0.01, 0.03), {}),
            (make_frustum_region_mesh, (0.02, 0.04, 0.03, -0.01), {}),
        ],
        ids=[
            "frustum_too_few_segments",
            "box_zero_half_extent",
            "box_negative_half_extent",
            "frustum_non_positive_radius",
            "frustum_inverted_z",
        ],
    )
    def test_factory_rejects_invalid_args(self, factory, args, kwargs):
        with pytest.raises(ValueError):
            factory(*args, **kwargs)
