# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for Franka Pour reset-bank tensor utilities."""

import math

import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_utils import (
    asymmetric_reset_offset_samples,
    polar_workspace_cells,
    reset_rotation_vector_samples,
    sample_index_pools,
    scale_randomization_rows_by_extent,
)


def test_scale_randomization_rows_preserves_balanced_rows_and_full_extent():
    rows = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.10, -0.04, 0.02],
            [-0.10, 0.04, -0.02],
        ]
    )
    extents = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75, 1.0)

    scaled = scale_randomization_rows_by_extent(rows, extents)

    assert scaled.shape == (len(extents), *rows.shape)
    for level, extent in enumerate(extents):
        torch.testing.assert_close(scaled[level], rows * extent)
        if extent == 0.0:
            torch.testing.assert_close(scaled[level], torch.zeros_like(rows), rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(torch.sign(scaled[level]), torch.sign(rows))
    torch.testing.assert_close(scaled[-1], rows, rtol=0.0, atol=0.0)


def test_reset_rotation_vectors_cover_angle_range_without_aligned_full_extent_sample():
    angle_range = (math.radians(15.0), math.radians(35.0))

    vectors = reset_rotation_vector_samples(angle_range, 11, dtype=torch.float64)
    magnitudes = torch.linalg.vector_norm(vectors, dim=-1)

    assert vectors.shape == (11, 3)
    assert vectors.dtype == torch.float64
    assert float(magnitudes.amin()) == pytest.approx(angle_range[0])
    assert float(magnitudes.amax()) == pytest.approx(angle_range[1])
    assert bool(torch.all(magnitudes > 0.0))
    assert int(torch.linalg.matrix_rank(vectors)) == 3
    torch.testing.assert_close(
        vectors,
        reset_rotation_vector_samples(angle_range, 11, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_reset_rotation_vectors_allow_disabled_rotation_randomization():
    vectors = reset_rotation_vector_samples((0.0, 0.0), 5)

    torch.testing.assert_close(vectors, torch.zeros_like(vectors), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "angle_range, sample_count, message",
    [
        ((0.0,), 5, "two values"),
        ((-0.1, 0.2), 5, "within"),
        ((0.3, 0.2), 5, "within"),
        ((0.0, math.pi + 0.1), 5, "within"),
        ((0.0, float("nan")), 5, "finite"),
        ((0.0, 0.2), 0, "positive integer"),
    ],
)
def test_reset_rotation_vectors_reject_invalid_configuration(angle_range, sample_count, message):
    with pytest.raises(ValueError, match=message):
        reset_rotation_vector_samples(angle_range, sample_count)


def test_polar_workspace_cells_cover_sector_and_preserve_exact_nominal_row():
    nominal = torch.tensor([0.50, 0.0, 0.02], dtype=torch.float64)
    radius_range = (0.48, 0.74)
    azimuth_range = math.radians(25.0)
    grid_size = 7

    cells = polar_workspace_cells(
        nominal,
        radius_range=radius_range,
        azimuth_half_range=azimuth_range,
        grid_size=grid_size,
    )

    assert cells.shape == (grid_size**2, 3)
    assert cells.dtype == nominal.dtype
    assert cells.device == nominal.device
    torch.testing.assert_close(cells[:, 2], nominal[2].expand(cells.shape[0]), rtol=0.0, atol=0.0)
    nominal_ring = grid_size // 2
    nominal_row = nominal_ring * grid_size + grid_size // 2
    torch.testing.assert_close(cells[nominal_row], nominal, rtol=0.0, atol=1.0e-12)

    radius = torch.linalg.vector_norm(cells[:, :2], dim=-1).reshape(grid_size, grid_size)
    azimuth = torch.atan2(cells[:, 1], cells[:, 0]).reshape(grid_size, grid_size)
    expected_radii = torch.cat(
        (
            torch.linspace(radius_range[0], float(nominal[0]), nominal_ring + 1, dtype=nominal.dtype)[:-1],
            torch.linspace(float(nominal[0]), radius_range[1], nominal_ring + 1, dtype=nominal.dtype),
        )
    )
    torch.testing.assert_close(radius, expected_radii[:, None].expand_as(radius))
    torch.testing.assert_close(radius[:, 0], radius[:, -1], rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(azimuth[0], azimuth[-1], rtol=0.0, atol=1.0e-12)
    assert bool(torch.all(radius[1:, 0] > radius[:-1, 0]))
    assert bool(torch.all(azimuth[0, 1:] > azimuth[0, :-1]))
    assert float(radius.amin()) == pytest.approx(radius_range[0])
    assert float(radius.amax()) == pytest.approx(radius_range[1])
    assert float(azimuth.amin()) == pytest.approx(-azimuth_range)
    assert float(azimuth.amax()) == pytest.approx(azimuth_range)

    # The task's conservative Cartesian half-extents must contain the complete polar sector.
    xy_offset = torch.abs(cells[:, :2] - nominal[:2])
    assert bool(torch.all(xy_offset <= nominal.new_tensor((0.25, 0.34)) + 1.0e-12))


def test_polar_workspace_cells_centers_azimuth_on_nonzero_nominal_bearing():
    nominal_azimuth = 0.4
    nominal = torch.tensor(
        (0.5 * math.cos(nominal_azimuth), 0.5 * math.sin(nominal_azimuth), 0.02),
        dtype=torch.float64,
    )

    cells = polar_workspace_cells(
        nominal,
        radius_range=(0.4, 0.7),
        azimuth_half_range=0.3,
        grid_size=3,
    )

    azimuth = torch.atan2(cells[:, 1], cells[:, 0]).reshape(3, 3)
    offsets = torch.atan2(
        torch.sin(azimuth - nominal_azimuth),
        torch.cos(azimuth - nominal_azimuth),
    )
    torch.testing.assert_close(offsets, nominal.new_tensor((-0.3, 0.0, 0.3)).expand_as(offsets))
    torch.testing.assert_close(
        cells,
        polar_workspace_cells(
            nominal,
            radius_range=(0.4, 0.7),
            azimuth_half_range=0.3,
            grid_size=3,
        ),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "nominal_radius, nominal_ring",
    [
        (0.4, 0),
        (0.8, 4),
    ],
)
def test_polar_workspace_cells_supports_nominal_radius_at_range_endpoint(nominal_radius, nominal_ring):
    grid_size = 5
    nominal = torch.tensor((nominal_radius, 0.0, 0.03), dtype=torch.float64)

    cells = polar_workspace_cells(
        nominal,
        radius_range=(0.4, 0.8),
        azimuth_half_range=0.2,
        grid_size=grid_size,
    )

    radii = torch.linalg.vector_norm(cells[:, :2], dim=-1).reshape(grid_size, grid_size)
    expected_radii = torch.linspace(0.4, 0.8, grid_size, dtype=nominal.dtype)
    torch.testing.assert_close(radii, expected_radii[:, None].expand_as(radii))
    nominal_row = nominal_ring * grid_size + grid_size // 2
    torch.testing.assert_close(cells[nominal_row], nominal, rtol=0.0, atol=0.0)
    assert torch.unique(cells, dim=0).shape[0] == cells.shape[0]


@pytest.mark.parametrize("grid_size", [2, 4, 3.0, True])
def test_polar_workspace_cells_rejects_invalid_grid_size(grid_size):
    with pytest.raises(ValueError, match="odd integer"):
        polar_workspace_cells(
            (0.5, 0.0, 0.0),
            radius_range=(0.4, 0.6),
            azimuth_half_range=0.2,
            grid_size=grid_size,
        )


@pytest.mark.parametrize(
    "nominal, radius_range, azimuth_half_range, message",
    [
        ((0.5, 0.0), (0.4, 0.6), 0.2, "three coordinates"),
        ((float("nan"), 0.0, 0.0), (0.4, 0.6), 0.2, "finite"),
        ((0.0, 0.0, 0.0), (0.4, 0.6), 0.2, "closed interval"),
        ((0.5, 0.0, 0.0), (0.4,), 0.2, "two values"),
        ((0.3, 0.0, 0.0), (0.4, 0.6), 0.2, "closed interval"),
        ((0.7, 0.0, 0.0), (0.4, 0.6), 0.2, "closed interval"),
        ((0.5, 0.0, 0.0), (-0.1, 0.6), 0.2, "strictly increasing"),
        ((0.5, 0.0, 0.0), (0.6, 0.4), 0.2, "strictly increasing"),
        ((0.5, 0.0, 0.0), (0.4, 0.6), 0.0, "azimuth_half_range"),
        ((0.5, 0.0, 0.0), (0.4, 0.6), math.pi, "azimuth_half_range"),
    ],
)
def test_polar_workspace_cells_rejects_invalid_geometry(nominal, radius_range, azimuth_half_range, message):
    with pytest.raises(ValueError, match=message):
        polar_workspace_cells(
            nominal,
            radius_range=radius_range,
            azimuth_half_range=azimuth_half_range,
            grid_size=3,
        )


def test_polar_workspace_cells_rejects_nonfloating_tensor_nominal():
    with pytest.raises(ValueError, match="floating-point dtype"):
        polar_workspace_cells(
            torch.tensor((1, 0, 0)),
            radius_range=(0.4, 1.2),
            azimuth_half_range=0.2,
            grid_size=3,
        )


def test_asymmetric_reset_offsets_include_bounds_zero_and_antithetic_interior():
    lower = torch.tensor((0.0, -0.10, 0.0), dtype=torch.float64)
    upper = torch.tensor((0.0, 0.10, 0.15), dtype=torch.float64)

    sample_count = 11
    offsets = asymmetric_reset_offset_samples(lower, upper, sample_count=sample_count)

    assert offsets.shape == (sample_count, 3)
    assert offsets.dtype == lower.dtype
    assert offsets.device == lower.device
    torch.testing.assert_close(offsets[0], torch.zeros_like(lower), rtol=0.0, atol=0.0)
    torch.testing.assert_close(offsets[1], lower, rtol=0.0, atol=0.0)
    torch.testing.assert_close(offsets[2], upper, rtol=0.0, atol=0.0)
    interior_pair_count = (sample_count - 3) // 2
    torch.testing.assert_close(
        offsets[3::2] + offsets[4::2],
        (lower + upper).expand(interior_pair_count, -1),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert bool(torch.all(offsets >= lower))
    assert bool(torch.all(offsets <= upper))
    torch.testing.assert_close(offsets.amin(dim=0), lower, rtol=0.0, atol=0.0)
    torch.testing.assert_close(offsets.amax(dim=0), upper, rtol=0.0, atol=0.0)

    active_axes = upper > lower
    normalized_interior = (offsets[3:, active_axes] - lower[active_axes]) / (upper[active_axes] - lower[active_axes])
    assert bool(torch.all((normalized_interior > 0.0) & (normalized_interior < 1.0)))
    torch.testing.assert_close(
        normalized_interior.reshape(-1, 2, int(active_axes.sum())).sum(dim=1),
        torch.ones((interior_pair_count, int(active_axes.sum())), dtype=lower.dtype),
    )
    expected_first = torch.frac(lower.new_tensor((0.754877666, 0.569840296, 0.438447187)) * 0.5)[active_axes]
    torch.testing.assert_close(normalized_interior[0], expected_first)
    torch.testing.assert_close(
        offsets,
        asymmetric_reset_offset_samples(lower, upper, sample_count),
        rtol=0.0,
        atol=0.0,
    )


def test_asymmetric_reset_offsets_support_minimum_count_and_degenerate_axes():
    offsets = asymmetric_reset_offset_samples(
        (-0.2, 0.0, 0.0),
        (0.0, 0.0, 0.15),
        3,
        dtype=torch.float64,
    )

    expected = torch.tensor(
        ((0.0, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.0, 0.15)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(offsets, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("sample_count", [1, 4, 5.0, True])
def test_asymmetric_reset_offsets_reject_invalid_sample_count(sample_count):
    with pytest.raises(ValueError, match="odd integer"):
        asymmetric_reset_offset_samples((-0.2, -0.1, 0.0), (0.0, 0.1, 0.15), sample_count)


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        ((-0.2, -0.1), (0.0, 0.1, 0.15), "three coordinates"),
        ((-0.2, -0.1, 0.0), (0.0, 0.1), "three coordinates"),
        ((float("nan"), -0.1, 0.0), (0.0, 0.1, 0.15), "finite"),
        ((0.01, -0.1, 0.0), (0.1, 0.1, 0.15), "contain zero"),
        ((-0.2, -0.1, 0.0), (0.0, -0.01, 0.15), "contain zero"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "positive width"),
    ],
)
def test_asymmetric_reset_offsets_reject_invalid_bounds(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        asymmetric_reset_offset_samples(lower, upper, 5)


def test_asymmetric_reset_offsets_reject_nonfloating_or_mismatched_tensor_bounds():
    with pytest.raises(ValueError, match="floating-point dtype"):
        asymmetric_reset_offset_samples(torch.tensor((-1, 0, 0)), torch.tensor((0, 1, 1)), 5)
    with pytest.raises(ValueError, match="same dtype"):
        asymmetric_reset_offset_samples(
            torch.tensor((-0.2, -0.1, 0.0), dtype=torch.float32),
            torch.tensor((0.0, 0.1, 0.15), dtype=torch.float64),
            5,
        )


def test_weighted_index_pool_sampling_is_repeatable_from_torch_seed():
    pools = (torch.tensor([4, 8, 12]), torch.tensor([1, 5, 9, 13]))
    weights = (torch.tensor([1.0, 2.0, 1.0]), torch.tensor([1.0, 1.0, 3.0, 1.0]))
    pool_ids = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])

    with torch.random.fork_rng():
        torch.manual_seed(1234)
        first = sample_index_pools(pools, pool_ids, weights=weights)
        torch.manual_seed(1234)
        second = sample_index_pools(pools, pool_ids, weights=weights)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "extents, message",
    [
        ((), "must not be empty"),
        ((-0.1, 1.0), "values in"),
        ((0.8, 0.7, 1.0), "strictly increasing"),
        ((0.5, 0.8), "must end at 1.0"),
    ],
)
def test_scale_randomization_rows_rejects_invalid_extent_levels(extents, message):
    with pytest.raises(ValueError, match=message):
        scale_randomization_rows_by_extent(torch.ones((3, 2)), extents)


def test_scale_randomization_rows_rejects_nonfinite_offsets():
    with pytest.raises(ValueError, match="finite floating-point"):
        scale_randomization_rows_by_extent(torch.tensor([[0.0, float("nan")]]), (1.0,))


def test_scale_randomization_rows_normalizes_tolerant_final_extent_to_full_amplitude():
    rows = torch.tensor([[0.10, -0.04]])

    scaled = scale_randomization_rows_by_extent(rows, (0.5, 1.0 - 5.0e-10))

    torch.testing.assert_close(scaled[-1], rows, rtol=0.0, atol=0.0)


def test_weighted_index_pool_sampling_uses_multinomial_and_maps_global_rows(monkeypatch):
    pools = (torch.tensor([4, 8]), torch.tensor([1, 5, 9]))
    weights = (torch.tensor([0.75, 0.25]), torch.tensor([0.0, 1.0, 3.0]))
    pool_ids = torch.tensor([0, 1, 1, 0])
    sampled_slots = iter((torch.tensor([1, 0]), torch.tensor([2, 1])))
    calls = []

    def sample_weighted(pool_weights, count, replacement):
        calls.append((pool_weights.clone(), count, replacement))
        return next(sampled_slots)

    monkeypatch.setattr(torch, "multinomial", sample_weighted)
    sampled = sample_index_pools(pools, pool_ids, weights=weights)

    assert sampled.tolist() == [8, 9, 5, 4]
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], weights[0])
    torch.testing.assert_close(calls[1][0], weights[1])
    assert [(count, replacement) for _, count, replacement in calls] == [(2, True), (2, True)]


@pytest.mark.parametrize(
    "weights, message",
    [
        ((), "one tensor per index pool"),
        ((torch.ones(3),), "shape and device"),
        ((torch.ones(2, dtype=torch.int64),), "finite, nonnegative floating-point"),
        ((torch.tensor([1.0, float("nan")]),), "finite, nonnegative floating-point"),
        ((torch.tensor([1.0, -0.1]),), "finite, nonnegative floating-point"),
        ((torch.zeros(2),), "positive sum"),
    ],
)
def test_weighted_index_pool_sampling_rejects_invalid_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        sample_index_pools((torch.tensor([4, 8]),), torch.tensor([0]), weights=weights)


def test_weighted_index_pool_sampling_rejects_weight_device_mismatch():
    with pytest.raises(ValueError, match="shape and device"):
        sample_index_pools(
            (torch.tensor([4, 8]),),
            torch.tensor([0]),
            weights=(torch.ones(2, device="meta"),),
        )
