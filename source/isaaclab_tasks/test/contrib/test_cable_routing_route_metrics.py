# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ordered cable-route geometry metrics."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.contrib.cable_routing.mdp.route_metrics import (
    benchmark_local_cable_spans,
    benchmark_winding_angle,
    ordered_route_state,
)


def _planar_cable(angles: torch.Tensor, radii: torch.Tensor | float = 0.5) -> torch.Tensor:
    """Build one ordered planar cable from polar samples about the origin."""
    radii = torch.as_tensor(radii, dtype=angles.dtype, device=angles.device).expand_as(angles)
    points = torch.zeros(angles.numel(), 3, dtype=angles.dtype, device=angles.device)
    points[:, 0] = radii * torch.cos(angles)
    points[:, 1] = radii * torch.sin(angles)
    return points


def test_benchmark_winding_is_clockwise_positive_and_counterclockwise_negative() -> None:
    clockwise = _planar_cable(torch.linspace(0.0, -2.0 * torch.pi, 9))
    counterclockwise = _planar_cable(torch.linspace(0.0, 2.0 * torch.pi, 9))
    cable_points = torch.stack((clockwise, counterclockwise))
    peg_positions = torch.zeros(2, 1, 3)

    winding = benchmark_winding_angle(cable_points, peg_positions, radial_cutoff=0.6, axial_cutoff=0.1)

    torch.testing.assert_close(winding[:, 0], torch.tensor([2.0 * torch.pi, -2.0 * torch.pi]), atol=1.0e-5, rtol=0.0)


def test_benchmark_winding_counts_only_edges_whose_endpoints_are_local() -> None:
    angles = torch.linspace(0.0, -2.0 * torch.pi, 5)
    # The middle point is outside the cutoff. Its two adjacent edges must both be excluded,
    # leaving only two clockwise quarter-circle edges in the accumulated winding.
    cable_points = _planar_cable(angles, torch.tensor([0.5, 0.5, 1.5, 0.5, 0.5]))[None]
    peg_positions = torch.zeros(1, 1, 3)

    local_winding = benchmark_winding_angle(cable_points, peg_positions, radial_cutoff=1.0, axial_cutoff=0.1)
    no_local_edges = benchmark_winding_angle(cable_points, peg_positions, radial_cutoff=0.49, axial_cutoff=0.1)

    torch.testing.assert_close(local_winding, torch.tensor([[torch.pi]]), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(no_local_edges, torch.zeros_like(no_local_edges))


def test_benchmark_metrics_reject_loops_outside_the_peg_axial_span() -> None:
    """A matching XY projection must not route a peg from above or below it."""
    loop = _planar_cable(torch.linspace(0.0, -2.0 * torch.pi, 9), radii=0.5)
    cable_points = loop.repeat(3, 1, 1)
    cable_points[1, :, 2] = 0.11
    cable_points[2, :, 2] = -0.11
    peg_positions = torch.zeros(3, 1, 3)

    winding = benchmark_winding_angle(
        cable_points,
        peg_positions,
        radial_cutoff=0.6,
        axial_cutoff=0.1,
    )
    span_count, local_length = benchmark_local_cable_spans(
        cable_points,
        peg_positions,
        radial_cutoff=0.6,
        axial_cutoff=0.1,
    )

    torch.testing.assert_close(winding[:, 0], torch.tensor([2.0 * torch.pi, 0.0, 0.0]), atol=1.0e-5, rtol=0.0)
    assert torch.equal(span_count[:, 0], torch.tensor([1, 0, 0]))
    assert float(local_length[0, 0]) > 0.0
    torch.testing.assert_close(local_length[1:, 0], torch.zeros(2))


@pytest.mark.parametrize("axial_cutoff", (0.0, -0.1, float("nan"), float("inf")))
def test_benchmark_metrics_reject_invalid_axial_cutoffs(axial_cutoff: float) -> None:
    cable_points = _planar_cable(torch.tensor((0.0, 0.5)))[None]
    peg_positions = torch.zeros(1, 1, 3)

    with pytest.raises(ValueError, match="axial_cutoff"):
        benchmark_winding_angle(cable_points, peg_positions, radial_cutoff=1.0, axial_cutoff=axial_cutoff)
    with pytest.raises(ValueError, match="axial_cutoff"):
        benchmark_local_cable_spans(cable_points, peg_positions, radial_cutoff=1.0, axial_cutoff=axial_cutoff)


def test_benchmark_local_cable_spans_separates_disconnected_bundle_strands() -> None:
    angles = torch.tensor([0.0, -0.5 * torch.pi, -torch.pi, -1.5 * torch.pi, -2.0 * torch.pi])
    cable_points = _planar_cable(angles, torch.tensor([0.5, 0.5, 1.5, 0.5, 0.5]))[None]
    peg_positions = torch.zeros(1, 1, 3)

    span_count, local_length = benchmark_local_cable_spans(
        cable_points,
        peg_positions,
        radial_cutoff=1.0,
        axial_cutoff=0.1,
    )

    assert torch.equal(span_count, torch.tensor([[2]]))
    torch.testing.assert_close(local_length, torch.tensor([[2.0**0.5]]), atol=1.0e-6, rtol=0.0)


def test_ordered_route_prefix_advances_one_contiguous_step_at_a_time() -> None:
    threshold = torch.pi
    winding = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [threshold, 0.0, threshold],
            [threshold, -threshold, 0.0],
            [threshold, -threshold, threshold],
        ]
    )
    peg_indices = torch.tensor([[0, 1, 2, -1]]).expand(4, -1)
    directions = torch.tensor([[1, -1, 1, 1]]).expand(4, -1)
    valid_steps = torch.tensor([[True, True, True, False]]).expand(4, -1)

    progress, completed, prefix_length, success = ordered_route_state(
        winding,
        peg_indices,
        directions,
        valid_steps,
        completion_threshold=threshold,
    )

    torch.testing.assert_close(prefix_length, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(success, torch.tensor([False, False, False, True]))
    # A later completed peg cannot jump over an incomplete earlier route step.
    assert torch.equal(completed[1], torch.tensor([True, False, True, False]))
    assert progress[1, 3] == 0.0


def test_ordered_route_success_requires_every_valid_step_in_order() -> None:
    threshold = torch.pi
    winding = torch.tensor(
        [
            [threshold, threshold],
            [0.0, threshold],
            [threshold, -threshold],
        ]
    )
    peg_indices = torch.tensor([[0, 1, -1]]).expand(3, -1)
    directions = torch.tensor([[1, 1, 1]]).expand(3, -1)
    valid_steps = torch.tensor([[True, True, False]]).expand(3, -1)

    _, completed, prefix_length, success = ordered_route_state(
        winding,
        peg_indices,
        directions,
        valid_steps,
        completion_threshold=threshold,
    )

    assert torch.equal(completed[:, :2], torch.tensor([[True, True], [False, True], [True, False]]))
    assert torch.equal(prefix_length, torch.tensor([2, 0, 1]))
    assert torch.equal(success, torch.tensor([True, False, False]))


def test_ordered_route_completion_rejects_bundles_and_multiple_wraps() -> None:
    completion_threshold = 2.6
    maximum_winding = 2.0 * torch.pi + 0.25
    winding = torch.tensor([[3.0, 3.0, 2.0 * torch.pi + 0.5]])
    peg_indices = torch.tensor([[0, 1, 2]])
    directions = torch.ones_like(winding)
    valid_steps = torch.ones_like(winding, dtype=torch.bool)
    single_span = torch.tensor([[True, False, True]])

    _, completed, prefix_length, success = ordered_route_state(
        winding,
        peg_indices,
        directions,
        valid_steps,
        completion_threshold,
        maximum_completion_winding=maximum_winding,
        completion_mask=single_span,
    )

    assert torch.equal(completed, torch.tensor([[True, False, False]]))
    assert torch.equal(prefix_length, torch.tensor([1]))
    assert not bool(success[0])
