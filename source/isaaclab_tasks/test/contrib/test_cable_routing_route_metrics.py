# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral contracts for cable-route geometry metrics."""

import torch

from isaaclab_tasks.contrib.cable_routing.mdp.route_metrics import (
    benchmark_local_cable_spans,
    benchmark_winding_angle,
    ordered_route_state,
)


def _planar_cable(angles: torch.Tensor, radii: torch.Tensor | float = 0.5) -> torch.Tensor:
    radii = torch.as_tensor(radii, dtype=angles.dtype).expand_as(angles)
    points = torch.zeros(angles.numel(), 3, dtype=angles.dtype)
    points[:, 0] = radii * torch.cos(angles)
    points[:, 1] = radii * torch.sin(angles)
    return points


def test_benchmark_metrics_enforce_local_single_span_geometry() -> None:
    """Winding is directed, local, axially bounded, and split into connected spans."""
    clockwise = _planar_cable(torch.linspace(0.0, -2.0 * torch.pi, 9))
    counterclockwise = _planar_cable(torch.linspace(0.0, 2.0 * torch.pi, 9))
    winding = benchmark_winding_angle(
        torch.stack((clockwise, counterclockwise)),
        torch.zeros(2, 1, 3),
        radial_cutoff=0.6,
        axial_cutoff=0.1,
    )
    torch.testing.assert_close(
        winding[:, 0],
        torch.tensor((2.0 * torch.pi, -2.0 * torch.pi)),
        atol=1.0e-5,
        rtol=0.0,
    )

    angles = torch.linspace(0.0, -2.0 * torch.pi, 5)
    split_loop = _planar_cable(angles, torch.tensor((0.5, 0.5, 1.5, 0.5, 0.5)))[None]
    split_winding = benchmark_winding_angle(
        split_loop,
        torch.zeros(1, 1, 3),
        radial_cutoff=1.0,
        axial_cutoff=0.1,
    )
    span_count, local_length = benchmark_local_cable_spans(
        split_loop,
        torch.zeros(1, 1, 3),
        radial_cutoff=1.0,
        axial_cutoff=0.1,
    )
    torch.testing.assert_close(split_winding, torch.tensor(((torch.pi,),)), atol=1.0e-5, rtol=0.0)
    assert torch.equal(span_count, torch.tensor(((2,),)))
    torch.testing.assert_close(local_length, torch.tensor(((2.0**0.5,),)), atol=1.0e-6, rtol=0.0)

    axial_loops = clockwise.repeat(3, 1, 1)
    axial_loops[1, :, 2] = 0.11
    axial_loops[2, :, 2] = -0.11
    axial_winding = benchmark_winding_angle(
        axial_loops,
        torch.zeros(3, 1, 3),
        radial_cutoff=0.6,
        axial_cutoff=0.1,
    )
    axial_spans, _ = benchmark_local_cable_spans(
        axial_loops,
        torch.zeros(3, 1, 3),
        radial_cutoff=0.6,
        axial_cutoff=0.1,
    )
    torch.testing.assert_close(
        axial_winding[:, 0],
        torch.tensor((2.0 * torch.pi, 0.0, 0.0)),
        atol=1.0e-5,
        rtol=0.0,
    )
    assert torch.equal(axial_spans[:, 0], torch.tensor((1, 0, 0)))


def test_ordered_route_state_requires_contiguous_single_wraps() -> None:
    """Later wraps cannot skip a step, and bundles or over-wraps cannot complete it."""
    threshold = torch.pi
    winding = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (threshold, 0.0, threshold),
            (threshold, -threshold, 0.0),
            (threshold, -threshold, threshold),
        )
    )
    peg_indices = torch.tensor(((0, 1, 2, -1),)).expand(4, -1)
    directions = torch.tensor(((1, -1, 1, 1),)).expand(4, -1)
    valid_steps = torch.tensor(((True, True, True, False),)).expand(4, -1)
    progress, completed, prefix, success = ordered_route_state(
        winding,
        peg_indices,
        directions,
        valid_steps,
        completion_threshold=threshold,
    )
    assert torch.equal(prefix, torch.tensor((0, 1, 2, 3)))
    assert torch.equal(success, torch.tensor((False, False, False, True)))
    assert torch.equal(completed[1], torch.tensor((True, False, True, False)))
    assert progress[1, 3] == 0.0

    _, completed, prefix, success = ordered_route_state(
        torch.tensor(((3.0, 3.0, 2.0 * torch.pi + 0.5),)),
        torch.tensor(((0, 1, 2),)),
        torch.ones(1, 3),
        torch.ones(1, 3, dtype=torch.bool),
        completion_threshold=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        completion_mask=torch.tensor(((True, False, True),)),
    )
    assert torch.equal(completed, torch.tensor(((True, False, False),)))
    assert torch.equal(prefix, torch.tensor((1,)))
    assert not bool(success[0])
