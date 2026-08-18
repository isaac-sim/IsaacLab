# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry utilities for ordered cable-routing goals."""

from __future__ import annotations

import math

import torch


def _peg_relative_xy_and_local_points(
    cable_points_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    radial_cutoff: float,
    axial_cutoff: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return peg-relative XY coordinates and the local-point mask."""
    if cable_points_w.ndim != 3 or cable_points_w.shape[-1] != 3:
        raise ValueError(f"Expected cable_points_w shape (N, S, 3), got {tuple(cable_points_w.shape)}.")
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"Expected peg_positions_w shape (N, P, 3), got {tuple(peg_positions_w.shape)}.")
    if cable_points_w.shape[0] != peg_positions_w.shape[0]:
        raise ValueError("Cable and peg tensors must contain the same number of environments.")
    if cable_points_w.shape[1] < 2:
        raise ValueError("At least two ordered cable points are required to evaluate peg-local edges.")
    if not math.isfinite(radial_cutoff) or radial_cutoff <= 0.0:
        raise ValueError("radial_cutoff must be finite and positive.")
    if axial_cutoff is not None and (not math.isfinite(axial_cutoff) or axial_cutoff <= 0.0):
        raise ValueError("axial_cutoff must be None or finite and positive.")

    relative_xy = cable_points_w[:, None, :, :2] - peg_positions_w[:, :, None, :2]
    local = torch.linalg.vector_norm(relative_xy, dim=-1) <= radial_cutoff
    if axial_cutoff is not None:
        relative_z = cable_points_w[:, None, :, 2] - peg_positions_w[:, :, None, 2]
        local &= relative_z.abs() <= axial_cutoff
    return relative_xy, local


def benchmark_local_cable_spans(
    cable_points_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    radial_cutoff: float,
    axial_cutoff: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure contiguous cable spans local to each peg.

    A valid single wrap should enter the peg neighborhood once, follow one contiguous
    section of cable, and leave once. Counting local spans prevents several disconnected
    strands from a nearby cable bundle from accumulating into one successful winding.

    Args:
        cable_points_w: Ordered cable points [m], shape ``(num_envs, num_points, 3)``.
        peg_positions_w: Peg centers [m], shape ``(num_envs, num_pegs, 3)``.
        radial_cutoff: Maximum XY distance from a peg for an edge to be local [m].
        axial_cutoff: Maximum absolute Z distance from the peg center for an edge
            to be local [m]. ``None`` preserves radial-only behavior for callers
            that do not model finite-height fixtures.

    Returns:
        A pair containing the number of contiguous local edge spans and their total
        three-dimensional cable length [m], each with shape ``(num_envs, num_pegs)``.
    """
    _, local_points = _peg_relative_xy_and_local_points(cable_points_w, peg_positions_w, radial_cutoff, axial_cutoff)
    local_edges = local_points[..., :-1] & local_points[..., 1:]
    previous_local = torch.zeros_like(local_edges)
    previous_local[..., 1:] = local_edges[..., :-1]
    span_count = (local_edges & ~previous_local).sum(dim=-1)

    edge_length = torch.linalg.vector_norm(cable_points_w[:, 1:] - cable_points_w[:, :-1], dim=-1)
    local_length = torch.where(local_edges, edge_length[:, None, :], 0.0).sum(dim=-1)
    return span_count, local_length


def benchmark_winding_angle(
    cable_points_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    radial_cutoff: float,
    axial_cutoff: float | None = None,
) -> torch.Tensor:
    """Compute signed cable winding around each peg [rad].

    The cable samples must be ordered from the benchmark start endpoint to the finish endpoint.
    ManipulationNet defines positive routing as clockwise when viewed from above, which is the
    opposite of the usual positive planar angle. This function therefore returns clockwise-positive
    angles.

    Args:
        cable_points_w: Ordered cable points [m], shape ``(num_envs, num_points, 3)``.
        peg_positions_w: Peg centers [m], shape ``(num_envs, num_pegs, 3)``.
        radial_cutoff: Maximum XY distance from a peg for a cable edge to contribute [m].
        axial_cutoff: Maximum absolute Z distance from the peg center for an edge
            to contribute [m]. ``None`` preserves radial-only behavior for callers
            that do not model finite-height fixtures.

    Returns:
        Clockwise-positive winding angle [rad], shape ``(num_envs, num_pegs)``.

    Raises:
        ValueError: If the tensor shapes or cutoffs are invalid.
    """
    relative_xy, local_points = _peg_relative_xy_and_local_points(
        cable_points_w, peg_positions_w, radial_cutoff, axial_cutoff
    )
    angle = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
    delta = angle[..., 1:] - angle[..., :-1]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))

    # Require the complete edge to be local to the peg. This prevents long entry/exit edges from
    # contributing an arbitrary angular jump while preserving a genuine local wrap.
    local_edge = local_points[..., :-1] & local_points[..., 1:]
    mathematical_winding = torch.where(local_edge, delta, 0.0).sum(dim=-1)
    return -mathematical_winding


def ordered_route_state(
    winding: torch.Tensor,
    peg_indices: torch.Tensor,
    directions: torch.Tensor,
    valid_steps: torch.Tensor,
    completion_threshold: float,
    maximum_completion_winding: float | None = None,
    completion_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate per-step progress and the longest completed route prefix.

    Args:
        winding: Clockwise-positive winding angles [rad], shape ``(N, P)``.
        peg_indices: Peg index for each padded route step, shape ``(N, G)``.
        directions: Required direction per step (``+1`` clockwise, ``-1`` counterclockwise), shape ``(N, G)``.
        valid_steps: Whether each padded route step is active, shape ``(N, G)``.
        completion_threshold: Required directed winding magnitude [rad].
        maximum_completion_winding: Largest directed winding accepted as one wrap [rad].
            When ``None``, no upper bound is applied.
        completion_mask: Optional per-step geometric eligibility mask, shape ``(N, G)``.

    Returns:
        Tuple ``(directed_progress, completed, prefix_length, success)``. Progress is normalized
        and clipped to ``[-1, 1]``; padded steps have zero progress.
    """
    if completion_threshold <= 0.0:
        raise ValueError("completion_threshold must be positive.")
    if maximum_completion_winding is not None and maximum_completion_winding <= completion_threshold:
        raise ValueError("maximum_completion_winding must exceed completion_threshold.")
    if peg_indices.shape != directions.shape or peg_indices.shape != valid_steps.shape:
        raise ValueError("peg_indices, directions, and valid_steps must have identical shapes.")
    if completion_mask is not None and completion_mask.shape != valid_steps.shape:
        raise ValueError("completion_mask and valid_steps must have identical shapes.")
    if winding.ndim != 2 or peg_indices.ndim != 2 or winding.shape[0] != peg_indices.shape[0]:
        raise ValueError("Expected winding (N, P) and route tensors (N, G).")

    selected_winding = torch.gather(winding, 1, peg_indices.clamp(min=0))
    directed_winding = selected_winding * directions
    directed_progress = (directed_winding / completion_threshold).clamp(min=-1.0, max=1.0)
    directed_progress = torch.where(valid_steps, directed_progress, 0.0)
    completed = valid_steps & (directed_winding >= completion_threshold)
    if maximum_completion_winding is not None:
        completed &= directed_winding <= maximum_completion_winding
    if completion_mask is not None:
        completed &= completion_mask

    # A step contributes only if it and every earlier step are complete. Padded steps are false,
    # so the cumulative product naturally stops at the configured route length.
    prefix_mask = torch.cumprod(completed.to(torch.int64), dim=1).to(torch.bool)
    prefix_length = prefix_mask.sum(dim=1)
    route_length = valid_steps.sum(dim=1)
    success = prefix_length >= route_length
    return directed_progress, completed, prefix_length, success
