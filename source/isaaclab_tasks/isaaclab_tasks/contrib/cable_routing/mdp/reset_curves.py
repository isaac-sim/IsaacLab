# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Route-conditioned cable curves for reset-state replay."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .events import cable_capsule_clearance_mask, cable_capsule_self_clearance_mask
from .reset_curve_xpbd import CableResetCurveXPBDCfg, relax_open_cable_curve_xpbd
from .route_metrics import benchmark_local_cable_spans, benchmark_winding_angle, ordered_route_state

__all__ = [
    "generate_route_conditioned_cable_poses",
    "planar_vertices_to_segment_poses",
    "validate_route_conditioned_cable_poses",
]


def _planar_bishop_quaternions(direction: torch.Tensor) -> torch.Tensor:
    """Build discrete parallel-transport frames whose local +Z follows planar directions."""
    first_quaternion = torch.stack(
        (
            -direction[:, 0, 1],
            direction[:, 0, 0],
            torch.zeros_like(direction[:, 0, 0]),
            torch.ones_like(direction[:, 0, 0]),
        ),
        dim=-1,
    )
    first_quaternion = torch.nn.functional.normalize(first_quaternion, dim=-1)
    previous_tangent = direction[:, :-1]
    tangent = direction[:, 1:]
    turn = torch.atan2(
        previous_tangent[..., 0] * tangent[..., 1] - previous_tangent[..., 1] * tangent[..., 0],
        (previous_tangent * tangent).sum(dim=-1),
    )
    cumulative_turn = torch.cat(
        (torch.zeros_like(turn[:, :1]), torch.cumsum(turn, dim=1)),
        dim=1,
    )
    half_turn = 0.5 * cumulative_turn
    transport = torch.stack(
        (
            torch.zeros_like(half_turn),
            torch.zeros_like(half_turn),
            torch.sin(half_turn),
            torch.cos(half_turn),
        ),
        dim=-1,
    )
    first_quaternion = first_quaternion[:, None].expand_as(transport)
    return torch.nn.functional.normalize(quat_mul(transport, first_quaternion), dim=-1)


def planar_vertices_to_segment_poses(
    vertices_w: torch.Tensor,
    z_w: torch.Tensor,
    reference_quaternions_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert exact-length planar vertices to Newton cable capsule poses.

    The geometric frame is parallel transported along the new centerline. When
    reference material frames are supplied, their per-segment roll about the
    cable tangent is transferred to the new frame. This preserves the VBD
    cable's authored rest-twist profile instead of resetting every material
    frame to zero roll.

    Args:
        vertices_w: Centerline vertices [m], shape ``(N, S + 1, 2)``.
        z_w: Cable center height [m], shape ``(N,)``.
        reference_quaternions_w: Optional authored material frames in ``xyzw``
            order, shape ``(N, S, 4)``.

    Returns:
        Segment poses ``(x, y, z, qx, qy, qz, qw)``, shape ``(N, S, 7)``.
    """
    if vertices_w.ndim != 3 or vertices_w.shape[-1] != 2 or vertices_w.shape[1] < 2:
        raise ValueError(f"vertices_w must have shape (N, S + 1, 2); got {tuple(vertices_w.shape)}.")
    if z_w.shape != (vertices_w.shape[0],):
        raise ValueError(f"z_w must have shape ({vertices_w.shape[0]},); got {tuple(z_w.shape)}.")
    expected_quaternion_shape = (vertices_w.shape[0], vertices_w.shape[1] - 1, 4)
    if reference_quaternions_w is not None and reference_quaternions_w.shape != expected_quaternion_shape:
        raise ValueError(
            f"reference_quaternions_w must have shape {expected_quaternion_shape}; "
            f"got {tuple(reference_quaternions_w.shape)}."
        )

    edge = vertices_w[:, 1:] - vertices_w[:, :-1]
    direction = torch.nn.functional.normalize(edge, dim=-1)
    centers = 0.5 * (vertices_w[:, 1:] + vertices_w[:, :-1])
    positions = torch.cat((centers, z_w[:, None, None].expand(-1, centers.shape[1], 1)), dim=-1)
    quaternion = _planar_bishop_quaternions(direction)
    if reference_quaternions_w is not None:
        reference_quaternion = torch.nn.functional.normalize(reference_quaternions_w, dim=-1)
        local_z = torch.zeros_like(reference_quaternion[..., :3])
        local_z[..., 2] = 1.0
        reference_tangent = torch.nn.functional.normalize(quat_apply(reference_quaternion, local_z)[..., :2], dim=-1)
        reference_bishop = _planar_bishop_quaternions(reference_tangent)

        # q_ref = q_spin(world tangent) * q_bishop. Projecting the relative
        # quaternion onto that tangent removes numerical components orthogonal
        # to the theoretically pure roll and transfers the same material-index
        # spin onto the new tangent.
        reference_spin = quat_mul(reference_quaternion, quat_conjugate(reference_bishop))
        reference_tangent_3d = torch.nn.functional.pad(reference_tangent, (0, 1))
        axial_sine = (reference_spin[..., :3] * reference_tangent_3d).sum(dim=-1)
        tangent_3d = torch.nn.functional.pad(direction, (0, 1))
        transferred_spin = torch.cat(
            (axial_sine[..., None] * tangent_3d, reference_spin[..., 3:4]),
            dim=-1,
        )
        transferred_spin = torch.nn.functional.normalize(transferred_spin, dim=-1)
        quaternion = torch.nn.functional.normalize(quat_mul(transferred_spin, quaternion), dim=-1)
    return torch.cat((positions, quaternion), dim=-1)


def _sample_control_polyline(
    controls: torch.Tensor,
    num_segments: int,
    rest_length: float,
    minimum_bend_radius: float,
) -> torch.Tensor:
    """Arc-length sample a guide polyline, then rebuild a bend-limited exact-length curve."""
    edge = controls[:, 1:] - controls[:, :-1]
    edge_length = torch.linalg.vector_norm(edge, dim=-1)
    cumulative = torch.cat((torch.zeros_like(edge_length[:, :1]), edge_length.cumsum(dim=1)), dim=1)
    required_length = num_segments * rest_length
    if bool((cumulative[:, -1] < required_length).any()):
        short = (cumulative[:, -1] < required_length).nonzero(as_tuple=False).squeeze(-1).tolist()
        raise RuntimeError(f"Route guide is shorter than the cable for rows {short}.")

    distance = torch.arange(num_segments + 1, device=controls.device, dtype=controls.dtype) * rest_length
    distance = distance[None].expand(len(controls), -1).contiguous()
    segment = torch.searchsorted(cumulative.contiguous(), distance, right=True) - 1
    segment = segment.clamp(min=0, max=controls.shape[1] - 2)
    start_s = torch.gather(cumulative, 1, segment)
    segment_length = torch.gather(edge_length, 1, segment).clamp_min(torch.finfo(controls.dtype).eps)
    fraction = ((distance - start_s) / segment_length).clamp(0.0, 1.0)
    start = torch.gather(controls, 1, segment[..., None].expand(-1, -1, 2))
    selected_edge = torch.gather(edge, 1, segment[..., None].expand(-1, -1, 2))
    sampled = start + fraction[..., None] * selected_edge

    return _reconstruct_exact_length_with_bend_limit(sampled, rest_length, minimum_bend_radius)


def _maximum_turn_from_bend_radius(rest_length: float, minimum_bend_radius: float) -> float:
    """Return the largest adjacent chord turn compatible with a discrete bend radius.

    Three consecutive equal-length vertices define a circumcircle with
    ``R = l / (2 sin(theta / 2))``, where ``theta`` is the turn between
    adjacent chord directions. This relation makes the acceptance threshold
    independent of polygon sampling convention.
    """
    if not math.isfinite(minimum_bend_radius) or minimum_bend_radius <= 0.5 * rest_length:
        raise ValueError("minimum_bend_radius must be finite and exceed half the cable rest length.")
    return 2.0 * math.asin(min(1.0, rest_length / (2.0 * minimum_bend_radius)))


def _reconstruct_exact_length_with_bend_limit(
    vertices: torch.Tensor,
    rest_length: float,
    minimum_bend_radius: float,
) -> torch.Tensor:
    """Reintegrate curve headings while limiting every discrete adjacent turn."""
    edge = vertices[:, 1:] - vertices[:, :-1]
    raw_heading = torch.atan2(edge[..., 1], edge[..., 0])
    wrapped_turn = torch.atan2(
        torch.sin(raw_heading[:, 1:] - raw_heading[:, :-1]),
        torch.cos(raw_heading[:, 1:] - raw_heading[:, :-1]),
    )
    unwrapped_heading = torch.cat(
        (raw_heading[:, :1], raw_heading[:, :1] + torch.cumsum(wrapped_turn, dim=1)),
        dim=1,
    )
    maximum_turn = _maximum_turn_from_bend_radius(rest_length, minimum_bend_radius)
    limited_heading = [unwrapped_heading[:, 0]]
    for segment in range(1, unwrapped_heading.shape[1]):
        turn = (unwrapped_heading[:, segment] - limited_heading[-1]).clamp(-maximum_turn, maximum_turn)
        limited_heading.append(limited_heading[-1] + turn)
    limited_heading = torch.stack(limited_heading, dim=1)
    direction = torch.stack((torch.cos(limited_heading), torch.sin(limited_heading)), dim=-1)
    return torch.cat(
        (
            vertices[:, :1],
            vertices[:, :1] + torch.cumsum(rest_length * direction, dim=1),
        ),
        dim=1,
    )


def _refine_projected_curve(
    vertices: torch.Tensor,
    waypoint_positions: torch.Tensor,
    waypoint_mask: torch.Tensor,
    peg_positions: torch.Tensor,
    *,
    rest_length: float,
    cable_radius: float,
    peg_radius: float,
    fixture_clearance: float,
    board_bounds: tuple[tuple[float, float], tuple[float, float]],
    board_clearance: float,
    self_clearance: float,
    minimum_bend_radius: float,
    iterations: int,
) -> torch.Tensor:
    """Repair a rejected topology seed with fixed-sweep open-chain projection."""
    if iterations <= 0:
        return vertices

    # Pin the benchmark connector endpoint and the directed route witnesses.
    # Duplicate witness-to-vertex assignments are averaged by the projector.
    waypoint_indices = torch.cdist(waypoint_positions, vertices).argmin(dim=-1)
    waypoint_mask = waypoint_mask & (waypoint_indices != 0)
    waypoint_indices = torch.cat((torch.zeros_like(waypoint_indices[:, :1]), waypoint_indices), dim=1)
    waypoint_positions = torch.cat((vertices[:, :1], waypoint_positions), dim=1)
    waypoint_mask = torch.cat((torch.ones_like(waypoint_mask[:, :1]), waypoint_mask), dim=1)

    settling_reserve = 0.001
    projected = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=rest_length,
        board_bounds=board_bounds,
        cfg=CableResetCurveXPBDCfg(
            self_separation_distance=2.0 * cable_radius + self_clearance + settling_reserve,
            bend_radius=minimum_bend_radius,
            iterations=iterations,
            board_margin=cable_radius + board_clearance + settling_reserve,
        ),
        waypoint_vertex_indices=waypoint_indices,
        waypoint_positions=waypoint_positions,
        waypoint_mask=waypoint_mask,
        peg_centers=peg_positions,
        peg_radii=cable_radius + peg_radius + fixture_clearance + settling_reserve,
    )

    # VBD starts from an effectively inextensible cable. Projection converges
    # edge lengths but does not make them exact, so retain only its headings,
    # rate-limit their turns, and rebuild every chord at the authored rest
    # length. The continuous capsule/topology validator below remains the
    # acceptance authority because reconstruction can move the centerline.
    return _reconstruct_exact_length_with_bend_limit(projected, rest_length, minimum_bend_radius)


def _route_tensors(
    route_ids: torch.Tensor,
    route_options: Sequence[Sequence[tuple[int, int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand Python route programs into padded tensors."""
    max_steps = max(len(route) for route in route_options)
    peg_indices = torch.zeros((len(route_ids), max_steps), dtype=torch.long, device=route_ids.device)
    directions = torch.zeros((len(route_ids), max_steps), dtype=torch.float32, device=route_ids.device)
    valid = torch.zeros((len(route_ids), max_steps), dtype=torch.bool, device=route_ids.device)
    for route_id, route in enumerate(route_options):
        rows = (route_ids == route_id).nonzero(as_tuple=False).squeeze(-1)
        for step, (peg_index, direction) in enumerate(route):
            peg_indices[rows, step] = peg_index
            directions[rows, step] = float(direction)
            valid[rows, step] = True
    return peg_indices, directions, valid


def validate_route_conditioned_cable_poses(
    segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    route_ids: torch.Tensor,
    active_steps: torch.Tensor,
    requested_active_winding: torch.Tensor,
    route_options: Sequence[Sequence[tuple[int, int]]],
    *,
    rest_length: float,
    completion_winding: float,
    maximum_completion_winding: float,
    radial_cutoff: float = 0.05,
    axial_cutoff: float | None = None,
    maximum_local_cable_length: float = 0.25,
    maximum_unrouted_winding: float = 0.6,
    cable_radius: float = 0.003,
    minimum_bend_radius: float | None = None,
    peg_radius: float = 0.0125,
    fixture_clearance: float = 0.002,
    self_clearance: float = 0.00025,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]] | None = ((-0.15, 0.15), (-0.20, 0.20)),
    board_clearance: float = 0.002,
    requested_progress_tolerance: float = 0.25,
    maximum_active_progress: float = 0.99,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate cable geometry and ordered topology for a nonterminal replay state.

    The same predicate is used before writing generated curves and after Newton
    settling. Keeping one acceptance authority prevents a visually plausible
    curve from entering the bank after its topology changes under dynamics.
    The default bend gate requires a discrete circumradius of at least 1.2
    segment lengths or two cable radii, whichever is larger.

    Returns:
        A validity mask and normalized ordered progress, both shape ``(N,)``.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError("segment_poses_w must have shape (N, S, 7).")
    num_envs = segment_poses_w.shape[0]
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[0] != num_envs or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"peg_positions_w must have shape ({num_envs}, P, 3).")
    if env_origins_w.shape != (num_envs, 3):
        raise ValueError(f"env_origins_w must have shape ({num_envs}, 3).")
    for name, value in (
        ("route_ids", route_ids),
        ("active_steps", active_steps),
        ("requested_active_winding", requested_active_winding),
    ):
        if value.shape != (num_envs,):
            raise ValueError(f"{name} must have shape ({num_envs},); got {tuple(value.shape)}.")
    if rest_length <= 0.0 or completion_winding <= 0.0:
        raise ValueError("rest_length and completion_winding must be positive.")
    if minimum_bend_radius is None:
        minimum_bend_radius = max(1.2 * rest_length, 2.0 * cable_radius)
    maximum_turn = _maximum_turn_from_bend_radius(rest_length, minimum_bend_radius)
    if maximum_completion_winding <= completion_winding:
        raise ValueError("maximum_completion_winding must exceed completion_winding.")
    if axial_cutoff is not None and (not math.isfinite(axial_cutoff) or axial_cutoff <= 0.0):
        raise ValueError("axial_cutoff must be None or finite and positive.")
    if maximum_unrouted_winding < 0.0 or maximum_unrouted_winding >= completion_winding:
        raise ValueError("maximum_unrouted_winding must lie in [0, completion_winding).")
    if requested_progress_tolerance < 0.0:
        raise ValueError("requested_progress_tolerance must be non-negative.")
    if not math.isfinite(maximum_active_progress) or not 0.02 < maximum_active_progress < 1.0:
        raise ValueError("maximum_active_progress must lie strictly inside (0.02, 1).")

    device = segment_poses_w.device
    route_ids = route_ids.to(device=device, dtype=torch.long)
    active_steps = active_steps.to(device=device, dtype=torch.long)
    requested_active_winding = requested_active_winding.to(device=device, dtype=segment_poses_w.dtype)
    peg_indices, directions, valid_steps = _route_tensors(route_ids, route_options)
    route_length = torch.tensor([len(route) for route in route_options], device=device)[route_ids]

    finite_pose = torch.isfinite(segment_poses_w).all(dim=(1, 2))
    valid = finite_pose.clone()
    local_axis = torch.zeros_like(segment_poses_w[..., :3])
    local_axis[..., 2] = 1.0
    tangent = quat_apply(segment_poses_w[..., 3:7], local_axis)[..., :2]
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    unit_tangent = tangent / tangent_norm[..., None].clamp_min(torch.finfo(tangent.dtype).eps)
    adjacent_cosine = (unit_tangent[:, 1:] * unit_tangent[:, :-1]).sum(dim=-1)
    minimum_cosine = math.cos(maximum_turn)
    cosine_tolerance = 32.0 * torch.finfo(segment_poses_w.dtype).eps
    tangent_valid = (tangent_norm > torch.finfo(tangent.dtype).eps).all(dim=1)
    bend_valid = (adjacent_cosine >= minimum_cosine - cosine_tolerance).all(dim=1)
    fixture_diagnostics: dict[str, torch.Tensor] | None = {} if diagnostics is not None else None
    fixture_clearance_valid = cable_capsule_clearance_mask(
        segment_poses_w,
        peg_positions_w,
        env_origins_w,
        rest_length=rest_length,
        cable_radius=cable_radius,
        peg_radius=peg_radius,
        fixture_clearance=fixture_clearance,
        board_bounds_b=board_bounds_b,
        board_clearance=board_clearance,
        diagnostics=fixture_diagnostics,
    )
    self_clearance_valid = cable_capsule_self_clearance_mask(
        segment_poses_w,
        rest_length=rest_length,
        cable_radius=cable_radius,
        self_clearance=self_clearance,
    )
    valid &= tangent_valid
    valid &= bend_valid
    valid &= fixture_clearance_valid
    valid &= self_clearance_valid
    if diagnostics is not None:
        diagnostics["finite_pose"] = finite_pose
        diagnostics["tangent"] = tangent_valid
        diagnostics["bend"] = bend_valid
        diagnostics["fixture_clearance"] = fixture_clearance_valid
        diagnostics["self_clearance"] = self_clearance_valid
        assert fixture_diagnostics is not None
        diagnostics.update({f"fixture_{name}": value for name, value in fixture_diagnostics.items()})
    winding = benchmark_winding_angle(
        segment_poses_w[..., :3],
        peg_positions_w,
        radial_cutoff,
        axial_cutoff,
    )
    span_count, local_length = benchmark_local_cable_spans(
        segment_poses_w[..., :3],
        peg_positions_w,
        radial_cutoff,
        axial_cutoff,
    )
    step_eligible = torch.gather(
        (span_count == 1) & (local_length <= maximum_local_cable_length),
        1,
        peg_indices,
    )
    directed_progress, _, prefix, success = ordered_route_state(
        winding,
        peg_indices,
        directions,
        valid_steps,
        completion_winding,
        maximum_completion_winding=maximum_completion_winding,
        completion_mask=step_eligible,
    )
    active_progress = torch.gather(directed_progress, 1, active_steps[:, None]).squeeze(1)
    active_eligible = torch.gather(step_eligible, 1, active_steps[:, None]).squeeze(1)
    requested_progress = requested_active_winding / completion_winding
    prefix_valid = prefix == active_steps
    nonterminal = ~success
    active_progress_min_valid = active_progress > 0.02
    active_progress_max_valid = active_progress < maximum_active_progress
    requested_progress_valid = (active_progress - requested_progress).abs() < requested_progress_tolerance
    valid &= prefix_valid
    valid &= nonterminal
    valid &= active_eligible
    valid &= active_progress_min_valid
    valid &= active_progress_max_valid
    valid &= requested_progress_valid
    if diagnostics is not None:
        diagnostics["ordered_prefix"] = prefix_valid
        diagnostics["nonterminal"] = nonterminal
        diagnostics["active_eligible"] = active_eligible
        diagnostics["active_progress_min"] = active_progress_min_valid
        diagnostics["active_progress_max"] = active_progress_max_valid
        diagnostics["requested_progress"] = requested_progress_valid

    # Earlier and active route pegs may carry winding. Future and unmentioned
    # pegs must remain untouched so the sampler cannot skip a later subgoal.
    step_ids = torch.arange(valid_steps.shape[1], device=device)
    authorized_steps = valid_steps & (step_ids[None] <= active_steps[:, None])
    authorized_peg_count = torch.zeros_like(winding, dtype=torch.long)
    authorized_peg_count.scatter_add_(1, peg_indices, authorized_steps.to(dtype=torch.long))
    authorized_pegs = authorized_peg_count > 0
    unrouted_valid = torch.where(authorized_pegs, 0.0, winding.abs()).amax(dim=1) < maximum_unrouted_winding
    valid &= unrouted_valid
    if diagnostics is not None:
        diagnostics["unrouted_winding"] = unrouted_valid

    progress = (prefix.float() + active_progress.clamp(min=0.0)) / route_length.float()
    return valid, progress


def _reset_entry_angle(peg_index: int, direction: int, *, completed: bool) -> float:
    """Return the nominal board-side entry angle for a peg wrap."""
    # Direction-specific gates keep partial arcs and their slack tails from
    # doubling back. A completed peg-0 clockwise arc uses the alternate gate
    # whose exit connects cleanly to the next fixture.
    if completed and peg_index == 0:
        return -2.10
    direction_specific_angles = {
        (0, 1): 2.75,
        (1, -1): -1.05,
    }
    if (peg_index, direction) in direction_specific_angles:
        return direction_specific_angles[(peg_index, direction)]
    entry_angle_by_peg = (-0.25 * math.pi, -2.55)
    if peg_index < 0 or peg_index >= len(entry_angle_by_peg):
        raise ValueError(f"No reset-guide entry angle is configured for peg index {peg_index}.")
    return entry_angle_by_peg[peg_index]


def _build_route_guides(
    peg_positions_b: torch.Tensor,
    route_ids: torch.Tensor,
    active_steps: torch.Tensor,
    active_winding: torch.Tensor,
    route_options: Sequence[Sequence[tuple[int, int]]],
    *,
    completed_winding: float,
    wrap_radius: torch.Tensor,
    entry_jitter: torch.Tensor,
    start_jitter: torch.Tensor,
    board_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build topology-correct control polygons and directed peg-arc witnesses."""
    device, dtype = peg_positions_b.device, peg_positions_b.dtype
    controls_per_row: list[torch.Tensor] = []
    witnesses_per_row: list[torch.Tensor] = []
    xmin, xmax = board_bounds[0]
    ymin, ymax = board_bounds[1]
    # Reintegrating equal-length chords through polyline corners can drift by
    # roughly one segment from the guide. Keep more than one rest length of
    # reserve inside the physical board/cable margin.
    boundary_margin = 0.025
    arc_samples = 20

    # Enter each physical fixture from a consistent board-side family instead
    # of keying geometry by route ID. This supports arbitrary valid programs
    # over the two pegs while jittering slack without changing homotopy.
    for row in range(len(route_ids)):
        route_id = int(route_ids[row])
        active_step = int(active_steps[row])
        route = route_options[route_id]
        control: list[torch.Tensor] = [
            # Keep the free start inside the perimeter lane. Long two-fixture
            # guides can traverse most of the board boundary; placing the
            # endpoint exactly on the bottom-left perimeter corner makes the
            # returning slack cross the beginning of the cable.
            torch.tensor((xmin + boundary_margin + 0.030, ymin + boundary_margin + 0.030), device=device, dtype=dtype)
            + start_jitter[row]
        ]
        witnesses: list[torch.Tensor] = []
        exit_tangent = torch.tensor((1.0, 0.0), device=device, dtype=dtype)
        exit_radial = torch.tensor((1.0, 0.0), device=device, dtype=dtype)

        for step, (peg_index, direction) in enumerate(route[: active_step + 1]):
            center = peg_positions_b[row, peg_index]
            if step == 0:
                theta_0 = _reset_entry_angle(peg_index, direction, completed=step < active_step)
                theta_0 += entry_jitter[row, step]
            else:
                # Aim the next radial gate directly at the preceding wrap's
                # exit. A fixed angle can make the inter-fixture strand cut
                # through the next peg neighborhood, adding unrequested
                # winding or even penetrating the fixture before its arc.
                incoming = control[-1] - center
                theta_0 = torch.atan2(incoming[1], incoming[0])
            winding = completed_winding if step < active_step else float(active_winding[row])
            theta = torch.linspace(0.0, 1.0, arc_samples, device=device, dtype=dtype)
            theta = theta_0 - float(direction) * winding * theta
            radius = wrap_radius[row, step]
            arc = center[None] + radius * torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

            # Enter and leave radially. A tangent line crosses the metric's
            # radial neighborhood while rotating by acos(r_wrap/r_cutoff) on
            # both sides, adding roughly two radians that were not requested
            # by the progress sample. Radial transitions contribute zero
            # winding; bend-limited reconstruction rounds their two junctions.
            entry_radial = torch.stack((torch.cos(theta[0]), torch.sin(theta[0])))
            approach = arc[0] + 0.040 * entry_radial
            control.extend((approach, *arc))
            witnesses.extend(arc)
            exit_radial = torch.stack((torch.cos(theta[-1]), torch.sin(theta[-1])))
            exit_tangent = exit_radial
            control.append(arc[-1] + 0.040 * exit_radial)

        exit_point = control[-1]
        right_x = xmax - boundary_margin
        left_x = xmin + boundary_margin
        top_y = ymax - boundary_margin
        bottom_y = ymin + boundary_margin
        # Leave toward the quadrant selected by the final arc radius. Going to
        # a fixed board corner can cut straight back through the just-routed
        # peg and create a second local span. At that corner, traverse the
        # perimeter in whichever direction best continues the exit tangent.
        corners = torch.tensor(
            ((left_x, bottom_y), (right_x, bottom_y), (right_x, top_y), (left_x, top_y)),
            device=device,
            dtype=dtype,
        )
        corner_index = (1 if float(exit_radial[0]) >= 0.0 else 0) + (2 if float(exit_radial[1]) >= 0.0 else 0)
        # Mapping the sign bits above to BL, BR, TR, TL.
        corner_index = (0, 1, 3, 2)[corner_index]
        next_ccw = (corner_index + 1) % 4
        next_cw = (corner_index - 1) % 4
        ccw_direction = torch.nn.functional.normalize(corners[next_ccw] - corners[corner_index], dim=0)
        cw_direction = torch.nn.functional.normalize(corners[next_cw] - corners[corner_index], dim=0)
        incoming_direction = torch.nn.functional.normalize(corners[corner_index] - exit_point, dim=0)
        perimeter_step = (
            1
            if float(torch.dot(ccw_direction, incoming_direction)) >= float(torch.dot(cw_direction, incoming_direction))
            else -1
        )
        perimeter = [corners[(corner_index + perimeter_step * index) % 4] for index in range(4)]
        control.extend((exit_point + 0.030 * exit_tangent, *perimeter))
        controls_per_row.append(torch.stack(control))
        witnesses_per_row.append(torch.stack(witnesses))

    max_controls = max(len(control) for control in controls_per_row)
    max_witnesses = max(len(witnesses) for witnesses in witnesses_per_row)
    controls = torch.empty((len(route_ids), max_controls, 2), device=device, dtype=dtype)
    witnesses = torch.zeros((len(route_ids), max_witnesses, 2), device=device, dtype=dtype)
    witness_mask = torch.zeros((len(route_ids), max_witnesses), device=device, dtype=torch.bool)
    for row, (control, witness) in enumerate(zip(controls_per_row, witnesses_per_row)):
        controls[row, : len(control)] = control
        controls[row, len(control) :] = control[-1]
        witnesses[row, : len(witness)] = witness
        witness_mask[row, : len(witness)] = True
    return controls, witnesses, witness_mask


def generate_route_conditioned_cable_poses(
    default_segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    route_ids: torch.Tensor,
    active_steps: torch.Tensor,
    active_winding: torch.Tensor,
    route_options: Sequence[Sequence[tuple[int, int]]],
    *,
    rest_length: float,
    completion_winding: float,
    maximum_completion_winding: float,
    completed_winding: float = 4.0,
    radial_cutoff: float = 0.05,
    axial_cutoff: float | None = None,
    maximum_local_cable_length: float = 0.25,
    maximum_unrouted_winding: float = 0.6,
    cable_radius: float = 0.003,
    minimum_bend_radius: float | None = None,
    peg_radius: float = 0.0125,
    fixture_clearance: float = 0.002,
    self_clearance: float = 0.00025,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]] = ((-0.15, 0.15), (-0.20, 0.20)),
    board_clearance: float = 0.002,
    wrap_radius_range: tuple[float, float] = (0.023, 0.029),
    entry_angle_jitter: float = 0.65,
    start_position_jitter: float = 0.004,
    curve_projection_iterations: int = 50,
    max_rejection_attempts: int = 24,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate valid nonterminal cable states near ordered route goals.

    Directed peg arcs establish the requested routing homotopy. Rejected seeds
    are repaired by a fixed-sweep Warp position projector before exact-length
    bend-limited reconstruction. Constructive and projected candidates share
    the same continuous capsule, bend-radius, and benchmark acceptance gate.

    Returns:
        A pair containing segment poses and normalized ordered route progress,
        with shapes ``(N, S, 7)`` and ``(N,)``.
    """
    if default_segment_poses_w.ndim != 3 or default_segment_poses_w.shape[-1] != 7:
        raise ValueError("default_segment_poses_w must have shape (N, S, 7).")
    num_envs, num_segments = default_segment_poses_w.shape[:2]
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[0] != num_envs or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"peg_positions_w must have shape ({num_envs}, P, 3).")
    if env_origins_w.shape != (num_envs, 3):
        raise ValueError(f"env_origins_w must have shape ({num_envs}, 3).")
    for name, value in (("route_ids", route_ids), ("active_steps", active_steps), ("active_winding", active_winding)):
        if value.shape != (num_envs,):
            raise ValueError(f"{name} must have shape ({num_envs},); got {tuple(value.shape)}.")
    if rest_length <= 0.0 or completion_winding <= 0.0:
        raise ValueError("rest_length and completion_winding must be positive.")
    if minimum_bend_radius is None:
        minimum_bend_radius = max(1.2 * rest_length, 2.0 * cable_radius)
    _maximum_turn_from_bend_radius(rest_length, minimum_bend_radius)
    if maximum_unrouted_winding < 0.0 or maximum_unrouted_winding >= completion_winding:
        raise ValueError("maximum_unrouted_winding must lie in [0, completion_winding).")
    if axial_cutoff is not None and (not math.isfinite(axial_cutoff) or axial_cutoff <= 0.0):
        raise ValueError("axial_cutoff must be None or finite and positive.")
    if not completion_winding < completed_winding < maximum_completion_winding:
        raise ValueError("completed_winding must lie strictly between completion and maximum winding.")
    if wrap_radius_range[0] < cable_radius + peg_radius + fixture_clearance:
        raise ValueError("wrap_radius_range starts inside the inflated peg collision radius.")
    if wrap_radius_range[0] > wrap_radius_range[1]:
        raise ValueError("wrap_radius_range must be ordered.")
    if max_rejection_attempts < 1:
        raise ValueError("max_rejection_attempts must be positive.")
    if isinstance(curve_projection_iterations, bool) or not isinstance(curve_projection_iterations, int):
        raise TypeError("curve_projection_iterations must be an integer.")
    if curve_projection_iterations < 0:
        raise ValueError("curve_projection_iterations must be non-negative.")

    device, dtype = default_segment_poses_w.device, default_segment_poses_w.dtype
    route_ids = route_ids.to(device=device, dtype=torch.long)
    active_steps = active_steps.to(device=device, dtype=torch.long)
    active_winding = active_winding.to(device=device, dtype=dtype)
    route_length = torch.tensor([len(route) for route in route_options], device=device)[route_ids]
    if bool(((active_steps < 0) | (active_steps >= route_length)).any()):
        raise ValueError("Every active step must index its selected route.")
    if bool(((active_winding <= 0.0) | (active_winding >= completion_winding)).any()):
        raise ValueError("Active winding must be positive and strictly nonterminal.")

    accepted = torch.zeros(num_envs, device=device, dtype=torch.bool)
    result = torch.empty_like(default_segment_poses_w)
    result_progress = torch.zeros(num_envs, device=device, dtype=dtype)
    peg_indices, directions, valid_steps = _route_tensors(route_ids, route_options)
    peg_positions_b = peg_positions_w[..., :2] - env_origins_w[:, None, :2]
    z_w = default_segment_poses_w[..., 2].mean(dim=1)

    for _ in range(max_rejection_attempts):
        rows = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            break
        shape = (len(rows), valid_steps.shape[1])
        wrap_radius = torch.empty(shape, device=device, dtype=dtype).uniform_(
            wrap_radius_range[0], wrap_radius_range[1], generator=generator
        )
        entry_jitter = torch.empty(shape, device=device, dtype=dtype).uniform_(
            -entry_angle_jitter, entry_angle_jitter, generator=generator
        )
        start_jitter = torch.empty((len(rows), 2), device=device, dtype=dtype).uniform_(
            -start_position_jitter, start_position_jitter, generator=generator
        )
        controls, waypoints, waypoint_mask = _build_route_guides(
            peg_positions_b[rows],
            route_ids[rows],
            active_steps[rows],
            active_winding[rows],
            route_options,
            completed_winding=completed_winding,
            wrap_radius=wrap_radius,
            entry_jitter=entry_jitter,
            start_jitter=start_jitter,
            board_bounds=board_bounds_b,
        )
        seed = _sample_control_polyline(controls, num_segments, rest_length, minimum_bend_radius)

        def accept_curve(curve: torch.Tensor, candidate_rows: torch.Tensor) -> torch.Tensor:
            """Validate one local curve batch and commit accepted rows."""
            # Keep the 1 cm centerline differences in the board-local frame
            # while deriving tangents and bend-limited material frames. Adding
            # a distant clone origin first loses enough float32 precision to
            # make a geometrically identical curve fail the strict 12 mm bend
            # gate. Translate the completed poses only after their local
            # directions have been constructed.
            origins_w = env_origins_w[candidate_rows]
            candidates = planar_vertices_to_segment_poses(
                curve,
                z_w[candidate_rows] - origins_w[:, 2],
                default_segment_poses_w[candidate_rows, :, 3:7],
            )
            candidates[..., :3] += origins_w[:, None]
            candidate_valid, score = validate_route_conditioned_cable_poses(
                candidates,
                peg_positions_w[candidate_rows],
                env_origins_w[candidate_rows],
                route_ids[candidate_rows],
                active_steps[candidate_rows],
                active_winding[candidate_rows],
                route_options,
                rest_length=rest_length,
                completion_winding=completion_winding,
                maximum_completion_winding=maximum_completion_winding,
                radial_cutoff=radial_cutoff,
                axial_cutoff=axial_cutoff,
                maximum_local_cable_length=maximum_local_cable_length,
                maximum_unrouted_winding=maximum_unrouted_winding,
                cable_radius=cable_radius,
                minimum_bend_radius=minimum_bend_radius,
                peg_radius=peg_radius,
                fixture_clearance=fixture_clearance,
                self_clearance=self_clearance,
                board_bounds_b=board_bounds_b,
                board_clearance=board_clearance,
            )

            accepted_rows = candidate_rows[candidate_valid]
            result[accepted_rows] = candidates[candidate_valid]
            result_progress[accepted_rows] = score[candidate_valid]
            accepted[accepted_rows] = True
            return candidate_valid

        # Constructive topology seeds are exact-length and often already pass
        # every continuous capsule/benchmark check. Keep them unchanged when
        # valid; refine only rejected rows, which is both safer and much faster.
        seed_valid = accept_curve(seed, rows)
        failed_local = (~seed_valid).nonzero(as_tuple=False).squeeze(-1)
        if len(failed_local) > 0 and curve_projection_iterations > 0:
            refined = _refine_projected_curve(
                seed[failed_local],
                waypoints[failed_local],
                waypoint_mask[failed_local],
                peg_positions_b[rows[failed_local]],
                rest_length=rest_length,
                cable_radius=cable_radius,
                peg_radius=peg_radius,
                fixture_clearance=fixture_clearance,
                board_bounds=board_bounds_b,
                board_clearance=board_clearance,
                self_clearance=self_clearance,
                minimum_bend_radius=minimum_bend_radius,
                iterations=curve_projection_iterations,
            )
            accept_curve(refined, rows[failed_local])

    if not bool(accepted.all()):
        failed = (~accepted).nonzero(as_tuple=False).squeeze(-1).tolist()
        failure_details = [
            {
                "row": row,
                "route_id": int(route_ids[row]),
                "active_step": int(active_steps[row]),
                "active_winding": float(active_winding[row]),
                "peg_positions_b": peg_positions_b[row].tolist(),
            }
            for row in failed
        ]
        raise RuntimeError(
            "Unable to construct valid route-conditioned cable curves after "
            f"{max_rejection_attempts} attempts: {failure_details}."
        )
    return result, result_progress
