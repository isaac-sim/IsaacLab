# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the cable-routing task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul

from .route_metrics import benchmark_winding_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "BENCHMARK_GRID_DIRECTIONS",
    "cable_capsule_clearance_mask",
    "cable_capsule_self_clearance_mask",
    "cable_unrouted_mask",
    "generate_collision_free_cable_poses",
    "reset_cable_state",
    "reset_peg_offsets",
    "sample_cable_heading_offsets",
    "sample_benchmark_grid_offsets",
    "sample_board_frame_se2",
    "shape_cable_poses_planar",
    "transform_cable_poses_se2",
]


BENCHMARK_GRID_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
"""The eight nonzero fixture offsets used by the ManipulationNet benchmark."""


def _resolve_env_ids(env_ids: torch.Tensor, num_envs: int, device: str | torch.device) -> torch.Tensor:
    """Resolve an index vector or full environment mask to an index vector."""
    env_ids = env_ids.to(device=device)
    if env_ids.dtype == torch.bool:
        if env_ids.ndim != 1 or env_ids.numel() != num_envs:
            raise ValueError(f"Boolean env mask must have shape ({num_envs},); got {tuple(env_ids.shape)}.")
        return env_ids.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.ndim != 1:
        raise ValueError(f"env_ids must be one-dimensional; got shape {tuple(env_ids.shape)}.")
    return env_ids.to(dtype=torch.long)


def sample_benchmark_grid_offsets(
    num_envs: int,
    num_assets: int,
    *,
    grid_pitch: float = 0.01,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample independent benchmark fixture offsets.

    Args:
        num_envs: Number of environment rows to sample.
        num_assets: Number of fixture offsets per environment.
        grid_pitch: Distance represented by one board-grid step [m].
        device: Device of the returned tensor.
        dtype: Floating-point dtype of the returned tensor.
        generator: Optional device-compatible generator for deterministic sampling.

    Returns:
        Sampled planar offsets [m], shape ``(num_envs, num_assets, 2)``.
    """
    if num_envs < 0 or num_assets < 0:
        raise ValueError(f"num_envs and num_assets must be non-negative; got {num_envs} and {num_assets}.")
    if grid_pitch <= 0.0:
        raise ValueError(f"grid_pitch must be positive; got {grid_pitch}.")

    directions = torch.tensor(BENCHMARK_GRID_DIRECTIONS, device=device, dtype=dtype)
    choices = torch.randint(
        len(BENCHMARK_GRID_DIRECTIONS),
        (num_envs, num_assets),
        device=device,
        generator=generator,
    )
    return directions[choices] * grid_pitch


def sample_board_frame_se2(
    num_envs: int,
    *,
    translation_jitter: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        (-0.02, 0.02),
        (-0.02, 0.02),
    ),
    yaw_jitter: tuple[float, float] = (-0.17453292519943295, 0.17453292519943295),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a planar rigid perturbation in the board frame.

    A two-scalar ``translation_jitter`` applies one ``(min, max)`` range to
    both planar axes. Two nested ranges configure the x and y axes separately.

    Args:
        num_envs: Number of transforms to sample.
        translation_jitter: Uniform x/y translation ranges [m].
        yaw_jitter: Uniform yaw range [rad].
        device: Device of the returned tensors.
        dtype: Floating-point dtype of the returned tensors.
        generator: Optional device-compatible generator for deterministic sampling.

    Returns:
        A pair ``(translation_xy, yaw)`` with shapes ``(num_envs, 2)`` and
        ``(num_envs,)``.
    """
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative; got {num_envs}.")

    translation_range = torch.as_tensor(translation_jitter, device=device, dtype=dtype)
    if translation_range.shape == (2,):
        translation_range = translation_range.expand(2, 2)
    elif translation_range.shape != (2, 2):
        raise ValueError(
            "translation_jitter must be one (min, max) range or separate x/y ranges; "
            f"got shape {tuple(translation_range.shape)}."
        )
    yaw_range = torch.as_tensor(yaw_jitter, device=device, dtype=dtype)
    if yaw_range.shape != (2,):
        raise ValueError(f"yaw_jitter must have shape (2,); got {tuple(yaw_range.shape)}.")
    if bool((translation_range[:, 0] > translation_range[:, 1]).any()) or bool(yaw_range[0] > yaw_range[1]):
        raise ValueError("Jitter ranges must be ordered as (min, max).")

    unit_samples = torch.rand((num_envs, 3), device=device, dtype=dtype, generator=generator)
    translation_xy = translation_range[:, 0] + unit_samples[:, :2] * (translation_range[:, 1] - translation_range[:, 0])
    yaw = yaw_range[0] + unit_samples[:, 2] * (yaw_range[1] - yaw_range[0])
    return translation_xy, yaw


def sample_cable_heading_offsets(
    num_envs: int,
    num_segments: int,
    *,
    max_heading_offset: float = 0.08,
    num_modes: int = 3,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample smooth planar heading offsets for heterogeneous cable resets.

    A short sine basis produces curves that differ in every environment without independently
    jittering cable segments. The perturbation is zero at both cable ends and its absolute value
    is bounded by ``max_heading_offset``.

    Returns:
        Heading offsets [rad], shape ``(num_envs, num_segments)``.
    """
    if num_envs < 0 or num_segments < 1:
        raise ValueError(f"Expected non-negative num_envs and positive num_segments; got {num_envs}, {num_segments}.")
    if max_heading_offset < 0.0:
        raise ValueError(f"max_heading_offset must be non-negative; got {max_heading_offset}.")
    if num_modes < 1:
        raise ValueError(f"num_modes must be positive; got {num_modes}.")

    phase = torch.linspace(0.0, 1.0, num_segments, device=device, dtype=dtype)
    modes = torch.arange(1, num_modes + 1, device=device, dtype=dtype)
    basis = torch.sin(torch.pi * phase[:, None] * modes[None, :])
    coefficients = 2.0 * torch.rand((num_envs, num_modes), device=device, dtype=dtype, generator=generator) - 1.0
    offsets = coefficients @ basis.T
    peak = offsets.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
    return offsets / peak * max_heading_offset


def shape_cable_poses_planar(
    segment_poses_w: torch.Tensor,
    rest_length: float,
    heading_offsets: torch.Tensor,
) -> torch.Tensor:
    """Bend cable segment poses in-plane while retaining an exactly connected centerline.

    Segment orientation is yawed by the sampled offset. Segment centers are then reconstructed
    from connected endpoints at ``rest_length`` spacing and recentered on the original cable.
    This avoids injecting axial strain during a reset.

    Args:
        segment_poses_w: Segment poses ``(x, y, z, qx, qy, qz, qw)``, shape ``(N, S, 7)``.
        rest_length: Rest length of each segment [m].
        heading_offsets: Per-segment planar heading offsets [rad], shape ``(N, S)``.

    Returns:
        Shaped segment poses with the same shape, device, and dtype as the input.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    if heading_offsets.shape != segment_poses_w.shape[:2]:
        raise ValueError(
            f"heading_offsets must have shape {tuple(segment_poses_w.shape[:2])}; got {tuple(heading_offsets.shape)}."
        )
    if rest_length <= 0.0:
        raise ValueError(f"rest_length must be positive; got {rest_length}.")

    shaped = segment_poses_w.clone()
    half_heading = 0.5 * heading_offsets
    yaw_quat = torch.zeros_like(segment_poses_w[..., 3:7])
    yaw_quat[..., 2] = torch.sin(half_heading)
    yaw_quat[..., 3] = torch.cos(half_heading)
    shaped[..., 3:7] = quat_mul(yaw_quat, segment_poses_w[..., 3:7])

    local_axis = torch.zeros_like(segment_poses_w[..., :3])
    local_axis[..., 2] = 1.0
    directions = quat_apply(shaped[..., 3:7], local_axis)
    endpoints = torch.empty(
        (segment_poses_w.shape[0], segment_poses_w.shape[1] + 1, 3),
        device=segment_poses_w.device,
        dtype=segment_poses_w.dtype,
    )
    endpoints[:, 0] = segment_poses_w[:, 0, :3] - 0.5 * rest_length * quat_apply(
        segment_poses_w[:, 0, 3:7], local_axis[:, 0]
    )
    endpoints[:, 1:] = endpoints[:, :1] + torch.cumsum(rest_length * directions, dim=1)
    centers = 0.5 * (endpoints[:, :-1] + endpoints[:, 1:])
    centers += segment_poses_w[..., :3].mean(dim=1, keepdim=True) - centers.mean(dim=1, keepdim=True)
    shaped[..., :3] = centers
    return shaped


def _cable_segment_endpoints_w(segment_poses_w: torch.Tensor, rest_length: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two centerline endpoints of every cable capsule."""
    local_axis = torch.zeros_like(segment_poses_w[..., :3])
    local_axis[..., 2] = 1.0
    half_axis = 0.5 * rest_length * quat_apply(segment_poses_w[..., 3:7], local_axis)
    return segment_poses_w[..., :3] - half_axis, segment_poses_w[..., :3] + half_axis


def cable_capsule_clearance_mask(
    segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    *,
    rest_length: float,
    cable_radius: float = 0.003,
    peg_radius: float = 0.0125,
    fixture_clearance: float = 0.002,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]] | None = ((-0.15, 0.15), (-0.20, 0.20)),
    board_clearance: float = 0.002,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Check cable-capsule clearance from peg cylinders and board edges.

    Peg clearance is evaluated from each peg center to every capsule centerline
    in the board plane. Board containment evaluates both centerline endpoints
    with the cable radius and requested clearance as an inset margin.

    Args:
        segment_poses_w: Cable segment poses, shape ``(N, S, 7)``.
        peg_positions_w: Peg centers, shape ``(N, P, 3)``. ``P`` may be zero.
        env_origins_w: Board-frame origins in world frame [m], shape ``(N, 3)``.
        rest_length: Cable capsule centerline length [m].
        cable_radius: Cable capsule radius [m].
        peg_radius: Peg cylinder radius [m].
        fixture_clearance: Additional cable-to-peg surface clearance [m].
        board_bounds_b: Board x/y bounds [m] relative to each environment
            origin, or ``None`` to disable the containment check.
        board_clearance: Additional cable-to-board-edge clearance [m].

    Returns:
        Per-environment validity mask, shape ``(N,)``.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    num_envs = segment_poses_w.shape[0]
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[0] != num_envs or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"peg_positions_w must have shape ({num_envs}, P, 3); got {tuple(peg_positions_w.shape)}.")
    if env_origins_w.shape != (num_envs, 3):
        raise ValueError(f"env_origins_w must have shape ({num_envs}, 3); got {tuple(env_origins_w.shape)}.")
    if rest_length <= 0.0:
        raise ValueError(f"rest_length must be positive; got {rest_length}.")
    if min(cable_radius, peg_radius, fixture_clearance, board_clearance) < 0.0:
        raise ValueError("Cable/peg radii and clearances must be non-negative.")

    segment_start_w, segment_end_w = _cable_segment_endpoints_w(segment_poses_w, rest_length)
    clear = torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)
    peg_clear = torch.ones_like(clear)
    board_clear = torch.ones_like(clear)
    if diagnostics is not None:
        infinite_margin = torch.full(
            (num_envs,),
            float("inf"),
            device=segment_poses_w.device,
            dtype=segment_poses_w.dtype,
        )
        peg_margin = infinite_margin
        board_margin = infinite_margin

    if peg_positions_w.shape[1] > 0:
        start_xy = segment_start_w[:, None, :, :2]
        edge_xy = (segment_end_w - segment_start_w)[:, None, :, :2]
        peg_xy = peg_positions_w[:, :, None, :2]
        projection = ((peg_xy - start_xy) * edge_xy).sum(dim=-1)
        projection /= edge_xy.square().sum(dim=-1).clamp_min(torch.finfo(segment_poses_w.dtype).eps)
        closest_xy = start_xy + projection.clamp(0.0, 1.0)[..., None] * edge_xy
        centerline_distance = torch.linalg.vector_norm(peg_xy - closest_xy, dim=-1)
        required_distance = cable_radius + peg_radius + fixture_clearance
        peg_clear = (centerline_distance >= required_distance).all(dim=(1, 2))
        if diagnostics is not None:
            peg_margin = (centerline_distance - required_distance).amin(dim=(1, 2))
        clear &= peg_clear

    if board_bounds_b is not None:
        bounds = torch.as_tensor(board_bounds_b, device=segment_poses_w.device, dtype=segment_poses_w.dtype)
        if bounds.shape != (2, 2) or bool((bounds[:, 0] >= bounds[:, 1]).any()):
            raise ValueError(f"board_bounds_b must contain ordered x/y bounds with shape (2, 2); got {board_bounds_b}.")
        endpoints_b_xy = (
            torch.cat((segment_start_w[..., :2], segment_end_w[..., :2]), dim=1) - env_origins_w[:, None, :2]
        )
        margin = cable_radius + board_clearance
        lower_bound = bounds[None, :, 0] + margin
        upper_bound = bounds[None, :, 1] - margin
        board_clear = (endpoints_b_xy >= lower_bound).all(dim=(1, 2))
        board_clear &= (endpoints_b_xy <= upper_bound).all(dim=(1, 2))
        if diagnostics is not None:
            lower_margin = endpoints_b_xy - lower_bound
            upper_margin = upper_bound - endpoints_b_xy
            board_margin = torch.minimum(lower_margin, upper_margin).amin(dim=(1, 2))
        clear &= board_clear

    if diagnostics is not None:
        diagnostics["peg_clearance"] = peg_clear
        diagnostics["peg_margin_m"] = peg_margin
        diagnostics["board_clearance"] = board_clear
        diagnostics["board_margin_m"] = board_margin

    return clear


def cable_capsule_self_clearance_mask(
    segment_poses_w: torch.Tensor,
    *,
    rest_length: float,
    cable_radius: float = 0.003,
    self_clearance: float = 0.00025,
    neighbor_exclusion: int = 1,
) -> torch.Tensor:
    """Check planar centerline separation between non-neighbor cable capsules.

    Adjacent capsules intentionally share a centerline endpoint and are excluded.
    The calculation is chunked across environments to bound temporary pairwise
    storage during large vectorized resets.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    if rest_length <= 0.0:
        raise ValueError(f"rest_length must be positive; got {rest_length}.")
    if cable_radius < 0.0 or self_clearance < 0.0:
        raise ValueError("Cable radius and self-clearance must be non-negative.")
    if neighbor_exclusion < 0:
        raise ValueError(f"neighbor_exclusion must be non-negative; got {neighbor_exclusion}.")

    num_envs, num_segments = segment_poses_w.shape[:2]
    if num_segments <= neighbor_exclusion + 1:
        return torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)
    segment_start_w, segment_end_w = _cable_segment_endpoints_w(segment_poses_w, rest_length)
    pair_index = torch.arange(num_segments, device=segment_poses_w.device)
    checked_pairs = pair_index[None, :] - pair_index[:, None] > neighbor_exclusion
    required_distance = 2.0 * cable_radius + self_clearance
    clear = torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)

    def point_segment_distance(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        edge = end - start
        projection = ((point - start) * edge).sum(dim=-1)
        projection /= edge.square().sum(dim=-1).clamp_min(torch.finfo(segment_poses_w.dtype).eps)
        closest = start + projection.clamp(0.0, 1.0)[..., None] * edge
        return torch.linalg.vector_norm(point - closest, dim=-1)

    # Roughly two million segment pairs per chunk keeps peak temporaries modest.
    chunk_size = max(1, 2_000_000 // (num_segments * num_segments))
    for chunk_start in range(0, num_envs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_envs)
        start = segment_start_w[chunk_start:chunk_end, :, :2]
        end = segment_end_w[chunk_start:chunk_end, :, :2]
        start_i, end_i = start[:, :, None], end[:, :, None]
        start_j, end_j = start[:, None, :], end[:, None, :]
        distance = torch.minimum(
            torch.minimum(
                point_segment_distance(start_i, start_j, end_j),
                point_segment_distance(end_i, start_j, end_j),
            ),
            torch.minimum(
                point_segment_distance(start_j, start_i, end_i),
                point_segment_distance(end_j, start_i, end_i),
            ),
        )

        # Endpoint distances do not detect a proper 2-D crossing, so mark those
        # pairs explicitly. Collinear overlap is already caught by the distances.
        edge_i = end_i - start_i
        edge_j = end_j - start_j
        relative_start = start_j - start_i

        def cross_2d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]

        denominator = cross_2d(edge_i, edge_j)
        nonparallel = denominator.abs() > 10.0 * torch.finfo(segment_poses_w.dtype).eps
        safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
        first_fraction = cross_2d(relative_start, edge_j) / safe_denominator
        second_fraction = cross_2d(relative_start, edge_i) / safe_denominator
        intersects = (
            nonparallel
            & (first_fraction >= 0.0)
            & (first_fraction <= 1.0)
            & (second_fraction >= 0.0)
            & (second_fraction <= 1.0)
        )
        distance = torch.where(intersects, torch.zeros_like(distance), distance)
        clear[chunk_start:chunk_end] = (distance[:, checked_pairs] >= required_distance).all(dim=1)
    return clear


def cable_unrouted_mask(
    segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    *,
    radial_cutoff: float = 0.10,
    axial_cutoff: float | None = None,
    max_abs_winding: float = 0.5,
) -> torch.Tensor:
    """Check that a reset cable is topologically unrouted around every live peg.

    This uses the exact winding metric consumed by the routing command. Keeping
    initial winding comfortably below the completion threshold prevents reset
    geometry from being reported as a successful route before the robot acts.

    Args:
        segment_poses_w: Ordered cable segment poses, shape ``(N, S, 7)``.
        peg_positions_w: Live randomized peg centers, shape ``(N, P, 3)``.
        radial_cutoff: Winding metric's local radial cutoff [m].
        axial_cutoff: Winding metric's local vertical cutoff from the peg center
            [m], or ``None`` for fixtures without a finite axial extent.
        max_abs_winding: Maximum allowed initial absolute winding [rad].

    Returns:
        Per-environment topology validity mask, shape ``(N,)``.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    num_envs = segment_poses_w.shape[0]
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[0] != num_envs or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"peg_positions_w must have shape ({num_envs}, P, 3); got {tuple(peg_positions_w.shape)}.")
    if not math.isfinite(radial_cutoff) or radial_cutoff <= 0.0:
        raise ValueError(f"radial_cutoff must be positive; got {radial_cutoff}.")
    if axial_cutoff is not None and (not math.isfinite(axial_cutoff) or axial_cutoff <= 0.0):
        raise ValueError(f"axial_cutoff must be None or positive; got {axial_cutoff}.")
    if not math.isfinite(max_abs_winding) or max_abs_winding < 0.0:
        raise ValueError(f"max_abs_winding must be non-negative; got {max_abs_winding}.")
    if peg_positions_w.shape[1] == 0:
        return torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)

    winding = benchmark_winding_angle(segment_poses_w[..., :3], peg_positions_w, radial_cutoff, axial_cutoff)
    return (winding.abs() <= max_abs_winding).all(dim=1)


def _generate_boundary_cable_poses(
    default_segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    *,
    rest_length: float,
    cable_radius: float,
    self_clearance: float,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]],
    board_clearance: float,
    winding_radial_cutoff: float,
    max_initial_abs_winding: float,
    side: str,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a connected cable into a self-clear strip along one board edge.

    The alternating horizontal runs retain the exact segment rest length. Their
    top edge stays outside the winding metric's radial neighborhood, making the
    reset topologically neutral without changing the benchmark goal metric.
    """
    num_envs, num_segments = default_segment_poses_w.shape[:2]
    device, dtype = default_segment_poses_w.device, default_segment_poses_w.dtype
    side_axes = {
        "bottom": (0, 1, True),
        "top": (0, 1, False),
        "left": (1, 0, True),
        "right": (1, 0, False),
    }
    if side not in side_axes:
        raise ValueError(f"side must be one of {tuple(side_axes)}; got {side!r}.")
    tangent_axis, normal_axis, lower_side = side_axes[side]
    bounds = torch.as_tensor(board_bounds_b, device=device, dtype=dtype)
    margin = cable_radius + board_clearance
    # Candidates are constructed in board coordinates but accepted after a
    # float32 round trip through world coordinates. Keep a physical reserve
    # larger than that quantization at typical cloned-environment origins.
    construction_reserve = 5.0e-5
    inset_min = bounds[:, 0] + margin + construction_reserve
    inset_max = bounds[:, 1] - margin - construction_reserve
    usable_tangent = float((inset_max[tangent_axis] - inset_min[tangent_axis]).item())
    max_run_segments = int((usable_tangent / rest_length) // 1)
    if max_run_segments < 1:
        return default_segment_poses_w.clone(), torch.zeros(num_envs, device=device, dtype=torch.bool)

    # With K lanes, K-1 segments connect the lanes and the remaining segments
    # form horizontal runs. Choose the fewest lanes whose runs fit the board.
    num_lanes = (num_segments + max_run_segments + 1) // (max_run_segments + 1)
    num_connectors = num_lanes - 1
    num_horizontal = num_segments - num_connectors
    run_base, run_remainder = divmod(num_horizontal, num_lanes)
    run_counts = [run_base + (lane < run_remainder) for lane in range(num_lanes)]
    if max(run_counts) > max_run_segments:
        return default_segment_poses_w.clone(), torch.zeros(num_envs, device=device, dtype=torch.bool)

    peg_normal_b = peg_positions_w[..., normal_axis] - env_origins_w[:, None, normal_axis]
    # For a straight pass at perpendicular distance d inside a radial cutoff r,
    # the largest local angular sweep is 2*acos(d/r). This bound lets the upper
    # lane enter the cutoff just enough to retain capsule self-clearance while
    # remaining below the configured initial-winding limit.
    topology_margin = max(2.5e-4, 10.0 * torch.finfo(dtype).eps)
    half_winding = min(0.5 * max_initial_abs_winding, 0.5 * torch.pi)
    min_peg_distance = winding_radial_cutoff * torch.cos(torch.as_tensor(half_winding, device=device, dtype=dtype))
    if lower_side:
        safe_inner_normal = peg_normal_b.amin(dim=1) - min_peg_distance - topology_margin
        available_normal = safe_inner_normal - inset_min[normal_axis]
    else:
        safe_inner_normal = peg_normal_b.amax(dim=1) + min_peg_distance + topology_margin
        available_normal = inset_max[normal_axis] - safe_inner_normal
    if num_connectors == 0:
        lane_pitch = torch.zeros(num_envs, device=device, dtype=dtype)
        can_fit_normally = available_normal >= 0.0
    else:
        max_pitch = torch.minimum(
            available_normal / num_connectors,
            torch.full((num_envs,), 0.999 * rest_length, device=device, dtype=dtype),
        )
        # Non-neighboring horizontal capsules stay at least one cable diameter
        # plus a small gap apart, so the compact reset contains no self-contact.
        min_pitch = 2.0 * cable_radius + self_clearance + construction_reserve
        can_fit_normally = max_pitch >= min_pitch
        unit_pitch = torch.rand((num_envs,), device=device, dtype=dtype, generator=generator)
        lane_pitch = min_pitch + unit_pitch * (max_pitch - min_pitch).clamp_min(0.0)

    direction_sign = torch.where(
        torch.rand((num_envs,), device=device, dtype=dtype, generator=generator) < 0.5,
        -torch.ones(num_envs, device=device, dtype=dtype),
        torch.ones(num_envs, device=device, dtype=dtype),
    )
    directions = torch.zeros((num_envs, num_segments, 3), device=device, dtype=dtype)
    cursor = 0
    sign = direction_sign
    for lane, run_count in enumerate(run_counts):
        directions[:, cursor : cursor + run_count, tangent_axis] = sign[:, None]
        cursor += run_count
        if lane < num_connectors:
            directions[:, cursor, tangent_axis] = -sign * torch.sqrt(
                (rest_length**2 - lane_pitch.square()).clamp_min(0.0)
            )
            directions[:, cursor, normal_axis] = lane_pitch if lower_side else -lane_pitch
            directions[:, cursor] /= rest_length
            cursor += 1
            sign = -sign

    relative_endpoints = torch.zeros((num_envs, num_segments + 1, 3), device=device, dtype=dtype)
    relative_endpoints[:, 1:] = torch.cumsum(rest_length * directions, dim=1)
    relative_min_tangent = relative_endpoints[..., tangent_axis].amin(dim=1)
    relative_max_tangent = relative_endpoints[..., tangent_axis].amax(dim=1)
    start_tangent_min = inset_min[tangent_axis] - relative_min_tangent
    start_tangent_max = inset_max[tangent_axis] - relative_max_tangent
    can_fit_tangentially = start_tangent_min <= start_tangent_max
    unit_tangent = torch.rand((num_envs,), device=device, dtype=dtype, generator=generator)
    start_tangent = start_tangent_min + unit_tangent * (start_tangent_max - start_tangent_min).clamp_min(0.0)

    relative_min_normal = relative_endpoints[..., normal_axis].amin(dim=1)
    relative_max_normal = relative_endpoints[..., normal_axis].amax(dim=1)
    start_normal_min = inset_min[normal_axis] - relative_min_normal
    start_normal_max = inset_max[normal_axis] - relative_max_normal
    if lower_side:
        start_normal_max = torch.minimum(start_normal_max, safe_inner_normal - relative_max_normal)
    else:
        start_normal_min = torch.maximum(start_normal_min, safe_inner_normal - relative_min_normal)
    can_fit_normally &= start_normal_min <= start_normal_max
    unit_normal = torch.rand((num_envs,), device=device, dtype=dtype, generator=generator)
    start_normal = start_normal_min + unit_normal * (start_normal_max - start_normal_min).clamp_min(0.0)

    relative_endpoints[..., tangent_axis] += start_tangent[:, None]
    relative_endpoints[..., normal_axis] += start_normal[:, None]
    local_z = (default_segment_poses_w[..., 2] - env_origins_w[:, None, 2]).mean(dim=1)
    relative_endpoints[..., 2] = local_z[:, None]
    endpoints_w = relative_endpoints + env_origins_w[:, None, :]
    centers_w = 0.5 * (endpoints_w[:, :-1] + endpoints_w[:, 1:])

    # Shortest-arc quaternion rotating each capsule's local +Z axis onto its
    # planar centerline direction: normalize((-dy, dx, 0, 1)).
    quaternion = torch.stack(
        (
            -directions[..., 1],
            directions[..., 0],
            torch.zeros_like(directions[..., 0]),
            torch.ones_like(directions[..., 0]),
        ),
        dim=-1,
    )
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    poses = torch.cat((centers_w, quaternion), dim=-1)
    return poses, can_fit_tangentially & can_fit_normally


def _sample_board_fitting_translation(
    segment_poses_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    rest_length: float,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]],
    board_margin: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample translations from the range that keeps each cable on the board."""
    bounds = torch.as_tensor(board_bounds_b, device=segment_poses_w.device, dtype=segment_poses_w.dtype)
    segment_start_w, segment_end_w = _cable_segment_endpoints_w(segment_poses_w, rest_length)
    endpoints_b_xy = torch.cat((segment_start_w[..., :2], segment_end_w[..., :2]), dim=1) - env_origins_w[:, None, :2]
    translation_min = bounds[None, :, 0] + board_margin - endpoints_b_xy.amin(dim=1)
    translation_max = bounds[None, :, 1] - board_margin - endpoints_b_xy.amax(dim=1)
    can_fit = (translation_min <= translation_max).all(dim=1)
    unit_sample = torch.rand(
        translation_min.shape,
        device=segment_poses_w.device,
        dtype=segment_poses_w.dtype,
        generator=generator,
    )
    translation = translation_min + unit_sample * (translation_max - translation_min).clamp_min(0.0)
    return translation, can_fit


def generate_collision_free_cable_poses(
    default_segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    *,
    translation_jitter: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        (-0.02, 0.02),
        (-0.02, 0.02),
    ),
    yaw_jitter: tuple[float, float] = (-0.17453292519943295, 0.17453292519943295),
    rest_length: float = 0.01,
    max_heading_offset: float = 0.08,
    num_shape_modes: int = 3,
    cable_radius: float = 0.003,
    self_clearance: float = 0.00025,
    peg_radius: float = 0.0125,
    fixture_clearance: float = 0.002,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]] | None = ((-0.15, 0.15), (-0.20, 0.20)),
    board_clearance: float = 0.002,
    max_rejection_attempts: int = 512,
    repair_max_heading_offset: float = 0.4,
    repair_num_shape_modes: int = 6,
    repair_yaw_jitter: tuple[float, float] = (-0.05, 0.05),
    winding_radial_cutoff: float = 0.10,
    winding_axial_cutoff: float | None = None,
    max_initial_abs_winding: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate independent, connected, penetration-free and unrouted cable reset states.

    The configured small perturbation is attempted first. Invalid environments
    are then independently rejection-sampled with a richer smooth heading basis
    and translations drawn from their exact board-feasible intervals. Every
    candidate is reconstructed from connected ``rest_length`` segments, so no
    axial strain is introduced. Failure raises instead of writing penetration.

    Returns:
        Tuple ``(segment_poses_w, translation_xy_b, yaw_b)`` for every input
        environment.

    Raises:
        RuntimeError: If any environment has no valid candidate within
            ``max_rejection_attempts``.
    """
    if default_segment_poses_w.ndim != 3 or default_segment_poses_w.shape[-1] != 7:
        raise ValueError(
            f"default_segment_poses_w must have shape (N, S, 7); got {tuple(default_segment_poses_w.shape)}."
        )
    if max_rejection_attempts < 0:
        raise ValueError(f"max_rejection_attempts must be non-negative; got {max_rejection_attempts}.")
    if repair_max_heading_offset < 0.0 or repair_num_shape_modes < 1:
        raise ValueError("Repair heading offset must be non-negative and repair_num_shape_modes must be positive.")
    if (
        not math.isfinite(winding_radial_cutoff)
        or winding_radial_cutoff <= 0.0
        or not math.isfinite(max_initial_abs_winding)
        or max_initial_abs_winding < 0.0
    ):
        raise ValueError("Winding radial cutoff must be positive and maximum initial winding must be non-negative.")
    if winding_axial_cutoff is not None and (not math.isfinite(winding_axial_cutoff) or winding_axial_cutoff <= 0.0):
        raise ValueError("Winding axial cutoff must be None or positive.")
    if self_clearance < 0.0:
        raise ValueError(f"self_clearance must be non-negative; got {self_clearance}.")

    num_envs, num_segments = default_segment_poses_w.shape[:2]
    device, dtype = default_segment_poses_w.device, default_segment_poses_w.dtype
    segment_poses = default_segment_poses_w.clone()
    if max_heading_offset > 0.0:
        heading_offsets = sample_cable_heading_offsets(
            num_envs,
            num_segments,
            max_heading_offset=max_heading_offset,
            num_modes=num_shape_modes,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        segment_poses = shape_cable_poses_planar(segment_poses, rest_length, heading_offsets)
    translation_xy, yaw = sample_board_frame_se2(
        num_envs,
        translation_jitter=translation_jitter,
        yaw_jitter=yaw_jitter,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    segment_poses = transform_cable_poses_se2(segment_poses, env_origins_w, translation_xy, yaw)
    accepted = cable_capsule_clearance_mask(
        segment_poses,
        peg_positions_w,
        env_origins_w,
        rest_length=rest_length,
        cable_radius=cable_radius,
        peg_radius=peg_radius,
        fixture_clearance=fixture_clearance,
        board_bounds_b=board_bounds_b,
        board_clearance=board_clearance,
    )
    accepted &= cable_unrouted_mask(
        segment_poses,
        peg_positions_w,
        radial_cutoff=winding_radial_cutoff,
        axial_cutoff=winding_axial_cutoff,
        max_abs_winding=max_initial_abs_winding,
    )
    accepted &= cable_capsule_self_clearance_mask(
        segment_poses,
        rest_length=rest_length,
        cable_radius=cable_radius,
        self_clearance=self_clearance,
    )

    for attempt in range(max_rejection_attempts):
        remaining = (~accepted).nonzero(as_tuple=False).squeeze(-1)
        if remaining.numel() == 0:
            break

        boundary_sides = ("bottom", "left", "top", "right")
        if attempt < len(boundary_sides) and board_bounds_b is not None and peg_positions_w.shape[1] > 0:
            candidate, can_fit = _generate_boundary_cable_poses(
                default_segment_poses_w[remaining],
                peg_positions_w[remaining],
                env_origins_w[remaining],
                rest_length=rest_length,
                cable_radius=cable_radius,
                self_clearance=self_clearance,
                board_bounds_b=board_bounds_b,
                board_clearance=board_clearance,
                winding_radial_cutoff=winding_radial_cutoff,
                max_initial_abs_winding=max_initial_abs_winding,
                side=boundary_sides[attempt],
                generator=generator,
            )
            candidate_translation = candidate[..., :2].mean(dim=1) - default_segment_poses_w[remaining, ..., :2].mean(
                dim=1
            )
            candidate_yaw = torch.zeros(len(remaining), device=device, dtype=dtype)
        else:
            candidate = default_segment_poses_w[remaining].clone()
            heading_offsets = sample_cable_heading_offsets(
                len(remaining),
                num_segments,
                max_heading_offset=repair_max_heading_offset,
                num_modes=repair_num_shape_modes,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            candidate = shape_cable_poses_planar(candidate, rest_length, heading_offsets)
            _, candidate_yaw = sample_board_frame_se2(
                len(remaining),
                translation_jitter=(0.0, 0.0),
                yaw_jitter=repair_yaw_jitter,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            zero_translation = torch.zeros((len(remaining), 2), device=device, dtype=dtype)
            candidate = transform_cable_poses_se2(
                candidate,
                env_origins_w[remaining],
                zero_translation,
                candidate_yaw,
            )

            if board_bounds_b is None:
                candidate_translation, _ = sample_board_frame_se2(
                    len(remaining),
                    translation_jitter=translation_jitter,
                    yaw_jitter=(0.0, 0.0),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                can_fit = torch.ones(len(remaining), device=device, dtype=torch.bool)
            else:
                candidate_translation, can_fit = _sample_board_fitting_translation(
                    candidate,
                    env_origins_w[remaining],
                    rest_length,
                    board_bounds_b,
                    cable_radius + board_clearance,
                    generator,
                )
            candidate[..., :2] += candidate_translation[:, None, :]
        candidate_valid = can_fit & cable_capsule_clearance_mask(
            candidate,
            peg_positions_w[remaining],
            env_origins_w[remaining],
            rest_length=rest_length,
            cable_radius=cable_radius,
            peg_radius=peg_radius,
            fixture_clearance=fixture_clearance,
            board_bounds_b=board_bounds_b,
            board_clearance=board_clearance,
        )
        candidate_valid &= cable_unrouted_mask(
            candidate,
            peg_positions_w[remaining],
            radial_cutoff=winding_radial_cutoff,
            axial_cutoff=winding_axial_cutoff,
            max_abs_winding=max_initial_abs_winding,
        )
        candidate_valid &= cable_capsule_self_clearance_mask(
            candidate,
            rest_length=rest_length,
            cable_radius=cable_radius,
            self_clearance=self_clearance,
        )
        accepted_rows = remaining[candidate_valid]
        segment_poses[accepted_rows] = candidate[candidate_valid]
        translation_xy[accepted_rows] = candidate_translation[candidate_valid]
        yaw[accepted_rows] = candidate_yaw[candidate_valid]
        accepted[accepted_rows] = True

    if not bool(accepted.all()):
        failed = (~accepted).nonzero(as_tuple=False).squeeze(-1).tolist()
        raise RuntimeError(
            "Unable to generate penetration-free, topologically unrouted cable resets for local environment rows "
            f"{failed} after {max_rejection_attempts} repair attempts."
        )
    return segment_poses, translation_xy, yaw


def transform_cable_poses_se2(
    segment_poses_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    translation_xy_b: torch.Tensor,
    yaw_b: torch.Tensor,
) -> torch.Tensor:
    """Apply one board-frame SE(2) transform to every segment of each cable.

    The board frame is assumed to be axis-aligned with the environment frame
    and to have its origin at :attr:`InteractiveScene.env_origins`. Yaw is
    premultiplied onto each segment orientation.

    Args:
        segment_poses_w: Segment poses ``(x, y, z, qx, qy, qz, qw)`` in world
            frame [m], shape ``(num_envs, num_segments, 7)``.
        env_origins_w: Board-frame origins in world frame [m], shape
            ``(num_envs, 3)``.
        translation_xy_b: Planar translations in board frame [m], shape
            ``(num_envs, 2)``.
        yaw_b: Yaw rotations in board frame [rad], shape ``(num_envs,)``.

    Returns:
        Transformed segment poses with the same shape and dtype as
        ``segment_poses_w``.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(
            f"segment_poses_w must have shape (num_envs, num_segments, 7); got {tuple(segment_poses_w.shape)}."
        )
    num_envs, num_segments = segment_poses_w.shape[:2]
    if env_origins_w.shape != (num_envs, 3):
        raise ValueError(f"env_origins_w must have shape ({num_envs}, 3); got {tuple(env_origins_w.shape)}.")
    if translation_xy_b.shape != (num_envs, 2):
        raise ValueError(f"translation_xy_b must have shape ({num_envs}, 2); got {tuple(translation_xy_b.shape)}.")
    if yaw_b.shape != (num_envs,):
        raise ValueError(f"yaw_b must have shape ({num_envs},); got {tuple(yaw_b.shape)}.")

    transformed = segment_poses_w.clone()
    local_positions = segment_poses_w[..., :3] - env_origins_w[:, None, :]
    cos_yaw = torch.cos(yaw_b)[:, None]
    sin_yaw = torch.sin(yaw_b)[:, None]
    transformed[..., 0] = (
        env_origins_w[:, None, 0]
        + cos_yaw * local_positions[..., 0]
        - sin_yaw * local_positions[..., 1]
        + translation_xy_b[:, None, 0]
    )
    transformed[..., 1] = (
        env_origins_w[:, None, 1]
        + sin_yaw * local_positions[..., 0]
        + cos_yaw * local_positions[..., 1]
        + translation_xy_b[:, None, 1]
    )
    transformed[..., 2] = env_origins_w[:, None, 2] + local_positions[..., 2]

    half_yaw = 0.5 * yaw_b
    yaw_quat = torch.zeros((num_envs, num_segments, 4), device=segment_poses_w.device, dtype=segment_poses_w.dtype)
    yaw_quat[..., 2] = torch.sin(half_yaw)[:, None]
    yaw_quat[..., 3] = torch.cos(half_yaw)[:, None]
    transformed[..., 3:7] = quat_mul(yaw_quat, segment_poses_w[..., 3:7])
    return transformed


def reset_peg_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_names: Sequence[str] = ("peg_0", "peg_1"),
    base_positions_b: Sequence[Sequence[float]] | torch.Tensor | None = None,
    grid_pitch: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reset pegs to base board positions plus nonzero benchmark offsets.

    When ``base_positions_b`` is omitted, each peg's configured default root
    position is used. Explicit base positions may contain x/y or x/y/z. The
    peg orientations and velocities always come from their configured defaults.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices or a full boolean environment mask to reset.
        asset_names: Kinematic rigid-object scene names, one per peg.
        base_positions_b: Base peg positions in board frame [m], shape
            ``(num_pegs, 2 or 3)``. Defaults to the assets' configured poses.
        grid_pitch: Distance represented by one board-grid step [m].
        generator: Optional device-compatible generator for deterministic sampling.

    Returns:
        Applied x/y offsets in board frame [m], shape
        ``(len(env_ids), num_pegs, 2)``.
    """
    env_ids = _resolve_env_ids(env_ids, len(env.scene.env_origins), env.scene.env_origins.device)
    num_reset = int(env_ids.numel())
    num_assets = len(asset_names)
    if num_reset == 0:
        return torch.empty((0, num_assets, 2), device=env.scene.env_origins.device)
    if num_assets == 0:
        raise ValueError("asset_names must contain at least one peg.")

    first_asset = env.scene[asset_names[0]]
    first_default_pose = first_asset.data.default_root_pose.torch
    dtype = first_default_pose.dtype
    device = first_default_pose.device
    env_ids = env_ids.to(device=device)
    env_origins = env.scene.env_origins[env_ids].to(device=device, dtype=dtype)
    offsets = sample_benchmark_grid_offsets(
        num_reset,
        num_assets,
        grid_pitch=grid_pitch,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    bases: torch.Tensor | None = None
    if base_positions_b is not None:
        bases = torch.as_tensor(base_positions_b, device=device, dtype=dtype)
        if bases.ndim != 2 or bases.shape[0] != num_assets or bases.shape[1] not in (2, 3):
            raise ValueError(
                f"base_positions_b must have shape (num_pegs, 2 or 3); got {tuple(bases.shape)} for {num_assets} pegs."
            )

    for asset_index, asset_name in enumerate(asset_names):
        asset = env.scene[asset_name]
        default_pose = asset.data.default_root_pose.torch[env_ids].clone()
        if default_pose.device != device or default_pose.dtype != dtype:
            raise ValueError("All peg assets must use the same device and dtype.")

        local_position = default_pose[:, :3]
        if bases is not None:
            local_position[:, :2] = bases[asset_index, :2]
            if bases.shape[1] == 3:
                local_position[:, 2] = bases[asset_index, 2]
        default_pose[:, :3] = local_position + env_origins
        default_pose[:, :2] += offsets[:, asset_index]
        asset.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=env_ids)

        default_velocity = asset.data.default_root_vel.torch[env_ids].clone()
        asset.write_root_velocity_to_sim_index(root_velocity=default_velocity, env_ids=env_ids)

    return offsets


def reset_cable_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    fixture_asset_names: Sequence[str] = ("peg_0", "peg_1"),
    translation_jitter: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        (-0.02, 0.02),
        (-0.02, 0.02),
    ),
    yaw_jitter: tuple[float, float] = (-0.17453292519943295, 0.17453292519943295),
    rest_length: float = 0.01,
    max_heading_offset: float = 0.08,
    num_shape_modes: int = 3,
    cable_radius: float = 0.003,
    self_clearance: float = 0.00025,
    peg_radius: float = 0.0125,
    fixture_clearance: float = 0.002,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]] | None = ((-0.15, 0.15), (-0.20, 0.20)),
    board_clearance: float = 0.002,
    max_rejection_attempts: int = 512,
    winding_radial_cutoff: float = 0.10,
    winding_axial_cutoff: float | None = None,
    max_initial_abs_winding: float = 0.5,
    generator: torch.Generator | None = None,
    full_scene_replay_command_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore selected cables with penetration-free board-frame variation.

    A command term may instead own an atomic full-scene replay reset. Passing
    its name avoids generating an ordinary cable curve that the command will
    immediately overwrite. The fast path is deliberately structural: it is
    taken only when the resolved term owns a non-``None`` ``reset_replay``
    object whose configuration is enabled. Disabled replay configurations,
    including interactive teleoperation, retain the ordinary cable reset.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices or a full boolean environment mask to reset.
        asset_cfg: Cable scene entity.
        fixture_asset_names: Peg rigid-object scene names to avoid.
        translation_jitter: Uniform board-frame x/y translation ranges [m].
        yaw_jitter: Uniform board-frame yaw range [rad].
        rest_length: Cable segment rest length [m].
        max_heading_offset: Maximum smooth per-segment reset bend [rad].
        num_shape_modes: Number of sine modes used to vary the cable shape.
        cable_radius: Cable capsule radius [m].
        self_clearance: Required gap between non-neighbor cable capsule surfaces [m].
        peg_radius: Peg cylinder radius [m].
        fixture_clearance: Additional cable-to-peg surface clearance [m].
        board_bounds_b: Board x/y bounds [m], or ``None`` to disable containment.
        board_clearance: Additional cable-to-board-edge clearance [m].
        max_rejection_attempts: Per-environment repair candidate budget.
        winding_radial_cutoff: Winding metric's local radial cutoff [m].
        winding_axial_cutoff: Winding metric's local vertical cutoff from the
            peg center [m], or ``None`` for fixtures without a finite axial extent.
        max_initial_abs_winding: Maximum allowed reset winding around any peg [rad].
        generator: Optional device-compatible generator for deterministic sampling.
        full_scene_replay_command_name: Optional command term that owns an
            enabled full-scene ``reset_replay``. When active, return zero reset
            transforms without touching the cable because that command restores
            the complete scene after reset events run.

    Returns:
        Applied ``(translation_xy, yaw)`` tensors. The shapes are
        ``(len(env_ids), 2)`` and ``(len(env_ids),)``.

    Raises:
        RuntimeError: If the cable has no native default segment state.
    """
    if full_scene_replay_command_name is not None:
        command = env.command_manager.get_term(full_scene_replay_command_name)
        replay = getattr(command, "reset_replay", None)
        replay_cfg = getattr(replay, "cfg", None)
        if replay is not None and bool(getattr(replay_cfg, "enabled", False)):
            origins = env.scene.env_origins
            selected_ids = _resolve_env_ids(env_ids, len(origins), origins.device)
            return (
                torch.zeros((len(selected_ids), 2), device=origins.device, dtype=origins.dtype),
                torch.zeros(len(selected_ids), device=origins.device, dtype=origins.dtype),
            )

    cable = env.scene[asset_cfg.name]
    default_pose_proxy = cable.data.default_segment_pose_w
    default_velocity_proxy = cable.data.default_segment_velocity_w
    if default_pose_proxy is None or default_velocity_proxy is None:
        raise RuntimeError(f"Cable asset '{asset_cfg.name}' has no initialized default segment state.")

    default_poses = default_pose_proxy.torch
    device, dtype = default_poses.device, default_poses.dtype
    env_ids = _resolve_env_ids(env_ids, len(default_poses), device)
    num_reset = int(env_ids.numel())
    if num_reset == 0:
        return torch.empty((0, 2), device=device, dtype=dtype), torch.empty((0,), device=device, dtype=dtype)

    segment_poses = default_poses[env_ids].clone()
    segment_velocities = default_velocity_proxy.torch[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids].to(device=device, dtype=dtype)
    if fixture_asset_names:
        peg_positions = torch.stack(
            [
                env.scene[name].data.root_pose_w.torch[env_ids, :3].to(device=device, dtype=dtype)
                for name in fixture_asset_names
            ],
            dim=1,
        )
    else:
        peg_positions = torch.empty((num_reset, 0, 3), device=device, dtype=dtype)

    segment_poses, translation_xy, yaw = generate_collision_free_cable_poses(
        segment_poses,
        peg_positions,
        env_origins,
        translation_jitter=translation_jitter,
        yaw_jitter=yaw_jitter,
        rest_length=rest_length,
        max_heading_offset=max_heading_offset,
        num_shape_modes=num_shape_modes,
        cable_radius=cable_radius,
        self_clearance=self_clearance,
        peg_radius=peg_radius,
        fixture_clearance=fixture_clearance,
        board_bounds_b=board_bounds_b,
        board_clearance=board_clearance,
        max_rejection_attempts=max_rejection_attempts,
        winding_radial_cutoff=winding_radial_cutoff,
        winding_axial_cutoff=winding_axial_cutoff,
        max_initial_abs_winding=max_initial_abs_winding,
        generator=generator,
    )

    cable.write_segment_pose_to_sim_index(segment_pose=segment_poses, env_ids=env_ids)
    cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocities, env_ids=env_ids)
    return translation_xy, yaw
