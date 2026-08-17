# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp position projection for planar open-cable reset curves.

The projector is intended for constructing replay-bank states, not for advancing the
physical cable.  It uses fixed-count, double-buffered position-based sweeps so every
vertex in one sweep reads the same previous iterate.  The public API allocates its
temporary buffers, while the launch sequence itself is deterministic and contains no
data-dependent host synchronization.  Main Jacobi sweeps use delayed Chebyshev
extrapolation, followed by a shrink-resistant Taubin smoothing tail and low-gain
unaccelerated cleanup.  CUDA work is enqueued on Torch's current stream so Torch writes,
Warp kernels, and the returned vertices stay ordered without a device-wide synchronize.

Despite the conventional ``xpbd`` module name, this is the zero-compliance geometric
limit of an XPBD-style solver: it does not accumulate constraint multipliers or model a
time step.  This is deliberate for reset-state authoring, where the desired output is a
feasible curve rather than compliant dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp

_TAUBIN_LAMBDA = 0.5
_TAUBIN_MU = -0.53

__all__ = [
    "CableResetCurveXPBDCfg",
    "relax_open_cable_curve_xpbd",
]


@dataclass(frozen=True)
class CableResetCurveXPBDCfg:
    """Configuration for fixed-sweep open-cable position projection.

    The two geometric length scales are intentionally independent.  A cable's
    self-clearance is set by its diameter, while its useful bend radius is usually much
    larger and depends on the reset task.

    Attributes:
        self_separation_distance: Target distance between non-local centerline
            vertices [m].
        bend_radius: Minimum local Menger radius targeted by bend regularization [m].
        iterations: Number of full constraint sweeps.
        neighbor_exclusion: Material-index half-window omitted from self-separation.
        separation_relaxation: Scale applied to the averaged self-separation update.
        length_relaxation: Scale applied to the averaged edge-length update.
        bend_relaxation: Scale applied to the bend-radius update.
        chebyshev_acceleration: Whether to extrapolate Jacobi iterates after warmup.
        chebyshev_warmup: Number of plain sweeps before extrapolation starts.
        chebyshev_rho: Spectral-radius estimate used by the Chebyshev recurrence.
        chebyshev_gamma: Damping applied to the fresh Jacobi displacement.
        taubin_smoothing_passes: Number of shrink-resistant Taubin smoothing passes
            before the cleanup projection.  Each pass applies a lambda and mu half-step.
        cleanup_iterations: Number of unaccelerated cleanup sweeps after the main solve.
        cleanup_length_relaxation: Edge-length relaxation used by cleanup sweeps.
        cleanup_separation_relaxation: Self-separation relaxation used by cleanup sweeps.
        board_margin: Inset added to every board boundary [m].
    """

    self_separation_distance: float
    bend_radius: float
    iterations: int = 50
    neighbor_exclusion: int = 1
    separation_relaxation: float = 1.0
    length_relaxation: float = 1.0
    bend_relaxation: float = 1.0
    chebyshev_acceleration: bool = True
    chebyshev_warmup: int = 8
    chebyshev_rho: float = 0.98
    chebyshev_gamma: float = 0.9
    taubin_smoothing_passes: int = 5
    cleanup_iterations: int = 10
    cleanup_length_relaxation: float = 0.5
    cleanup_separation_relaxation: float = 0.25
    board_margin: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        non_negative_floats = {
            "self_separation_distance": self.self_separation_distance,
            "bend_radius": self.bend_radius,
            "separation_relaxation": self.separation_relaxation,
            "length_relaxation": self.length_relaxation,
            "bend_relaxation": self.bend_relaxation,
            "cleanup_length_relaxation": self.cleanup_length_relaxation,
            "cleanup_separation_relaxation": self.cleanup_separation_relaxation,
            "board_margin": self.board_margin,
        }
        for name, value in non_negative_floats.items():
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real scalar; got {type(value).__name__}.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
        for name, value in (
            ("iterations", self.iterations),
            ("neighbor_exclusion", self.neighbor_exclusion),
            ("chebyshev_warmup", self.chebyshev_warmup),
            ("taubin_smoothing_passes", self.taubin_smoothing_passes),
            ("cleanup_iterations", self.cleanup_iterations),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative; got {value}.")
        if not isinstance(self.chebyshev_acceleration, bool):
            raise TypeError("chebyshev_acceleration must be a bool.")
        if not isinstance(self.chebyshev_rho, int | float) or isinstance(self.chebyshev_rho, bool):
            raise TypeError("chebyshev_rho must be a real scalar.")
        if not 0.0 <= float(self.chebyshev_rho) < 1.0:
            raise ValueError(f"chebyshev_rho must lie in [0, 1); got {self.chebyshev_rho!r}.")
        if not isinstance(self.chebyshev_gamma, int | float) or isinstance(self.chebyshev_gamma, bool):
            raise TypeError("chebyshev_gamma must be a real scalar.")
        if not math.isfinite(float(self.chebyshev_gamma)) or not 0.0 < float(self.chebyshev_gamma) <= 1.0:
            raise ValueError(f"chebyshev_gamma must lie in (0, 1]; got {self.chebyshev_gamma!r}.")
        if self.cleanup_length_relaxation > 1.0:
            raise ValueError("cleanup_length_relaxation must not exceed 1.0.")


@wp.func
def _deterministic_unit(row: int, first: int, second: int, family: int) -> wp.vec2f:
    """Return a reproducible direction for a geometrically degenerate constraint."""
    angle = (
        float(row + 1) * 0.754877666
        + float(first + 1) * 1.324717957
        + float(second + 1) * 2.414213562
        + float(family + 1) * 0.618033989
    )
    return wp.vec2f(wp.cos(angle), wp.sin(angle))


@wp.func
def _is_fixed_vertex(
    waypoint_indices: wp.array(dtype=wp.int32, ndim=2),
    waypoint_mask: wp.array(dtype=wp.int32, ndim=2),
    row: int,
    vertex: int,
    waypoint_count: int,
) -> int:
    fixed = int(0)
    for waypoint in range(waypoint_count):
        if waypoint_mask[row, waypoint] != 0 and waypoint_indices[row, waypoint] == vertex:
            fixed = int(1)
    return fixed


@wp.kernel(enable_backward=False)
def _pin_waypoints_kernel(
    positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_indices: wp.array(dtype=wp.int32, ndim=2),
    waypoint_positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_mask: wp.array(dtype=wp.int32, ndim=2),
    vertex_count: int,
    waypoint_count: int,
):
    tid = wp.tid()
    row = tid // vertex_count
    vertex = tid - row * vertex_count
    target = wp.vec2f(0.0, 0.0)
    count = int(0)
    for waypoint in range(waypoint_count):
        if waypoint_mask[row, waypoint] != 0 and waypoint_indices[row, waypoint] == vertex:
            target = target + waypoint_positions[row, waypoint]
            count += 1
    if count > 0:
        positions[row, vertex] = target / float(count)


@wp.kernel(enable_backward=False)
def _project_open_curve_kernel(
    positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_indices: wp.array(dtype=wp.int32, ndim=2),
    waypoint_positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_mask: wp.array(dtype=wp.int32, ndim=2),
    peg_centers: wp.array(dtype=wp.vec2f, ndim=2),
    peg_radii: wp.array(dtype=wp.float32, ndim=2),
    bounds_lower: wp.array(dtype=wp.vec2f),
    bounds_upper: wp.array(dtype=wp.vec2f),
    vertex_count: int,
    waypoint_count: int,
    peg_count: int,
    neighbor_exclusion: int,
    rest_length: float,
    separation_distance: float,
    bend_radius: float,
    separation_relaxation: float,
    length_relaxation: float,
    bend_relaxation: float,
    out: wp.array(dtype=wp.vec2f, ndim=2),
):
    """Apply one Jacobi sweep over an open chain."""
    tid = wp.tid()
    row = tid // vertex_count
    vertex = tid - row * vertex_count

    fixed_target = wp.vec2f(0.0, 0.0)
    fixed_count = int(0)
    for waypoint in range(waypoint_count):
        if waypoint_mask[row, waypoint] != 0 and waypoint_indices[row, waypoint] == vertex:
            fixed_target = fixed_target + waypoint_positions[row, waypoint]
            fixed_count += 1
    if fixed_count > 0:
        out[row, vertex] = fixed_target / float(fixed_count)
        return

    point = positions[row, vertex]
    epsilon = float(1.0e-9)

    separation = wp.vec2f(0.0, 0.0)
    separation_count = int(0)
    if separation_distance > 0.0 and separation_relaxation > 0.0:
        for other in range(vertex_count):
            material_distance = wp.abs(vertex - other)
            if material_distance > neighbor_exclusion:
                difference = point - positions[row, other]
                distance = wp.length(difference)
                if distance < separation_distance:
                    direction = wp.vec2f(0.0, 0.0)
                    if distance > epsilon:
                        direction = difference / distance
                    else:
                        low = wp.min(vertex, other)
                        high = wp.max(vertex, other)
                        direction = _deterministic_unit(row, low, high, 0)
                        if vertex == low:
                            direction = -direction
                    separation = separation + 0.5 * (separation_distance - distance) * direction
                    separation_count += 1
    if separation_count > 0:
        separation = separation / float(separation_count)

    length_update = wp.vec2f(0.0, 0.0)
    incident_count = int(0)
    if length_relaxation > 0.0:
        if vertex > 0:
            neighbor = vertex - 1
            delta = positions[row, neighbor] - point
            distance = wp.max(wp.length(delta), epsilon)
            neighbor_mobile = 1 - _is_fixed_vertex(waypoint_indices, waypoint_mask, row, neighbor, waypoint_count)
            endpoint_share = 1.0 / float(1 + neighbor_mobile)
            length_update = length_update + endpoint_share * (distance - rest_length) * delta / distance
            incident_count += 1
        if vertex + 1 < vertex_count:
            neighbor = vertex + 1
            delta = positions[row, neighbor] - point
            distance = wp.max(wp.length(delta), epsilon)
            neighbor_mobile = 1 - _is_fixed_vertex(waypoint_indices, waypoint_mask, row, neighbor, waypoint_count)
            endpoint_share = 1.0 / float(1 + neighbor_mobile)
            length_update = length_update + endpoint_share * (distance - rest_length) * delta / distance
            incident_count += 1
    if incident_count > 0:
        length_update = length_update / float(incident_count)

    bend_update = wp.vec2f(0.0, 0.0)
    if bend_radius > 0.0 and bend_relaxation > 0.0 and vertex > 0 and vertex + 1 < vertex_count:
        previous = positions[row, vertex - 1]
        following = positions[row, vertex + 1]
        incoming = point - previous
        outgoing = following - point
        chord = following - previous
        denominator = wp.max(wp.length(incoming) * wp.length(outgoing) * wp.length(chord), 1.0e-12)
        twice_area = wp.abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        curvature = 2.0 * twice_area / denominator
        radius = 1.0 / wp.max(curvature, 1.0e-12)
        if radius < bend_radius:
            midpoint = 0.5 * (previous + following)
            fraction = wp.min(bend_relaxation * (bend_radius - radius) / bend_radius, 1.0)
            bend_update = fraction * (midpoint - point)

    projected = point
    projected = projected + separation_relaxation * separation
    projected = projected + length_relaxation * length_update
    projected = projected + bend_update

    # Obstacles have zero inverse mass, so a cable vertex receives the full projection.
    # Applying disks sequentially is deterministic and converges for the non-overlapping
    # fixture layouts used by this task.
    for peg in range(peg_count):
        center = peg_centers[row, peg]
        radius = peg_radii[row, peg]
        offset = projected - center
        distance = wp.length(offset)
        if radius > 0.0 and distance < radius:
            direction = wp.vec2f(0.0, 0.0)
            if distance > epsilon:
                direction = offset / distance
            else:
                direction = _deterministic_unit(row, vertex, peg, 1)
            projected = center + radius * direction

    lower = bounds_lower[row]
    upper = bounds_upper[row]
    projected[0] = wp.clamp(projected[0], lower[0], upper[0])
    projected[1] = wp.clamp(projected[1], lower[1], upper[1])
    out[row, vertex] = projected


@wp.kernel(enable_backward=False)
def _chebyshev_blend_kernel(
    jacobi: wp.array(dtype=wp.vec2f, ndim=2),
    current: wp.array(dtype=wp.vec2f, ndim=2),
    previous: wp.array(dtype=wp.vec2f, ndim=2),
    omega: float,
    gamma: float,
    out: wp.array(dtype=wp.vec2f, ndim=2),
):
    row, vertex = wp.tid()
    current_point = current[row, vertex]
    previous_point = previous[row, vertex]
    out[row, vertex] = previous_point + omega * (
        gamma * (jacobi[row, vertex] - current_point) + current_point - previous_point
    )


@wp.kernel(enable_backward=False)
def _taubin_open_curve_kernel(
    positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_indices: wp.array(dtype=wp.int32, ndim=2),
    waypoint_positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_mask: wp.array(dtype=wp.int32, ndim=2),
    vertex_count: int,
    waypoint_count: int,
    factor: float,
    out: wp.array(dtype=wp.vec2f, ndim=2),
):
    """Apply one fixed-waypoint-aware Taubin half-step to an open chain."""
    tid = wp.tid()
    row = tid // vertex_count
    vertex = tid - row * vertex_count

    # Pin active waypoints exactly on both half-steps.  Duplicate active entries for
    # one vertex retain the same averaging semantics as the main projection.
    fixed_target = wp.vec2f(0.0, 0.0)
    fixed_count = int(0)
    for waypoint in range(waypoint_count):
        if waypoint_mask[row, waypoint] != 0 and waypoint_indices[row, waypoint] == vertex:
            fixed_target = fixed_target + waypoint_positions[row, waypoint]
            fixed_count += 1
    if fixed_count > 0:
        out[row, vertex] = fixed_target / float(fixed_count)
        return

    # Unlike TrackGen's closed tracks, cable endpoints have no opposite neighbour.
    # Treating them as fixed during smoothing prevents endpoint drift and makes a
    # straight, uniformly sampled open chain an exact fixed point of the tail.
    if vertex == 0 or vertex + 1 == vertex_count:
        out[row, vertex] = positions[row, vertex]
        return

    point = positions[row, vertex]
    laplacian = 0.5 * (positions[row, vertex - 1] + positions[row, vertex + 1]) - point
    out[row, vertex] = point + factor * laplacian


@wp.kernel(enable_backward=False)
def _finalize_fixed_constraints_kernel(
    positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_indices: wp.array(dtype=wp.int32, ndim=2),
    waypoint_positions: wp.array(dtype=wp.vec2f, ndim=2),
    waypoint_mask: wp.array(dtype=wp.int32, ndim=2),
    peg_centers: wp.array(dtype=wp.vec2f, ndim=2),
    peg_radii: wp.array(dtype=wp.float32, ndim=2),
    bounds_lower: wp.array(dtype=wp.vec2f),
    bounds_upper: wp.array(dtype=wp.vec2f),
    vertex_count: int,
    waypoint_count: int,
    peg_count: int,
):
    """End with exact fixed-waypoint, disk, and box constraints."""
    tid = wp.tid()
    row = tid // vertex_count
    vertex = tid - row * vertex_count
    point = positions[row, vertex]
    epsilon = float(1.0e-9)

    target = wp.vec2f(0.0, 0.0)
    target_count = int(0)
    for waypoint in range(waypoint_count):
        if waypoint_mask[row, waypoint] != 0 and waypoint_indices[row, waypoint] == vertex:
            target = target + waypoint_positions[row, waypoint]
            target_count += 1
    if target_count > 0:
        positions[row, vertex] = target / float(target_count)
        return

    for peg in range(peg_count):
        center = peg_centers[row, peg]
        radius = peg_radii[row, peg]
        offset = point - center
        distance = wp.length(offset)
        if radius > 0.0 and distance < radius:
            direction = wp.vec2f(0.0, 0.0)
            if distance > epsilon:
                direction = offset / distance
            else:
                direction = _deterministic_unit(row, vertex, peg, 2)
            point = center + radius * direction

    lower = bounds_lower[row]
    upper = bounds_upper[row]
    point[0] = wp.clamp(point[0], lower[0], upper[0])
    point[1] = wp.clamp(point[1], lower[1], upper[1])
    positions[row, vertex] = point


def _chebyshev_weights(iterations: int, rho: float, warmup: int) -> list[float]:
    """Build the fixed host-side Chebyshev weight sequence."""
    weights = [1.0] * iterations
    rho_squared = rho * rho
    omega = 1.0
    for iteration in range(warmup, iterations):
        accelerated_index = iteration - warmup
        if accelerated_index == 0:
            omega = 1.0
        elif accelerated_index == 1:
            omega = 2.0 / (2.0 - rho_squared)
        else:
            omega = 4.0 / (4.0 - rho_squared * omega)
        weights[iteration] = omega
    return weights


def _normalize_waypoints(
    vertices: torch.Tensor,
    waypoint_vertex_indices: torch.Tensor | None,
    waypoint_positions: torch.Tensor | None,
    waypoint_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Validate waypoint inputs and create graph-friendly non-empty buffers."""
    batch_size, vertex_count = vertices.shape[:2]
    supplied = tuple(value is not None for value in (waypoint_vertex_indices, waypoint_positions, waypoint_mask))
    if any(supplied) and not all(supplied):
        raise ValueError("waypoint indices, positions, and mask must be supplied together.")
    if not any(supplied):
        return (
            torch.zeros((batch_size, 1), device=vertices.device, dtype=torch.int32),
            torch.zeros((batch_size, 1, 2), device=vertices.device, dtype=vertices.dtype),
            torch.zeros((batch_size, 1), device=vertices.device, dtype=torch.int32),
            0,
        )

    assert waypoint_vertex_indices is not None
    assert waypoint_positions is not None
    assert waypoint_mask is not None
    if waypoint_vertex_indices.ndim != 2 or waypoint_vertex_indices.shape[0] != batch_size:
        raise ValueError(f"waypoint_vertex_indices must have shape ({batch_size}, W).")
    waypoint_count = waypoint_vertex_indices.shape[1]
    if waypoint_positions.shape != (batch_size, waypoint_count, 2):
        raise ValueError(f"waypoint_positions must have shape ({batch_size}, {waypoint_count}, 2).")
    if waypoint_mask.shape != (batch_size, waypoint_count):
        raise ValueError(f"waypoint_mask must have shape ({batch_size}, {waypoint_count}).")
    if waypoint_vertex_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("waypoint_vertex_indices must use an integer dtype.")

    indices = waypoint_vertex_indices.to(device=vertices.device, dtype=torch.int32).contiguous()
    positions = waypoint_positions.to(device=vertices.device, dtype=vertices.dtype).contiguous()
    mask_bool = waypoint_mask.to(device=vertices.device, dtype=torch.bool).contiguous()
    if waypoint_count > 0:
        active_indices = indices[mask_bool]
        if active_indices.numel() > 0 and bool(((active_indices < 0) | (active_indices >= vertex_count)).any()):
            raise ValueError("Every active waypoint vertex index must lie inside the open curve.")
        if mask_bool.any() and not bool(torch.isfinite(positions[mask_bool]).all()):
            raise ValueError("Active waypoint positions must be finite.")
    if waypoint_count == 0:
        return (
            torch.zeros((batch_size, 1), device=vertices.device, dtype=torch.int32),
            torch.zeros((batch_size, 1, 2), device=vertices.device, dtype=vertices.dtype),
            torch.zeros((batch_size, 1), device=vertices.device, dtype=torch.int32),
            0,
        )
    indices = torch.where(mask_bool, indices, torch.zeros_like(indices))
    return indices, positions, mask_bool.to(dtype=torch.int32), waypoint_count


def _normalize_pegs(
    vertices: torch.Tensor,
    peg_centers: torch.Tensor | None,
    peg_radii: torch.Tensor | float | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate per-row exclusion disks and create non-empty Warp buffers."""
    batch_size = vertices.shape[0]
    if peg_centers is None:
        if peg_radii is not None:
            raise ValueError("peg_radii cannot be supplied without peg_centers.")
        return (
            torch.zeros((batch_size, 1, 2), device=vertices.device, dtype=vertices.dtype),
            torch.zeros((batch_size, 1), device=vertices.device, dtype=vertices.dtype),
            0,
        )
    if peg_centers.ndim != 3 or peg_centers.shape[0] != batch_size or peg_centers.shape[2] != 2:
        raise ValueError(f"peg_centers must have shape ({batch_size}, P, 2).")
    peg_count = peg_centers.shape[1]
    centers = peg_centers.to(device=vertices.device, dtype=vertices.dtype).contiguous()
    if not bool(torch.isfinite(centers).all()):
        raise ValueError("peg_centers must be finite.")
    if peg_radii is None:
        raise ValueError("peg_radii must be supplied with peg_centers.")
    radii = torch.as_tensor(peg_radii, device=vertices.device, dtype=vertices.dtype)
    if radii.ndim == 0:
        radii = radii.expand(batch_size, peg_count)
    elif radii.ndim == 1 and radii.shape == (peg_count,):
        radii = radii[None].expand(batch_size, -1)
    elif radii.shape != (batch_size, peg_count):
        raise ValueError(f"peg_radii must be scalar or have shape ({peg_count},) or ({batch_size}, {peg_count}).")
    radii = radii.contiguous()
    if not bool(torch.isfinite(radii).all()) or bool((radii < 0.0).any()):
        raise ValueError("peg_radii must be finite and non-negative.")
    if peg_count == 0:
        return (
            torch.zeros((batch_size, 1, 2), device=vertices.device, dtype=vertices.dtype),
            torch.zeros((batch_size, 1), device=vertices.device, dtype=vertices.dtype),
            0,
        )
    return centers, radii, peg_count


def _normalize_bounds(
    vertices: torch.Tensor,
    board_bounds: torch.Tensor | tuple[tuple[float, float], tuple[float, float]],
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert axis-major board intervals into per-row lower and upper corners."""
    batch_size = vertices.shape[0]
    bounds = torch.as_tensor(board_bounds, device=vertices.device, dtype=vertices.dtype)
    if bounds.shape == (2, 2):
        bounds = bounds[None].expand(batch_size, -1, -1)
    elif bounds.shape != (batch_size, 2, 2):
        raise ValueError(f"board_bounds must have shape (2, 2) or ({batch_size}, 2, 2).")
    if not bool(torch.isfinite(bounds).all()):
        raise ValueError("board_bounds must be finite.")
    lower = torch.stack((bounds[:, 0, 0], bounds[:, 1, 0]), dim=-1) + margin
    upper = torch.stack((bounds[:, 0, 1], bounds[:, 1, 1]), dim=-1) - margin
    if bool((lower >= upper).any()):
        raise ValueError("board_bounds must contain ordered intervals wider than twice board_margin.")
    return lower.contiguous(), upper.contiguous()


def relax_open_cable_curve_xpbd(
    vertices: torch.Tensor,
    *,
    rest_length: float,
    board_bounds: torch.Tensor | tuple[tuple[float, float], tuple[float, float]],
    cfg: CableResetCurveXPBDCfg,
    waypoint_vertex_indices: torch.Tensor | None = None,
    waypoint_positions: torch.Tensor | None = None,
    waypoint_mask: torch.Tensor | None = None,
    peg_centers: torch.Tensor | None = None,
    peg_radii: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Relax batched planar open-cable vertices with deterministic Warp sweeps.

    Active waypoints are hard positional constraints and take precedence over fixture
    and board projection.  ``peg_radii`` are centerline exclusion radii [m]; callers
    should therefore provide the physical peg radius plus cable radius and desired
    surface clearance.

    Edge-length projection converges toward ``rest_length`` but intentionally does not
    claim exact inextensibility. The caller applies bend-limited exact-length heading
    reintegration with a 12 mm gate for the task's 10 mm edges, then revalidates because
    reintegration can reintroduce fixture or self penetration. Likewise, the returned
    the caller performs continuous capsule and replay-forward validation before a curve
    enters the reset bank.

    Args:
        vertices: Initial planar vertices [m], shape ``(B, V, 2)``, float32 on CPU or CUDA.
        rest_length: Authored cable edge rest length [m].
        board_bounds: Axis-major ``((xmin, xmax), (ymin, ymax))`` bounds [m], shared
            with shape ``(2, 2)`` or per-row with shape ``(B, 2, 2)``.
        cfg: Fixed-sweep solver configuration.
        waypoint_vertex_indices: Padded vertex indices, shape ``(B, W)``.
        waypoint_positions: Padded fixed positions [m], shape ``(B, W, 2)``.
        waypoint_mask: Active waypoint mask, shape ``(B, W)``.
        peg_centers: Per-row exclusion-disk centers [m], shape ``(B, P, 2)``.
        peg_radii: Centerline exclusion radii [m], scalar or shape ``(P,)`` or ``(B, P)``.

    Returns:
        Relaxed vertices with shape ``(B, V, 2)``, detached from autograd.
    """
    if not isinstance(vertices, torch.Tensor):
        raise TypeError(f"vertices must be a torch.Tensor; got {type(vertices).__name__}.")
    if vertices.ndim != 3 or vertices.shape[-1] != 2 or vertices.shape[1] < 2:
        raise ValueError(f"vertices must have shape (B, V, 2) with V >= 2; got {tuple(vertices.shape)}.")
    if vertices.shape[0] < 1:
        raise ValueError("vertices must contain at least one batch row.")
    if vertices.dtype != torch.float32:
        raise TypeError(f"vertices must use torch.float32 for Warp vec2f interop; got {vertices.dtype}.")
    if vertices.device.type not in ("cpu", "cuda"):
        raise ValueError(f"vertices must be on a CPU or CUDA device; got {vertices.device}.")
    if not bool(torch.isfinite(vertices).all()):
        raise ValueError("vertices must be finite.")
    if not isinstance(rest_length, int | float) or isinstance(rest_length, bool):
        raise TypeError("rest_length must be a real scalar.")
    if not math.isfinite(float(rest_length)) or rest_length <= 0.0:
        raise ValueError(f"rest_length must be finite and positive; got {rest_length!r}.")
    if not isinstance(cfg, CableResetCurveXPBDCfg):
        raise TypeError(f"cfg must be CableResetCurveXPBDCfg; got {type(cfg).__name__}.")

    wp.init()
    batch_size, vertex_count = vertices.shape[:2]
    indices, targets, mask, waypoint_count = _normalize_waypoints(
        vertices, waypoint_vertex_indices, waypoint_positions, waypoint_mask
    )
    centers, radii, peg_count = _normalize_pegs(vertices, peg_centers, peg_radii)
    bounds_lower, bounds_upper = _normalize_bounds(vertices, board_bounds, cfg.board_margin)

    current_tensor = vertices.detach().clone().contiguous()
    next_tensor = torch.empty_like(current_tensor)
    previous_tensor = torch.empty_like(current_tensor)
    current = wp.from_torch(current_tensor, dtype=wp.vec2f)
    next_positions = wp.from_torch(next_tensor, dtype=wp.vec2f)
    previous = wp.from_torch(previous_tensor, dtype=wp.vec2f)
    waypoint_indices_wp = wp.from_torch(indices, dtype=wp.int32)
    waypoint_positions_wp = wp.from_torch(targets, dtype=wp.vec2f)
    waypoint_mask_wp = wp.from_torch(mask, dtype=wp.int32)
    peg_centers_wp = wp.from_torch(centers, dtype=wp.vec2f)
    peg_radii_wp = wp.from_torch(radii, dtype=wp.float32)
    bounds_lower_wp = wp.from_torch(bounds_lower, dtype=wp.vec2f)
    bounds_upper_wp = wp.from_torch(bounds_upper, dtype=wp.vec2f)
    warp_device = current.device
    # ``wp.from_torch`` shares storage without inserting a stream dependency.
    # Launch on Torch's current stream so preceding Torch writes are ordered
    # without a full-device synchronization.
    launch_stream = wp.stream_from_torch(vertices.device) if vertices.device.type == "cuda" else None
    flat_dimension = batch_size * vertex_count

    wp.launch(
        _pin_waypoints_kernel,
        dim=flat_dimension,
        inputs=[current, waypoint_indices_wp, waypoint_positions_wp, waypoint_mask_wp, vertex_count, waypoint_count],
        device=warp_device,
        stream=launch_stream,
    )
    weights = _chebyshev_weights(cfg.iterations, float(cfg.chebyshev_rho), cfg.chebyshev_warmup)
    for iteration in range(cfg.iterations):
        wp.launch(
            _project_open_curve_kernel,
            dim=flat_dimension,
            inputs=[
                current,
                waypoint_indices_wp,
                waypoint_positions_wp,
                waypoint_mask_wp,
                peg_centers_wp,
                peg_radii_wp,
                bounds_lower_wp,
                bounds_upper_wp,
                vertex_count,
                waypoint_count,
                peg_count,
                cfg.neighbor_exclusion,
                float(rest_length),
                float(cfg.self_separation_distance),
                float(cfg.bend_radius),
                float(cfg.separation_relaxation),
                float(cfg.length_relaxation),
                float(cfg.bend_relaxation),
                next_positions,
            ],
            device=warp_device,
            stream=launch_stream,
        )
        accelerated = cfg.chebyshev_acceleration and iteration >= cfg.chebyshev_warmup
        if accelerated:
            blend_previous = current if iteration == cfg.chebyshev_warmup else previous
            wp.launch(
                _chebyshev_blend_kernel,
                dim=(batch_size, vertex_count),
                inputs=[
                    next_positions,
                    current,
                    blend_previous,
                    float(weights[iteration]),
                    float(cfg.chebyshev_gamma),
                    next_positions,
                ],
                device=warp_device,
                stream=launch_stream,
            )
            current, previous, next_positions = next_positions, current, previous
            current_tensor, previous_tensor, next_tensor = next_tensor, current_tensor, previous_tensor
        else:
            current, next_positions = next_positions, current
            current_tensor, next_tensor = next_tensor, current_tensor

    # The Chebyshev-accelerated constraint phase is followed by a shrink-resistant
    # Taubin tail before spacing is restored. One pass is a positive Laplacian
    # half-step followed by a slightly stronger negative one. Both launches use
    # Jacobi ping-pong semantics, keep open endpoints stable, and re-pin active
    # waypoints exactly. Obstacles and board bounds are projected by every cleanup
    # sweep and receive final authority below even when cleanup is disabled.
    for _ in range(cfg.taubin_smoothing_passes):
        wp.launch(
            _taubin_open_curve_kernel,
            dim=flat_dimension,
            inputs=[
                current,
                waypoint_indices_wp,
                waypoint_positions_wp,
                waypoint_mask_wp,
                vertex_count,
                waypoint_count,
                float(_TAUBIN_LAMBDA),
                next_positions,
            ],
            device=warp_device,
            stream=launch_stream,
        )
        current, next_positions = next_positions, current
        current_tensor, next_tensor = next_tensor, current_tensor
        wp.launch(
            _taubin_open_curve_kernel,
            dim=flat_dimension,
            inputs=[
                current,
                waypoint_indices_wp,
                waypoint_positions_wp,
                waypoint_mask_wp,
                vertex_count,
                waypoint_count,
                float(_TAUBIN_MU),
                next_positions,
            ],
            device=warp_device,
            stream=launch_stream,
        )
        current, next_positions = next_positions, current
        current_tensor, next_tensor = next_tensor, current_tensor

    # Unaccelerated low-gain sweeps remove extrapolation and smoothing residue and
    # improve spacing without re-introducing high-frequency bend motion.
    for _ in range(cfg.cleanup_iterations):
        wp.launch(
            _project_open_curve_kernel,
            dim=flat_dimension,
            inputs=[
                current,
                waypoint_indices_wp,
                waypoint_positions_wp,
                waypoint_mask_wp,
                peg_centers_wp,
                peg_radii_wp,
                bounds_lower_wp,
                bounds_upper_wp,
                vertex_count,
                waypoint_count,
                peg_count,
                cfg.neighbor_exclusion,
                float(rest_length),
                float(cfg.self_separation_distance),
                0.0,
                float(cfg.cleanup_separation_relaxation),
                float(cfg.cleanup_length_relaxation),
                0.0,
                next_positions,
            ],
            device=warp_device,
            stream=launch_stream,
        )
        current, next_positions = next_positions, current
        current_tensor, next_tensor = next_tensor, current_tensor

    wp.launch(
        _finalize_fixed_constraints_kernel,
        dim=flat_dimension,
        inputs=[
            current,
            waypoint_indices_wp,
            waypoint_positions_wp,
            waypoint_mask_wp,
            peg_centers_wp,
            peg_radii_wp,
            bounds_lower_wp,
            bounds_upper_wp,
            vertex_count,
            waypoint_count,
            peg_count,
        ],
        device=warp_device,
        stream=launch_stream,
    )

    return current_tensor
