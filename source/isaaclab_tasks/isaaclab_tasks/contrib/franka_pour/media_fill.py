# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Granular-media fill for the Franka pour source bowl.

:func:`cube_fill_points` builds a deterministic, jittered axis-aligned lattice
clipped to the analytic hollow-cube bowl's inner cavity.

Particles must be seeded as a jittered lattice at ``spacing = voxel_size / particles_per_cell`` and
inset from the walls by ``clearance``: a particle spawned inside the grid-level collider shell is
ejected on the first solve, and any overlap explodes at the near-incompressible MPM stiffness.
"""

from __future__ import annotations

import numpy as np

# Default wall inset: at least one particle spacing, and clear of the collider margin band.
_DEFAULT_MARGIN = 0.002


def _resolve_clearance(spacing: float, clearance: float | None) -> float:
    return float(clearance) if clearance is not None else max(float(spacing), 3.0 * _DEFAULT_MARGIN)


def _fill_region(
    inner_lo: np.ndarray,
    inner_hi: np.ndarray,
    spacing: float,
    fill_frac: float,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the inset ``(region_lo, region_hi)`` the lattice fills."""
    inner_lo = np.asarray(inner_lo, dtype=np.float64)
    inner_hi = np.asarray(inner_hi, dtype=np.float64)
    cavity_h = float(inner_hi[2] - inner_lo[2])
    fill_depth = max(0.0, min(float(fill_frac) * cavity_h, cavity_h - 2.0 * clearance))
    region_lo = np.array([inner_lo[0] + clearance, inner_lo[1] + clearance, inner_lo[2] + clearance])
    region_hi = np.array([inner_hi[0] - clearance, inner_hi[1] - clearance, inner_lo[2] + clearance + fill_depth])
    return region_lo, region_hi


def _axis_samples(lo: float, hi: float, spacing: float) -> np.ndarray:
    """Regularly spaced samples in ``[lo, hi]`` (at least one, centred when the span is short)."""
    span = hi - lo
    if span <= 0.0:
        return np.array([0.5 * (lo + hi)], dtype=np.float64)
    n = int(np.floor(span / spacing))
    coords = lo + spacing * np.arange(n + 1, dtype=np.float64)
    # Centre the lattice in the span so both walls get equal clearance.
    coords = coords + 0.5 * (span - spacing * n)
    return coords


def expected_fill_count(
    inner_lo: np.ndarray,
    inner_hi: np.ndarray,
    spacing: float,
    fill_frac: float = 1.0,
    clearance: float | None = None,
) -> int:
    """Analytic particle count :func:`cube_fill_points` will produce (for sizing/asserts)."""
    clr = _resolve_clearance(spacing, clearance)
    region_lo, region_hi = _fill_region(inner_lo, inner_hi, spacing, fill_frac, clr)
    counts = [len(_axis_samples(region_lo[a], region_hi[a], float(spacing))) for a in range(3)]
    return int(counts[0] * counts[1] * counts[2])


def cube_fill_points(
    inner_lo: np.ndarray,
    inner_hi: np.ndarray,
    spacing: float,
    fill_frac: float = 1.0,
    clearance: float | None = None,
    jitter: float = 0.05,
    seed: int = 7,
) -> np.ndarray:
    """Build a jittered lattice of particle positions filling a box cavity.

    Args:
        inner_lo: Cavity floor corner ``(3,)`` [m] (e.g. from
            :func:`.cube_bowl_mesh.cube_bowl_inner_bounds`).
        inner_hi: Cavity rim corner ``(3,)`` [m].
        spacing: Particle lattice spacing [m] (``voxel_size / particles_per_cell``).
        fill_frac: Fraction of the cavity height to fill (1.0 = up to the rim, capped to leave
            ``clearance`` below the rim).
        clearance: Wall/floor inset [m]; defaults to ``max(spacing, 3 * 0.002)``.
        jitter: Uniform per-particle jitter as a fraction of ``spacing`` (0 = a perfect lattice).
        seed: RNG seed; identical ``seed`` gives identical points (per-env determinism).

    Returns:
        ``(K, 3)`` float32 particle positions in the bowl local frame.
    """
    clr = _resolve_clearance(spacing, clearance)
    region_lo, region_hi = _fill_region(inner_lo, inner_hi, spacing, fill_frac, clr)
    axes = [_axis_samples(region_lo[a], region_hi[a], float(spacing)) for a in range(3)]
    grid = np.stack(np.meshgrid(axes[0], axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 3)
    if jitter > 0.0:
        rng = np.random.default_rng(int(seed))
        grid = grid + (rng.random(grid.shape) - 0.5) * 2.0 * float(jitter) * float(spacing)
    return grid.astype(np.float32)
