# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporal filtering of the liquid's render field, independent of MPM physics."""

from __future__ import annotations

import math

import warp as wp
from newton.geometry import ParticleSurface


class SurfaceHistory:
    """Filter density at fixed spatial coordinates before marching cubes.

    Mesh vertex indices change on each extraction, so averaging vertices would
    connect unrelated locations. Instead, match the current and previous native
    sparse fields by voxel coordinates. Newly occupied voxels use their current
    values. Only the render field changes; physical particles remain untouched.

    The single-world presentation uses exact allocations, keeping the previous
    field's volume and values alive until the next update. Rebuildable volumes
    cannot retain history this way because their topology changes in place.

    Args:
        time_constant: Exponential filter time constant [s]; zero disables filtering.
    """

    def __init__(self, time_constant: float = 0.06):
        if not math.isfinite(time_constant) or time_constant < 0.0:
            raise ValueError("Surface smoothing time must be finite and nonnegative.")
        self._time_constant = time_constant
        self._previous: ParticleSurface.SparseField | None = None
        self._coordinates: wp.array | None = None

    def apply(self, field: ParticleSurface.SparseField | None, dt: float) -> None:
        """Filter a newly allocated native field after ``dt`` simulated seconds."""
        previous = self._previous
        if field is not None and previous is not None and self._time_constant > 0.0 and dt > 0.0:
            count = field.volume.get_voxel_count()
            if self._coordinates is None or self._coordinates.size < count:
                self._coordinates = wp.empty(count, dtype=wp.vec3i, device=field.voxel_data.device)
            field.volume.get_voxels(self._coordinates)
            wp.launch(
                _filter_density,
                dim=count,
                inputs=[
                    self._coordinates,
                    field.volume.id,
                    field.voxel_data,
                    previous.volume.id,
                    previous.voxel_data,
                    1.0 - math.exp(-dt / self._time_constant),
                ],
                device=field.voxel_data.device,
            )
        self._previous = field if self._time_constant > 0.0 else None

    def reset(self) -> None:
        """Discard history so an episode reset cannot leave a ghost surface."""
        self._previous = None


@wp.kernel
def _filter_density(
    coordinates: wp.array[wp.vec3i],
    grid: wp.uint64,
    density: wp.array[float],
    previous_grid: wp.uint64,
    previous_density: wp.array[float],
    fraction: float,
):
    coordinate = coordinates[wp.tid()]
    current = wp.volume_lookup_index(grid, coordinate[0], coordinate[1], coordinate[2])
    previous = wp.volume_lookup_index(previous_grid, coordinate[0], coordinate[1], coordinate[2])
    if current >= 0 and previous >= 0:
        density[current] = wp.lerp(previous_density[previous], density[current], fraction)
