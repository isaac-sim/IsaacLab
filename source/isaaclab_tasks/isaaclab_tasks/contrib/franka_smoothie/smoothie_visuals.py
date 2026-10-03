# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-only tap stream and cup fill; these meshes never enter the physics model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .smoothie_asset import NOZZLE_POSITION_M

FILL_RADIUS_M = 0.039
FILL_BOTTOM_LOCAL_M = 0.012
FILL_TOP_LOCAL_M = 0.145
STREAM_RADIUS_M = 0.004
STREAM_END_WORLD_Z_M = 0.18
MILK_COLOR = (0.95, 0.96, 0.88)


def _cylinder(radius: float, bottom: float, top: float, segments: int = 24) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(segments) * (2 * math.pi / segments)
    xy = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.concatenate(
        (
            np.column_stack((xy, np.full(segments, bottom))),
            np.column_stack((xy, np.full(segments, top))),
            [[0, 0, bottom], [0, 0, top]],
        )
    )
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(
            (
                (i, j, segments + j),
                (i, segments + j, segments + i),
                (2 * segments, j, i),
                (2 * segments + 1, segments + i, segments + j),
            )
        )
    return points.astype(np.float32), np.asarray(faces, dtype=np.int32).reshape(-1)


def fill_vertices(cup_pose: np.ndarray, fill_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return renderer vertices [m] from a measured cup pose [xyz, xyzw] and fill fraction.

    The decorative fill follows the cup, including after inversion; no fluid
    dynamics, mass, contacts, particle emission, or rigid-state writes occur.
    """
    pose = np.asarray(cup_pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isfinite(fill_level):
        raise ValueError("Expected a finite cup pose [xyz, xyzw] and fill fraction.")
    norm = np.linalg.norm(pose[3:])
    if norm <= 1e-12 or not 0.0 <= fill_level <= 1.0:
        raise ValueError("Require a nonzero quaternion and fill fraction in [0, 1].")
    x, y, z, w = pose[3:] / norm
    rotation = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    height = FILL_BOTTOM_LOCAL_M + max(fill_level, 1e-5) * (FILL_TOP_LOCAL_M - FILL_BOTTOM_LOCAL_M)
    points, faces = _cylinder(FILL_RADIUS_M, FILL_BOTTOM_LOCAL_M, height)
    return (points @ rotation.T + pose[:3]).astype(np.float32), faces


class SmoothieVisuals:
    """Render a decorative stream and fill through NewtonGL without adding simulation bodies.

    Call :meth:`update` immediately before ``viewer.step(dt)``. The installed
    Isaac Lab wrapper exposes its native Newton viewer through ``_viewer``;
    direct native NewtonGL instances are also accepted. This small adapter is
    the only place that accesses that wrapper detail.
    """

    def __init__(self, name: str = "tap_demo") -> None:
        self.name = name
        self._raw = None
        self._meshes = {}
        self._last_fill = None
        self._last_tap_on = None

    def update(self, viewer: Any, cup_pose: np.ndarray, fill_level: float, tap_on: bool) -> None:
        """Update only renderer meshes using cup pose [m, xyzw] and a visual fill fraction."""
        import warp as wp

        raw = getattr(viewer, "_viewer", viewer)
        if raw is None or not hasattr(raw, "log_mesh"):
            raise TypeError("SmoothieVisuals requires an initialized NewtonGL viewer.")
        if self._raw is not None and raw is not self._raw:
            raise ValueError("Create one SmoothieVisuals instance per viewer.")
        self._raw = raw
        pose = np.asarray(cup_pose, dtype=np.float64)
        points, indices = fill_vertices(pose, fill_level)
        signature = (tuple(pose), float(fill_level))
        if signature != self._last_fill:
            self._draw(raw, wp, "fill", points, indices, hidden=fill_level <= 0.0)
            self._last_fill = signature
        if bool(tap_on) != self._last_tap_on:
            stream, triangles = _cylinder(STREAM_RADIUS_M, STREAM_END_WORLD_Z_M, NOZZLE_POSITION_M[2], 12)
            stream[:, :2] += np.asarray(NOZZLE_POSITION_M[:2])
            self._draw(raw, wp, "stream", stream, triangles, hidden=not tap_on)
            self._last_tap_on = bool(tap_on)

    def _draw(self, raw, wp, suffix, points, indices, *, hidden):
        name = f"{self.name}/{suffix}"
        if suffix not in self._meshes:
            self._meshes[suffix] = (
                wp.array(points, dtype=wp.vec3, device=raw.device),
                wp.array(indices, dtype=wp.int32, device=raw.device),
            )
        vertices, faces = self._meshes[suffix]
        vertices.assign(points)
        raw.log_mesh(
            name,
            vertices,
            faces,
            color=MILK_COLOR,
            roughness=0.25,
            metallic=0.0,
            dynamic=True,
            backface_culling=False,
            hidden=hidden,
        )

    def close(self) -> None:
        """Hide this helper's two renderer meshes; physics remains untouched."""
        if self._raw is not None:
            for suffix in self._meshes:
                mesh = self._raw.objects.get(f"{self.name}/{suffix}")
                if mesh is not None:
                    mesh.hidden = True
        self._meshes.clear()
