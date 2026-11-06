# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Newton viewer geometry handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

VISUALIZER_INFINITE_PLANE_SIZE = 1000.0
"""Finite render size used for Newton planes encoded as infinite."""


def expand_infinite_plane_scale(
    geo_scale: tuple[float, ...], plane_size: float = VISUALIZER_INFINITE_PLANE_SIZE
) -> tuple[float, ...]:
    """Return a finite visual scale for Newton planes encoded with non-positive extents.

    Newton uses non-positive X/Y plane scale values to represent an effectively
    infinite plane. Newton GL renders those with a large finite mesh; web viewers
    also need a finite size, otherwise their world-extents heuristic can shrink
    the floor to just the actor bounds.
    """
    scale = tuple(float(value) for value in geo_scale)
    width = scale[0] if len(scale) > 0 else 0.0
    length = scale[1] if len(scale) > 1 else 0.0
    if width > 0.0 and length > 0.0:
        return scale
    tail = scale[2:] if len(scale) > 2 else ()
    return (
        width if width > 0.0 else float(plane_size),
        length if length > 0.0 else float(plane_size),
        *tail,
    )


def log_geo_with_expanded_plane_scale(
    super_log_geo: Callable[..., Any],
    plane_geo_type: int,
    name: str,
    geo_type: int,
    geo_scale: tuple[float, ...],
    geo_thickness: float,
    geo_is_solid: bool,
    geo_src=None,
    hidden: bool = False,
):
    """Log geometry after expanding Newton infinite-plane extents for web viewers."""
    if geo_type == plane_geo_type:
        geo_scale = expand_infinite_plane_scale(geo_scale)
    return super_log_geo(name, geo_type, geo_scale, geo_thickness, geo_is_solid, geo_src, hidden)


# TODO: Newton GL's checker floor (GeoType.PLANE, material.z=1.0) renders in the Newton GL
# OpenGL viewport but the streaming view seen in the browser shows composited Isaac Sim
# camera-sensor frames (RTX-rendered).  Patching Newton's GLSL checker_scale only affects
# the OpenGL window, not the camera images, so it has no visible effect from the user's
# perspective.  To fix the floor appearance in the streaming view, the Isaac Sim USD scene
# for each task needs an updated floor material/texture (see the Kuka Allegro env for a
# reference with a blue-grid floor).
