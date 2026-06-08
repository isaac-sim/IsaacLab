# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for viewer env selection (Newton viewers and Kit partial USD visibility)."""

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


def resolve_visible_env_indices(
    env_ids: list[int] | None,
    max_visible_envs: int | None,
    num_envs: int,
) -> list[int] | None:
    """Resolve which env indices stay visible (same rules as :func:`apply_viewer_visible_worlds`).

    * Cap-only path (``env_ids`` is ``None``): contiguous ``0 .. min(cap, num_envs) - 1`` when ``max_visible_envs``
      is set; otherwise ``None`` (viewer shows all worlds). (Random cap-only selection is applied earlier by
      turning it into explicit ``env_ids``.)
    * Explicit path (``env_ids`` is a list): if ``max_visible_envs`` is set, keep only the first *cap* indices
      (truncate from the end); if ``None``, use the full list.

    Returns:
        Selected indices, or ``None`` when all environments should be visible (cap-only, no limit).
    """
    if env_ids is not None:
        out = list(env_ids)
        if max_visible_envs is not None:
            out = out[: max(0, int(max_visible_envs))]
        return out
    if max_visible_envs is not None and num_envs > 0:
        n = min(int(max_visible_envs), num_envs)
        return list(range(n))
    return None


def apply_viewer_visible_worlds(
    viewer,
    *,
    env_ids: list[int] | None,
    max_visible_envs: int | None,
    num_envs: int,
) -> None:
    """Select which simulation worlds are visualized; no-op if the viewer does not support it.

    Prefer this over ``set_model(..., max_worlds=...)`` (deprecated in Newton).

    Args:
        viewer: Newton viewer (ViewerGL, ViewerRerun, ViewerViser, etc.).
        env_ids: Env indices from ``visible_env_indices`` (after validation), or ``None`` for the cap-only
            contiguous path (see ``VisualizerCfg``).
        max_visible_envs: When ``env_ids`` is ``None``, caps the contiguous count; otherwise truncates the list to
            the first *N* indices.
        num_envs: Total environment count from scene metadata.
    """
    if not hasattr(viewer, "set_visible_worlds"):
        return
    resolved = resolve_visible_env_indices(env_ids, max_visible_envs, num_envs)
    if resolved is None:
        viewer.set_visible_worlds(None)
    else:
        viewer.set_visible_worlds(resolved)
