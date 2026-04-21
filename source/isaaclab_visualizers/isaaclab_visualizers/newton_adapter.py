# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for viewer env selection (Newton viewers and Kit partial USD visibility)."""

from __future__ import annotations


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
