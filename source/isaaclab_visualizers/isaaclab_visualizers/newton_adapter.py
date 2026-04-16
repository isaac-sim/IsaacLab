# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for viewer env selection (Newton viewers and Kit partial USD visibility)."""

from __future__ import annotations


def resolve_visible_env_indices(
    env_ids: list[int] | None,
    env_selection_max_visible: int | None,
    num_envs: int,
) -> list[int] | None:
    """Resolve which env indices stay visible (same rules as :func:`apply_viewer_visible_worlds`).

    Returns:
        Selected indices, or ``None`` when all environments should be visible.
    """
    if env_ids is not None:
        return list(env_ids)
    if env_selection_max_visible is not None and num_envs > 0:
        n = min(int(env_selection_max_visible), num_envs)
        return list(range(n))
    return None


def apply_viewer_visible_worlds(
    viewer,
    *,
    env_ids: list[int] | None,
    env_selection_max_visible: int | None,
    num_envs: int,
) -> None:
    """Select which simulation worlds are visualized; no-op if the viewer does not support it.

    Prefer this over ``set_model(..., max_worlds=...)`` (deprecated in Newton).

    Args:
        viewer: Newton viewer (ViewerGL, ViewerRerun, ViewerViser, etc.).
        env_ids: Explicit env indices from ``env_selection_*`` config, or ``None`` when showing all
            unless :attr:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg.env_selection_max_visible` limits the count.
        env_selection_max_visible: Optional cap on the number of worlds (``0..num_envs-1``) when ``env_ids`` is
            ``None``.
        num_envs: Total environment count from scene metadata.
    """
    if not hasattr(viewer, "set_visible_worlds"):
        return
    resolved = resolve_visible_env_indices(env_ids, env_selection_max_visible, num_envs)
    if resolved is None:
        viewer.set_visible_worlds(None)
    else:
        viewer.set_visible_worlds(resolved)
