# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the visual-color randomization writers.

The reset-time visual-color writers (Newton, OVRTX, and the Kit/Replicator backend behind
:class:`~isaaclab.envs.mdp.events.randomize_visual_color`) each sample one color per target visual
prim, then write only the targets whose environment is being reset. The per-environment selection and
shape validation are identical across backends and live here so they can be tested once; each writer
applies the selected colors in its own backend format.
"""

from __future__ import annotations

import re

import torch

_ENV_INDEX_PATTERN = re.compile(r"/env_(\d+)(?:/|$)")


def env_index_from_prim_path(prim_path: str) -> int | None:
    """Return the environment index parsed from a ``/World/envs/env_<i>/...`` prim path, or None.

    Shared by the Replicator (Kit/RTX) and OVRTX writers. The Newton writer parses env indices from
    ``model.shape_label`` with its own configurable regex.

    Args:
        prim_path: A USD prim path string, e.g. ``/World/envs/env_3/Robot/cart/visuals``.

    Returns:
        The integer env index (``3`` in the example) when ``/env_<digits>/`` is present, else ``None``.
    """
    match = _ENV_INDEX_PATTERN.search(prim_path)
    return int(match.group(1)) if match is not None else None


def select_visual_color_targets(
    env_ids: torch.Tensor,
    colors: torch.Tensor,
    env_of_target: list[int | None],
    num_targets: int,
) -> list[int]:
    """Select which visual-color targets to write for a reset subset.

    Args:
        env_ids: Environment indices being reset, shape ``(E,)``. An empty tensor is a no-op.
        colors: Per-target sRGB colors, shape ``(num_targets, 3)``, values in [0, 1].
        env_of_target: Length-``num_targets`` map from a target's index to its environment index
            (``None`` for a target whose environment could not be resolved; never selected).
        num_targets: Expected number of targets (one color sample per target).

    Returns:
        The target indices whose environment is in ``env_ids``, in ascending order (empty when
        ``env_ids`` is empty). Index ``g`` selects both ``colors[g]`` and the writer's ``g``-th target.

    Raises:
        ValueError: If ``colors`` is not shaped ``(num_targets, 3)`` (checked only when ``env_ids``
            is non-empty, so an empty-subset no-op never raises).
    """
    if env_ids.numel() == 0:
        return []
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != num_targets:
        raise ValueError(f"`colors` must have shape ({num_targets}, 3); got {tuple(colors.shape)}.")
    env_set = {int(e) for e in env_ids.detach().to(device="cpu", dtype=torch.long).tolist()}
    return [g for g, env_id in enumerate(env_of_target) if env_id in env_set]
