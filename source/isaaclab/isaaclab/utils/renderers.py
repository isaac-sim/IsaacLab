# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-related utility helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)

ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING = "/rtx/scenePartitioning/showAllPartitionsByDefault"
"""RTX setting that exposes partitioned content to cameras without a partition token."""

_SCENE_PARTITION_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


def _resolve_env_light_partitions(
    stage: Usd.Stage,
    env_paths: Sequence[str],
    *,
    copies: Sequence[tuple[str, Sequence[str]]] = (),
) -> dict[str, str]:
    """Resolve scene-partition tokens for environment-owned lights and their copies."""
    from pxr import Sdf, Usd, UsdLux

    partitions = {Sdf.Path(path): f"env_{index}" for index, path in enumerate(env_paths)}
    roots: dict[Sdf.Path, list[Sdf.Path]] = {path: [] for path in partitions if stage.GetPrimAtPath(path)}
    for source, destinations in copies:
        source_path = Sdf.Path(source)
        if not stage.GetPrimAtPath(source_path):
            raise ValueError(f"Scene partition source {source!r} does not exist.")
        targets = roots.setdefault(source_path, [])
        targets.extend(Sdf.Path(destination) for destination in destinations if Sdf.Path(destination) not in targets)

    assignments: dict[str, str] = {}
    has_instance_proxy = False
    for source_path, destinations in roots.items():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(source_path), Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            path = prim.GetPath()
            targets = (path, *(path.ReplacePrefix(source_path, destination) for destination in destinations))
            resolved = []
            for target in targets:
                ancestor = target
                while ancestor != Sdf.Path.absoluteRootPath:
                    if ancestor in partitions:
                        resolved.append((str(target), partitions[ancestor]))
                        break
                    ancestor = ancestor.GetParentPath()
            if prim.IsInstanceProxy():
                if resolved:
                    has_instance_proxy = True
            else:
                assignments.update(resolved)

    if has_instance_proxy:
        logger.warning(
            "Lights inside USD instances cannot be assigned to environment scene partitions and may affect other"
            " environments."
        )
    return assignments


def isaac_rtx_per_env_scene_partition_enabled() -> bool:
    """Return the legacy environment-derived default for Isaac RTX scene partitioning.

    :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg` uses this helper as
    the construction default for ``enable_scene_partitioning``. Partitioning
    is enabled when the environment variable is absent. Set
    ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0`` to disable authoring of
    ``primvars:omni:scenePartition`` and ``omni:scenePartition`` on the USD stage.
    This setting does not affect OVRTX scene partitioning.

    Raises:
        ValueError: If the environment variable is set to a value other than ``"0"`` or ``"1"``.
    """
    value = os.environ.get(_SCENE_PARTITION_ENV_VAR)
    if value is None:
        return True
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid value for {_SCENE_PARTITION_ENV_VAR}: {value!r}. Expected '0' or '1'.")
    return value == "1"
