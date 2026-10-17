# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-related utility helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING = "/rtx/scenePartitioning/showAllPartitionsByDefault"
"""RTX setting that exposes partitioned content to cameras without a partition token."""

_SCENE_PARTITION_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


def scene_partition_attributes(
    stage: Usd.Stage,
    env_paths: Sequence[str],
    *,
    copies: Sequence[tuple[str, Sequence[str]]] = (),
) -> dict[str, dict[str, str]]:
    """Resolve geometry, camera, and light partitions, including detached scene copies.

    Geometry inherits an environment-root primvar. Cameras and lights require a direct
    selector, which must be reassigned when a native clone copies another world's token.
    Prims outside the environment roots retain their authored, shared-light behavior.

    Args:
        stage: Authoring stage containing the clone prototypes.
        env_paths: Environment root paths in world-index order.
        copies: Source subtrees and their native destination paths, if any.

    Returns:
        Attribute names mapped to prim-path/partition-token assignments.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    partitions = {Sdf.Path(path): f"env_{i}" for i, path in enumerate(env_paths)}
    selectors: dict[str, str] = {}
    if copies:
        roots = copies
    else:
        roots = [(path, ()) for path in env_paths if stage.GetPrimAtPath(path)]
    for source, destinations in roots:
        root = stage.GetPrimAtPath(source)
        if not root:
            raise ValueError(f"Scene partition source {source!r} does not exist.")
        for prim in Usd.PrimRange(root):
            if not (prim.IsA(UsdGeom.Camera) or prim.HasAPI(UsdLux.LightAPI)):
                continue
            path = prim.GetPath()
            targets = [path]
            targets.extend(path.ReplacePrefix(root.GetPath(), Sdf.Path(dst)) for dst in destinations)
            for target in targets:
                ancestor = target
                while ancestor != Sdf.Path.absoluteRootPath:
                    if ancestor in partitions:
                        selectors[str(target)] = partitions[ancestor]
                        break
                    ancestor = ancestor.GetParentPath()
    return {
        "primvars:omni:scenePartition": {str(path): token for path, token in partitions.items()},
        "omni:scenePartition": selectors,
    }


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
