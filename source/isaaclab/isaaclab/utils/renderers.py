# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-related utility helpers."""

import os

ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING = "/rtx/scenePartitioning/showAllPartitionsByDefault"
"""RTX setting that exposes partitioned content to cameras without a partition token."""

_SCENE_PARTITION_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


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
