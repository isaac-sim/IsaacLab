# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer-related utility helpers."""

import os


def isaac_rtx_per_env_scene_partition_enabled() -> bool:
    """Return whether per-environment RTX scene partitioning is enabled.

    Partitioning is opt-in: set ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1``
    to enable authoring of ``primvars:omni:scenePartition`` and ``omni:scenePartition``
    on the USD stage.
    """
    return os.environ.get("ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION", "0") == "1"
