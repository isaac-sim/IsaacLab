# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data providers for visualizers and renderers."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .physx_scene_data_provider import PhysxSceneDataProvider
    from .scene_data_provider import SceneDataProvider

from isaaclab.utils.module import lazy_export

lazy_export(
    ("physx_scene_data_provider", "PhysxSceneDataProvider"),
    ("scene_data_provider", "SceneDataProvider"),
)
