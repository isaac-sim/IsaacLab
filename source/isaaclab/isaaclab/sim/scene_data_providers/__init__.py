# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for scene data provider implementations.

This sub-package contains the abstract base class and concrete implementations
for providing scene data from various physics backends (Newton, PhysX, etc.)
to visualizers and renderers.

The scene data provider abstraction allows visualizers and renderers to work
with any physics backend without directly coupling to specific backend implementations.
"""

from .scene_data_provider import SceneDataProvider
from .newton_scene_data_provider import NewtonSceneDataProvider

__all__ = [
    "SceneDataProvider",
    "NewtonSceneDataProvider",
]

