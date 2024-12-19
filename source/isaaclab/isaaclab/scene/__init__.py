# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing an interactive scene definition.

A scene is a collection of entities (e.g., terrain, articulations, sensors, lights, etc.) that can be added to the
simulation. However, only a subset of these entities are of direct interest for the user to interact with.
For example, the user may want to interact with a robot in the scene, but not with the terrain or the lights.
For this reason, we integrate the different entities into a single class called :class:`InteractiveScene`.

The interactive scene performs the following tasks:

1. It parses the configuration class :class:`InteractiveSceneCfg` to create the scene. This configuration class is
   inherited by the user to add entities to the scene.
2. It clones the entities based on the number of environments specified by the user.
3. It clubs the entities into different groups based on their type (e.g., articulations, sensors, etc.).
4. It provides a set of methods to unify the common operations on the entities in the scene (e.g., resetting internal
   buffers, writing buffers to simulation and updating buffers from simulation).

The interactive scene can be passed around to different modules in the framework to perform different tasks.
For instance, computing the observations based on the state of the scene, or randomizing the scene, or applying
actions to the scene. All these are handled by different "managers" in the framework. Please refer to the
:mod:`isaaclab.managers` sub-package for more details.
"""

from .interactive_scene import InteractiveScene
from .interactive_scene_cfg import InteractiveSceneCfg
