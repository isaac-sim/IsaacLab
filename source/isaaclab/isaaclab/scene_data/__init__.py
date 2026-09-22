# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing the scene data provider and backend interface.

The :class:`SceneDataProvider` bridges physics simulation backends and the
consumers that read scene transforms (renderers and visualizers). Physics
backends implement :class:`SceneDataBackend` to expose their current
transforms in one of the :class:`SceneDataFormat` Warp struct variants;
the provider converts and remaps them on demand for each consumer.

This package is deliberately separate from :mod:`isaaclab.scene` so that
physics backends (``isaaclab_physx``, ``isaaclab_newton``) can subclass
:class:`SceneDataBackend` without pulling :mod:`isaaclab.scene` into the
``AppLauncher`` pre-launch import chain.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
