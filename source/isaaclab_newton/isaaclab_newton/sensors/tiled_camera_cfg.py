# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TiledCamera configuration that wires in the Newton warp renderer."""

from __future__ import annotations

from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class TiledCameraCfgNewtonWarpRenderer(TiledCameraCfg):
    """TiledCameraCfg variant that uses :class:`NewtonWarpRenderer`.

    The ``renderer_class`` field holds a ``"module:ClassName"`` string that is
    resolved lazily by the environment's ``_setup_scene`` via
    ``isaaclab.utils.string.string_to_callable``.  Using a plain string (rather
    than :class:`DeferredClass`) keeps the field serialisation-friendly and
    prevents ``cfg.validate()`` from recursing into warp type internals.
    """

    renderer_class: str = "isaaclab.renderers.newton_warp_renderer:NewtonWarpRenderer"
