# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the world-model renderer."""

from __future__ import annotations

from typing import ClassVar

import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class WorldModelRendererCfg(RendererCfg):
    """Select the world-model renderer through a camera's ``renderer_cfg`` (the task's ``renderer=`` preset)."""

    class_type: type | str = "{DIR}.world_model_renderer:WorldModelRenderer"
    renderer_type: str = "world_model"

    provides_frame_history: ClassVar[bool] = True
    """The renderer fills the camera's frame history with each predicted chunk, so the camera does not push its
    captures into it."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab.cloner:UsdReplicateContext",)
    """Camera prims are replicated in USD; there is no physics scene to clone."""

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Only RGB is produced; the frames are generated, not rendered from geometry."""
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        }
