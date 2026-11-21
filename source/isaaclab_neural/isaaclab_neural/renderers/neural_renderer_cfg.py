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
class NeuralRendererCfg(RendererCfg):
    """Select the world-model renderer through a camera's ``renderer_cfg`` (the task's ``renderer=`` preset)."""

    class_type: type | str = "{DIR}.neural_renderer:NeuralRenderer"
    renderer_type: str = "neural"

    outputs: dict[RenderBufferKind, RenderBufferSpec] = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
    }
    """Layout of each camera output the world model predicts. Defaults to RGB, with RGBA backing it.

    The model is built only when the scene starts, after the camera configuration is checked, so the outputs are
    declared here; the camera checks them against the model's
    :attr:`~isaaclab_neural.world_model.WorldModelSpec.outputs` and start frames when it binds.
    """

    view: str | None = None
    """The model's view this camera shows, one of :attr:`~isaaclab_neural.world_model.WorldModelSpec.views`. ``None``
    selects the only view of a model that predicts one."""

    provides_frame_history: ClassVar[bool] = True
    """The renderer fills the camera's frame history with each predicted chunk, so the camera does not push its
    captures into it."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab.cloner:UsdReplicateContext",)
    """Camera prims are replicated in USD; there is no physics scene to clone."""

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """The declared :attr:`outputs`; the frames are generated, not rendered from geometry."""
        return dict(self.outputs)
