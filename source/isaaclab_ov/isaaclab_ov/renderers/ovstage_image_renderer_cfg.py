# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the renderer that reads camera images from the simulation's OVStage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .ovstage_image_renderer import OvstageImageRenderer


@configclass
class OvstageImageRendererCfg(RendererCfg):
    """Copy RGB images that another component wrote into the simulation's OVStage into a camera.

    Select it through a camera's ``renderer_cfg``. See :class:`~isaaclab_ov.renderers.OvstageImageRenderer` for the
    layout a producer writes.
    """

    class_type: type[OvstageImageRenderer] | str = "{DIR}.ovstage_image_renderer:OvstageImageRenderer"

    renderer_type: str = "ovstage_image"

    attribute: str = "sensor:rgb"
    """Attribute of each camera prim that holds its image."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab.cloner:UsdReplicateContext",)
    """Camera prims are replicated in USD; the images come from the producer, not from a native scene."""

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """RGB, with RGBA backing it."""
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
        }
