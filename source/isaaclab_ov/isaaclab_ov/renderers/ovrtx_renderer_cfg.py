# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .ovrtx_renderer import OVRTXRenderer


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.

    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    width, height, num_envs, and data_types are obtained from the
    :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` when
    :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data` is called
    (same pattern as Isaac RTX).
    """

    class_type: type[OVRTXRenderer] | str = "{DIR}.ovrtx_renderer:OVRTXRenderer"
    """Renderer implementation class."""

    renderer_type: str = "ovrtx"
    """Type identifier for OVRTX renderer."""

    temp_usd_dir: str | None = None
    """Directory for temporary USD debug dumps written during OVRTX stage preparation.

    When set, the renderer writes ``pre_ovrtx_renderer_stage.usda`` (raw stage before
    partition attributes and export trimming) and ``ovrtx_renderer_stage.usda`` (exported
    stage plus injected render products) under this directory. Must be writable.
    """

    log_level: str = "verbose"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Path for OVRTX log file. Defaults to ``<system temp>/ovrtx_renderer.log``."""

    enable_shadows: bool = False
    """Whether lights cast shadows in RTX Minimal mode. Defaults to False.

    Shadow rays cost render time that rarely changes what a policy learns, so they are turned off
    and opted back into for visually faithful renders.

    Only the ``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl`` and
    ``simple_shading_full_mdl`` data types are affected, because those are the ones that put the
    render product into RTX Minimal mode, whose ``omni:rtx:minimal:castShadows`` switch this drives.
    OVRTX's path-traced modes offer no equivalent switch and always cast shadows, so this setting
    does not change ``rgb`` and the other AOV outputs.
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize semantic segmentation output. Defaults to True.

    If True, semantic IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw semantic IDs are returned as an ``int32`` 1-channel array.

    Regardless of this setting, the semantic ID (or color) to label mapping is exposed via
    ``camera.data.info["semantic_segmentation"]["idToLabels"]``.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize instance segmentation output. Defaults to True.

    If True, instance IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw instance IDs are returned as a ``uint32`` 1-channel array.

    Regardless of this setting, the instance ID (or color) to prim-path mapping is exposed via
    ``camera.data.info["instance_segmentation"]["idToLabels"]`` and the instance ID (or color) to
    semantic-label mapping via ``camera.data.info["instance_segmentation"]["idToSemantics"]``.
    """

    colorize_instance_id_segmentation: bool = True
    """Whether to colorize instance ID segmentation output. Defaults to True.

    If True, instance IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw instance IDs are returned as a ``uint32`` 1-channel array.
    """

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Return the per-output layouts supported by the OVRTX renderer."""

        def segmentation_spec(colorize: bool) -> RenderBufferSpec:
            return RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_FULL_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SEMANTIC_SEGMENTATION: segmentation_spec(self.colorize_semantic_segmentation),
            RenderBufferKind.INSTANCE_SEGMENTATION: segmentation_spec(self.colorize_instance_segmentation),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }
