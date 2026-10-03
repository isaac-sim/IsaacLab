# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP configuration and callbacks for visual observation processing."""

from __future__ import annotations

import warp as wp

from isaaclab.renderers import RenderBufferSpec
from isaaclab.sensors.post_processing import CameraPostProcessorContext, SensorPostProcessor, SensorPostProcessorCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

from .cfg import PpispCfg, PpispDiscoveryMode, resolve_and_normalize
from .pipeline import PpispPipeline


def resolve_ppisp_processor(cfg: PpispProcessorCfg, context: CameraPostProcessorContext) -> SensorPostProcessor | None:
    """Resolve camera PPISP attributes and create independent processing callbacks.

    Args:
        cfg: PPISP configuration or camera discovery mode.
        context: Camera batch dimensions, device, and USD discovery context.

    Returns:
        Processor callbacks, or ``None`` when discovery finds no PPISP camera.
    """
    camera_prim_path = context.camera_prim_paths[0] if context.camera_prim_paths else None
    ppisp_cfg = resolve_and_normalize(cfg.isp_cfg, context.stage, camera_prim_path)
    if ppisp_cfg is None:
        return None
    if ppisp_cfg.controller_weights is not None and not wp.get_device(context.device).is_cuda:
        raise ValueError("Camera PPISP controller requires a CUDA device.")

    pipeline = PpispPipeline(ppisp_cfg)
    hdr: wp.array | None = None
    rgba: wp.array | None = None

    def initialize(inputs: dict[str, ProxyArray], outputs: dict[str, ProxyArray]) -> None:
        nonlocal hdr, rgba
        hdr = inputs["rgb_radiance"].warp
        rgba = outputs["rgba"].warp
        pipeline.initialize(hdr)

    def process(env_mask: wp.array) -> None:
        if hdr is None or rgba is None:
            raise RuntimeError("PPISP processor must be initialized before processing frames.")
        # PPISP has no temporal state; processing the persistent batch avoids
        # gathering partial views and allocating intermediate image copies.
        pipeline.apply(hdr, rgba)

    def close() -> None:
        nonlocal hdr, rgba
        hdr = None
        rgba = None
        pipeline.close()

    return SensorPostProcessor(
        inputs=cfg.inputs,
        outputs=cfg.outputs,
        initialize=initialize,
        process=process,
        close=close,
    )


@configclass
class PpispProcessorCfg(SensorPostProcessorCfg):
    """PPISP stage for :class:`~isaaclab.envs.mdp.visual_observations.processed_image`.

    Consumes unexposed scene-linear ``rgb_radiance`` and produces RGB/RGBA with PPISP's camera response
    function (``color_space="camera_response"``). The RGB output aliases the first three channels of RGBA. Inputs
    and intermediate outputs are allocated even when omitted from the camera's
    requested public outputs. Each observation term owns independent processing state.
    """

    func = resolve_ppisp_processor

    inputs: dict[str, RenderBufferSpec] = {
        "rgb_radiance": RenderBufferSpec(channels=3, dtype=wp.float32, color_space="scene_linear"),
    }

    outputs: dict[str, RenderBufferSpec] = {
        "rgba": RenderBufferSpec(channels=4, dtype=wp.uint8, color_space="camera_response"),
        "rgb": RenderBufferSpec(channels=3, dtype=wp.uint8, color_space="camera_response"),
    }

    isp_cfg: PpispCfg | PpispDiscoveryMode | None = PpispDiscoveryMode.AUTO_CAMERA
    """Explicit PPISP values, USD discovery mode, or ``None`` to disable this processor."""
