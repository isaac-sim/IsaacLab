# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP operation for camera and observation render-buffer pipes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import warp as wp

from isaaclab.utils import configclass
from isaaclab.utils.images import PreparedBufferOperation, RenderBufferStepCfg
from isaaclab.utils.warp import ProxyArray

from .cfg import PpispCfg
from .pipeline import PpispControllerWeights, PpispPipeline


def prepare_ppisp(
    inputs: Mapping[str, ProxyArray],
    *,
    cfg: PpispCfg,
    channels: int = 3,
    controller_weights: PpispControllerWeights | None = None,
) -> PreparedBufferOperation:
    """Prepare PPISP and allocate its persistent output from an actual radiance buffer.

    The producer must supply scene-linear color before camera exposure and response. This semantic
    requirement cannot be inferred from the array's dtype, shape, or pixel values.

    Args:
        inputs: One ``radiance`` input: float32 with shape ``(N, H, W, 3)``.
        cfg: Concrete configuration. Resolve USD camera attributes before preparing the pipe.
        channels: Number of output channels, three for RGB or four for opaque RGBA.
        controller_weights: Optional read-only device weights shared across operations. Each operation
            retains its own mutable controller workspace; weights must be ready on the calling stream.

    Returns:
        A prepared operation owning a uint8 ``color`` output with three or four channels.
    """
    if set(inputs) != {"radiance"}:
        raise ValueError("PPISP requires exactly one input role named 'radiance'.")
    source = inputs["radiance"]
    if not isinstance(source, ProxyArray):
        raise TypeError("PPISP radiance must be a ProxyArray.")
    if len(source.shape) != 4 or source.shape[-1] != 3 or any(size <= 0 for size in source.shape):
        raise ValueError("PPISP radiance must have shape (N, H, W, 3) with positive dimensions.")
    if source.dtype is not wp.float32:
        raise ValueError("PPISP radiance must have dtype wp.float32.")
    if channels not in (3, 4):
        raise ValueError("PPISP color output requires three RGB or four RGBA channels.")
    if not isinstance(cfg, PpispCfg) or cfg.camera_prim_path is not None:
        raise ValueError("PPISP stages require a concrete PpispCfg; resolve USD camera attributes before preparation.")
    pipeline = PpispPipeline(deepcopy(cfg), controller_weights=controller_weights)
    try:
        pipeline.initialize(source.shape, source.device)
        color = ProxyArray(wp.empty((*source.shape[:3], channels), dtype=wp.uint8, device=source.device))

        def apply(inputs: Mapping[str, ProxyArray], outputs: Mapping[str, ProxyArray]) -> None:
            pipeline.apply(inputs["radiance"].warp, outputs["color"].warp)

        return PreparedBufferOperation(outputs={"color": color}, apply=apply, close=pipeline.close)
    except Exception:
        pipeline.close()
        raise


@configclass
class PpispStepCfg(RenderBufferStepCfg):
    """Route scene-linear radiance through PPISP into a public RGB buffer.

    Supply concrete settings with ``params={"cfg": ppisp_cfg}``. To produce RGBA, set
    ``outputs={"color": "rgba"}`` and ``params={"cfg": ppisp_cfg, "channels": 4}``.
    Both camera and observation pipes use the same operation implementation.
    """

    prepare = "isaaclab_ppisp:prepare_ppisp"
    inputs: dict[str, str] = {"radiance": "rgb_radiance"}
    outputs: dict[str, str] = {"color": "rgb"}
    params: dict[str, Any] = {"cfg": PpispCfg()}
