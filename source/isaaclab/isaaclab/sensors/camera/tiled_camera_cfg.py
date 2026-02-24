# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .camera_cfg import CameraCfg
from .tiled_camera import TiledCamera

if TYPE_CHECKING:
    from isaaclab.renderer import RendererCfg


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor."""

    class_type: type = TiledCamera

    renderer_cfg: RendererCfg | None = None
    """Renderer config (e.g. IsaacRtxRendererCfg, NewtonWarpRendererCfg). If ``None``, RTX is used.
    Set by the scene from ``renderer_type`` string; Hydra override:
    ``env.scene.base_camera.renderer_type=warp_renderer`` or ``=rtx``."""

    renderer_type: str | None = None
    """Legacy / Hydra: ``"rtx"`` or ``None`` = RTX, ``"warp_renderer"`` = Warp. When set (e.g. by
    Hydra), the scene sets ``renderer_cfg`` from this. Prefer setting ``renderer_cfg`` directly."""
