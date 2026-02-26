# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .camera_cfg import CameraCfg
from .tiled_camera import TiledCamera

if TYPE_CHECKING:
    from isaaclab.renderers import RendererCfg


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor.

    If :attr:`renderer_type` is set (e.g. via Hydra env.scene.base_camera.renderer_type=warp_renderer),
    TiledCamera resolves :attr:`~.camera_cfg.CameraCfg.renderer_cfg` in _initialize_impl(); no scene
    __post_init__ logic required.
    """

    class_type: type = TiledCamera

    renderer_type: str | None = None
    """If set, TiledCamera builds renderer_cfg from this in _initialize_impl() so any task works."""

    def __post_init__(self):
        # So validation passes when Hydra sets only renderer_type (renderer_cfg.data_types is MISSING by default).
        # Skip when data_types is MISSING (e.g. Kuka scene sets it in scene __post_init__).
        if hasattr(self, "renderer_cfg") and self.renderer_cfg is not None and hasattr(self.renderer_cfg, "data_types"):
            if hasattr(self, "data_types") and self.data_types is not None and self.data_types is not MISSING:
                self.renderer_cfg.data_types = list(self.data_types)
