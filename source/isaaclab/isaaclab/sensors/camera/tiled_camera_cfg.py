# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .camera_cfg import CameraCfg
from .tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor."""

    class_type: type = TiledCamera

    renderer_type: str | None = None
    """Renderer backend selector (isaac_rtx, newton_warp). Hydra instantiates renderer_cfg from this; otherwise TiledCamera does in _initialize_impl()."""

    def __post_init__(self):
        # Sync camera data_types to renderer_cfg (source of truth is the camera).
        if hasattr(self, "renderer_cfg") and self.renderer_cfg is not None and hasattr(self.renderer_cfg, "data_types"):
            if hasattr(self, "data_types") and self.data_types:
                self.renderer_cfg.data_types = list(self.data_types)
