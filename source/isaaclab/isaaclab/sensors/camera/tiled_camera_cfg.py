# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .camera_cfg import CameraCfg
from .tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor."""

    class_type: type = TiledCamera

    # Required by Hydra overrides
    # Overrides like env.scene.base_camera.renderer_type=newton_warp 
    #   only work if the composed config has that attribute.
    renderer_type: str | None = None
    """Renderer backend selector (isaac_rtx, newton_warp). Hydra instantiates renderer_cfg from this; otherwise TiledCamera does in _initialize_impl()."""

    def __post_init__(self):
        # So validation passes when Hydra sets only renderer_type (renderer_cfg.data_types is MISSING by default).
        # Skip when data_types is MISSING (e.g. Kuka scene sets it in scene __post_init__).
        if hasattr(self, "renderer_cfg") and self.renderer_cfg is not None and hasattr(self.renderer_cfg, "data_types"):
            if hasattr(self, "data_types") and self.data_types is not None and self.data_types is not MISSING:
                self.renderer_cfg.data_types = list(self.data_types)
