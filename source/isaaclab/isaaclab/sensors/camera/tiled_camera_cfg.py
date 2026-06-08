# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .camera_cfg import CameraCfg

if TYPE_CHECKING:
    from .tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor.

    .. deprecated:: 4.6.0
        :class:`TiledCameraCfg` is deprecated. Use :class:`CameraCfg` directly —
        :class:`~isaaclab.sensors.camera.Camera` now includes TiledCamera's vectorized
        rendering optimizations via the same renderer abstraction.
    """

    class_type: type["TiledCamera"] | str = "{DIR}.tiled_camera:TiledCamera"

    def __post_init__(self):
        # TODO when Camera.__init__ moves rtx_sensor setting out of camera initialization
        # the default renderer config instantiation can be moved into the render factory
        # and get_default_render_cfg method can be removed from backend_utils
        renderer_type = getattr(self.renderer_cfg, "renderer_type", None)
        if renderer_type == "default":
            from isaaclab.utils.backend_utils import get_default_renderer_cfg

            self.renderer_cfg = get_default_renderer_cfg()
        warnings.warn(
            "TiledCameraCfg is deprecated. Use CameraCfg directly — "
            "Camera now includes TiledCamera's vectorized rendering optimizations.",
            DeprecationWarning,
            stacklevel=2,
        )
