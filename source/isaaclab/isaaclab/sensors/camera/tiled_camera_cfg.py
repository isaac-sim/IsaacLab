# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import DeferredClass, configclass

from .camera_cfg import CameraCfg


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor."""

    class_type: type | DeferredClass = DeferredClass("isaaclab.sensors.camera.tiled_camera:TiledCamera")
