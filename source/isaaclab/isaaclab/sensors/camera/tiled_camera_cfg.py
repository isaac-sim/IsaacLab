# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .camera_cfg import CameraCfg
from .tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor."""

    class_type: type = TiledCamera

    return_latest_camera_pose: bool = False
    """Whether to return the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is returned in the camera's data which will slow down performance
    due to the use of :class:`XformPrimView`.
    If False, the pose of the camera during initialization is returned.
    """
