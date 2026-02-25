# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for camera wrapper around USD camera prim."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["utils"],
    submod_attrs={
        "camera": ["Camera"],
        "camera_cfg": ["CameraCfg"],
        "camera_data": ["CameraData"],
        "tiled_camera": ["TiledCamera"],
        "tiled_camera_cfg": ["TiledCameraCfg"],
        "utils": ["save_images_to_file"],
    },
)
