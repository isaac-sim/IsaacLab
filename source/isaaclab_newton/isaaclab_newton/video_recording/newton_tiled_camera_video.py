# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backend tiled-camera grid video capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .newton_tiled_camera_video_cfg import NewtonTiledCameraVideoCfg


def create_newton_tiled_camera_video(cfg: NewtonTiledCameraVideoCfg, scene: InteractiveScene):
    """Instantiate tiled capture from ``cfg.class_type`` (default: :class:`TiledCameraGridVideoCapture`)."""
    ct = cfg.class_type
    if isinstance(ct, type):
        return ct(
            scene,
            video_num_tiles=cfg.video_num_tiles,
            fallback_camera_cfg=cfg.fallback_camera_cfg,
        )
    from isaaclab.utils.string import string_to_callable

    cls = string_to_callable(str(ct))
    return cls(
        scene,
        video_num_tiles=cfg.video_num_tiles,
        fallback_camera_cfg=cfg.fallback_camera_cfg,
    )
