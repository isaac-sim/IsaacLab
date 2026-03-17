# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim tiled-camera grid video capture (Kit / PhysX sensor path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .isaacsim_tiled_camera_video_cfg import IsaacsimTiledCameraVideoCfg


def create_isaacsim_tiled_camera_video(cfg: IsaacsimTiledCameraVideoCfg, scene: InteractiveScene):
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
