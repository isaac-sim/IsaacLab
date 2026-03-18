# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton-backend tiled-camera grid video capture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs.utils.tiled_camera_grid_video import TiledCameraGridVideoCapture


@configclass
class NewtonTiledCameraVideoCfg:
    """Settings for tiled RGB recording via TiledCamera (Newton / Newton Warp setups)."""

    class_type: type[Any] | str = (
        "isaaclab.envs.utils.tiled_camera_grid_video:TiledCameraGridVideoCapture"
    )
    """Implementation class; default is :class:`~isaaclab.envs.utils.tiled_camera_grid_video.TiledCameraGridVideoCapture`."""

    video_num_tiles: int = -1
    """Max environments per frame (``-1`` = all). Tiles fill a square grid with padding."""

    fallback_camera_cfg: object | None = None
    """Spawned when no observation TiledCamera exists; ``None`` disables fallback spawn."""

    preferred_renderer_types: tuple[str, ...] = ("newton_warp",)
    """Tiled video uses only TiledCameras with Newton Warp renderer (matches Newton GL backend)."""
