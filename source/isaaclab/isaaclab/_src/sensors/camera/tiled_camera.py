# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated module. Use :class:`~isaaclab.sensors.camera.Camera` instead.

.. deprecated:: 4.6.0
    :class:`TiledCamera` is deprecated. :class:`~isaaclab.sensors.camera.Camera` now includes
    TiledCamera's vectorized rendering optimizations via the same :class:`~isaaclab.renderers.Renderer`
    abstraction. Use :class:`~isaaclab.sensors.camera.Camera` with
    :class:`~isaaclab.sensors.camera.CameraCfg` (or :class:`~isaaclab.sensors.camera.TiledCameraCfg`)
    directly.
"""

from __future__ import annotations

import warnings

from .camera import Camera
from .tiled_camera_cfg import TiledCameraCfg


class TiledCamera(Camera):
    """Deprecated alias for :class:`Camera`.

    .. deprecated:: 4.6.0
        Use :class:`Camera` directly — it now includes TiledCamera's vectorized rendering
        optimizations via the same Renderer abstraction.
    """

    def __init__(self, cfg: TiledCameraCfg):
        warnings.warn(
            "TiledCamera is deprecated and will be removed in a future release. "
            "Use Camera directly — it now includes TiledCamera's vectorized rendering "
            "optimizations via the same Renderer abstraction.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cfg)
