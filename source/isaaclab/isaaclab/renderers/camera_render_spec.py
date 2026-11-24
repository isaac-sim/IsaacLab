# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated subclass of :class:`~isaaclab.renderers.sensor_render_spec.SensorRenderSpec`.

Construct :class:`SensorRenderSpec` instead. Pass sensor prims as ``prim_paths``.
``camera_prim_paths`` still reads and sets that field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import deprecated

from .sensor_render_spec import SensorRenderSpec

if TYPE_CHECKING:
    from ..sensors.camera.camera_cfg import CameraCfg
    from ..sensors.lidar.lidar_cfg import LidarCfg


@deprecated("CameraRenderSpec is deprecated. Use SensorRenderSpec instead.")
class CameraRenderSpec(SensorRenderSpec):
    """Deprecated subclass of :class:`SensorRenderSpec`.

    .. deprecated::
        Construct :class:`SensorRenderSpec` for a camera or a lidar. Pass sensor prims as
        ``prim_paths``. ``camera_prim_paths`` still reads and sets that field.
    """

    def __init__(
        self,
        cfg: CameraCfg | LidarCfg,
        device: str,
        num_instances: int,
        prim_paths: tuple[str, ...] | None = None,
        view_count: int | None = None,
        *,
        camera_prim_paths: tuple[str, ...] | None = None,
    ) -> None:
        if camera_prim_paths is not None:
            if prim_paths is not None:
                raise TypeError("Pass only one of 'prim_paths' and 'camera_prim_paths'.")
            prim_paths = camera_prim_paths

        # To allow camera_prim_paths to omit prim_paths, None defaults are used.
        if prim_paths is None:
            raise TypeError("CameraRenderSpec.__init__() missing required argument: 'prim_paths'.")

        # view_count needs a default because it follows prim_paths in the function signature.
        if view_count is None:
            raise TypeError("CameraRenderSpec.__init__() missing required argument: 'view_count'.")

        super().__init__(cfg, device, num_instances, prim_paths, view_count)

    @property
    def camera_prim_paths(self) -> tuple[str, ...]:
        """Alias of :attr:`prim_paths`."""
        return self.prim_paths
