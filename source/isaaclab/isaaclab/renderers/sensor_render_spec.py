# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a tiled camera or lidar passed to render backends."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..sensors.camera.camera_cfg import CameraCfg
from ..sensors.lidar.lidar_cfg import LidarCfg


@dataclass(frozen=True, init=False)
class SensorRenderSpec:
    """Stable inputs for :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`.

    Backends use this instead of holding a reference to the camera or lidar sensor instance,
    avoiding circular dependencies between sensors and render data.

    Args:
        cfg: Camera or lidar configuration.
        device: Torch device string (e.g. ``"cuda:0"``) used by GPU annotators and Warp.
        num_instances: Number of tiled sensor instances (environments).
        prim_paths: Absolute paths of the authored sensor prims. When the renderer clones
            environments internally, this may contain only the source prim path; the renderer
            resolves its logical per-environment paths during registration.
        view_count: Number of logical sensor instances in the sensor view.
    """

    cfg: CameraCfg | LidarCfg
    device: str
    num_instances: int
    prim_paths: tuple[str, ...]
    view_count: int

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

            warnings.warn("camera_prim_paths is deprecated. Use prim_paths.", FutureWarning, stacklevel=2)
            prim_paths = camera_prim_paths

        # To allow camera_prim_paths to omit prim_paths, None defaults are used.
        if prim_paths is None:
            raise TypeError("SensorRenderSpec.__init__() missing required argument: 'prim_paths'.")

        # view_count needs a default because it follows prim_paths in the function signature.
        if view_count is None:
            raise TypeError("SensorRenderSpec.__init__() missing required argument: 'view_count'.")

        # Frozen dataclasses reject self.field = value, including in a custom __init__.
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_instances", num_instances)
        object.__setattr__(self, "prim_paths", prim_paths)
        object.__setattr__(self, "view_count", view_count)

    @property
    def camera_prim_paths(self) -> tuple[str, ...]:
        """Deprecated name for :attr:`prim_paths`."""
        warnings.warn("camera_prim_paths is deprecated. Use prim_paths.", FutureWarning, stacklevel=2)
        return self.prim_paths
