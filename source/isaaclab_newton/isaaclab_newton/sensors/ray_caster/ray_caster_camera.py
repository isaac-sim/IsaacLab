# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster import BaseRayCasterCamera

from .ray_caster import RayCaster

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import RayCasterCameraCfg


class RayCasterCamera(BaseRayCasterCamera, RayCaster):
    """Newton backend for the ray-caster camera sensor.

    Camera buffers/intrinsics from :class:`BaseRayCasterCamera`, body tracker
    (body-attached site + :class:`~newton.sensors.SensorFrameTransform`) from
    :class:`RayCaster`.
    """

    cfg: RayCasterCameraCfg
    __backend_name__: str = "newton"
