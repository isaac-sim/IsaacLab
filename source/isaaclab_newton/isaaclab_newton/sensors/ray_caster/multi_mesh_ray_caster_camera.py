# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster import BaseMultiMeshRayCasterCamera

from .multi_mesh_ray_caster import MultiMeshRayCaster
from .ray_caster_camera import RayCasterCamera

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg


class MultiMeshRayCasterCamera(BaseMultiMeshRayCasterCamera, MultiMeshRayCaster, RayCasterCamera):
    """Newton backend for the multi-mesh ray-cast camera sensor.

    Multi-mesh + camera pipeline from :class:`BaseMultiMeshRayCasterCamera`,
    target-mesh + body trackers from :class:`MultiMeshRayCaster` and
    :class:`RayCasterCamera`.
    """

    cfg: MultiMeshRayCasterCameraCfg
    __backend_name__: str = "newton"
