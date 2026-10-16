# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Data container for the multi-mesh ray-cast sensor."""

from ...utils.leapp import LeappTensorSemantics
from ...utils.warp import ProxyArray
from .ray_caster_data import RayCasterData


class MultiMeshRayCasterData(RayCasterData):
    """Data container for the multi-mesh ray-cast sensor."""

    _leapp_field_semantics = {
        "ray_mesh_ids": LeappTensorSemantics(kind="state/sensor/ray_mesh_id"),
    }

    ray_mesh_ids: ProxyArray = None
    """The mesh ids of the ray hits.

    Shape is (N, B, 1), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """
