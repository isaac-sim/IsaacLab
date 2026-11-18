# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Data container for the multi-mesh ray-cast sensor."""

from ...utils.warp import ProxyArray
from .ray_caster_data import RayCasterData


class MultiMeshRayCasterData(RayCasterData):
    """Data container for the multi-mesh ray-cast sensor."""

    ray_mesh_ids: ProxyArray = None
    """The mesh ids of the ray hits.

    Shape is (N, B), dtype is ``wp.int16``, where N is the number of sensors and B is the number of rays
    in the scan pattern per sensor. In Torch this resolves to shape (N, B) with dtype ``torch.int16``.
    This field is None when :attr:`~isaaclab.sensors.MultiMeshRayCasterCfg.update_mesh_ids` is disabled.
    """
