# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster

from .ray_caster import _NewtonRayCasterMixin


class MultiMeshRayCaster(_NewtonRayCasterMixin, BaseMultiMeshRayCaster):
    """Newton MultiMeshRayCaster implementation."""
