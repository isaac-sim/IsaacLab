# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors.ray_caster import BaseMultiMeshRayCasterCamera

from .ray_caster import _PhysXRayCasterMixin


class MultiMeshRayCasterCamera(_PhysXRayCasterMixin, BaseMultiMeshRayCasterCamera):
    """PhysX MultiMeshRayCasterCamera implementation."""
