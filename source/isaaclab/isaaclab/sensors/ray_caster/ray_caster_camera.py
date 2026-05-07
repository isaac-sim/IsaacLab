# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_ray_caster_camera import BaseRayCasterCamera


class RayCasterCamera(FactoryBase, BaseRayCasterCamera):
    """Backend-dispatching ray-caster camera sensor."""

    _backend_class_names = {"physx": "RayCasterCamera", "newton": "RayCasterCamera"}
