# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_ray_caster import BaseRayCaster


class RayCaster(FactoryBase, BaseRayCaster):
    """Backend-dispatching ray-caster sensor."""

    _backend_class_names = {"physx": "RayCaster", "newton": "RayCaster"}
