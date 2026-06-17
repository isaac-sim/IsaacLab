# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera


class MultiMeshRayCasterCamera(FactoryBase, BaseMultiMeshRayCasterCamera):
    """Backend-dispatching multi-mesh ray-caster camera sensor."""

    _backend_class_names = {
        "physx": "MultiMeshRayCasterCamera",
        "newton": "MultiMeshRayCasterCamera",
        "ovphysx": "MultiMeshRayCasterCamera",
    }
