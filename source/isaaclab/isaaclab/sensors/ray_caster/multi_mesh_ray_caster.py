# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.backend_utils import FactoryBase

from .base_multi_mesh_ray_caster import BaseMultiMeshRayCaster


class MultiMeshRayCaster(FactoryBase, BaseMultiMeshRayCaster):
    """Backend-dispatching multi-mesh ray-caster sensor."""

    _backend_class_names = {"physx": "MultiMeshRayCaster", "newton": "MultiMeshRayCaster"}
