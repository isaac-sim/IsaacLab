# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "spawn_deformable_body_material",
    "DeformableBodyMaterialCfg",
    "SurfaceDeformableBodyMaterialCfg",
]

from .physics_materials import spawn_deformable_body_material
from .physics_materials_cfg import (
    DeformableBodyMaterialCfg,
    SurfaceDeformableBodyMaterialCfg,
)
