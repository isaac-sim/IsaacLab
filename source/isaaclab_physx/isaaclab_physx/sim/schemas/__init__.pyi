# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "define_deformable_body_properties",
    "modify_deformable_body_properties",
    "DeformableBodyPropertiesCfg",
]

from .schemas import (
    define_deformable_body_properties,
    modify_deformable_body_properties,
)
from .schemas_cfg import DeformableBodyPropertiesCfg
