# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for deformable object assets."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "deformable_object": ["DeformableObject"],
        "deformable_object_cfg": ["DeformableObjectCfg"],
        "deformable_object_data": ["DeformableObjectData"],
    },
)
