# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton deformable object backend.

The implementation lives in :mod:`isaaclab_experimental.deformable`.
This stub re-exports the class so the :class:`FactoryBase` auto-discovery
(which tries ``isaaclab_newton.assets.deformable_object``) finds it.
"""

from isaaclab_experimental.deformable.deformable_object import DeformableObject
from isaaclab_experimental.deformable.deformable_object_data import DeformableObjectData

__all__ = ["DeformableObject", "DeformableObjectData"]
