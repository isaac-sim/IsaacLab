# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrappers for deformable schema writers.

The deformable schema writers are backend-aware but remain unified in
:mod:`isaaclab.sim.schemas`.
"""

from isaaclab.sim.schemas.schemas import define_deformable_body_properties, modify_deformable_body_properties

__all__ = ["define_deformable_body_properties", "modify_deformable_body_properties"]
