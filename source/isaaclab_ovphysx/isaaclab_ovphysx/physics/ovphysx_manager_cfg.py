# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the OvPhysX physics manager."""

from __future__ import annotations

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass


@configclass
class OvPhysxCfg(PhysicsCfg):
    """Configuration for the ovphysx physics manager.

    PhysX scene-level parameters (solver iterations, GPU buffer sizes, etc.) are
    read from the USD PhysicsScene prim.  Only ovphysx-specific settings that are
    not captured in USD live here.
    """

    class_type = "{DIR}.ovphysx_manager:OvPhysxManager"

    gpu_max_rigid_contact_count: int = 2**23
    """Size of the GPU rigid-body contact buffer."""

    gpu_max_rigid_patch_count: int = 5 * 2**15
    """Size of the GPU rigid-body patch buffer."""

    gpu_found_lost_pairs_capacity: int = 2**21
    """Capacity for GPU found/lost broadphase pairs."""

    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    """Capacity for GPU found/lost *aggregate* broadphase pairs."""

    gpu_total_aggregate_pairs_capacity: int = 2**21
    """Capacity for total GPU aggregate broadphase pairs."""

    gpu_collision_stack_size: int = 2**26
    """GPU collision stack size in bytes."""
