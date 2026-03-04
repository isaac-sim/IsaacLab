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

    gpu_max_rigid_contact_count: int = 524288
    """Size of the GPU rigid-body contact buffer. Default 512k contacts."""

    gpu_max_rigid_patch_count: int = 81920
    """Size of the GPU rigid-body patch buffer. Default 80k patches."""

    gpu_found_lost_pairs_capacity: int = 262144
    """Capacity for GPU found/lost broadphase pairs. Default 256k."""

    gpu_collision_stack_size: int = 67108864
    """GPU collision stack size in bytes. Default 64 MB."""
