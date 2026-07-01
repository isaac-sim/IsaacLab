# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.sim.spawners.mpm import MPMParticleSpawnerCfg

if TYPE_CHECKING:
    from .mpm_object import MPMObject


@configclass
class MPMObjectCfg(DeformableObjectCfg):
    """Configuration parameters for a Newton MPM particle object."""

    class_type: type[MPMObject] | str = "{DIR}.mpm_object:MPMObject"

    spawn: MPMParticleSpawnerCfg = MISSING
    """Particle generation configuration for this MPM object."""
