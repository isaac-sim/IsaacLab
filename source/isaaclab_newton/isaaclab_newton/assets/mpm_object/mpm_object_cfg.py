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

    cloning_contexts: tuple[str | type, ...] | None = ("isaaclab_newton.cloner:NewtonReplicateContext",)
    """Physics cloning context for the MPM object: Newton replication.

    The authored simulation points are USD-clonable. A separate mutable particle
    cloud mirrors the simulated positions for renderer compatibility.
    """

    class_type: type[MPMObject] | str = "{DIR}.mpm_object:MPMObject"

    spawn: MPMParticleSpawnerCfg = MISSING
    """Particle generation configuration for this MPM object."""
