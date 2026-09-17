# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg
from isaaclab.utils import REQUIRED

from isaaclab_newton.sim.spawners.mpm import MPMParticleSpawnerCfg

if TYPE_CHECKING:
    from .mpm_object import MPMObject


@dataclass
class MPMObjectCfg(DeformableObjectCfg):
    """Configuration parameters for a Newton MPM particle object."""

    cloning_contexts: tuple[str | type, ...] | None = field(
        default_factory=lambda: ("isaaclab_newton.cloner:NewtonReplicateContext",)
    )
    """Physics cloning context for the MPM object: Newton replication. The spawner authors
    only an empty Xform, which is fully USD-clonable, so USD clones are still created under
    Kit like any spawned asset; particle positions then sync through Fabric on top of the
    per-environment prims."""

    class_type: type[MPMObject] | str = "isaaclab_newton.assets.mpm_object.mpm_object:MPMObject"

    spawn: MPMParticleSpawnerCfg = REQUIRED
    """Particle generation configuration for this MPM object."""
