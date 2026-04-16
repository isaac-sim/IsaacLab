# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.sim import schemas

    # deformables only supported on PhysX backend
    from isaaclab_physx.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg


@configclass
class DeformableObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a deformable asset.

    Unlike rigid objects, deformable objects are affected by forces and can deform when subjected to
    external forces. This class is used to configure the properties of the deformable object.

    Deformable bodies don't have a separate collision mesh. The collision mesh is the same as the visual mesh.
    The collision properties such as rest and collision offsets are specified in the :attr:`deformable_props`.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    mass_props: schemas.MassPropertiesCfg | None = None
    """Mass properties."""

    deformable_props: DeformableBodyPropertiesCfg | None = None
    """Deformable body properties. Only supported on PhysX backend for now."""
