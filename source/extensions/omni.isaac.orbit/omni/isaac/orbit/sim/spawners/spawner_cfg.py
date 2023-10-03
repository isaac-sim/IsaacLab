# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from pxr import Usd

from omni.isaac.orbit.sim import schemas
from omni.isaac.orbit.utils import configclass


@configclass
class SpawnerCfg:
    """Configuration parameters for spawning an asset.

    Spawning an asset is done by calling the :attr:`func` function. The function takes in the
    prim path to spawn the asset at, the configuration instance and transformation, and returns the
    prim path of the spawned asset.

    The function is typically decorated with :func:`omni.isaac.orbit.sim.spawner.utils.clone` decorator
    that checks if input prim path is a regex expression and spawns the asset at all matching prims.
    For this, the decorator uses the Cloner API from Isaac Sim and handles the :attr:`copy_from_source`
    parameter.
    """

    func: Callable[..., Usd.Prim] = MISSING
    """Function to use for spawning the asset.

    The function takes in the prim path (or expression) to spawn the asset at, the configuration instance
    and transformation, and returns the source prim spawned.
    """

    visible: bool = True
    """Whether the spawned asset should be visible. Defaults to True."""

    copy_from_source: bool = True
    """Whether to copy the asset from the source prim or inherit it. Defaults to True.

    This parameter is only used when cloning prims. If False, then the asset will be inherited from
    the source prim, i.e. all USD changes to the source prim will be reflected in the cloned prims.

    .. versionadded:: 2023.1

        This parameter is only supported from Isaac Sim 2023.1 onwards. If you are using an older
        version of Isaac Sim, this parameter will be ignored.
    """


@configclass
class RigidObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a rigid asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    mass_props: schemas.MassPropertiesCfg | None = None
    """Mass properties."""
    rigid_props: schemas.RigidBodyPropertiesCfg | None = None
    """Rigid body properties."""
    collision_props: schemas.CollisionPropertiesCfg | None = None
    """Properties to apply to all collision meshes."""

    activate_contact_sensors: bool = False
    """Activate contact reporting on all rigid bodies. Defaults to False.

    This adds the PhysxContactReporter API to all the rigid bodies in the given prim path and its children.
    """
