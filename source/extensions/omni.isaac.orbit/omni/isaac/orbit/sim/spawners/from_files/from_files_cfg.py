# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from omni.isaac.orbit.sim import converters, schemas
from omni.isaac.orbit.sim.spawners import materials
from omni.isaac.orbit.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from . import from_files


@configclass
class FileCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    articulation_props: schemas.ArticulationPropertiesCfg | None = None
    """Properties to apply to the articulation root."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class UsdFileCfg(FileCfg):
    """USD file to spawn asset from.

    See :meth:`spawn_from_usd` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.
    """

    func: Callable = from_files.spawn_from_usd

    usd_path: str = MISSING
    """Path to the USD file to spawn asset from."""


@configclass
class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg):
    """URDF file to spawn asset from.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF and spawns the imported
    USD file. See :meth:`spawn_from_urdf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.
    """

    func: Callable = from_files.spawn_from_urdf


"""
Spawning ground plane.
"""


@configclass
class GroundPlaneCfg(SpawnerCfg):
    """Create a ground plane prim.

    This uses the USD for the standard grid-world ground plane from Isaac Sim by default.
    """

    func: Callable = from_files.spawn_ground_plane

    usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
    """Path to the USD file to spawn asset from. Defaults to the grid-world ground plane."""

    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """The color of the ground plane. Defaults to (0.0, 0.0, 0.0).

    If None, then the color remains unchanged.
    """

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""

    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""
