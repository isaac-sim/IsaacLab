# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from omni.isaac.orbit.sim import loaders
from omni.isaac.orbit.sim.spawners import materials
from omni.isaac.orbit.sim.spawners.spawner_cfg import ArticulationSpawnerCfg, SpawnerCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from . import from_files


@configclass
class UsdFileCfg(ArticulationSpawnerCfg):
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
class UrdfFileCfg(ArticulationSpawnerCfg, loaders.UrdfLoaderCfg):
    """URDF file to spawn asset from.

    It uses the :class:`UrdfLoader` class to create a USD file from URDF and spawns the imported
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
    height: float = 0.0
    """The height of the ground plane. Defaults to 0.0."""
    color: tuple[float, float, float] | None = (0.065, 0.0725, 0.080)
    """The color of the ground plane. Defaults to (0.065, 0.0725, 0.080).

    If None, then the color remains unchanged.
    """
    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""
    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""
