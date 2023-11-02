# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.orbit.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from omni.isaac.orbit.sim.schemas import schemas_cfg
from omni.isaac.orbit.utils import configclass


@configclass
class MeshConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MeshConverter."""

    mass_props: schemas_cfg.MassPropertiesCfg = None
    """Mass properties to apply to the USD. Defaults to None.

    Note:
        If None, then no mass properties will be added.
    """

    rigid_props: schemas_cfg.RigidBodyPropertiesCfg = None
    """Rigid body properties to apply to the USD. Defaults to None.

    Note:
        If None, then no rigid body properties will be added.
    """

    collision_props: schemas_cfg.CollisionPropertiesCfg = None
    """Collision properties to apply to the USD. Defaults to None.

    Note:
        If None, then no collision properties will be added.
    """

    collision_approximation: str = "convexDecomposition"
    """Collision approximation method to use. Defaults to "convexDecomposition".

    Valid options are:
    "convexDecomposition", "convexHull", "boundingCube",
    "boundingSphere", "meshSimplification", or "none"

    "none" causes no collision mesh to be added.
    """
