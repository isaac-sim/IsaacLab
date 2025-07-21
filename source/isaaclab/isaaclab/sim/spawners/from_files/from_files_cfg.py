# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim import converters, schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import from_files


@configclass
class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    This class is a base class for spawning assets from files. It includes the common parameters
    for spawning assets from files, such as the path to the file and the function to use for spawning
    the asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Properties to apply to the articulation root."""

    fixed_tendons_props: schemas.FixedTendonsPropertiesCfg | None = None
    """Properties to apply to the fixed tendons (if any)."""

    spatial_tendons_props: schemas.SpatialTendonsPropertiesCfg | None = None
    """Properties to apply to the spatial tendons (if any)."""

    joint_drive_props: schemas.JointDrivePropertiesCfg | None = None
    """Properties to apply to a joint.

    .. note::
        The joint drive properties set the USD attributes of all the joint drives in the asset.
        We recommend using this attribute sparingly and only when necessary. Instead, please use the
        :attr:`~isaaclab.assets.ArticulationCfg.actuators` parameter to set the joint drive properties
        for specific joints in an articulation.
    """

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

    USD files are imported directly into the scene. However, given their complexity, there are various different
    operations that can be performed on them. For example, selecting variants, applying materials, or modifying
    existing properties.

    To prevent the explosion of configuration parameters, the available operations are limited to the most common
    ones. These include:

    - **Selecting variants**: This is done by specifying the :attr:`variants` parameter.
    - **Creating and applying materials**: This is done by specifying the :attr:`visual_material` parameter.
    - **Modifying existing properties**: This is done by specifying the respective properties in the configuration
      class. For instance, to modify the scale of the imported prim, set the :attr:`scale` parameter.

    See :meth:`spawn_from_usd` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    func: Callable = from_files.spawn_from_usd

    usd_path: str = MISSING
    """Path to the USD file to spawn asset from."""

    variants: object | dict[str, str] | None = None
    """Variants to select from in the input USD file. Defaults to None, in which case no variants are applied.

    This can either be a configclass object, in which case each attribute is used as a variant set name and its specified value,
    or a dictionary mapping between the two. Please check the :meth:`~isaaclab.sim.utils.select_usd_variants` function
    for more information.
    """


@configclass
class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg):
    """URDF file to spawn asset from.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF and spawns the imported
    USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
    the respective properties in the configuration class.

    See :meth:`spawn_from_urdf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.

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
