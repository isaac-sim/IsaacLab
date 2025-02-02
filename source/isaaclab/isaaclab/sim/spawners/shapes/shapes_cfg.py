# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from . import shapes


@configclass
class ShapeCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """
    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """
    physics_material: materials.PhysicsMaterialCfg | None = None
    """Physics material properties.

    Note:
        If None, then no physics material will be added.
    """


@configclass
class SphereCfg(ShapeCfg):
    """Configuration parameters for a sphere prim.

    See :meth:`spawn_sphere` for more information.
    """

    func: Callable = shapes.spawn_sphere

    radius: float = MISSING
    """Radius of the sphere (in m)."""


@configclass
class CuboidCfg(ShapeCfg):
    """Configuration parameters for a cuboid prim.

    See :meth:`spawn_cuboid` for more information.
    """

    func: Callable = shapes.spawn_cuboid

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid."""


@configclass
class CylinderCfg(ShapeCfg):
    """Configuration parameters for a cylinder prim.

    See :meth:`spawn_cylinder` for more information.
    """

    func: Callable = shapes.spawn_cylinder

    radius: float = MISSING
    """Radius of the cylinder (in m)."""
    height: float = MISSING
    """Height of the cylinder (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cylinder. Defaults to "Z"."""


@configclass
class CapsuleCfg(ShapeCfg):
    """Configuration parameters for a capsule prim.

    See :meth:`spawn_capsule` for more information.
    """

    func: Callable = shapes.spawn_capsule

    radius: float = MISSING
    """Radius of the capsule (in m)."""
    height: float = MISSING
    """Height of the capsule (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the capsule. Defaults to "Z"."""


@configclass
class ConeCfg(ShapeCfg):
    """Configuration parameters for a cone prim.

    See :meth:`spawn_cone` for more information.
    """

    func: Callable = shapes.spawn_cone

    radius: float = MISSING
    """Radius of the cone (in m)."""
    height: float = MISSING
    """Height of the v (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""
