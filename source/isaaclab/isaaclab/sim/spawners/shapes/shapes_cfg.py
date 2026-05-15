# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.utils.configclass import configclass


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

    func: Callable | str = "{DIR}.shapes:spawn_sphere"

    radius: float = MISSING
    """Radius of the sphere (in m)."""


@configclass
class CuboidCfg(ShapeCfg):
    """Configuration parameters for a cuboid prim.

    See :meth:`spawn_cuboid` for more information.
    """

    func: Callable | str = "{DIR}.shapes:spawn_cuboid"

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid."""


@configclass
class CylinderCfg(ShapeCfg):
    """Configuration parameters for a cylinder prim.

    See :meth:`spawn_cylinder` for more information.
    """

    func: Callable | str = "{DIR}.shapes:spawn_cylinder"

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

    func: Callable | str = "{DIR}.shapes:spawn_capsule"

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

    func: Callable | str = "{DIR}.shapes:spawn_cone"

    radius: float = MISSING
    """Radius of the cone (in m)."""
    height: float = MISSING
    """Height of the v (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""


@configclass
class CableCfg(ShapeCfg):
    """Configuration parameters for a 1D cable / rod prim.

    Authors a ``UsdGeomBasisCurves`` prim at ``{prim_path}/curve`` from an
    explicit list of control points. Physics is materialized at model-build time
    by the Newton replicate hook calling :meth:`newton.ModelBuilder.add_rod_graph`.

    The cable's stretch/bend stiffness, damping, and density live on
    ``physics_material`` (a :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg` instance from
    :mod:`isaaclab_newton.sim.spawners.materials`, inherited slot from
    :class:`ShapeCfg`). ``rigid_props``, ``mass_props``, ``collision_props`` are
    inherited from :class:`ShapeCfg` but are not used by cables — :func:`spawn_cable`
    raises ``ValueError`` if any is non-None.
    """

    func: Callable | str = "{DIR}.shapes:spawn_cable"

    positions: list[tuple[float, float, float]] = MISSING
    """Control points in cable-local frame [m]. Must contain at least 2 points.
    Adjacent pairs define one cable segment each."""

    width: float = MISSING
    """Capsule diameter for each segment [m]."""

    visual_material_path: str = "visual_material"
    """Path for the visual material prim, relative to ``prim_path``. Overrides
    :attr:`ShapeCfg.visual_material_path` so visual and physics materials don't
    collide at the same sub-path (cables don't use a ``/geometry/`` intermediate
    like mesh spawners do)."""

    physics_material_path: str = "physics_material"
    """Path for the physics material prim, relative to ``prim_path``. Overrides
    :attr:`ShapeCfg.physics_material_path`. See :attr:`visual_material_path`."""
