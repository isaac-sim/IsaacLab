# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import REQUIRED

if TYPE_CHECKING:
    from isaaclab.sim import schemas


@dataclass
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
    physics_material: (
        materials.RigidBodyMaterialBaseCfg
        | materials.RigidBodyMaterialFragment
        | list[materials.RigidBodyMaterialFragment]
        | None
    ) = None
    """Physics material properties.

    Since shapes are rigid-only spawners, this slot accepts the rigid material base class or
    rigid-material fragments (single-namespace :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`
    instances or lists thereof).

    Note:
        If None, then no physics material will be added.
    """


@dataclass
class SphereCfg(ShapeCfg):
    """Configuration parameters for a sphere prim.

    See :meth:`spawn_sphere` for more information.
    """

    func: Callable | str = "isaaclab.sim.spawners.shapes.shapes:spawn_sphere"

    radius: float = REQUIRED
    """Radius of the sphere (in m)."""


@dataclass
class CuboidCfg(ShapeCfg):
    """Configuration parameters for a cuboid prim.

    See :meth:`spawn_cuboid` for more information.
    """

    func: Callable | str = "isaaclab.sim.spawners.shapes.shapes:spawn_cuboid"

    size: tuple[float, float, float] = REQUIRED
    """Size of the cuboid."""


@dataclass
class CylinderCfg(ShapeCfg):
    """Configuration parameters for a cylinder prim.

    See :meth:`spawn_cylinder` for more information.
    """

    func: Callable | str = "isaaclab.sim.spawners.shapes.shapes:spawn_cylinder"

    radius: float = REQUIRED
    """Radius of the cylinder (in m)."""
    height: float = REQUIRED
    """Height of the cylinder (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cylinder. Defaults to "Z"."""


@dataclass
class CapsuleCfg(ShapeCfg):
    """Configuration parameters for a capsule prim.

    See :meth:`spawn_capsule` for more information.
    """

    func: Callable | str = "isaaclab.sim.spawners.shapes.shapes:spawn_capsule"

    radius: float = REQUIRED
    """Radius of the capsule (in m)."""
    height: float = REQUIRED
    """Height of the capsule (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the capsule. Defaults to "Z"."""


@dataclass
class ConeCfg(ShapeCfg):
    """Configuration parameters for a cone prim.

    See :meth:`spawn_cone` for more information.
    """

    func: Callable | str = "isaaclab.sim.spawners.shapes.shapes:spawn_cone"

    radius: float = REQUIRED
    """Radius of the cone (in m)."""
    height: float = REQUIRED
    """Height of the v (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""


@dataclass
class CableCfg(SpawnerCfg):
    """Configuration parameters for an open linear cable."""

    func: Callable | str = "isaaclab.sim.spawners.shapes.shapes:spawn_cable"
    visual_material_path: str = "material"
    """Path to the visual material, relative to the cable geometry prim."""

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties."""

    positions: Sequence[tuple[float, float, float]] = REQUIRED
    """Control points in the cable-local frame [m].

    Requires at least three finite points with consecutive points separated by more than 1e-8 m.
    """

    physics_material_path: str = "physics_material"
    """Path to the physics material, relative to the cable geometry prim."""

    physics_material: materials.CableMaterialCfg = REQUIRED
    """Cable physics material."""

    collision_props: (
        schemas.CollisionPropertiesCfg | schemas.CollisionFragment | list[schemas.CollisionFragment] | None
    ) = None
    """Collision properties applied to the cable geometry."""
