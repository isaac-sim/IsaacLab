# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.configclass import configclass


@configclass
class MeshConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MeshConverter."""

    mass_props: dict[str, list[schemas_cfg.MassFragment]] | schemas_cfg.MassPropertiesCfg | None = None
    """Mass properties to apply to the USD. Defaults to None.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.MassFragment` fragments (e.g. ``{"": [MassCfg(...)]}``) or a
    single legacy :class:`~isaaclab.sim.schemas.MassPropertiesCfg`. Keys are target patterns
    relative to the root Xform prim of the converted asset; an empty string targets that prim
    itself. Entries are applied in insertion order, so on overlapping targets later entries
    override earlier ones per attribute.

    Note:
        If None, then no mass properties will be added.
    """

    rigid_props: dict[str, list[schemas_cfg.RigidBodyFragment]] | schemas_cfg.RigidBodyBaseCfg | None = None
    """Rigid body properties to apply to the USD. Defaults to None.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.RigidBodyFragment` fragments or a single legacy cfg
    (e.g. :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg`). Keys are target patterns relative to
    the root Xform prim of the converted asset; an empty string targets that prim itself. Entries
    are applied in insertion order, so on overlapping targets later entries override earlier ones
    per attribute.

    Note:
        If None, then no rigid body properties will be added.
    """

    collision_props: dict[str, list[schemas_cfg.CollisionFragment]] | schemas_cfg.CollisionPropertiesCfg | None = None
    """Collision properties to apply to the USD. Defaults to None.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.CollisionFragment` fragments or a single legacy cfg
    (e.g. :class:`~isaaclab.sim.schemas.CollisionBaseCfg`). Keys are target patterns relative to
    each Mesh prim of the converted asset; an empty string targets the mesh prim itself. Entries
    are applied in insertion order, so on overlapping targets later entries override earlier ones
    per attribute.

    Note:
        If None, then no collision properties will be added.
    """
    mesh_collision_props: (
        schemas_cfg.MeshCollisionBaseCfg
        | schemas_cfg.MeshCollisionFragment
        | list[schemas_cfg.MeshCollisionFragment]
        | None
    ) = None
    """Mesh approximation properties to apply to all collision meshes in the USD.

    Accepts either a single legacy cfg (e.g. :class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg` or
    a ``Physx*PropertiesCfg`` cooking cfg) or a list of
    :class:`~isaaclab.sim.schemas.MeshCollisionFragment` fragments (e.g.
    ``[UsdPhysicsMeshCollisionCfg(...), PhysxConvexHullCfg(...)]``). When a fragment list is given,
    ``UsdPhysics.MeshCollisionAPI`` is applied as the implicit anchor, the ``physics:approximation``
    token is resolved from whichever cooking fragment is present, and each fragment writes its own
    namespace.

    Note:
        If None, then no mesh approximation properties will be added.
    """

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The translation of the mesh to the origin. Defaults to (0.0, 0.0, 0.0)."""

    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """The rotation of the mesh in quaternion format (x, y, z, w). Defaults to (0.0, 0.0, 0.0, 1.0)."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The scale of the mesh. Defaults to (1.0, 1.0, 1.0)."""
