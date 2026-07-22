# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim import schemas


@configclass
class SpawnerCfg:
    """Configuration parameters for spawning an asset.

    Spawning an asset is done by calling the :attr:`func` function. The function takes in the
    prim path to spawn the asset at, the configuration instance and transformation, and returns the
    prim path of the spawned asset.

    The function is typically decorated with :func:`isaaclab.sim.spawner.utils.clone` decorator
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

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset. Defaults to None,
    which means no semantic tags will be added.

    The semantic tags follow the `Replicator Semantic` tagging system. Each tag is a tuple of the
    form ``(type, data)``, where ``type`` is the type of the tag and ``data`` is the semantic label
    associated with the tag. For example, to annotate a spawned asset in the class avocado, the semantic
    tag would be ``[("class", "avocado")]``.

    You can specify multiple semantic tags by passing in a list of tags. For example, to annotate a
    spawned asset in the class avocado and the color green, the semantic tags would be
    ``[("class", "avocado"), ("color", "green")]``.

    .. seealso::

        For more information on the semantics filter, see the documentation for the `semantics schema editor`_.

    .. _semantics schema editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering

    """

    copy_from_source: bool = True
    """Whether to copy the asset from the source prim or inherit it. Defaults to True.

    This parameter is only used when cloning prims. If False, then the asset will be inherited from
    the source prim, i.e. all USD changes to the source prim will be reflected in the cloned prims.
    """

    spawn_path: str | None = None
    """Path where the prototype is spawned. Defaults to None."""


@configclass
class RigidObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a rigid asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    mass_props: dict[str, list[schemas.MassFragment]] | schemas.MassPropertiesCfg | None = None
    """Mass properties.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.MassFragment` fragments (e.g. ``{"**": [MassCfg(...)]}``) or a
    single legacy :class:`~isaaclab.sim.schemas.MassPropertiesCfg`. On the fragment path each
    fragment writes its own namespace.

    Keys are target patterns relative to the prim the spawner anchors this family on (for USD
    assets: the spawn prim; for shapes and meshes: the container prim). Each ``/``-separated token
    is a regular expression matched per level, a trailing ``**`` token matches a prim and all its
    descendants, and an empty string targets the anchor prim itself. Entries are applied in
    insertion order, so on overlapping targets later entries override earlier ones per attribute.
    """

    mass_props_create_if_missing: bool = False
    """Whether the mass writer may apply ``UsdPhysics.MassAPI`` to matched prims that lack it.
    Defaults to False. The flag applies to every entry of the :attr:`mass_props` mapping.

    Only consumed when :attr:`mass_props` is given as fragments and the asset is spawned from a
    USD file; the shape and mesh spawners always create the API on the bare prim they author.
    """

    rigid_props: dict[str, list[schemas.RigidBodyFragment]] | schemas.RigidBodyBaseCfg | None = None
    """Rigid body properties.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.RigidBodyFragment` fragments
    (e.g. ``{"**": [UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]}``) or a single legacy cfg
    (e.g. :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg`). On the fragment path each fragment
    writes its own namespace.

    Keys are target patterns relative to the prim the spawner anchors this family on (for USD
    assets: the spawn prim; for shapes and meshes: the container prim). Each ``/``-separated token
    is a regular expression matched per level, a trailing ``**`` token matches a prim and all its
    descendants, and an empty string targets the anchor prim itself. Entries are applied in
    insertion order, so on overlapping targets later entries override earlier ones per attribute.

    For making a rigid object static, set the :attr:`schemas.RigidBodyBaseCfg.kinematic_enabled`
    (or :attr:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg.kinematic_enabled`) as True. This will
    make the object static and will not be affected by gravity or other forces.
    """

    collision_props: dict[str, list[schemas.CollisionFragment]] | schemas.CollisionPropertiesCfg | None = None
    """Properties to apply to all collision meshes.

    Accepts either a mapping from target pattern to a list of
    :class:`~isaaclab.sim.schemas.CollisionFragment` fragments
    (e.g. ``{"**": [UsdPhysicsCollisionCfg(...), PhysxCollisionCfg(...)]}``) or a single legacy cfg
    (e.g. :class:`~isaaclab.sim.schemas.CollisionBaseCfg`). On the fragment path each fragment
    writes its own namespace.

    Keys are target patterns relative to the prim the spawner anchors this family on (for USD
    assets: the spawn prim; for shapes and meshes: the geometry prim the spawner authors). Each
    ``/``-separated token is a regular expression matched per level, a trailing ``**`` token
    matches a prim and all its descendants, and an empty string targets the anchor prim itself.
    Entries are applied in insertion order, so on overlapping targets later entries override
    earlier ones per attribute.
    """

    activate_contact_sensors: bool = False
    """Activate contact reporting on all rigid bodies. Defaults to False.

    This adds the PhysxContactReporter API to all the rigid bodies in the given prim path and its children.
    """


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

    deformable_props: schemas.DeformableBodyPropertiesBaseCfg | None = None
    """Deformable body properties."""
