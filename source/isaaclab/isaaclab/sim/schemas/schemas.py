# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from ..utils import (
    apply_nested,
    create_prim,
    find_global_fixed_joint_prim,
    get_all_matching_child_prims,
    safe_set_attribute_on_usd_prim,
    safe_set_attribute_on_usd_schema,
)
from . import schemas_cfg

# import logger
logger = logging.getLogger(__name__)


"""
Constants.
"""

# Mapping from string names to USD/PhysX tokens for mesh collision approximation
# Refer to omniverse documentation
# https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html#mesh-geometry-colliders
# for available tokens.
MESH_APPROXIMATION_TOKENS = {
    "boundingCube": UsdPhysics.Tokens.boundingCube,
    "boundingSphere": UsdPhysics.Tokens.boundingSphere,
    "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    "convexHull": UsdPhysics.Tokens.convexHull,
    "none": UsdPhysics.Tokens.none,
    "meshSimplification": UsdPhysics.Tokens.meshSimplification,
    "sdf": "sdf",  # PhysX SDF mesh token; use string (pxr.Tf.Token not available in all envs)
}


# Lazy accessors. These lists were used by the legacy ``usd_api`` / ``physx_api`` instance-
# field dispatch in ``modify_mesh_collision_properties``. The new metadata-driven writer
# does not consult them, but they are preserved as a public API so external code that
# imported them keeps working. The PhysX leaves now live in ``isaaclab_physx``; we resolve
# them lazily so this module does not import ``isaaclab_physx`` at load time.
def _get_physx_mesh_collision_cfgs() -> list:
    from isaaclab_physx.sim.schemas import schemas_cfg as _physx_cfg

    return [
        _physx_cfg.PhysxConvexHullPropertiesCfg,
        _physx_cfg.PhysxConvexDecompositionPropertiesCfg,
        _physx_cfg.PhysxTriangleMeshPropertiesCfg,
        _physx_cfg.PhysxTriangleMeshSimplificationPropertiesCfg,
        _physx_cfg.PhysxSDFMeshPropertiesCfg,
        # legacy deprecation aliases
        _physx_cfg.ConvexHullPropertiesCfg,
        _physx_cfg.ConvexDecompositionPropertiesCfg,
        _physx_cfg.TriangleMeshPropertiesCfg,
        _physx_cfg.TriangleMeshSimplificationPropertiesCfg,
        _physx_cfg.SDFMeshPropertiesCfg,
    ]


class _LazyList:
    """Lazy list whose contents are produced on first access.

    Used to keep the public ``PHYSX_MESH_COLLISION_CFGS`` / ``USD_MESH_COLLISION_CFGS`` symbols
    resolvable for callers that imported them, without triggering an ``isaaclab_physx`` import
    at this module's load time.
    """

    def __init__(self, factory):
        self._factory = factory
        self._cache = None

    def _resolved(self):
        if self._cache is None:
            self._cache = list(self._factory())
        return self._cache

    def __iter__(self):
        return iter(self._resolved())

    def __contains__(self, item):
        return item in self._resolved()

    def __len__(self):
        return len(self._resolved())

    def __getitem__(self, index):
        return self._resolved()[index]


PHYSX_MESH_COLLISION_CFGS = _LazyList(_get_physx_mesh_collision_cfgs)

USD_MESH_COLLISION_CFGS = _LazyList(
    lambda: [
        schemas_cfg.BoundingCubePropertiesCfg,
        schemas_cfg.BoundingSpherePropertiesCfg,
    ]
)


"""
Schema-application helper.
"""


def _get_field_declaring_class(cfg_class: type, field_name: str) -> type | None:
    """Return the most-base class in the MRO that declares ``field_name``.

    Each cfg field is owned by a single class in the hierarchy (the one whose body
    contains its annotation). This function walks the MRO in reverse so a base class
    declaration wins over a subclass redeclaration with the same name -- the field's
    USD namespace follows where it semantically lives, not where it was last
    overridden for default values.
    """
    for cls in reversed(cfg_class.__mro__):
        if field_name in getattr(cls, "__annotations__", {}):
            return cls
    return None


def _apply_namespaced_schemas(prim, cfg, cfg_dict: dict) -> None:
    """Route every cfg field to its declaring class's namespace and apply schemas.

    The helper handles the common ``AddAppliedSchema`` + namespaced-attribute write
    logic shared by every metadata-driven writer. Caller is responsible for popping
    fields that need typed-API writes (multi-instance ``UsdPhysics.DriveAPI``,
    ``TfToken`` attributes with ``allowedTokens``) out of ``cfg_dict`` first.

    USD attribute names are derived by snake_case -> camelCase conversion of cfg field
    names. The codebase enforces this as a convention: any cfg field whose
    snake_case name does not produce the correct USD camelCase attr is renamed (with a
    deprecation alias forwarded in ``__post_init__``) rather than mapped via metadata.

    Two passes:

    1. **Per-field exceptions** -- ``cfg._usd_field_exceptions`` is a mapping
       ``applied_schema -> (namespace, [cfg_field, ...])``. For each schema, if any
       listed field is non-None, the schema is applied (once) and each non-None field is
       written under that schema's namespace. Fields are popped from ``cfg_dict``.
    2. **Per-declaring-class routing** -- each remaining non-None field is grouped by the
       class that declares it (walking the MRO). Each group writes under that class's
       ``_usd_namespace`` and applies that class's ``_usd_applied_schema`` (if any). This
       means base-class fields go under the base namespace (e.g. ``physics:*``) even when
       the cfg instance is a PhysX subclass -- the subclass's ``_usd_namespace =
       "physxRigidBody"`` only governs *its own* fields.

    Args:
        prim: The USD prim to author on.
        cfg: The cfg instance carrying the metadata.
        cfg_dict: A mutable dict view of the cfg's non-metadata fields. Modified in place.

    Raises:
        ValueError: If a non-None field's declaring class does not define ``_usd_namespace``.
    """
    cfg_class = type(cfg)

    # 1. Per-field exceptions (overrides per-class routing for codeless-PhysX-namespace
    #    fields like ``disable_gravity`` on RigidBodyBaseCfg).
    field_exceptions = getattr(cfg, "_usd_field_exceptions", {}) or {}
    for applied_schema, (exc_ns, fields) in field_exceptions.items():
        triggered: list[tuple[str, object]] = []
        for cfg_field in fields:
            if cfg_field in cfg_dict:
                value = cfg_dict.pop(cfg_field)
                if value is not None:
                    triggered.append((to_camel_case(cfg_field, "cC"), value))
        if not triggered:
            continue
        if applied_schema and applied_schema not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(applied_schema)
        for usd_attr, value in triggered:
            safe_set_attribute_on_usd_prim(prim, f"{exc_ns}:{usd_attr}", value, camel_case=False)

    # 2. Group remaining non-None writes by declaring class.
    by_class: dict[type, list[tuple[str, object]]] = {}
    for cfg_field, value in list(cfg_dict.items()):
        if value is None:
            continue
        decl_class = _get_field_declaring_class(cfg_class, cfg_field)
        if decl_class is None:
            continue
        by_class.setdefault(decl_class, []).append((to_camel_case(cfg_field, "cC"), value))

    for decl_class, writes in by_class.items():
        # Read namespace/schema from the declaring class's own ``__dict__`` (not via
        # ``getattr``) so subclass overrides don't leak into base-field routing.
        namespace = decl_class.__dict__.get("_usd_namespace", None)
        applied_schema = decl_class.__dict__.get("_usd_applied_schema", None)
        if namespace is None:
            raise ValueError(
                f"{decl_class.__name__} declares fields {[a for a, _ in writes]} but does"
                " not define '_usd_namespace'. Add '_usd_namespace' to the class metadata"
                " or route the fields via '_usd_field_exceptions'."
            )
        if applied_schema and applied_schema not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(applied_schema)
        for usd_attr, value in writes:
            safe_set_attribute_on_usd_prim(prim, f"{namespace}:{usd_attr}", value, camel_case=False)


"""
Articulation root properties.
"""


def define_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootBaseCfg, stage: Usd.Stage | None = None
):
    """Apply the articulation root schema on the input prim and set its properties.

    See :func:`modify_articulation_root_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the articulation root schema.
        cfg: The configuration for the articulation root.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        TypeError: When the prim already has conflicting API schemas.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get articulation USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(prim):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    # set articulation root properties
    modify_articulation_root_properties(prim_path, cfg, stage)


@apply_nested
def modify_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootBaseCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for an articulation root prim.

    The `articulation root`_ marks the root of an articulation tree. For floating articulations, this should be on
    the root body. For fixed articulations, this API can be on a direct or indirect parent of the root joint
    which is fixed to the world.

    The schema comprises of attributes that belong to the `ArticulationRootAPI`_ and `PhysxArticulationAPI`_.
    schemas. The latter contains the PhysX parameters for the articulation root.

    The properties are applied to the articulation root prim. The common properties (such as solver position
    and velocity iteration counts, sleep threshold, stabilization threshold) take precedence over those specified
    in the rigid body schemas for all the rigid bodies in the articulation.

    .. caution::
        When the attribute :attr:`schemas_cfg.ArticulationRootPropertiesCfg.fix_root_link` is set to True,
        a fixed joint is created between the root link and the world frame (if it does not already exist). However,
        to deal with physics parser limitations, the articulation root schema needs to be applied to the parent of
        the root link.

    .. note::
        This function is decorated with :func:`apply_nested` that set the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _articulation root: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Articulations.html
    .. _ArticulationRootAPI: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html
    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html

    Args:
        prim_path: The prim path to the articulation root.
        cfg: The configuration for the articulation root.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        NotImplementedError: When the root prim is not a rigid body and a fixed joint is to be created.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get articulation USD prim
    articulation_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(articulation_prim):
        return False

    # convert to dict, filtering out class metadata (underscore-prefixed keys)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    # extract writer-side (non-USD) properties
    fix_root_link = cfg_dict.pop("fix_root_link", None)

    # apply per-field exceptions + main-namespace writes
    _apply_namespaced_schemas(articulation_prim, cfg, cfg_dict)

    # fix root link based on input
    # we do the fixed joint processing later to not interfere with setting other properties
    if fix_root_link is not None:
        # check if a global fixed joint exists under the root prim
        existing_fixed_joint_prim = find_global_fixed_joint_prim(prim_path)

        # if we found a fixed joint, enable/disable it based on the input
        # otherwise, create a fixed joint between the world and the root link
        if existing_fixed_joint_prim is not None:
            logger.info(
                f"Found an existing fixed joint for the articulation: '{prim_path}'. Setting it to: {fix_root_link}."
            )
            existing_fixed_joint_prim.GetJointEnabledAttr().Set(fix_root_link)
        elif fix_root_link:
            logger.info(f"Creating a fixed joint for the articulation: '{prim_path}'.")

            # note: we have to assume that the root prim is a rigid body,
            #   i.e. we don't handle the case where the root prim is not a rigid body but has articulation api on it
            # Currently, there is no obvious way to get first rigid body link identified by the PhysX parser
            if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise NotImplementedError(
                    f"The articulation prim '{prim_path}' does not have the RigidBodyAPI applied."
                    " To create a fixed joint, we need to determine the first rigid body link in"
                    " the articulation tree. However, this is not implemented yet."
                )

            # create a fixed joint between the root link and the world frame
            from omni.physx.scripts import utils as physx_utils

            physx_utils.createJoint(stage=stage, joint_type="Fixed", from_prim=None, to_prim=articulation_prim)

            # Having a fixed joint on a rigid body is not treated as "fixed base articulation".
            # instead, it is treated as a part of the maximal coordinate tree.
            # Moving the articulation root to the parent solves this issue. This is a limitation of the PhysX parser.
            # get parent prim
            parent_prim = articulation_prim.GetParent()
            # apply api to parent
            UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
            parent_applied = parent_prim.GetAppliedSchemas()
            if "PhysxArticulationAPI" not in parent_applied:
                parent_prim.AddAppliedSchema("PhysxArticulationAPI")

            # copy the attributes
            # -- usd attributes
            usd_articulation_api = UsdPhysics.ArticulationRootAPI(articulation_prim)
            for attr_name in usd_articulation_api.GetSchemaAttributeNames():
                attr = articulation_prim.GetAttribute(attr_name)
                parent_attr = parent_prim.GetAttribute(attr_name)
                if not parent_attr:
                    parent_attr = parent_prim.CreateAttribute(attr_name, attr.GetTypeName())
                parent_attr.Set(attr.Get())
            # -- physx attributes (copy by name prefix)
            for attr in articulation_prim.GetAttributes():
                aname = attr.GetName()
                if aname.startswith("physxArticulation:"):
                    parent_attr = parent_prim.GetAttribute(aname)
                    if not parent_attr:
                        parent_attr = parent_prim.CreateAttribute(aname, attr.GetTypeName())
                    parent_attr.Set(attr.Get())

            # remove api from root
            articulation_prim.RemoveAppliedSchema("PhysxArticulationAPI")
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    # success
    return True


"""
Rigid body properties.
"""


def define_rigid_body_properties(prim_path: str, cfg: schemas_cfg.RigidBodyBaseCfg, stage: Usd.Stage | None = None):
    """Apply the rigid body schema on the input prim and set its properties.

    See :func:`modify_rigid_body_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the rigid body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        TypeError: When the prim already has conflicting API schemas.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has rigid body applied on it
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    # set rigid body properties
    modify_rigid_body_properties(prim_path, cfg, stage)


@apply_nested
def modify_rigid_body_properties(
    prim_path: str, cfg: schemas_cfg.RigidBodyBaseCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify parameters for a rigid body prim.

    A `rigid body`_ is a single body that can be simulated by a physics engine. It can be either dynamic
    or kinematic. A dynamic body responds to forces and collisions. A `kinematic body`_ can be moved by
    the user, but does not respond to forces.

    Solver-common properties (from `RigidBodyAPI`_) are always written. Solver-specific properties are
    written based on the cfg subclass metadata (``_usd_namespace``, ``_usd_applied_schema``).

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _rigid body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyOverview.html
    .. _kinematic body: https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies
    .. _RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html

    Args:
        prim_path: The prim path to the rigid body.
        cfg: The configuration for the rigid body. Accepts
            :class:`~schemas_cfg.RigidBodyBaseCfg` for solver-common properties,
            :class:`~schemas_cfg.PhysxRigidBodyPropertiesCfg` for PhysX properties, or
            :class:`~schemas_cfg.MujocoRigidBodyPropertiesCfg` for Newton (MuJoCo) properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get rigid-body USD prim
    rigid_body_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has rigid-body applied on it
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        return False
    # convert to dict, filtering out class metadata (underscore-prefixed keys)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    # All fields routed by the helper via per-declaring-class lookup: base
    # ``rigid_body_enabled`` / ``kinematic_enabled`` go under ``physics:*``;
    # ``disable_gravity`` via field exceptions; PhysX-subclass fields under
    # ``physxRigidBody:*``.
    _apply_namespaced_schemas(rigid_body_prim, cfg, cfg_dict)
    return True


"""
Collision properties.
"""


def define_collision_properties(
    prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the collision schema on the input prim and set its properties.

    See :func:`modify_collision_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
    # set collision properties
    modify_collision_properties(prim_path, cfg, stage)


@apply_nested
def modify_collision_properties(
    prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX properties of collider prim.

    These properties are based on the `UsdPhysics.CollisionAPI`_ and `PhysxSchema.PhysxCollisionAPI`_ schemas.
    For more information on the properties, please refer to the official documentation.

    Tuning these parameters influence the contact behavior of the rigid body. For more information on
    tune them and their effect on the simulation, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>`__.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    .. _PhysxSchema.PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html

    Args:
        prim_path: The prim path of parent.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    collider_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    # dispatch nested mesh-collision cfg if present (preserve legacy behavior)
    mesh_collision_cfg = getattr(cfg, "mesh_collision_property", None)
    if mesh_collision_cfg is not None:
        modify_mesh_collision_properties(prim_path, mesh_collision_cfg, stage)

    # convert to dict, filtering out class metadata (underscore-prefixed keys)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    # pop the mesh_collision_property since it is already dispatched above
    cfg_dict.pop("mesh_collision_property", None)

    # All fields routed by the helper via per-declaring-class lookup: base
    # ``collision_enabled`` goes under ``physics:*``; ``contact_offset`` /
    # ``rest_offset`` via field exceptions; PhysX-subclass fields under
    # ``physxCollision:*``.
    _apply_namespaced_schemas(collider_prim, cfg, cfg_dict)
    # success
    return True


"""
Mass properties.
"""


def define_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage | None = None):
    """Apply the mass schema on the input prim and set its properties.

    See :func:`modify_mass_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has mass applied on it
    if not UsdPhysics.MassAPI(prim):
        UsdPhysics.MassAPI.Apply(prim)
    # set mass properties
    modify_mass_properties(prim_path, cfg, stage)


@apply_nested
def modify_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage | None = None) -> bool:
    """Set properties for the mass of a rigid body prim.

    These properties are based on the `UsdPhysics.MassAPI` schema. If the mass is not defined, the density is used
    to compute the mass. However, in that case, a collision approximation of the rigid body is used to
    compute the density. For more information on the properties, please refer to the
    `documentation <https://openusd.org/release/wp_rigid_body_physics.html#body-mass-properties>`__.

    .. caution::

        The mass of an object can be specified in multiple ways and have several conflicting settings
        that are resolved based on precedence. Please make sure to understand the precedence rules
        before using this property.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html

    Args:
        prim_path: The prim path of the rigid body.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    rigid_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has mass API applied on it
    if not UsdPhysics.MassAPI(rigid_prim):
        return False

    # ``mass`` / ``density`` (``physics:*``) routed via the helper's per-declaring-class lookup.
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    _apply_namespaced_schemas(rigid_prim, cfg, cfg_dict)
    # success
    return True


"""
Contact sensor.
"""


def activate_contact_sensors(prim_path: str, threshold: float = 0.0, stage: Usd.Stage = None):
    """Activate the contact sensor on all rigid bodies under a specified prim path.

    This function adds the PhysX contact report API to all rigid bodies under the specified prim path.
    It also sets the force threshold beyond which the contact sensor reports the contact. The contact
    reporting API can only be added to rigid bodies.

    Args:
        prim_path: The prim path under which to search and prepare contact sensors.
        threshold: The threshold for the contact sensor. Defaults to 0.0.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: If the input prim path is not valid.
        ValueError: If there are no rigid bodies under the prim path.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get prim
    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # iterate over all children
    num_contact_sensors = 0
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # check if prim is a rigid body
        # nested rigid bodies are not allowed by SDK so we can safely assume that
        # if a prim has a rigid body API, it is a rigid body and we don't need to
        # check its children
        if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # set sleep threshold to zero
            child_applied = child_prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in child_applied:
                child_prim.AddAppliedSchema("PhysxRigidBodyAPI")
            safe_set_attribute_on_usd_prim(child_prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
            # add contact report API with threshold of zero
            if "PhysxContactReportAPI" not in child_applied:
                child_prim.AddAppliedSchema("PhysxContactReportAPI")
            safe_set_attribute_on_usd_prim(child_prim, "physxContactReport:threshold", threshold, camel_case=False)
            # increment number of contact sensors
            num_contact_sensors += 1
        else:
            # add all children to tree
            all_prims += child_prim.GetChildren()
    # check if no contact sensors were found
    if num_contact_sensors == 0:
        descendant_count = 0
        frontier = [prim]
        while frontier:
            node = frontier.pop(0)
            children = list(node.GetChildren())
            descendant_count += len(children)
            frontier.extend(children)
        logger.warning(
            "[activate_contact_sensors] no rigid bodies found under prim=%r (type=%r, descendants=%d)",
            prim_path,
            prim.GetTypeName(),
            descendant_count,
        )
        raise ValueError(
            f"No contact sensors added to the prim: '{prim_path}'. This means that no rigid bodies"
            " are present under this prim. Please check the prim path."
        )
    # success
    return True


"""
Joint drive properties.
"""


@apply_nested
def modify_joint_drive_properties(
    prim_path: str, cfg: schemas_cfg.JointDriveBaseCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify parameters for a joint prim.

    This function checks if the input prim is a prismatic or revolute joint and applies the joint drive schema
    on it. If the joint is a tendon (i.e., it has the `PhysxTendonAxisAPI`_ schema applied on it), then the joint
    drive schema is not applied.

    Solver-common properties (from `UsdPhysics.DriveAPI`_) are always written. Solver-specific properties
    are written based on the cfg subclass metadata (``_usd_namespace``, ``_usd_applied_schema``).

    .. caution::

        We highly recommend modifying joint properties of articulations through the functionalities in the
        :mod:`isaaclab.actuators` module. The methods here are for setting simulation low-level
        properties only.

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    .. _PhysxTendonAxisAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_a_p_i.html

    Args:
        prim_path: The prim path where to apply the joint drive schema.
        cfg: The configuration for the joint drive. Accepts
            :class:`~schemas_cfg.JointDriveBaseCfg` for solver-common properties,
            :class:`~schemas_cfg.PhysxJointDrivePropertiesCfg` for PhysX properties, or
            :class:`~schemas_cfg.MujocoJointDrivePropertiesCfg` for Newton (MuJoCo) properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # check if prim has joint drive applied on it
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive_api_name = "angular"
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        drive_api_name = "linear"
    else:
        return False
    # check that prim is not a tendon child prim
    applied_schemas_str = str(prim.GetAppliedSchemas())
    if "PhysxTendonAxisAPI" in applied_schemas_str and "PhysxTendonAxisRootAPI" not in applied_schemas_str:
        return False

    # check if prim has joint drive applied on it
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name)
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_api_name)

    # ``drive_type`` is a permanent inline carve-out: the USD attribute is named ``type``
    # (a Python keyword-like name we cannot use as a cfg field). All other solver-common
    # joint-drive fields follow the snake_case = camelCase convention.
    # convert to dict, filtering out class metadata (underscore-prefixed keys)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    # ensure_drives_exist: if both stiffness and damping are zero on the authored drive,
    # set a minimal stiffness so that backends like Newton recognise the drive as active.
    ensure_drives = cfg_dict.pop("ensure_drives_exist", False)
    if ensure_drives and cfg_dict["stiffness"] is None and cfg_dict["damping"] is None:
        # read the current values from the drive
        cur_stiffness = usd_drive_api.GetStiffnessAttr().Get()
        cur_damping = usd_drive_api.GetDampingAttr().Get()
        if (cur_stiffness is None or cur_stiffness == 0.0) and (cur_damping is None or cur_damping == 0.0):
            cfg_dict["stiffness"] = 1e-3

    # check if linear drive
    is_linear_drive = prim.IsA(UsdPhysics.PrismaticJoint)
    # convert values for angular drives from radians to degrees units
    if not is_linear_drive:
        if cfg_dict.get("max_joint_velocity") is not None:
            # rad / s --> deg / s (PhysX angular convention is degrees)
            cfg_dict["max_joint_velocity"] = cfg_dict["max_joint_velocity"] * 180.0 / math.pi
        if cfg_dict["stiffness"] is not None:
            # N-m/rad --> N-m/deg
            cfg_dict["stiffness"] = cfg_dict["stiffness"] * math.pi / 180.0
        if cfg_dict["damping"] is not None:
            # N-m-s/rad --> N-m-s/deg
            cfg_dict["damping"] = cfg_dict["damping"] * math.pi / 180.0

    # set into USD API (solver-common properties; UsdPhysics.DriveAPI fields). Pop only
    # the solver-common fields here; the helper handles the PhysX-namespaced remainder.
    for attr_name in ["drive_type", "max_force", "stiffness", "damping"]:
        if attr_name not in cfg_dict:
            continue
        attr_value = cfg_dict.pop(attr_name)
        usd_attr_name = "type" if attr_name == "drive_type" else attr_name
        safe_set_attribute_on_usd_schema(usd_drive_api, usd_attr_name, attr_value, camel_case=True)

    # apply per-field exceptions (max_velocity -> physxJoint:maxJointVelocity) + any
    # PhysX-subclass main-namespace writes
    _apply_namespaced_schemas(prim, cfg, cfg_dict)

    return True


"""
Fixed tendon properties.
"""


@apply_nested
def modify_fixed_tendon_properties(
    prim_path: str, cfg: schemas_cfg.PhysxFixedTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a fixed tendon attachment prim.

    A `fixed tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises of attributes that belong to the `PhysxTendonAxisRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _fixed tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationFixedTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html

    Args:
        prim_path: The prim path to the tendon attachment.
        cfg: The configuration for the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has fixed tendon applied on it or if the mjc tendon prim exiss
    applied_schemas = tendon_prim.GetAppliedSchemas()
    prim_type = tendon_prim.GetTypeName()
    if not any("PhysxTendonAxisRootAPI" in s for s in applied_schemas) and prim_type != "MjcTendon":
        return False

    # resolve all available instances of the schema since it is multi-instance
    cfg = cfg.to_dict()
    if prim_type != "MjcTendon":
        for schema_name in applied_schemas:
            if "PhysxTendonAxisRootAPI" not in schema_name:
                continue
            # set into PhysX API by attribute prefix schema_name: (e.g. PhysxTendonAxisRootAPI:default:stiffness)
            for attr_name, value in cfg.items():
                safe_set_attribute_on_usd_prim(
                    tendon_prim,
                    f"{schema_name}:{to_camel_case(attr_name, 'cC')}",
                    value,
                    camel_case=False,
                )
    else:
        # only stiffness and damping in the cfg map to mjc attributes
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_prim(
                tendon_prim, f"mjc:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
            )
    # success
    return True


"""
Spatial tendon properties.
"""


@apply_nested
def modify_spatial_tendon_properties(
    prim_path: str, cfg: schemas_cfg.PhysxSpatialTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a spatial tendon attachment prim.

    A `spatial tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises of attributes that belong to the `PhysxTendonAxisRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _spatial tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationSpatialTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html
    .. _PhysxTendonAttachmentRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_root_a_p_i.html
    .. _PhysxTendonAttachmentLeafAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_leaf_a_p_i.html

    Args:
        prim_path: The prim path to the tendon attachment.
        cfg: The configuration for the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has spatial tendon applied on it
    applied_schemas = tendon_prim.GetAppliedSchemas()
    has_spatial = any(
        "PhysxTendonAttachmentRootAPI" in s or "PhysxTendonAttachmentLeafAPI" in s for s in applied_schemas
    )
    if not has_spatial:
        return False

    cfg = cfg.to_dict()
    for schema_name in applied_schemas:
        if "PhysxTendonAttachmentRootAPI" not in schema_name and "PhysxTendonAttachmentLeafAPI" not in schema_name:
            continue
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_prim(
                tendon_prim,
                f"{schema_name}:{to_camel_case(attr_name, 'cC')}",
                value,
                camel_case=False,
            )
    # success
    return True


"""
Collision mesh properties.
"""


def define_mesh_collision_properties(
    prim_path: str, cfg: schemas_cfg.MeshCollisionBaseCfg, stage: Usd.Stage | None = None
):
    """Apply the mesh collision schema on the input prim and set its properties.

    See :func:`modify_mesh_collision_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the mesh collision schema.
        cfg: The configuration for the mesh collision properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # Always apply the standard ``UsdPhysics.MeshCollisionAPI`` so the approximation token is
    # writable. The PhysX cooking schema (if any) is applied lazily by the writer below
    # only when the user authored at least one PhysX-namespaced tuning field.
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)

    modify_mesh_collision_properties(prim_path=prim_path, cfg=cfg, stage=stage)


@apply_nested
def modify_mesh_collision_properties(
    prim_path: str, cfg: schemas_cfg.MeshCollisionBaseCfg, stage: Usd.Stage | None = None
) -> bool:
    """Set properties for the mesh collision of a prim.

    Metadata-driven writer. The standard ``UsdPhysics.MeshCollisionAPI`` is applied
    unconditionally (it is the carrier of the ``physics:approximation`` token). The
    PhysX cooking schema declared by ``_usd_applied_schema`` (e.g.
    ``PhysxConvexHullCollisionAPI``) is gated on the user authoring at least one
    non-``None`` namespaced tuning field, mirroring the gating used by the other
    consumption-gated writers (rigid body, joint drive, collision, articulation root).

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to
        all the prims (that have the schema applied on them) under the input prim path.

    .. _UsdPhysics.MeshCollisionAPI: https://openusd.org/release/api/class_usd_physics_mesh_collision_a_p_i.html

    Args:
        prim_path: The prim path of the rigid body. This prim should be a Mesh prim.
        cfg: The configuration for the mesh collision properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: When the mesh approximation name is invalid.
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)

    # we need MeshCollisionAPI to set mesh collision approximation attribute
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)

    # convert to dict, filtering out class metadata (underscore-prefixed keys)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    # write the standard ``physics:approximation`` token via UsdPhysics.MeshCollisionAPI
    approximation_name = cfg_dict.pop("mesh_approximation_name", "none")
    if approximation_name not in MESH_APPROXIMATION_TOKENS:
        raise ValueError(
            f"Invalid mesh approximation name: '{approximation_name}'. "
            f"Valid options are: {list(MESH_APPROXIMATION_TOKENS.keys())}"
        )
    approximation_token = MESH_APPROXIMATION_TOKENS[approximation_name]
    safe_set_attribute_on_usd_schema(
        UsdPhysics.MeshCollisionAPI(prim), "Approximation", approximation_token, camel_case=False
    )

    # The standard ``UsdPhysics.MeshCollisionAPI`` is already applied above. The base
    # ``MeshCollisionBaseCfg`` declares ``_usd_applied_schema = "MeshCollisionAPI"`` so the
    # helper would re-apply (idempotent) if any base-namespace write fired. PhysX cooking
    # subclasses (ConvexHull / TriangleMesh / SDF / ...) override the schema and namespace
    # to author their tuning fields under e.g. ``physxConvexHullCollision:*``; the helper
    # gates ``Physx*CollisionAPI`` application on at least one non-None tuning field, so
    # Newton-targeted prims stay free of PhysX cooking schemas they did not opt in to.
    _apply_namespaced_schemas(prim, cfg, cfg_dict)

    # success
    return True


"""
Deformable body properties.
"""


@wp.kernel
def _fix_tet_winding_kernel(
    points: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(ndim=2, dtype=wp.int32),
):
    """Flip any tet with negative signed volume by swapping its last two vertex indices.

    ``UsdGeom.TetMesh`` and :meth:`UsdGeom.TetMesh.ComputeSurfaceFaces` require a
    right-handed tet winding (positive signed volume). Swapping indices 2 and 3
    reverses the orientation without changing which four vertices form the tet.
    """
    i = wp.tid()
    v0 = tet_indices[i, 0]
    v1 = tet_indices[i, 1]
    v2 = tet_indices[i, 2]
    v3 = tet_indices[i, 3]
    p0 = points[v0]
    e1 = points[v1] - p0
    e2 = points[v2] - p0
    e3 = points[v3] - p0
    signed_volume = wp.dot(e1, wp.cross(e2, e3))
    if signed_volume < 0.0:
        tet_indices[i, 2] = v3
        tet_indices[i, 3] = v2


def define_deformable_body_properties(
    prim_path: str,
    cfg: schemas_cfg.DeformableBodyPropertiesBaseCfg,
    stage: Usd.Stage | None = None,
    deformable_type: str = "volume",
    sim_mesh_prim_path: str | None = None,
):
    """Apply the deformable body schema on the input prim and set its properties. The input prim should
    have a visual surface mesh as child. Volume deformables will have their simulation tetrahedral mesh
    automatically computed from the surface mesh of the input prim. Surface deformables simply copy the visual mesh
    as simulation mesh.

    See :func:`modify_deformable_body_properties` for more details on how the properties are set.

    .. note::
        If the input prim is not a mesh, this function will traverse the prim and find the first mesh
        under it. If no mesh or multiple meshes are found, an error is raised. This is because the deformable
        body schema can only be applied to a single mesh.

    Args:
        prim_path: The prim path where to apply the deformable body schema.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
        deformable_type: The type of the deformable body (surface or volume).
            This is used to determine which USD API to use for the deformable body. Defaults to "volume".
        sim_mesh_prim_path: Optional override for the simulation mesh creation prim path.
            Ignored when pre-tetrahedralized mesh is found for volume deformables.
            If None, it is set to ``{prim_path}/sim_mesh``.

    Raises:
        ValueError: When the prim path is not valid.
        ValueError: When the prim has no mesh or multiple meshes.
        RuntimeError: When setting the deformable body properties fails.
    """
    from omni.physx.scripts import deformableUtils

    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not root_prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    sim_mesh_prim = None
    # for volume deformables, we check if a pre-tetrahedralized TetMesh exists for the sim_mesh
    if deformable_type == "volume":
        matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "TetMesh")
        if len(matching_prims) == 0:
            sim_mesh_prim = None
        elif len(matching_prims) > 1:
            # get list of all meshes found
            mesh_paths = [p.GetPrimPath() for p in matching_prims]
            raise ValueError(
                f"Found multiple tetrahedral meshes in '{prim_path}': {mesh_paths}."
                " Deformable body schema can only be applied to one mesh for now."
            )
        else:
            # found existing tetmesh
            sim_mesh_prim = matching_prims[0]
            if not sim_mesh_prim.IsValid():
                raise ValueError(f"Mesh prim path '{sim_mesh_prim.GetPrimPath()}' is not valid.")

    # Search for a visual surface mesh for both surface and volume deformables
    matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh")
    # check if the visual surface mesh is valid
    if len(matching_prims) == 0:
        # in case a TetMesh is found but no Mesh is found, we use the TetMesh surface as visual.
        if sim_mesh_prim is not None:
            tet_mesh_prim = UsdGeom.TetMesh(sim_mesh_prim)
            surface_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(tet_mesh_prim, Usd.TimeCode.Default())
            if surface_indices is None or len(surface_indices) == 0:
                raise ValueError(
                    f"Deformable body at '{prim_path}' has no surface indices on its TetMesh prim; "
                    "cannot sync to visual mesh."
                )
            # create visual mesh
            vis_mesh_prim = create_prim(
                prim_path + "/vis_mesh",
                prim_type="Mesh",
                attributes={
                    "points": tet_mesh_prim.GetPointsAttr().Get(),
                    "faceVertexIndices": np.asarray(surface_indices).flatten(),
                    "faceVertexCounts": [3] * len(surface_indices),
                },
                stage=stage,
            )
            matching_prims = [vis_mesh_prim]
        else:
            raise ValueError(f"Could not find any visual mesh in '{prim_path}'. Please check asset.")
    if len(matching_prims) > 1:
        # get list of all meshes found
        mesh_paths = [p.GetPrimPath() for p in matching_prims]
        raise ValueError(
            f"Found multiple visual meshes in '{prim_path}': {mesh_paths}."
            " Deformable body schema can only be applied to one mesh for now."
        )
    vis_mesh_prim = matching_prims[0]

    # check if the prim is valid
    if not vis_mesh_prim.IsValid():
        raise ValueError(f"Mesh prim path '{vis_mesh_prim.GetPrimPath()}' is not valid.")

    # remove potential previous configuration
    deformableUtils.remove_deformable_body(stage, prim_path)

    # create and set simulation/root prim properties based on the type of the deformable mesh (surface vs volume)
    sim_mesh_prim_path = prim_path + "/sim_mesh" if sim_mesh_prim_path is None else sim_mesh_prim_path
    # extract visual surface mesh vertices and faces
    vertices = np.array(vis_mesh_prim.GetAttribute("points").Get())
    faces = np.array(vis_mesh_prim.GetAttribute("faceVertexIndices").Get()).flatten()
    face_counts = np.array(vis_mesh_prim.GetAttribute("faceVertexCounts").Get())
    if deformable_type == "surface":
        # create simulation mesh as copy of visual mesh
        sim_mesh_prim = create_prim(
            sim_mesh_prim_path,
            prim_type="Mesh",
            attributes={
                "points": vertices,
                "faceVertexIndices": faces,
                "faceVertexCounts": face_counts,
            },
            stage=stage,
        )

        # apply sim API
        if not sim_mesh_prim.ApplyAPI("OmniPhysicsSurfaceDeformableSimAPI"):
            raise RuntimeError(f"Failed to set surface deformable body API on prim '{sim_mesh_prim_path}'.")

        # apply collision API
        if not sim_mesh_prim.ApplyAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Failed to set surface deformable collision API on prim '{sim_mesh_prim_path}'.")

        # set rest-shape attributes required by OmniPhysicsSurfaceDeformableSimAPI
        sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(vertices)
        sim_mesh_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(faces)

    elif deformable_type == "volume":
        if sim_mesh_prim is None:
            try:
                from pytetwild import tetrahedralize
            except ImportError as exc:
                raise ImportError(
                    "Automatic tetrahedralization of volume deformables requires the optional 'pytetwild' "
                    "package. Install pytetwild or provide a pre-tetrahedralized UsdGeom.TetMesh under the "
                    f"deformable prim '{prim_path}'."
                ) from exc

            tet_mesh_points, tet_mesh_indices = tetrahedralize(
                vertices,
                faces.reshape(-1, 3),
                edge_length_fac=0.1,
                simplify=False,
                epsilon=1e-2,
                coarsen=True,
            )
            # pytetwild's default ordering does not guarantee positive signed volume, which
            # ``UsdGeom.TetMesh`` and ``ComputeSurfaceFaces`` require. Flip any inverted tets.
            device = "cpu"
            _tet_points_wp = wp.array(tet_mesh_points.astype(np.float32), dtype=wp.vec3, device=device)
            _tet_indices_wp = wp.array(
                np.asarray(tet_mesh_indices, dtype=np.int32).reshape(-1, 4), dtype=wp.int32, device=device
            )
            wp.launch(
                _fix_tet_winding_kernel,
                dim=_tet_indices_wp.shape[0],
                inputs=[_tet_points_wp, _tet_indices_wp],
                device=device,
            )
            tet_mesh_indices = _tet_indices_wp.numpy()
            sim_mesh_prim = create_prim(
                sim_mesh_prim_path,
                prim_type="TetMesh",
                attributes={
                    "points": tet_mesh_points,
                    "tetVertexIndices": tet_mesh_indices,
                },
                stage=stage,
            )

        # apply sim API
        if not sim_mesh_prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI"):
            raise RuntimeError(f"Failed to set volume deformable body API on prim '{sim_mesh_prim_path}'.")

        # apply collision API
        if not sim_mesh_prim.ApplyAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Failed to set volume deformable collision API on prim '{sim_mesh_prim_path}'.")

        # set surface faces and rest-shape attributes required by OmniPhysicsVolumeDeformableSimAPI
        surface_face_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(
            UsdGeom.TetMesh(sim_mesh_prim), Usd.TimeCode.Default()
        )
        UsdGeom.TetMesh(sim_mesh_prim).GetSurfaceFaceVertexIndicesAttr().Set(surface_face_indices)
        sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(sim_mesh_prim.GetAttribute("points").Get())
        sim_mesh_prim.GetAttribute("omniphysics:restTetVtxIndices").Set(
            sim_mesh_prim.GetAttribute("tetVertexIndices").Get()
        )

    else:
        raise ValueError(
            f"""Unsupported deformable type: '{deformable_type}'.
            Only surface and volume deformables are supported."""
        )

    # TODO: Temporary solution: Overwrite visual mesh with tet mesh surface points or copy
    # surface sim mesh to vis mesh. In the future we can have separate visual from simulation mesh.
    # This currently does not work if an asset is loaded where the visual mesh is not the simulation mesh surface.
    vis_mesh = UsdGeom.Mesh(vis_mesh_prim)
    if deformable_type == "volume":
        tet_mesh_prim = UsdGeom.TetMesh(sim_mesh_prim)
        surface_indices = tet_mesh_prim.GetSurfaceFaceVertexIndicesAttr().Get()
        if surface_indices is None or len(surface_indices) == 0:
            raise ValueError(
                f"Deformable body at '{prim_path}' has no surface indices on its TetMesh prim; "
                "cannot sync to visual mesh."
            )
        vis_mesh.GetPointsAttr().Set(tet_mesh_prim.GetPointsAttr().Get())
        vis_mesh.GetFaceVertexIndicesAttr().Set(np.asarray(surface_indices).flatten())
        vis_mesh.GetFaceVertexCountsAttr().Set([3] * len(surface_indices))
    else:
        sim_mesh = UsdGeom.Mesh(sim_mesh_prim)
        vis_mesh.GetFaceVertexIndicesAttr().Set(sim_mesh.GetFaceVertexIndicesAttr().Get())
        vis_mesh.GetFaceVertexCountsAttr().Set(sim_mesh.GetFaceVertexCountsAttr().Get())

    # bind visual to sim mesh by applying bind pose deformable pose API
    purposes = ["bindPose"]
    vis_mesh_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    vis_mesh_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        purposes
    )
    points = UsdGeom.PointBased(vis_mesh_prim).GetPointsAttr().Get()
    vis_mesh_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(
        points
    )

    sim_mesh_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    sim_mesh_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        purposes
    )

    # disable simulation mesh for rendering
    UsdGeom.Imageable(sim_mesh_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    # apply deformable body api
    if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        raise RuntimeError(f"Failed to set deformable body API on prim '{prim_path}'.")

    # set deformable body properties
    modify_deformable_body_properties(prim_path, cfg, stage)


@apply_nested
def modify_deformable_body_properties(
    prim_path: str, cfg: schemas_cfg.DeformableBodyPropertiesBaseCfg, stage: Usd.Stage | None = None
):
    """Modify deformable body parameters for a deformable body prim.

    A `deformable body`_ is a single body (either surface or volume deformable) that can be simulated by PhysX
    or Newton. Unlike rigid bodies, deformable bodies support relative motion of the nodes in the mesh.
    Consequently, they can be used to simulate deformations under applied forces.

    PhysX deformable body simulation employs Finite Element Analysis (FEA) to simulate the deformations of the mesh.
    It uses two meshes to represent the deformable body:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    For most applications, we assume that the above two meshes are computed from the "render mesh" of the deformable
    body. The render mesh is the mesh that is visible in the scene and is used for rendering purposes. It is composed
    of triangles, while the simulation mesh is composed of tetrahedrons for volume deformables,
    and triangles for surface deformables.

    We apply similar design choices to the simulation in Newton with a separate visual, simulation and collision mesh.

    .. caution::
        The deformable body schema is still under development by the Omniverse team. The current implementation
        works with the PhysX schemas shipped with Isaac Sim 6.0.0 onwards. It may change in future releases.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _deformable body: https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/DeformableVolume.html
    .. _PhysxDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/annotated.html

    Args:
        prim_path: The prim path to the deformable body.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get deformable-body USD prim
    deformable_body_prim = stage.GetPrimAtPath(prim_path)
    # check if the prim is valid
    if not deformable_body_prim.IsValid():
        return False
    # check if deformable body API is applied
    if "OmniPhysicsDeformableBodyAPI" not in deformable_body_prim.GetAppliedSchemas():
        return False

    # build cfg dict from dataclass fields only; USD routing is driven by the
    # declaring classes' ``_usd_namespace`` / ``_usd_applied_schema`` metadata.
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    if cfg_dict.get("kinematic_enabled"):
        logger.warning(
            "Kinematic deformable bodies are not fully supported in the current version of Omni Physics. "
            "Setting kinematic_enabled to True may lead to unexpected behavior."
        )

    _apply_namespaced_schemas(deformable_body_prim, cfg, cfg_dict)
    # success
    return True
