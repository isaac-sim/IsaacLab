# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import dataclasses
import logging
import math

from pxr import Usd, UsdPhysics

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from ..utils import (
    apply_nested,
    find_global_fixed_joint_prim,
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
    """Modify PhysX parameters for a rigid body prim.

    A `rigid body`_ is a single body that can be simulated by PhysX. It can be either dynamic or kinematic.
    A dynamic body responds to forces and collisions. A `kinematic body`_ can be moved by the user, but does not
    respond to forces. They are similar to having static bodies that can be moved around.

    The schema comprises of attributes that belong to the `RigidBodyAPI`_ and `PhysxRigidBodyAPI`_.
    schemas. The latter contains the PhysX parameters for the rigid body.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _rigid body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyOverview.html
    .. _kinematic body: https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies
    .. _RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html

    Args:
        prim_path: The prim path to the rigid body.
        cfg: The configuration for the rigid body.
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
    """Modify PhysX parameters for a joint prim.

    This function checks if the input prim is a prismatic or revolute joint and applies the joint drive schema
    on it. If the joint is a tendon (i.e., it has the `PhysxTendonAxisAPI`_ schema applied on it), then the joint
    drive schema is not applied.

    Based on the configuration, this method modifies the properties of the joint drive. These properties are
    based on the `UsdPhysics.DriveAPI`_ schema. For more information on the properties, please refer to the
    official documentation.

    .. caution::

        We highly recommend modifying joint properties of articulations through the functionalities in the
        :mod:`isaaclab.actuators` module. The methods here are for setting simulation low-level
        properties only.

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    .. _PhysxTendonAxisAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_a_p_i.html

    Args:
        prim_path: The prim path where to apply the joint drive schema.
        cfg: The configuration for the joint drive.
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
    # check if prim has fixed tendon applied on it
    applied_schemas = tendon_prim.GetAppliedSchemas()
    if not any("PhysxTendonAxisRootAPI" in s for s in applied_schemas):
        return False

    # resolve all available instances of the schema since it is multi-instance
    cfg = cfg.to_dict()
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
