# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Iterable

import numpy as np
import warp as wp
from typing_extensions import deprecated

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.utils.string import string_to_callable, to_camel_case

from ..utils import (
    apply_nested,
    create_prim,
    find_global_fixed_joint_prim,
    find_matching_prims,
    get_all_matching_child_prims,
    has_deformable_body_api,
    safe_set_attribute_on_usd_prim,
    safe_set_attribute_on_usd_schema,
)
from ..utils.stage import get_current_stage
from . import schemas_cfg
from ._backend_hooks import _skip_joint_drive

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


def _cfg_fields(cfg) -> dict[str, object]:
    """Return the dataclass field values of a cfg; class-level ``_usd_*`` metadata is not a field."""
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def _resolve_func(cfg: schemas_cfg.SchemaFragment) -> Callable:
    """Return a fragment's applier, resolving a ``module:attr`` import string."""
    return cfg.func if callable(cfg.func) else string_to_callable(cfg.func)


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
    for cfg_field, value in cfg_dict.items():
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


def apply_namespaced(cfg: schemas_cfg.SchemaFragment, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Default fragment applier: apply the fragment's schema and write its namespaced attrs.

    Reads :attr:`~isaaclab.sim.schemas.SchemaFragment._usd_namespace` /
    :attr:`~isaaclab.sim.schemas.SchemaFragment._usd_applied_schema` from the cfg's class. If the
    fragment owns an applied schema, it is applied (once). Each non-``None`` dataclass field is
    written as ``<namespace>:<camelCase(field)>``; the ``func`` field is skipped. ``None`` fields
    are left unchanged on the prim (partial update).

    Args:
        cfg: The fragment instance carrying ``_usd_namespace`` / ``_usd_applied_schema`` metadata.
        prim_path: The prim path to author on.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the properties were successfully set.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    namespace = type(cfg)._usd_namespace
    applied = type(cfg)._usd_applied_schema
    if namespace is None:
        raise ValueError(
            f"Fragment '{type(cfg).__name__}' has no '_usd_namespace' set. Every fragment field is"
            " authored as '<namespace>:<attr>', so a USD namespace is required; non-USD state must"
            " live on the spawner cfg or be passed as a writer keyword argument, not as a fragment"
            " field."
        )
    if applied and applied not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(applied)
    # ``func`` is the only non-USD field; ``mesh_approximation_name`` is the shared ``physics:approximation``
    # token written by ``apply_mesh_collision``, not a ``<namespace>:meshApproximationName`` attribute
    for name, value in _cfg_fields(cfg).items():
        if name in ("func", "mesh_approximation_name") or value is None:
            continue
        safe_set_attribute_on_usd_prim(prim, f"{namespace}:{to_camel_case(name, 'cC')}", value, camel_case=False)
    return True


"""
Articulation root properties.
"""


def apply_articulation_root_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.ArticulationRootFragment],
    stage: Usd.Stage | None = None,
    fix_root_link: bool | None = None,
    create_if_missing: bool = False,
) -> bool:
    """Apply a list of articulation-root fragments to the roots matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.ArticulationRootAPI`` are the targets: each fragment is
    dispatched to every target via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func`.
    Sibling roots (independent articulations matched by one expression) are all processed.
    Nested targets are authored as matched, with a warning -- resolving nested roots is the
    asset author's responsibility.

    With :paramref:`create_if_missing`, the API is applied to every matched prim that lacks
    it. Zero targets warn and return False. Instanced matches are skipped with a warning.

    An empty fragment list is an authoring no-op: it returns True immediately when
    :paramref:`fix_root_link` is None, but still resolves targets and adjusts topology when the
    flag is set. When :paramref:`fix_root_link` is True, the active physics manager creates or
    enables the world joint on each target and returns the backend's final root prim; False
    only disables an existing joint.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: Articulation-root fragments to apply.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.
        fix_root_link: Whether to fix the root link. None leaves topology unchanged.
        create_if_missing: Whether to apply ``UsdPhysics.ArticulationRootAPI`` to every
            matched prim that does not carry it. Defaults to False.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.

    Raises:
        TypeError: If fragments contains a non-articulation fragment.
        RuntimeError: If fixing cannot resolve the active backend or relocate the root.
        NotImplementedError: If the backend cannot fix the resolved root.
    """
    fragments = list(fragments)
    for fragment in fragments:
        if not isinstance(fragment, schemas_cfg.ArticulationRootFragment):
            raise TypeError(
                f"Expected ArticulationRootFragment, got '{type(fragment).__name__}'."
                " Pass legacy cfgs to modify_articulation_root_properties."
            )
    dispatchers = [_resolve_func(fragment) for fragment in fragments]
    if stage is None:
        stage = get_current_stage()
    if not fragments and fix_root_link is None:
        return True

    targets, creation_candidates, any_skipped = _match_fragment_targets(
        prim_path_expr, lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI), stage
    )
    if create_if_missing:
        for prim in creation_candidates:
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            targets.append(prim)
    target_paths = [t.GetPath() for t in targets]
    if any(path != other and path.HasPrefix(other) for path in target_paths for other in target_paths):
        logger.warning(
            "Expression '%s' targets nested articulation roots (%s); authoring on all of them.",
            prim_path_expr,
            [p.pathString for p in target_paths],
        )
    if not targets:
        logger.warning("No articulation-root targets matched expression '%s'; nothing was authored.", prim_path_expr)
        return False

    if fix_root_link:
        from .. import SimulationContext

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError(
                f"Cannot fix articulation roots matched by '{prim_path_expr}' without an active simulation."
            )

    # aggregate per-target, per-fragment results so a reported failure is not masked
    success = not any_skipped
    for root in targets:
        if fix_root_link:
            root = sim.physics_manager.fix_articulation_root(root, stage)
        elif fix_root_link is False:
            joint = find_global_fixed_joint_prim(root.GetPath().pathString, stage=stage)
            if joint is not None:
                joint.GetJointEnabledAttr().Set(False)

        root_path = root.GetPath().pathString
        for fragment, func in zip(fragments, dispatchers):
            success = bool(func(fragment, root_path, stage)) and success

    return success


@deprecated(
    "define_articulation_root_properties is deprecated. Use apply_articulation_root_properties with schema fragments"
    " instead; define_articulation_root_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_articulation_root_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not UsdPhysics.ArticulationRootAPI(prim):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    # ``__wrapped__`` skips only the deprecation warning and keeps the nested traversal
    modify_articulation_root_properties.__wrapped__(prim_path, cfg, stage)


def create_world_fixed_joint(articulation_prim: Usd.Prim, stage: Usd.Stage) -> None:
    """Author a ``UsdPhysics.FixedJoint`` fixing an articulation root link to the world frame.

    This is a pure-USD equivalent of
    ``omni.physx.scripts.utils.createJoint(joint_type="Fixed", from_prim=None, to_prim=articulation_prim)``.
    Authoring directly with USD keeps the fixed-root-link spawn path backend-agnostic:
    it works identically under Kit/PhysX and on kitless backends (e.g. Newton) where
    ``omni.physx`` is unavailable. Only the single-body (world-attached) case is handled,
    matching the fixed-root-link spawn path.

    Args:
        articulation_prim: The articulation root link prim to fix to the world.
        stage: The stage that owns the prim.
    """
    # ``MAX_FLOAT`` used by ``omni.physx.createJoint`` for an effectively unbreakable joint.
    max_break = 3.40282347e38

    to_path = articulation_prim.GetPath().pathString

    # Instanceable/prototype/instance-proxy prims are not authorable; walk up to the first
    # writable ancestor so the joint can be defined there (mirrors ``omni.physx.createJoint``).
    base_prim = articulation_prim
    pseudo_root = stage.GetPseudoRoot()
    while base_prim != pseudo_root and (
        base_prim.IsInPrototype() or base_prim.IsInstanceProxy() or base_prim.IsInstanceable()
    ):
        base_prim = base_prim.GetParent()
    joint_base_path = str(base_prim.GetPrimPath())
    if joint_base_path == "/":
        joint_base_path = ""

    # Find a unique joint name under the writable base (mirrors ``create_unused_path``).
    joint_name = "FixedJoint"
    if stage.GetPrimAtPath(f"{joint_base_path}/{joint_name}").IsValid():
        uniquifier = 0
        while stage.GetPrimAtPath(f"{joint_base_path}/{joint_name}{uniquifier}").IsValid():
            uniquifier += 1
        joint_name = f"{joint_name}{uniquifier}"
    joint = UsdPhysics.FixedJoint.Define(stage, f"{joint_base_path}/{joint_name}")

    # Anchor the joint at the root link's world pose (body0 = world, body1 = root link).
    world_pose = UsdGeom.XformCache().GetLocalToWorldTransform(articulation_prim).RemoveScaleShear()
    joint.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world_pose.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(world_pose.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateBreakForceAttr().Set(max_break)
    joint.CreateBreakTorqueAttr().Set(max_break)


@deprecated(
    "modify_articulation_root_properties is deprecated. Use apply_articulation_root_properties with schema fragments"
    " instead; modify_articulation_root_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_articulation_root_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    articulation_prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.ArticulationRootAPI(articulation_prim):
        return False

    cfg_dict = _cfg_fields(cfg)
    # writer-side (non-USD) flag; the joint is processed after the attribute writes
    fix_root_link = cfg_dict.pop("fix_root_link", None)
    _apply_namespaced_schemas(articulation_prim, cfg, cfg_dict)

    if fix_root_link is not None:
        existing_fixed_joint_prim = find_global_fixed_joint_prim(prim_path, stage=stage)
        # enable/disable an existing world joint, otherwise create one
        if existing_fixed_joint_prim is not None:
            logger.info(
                f"Found an existing fixed joint for the articulation: '{prim_path}'. Setting it to: {fix_root_link}."
            )
            existing_fixed_joint_prim.GetJointEnabledAttr().Set(fix_root_link)
        elif fix_root_link:
            logger.info(f"Creating a fixed joint for the articulation: '{prim_path}'.")

            # the root prim must be a rigid body: there is no obvious way to get the first rigid body link
            # identified by the PhysX parser otherwise
            if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise NotImplementedError(
                    f"The articulation prim '{prim_path}' does not have the RigidBodyAPI applied."
                    " To create a fixed joint, we need to determine the first rigid body link in"
                    " the articulation tree. However, this is not implemented yet."
                )

            create_world_fixed_joint(articulation_prim, stage)

            # The PhysX parser treats a fixed joint on a rigid body as part of a maximal-coordinate tree
            # rather than a fixed-base articulation; moving the articulation root to the parent avoids this.
            parent_prim = articulation_prim.GetParent()
            UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
            if "PhysxArticulationAPI" not in parent_prim.GetAppliedSchemas():
                parent_prim.AddAppliedSchema("PhysxArticulationAPI")

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
            # -- Newton root schema and its authored properties
            newton_root_schema = "NewtonArticulationRootAPI"
            if newton_root_schema in articulation_prim.GetAppliedSchemas():
                if not parent_prim.AddAppliedSchema(newton_root_schema):
                    raise RuntimeError(f"Failed to apply '{newton_root_schema}' to '{parent_prim.GetPath()}'.")
                schema_definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(newton_root_schema)
                newton_properties = []
                if schema_definition is not None:
                    for property_name in schema_definition.GetPropertyNames():
                        prop = articulation_prim.GetProperty(property_name)
                        if prop and prop.IsAuthored():
                            newton_properties.append(prop)
                for prop in newton_properties:
                    if not prop.FlattenTo(parent_prim):
                        raise RuntimeError(f"Failed to move '{prop.GetPath()}' to '{parent_prim.GetPath()}'.")
                for prop in newton_properties:
                    if not articulation_prim.RemoveProperty(prop.GetName()):
                        raise RuntimeError(f"Failed to remove '{prop.GetPath()}' from the former articulation root.")
                if not articulation_prim.RemoveAppliedSchema(newton_root_schema):
                    raise RuntimeError(f"Failed to remove '{newton_root_schema}' from '{articulation_prim.GetPath()}'.")

            articulation_prim.RemoveAppliedSchema("PhysxArticulationAPI")
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            articulation_prim = parent_prim

    # mirrored after any root relocation so the Newton API does not recreate a root on the former root link
    enabled_self_collisions = cfg_dict.get("enabled_self_collisions")
    if enabled_self_collisions is not None:
        if "NewtonArticulationRootAPI" not in articulation_prim.GetAppliedSchemas():
            articulation_prim.AddAppliedSchema("NewtonArticulationRootAPI")
        safe_set_attribute_on_usd_prim(
            articulation_prim, "newton:selfCollisionEnabled", enabled_self_collisions, camel_case=False
        )
    return True


"""
Fragment-writer helpers.
"""


def _match_fragment_targets(
    prim_path_expr: str,
    is_target: Callable[[Usd.Prim], bool],
    stage: Usd.Stage,
) -> tuple[list[Usd.Prim], list[Usd.Prim], bool]:
    """Resolve fragment-writer targets from a prim path expression.

    Matches ``prim_path_expr`` with :func:`~isaaclab.sim.utils.find_matching_prims` (a plain
    regular expression over whole prim paths) and splits the matches: writable prims
    passing ``is_target`` are targets, writable prims failing it are creation candidates, and
    instanced prims passing it are skipped with a warning since prototypes cannot be authored on.

    Args:
        prim_path_expr: The prim path expression to match. Path-like objects (e.g.
            ``Sdf.Path``) are accepted and converted with ``str``.
        is_target: Predicate deciding whether a matched prim is a valid family target.
        stage: The stage to match on.

    Returns:
        A tuple ``(targets, creation_candidates, any_skipped)``.
    """
    # tolerate path-like inputs such as Sdf.Path, whose string form is the path itself
    prim_path_expr = str(prim_path_expr)
    targets = []
    creation_candidates = []
    skipped = []
    for prim in find_matching_prims(prim_path_expr, stage):
        instanced = prim.IsInstance() or prim.IsInstanceProxy()
        if is_target(prim):
            (skipped if instanced else targets).append(prim)
        elif not instanced:
            creation_candidates.append(prim)
    if skipped:
        logger.warning(
            "Skipping fragment updates on instanced prims matched by '%s': %s.",
            prim_path_expr,
            [p.GetPath().pathString for p in skipped],
        )
    return targets, creation_candidates, bool(skipped)


def _apply_api_fragments(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.SchemaFragment],
    api: type[Usd.APISchemaBase],
    family: str,
    create_if_missing: bool,
    stage: Usd.Stage | None,
) -> bool:
    """Dispatch fragments to every matched prim carrying ``api``, the family's implicit anchor.

    Shared by the rigid-body, collision and mass writers: an empty fragment list is a no-op that
    returns True, ``create_if_missing`` applies ``api`` to matched prims lacking it, zero targets
    warn and return False, and per-target results are aggregated so a failure is never masked.
    """
    fragments = list(fragments)
    if stage is None:
        stage = get_current_stage()
    if not fragments:
        return True
    targets, creation_candidates, any_skipped = _match_fragment_targets(prim_path_expr, lambda p: p.HasAPI(api), stage)
    if create_if_missing:
        for prim in creation_candidates:
            api.Apply(prim)
            targets.append(prim)
    if not targets:
        logger.warning("No %s targets matched expression '%s'; nothing was authored.", family, prim_path_expr)
        return False
    dispatchers = [_resolve_func(cfg) for cfg in fragments]
    success = not any_skipped
    for target in targets:
        target_path = target.GetPath().pathString
        for cfg, func in zip(fragments, dispatchers):
            success = bool(func(cfg, target_path, stage)) and success
    return success


"""
Rigid body properties.
"""


def apply_rigid_body_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.RigidBodyFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
) -> bool:
    """Apply a list of rigid-body fragments to the rigid bodies matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.RigidBodyAPI`` are modified in place: each fragment is
    dispatched to every such target via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. Backend fragments carry backend-specific
    funcs, so core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. With
    :paramref:`create_if_missing`, ``UsdPhysics.RigidBodyAPI`` is applied to every matched
    prim that lacks it; only the asset's joints decide which bodies participate in the
    articulation, so the expression is trusted as written. Zero targets warn and return
    False. Instanced matches are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.RigidBodyFragment` instances.
        create_if_missing: Whether to apply ``UsdPhysics.RigidBodyAPI`` to every matched
            prim that does not carry it. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.
    """
    return _apply_api_fragments(
        prim_path_expr, fragments, UsdPhysics.RigidBodyAPI, "rigid-body", create_if_missing, stage
    )


def apply_mesh_collision(
    cfg: schemas_cfg.MeshCollisionFragment, prim_path: str, stage: Usd.Stage | None = None
) -> bool:
    """Apply a single mesh-collision fragment: its namespaced cooking attrs plus the shared token.

    This is the default :attr:`~isaaclab.sim.schemas.SchemaFragment.func` for every
    :class:`~isaaclab.sim.schemas.MeshCollisionFragment`. Unlike the generic :func:`apply_namespaced`,
    a mesh-collision fragment additionally authors the shared ``physics:approximation`` token (via the
    standard ``UsdPhysics.MeshCollisionAPI``) on top of its own backend cooking namespace.

    The token is *not* a plain namespaced attribute -- it is shared state on the family anchor implied
    by the present cooking fragment. Each fragment carries a :attr:`mesh_approximation_name` whose
    default encodes the token its schema implies (e.g. ``"convexHull"`` for :class:`PhysxConvexHullCfg`,
    ``"sdf"`` for :class:`PhysxSDFMeshCfg`). A name of ``"none"`` leaves the token unchanged, so when
    several fragments are dispatched in order by :func:`apply_mesh_collision_properties` the last one
    with a non-``"none"`` name wins -- this is how a core fragment composes with a backend cooking
    fragment. The name is validated against :const:`MESH_APPROXIMATION_TOKENS`; an unknown name raises
    ``ValueError``. :attr:`mesh_approximation_name` is skipped by :func:`apply_namespaced`, so it is
    never authored as a spurious ``<namespace>:meshApproximationName`` attribute.

    Args:
        cfg: The mesh-collision fragment to apply.
        prim_path: The prim path to author on. This prim should be a Mesh.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the fragment was applied successfully.

    Raises:
        ValueError: If the prim at ``prim_path`` is not valid, or when the fragment's mesh
            approximation name is not in :const:`MESH_APPROXIMATION_TOKENS`.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # the standard MeshCollisionAPI anchor carries ``physics:approximation``
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # the generic applier skips ``mesh_approximation_name``; the shared token is written below
    success = apply_namespaced(cfg, prim_path, stage)
    # ``"none"`` leaves the token untouched so a later non-"none" fragment in a list dispatch wins
    name = getattr(cfg, "mesh_approximation_name", None)
    if name is not None and name != "none":
        _write_mesh_approximation(prim, name)
    return success


def _write_mesh_approximation(prim: Usd.Prim, name: str) -> None:
    """Author ``physics:approximation`` on a ``MeshCollisionAPI`` prim from a token name."""
    if name not in MESH_APPROXIMATION_TOKENS:
        raise ValueError(
            f"Invalid mesh approximation name: '{name}'. Valid options are: {list(MESH_APPROXIMATION_TOKENS)}"
        )
    safe_set_attribute_on_usd_schema(
        UsdPhysics.MeshCollisionAPI(prim), "Approximation", MESH_APPROXIMATION_TOKENS[name], camel_case=False
    )


def apply_mesh_collision_properties(
    prim_path: str, fragments: Iterable[schemas_cfg.MeshCollisionFragment], stage: Usd.Stage | None = None
) -> bool:
    """Apply a list of mesh-collision fragments to a prim.

    Applies ``UsdPhysics.MeshCollisionAPI`` as the implicit anchor (the carrier of the
    ``physics:approximation`` token), then dispatches each fragment via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. The default mesh-collision func
    (:func:`apply_mesh_collision`) authors both the fragment's backend cooking namespace and the
    shared approximation token it implies, so composing a core fragment with a backend cooking
    fragment lets the last fragment with a non-``"none"`` :attr:`mesh_approximation_name` set the
    token. Backend cooking fragments carry their own funcs, so core never imports a backend.

    Args:
        prim_path: The prim path to apply the mesh-collision schemas on. This prim should be a Mesh.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.MeshCollisionFragment` instances.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if all fragments applied successfully, False if any fragment reported failure.

    Raises:
        ValueError: If the prim at ``prim_path`` is not valid, or when a fragment's mesh
            approximation name is not in :const:`MESH_APPROXIMATION_TOKENS`.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # aggregate per-fragment results so a reported failure is not masked
    success = True
    for cfg in fragments:
        success = bool(_resolve_func(cfg)(cfg, prim_path, stage)) and success
    return success


@deprecated(
    "define_rigid_body_properties is deprecated. Use apply_rigid_body_properties with schema fragments"
    " instead; define_rigid_body_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_rigid_body_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    modify_rigid_body_properties.__wrapped__(prim_path, cfg, stage)


# rigid bodies nest when child links are authored under their parent link prim (URDF importer
# in Isaac Sim 6.0+), so keep descending after a success to reach every link
@deprecated(
    "modify_rigid_body_properties is deprecated. Use apply_rigid_body_properties with schema fragments"
    " instead; modify_rigid_body_properties will be removed in 3.2."
)
@apply_nested(stop_on_success=False)
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

    .. deprecated:: 3.1
        Use :func:`apply_rigid_body_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    rigid_body_prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        return False
    # base fields route to ``physics:*``, ``disable_gravity`` via field exceptions, PhysX-subclass
    # fields to ``physxRigidBody:*``
    _apply_namespaced_schemas(rigid_body_prim, cfg, _cfg_fields(cfg))
    return True


"""
Collision properties.
"""


def apply_collision_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.CollisionFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
) -> bool:
    """Apply a list of collision fragments to the colliders matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.CollisionAPI`` are modified in place: each fragment is
    dispatched to every such target via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. Backend fragments carry backend-specific
    funcs, so core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. With
    :paramref:`create_if_missing`, ``UsdPhysics.CollisionAPI`` is applied to every matched
    prim that lacks it. When no target remains, a warning is emitted and False is
    returned without authoring anything. Matched prims inside instances cannot be authored on
    and are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.CollisionFragment` instances.
        create_if_missing: Whether to apply ``UsdPhysics.CollisionAPI`` to matched prims that
            do not carry it. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.
    """
    return _apply_api_fragments(
        prim_path_expr, fragments, UsdPhysics.CollisionAPI, "collision", create_if_missing, stage
    )


@deprecated(
    "define_collision_properties is deprecated. Use apply_collision_properties with schema fragments"
    " instead; define_collision_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_collision_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
    modify_collision_properties.__wrapped__(prim_path, cfg, stage)


@deprecated(
    "modify_collision_properties is deprecated. Use apply_collision_properties with schema fragments"
    " instead; modify_collision_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_collision_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    collider_prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    cfg_dict = _cfg_fields(cfg)
    # the nested mesh-collision cfg is dispatched to its own legacy writer
    mesh_collision_cfg = cfg_dict.pop("mesh_collision_property", None)
    if mesh_collision_cfg is not None:
        modify_mesh_collision_properties.__wrapped__(prim_path, mesh_collision_cfg, stage)
    # ``collision_enabled`` routes to ``physics:*``, the offsets via field exceptions, PhysX-subclass
    # fields to ``physxCollision:*``
    _apply_namespaced_schemas(collider_prim, cfg, cfg_dict)
    return True


"""
Mass properties.
"""


def apply_mass_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.MassFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
) -> bool:
    """Apply a list of mass fragments to the mass-bearing prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.MassAPI`` are modified in place: each fragment is dispatched to
    every such target via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. Backend
    fragments carry backend-specific funcs, so core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. With
    :paramref:`create_if_missing`, ``UsdPhysics.MassAPI`` is applied to every matched prim
    that lacks it; pairing the mass with a rigid body is the caller's responsibility. Zero
    targets warn and return False. Instanced matches are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.MassFragment` instances.
        create_if_missing: Whether to apply ``UsdPhysics.MassAPI`` to every matched prim
            that does not carry it. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.
    """
    return _apply_api_fragments(prim_path_expr, fragments, UsdPhysics.MassAPI, "mass", create_if_missing, stage)


@deprecated(
    "define_mass_properties is deprecated. Use apply_mass_properties with schema fragments"
    " instead; define_mass_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_mass_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not UsdPhysics.MassAPI(prim):
        UsdPhysics.MassAPI.Apply(prim)
    modify_mass_properties.__wrapped__(prim_path, cfg, stage)


# mass is authored on the same link prims as the rigid-body schema, which may nest (see
# modify_rigid_body_properties above), so keep descending after a success
@deprecated(
    "modify_mass_properties is deprecated. Use apply_mass_properties with schema fragments"
    " instead; modify_mass_properties will be removed in 3.2."
)
@apply_nested(stop_on_success=False)
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

    .. deprecated:: 3.1
        Use :func:`apply_mass_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    rigid_prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.MassAPI(rigid_prim):
        return False
    _apply_namespaced_schemas(rigid_prim, cfg, _cfg_fields(cfg))
    return True


"""
Contact sensor.
"""


def activate_contact_sensors(prim_path: str, threshold: float = 0.0, stage: Usd.Stage | None = None):
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
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # nested rigid-body trees are included
    rigid_body_prims = get_all_matching_child_prims(
        prim_path,
        predicate=lambda child_prim: child_prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=stage,
        traverse_instance_prims=False,
    )
    if not rigid_body_prims:
        descendant_count = sum(1 for _ in Usd.PrimRange(prim)) - 1
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
    for child_prim in rigid_body_prims:
        child_applied = child_prim.GetAppliedSchemas()
        # a zero sleep threshold keeps the body awake so contacts are always reported
        if "PhysxRigidBodyAPI" not in child_applied:
            child_prim.AddAppliedSchema("PhysxRigidBodyAPI")
        safe_set_attribute_on_usd_prim(child_prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
        if "PhysxContactReportAPI" not in child_applied:
            child_prim.AddAppliedSchema("PhysxContactReportAPI")
        safe_set_attribute_on_usd_prim(child_prim, "physxContactReport:threshold", threshold, camel_case=False)
    return True


"""
Joint drive properties.
"""


def _drive_instance_name(prim) -> str | None:
    """Return the ``UsdPhysics.DriveAPI`` instance for a joint prim, or ``None`` if it has no drive.

    Revolute joints use the ``"angular"`` instance, prismatic joints the ``"linear"`` instance; any
    other prim type has no joint drive. Shared by :func:`apply_drive` and :func:`_ensure_drive_exists`.

    Args:
        prim: The candidate joint prim.

    Returns:
        ``"angular"``, ``"linear"``, or ``None`` when the prim is not a revolute/prismatic joint.
    """
    if prim.IsA(UsdPhysics.RevoluteJoint):
        return "angular"
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return "linear"
    return None


def apply_drive(cfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Apply a :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment to a single joint prim.

    This is the override ``func`` for the ``UsdPhysics.DriveAPI`` fragment: the drive attributes
    live under a multi-instance schema, so the generic :func:`apply_namespaced` writer cannot be
    used. The writer reproduces the solver-common drive logic of
    :func:`modify_joint_drive_properties`:

    * Selects the drive instance: ``"angular"`` for a revolute joint, ``"linear"`` for a prismatic
      joint. For any other prim type, the function is a no-op and returns ``False``.
    * Skips joints excluded by a backend-registered predicate (see
      :func:`register_joint_drive_skip_predicate`, e.g. PhysX tendon members), returning ``False``.
    * Applies ``UsdPhysics.DriveAPI`` for the selected instance (presence-gated -- only applied when
      this fragment is present).
    * Converts angular-drive :attr:`stiffness` and :attr:`damping` from radians to degrees
      (``N·m/rad`` -> ``N·m/deg`` and ``N·m·s/rad`` -> ``N·m·s/deg``); linear drives are written
      as-is.
    * Writes the typed ``drive:<inst>:physics:{type,maxForce,stiffness,damping}`` attributes,
      mapping the :attr:`drive_type` field to the USD attribute named ``type``.

    Args:
        cfg: The :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment to apply.
        prim_path: The joint prim path to author on.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the drive was applied to a joint prim, False if the prim is not a revolute or
        prismatic joint (or is a tendon child).
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    drive_api_name = _drive_instance_name(prim)
    # skip non-joints and joints a backend owns (e.g. PhysX tendon members)
    if drive_api_name is None or _skip_joint_drive(prim):
        return False
    usd_drive_api = _get_or_apply_drive_api(prim, drive_api_name)
    _write_drive_attributes(usd_drive_api, drive_api_name, cfg.drive_type, cfg.max_force, cfg.stiffness, cfg.damping)
    return True


def _get_or_apply_drive_api(prim: Usd.Prim, drive_api_name: str) -> UsdPhysics.DriveAPI:
    """Return the joint's ``UsdPhysics.DriveAPI`` instance, applying it when absent."""
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name)
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_api_name)
    return usd_drive_api


def _write_drive_attributes(
    usd_drive_api: UsdPhysics.DriveAPI,
    drive_api_name: str,
    drive_type: str | None,
    max_force: float | None,
    stiffness: float | None,
    damping: float | None,
) -> None:
    """Write the solver-common ``UsdPhysics.DriveAPI`` attributes, skipping ``None`` values.

    Angular drives are stored in degree units in USD, so stiffness [N·m/rad] and damping
    [N·m·s/rad] are converted to per-degree values. ``drive_type`` maps to the USD attribute
    ``type``; every other field follows the snake_case -> camelCase convention.
    """
    if drive_api_name == "angular":
        if stiffness is not None:
            stiffness = stiffness * math.pi / 180.0
        if damping is not None:
            damping = damping * math.pi / 180.0
    for attr_name, value in (
        ("type", drive_type),
        ("max_force", max_force),
        ("stiffness", stiffness),
        ("damping", damping),
    ):
        if value is not None:
            safe_set_attribute_on_usd_schema(usd_drive_api, attr_name, value, camel_case=True)


def apply_joint_drive_properties(
    prim_path_expr: str,
    fragments,
    stage: Usd.Stage | None = None,
    ensure_drives_exist: bool = False,
    create_if_missing: bool = False,
) -> bool:
    """Apply a list of joint-drive fragments to the joint prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. The fragments are
    dispatched to every matched revolute/prismatic joint prim that is not excluded by a
    backend-registered skip predicate (see :func:`register_joint_drive_skip_predicate`, e.g.
    PhysX tendon members). Non-joint matches are ignored silently -- a subtree expression
    matches every descendant, so per-prim warnings would spam. Matched prims inside
    instances cannot be authored on and are skipped with a warning.

    Unlike :func:`apply_rigid_body_properties`, the joint-drive family has no implicit anchor:
    ``UsdPhysics.DriveAPI`` is *presence-gated* and applied only by :func:`apply_drive` when a
    :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment is present in ``fragments``. Each
    fragment is dispatched via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func`, so backend
    fragments carry backend-specific funcs and core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. When no fragment succeeds on
    any joint, a warning is emitted and False is returned.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.JointDriveFragment` instances.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.
        ensure_drives_exist: If True, write a minimal stiffness (``1e-3``) to any drive whose
            authored stiffness *and* damping are both zero, so that backends (e.g. Newton) treat
            the drive as active. Reproduces the legacy
            :attr:`~isaaclab.sim.schemas.JointDriveBaseCfg.ensure_drives_exist` behaviour. This is
            a spawner-level flag, not a fragment field.
        create_if_missing: If True, apply the axis-appropriate ``UsdPhysics.DriveAPI`` instance
            (``"angular"`` for revolute joints, ``"linear"`` for prismatic joints) on matched
            joints that do not carry it, before dispatching the fragments. Distinct from
            :paramref:`ensure_drives_exist`: this flag creates the drive API itself, whereas
            :paramref:`ensure_drives_exist` seeds a minimal stiffness on fully-passive drives
            that already exist. Defaults to False.

    Returns:
        True if the fragments were applied to at least one joint prim and no instanced joint
        was skipped, False otherwise.
    """
    if stage is None:
        stage = get_current_stage()
    fragments = list(fragments)
    if not fragments:
        return True
    # ``ensure_drives_exist`` only makes sense for the solver-common drive fragment
    drive_cfg = next((f for f in fragments if isinstance(f, schemas_cfg.UsdPhysicsDriveCfg)), None)
    dispatchers = [_resolve_func(cfg) for cfg in fragments]

    # non-joint matches are ignored silently since a subtree expression matches every descendant
    targets, _, any_skipped = _match_fragment_targets(
        prim_path_expr, lambda p: _drive_instance_name(p) is not None and not _skip_joint_drive(p), stage
    )

    count_success = 0
    for joint_prim in targets:
        joint_prim_path = joint_prim.GetPath().pathString
        drive_api_name = _drive_instance_name(joint_prim)
        if create_if_missing:
            _get_or_apply_drive_api(joint_prim, drive_api_name)
        results = [bool(func(cfg, joint_prim_path, stage)) for cfg, func in zip(fragments, dispatchers)]
        if not any(results):
            continue
        count_success += 1
        if ensure_drives_exist and drive_cfg is not None:
            _ensure_drive_exists(drive_cfg, joint_prim, drive_api_name)

    # instanced skips were already reported by the matcher; only warn when nothing matched at all
    if count_success == 0 and not any_skipped:
        logger.warning(
            "Could not apply joint-drive properties on any joints matched by '%s'."
            " No revolute/prismatic joint prims matched or every fragment reported failure.",
            prim_path_expr,
        )
    return count_success > 0 and not any_skipped


def _is_passive_drive(usd_drive_api: UsdPhysics.DriveAPI) -> bool:
    """Return whether a drive has neither stiffness nor damping authored (or both are zero)."""
    return not usd_drive_api.GetStiffnessAttr().Get() and not usd_drive_api.GetDampingAttr().Get()


def _ensure_drive_exists(drive_cfg: schemas_cfg.UsdPhysicsDriveCfg, prim: Usd.Prim, drive_api_name: str) -> None:
    """Seed a minimal stiffness on a fully-passive drive so backends treat it as active.

    Reproduces the legacy ``ensure_drives_exist`` behaviour: if the drive fragment authored
    neither :attr:`stiffness` nor :attr:`damping` and the authored drive currently has zero
    (or unset) stiffness *and* damping, write a minimal stiffness of ``1e-3`` directly to the
    drive API (converted to degree units for angular drives, matching :func:`apply_drive`). The
    fragment is not mutated, so this is safe across multiple joint prims sharing one fragment.

    Args:
        drive_cfg: The :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment.
        prim: The joint prim being authored.
        drive_api_name: The drive instance of the joint (``"angular"`` or ``"linear"``).
    """
    if drive_cfg.stiffness is not None or drive_cfg.damping is not None:
        return
    usd_drive_api = _get_or_apply_drive_api(prim, drive_api_name)
    if _is_passive_drive(usd_drive_api):
        _write_drive_attributes(usd_drive_api, drive_api_name, None, None, 1e-3, None)


@deprecated(
    "modify_joint_drive_properties is deprecated. Use apply_joint_drive_properties with schema fragments"
    " instead; modify_joint_drive_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_joint_drive_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    drive_api_name = _drive_instance_name(prim)
    if drive_api_name is None:
        return False
    # tendon child prims are controlled by the tendon, not a drive
    applied_schemas_str = str(prim.GetAppliedSchemas())
    if "PhysxTendonAxisAPI" in applied_schemas_str and "PhysxTendonAxisRootAPI" not in applied_schemas_str:
        return False
    usd_drive_api = _get_or_apply_drive_api(prim, drive_api_name)

    cfg_dict = _cfg_fields(cfg)
    # seed a minimal stiffness on a passive drive so backends like Newton treat it as active
    ensure_drives = cfg_dict.pop("ensure_drives_exist", False)
    if ensure_drives and cfg_dict["stiffness"] is None and cfg_dict["damping"] is None:
        if _is_passive_drive(usd_drive_api):
            cfg_dict["stiffness"] = 1e-3
    # PhysX stores angular velocities in deg/s
    if drive_api_name == "angular" and cfg_dict.get("max_joint_velocity") is not None:
        cfg_dict["max_joint_velocity"] = cfg_dict["max_joint_velocity"] * 180.0 / math.pi

    # solver-common ``UsdPhysics.DriveAPI`` fields; the remainder is PhysX-namespaced
    drive_values = [cfg_dict.pop(name, None) for name in ("drive_type", "max_force", "stiffness", "damping")]
    _write_drive_attributes(usd_drive_api, drive_api_name, *drive_values)
    _apply_namespaced_schemas(prim, cfg, cfg_dict)
    return True


"""
Fixed tendon properties.
"""


_FIXED_TENDON_SCHEMAS = ("PhysxTendonAxisRootAPI", "PhysxTendonAxisAPI")
_SPATIAL_TENDON_SCHEMAS = ("PhysxTendonAttachmentRootAPI",)


def _write_tendon_properties(prim: Usd.Prim, values: dict[str, object], schema_type: str) -> bool:
    """Write ``physxTendon:<instance>:*`` values to every applied instance of a multi-apply tendon schema.

    Returns:
        True if at least one instance of ``schema_type`` was found on the prim.
    """
    authored = False
    for schema in prim.GetAppliedSchemas():
        applied_type, instance = Usd.SchemaRegistry.GetTypeNameAndInstance(str(schema))
        if applied_type != schema_type or not instance:
            continue
        authored = True
        for name, value in values.items():
            attribute = f"physxTendon:{instance}:{to_camel_case(name, 'cC')}"
            safe_set_attribute_on_usd_prim(prim, attribute, value, camel_case=False)
    return authored


def _apply_tendon_fragments(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.SchemaFragment],
    schema_types: tuple[str, ...],
    prim_types: tuple[str, ...],
    family: str,
    stage: Usd.Stage | None,
) -> bool:
    """Dispatch tune-not-apply tendon fragments to prims carrying a tendon schema or prim type.

    A fragment succeeds when its func returns True on at least one target, so a mixed-backend
    target set (each func no-ops on the other backend's prims) does not fail the write.
    """
    fragments = list(fragments)
    if stage is None:
        stage = get_current_stage()
    if not fragments:
        return True
    targets, _, any_skipped = _match_fragment_targets(
        prim_path_expr,
        lambda prim: (
            prim.GetTypeName() in prim_types
            or any(
                Usd.SchemaRegistry.GetTypeNameAndInstance(str(schema))[0] in schema_types
                for schema in prim.GetAppliedSchemas()
            )
        ),
        stage,
    )
    if not targets:
        logger.warning("No %s-tendon targets matched expression '%s'; nothing was authored.", family, prim_path_expr)
        return False
    target_paths = [target.GetPath().pathString for target in targets]
    success = not any_skipped
    for cfg in fragments:
        func = _resolve_func(cfg)
        # every target is visited; the list keeps ``any`` from short-circuiting the dispatch
        results = [bool(func(cfg, path, stage)) for path in target_paths]
        success = any(results) and success
    return success


def apply_fixed_tendon_properties(
    prim_path_expr: str, fragments: Iterable[schemas_cfg.FixedTendonFragment], stage: Usd.Stage | None = None
) -> bool:
    """Apply a list of fixed-tendon fragments to the tendon prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. A matched prim is a
    fixed-tendon target when it carries an applied ``PhysxTendonAxisRootAPI`` or
    ``PhysxTendonAxisAPI`` instance, or is a ``MjcTendon`` prim.

    Fixed tendons are a *tune-not-apply* family: the tendon topology is authored in the source
    asset, so this writer never creates instances -- it only dispatches each fragment via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func` to every matched target. Backend
    fragments carry backend-specific funcs, so core never imports a backend. A fragment
    succeeds when its func returns True on at least one target: each func only tunes its own
    backend's representation and no-ops (returns False) on the other backend's prims, so a
    mixed-backend target set does not fail the write.

    An empty fragment list is an authoring no-op and returns True. When no target matches, a
    warning is emitted and False is returned without authoring anything. Matched prims inside
    instances cannot be authored on and are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.FixedTendonFragment` instances.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every fragment tuned at least one target and no instanced prim was skipped.
    """
    return _apply_tendon_fragments(prim_path_expr, fragments, _FIXED_TENDON_SCHEMAS, ("MjcTendon",), "fixed", stage)


@deprecated(
    "modify_fixed_tendon_properties is deprecated. Use apply_fixed_tendon_properties with schema fragments"
    " instead; modify_fixed_tendon_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_fixed_tendon_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    tendon_prim = stage.GetPrimAtPath(prim_path)
    values = cfg.to_dict()
    if tendon_prim.GetTypeName() != "MjcTendon":
        return _write_tendon_properties(tendon_prim, values, "PhysxTendonAxisRootAPI")
    # MuJoCo tendons only share the stiffness/damping fields
    for name in ("stiffness", "damping"):
        safe_set_attribute_on_usd_prim(tendon_prim, f"mjc:{name}", values.get(name), camel_case=False)
    return True


"""
Spatial tendon properties.
"""


def apply_spatial_tendon_properties(
    prim_path_expr: str, fragments: Iterable[schemas_cfg.SpatialTendonFragment], stage: Usd.Stage | None = None
) -> bool:
    """Apply a list of spatial-tendon fragments to the tendon prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. A matched prim is a
    spatial-tendon target when it carries an applied ``PhysxTendonAttachmentRootAPI`` instance.

    Spatial tendons are a *tune-not-apply* family: the tendon topology is authored in the
    source asset, so this writer never creates instances -- it only dispatches each fragment
    via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func` to every matched target. Backend
    fragments carry backend-specific funcs, so core never imports a backend. A fragment
    succeeds when its func returns True on at least one target: each func only tunes its own
    backend's representation and no-ops (returns False) on the other backend's prims, so a
    mixed-backend target set does not fail the write.

    An empty fragment list is an authoring no-op and returns True. When no target matches, a
    warning is emitted and False is returned without authoring anything. Matched prims inside
    instances cannot be authored on and are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.SpatialTendonFragment` instances.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every fragment tuned at least one target and no instanced prim was skipped.
    """
    return _apply_tendon_fragments(prim_path_expr, fragments, _SPATIAL_TENDON_SCHEMAS, (), "spatial", stage)


@deprecated(
    "modify_spatial_tendon_properties is deprecated. Use apply_spatial_tendon_properties with schema fragments"
    " instead; modify_spatial_tendon_properties will be removed in 3.2."
)
@apply_nested
def modify_spatial_tendon_properties(
    prim_path: str, cfg: schemas_cfg.PhysxSpatialTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a spatial tendon attachment prim.

    A `spatial tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises attributes that belong to the `PhysxTendonAttachmentRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _spatial tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationSpatialTendon.html
    .. _PhysxTendonAttachmentRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_root_a_p_i.html

    Args:
        prim_path: The prim path to the tendon attachment.
        cfg: The configuration for the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.

    .. deprecated:: 3.1
        Use :func:`apply_spatial_tendon_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    tendon_prim = stage.GetPrimAtPath(prim_path)
    return _write_tendon_properties(tendon_prim, cfg.to_dict(), "PhysxTendonAttachmentRootAPI")


"""
Collision mesh properties.
"""


@deprecated(
    "define_mesh_collision_properties is deprecated. Use apply_mesh_collision_properties with schema fragments"
    " instead; define_mesh_collision_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_mesh_collision_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # the standard MeshCollisionAPI is always applied; the PhysX cooking schema (if any) is applied
    # by the writer only when a PhysX-namespaced tuning field is set
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    modify_mesh_collision_properties.__wrapped__(prim_path, cfg, stage)


@deprecated(
    "modify_mesh_collision_properties is deprecated. Use apply_mesh_collision_properties with schema fragments"
    " instead; modify_mesh_collision_properties will be removed in 3.2."
)
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

    .. deprecated:: 3.1
        Use :func:`apply_mesh_collision_properties` with schema fragments instead. This function will be removed
        in 3.2.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    cfg_dict = _cfg_fields(cfg)
    _write_mesh_approximation(prim, cfg_dict.pop("mesh_approximation_name", "none"))
    # PhysX cooking subclasses author their tuning fields under e.g. ``physxConvexHullCollision:*``;
    # the helper applies the cooking schema only when a tuning field is set, so Newton-targeted
    # prims stay free of PhysX schemas they did not opt in to
    _apply_namespaced_schemas(prim, cfg, cfg_dict)
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


def _tetrahedralize_surface(
    vertices: np.ndarray, faces: np.ndarray, edge_length_fac: float, prim_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a triangle surface mesh with pytetwild, returning right-handed tets.

    Args:
        vertices: Surface vertex positions [m], shape [N, 3].
        faces: Flattened triangle vertex indices, shape [3 * F].
        edge_length_fac: Relative target edge length for the tetrahedralization.
        prim_path: The deformable prim path, used in the error message when the dependency is missing.

    Returns:
        The tetrahedral mesh points [m], shape [M, 3], and tet vertex indices, shape [T, 4].

    Raises:
        ModuleNotFoundError: If the optional tetrahedralization dependencies are not installed.
    """
    try:
        from pytetwild import tetrahedralize
    except ModuleNotFoundError as exc:
        if exc.name not in {"pytetwild", "pyvista", "vtk", "vtkmodules"}:
            raise
        raise ModuleNotFoundError(
            "Automatic tetrahedralization of volume deformables requires the optional "
            "tetrahedralization dependencies. Install them with "
            "uv sync --inexact --extra tetrahedralization from a source checkout "
            "(or ./isaaclab.sh -i tetrahedralization with the legacy installer), or "
            'pip install "isaaclab[tetrahedralization]" from a wheel. Alternatively, provide '
            f"a pre-tetrahedralized UsdGeom.TetMesh under the deformable prim {prim_path}."
        ) from exc

    tet_mesh_points, tet_mesh_indices = tetrahedralize(
        vertices, faces.reshape(-1, 3), edge_length_fac=edge_length_fac, simplify=False, epsilon=1e-2, coarsen=True
    )
    # pytetwild's default ordering does not guarantee positive signed volume, which
    # ``UsdGeom.TetMesh`` and ``ComputeSurfaceFaces`` require. Flip any inverted tets.
    tet_points_wp = wp.array(tet_mesh_points.astype(np.float32), dtype=wp.vec3, device="cpu")
    tet_indices_wp = wp.array(np.asarray(tet_mesh_indices, dtype=np.int32).reshape(-1, 4), dtype=wp.int32, device="cpu")
    wp.launch(
        _fix_tet_winding_kernel, dim=tet_indices_wp.shape[0], inputs=[tet_points_wp, tet_indices_wp], device="cpu"
    )
    return tet_mesh_points, tet_indices_wp.numpy()


def define_deformable_curve_properties(prim_path: str, stage: Usd.Stage | None = None) -> None:
    """Apply the deformable curve simulation schema.

    Args:
        prim_path: The path of the ``UsdGeom.BasisCurves`` prim.
        stage: The stage where the prim exists. Defaults to the current stage.

    Raises:
        ValueError: If the prim path is invalid or is not a ``UsdGeom.BasisCurves`` prim.
        RuntimeError: If the schema cannot be applied.
    """
    if stage is None:
        stage = get_current_stage()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not prim.IsA(UsdGeom.BasisCurves):
        raise ValueError(f"Prim path '{prim_path}' is not a UsdGeom.BasisCurves prim.")

    schema_name = "PhysicsCurvesDeformableSimAPI"
    if schema_name in prim.GetPrimTypeInfo().GetAppliedAPISchemas():
        return
    if not prim.AddAppliedSchema(schema_name):
        raise RuntimeError(f"Failed to set deformable curve API on prim '{prim_path}'.")


def define_deformable_body_properties(
    prim_path: str,
    cfg: schemas_cfg.DeformableBodyPropertiesBaseCfg,
    stage: Usd.Stage | None = None,
    deformable_type: str = "volume",
    sim_mesh_prim_path: str | None = None,
    tetrahedralization_edge_length_fac: float = 0.1,
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

    .. note::
        This function authors a new deformable body setup from scratch. It does not remove or clear existing
        deformable body schemas, simulation meshes, or pose data. Use :func:`modify_deformable_body_properties`
        to update properties on an existing deformable body, or clear any previous setup before calling this
        function.

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
        tetrahedralization_edge_length_fac: Relative target edge length for automatic tetrahedralization.
            Defaults to ``0.1``.

    Raises:
        ValueError: When the prim path is not valid.
        ValueError: When the prim has no mesh or multiple meshes.
        ModuleNotFoundError: When automatic volume tetrahedralization is requested
            without its optional dependencies.
        RuntimeError: When setting the deformable body properties fails.
    """
    if stage is None:
        stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # volume deformables may ship a pre-tetrahedralized TetMesh to use as the simulation mesh
    sim_mesh_prim = None
    if deformable_type == "volume":
        matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "TetMesh")
        if len(matching_prims) > 1:
            mesh_paths = [p.GetPrimPath() for p in matching_prims]
            raise ValueError(
                f"Found multiple tetrahedral meshes in '{prim_path}': {mesh_paths}."
                " Deformable body schema can only be applied to one mesh for now."
            )
        if matching_prims:
            sim_mesh_prim = matching_prims[0]
            if not sim_mesh_prim.IsValid():
                raise ValueError(f"Mesh prim path '{sim_mesh_prim.GetPrimPath()}' is not valid.")

    matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh")
    if len(matching_prims) == 0:
        # with a TetMesh but no Mesh, the TetMesh surface becomes the visual mesh
        if sim_mesh_prim is not None:
            tet_mesh_prim = UsdGeom.TetMesh(sim_mesh_prim)
            surface_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(tet_mesh_prim, Usd.TimeCode.Default())
            if surface_indices is None or len(surface_indices) == 0:
                raise ValueError(
                    f"Deformable body at '{prim_path}' has no surface indices on its TetMesh prim; "
                    "cannot sync to visual mesh."
                )
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
        mesh_paths = [p.GetPrimPath() for p in matching_prims]
        raise ValueError(
            f"Found multiple visual meshes in '{prim_path}': {mesh_paths}."
            " Deformable body schema can only be applied to one mesh for now."
        )
    vis_mesh_prim = matching_prims[0]
    if not vis_mesh_prim.IsValid():
        raise ValueError(f"Mesh prim path '{vis_mesh_prim.GetPrimPath()}' is not valid.")

    # the cfg's USD namespace selects between the OmniPhysics (PhysX) and Newton deformable APIs
    use_omni_physics_apis = getattr(cfg, "_usd_namespace", None) != "newton"

    if sim_mesh_prim_path is None:
        sim_mesh_prim_path = prim_path + "/sim_mesh"
    vertices = np.array(vis_mesh_prim.GetAttribute("points").Get())
    faces = np.array(vis_mesh_prim.GetAttribute("faceVertexIndices").Get()).flatten()
    face_counts = np.array(vis_mesh_prim.GetAttribute("faceVertexCounts").Get())
    if deformable_type == "surface":
        # the simulation mesh is a copy of the visual mesh
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
        if use_omni_physics_apis:
            if not sim_mesh_prim.ApplyAPI("OmniPhysicsSurfaceDeformableSimAPI"):
                raise RuntimeError(f"Failed to set surface deformable body API on prim '{sim_mesh_prim_path}'.")
            # set rest-shape attributes required by OmniPhysicsSurfaceDeformableSimAPI
            sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(vertices)
            sim_mesh_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(faces)
        else:
            if not sim_mesh_prim.AddAppliedSchema("PhysicsSurfaceDeformableSimAPI"):
                raise RuntimeError(f"Failed to set surface deformable body API on prim '{sim_mesh_prim_path}'.")

    elif deformable_type == "volume":
        if sim_mesh_prim is None:
            tet_mesh_points, tet_mesh_indices = _tetrahedralize_surface(
                vertices, faces, tetrahedralization_edge_length_fac, prim_path
            )
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
        if use_omni_physics_apis:
            if not sim_mesh_prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI"):
                raise RuntimeError(f"Failed to set volume deformable body API on prim '{sim_mesh_prim_path}'.")
        else:
            if not sim_mesh_prim.AddAppliedSchema("PhysicsVolumeDeformableSimAPI"):
                raise RuntimeError(f"Failed to set volume deformable body API on prim '{sim_mesh_prim_path}'.")

        # set surface faces and rest-shape attributes required by OmniPhysicsVolumeDeformableSimAPI
        surface_face_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(
            UsdGeom.TetMesh(sim_mesh_prim), Usd.TimeCode.Default()
        )
        UsdGeom.TetMesh(sim_mesh_prim).GetSurfaceFaceVertexIndicesAttr().Set(surface_face_indices)
        if use_omni_physics_apis:
            sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(sim_mesh_prim.GetAttribute("points").Get())
            sim_mesh_prim.GetAttribute("omniphysics:restTetVtxIndices").Set(
                sim_mesh_prim.GetAttribute("tetVertexIndices").Get()
            )

    else:
        raise ValueError(
            f"Unsupported deformable type: '{deformable_type}'. Only surface and volume deformables are supported."
        )

    if not sim_mesh_prim.ApplyAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Failed to set {deformable_type} deformable collision API on prim '{sim_mesh_prim_path}'.")
    # the simulation mesh is not rendered
    UsdGeom.Imageable(sim_mesh_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    if use_omni_physics_apis:
        # PhysX binds the visual mesh to the simulation mesh through the bind-pose deformable pose API
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
        if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
            raise RuntimeError(f"Failed to set deformable body API on prim '{prim_path}'.")
    else:
        # TODO: Temporary solution for Newton: Overwrite visual mesh with tet mesh surface points or copy
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
        if not root_prim.AddAppliedSchema("PhysicsDeformableBodyAPI"):
            raise RuntimeError(f"Failed to set deformable body API on prim '{prim_path}'.")

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
    if stage is None:
        stage = get_current_stage()
    deformable_body_prim = stage.GetPrimAtPath(prim_path)
    if not deformable_body_prim.IsValid() or not has_deformable_body_api(deformable_body_prim):
        return False
    cfg_dict = _cfg_fields(cfg)
    if cfg_dict.get("kinematic_enabled"):
        logger.warning(
            "Kinematic deformable bodies are not fully supported in the current version of Omni Physics. "
            "Setting kinematic_enabled to True may lead to unexpected behavior."
        )
    _apply_namespaced_schemas(deformable_body_prim, cfg, cfg_dict)
    return True
