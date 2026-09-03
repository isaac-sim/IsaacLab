# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX schema-fragment appliers and compatibility wrappers.

The deformable schema writers are backend-aware but remain unified in
:mod:`isaaclab.sim.schemas`. This module additionally hosts the PhysX-specific fragment
applier funcs that override :attr:`~isaaclab.sim.schemas.SchemaFragment.func` for the
joint-drive and multi-instance tendon schemas, keeping the backend func out of the core package.
"""

from __future__ import annotations

import dataclasses
import math

from pxr import Usd, UsdPhysics

from isaaclab.sim.schemas.schemas import (
    define_deformable_body_properties,
    modify_deformable_body_properties,
)
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from .schemas_cfg import PhysxFixedTendonAxisCfg, PhysxFixedTendonCfg, PhysxSpatialTendonCfg

__all__ = [
    "apply_fixed_tendon",
    "apply_fixed_tendon_axis",
    "apply_physx_joint",
    "apply_spatial_tendon",
    "define_deformable_body_properties",
    "modify_deformable_body_properties",
]


def apply_physx_joint(cfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Apply a :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` fragment to a joint prim.

    Like :func:`~isaaclab.sim.schemas.apply_namespaced`, this applies ``PhysxJointAPI`` and writes
    each non-``None`` field under the ``physxJoint:`` namespace. It additionally reproduces the
    legacy joint-drive unit convention: for angular (revolute) joints, ``max_joint_velocity`` is
    converted from rad/s to deg/s, since PhysX stores angular joint velocity limits in degrees.

    Args:
        cfg: The :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` fragment.
        prim_path: The prim path of the joint.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the properties were successfully set.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    namespace = type(cfg)._usd_namespace
    applied = type(cfg)._usd_applied_schema
    if applied and applied not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(applied)
    # angular (revolute) joints store velocity limits in degrees; linear (prismatic) in meters.
    is_angular = prim.IsA(UsdPhysics.RevoluteJoint)
    for f in dataclasses.fields(cfg):
        if f.name == "func":
            continue
        value = getattr(cfg, f.name)
        if value is None:
            continue
        if f.name == "max_joint_velocity" and is_angular:
            value = value * 180.0 / math.pi  # rad/s -> deg/s
        safe_set_attribute_on_usd_prim(prim, f"{namespace}:{to_camel_case(f.name, 'cC')}", value, camel_case=False)
    return True


def _selected_tendon_instances(prim: Usd.Prim, schema_type: str, selection) -> list[str]:
    """Return existing instances of ``schema_type`` selected by a tendon fragment."""
    if selection is None:
        requested = None
    elif isinstance(selection, str):
        if not selection:
            raise ValueError("'instance_names' must contain at least one non-empty instance name.")
        requested = {selection}
    elif isinstance(selection, list):
        if not selection or any(not isinstance(name, str) or not name for name in selection):
            raise ValueError("'instance_names' must contain at least one non-empty instance name.")
        requested = set(selection)
    else:
        raise TypeError("'instance_names' must be a string, list of strings, or None.")

    instances = []
    for applied_schema in prim.GetAppliedSchemas():
        applied_type, instance = Usd.SchemaRegistry.GetTypeNameAndInstance(str(applied_schema))
        if applied_type == schema_type and instance and (requested is None or instance in requested):
            instances.append(instance)
    return instances


def _tune_tendon_schema(cfg, prim_path: str, stage: Usd.Stage | None, schema_type: str) -> bool:
    """Tune selected instances of one concrete PhysX tendon API on one prim."""
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    instances = _selected_tendon_instances(prim, schema_type, cfg.instance_names)
    if not instances:
        return False

    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_type)
    if definition is None:
        raise RuntimeError(f"USD schema '{schema_type}' is not registered.")
    templates = {
        str(Usd.SchemaRegistry.GetMultipleApplyNameTemplateBaseName(str(name))): str(name)
        for name in definition.GetPropertyNames()
        if "__INSTANCE_NAME__" in str(name)
    }
    for instance in instances:
        for field in dataclasses.fields(cfg):
            if field.name in ("func", "instance_names"):
                continue
            value = getattr(cfg, field.name)
            if value is None:
                continue
            property_name = to_camel_case(field.name, "cC")
            template = templates.get(property_name)
            if template is None:
                raise TypeError(f"'{field.name}' is not a property of USD schema '{schema_type}'.")
            attr_name = Usd.SchemaRegistry.MakeMultipleApplyNameInstance(template, instance)
            attr = prim.GetAttribute(attr_name)
            if not attr:
                raise RuntimeError(f"USD schema '{schema_type}' did not expose attribute '{attr_name}'.")
            if not attr.Set(value):
                raise ValueError(f"Failed to set '{attr_name}' on prim '{prim_path}'.")
    return True


def apply_fixed_tendon(cfg: PhysxFixedTendonCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Tune selected ``PhysxTendonAxisRootAPI`` instances on a prim.

    The source asset owns the fixed-tendon topology, so this function applies no schema. It writes
    only whole-tendon root properties to the instances selected by :attr:`PhysxFixedTendonCfg.instance_names`.

    Args:
        cfg: The :class:`PhysxFixedTendonCfg` fragment to apply.
        prim_path: The prim path carrying the fixed-tendon schemas.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if at least one selected root instance exists, otherwise False.
    """
    return _tune_tendon_schema(cfg, prim_path, stage, "PhysxTendonAxisRootAPI")


def apply_fixed_tendon_axis(cfg: PhysxFixedTendonAxisCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Tune selected ``PhysxTendonAxisAPI`` instances on a joint prim.

    The source asset owns the fixed-tendon topology, so this function applies no schema. A root API
    includes the axis API for the same instance; consequently this writer works for both root joints
    and AxisAPI-only child joints.

    Args:
        cfg: Per-axis properties and the existing instances to select.
        prim_path: The joint prim carrying the tendon-axis instances.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if at least one selected axis instance exists, otherwise False.
    """
    return _tune_tendon_schema(cfg, prim_path, stage, "PhysxTendonAxisAPI")


def apply_spatial_tendon(cfg: PhysxSpatialTendonCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Tune selected ``PhysxTendonAttachmentRootAPI`` instances on a prim.

    The fragment contains whole-tendon dynamics declared by the attachment root. Attachment and
    leaf properties describe topology and are not modified by this function.

    Args:
        cfg: The :class:`PhysxSpatialTendonCfg` fragment to apply.
        prim_path: The prim path carrying the spatial-tendon schemas.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if at least one selected attachment-root instance exists, otherwise False.
    """
    return _tune_tendon_schema(cfg, prim_path, stage, "PhysxTendonAttachmentRootAPI")
