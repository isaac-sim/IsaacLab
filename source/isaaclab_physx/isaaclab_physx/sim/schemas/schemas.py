# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX schema-fragment appliers and compatibility wrappers.

The deformable schema writers are backend-aware but remain unified in
:mod:`isaaclab.sim.schemas`. This module additionally hosts PhysX-specific fragment
implementation used by joint-drive and multi-instance tendon configs, keeping backend behavior
out of the core package.
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

__all__ = [
    "apply_physx_joint",
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


def _tune_tendon_schema(cfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Tune selected instances of one concrete PhysX tendon API on one prim."""
    schema_type = type(cfg)._usd_applied_schema
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)

    selected = [cfg.instance_names] if isinstance(cfg.instance_names, str) else cfg.instance_names
    if selected is not None and (
        not selected or not all(isinstance(instance, str) and instance for instance in selected)
    ):
        raise ValueError("'instance_names' must contain at least one non-empty instance name.")
    instances = []
    for applied_schema in prim.GetAppliedSchemas():
        applied_type, instance = Usd.SchemaRegistry.GetTypeNameAndInstance(str(applied_schema))
        if applied_type == schema_type and instance and (selected is None or instance in selected):
            instances.append(instance)
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
            value = getattr(cfg, field.name)
            if field.name in ("func", "instance_names") or value is None:
                continue
            template = templates.get(to_camel_case(field.name, "cC"))
            if template is None:
                raise TypeError(f"'{field.name}' is not a property of USD schema '{schema_type}'.")
            attr_name = Usd.SchemaRegistry.MakeMultipleApplyNameInstance(template, instance)
            if not prim.GetAttribute(attr_name).Set(value):
                raise ValueError(f"Failed to set '{attr_name}' on prim '{prim_path}'.")
    return True
