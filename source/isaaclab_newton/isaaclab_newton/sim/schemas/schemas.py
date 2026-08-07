# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific schema-fragment appliers.

Hosts the custom ``func`` overrides for Newton/MuJoCo fragments whose application needs backend
logic beyond the generic :func:`~isaaclab.sim.schemas.apply_namespaced` writer. Keeping these here
(rather than in the core spawner) means ``isaaclab`` never imports a backend.
"""

from __future__ import annotations

import dataclasses

from pxr import Sdf, Usd, UsdPhysics, Vt

from isaaclab.sim.schemas.schemas import apply_namespaced
from isaaclab.sim.utils import change_prim_property, safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from .schemas_cfg import MujocoCollisionCfg, MujocoFixedTendonCfg, MujocoJointCfg

__all__ = ["apply_mujoco_collision", "apply_mujoco_fixed_tendon", "apply_mujoco_joint"]


def apply_mujoco_collision(cfg: MujocoCollisionCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Write MuJoCo contact attributes on a collision prim.

    Custom ``func`` override for :class:`~isaaclab_newton.sim.schemas.MujocoCollisionCfg`.
    Scalar fields are written as regular ``mjc:*`` attributes. The fixed-length :attr:`solimp`
    and :attr:`solref` fields are written as canonical USD ``double[]`` arrays.

    Args:
        cfg: The :class:`MujocoCollisionCfg` fragment to apply.
        prim_path: The collision prim path to author on.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if all authored attributes were written successfully.

    Raises:
        ValueError: If the prim does not exist or a field has an invalid value or array length.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    if cfg.condim is not None and cfg.condim not in (1, 3, 4, 6):
        raise ValueError(f"'condim' must be one of 1, 3, 4, or 6. Received: {cfg.condim}.")
    if cfg.group is not None and not 0 <= cfg.group <= 5:
        raise ValueError(f"'group' must be between 0 and 5. Received: {cfg.group}.")
    if cfg.priority is not None and cfg.priority < 0:
        raise ValueError(f"'priority' must be non-negative. Received: {cfg.priority}.")
    if cfg.solmix is not None and cfg.solmix < 0.0:
        raise ValueError(f"'solmix' must be non-negative. Received: {cfg.solmix}.")

    success = True
    for attr_name in ("condim", "group", "priority", "solmix"):
        value = getattr(cfg, attr_name)
        if value is not None:
            safe_set_attribute_on_usd_prim(prim, f"mjc:{attr_name}", value, camel_case=False)

    for attr_name, expected_length in (("solimp", 5), ("solref", 2)):
        value = getattr(cfg, attr_name)
        if value is None:
            continue
        if len(value) != expected_length:
            raise ValueError(f"'{attr_name}' must contain exactly {expected_length} values. Received: {len(value)}.")
        success = (
            change_prim_property(
                prop_path=f"{prim.GetPath()}.mjc:{attr_name}",
                value=Vt.DoubleArray(value),
                stage=stage,
                type_to_create_if_not_exist=Sdf.ValueTypeNames.DoubleArray,
            )
            and success
        )
    return success


def apply_mujoco_fixed_tendon(cfg: MujocoFixedTendonCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Write ``mjc:*`` fixed-tendon attributes on a ``MjcTendon`` prim.

    Custom ``func`` override for :class:`~isaaclab_newton.sim.schemas.MujocoFixedTendonCfg`.
    No-op (returns False) on any prim whose type is not ``MjcTendon``.

    Args:
        cfg: The :class:`MujocoFixedTendonCfg` fragment to apply.
        prim_path: The prim path of the ``MjcTendon`` prim.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if the prim is a ``MjcTendon`` and was tuned, False otherwise.

    Raises:
        ValueError: If the prim at ``prim_path`` does not exist in the stage.
    """
    if stage is None:
        stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    values = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if f.name != "func" and getattr(cfg, f.name) is not None
    }
    # Descend the whole subtree (matching legacy apply_nested): ``MjcTendon`` prims may sit below
    # the prim_path the spawner targets. Write ``mjc:*`` on every ``MjcTendon`` descendant.
    found = False
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "MjcTendon":
            continue
        found = True
        for attr_name, value in values.items():
            safe_set_attribute_on_usd_prim(prim, f"mjc:{to_camel_case(attr_name, 'cC')}", value, camel_case=False)
    return found


def apply_mujoco_joint(cfg: MujocoJointCfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Apply a :class:`MujocoJointCfg` fragment, including its body-level gravcomp coupling.

    Custom ``func`` override for :class:`MujocoJointCfg`. Writes the joint's ``mjc:*`` attributes via
    :func:`~isaaclab.sim.schemas.apply_namespaced`, then enforces the MuJoCo coupling that joint-level
    ``actuatorgravcomp`` requires body-level ``gravcomp``: in MuJoCo ``actuatorgravcomp`` is a per-joint
    flag (``jnt_actgravcomp``) that routes the gravity-compensation force of the joint's actuated body
    through the actuator, and it is inert unless that body's ``mjc:gravcomp`` is non-zero. So when
    :attr:`~MujocoJointCfg.actuatorgravcomp` is requested, this enables ``mjc:gravcomp = 1.0`` on the
    joint's child body (its ``physics:body1`` target) when the body has not authored it. An explicitly
    authored body gravcomp is preserved. Keeping this coupling in the Newton applier (not the core
    spawner) keeps the core package free of any backend dependency.

    Args:
        cfg: The :class:`MujocoJointCfg` fragment to apply.
        prim_path: The joint prim path to author on.
        stage: The stage where to find the prim. Defaults to the current stage.

    Returns:
        True if the joint fragment was applied successfully.
    """
    if stage is None:
        stage = get_current_stage()
    success = apply_namespaced(cfg, prim_path, stage)
    # actuatorgravcomp is inert unless the actuated body has non-zero gravcomp; flip it on the joint's
    # child body when requested and unset (per-joint dispatch covers every actuated body in the
    # articulation; the non-actuated base has no parent joint and needs no compensation).
    if cfg.actuatorgravcomp:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(prim_path))
        targets = joint.GetBody1Rel().GetTargets() if joint else []
        for body_path in targets:
            body = stage.GetPrimAtPath(body_path)
            if not body.IsValid():
                continue
            current = body.GetAttribute("mjc:gravcomp").Get()
            if current is None or current == 0.0:
                safe_set_attribute_on_usd_prim(body, "mjc:gravcomp", 1.0, camel_case=False)
    return success
