# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD schema authoring for Newton-native actuators.

:func:`define_actuator_properties` translates IsaacLab actuator configs
into ``NewtonActuator`` USD prims. Both the Newton ``ModelBuilder.add_usd``
path and the PhysX adapter's
:meth:`~isaaclab.actuators.newton.adapter.NewtonActuatorAdapter.from_usd`
read the same authored prims, ensuring both backends construct
:class:`~newton.actuators.Actuator` instances with matching parameters.

This module lives on the schema side so that authoring is a regular
``define_*_properties`` step in the spawner pipeline, alongside
:func:`define_articulation_root_properties` and friends, rather than a
side effect of asset construction.
"""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch._export.converter import TS2EPConverter

from pxr import Sdf, Usd, UsdPhysics

from ...actuators.actuator_bam_cfg import BAM_DRIVE_API, BAM_MODEL_FLAGS, BamActuatorCfg
from ...actuators.actuator_base import ActuatorBase
from ...actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from ...actuators.actuator_compat import resolve_limit_aliases
from ...actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from ...actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from ...actuators.actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator, RemotizedPDActuator
from ...actuators.actuator_pd_cfg import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, RemotizedPDActuatorCfg
from ...utils import clone, validate
from ...utils.assets import retrieve_file_path
from ...utils.backend_utils import FactoryBase
from ...utils.string import (
    find_unique_string_name,
    resolve_matching_names,
    resolve_matching_names_values,
    string_to_callable,
    to_camel_case,
)
from ..simulation_context import SimulationContext
from ..utils.queries import find_matching_prims
from ..utils.stage import get_current_stage
from .schemas import drive_instance_name

logger = logging.getLogger(__name__)


def _resolve_actuator_class(class_type: type | str) -> type:
    """Resolve and validate an actuator class reference for authoring identity checks."""
    if isinstance(class_type, str):
        try:
            class_type = string_to_callable(str(class_type))
        except (AttributeError, ImportError, ValueError) as error:
            raise ValueError(f"Unable to resolve actuator class '{class_type}'.") from error
    if not isinstance(class_type, type) or not issubclass(class_type, ActuatorBase):
        raise ValueError(f"Actuator class must derive from ActuatorBase, got {class_type!r}.")
    return class_type


def _is_newton_native_actuator_cfg(cfg: Any) -> bool:
    """Return whether an actuator config can be authored as a Newton actuator."""
    if isinstance(cfg, BamActuatorCfg):
        return cfg.class_type is None

    supported_cfg_types = (
        (ActuatorNetMLPCfg, ActuatorNetMLP),
        (ActuatorNetLSTMCfg, ActuatorNetLSTM),
        (RemotizedPDActuatorCfg, RemotizedPDActuator),
        (DelayedPDActuatorCfg, DelayedPDActuator),
        (DCMotorCfg, DCMotor),
        (IdealPDActuatorCfg, IdealPDActuator),
    )
    try:
        resolved_class = _resolve_actuator_class(cfg.class_type)
    except ValueError:
        return False
    for cfg_type, actuator_type in supported_cfg_types:
        if isinstance(cfg, cfg_type):
            return resolved_class is actuator_type
    return False


def validate_newton_native_actuator_cfgs(actuator_cfgs: dict[str, Any]) -> None:
    """Reject explicit actuator configurations that Newton cannot author."""
    unsupported_groups = []
    for group_name, cfg in actuator_cfgs.items():
        try:
            is_implicit = _is_implicit_actuator_cfg(cfg)
        except ValueError:
            is_implicit = False
        if not is_implicit and not _is_newton_native_actuator_cfg(cfg):
            unsupported_groups.append(f"'{group_name}' ({type(cfg).__name__})")
    if unsupported_groups:
        raise ValueError(
            "Newton-native actuator execution does not support "
            f"{', '.join(unsupported_groups)}. Disable 'use_newton_actuators' or use a supported actuator config."
        )


def resolve_per_dof(
    value: dict[str, float | int] | float | int | None,
    joint_names: list[str],
    cast: type = float,
) -> dict[str, float | int]:
    """Expand a scalar or regex-keyed dict cfg value into a per-joint mapping.

    Used by :func:`define_actuator_properties` to flatten the various
    accepted forms of a per-DOF config field (``stiffness``, ``damping``,
    ``effort_limit``, …) into a single ``{joint_name: value}`` dict. Joints
    the value does not cover are left out, so authoring can tell a value the
    config sets from one it leaves to the USD asset.

    Accepted forms of *value*:

    * ``None`` → empty dict.
    * scalar (``int`` / ``float``) → broadcast to every joint name.
    * dict → keys are treated as regex patterns and matched against
      *joint_names* via :func:`re.fullmatch`. The first matching pattern
      wins per joint name.
    """
    if value is None:
        return {}
    if isinstance(value, (int, float)):
        return {name: cast(value) for name in joint_names}
    if isinstance(value, dict):
        result: dict[str, float | int] = {}
        for name in joint_names:
            for pattern, v in value.items():
                if re.fullmatch(pattern, name):
                    result[name] = cast(v)
                    break
        return result
    return {}


def define_actuator_properties(
    prim_path: str,
    actuator_cfgs: dict[str, Any],
    stage: Any | None = None,
) -> None:
    """Author ``NewtonActuator`` USD prims under an articulation root.

    Every joint covered by an explicit (non-implicit) Lab actuator config ends
    up with one active ``NewtonActuator`` prim:

    * If the joint has exactly one USD-authored actuator prim whose drive schema
      matches the config's, the config is authored onto that prim in the stage's
      edit target. Only the values the config sets are written; a value it leaves
      unset (``None``, or a joint a regex mapping does not match) keeps the asset's
      opinion through USD composition. The config type still decides which
      components the actuator has: clamping and delay schemas it does not model
      are removed from the prim, and setting ``max_delay`` to 0 removes a delay.
    * Otherwise (no actuator prim, several, or another drive) the existing prims
      are deactivated and a new prim is defined from the config, with code
      defaults for unset values (e.g. zero gains).

    Joints **not** covered by any Lab config keep their USD-authored actuators
    unchanged.

    The supported config-to-schema mapping is:

    * :class:`~isaaclab.actuators.IdealPDActuatorCfg` →
      ``NewtonPDControlAPI`` + ``NewtonMaxEffortClampingAPI``
    * :class:`~isaaclab.actuators.DCMotorCfg` →
      ``NewtonPDControlAPI`` + ``NewtonDCMotorClampingAPI``
    * :class:`~isaaclab.actuators.DelayedPDActuatorCfg` →
      same as ``IdealPDActuatorCfg`` + ``NewtonActuatorDelayAPI``
    * :class:`~isaaclab.actuators.RemotizedPDActuatorCfg` →
      ``NewtonPDControlAPI`` + ``NewtonPositionBasedClampingAPI`` + ``NewtonActuatorDelayAPI``
    * :class:`~isaaclab.actuators.ActuatorNetMLPCfg` /
      :class:`~isaaclab.actuators.ActuatorNetLSTMCfg` →
      ``NewtonNeuralControlAPI`` + ``NewtonDCMotorClampingAPI``
    * :class:`~isaaclab.actuators.BamActuatorCfg` →
      ``NewtonBamDriveAPI`` (effort limit as a drive attribute, no clamping schema)

    No-ops (returns immediately) when:

    * the active :class:`~isaaclab.sim.SimulationContext` was configured
      with ``use_newton_actuators=False`` and no BAM groups (or no context is active), or
    * *prim_path* does not resolve to a valid prim on the stage.

    Must be called **after** the articulation is spawned (joint prims
    exist on stage) and **before** the cloner / ``ModelBuilder.add_usd``
    reads the stage.

    Args:
        prim_path: Root prim path of the articulation (e.g.
            ``"/World/Env_0/Robot"``). May contain a regex pattern; every matching prim is
            authored, because a multi-asset spawner creates a distinct articulation per match.
        actuator_cfgs: Mapping of group name to
            :class:`~isaaclab.actuators.ActuatorBaseCfg`.
        stage: USD stage to author on. When ``None``, the current stage
            is used.

    Raises:
        ValueError: If BAM is configured without native Newton execution, if Newton-native
            execution is enabled and an explicit actuator config is unsupported, or if a required
            value is set neither by the config nor by the USD actuator it is authored onto.
    """
    sim_ctx = SimulationContext.instance()
    sim_cfg = sim_ctx.cfg if sim_ctx is not None else None
    if sim_cfg is not None and any(isinstance(cfg, BamActuatorCfg) for cfg in actuator_cfgs.values()):
        if not sim_cfg.use_newton_actuators or FactoryBase._get_backend() != "newton":
            raise ValueError("BAM requires use_newton_actuators=True with the Newton backend and MJWarp solver.")
    if sim_cfg is None or not sim_cfg.use_newton_actuators:
        return

    if stage is None:
        stage = get_current_stage()

    # Each match is a separately spawned articulation. A multi-asset spawner creates one per asset
    # before cloning, and the Newton builder only sees actuators authored on the prim it imports.
    for prim in find_matching_prims(prim_path, stage):
        author_actuator_prims(stage, str(prim.GetPath()), actuator_cfgs)


def author_actuator_prims(
    stage: Any,
    articulation_prim_path: str,
    actuator_cfgs: dict[str, Any],
) -> None:
    """Inner authoring routine; exposed separately for test fixtures.

    Every config is resolved and checked before the stage is edited, so a rejected
    config leaves the stage unchanged.
    """
    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not art_prim.IsValid():
        raise ValueError(f"Articulation prim not found: {articulation_prim_path}")

    validate_newton_native_actuator_cfgs(actuator_cfgs)

    joint_inventory = _collect_joint_prims(art_prim)
    all_joint_names = list(joint_inventory.keys())
    actuator_prims = _collect_actuator_prims(art_prim)

    plans: list[_ActuatorPlan] = []
    for group_name, cfg in actuator_cfgs.items():
        if _is_implicit_actuator_cfg(cfg):
            continue

        _ids, joint_names = resolve_matching_names(cfg.joint_names_expr, all_joint_names)
        if not joint_names:
            continue

        resolved_cfg = clone(cfg)
        # Collection construction emits the deprecation warning later in the
        # normal asset lifecycle. Authoring only needs the normalized value.
        resolve_limit_aliases(group_name, resolved_cfg, joint_names, warn_deprecated=False)
        if isinstance(resolved_cfg, BamActuatorCfg):
            validate(resolved_cfg)
        plans.extend(
            _plan_group(
                stage, articulation_prim_path, group_name, resolved_cfg, joint_inventory, joint_names, actuator_prims
            )
        )

    for plan in plans:
        _apply_plan(stage, plan)


"""
Private helpers.
"""

_JOINT_TYPES = frozenset({"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"})

_PD_API = "NewtonPDControlAPI"
_PID_API = "NewtonPIDControlAPI"
_NEURAL_API = "NewtonNeuralControlAPI"
_MAX_EFFORT_API = "NewtonMaxEffortClampingAPI"
_DC_MOTOR_API = "NewtonDCMotorClampingAPI"
_POSITION_BASED_API = "NewtonPositionBasedClampingAPI"
_DELAY_API = "NewtonActuatorDelayAPI"

_DRIVE_APIS = frozenset({_PD_API, _PID_API, _NEURAL_API, BAM_DRIVE_API})
"""Drive schemas; Newton rejects an actuator prim with more than one."""

_COMPONENT_APIS = _DRIVE_APIS | {_MAX_EFFORT_API, _DC_MOTOR_API, _POSITION_BASED_API, _DELAY_API}
"""Actuator component schemas whose presence an actuator config decides."""

_KNOWN_ACTUATOR_APIS = _COMPONENT_APIS | {"NewtonActuatorControlBaseAPI", "NewtonActuatorClampingBaseAPI"}
"""Every schema token a config can be authored next to; base APIs are built into the component schemas."""


@dataclass(frozen=True)
class _ActuatorField:
    """A ``NewtonActuator`` attribute that an actuator config can author."""

    schema: str
    """API schema that owns the attribute."""

    attr: str
    """USD attribute name."""

    type_name: Sdf.ValueTypeName
    """USD value type of the attribute."""

    source: str | Callable[[Any, list[str]], dict[str, Any]]
    """Config attribute holding a scalar or a regex-keyed mapping, or a callable returning the
    values the config sets keyed by joint name. Joints left out keep the USD value."""

    cast: type = float
    """Value type of a config attribute :attr:`source`."""

    fresh_default: Any = None
    """Value for a joint the config leaves unset when a new prim is defined. A callable receives
    the config and the joint prim. None authors nothing, leaving Newton's default."""

    required: bool = False
    """Whether the config or the USD prim it is authored onto must provide the value."""

    off_value: Any = None
    """Configured value that removes :attr:`schema` instead of authoring the attribute."""

    def configured(self, cfg: Any, joint_names: list[str]) -> dict[str, Any]:
        """Return the values *cfg* sets, keyed by joint name."""
        if callable(self.source):
            return self.source(cfg, joint_names)
        return resolve_per_dof(getattr(cfg, self.source), joint_names, cast=self.cast)

    def fresh_value(self, cfg: Any, joint_prim: Usd.Prim) -> Any:
        """Return the value a new prim gets when *cfg* leaves this attribute unset."""
        return self.fresh_default(cfg, joint_prim) if callable(self.fresh_default) else self.fresh_default


@dataclass(frozen=True)
class _ActuatorLayout:
    """The ``NewtonActuator`` components and attributes of one actuator config type."""

    drive: str
    """Drive schema; an existing prim is authored onto only when it has this drive."""

    fields: tuple[_ActuatorField, ...]
    """Attributes the config type can author. Their schemas, with :attr:`drive`, are the only
    component schemas an actuator of this type keeps."""

    joint_attrs: tuple[tuple[str, Sdf.ValueTypeName, Any], ...] = ()
    """Attributes authored on each driven joint prim, as ``(name, type, value)``."""

    @property
    def schemas(self) -> frozenset[str]:
        """Component schemas an actuator of this type may carry."""
        return frozenset({self.drive, *(field.schema for field in self.fields)})


@dataclass
class _ActuatorPlan:
    """Stage edits that give one joint its configured actuator."""

    joint_path: str
    prim_path: str
    """Existing actuator prim to author onto, or the path of a new prim."""
    define: bool
    """Whether a new prim targeting :attr:`joint_path` is defined at :attr:`prim_path`."""
    layout: _ActuatorLayout
    values: list[tuple[_ActuatorField, Any]]
    add_schemas: list[str]
    remove_schemas: list[str]
    deactivate: list[Usd.Prim]


def _plan_group(
    stage: Usd.Stage,
    articulation_prim_path: str,
    group_name: str,
    cfg: Any,
    joint_inventory: dict[str, str],
    joint_names: list[str],
    actuator_prims: dict[str, list[Usd.Prim]],
) -> list[_ActuatorPlan]:
    """Plan the stage edits of one actuator group, raising before any edit if a value is missing."""
    layout = _actuator_layout(cfg)
    configured = {field.attr: field.configured(cfg, joint_names) for field in layout.fields}
    plans: list[_ActuatorPlan] = []
    replaced: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}
    for joint_name in joint_names:
        joint_path = joint_inventory[joint_name]
        joint_values = {
            field.attr: configured[field.attr][joint_name]
            for field in layout.fields
            if joint_name in configured[field.attr]
        }
        disabled = {
            field.schema
            for field in layout.fields
            if field.off_value is not None and joint_values.get(field.attr) == field.off_value
        }
        existing = actuator_prims.get(joint_path, [])
        target, reason = _overlay_target(existing, layout.drive)
        if target is not None:
            plan = _plan_overlay(group_name, joint_name, joint_path, target, layout, joint_values, disabled)
            for schema in plan.remove_schemas:
                removed.setdefault(schema, []).append(joint_name)
        else:
            if reason is not None:
                replaced.setdefault(reason, []).append(joint_name)
            prim_path = find_unique_string_name(
                f"{articulation_prim_path}/{group_name}_{joint_name}_actuator",
                lambda path: not stage.GetPrimAtPath(path).IsValid(),
            )
            plan = _plan_new_prim(
                group_name, joint_name, cfg, stage.GetPrimAtPath(joint_path), prim_path, layout, joint_values, disabled
            )
            plan.deactivate = existing
        plans.append(plan)

    for reason, names in replaced.items():
        logger.warning(
            f"Actuator group '{group_name}' replaces the USD actuators of joints {names}: {reason}. Values the"
            " configuration leaves unset use code defaults instead of the asset's."
        )
    for schema, names in removed.items():
        logger.warning(
            f"Actuator group '{group_name}' removes {schema} from the USD actuators of joints {names}, because"
            f" {type(cfg).__name__} does not model it."
        )
    return plans


def _plan_overlay(
    group_name: str,
    joint_name: str,
    joint_path: str,
    prim: Usd.Prim,
    layout: _ActuatorLayout,
    joint_values: dict[str, Any],
    disabled: set[str],
) -> _ActuatorPlan:
    """Plan authoring the configured values onto a USD actuator prim with a matching drive."""
    present = _applied_schemas(prim)
    remove = sorted(((present & _COMPONENT_APIS) - layout.schemas) | (present & disabled))
    values = [
        (field, joint_values[field.attr])
        for field in layout.fields
        if field.attr in joint_values and field.schema not in disabled
    ]
    kept = present - set(remove)
    for field in layout.fields:
        if field.required and field.attr not in joint_values and not _has_authored_value(prim, field, kept):
            raise ValueError(
                f"Actuator group '{group_name}' does not set '{field.source}' for joint '{joint_name}', and its USD"
                f" actuator '{prim.GetPath()}' does not author '{field.attr}'."
            )
    add = list(dict.fromkeys(field.schema for field, _ in values if field.schema not in present))
    return _ActuatorPlan(
        joint_path=joint_path,
        prim_path=str(prim.GetPath()),
        define=False,
        layout=layout,
        values=values,
        add_schemas=add,
        remove_schemas=remove,
        deactivate=[],
    )


def _plan_new_prim(
    group_name: str,
    joint_name: str,
    cfg: Any,
    joint_prim: Usd.Prim,
    prim_path: str,
    layout: _ActuatorLayout,
    joint_values: dict[str, Any],
    disabled: set[str],
) -> _ActuatorPlan:
    """Plan a new actuator prim that holds every configured value and code defaults for the rest."""
    values = []
    for field in layout.fields:
        if field.schema in disabled:
            continue
        value = joint_values[field.attr] if field.attr in joint_values else field.fresh_value(cfg, joint_prim)
        if value is None:
            if field.required:
                raise ValueError(
                    f"Actuator group '{group_name}' does not set '{field.source}' for joint '{joint_name}', which"
                    " has no USD actuator to take it from."
                )
            continue
        values.append((field, value))
    schemas = list(dict.fromkeys([layout.drive, *(field.schema for field, _ in values)]))
    return _ActuatorPlan(
        joint_path=str(joint_prim.GetPath()),
        prim_path=prim_path,
        define=True,
        layout=layout,
        values=values,
        add_schemas=schemas,
        remove_schemas=[],
        deactivate=[],
    )


def _apply_plan(stage: Usd.Stage, plan: _ActuatorPlan) -> None:
    """Apply one planned actuator to the stage's edit target."""
    for prim in plan.deactivate:
        prim.SetActive(False)
    if plan.define:
        act_prim = stage.DefinePrim(plan.prim_path, "NewtonActuator")
        act_prim.CreateRelationship("newton:targets").SetTargets([Sdf.Path(plan.joint_path)])
    else:
        act_prim = stage.GetPrimAtPath(plan.prim_path)
    for schema in plan.remove_schemas:
        act_prim.RemoveAppliedSchema(schema)
    for schema in plan.add_schemas:
        act_prim.AddAppliedSchema(schema)
    for field, value in plan.values:
        act_prim.CreateAttribute(field.attr, field.type_name).Set(value)
    joint_prim = stage.GetPrimAtPath(plan.joint_path)
    for name, type_name, value in plan.layout.joint_attrs:
        joint_prim.CreateAttribute(name, type_name).Set(value)


def _overlay_target(prims: list[Usd.Prim], drive: str) -> tuple[Usd.Prim | None, str | None]:
    """Choose the USD actuator prim a config with *drive* is authored onto.

    Returns:
        The prim, or None with the reason it must be replaced (None when the joint has no actuator).
    """
    if not prims:
        return None, None
    if len(prims) > 1:
        return None, f"{len(prims)} actuator prims drive each joint"
    schemas = _applied_schemas(prims[0])
    drives = schemas & _DRIVE_APIS
    if drives != {drive}:
        return None, f"their drive {sorted(drives)} is not {drive}"
    unknown = schemas - _KNOWN_ACTUATOR_APIS
    if unknown:
        return None, f"they apply schemas {sorted(unknown)} the configuration cannot keep"
    return prims[0], None


def _applied_schemas(prim: Usd.Prim) -> set[str]:
    """Return the composed API schema tokens of *prim*.

    Reads the composed ``apiSchemas`` list as well, since an unregistered schema such as BAM's
    drive is missing from :meth:`Usd.Prim.GetAppliedSchemas`.
    """
    schemas = set(prim.GetAppliedSchemas())
    list_op = prim.GetMetadata("apiSchemas")
    if list_op is not None:
        schemas.update(list_op.GetAddedOrExplicitItems())
    return schemas


def _has_authored_value(prim: Usd.Prim, field: _ActuatorField, schemas: set[str]) -> bool:
    """Return whether *prim* authors *field* under a schema it keeps."""
    attr = prim.GetAttribute(field.attr)
    return field.schema in schemas and attr.IsValid() and attr.HasAuthoredValue()


def _actuator_target(prim: Usd.Prim) -> str:
    """Return the joint path Newton drives with *prim*: the first ``newton:targets`` target."""
    rel = prim.GetRelationship("newton:targets")
    targets = rel.GetTargets() if rel else []
    return str(targets[0]) if targets else ""


def _get_authored_joint_effort_limit(joint_prim: Usd.Prim) -> float | None:
    """Read a revolute or prismatic joint's authored USD drive effort limit."""
    drive_name = drive_instance_name(joint_prim)
    if drive_name is None:
        return None
    value = UsdPhysics.DriveAPI(joint_prim, drive_name).GetMaxForceAttr().Get()
    return None if value is None else float(value)


def _collect_joint_prims(art_prim: Any) -> dict[str, str]:
    """Collect all joint prims under an articulation subtree.

    Returns:
        Ordered mapping of joint name to full prim path.
    """
    return {
        prim.GetName(): str(prim.GetPath()) for prim in Usd.PrimRange(art_prim) if prim.GetTypeName() in _JOINT_TYPES
    }


def _collect_actuator_prims(art_prim: Any) -> dict[str, list[Usd.Prim]]:
    """Map joint paths to the active ``NewtonActuator`` prims that drive them.

    Newton drives only the first target of an actuator prim, so later targets are
    ignored. Inactive prims and prims inside instances are skipped, as
    ``ModelBuilder.add_usd`` skips them. Only prims under the *art_prim* subtree
    are considered.
    """
    actuator_prims: dict[str, list[Usd.Prim]] = {}
    for prim in Usd.PrimRange(art_prim):
        if prim.GetTypeName() == "NewtonActuator" and (target := _actuator_target(prim)):
            actuator_prims.setdefault(target, []).append(prim)
    return actuator_prims


def _configured_effort_limits(cfg: Any, joint_names: list[str]) -> dict[str, float]:
    """Return the actuator effort limits *cfg* sets: a scalar for every joint, or the joints a mapping matches."""
    limit = cfg.actuator_effort_limit
    if limit is None:
        return {}
    if isinstance(limit, (int, float)):
        return dict.fromkeys(joint_names, float(limit))
    _, names, values = resolve_matching_names_values(limit, joint_names)
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _fresh_effort_limit(cfg: Any, joint_prim: Usd.Prim) -> float | None:
    """Return a new prim's effort limit for a joint *cfg* does not set.

    A configured mapping resolves unmatched joints to zero, like the Isaac Lab actuator models.
    Without a configured limit, the joint's authored USD drive limit applies.
    """
    if cfg.actuator_effort_limit is not None:
        return 0.0
    return _get_authored_joint_effort_limit(joint_prim)


def _lookup_column(column: int) -> Callable[[Any, list[str]], dict[str, list[float]]]:
    """Return a source of one column of a remotized actuator's joint parameter lookup table."""

    def source(cfg: Any, joint_names: list[str]) -> dict[str, list[float]]:
        return dict.fromkeys(joint_names, [float(row[column]) for row in cfg.joint_parameter_lookup])

    return source


def _exported_model_path(cfg: Any, joint_names: list[str]) -> dict[str, Sdf.AssetPath]:
    """Export a neural actuator's checkpoint with its config metadata and return its path for every joint."""
    meta: dict[str, Any] = {}
    if isinstance(cfg, ActuatorNetMLPCfg):
        meta["model_type"] = "mlp"
        meta["input_order"] = cfg.input_order
        meta["input_idx"] = list(cfg.input_idx)
        meta["pos_scale"] = cfg.pos_scale
        meta["vel_scale"] = cfg.vel_scale
        meta["torque_scale"] = cfg.torque_scale
    else:
        meta["model_type"] = "lstm"
    return dict.fromkeys(joint_names, Sdf.AssetPath(resave_checkpoint_with_metadata(cfg.network_file, meta)))


def _bam_value(name: str, type_name: Sdf.ValueTypeName, cast: type, fresh_default: Any = None) -> _ActuatorField:
    """Return a BAM drive attribute set from the :class:`BamActuatorCfg` attribute *name*."""
    return _ActuatorField(
        BAM_DRIVE_API, f"newton:{to_camel_case(name)}", type_name, name, cast=cast, fresh_default=fresh_default
    )


def _bam_motor_value(name: str, fresh_default: float | None = None) -> _ActuatorField:
    """Return a BAM drive attribute set from the :class:`BamMotorCfg` attribute *name*."""

    def source(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, float]:
        return resolve_per_dof(getattr(cfg.motor, name) if cfg.motor is not None else None, joint_names)

    return _ActuatorField(
        BAM_DRIVE_API, f"newton:{to_camel_case(name)}", Sdf.ValueTypeNames.Float, source, fresh_default=fresh_default
    )


def _bam_model_flag(flag: str) -> _ActuatorField:
    """Return a BAM drive flag set by the friction model of the :class:`BamMotorCfg`."""

    def source(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, int]:
        model = cfg.motor.model if cfg.motor is not None else None
        return resolve_per_dof(BAM_MODEL_FLAGS[model][flag] if model is not None else None, joint_names, cast=int)

    return _ActuatorField(BAM_DRIVE_API, f"newton:{to_camel_case(flag)}", Sdf.ValueTypeNames.Int, source)


def _bam_backlash_flag(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, int]:
    """Flag backlash drives so they never merge with regular ones."""
    return dict.fromkeys(joint_names, int(cfg.backlash_joint_template is not None))


def _bam_effort_limits(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, float]:
    """Return the configured BAM effort limit, defaulting to the stall torque at the highest sampled voltage."""
    if cfg.actuator_effort_limit is not None:
        return _configured_effort_limits(cfg, joint_names)
    motor = cfg.motor
    voltage = max(cfg.vin_range) if cfg.vin_range is not None else cfg.vin
    if motor is None or motor.kt is None or motor.resistance is None or voltage is None:
        return {}
    return dict.fromkeys(joint_names, voltage * motor.kt / motor.resistance)


_FLOAT = Sdf.ValueTypeNames.Float
_INT = Sdf.ValueTypeNames.Int

_PD_GAINS = (
    _ActuatorField(_PD_API, "newton:kp", _FLOAT, "stiffness", fresh_default=0.0),
    _ActuatorField(_PD_API, "newton:kd", _FLOAT, "damping", fresh_default=0.0),
)
_MAX_EFFORT = _ActuatorField(
    _MAX_EFFORT_API, "newton:maxEffort", _FLOAT, _configured_effort_limits, fresh_default=_fresh_effort_limit
)
_DC_MOTOR = (
    _ActuatorField(_DC_MOTOR_API, "newton:saturationEffort", _FLOAT, "saturation_effort", required=True),
    _ActuatorField(_DC_MOTOR_API, "newton:velocityLimit", _FLOAT, "actuator_velocity_limit"),
    _ActuatorField(
        _DC_MOTOR_API, "newton:maxMotorEffort", _FLOAT, _configured_effort_limits, fresh_default=_fresh_effort_limit
    ),
)
_DELAY = _ActuatorField(_DELAY_API, "newton:delaySteps", _INT, "max_delay", cast=int, off_value=0)
_POSITION_BASED = (
    _ActuatorField(_POSITION_BASED_API, "newton:lookupPositions", Sdf.ValueTypeNames.FloatArray, _lookup_column(0)),
    _ActuatorField(_POSITION_BASED_API, "newton:lookupEfforts", Sdf.ValueTypeNames.FloatArray, _lookup_column(2)),
)
_NEURAL_MODEL = _ActuatorField(_NEURAL_API, "newton:modelPath", Sdf.ValueTypeNames.Asset, _exported_model_path)

# BAM attributes are float-typed even when integral: an int-typed attribute would truncate a float override.
_BAM_FIELDS = (
    _bam_motor_value("kt"),
    _bam_motor_value("resistance"),
    _bam_motor_value("error_gain"),
    _bam_motor_value("max_pwm", fresh_default=1.0),
    _bam_motor_value("max_current"),
    _bam_motor_value("friction_base"),
    _bam_motor_value("friction_stribeck", fresh_default=0.0),
    _bam_motor_value("dtheta_stribeck", fresh_default=1.0),
    _bam_motor_value("alpha", fresh_default=1.0),
    _bam_motor_value("load_friction_motor", fresh_default=0.0),
    _bam_motor_value("load_friction_external", fresh_default=0.0),
    _bam_motor_value("load_friction_motor_stribeck", fresh_default=0.0),
    _bam_motor_value("load_friction_external_stribeck", fresh_default=0.0),
    _bam_motor_value("load_friction_motor_quad", fresh_default=0.0),
    _bam_motor_value("load_friction_external_quad", fresh_default=0.0),
    _bam_model_flag("stribeck"),
    _bam_model_flag("load_dependent"),
    _bam_model_flag("quadratic"),
    _bam_value("kp_fw", _FLOAT, float),
    _bam_value("vin", _FLOAT, float),
    _bam_value("vin_min", _FLOAT, float, fresh_default=-math.inf),
    _bam_value("delay_hold_prob", _FLOAT, float, fresh_default=0.0),
    _bam_value("min_delay", _INT, int, fresh_default=0),
    _bam_value("max_delay", _INT, int, fresh_default=0),
    _bam_value("delay_update_period", _INT, int, fresh_default=0),
    _ActuatorField(BAM_DRIVE_API, "newton:hasBacklash", _INT, _bam_backlash_flag),
    _ActuatorField(BAM_DRIVE_API, "newton:maxEffort", _FLOAT, _bam_effort_limits),
)

_ACTUATOR_LAYOUTS: dict[type, _ActuatorLayout] = {
    IdealPDActuatorCfg: _ActuatorLayout(_PD_API, (*_PD_GAINS, _MAX_EFFORT)),
    DelayedPDActuatorCfg: _ActuatorLayout(_PD_API, (*_PD_GAINS, _MAX_EFFORT, _DELAY)),
    RemotizedPDActuatorCfg: _ActuatorLayout(_PD_API, (*_PD_GAINS, *_POSITION_BASED, _DELAY)),
    DCMotorCfg: _ActuatorLayout(_PD_API, (*_PD_GAINS, *_DC_MOTOR)),
    ActuatorNetMLPCfg: _ActuatorLayout(_NEURAL_API, (_NEURAL_MODEL, *_DC_MOTOR)),
    ActuatorNetLSTMCfg: _ActuatorLayout(_NEURAL_API, (_NEURAL_MODEL, *_DC_MOTOR)),
    # Allocate the joint friction constraint; BAM overwrites this seed before each solve.
    BamActuatorCfg: _ActuatorLayout(BAM_DRIVE_API, _BAM_FIELDS, joint_attrs=(("newton:friction", _FLOAT, 1.0),)),
}
"""Newton actuator layout of each supported config type; subclasses use their closest base's layout."""


def _actuator_layout(cfg: Any) -> _ActuatorLayout:
    """Return the Newton actuator layout of *cfg*'s type."""
    return next(_ACTUATOR_LAYOUTS[cls] for cls in type(cfg).__mro__ if cls in _ACTUATOR_LAYOUTS)


def resave_checkpoint_with_metadata(original_path: str, metadata: dict[str, Any]) -> str:
    """Re-save a neural-network checkpoint with updated metadata.

    Resolves the configured path through the shared asset cache, loads the
    original TorchScript checkpoint, merges *metadata* into any existing
    metadata (Lab config values take precedence), and exports Newton's ``.pt2``
    format with a dynamic batch dimension. The temporary file persists for the
    lifetime of the process.

    Args:
        original_path: Path or URL of a TorchScript actuator checkpoint.
        metadata: Overrides for the checkpoint's metadata. The merged metadata must include
            ``model_type`` (``"mlp"`` or ``"lstm"``) and, for MLP models, ``input_idx``.

    Returns:
        Path to the temporary ``.pt2`` checkpoint.
    """
    local_path = retrieve_file_path(original_path)

    extra_files: dict[str, str] = {"metadata.json": ""}
    try:
        net = torch.jit.load(local_path, map_location="cpu", _extra_files=extra_files)
        existing_meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint at '{original_path}'; expected a TorchScript archive") from exc

    merged = {**existing_meta, **metadata}
    batch = torch.export.Dim("batch")
    if merged["model_type"] == "lstm":
        weights = net.lstm.state_dict()
        merged["num_layers"] = len(weights) // 4
        merged["hidden_size"] = weights["weight_hh_l0"].shape[1]
        state_shape = (merged["num_layers"], 3, merged["hidden_size"])
        inputs = (torch.zeros(3, 1, 2), (torch.zeros(state_shape), torch.zeros(state_shape)))
        dynamic_shapes = ({0: batch}, ({1: batch}, {1: batch}))
    else:
        inputs = (torch.zeros(3, 2 * len(merged["input_idx"])),)
        dynamic_shapes = ({0: batch},)

    converter = TS2EPConverter(net.eval(), inputs)
    # Fold TorchScript's rank checks before converting LSTM control flow.
    torch._C._jit_pass_peephole(converter.ts_graph, False)
    torch._C._jit_pass_constant_propagation(converter.ts_graph)
    program = converter.convert()
    graph = program.module(check_guards=False)
    # Lifted LSTM weights must move with the module when Newton selects its device.
    for name, tensor in program.constants.items():
        if isinstance(tensor, torch.Tensor):
            delattr(graph, name)
            graph.register_buffer(name, tensor)
    # The converter specializes LSTM batches; export the graph with an explicit batch contract.
    program = torch.export.export(graph, inputs, dynamic_shapes=dynamic_shapes)

    with tempfile.NamedTemporaryFile(suffix=".pt2", delete=False) as tmp:
        tmp_path = tmp.name
    extra_out = {"metadata.json": json.dumps(merged)}
    torch.export.save(program, tmp_path, extra_files=extra_out)

    return tmp_path
