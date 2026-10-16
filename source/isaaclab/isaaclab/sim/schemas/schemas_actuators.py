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
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import torch
from torch._export.converter import TS2EPConverter

from pxr import Sdf, Usd, UsdPhysics

from ...actuators.actuator_bam_cfg import BAM_DRIVE_API, BAM_MODEL_FLAGS, BamActuatorCfg
from ...actuators.actuator_base import ActuatorBase
from ...actuators.actuator_base_cfg import ActuatorBaseCfg, _is_implicit_actuator_cfg
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


def _is_newton_native_actuator_cfg(cfg: ActuatorBaseCfg) -> bool:
    """Return whether an actuator config can be authored as a Newton actuator."""
    layout = _actuator_layout(cfg)
    if layout is None:
        return False
    if layout.actuator_class is None:
        return cfg.class_type is None
    try:
        return _resolve_actuator_class(cfg.class_type) is layout.actuator_class
    except ValueError:
        return False


def validate_newton_native_actuator_cfgs(actuator_cfgs: dict[str, ActuatorBaseCfg]) -> None:
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


def define_actuator_properties(
    prim_path: str,
    actuator_cfgs: dict[str, ActuatorBaseCfg],
    stage: Usd.Stage | None = None,
) -> None:
    """Author ``NewtonActuator`` USD prims under an articulation root.

    Every joint covered by an explicit (non-implicit) Lab actuator config ends up with one
    active ``NewtonActuator`` prim. A config is authored onto the joint's USD actuator when
    its drive matches, so values the config leaves unset keep the asset's value; otherwise
    the USD actuators are replaced by a new prim. See :ref:`USD actuator overrides <actuators-usd-override>` for the
    rules and the schemas each config type authors. Joints **not** covered by any Lab
    config keep their USD-authored actuators unchanged.

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
            execution is enabled and an explicit actuator config is unsupported, or if an actuator
            component the config adds lacks a required value.
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
    stage: Usd.Stage,
    articulation_prim_path: str,
    actuator_cfgs: dict[str, ActuatorBaseCfg],
) -> None:
    """Author the ``NewtonActuator`` prims of one articulation.

    Args:
        stage: USD stage to author on.
        articulation_prim_path: Root prim path of one articulation.
        actuator_cfgs: Mapping of group name to actuator config.

    Raises:
        ValueError: If the articulation prim is missing, a config is unsupported, a mapping pattern matches no
            joint, or an actuator component the config adds lacks a required value.
    """
    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not art_prim.IsValid():
        raise ValueError(f"Articulation prim not found: {articulation_prim_path}")

    validate_newton_native_actuator_cfgs(actuator_cfgs)

    joint_inventory = _collect_joint_prims(art_prim)
    all_joint_names = list(joint_inventory.keys())
    actuator_prims = _collect_actuator_prims(art_prim)

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
        joints = {name: stage.GetPrimAtPath(joint_inventory[name]) for name in joint_names}
        _author_group(stage, articulation_prim_path, group_name, resolved_cfg, joints, actuator_prims)


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

_FLOAT = Sdf.ValueTypeNames.Float
_INT = Sdf.ValueTypeNames.Int


@dataclass(frozen=True)
class _ActuatorField:
    """A ``NewtonActuator`` attribute that an actuator config can author."""

    name: str
    """Config field that sets the attribute, as a dotted path; also named in errors."""

    schema: str
    """API schema that owns the attribute."""

    attr: str
    """USD attribute name."""

    type_name: Sdf.ValueTypeName = _FLOAT
    """USD value type of the attribute."""

    source: Callable[[ActuatorBaseCfg, list[str]], dict[str, Any]] | None = None
    """Values the config sets, keyed by joint name; joints left out keep the USD value. Defaults to :attr:`name`."""

    new_prim_default: float | Callable[[ActuatorBaseCfg, Usd.Prim, bool], float | None] | None = None
    """Value of a component the config adds when the config leaves it unset, or a callable of the config, the joint
    prim and whether the actuator prim is new."""

    required: bool = False
    """Whether a component the config adds needs the value; Newton validates the components a USD prim keeps."""

    disabled_by: int | None = None
    """Configured value that removes :attr:`schema` instead of authoring the attribute."""

    def configured(self, cfg: ActuatorBaseCfg, joint_names: list[str]) -> dict[str, Any]:
        """Return the values *cfg* sets, keyed by joint name."""
        if self.source is not None:
            return self.source(cfg, joint_names)
        return _configured_values(attrgetter(self.name)(cfg), joint_names, int if self.type_name == _INT else float)


@dataclass(frozen=True)
class _ActuatorLayout:
    """The ``NewtonActuator`` components and attributes of one actuator config type."""

    actuator_class: type[ActuatorBase] | None
    """Isaac Lab actuator class the config must name, or None when the config names none."""

    drive: str
    """Drive schema; an existing prim is authored onto only when it has this drive."""

    fields: tuple[_ActuatorField, ...]
    """Attributes the config type can author; their schemas and the drive are all it keeps."""

    joint_attrs: tuple[tuple[str, Sdf.ValueTypeName, Any], ...] = ()
    """Attributes authored on each driven joint prim, as ``(name, type, value)``."""


def _author_group(
    stage: Usd.Stage,
    articulation_prim_path: str,
    group_name: str,
    cfg: ActuatorBaseCfg,
    joints: dict[str, Usd.Prim],
    actuator_prims: dict[str, list[Usd.Prim]],
) -> None:
    """Author the actuators of one group, raising before a joint's edits if a required value is missing.

    A value the config sets is authored. An unset value keeps the prim's opinion when the prim keeps
    that value's schema; otherwise it takes the field's new-prim default.
    """
    layout = _actuator_layout(cfg)
    configured = {field.attr: field.configured(cfg, list(joints)) for field in layout.fields}
    allowed = {layout.drive, *(field.schema for field in layout.fields)}
    for joint_name, joint_prim in joints.items():
        existing = actuator_prims.get(str(joint_prim.GetPath()), [])
        prim, reason = _override_target(existing, layout.drive)
        present = _applied_schemas(prim) if prim is not None else set()
        disabled = {
            field.schema
            for field in layout.fields
            if field.disabled_by is not None and configured[field.attr].get(joint_name) == field.disabled_by
        }
        dropped = sorted((present & _COMPONENT_APIS) - allowed)
        remove = sorted({*dropped, *(present & disabled)})
        values = []
        for field in layout.fields:
            value = configured[field.attr].get(joint_name)
            if field.schema in disabled or (value is None and field.schema in present):
                # A component the prim keeps keeps its USD values.
                continue
            if value is None:
                default = field.new_prim_default
                value = default(cfg, joint_prim, prim is None) if callable(default) else default
            if value is None:
                if field.required:
                    raise ValueError(
                        f"Actuator group '{group_name}' does not set '{field.name}' for joint '{joint_name}', and the"
                        f" joint has no USD actuator with {field.schema} to provide it."
                    )
                continue
            values.append((field, value))
        if reason is not None:
            logger.warning(
                f"Actuator group '{group_name}' replaces the USD actuators of joint '{joint_name}': {reason}. Values"
                " the configuration leaves unset use code defaults instead of the asset's."
            )
        if dropped:
            logger.warning(
                f"Actuator group '{group_name}' removes {dropped} from the USD actuator of joint '{joint_name}':"
                f" {type(cfg).__name__} does not model them."
            )
        if prim is None:
            # Define the new prim before deactivating the old ones, so a failure never leaves the joint without one.
            prim_path = find_unique_string_name(
                f"{articulation_prim_path}/{group_name}_{joint_name}_actuator",
                lambda path: not stage.GetPrimAtPath(path).IsValid(),
            )
            prim = stage.DefinePrim(prim_path, "NewtonActuator")
            prim.CreateRelationship("newton:targets").SetTargets([joint_prim.GetPath()])
            for old_prim in existing:
                old_prim.SetActive(False)
        for schema in remove:
            prim.RemoveAppliedSchema(schema)
        for schema in dict.fromkeys([layout.drive, *(f.schema for f, _ in values)]):
            if schema not in present:
                prim.AddAppliedSchema(schema)
        for field, value in values:
            prim.CreateAttribute(field.attr, field.type_name).Set(value)
        for name, type_name, value in layout.joint_attrs:
            joint_prim.CreateAttribute(name, type_name).Set(value)


def _override_target(prims: list[Usd.Prim], drive: str) -> tuple[Usd.Prim | None, str | None]:
    """Choose the USD actuator prim a config with *drive* is authored onto.

    Returns:
        The prim, or None with the reason it must be replaced (None when the joint has no actuator).
    """
    if not prims:
        return None, None
    if len(prims) > 1:
        return None, f"{len(prims)} actuator prims drive it"
    schemas = _applied_schemas(prims[0])
    drives = schemas & _DRIVE_APIS
    if drives != {drive}:
        return None, f"their drive {sorted(drives)} is not {drive}"
    unknown = schemas - _KNOWN_ACTUATOR_APIS
    if unknown:
        return None, f"they apply schemas {sorted(unknown)} the configuration cannot keep"
    return prims[0], None


def _applied_schemas(prim: Usd.Prim) -> set[str]:
    """Return the composed ``apiSchemas`` tokens of *prim*, including unregistered ones such as BAM's drive."""
    list_op = prim.GetMetadata("apiSchemas")
    return set(list_op.GetAddedOrExplicitItems()) if list_op is not None else set()


def _get_authored_joint_effort_limit(joint_prim: Usd.Prim) -> float | None:
    """Read a revolute or prismatic joint's authored USD drive effort limit."""
    drive_name = drive_instance_name(joint_prim)
    if drive_name is None:
        return None
    value = UsdPhysics.DriveAPI(joint_prim, drive_name).GetMaxForceAttr().Get()
    return None if value is None else float(value)


def _collect_joint_prims(art_prim: Usd.Prim) -> dict[str, str]:
    """Collect all joint prims under an articulation subtree.

    Returns:
        Ordered mapping of joint name to full prim path.
    """
    return {
        prim.GetName(): str(prim.GetPath()) for prim in Usd.PrimRange(art_prim) if prim.GetTypeName() in _JOINT_TYPES
    }


def _collect_actuator_prims(art_prim: Usd.Prim) -> dict[str, list[Usd.Prim]]:
    """Map joint paths to the active ``NewtonActuator`` prims that drive them.

    Newton drives only the first ``newton:targets`` target of an actuator prim, so later
    targets are ignored. Inactive prims and prims inside instances are skipped, as
    ``ModelBuilder.add_usd`` skips them. Only prims under the *art_prim* subtree are considered.
    """
    actuator_prims: dict[str, list[Usd.Prim]] = {}
    for prim in Usd.PrimRange(art_prim):
        rel = prim.GetRelationship("newton:targets") if prim.GetTypeName() == "NewtonActuator" else None
        targets = rel.GetTargets() if rel else []
        if targets:
            actuator_prims.setdefault(str(targets[0]), []).append(prim)
    return actuator_prims


def _configured_values(value: Any, joint_names: list[str], cast: type = float) -> dict[str, Any]:
    """Return the values a scalar or regex-keyed mapping sets, keyed by the joints it matches.

    Raises:
        ValueError: If a mapping pattern matches no joint, or a joint matches several patterns.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        _, names, values = resolve_matching_names_values(value, joint_names)
        return {name: cast(v) for name, v in zip(names, values, strict=True)}
    return dict.fromkeys(joint_names, cast(value))


def _new_prim_effort_limit(cfg: ActuatorBaseCfg, joint_prim: Usd.Prim, new_prim: bool) -> float | None:
    """Return the limit of an effort clamp the config adds for a joint it does not set.

    On a new prim, a configured mapping resolves unmatched joints to zero, like the Isaac Lab actuator models.
    Otherwise the joint's authored USD drive limit applies.
    """
    if new_prim and cfg.actuator_effort_limit is not None:
        return 0.0
    return _get_authored_joint_effort_limit(joint_prim)


def _exported_model_path(cfg: ActuatorNetMLPCfg | ActuatorNetLSTMCfg, joint_names: list[str]) -> dict[str, Any]:
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


def _bam_value(
    name: str, type_name: Sdf.ValueTypeName = _FLOAT, default: float | None = None, required: bool = False
) -> _ActuatorField:
    """Return the BAM drive attribute set by a :class:`BamActuatorCfg` field, named after its DriveBam argument."""
    attr = f"newton:{to_camel_case(name.removeprefix('motor.'))}"
    return _ActuatorField(name, BAM_DRIVE_API, attr, type_name, new_prim_default=default, required=required)


def _bam_model_flag(flag: str) -> _ActuatorField:
    """Return a BAM drive flag set by the friction model of the :class:`BamMotorCfg`."""

    def source(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, int]:
        model = cfg.motor.model
        return _configured_values(BAM_MODEL_FLAGS[model][flag] if model is not None else None, joint_names, int)

    return _ActuatorField("motor.model", BAM_DRIVE_API, f"newton:{to_camel_case(flag)}", _INT, source, required=True)


def _bam_effort_limits(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, float]:
    """Return the configured BAM effort limit, defaulting to the stall torque at the highest sampled voltage."""
    if cfg.actuator_effort_limit is not None:
        return _configured_values(cfg.actuator_effort_limit, joint_names)
    voltage = max(cfg.vin_range) if cfg.vin_range is not None else cfg.vin
    if cfg.motor.kt is None or cfg.motor.resistance is None or voltage is None:
        if any(v is not None for v in (voltage, cfg.motor.kt, cfg.motor.resistance)):
            logger.warning(
                f"The BAM configuration of joints {joint_names} changes only part of the stall torque inputs, so their"
                " USD actuators keep their effort limit. Set actuator_effort_limit to change it."
            )
        return {}
    return dict.fromkeys(joint_names, voltage * cfg.motor.kt / cfg.motor.resistance)


_PD_GAINS = (
    _ActuatorField("stiffness", _PD_API, "newton:kp", new_prim_default=0.0),
    _ActuatorField("damping", _PD_API, "newton:kd", new_prim_default=0.0),
)
_MAX_EFFORT = _ActuatorField(
    "actuator_effort_limit", _MAX_EFFORT_API, "newton:maxEffort", new_prim_default=_new_prim_effort_limit
)
_DC_MOTOR = (
    _ActuatorField("saturation_effort", _DC_MOTOR_API, "newton:saturationEffort", required=True),
    _ActuatorField("actuator_velocity_limit", _DC_MOTOR_API, "newton:velocityLimit"),
    _ActuatorField(
        "actuator_effort_limit", _DC_MOTOR_API, "newton:maxMotorEffort", new_prim_default=_new_prim_effort_limit
    ),
)
_DELAY = _ActuatorField("max_delay", _DELAY_API, "newton:delaySteps", _INT, disabled_by=0)
_POSITION_BASED = tuple(
    _ActuatorField(
        "joint_parameter_lookup",
        _POSITION_BASED_API,
        attr,
        Sdf.ValueTypeNames.FloatArray,
        lambda cfg, names, column=column: dict.fromkeys(names, [float(r[column]) for r in cfg.joint_parameter_lookup]),
    )
    for attr, column in (("newton:lookupPositions", 0), ("newton:lookupEfforts", 2))
)
_NEURAL_MODEL = _ActuatorField(
    "network_file", _NEURAL_API, "newton:modelPath", Sdf.ValueTypeNames.Asset, _exported_model_path
)

# BAM attributes are float-typed even when integral: an int-typed attribute would truncate a float override.
# A new prim requires the fit and supply: the drive would silently fall back to m1 friction and no current limit.
_BAM_FIELDS = (
    *(_bam_value(f"motor.{name}", required=True) for name in ("kt", "resistance", "error_gain")),
    _bam_value("motor.max_pwm", default=1.0),
    *(_bam_value(f"motor.{name}", required=True) for name in ("max_current", "friction_base")),
    *(_bam_value(f"motor.{name}", default=1.0) for name in ("dtheta_stribeck", "alpha")),
    *(
        _bam_value(f"motor.{name}", default=0.0)
        for name in (
            "friction_stribeck",
            "load_friction_motor",
            "load_friction_external",
            "load_friction_motor_stribeck",
            "load_friction_external_stribeck",
            "load_friction_motor_quad",
            "load_friction_external_quad",
        )
    ),
    *(_bam_model_flag(flag) for flag in ("stribeck", "load_dependent", "quadratic")),
    *(_bam_value(name, required=True) for name in ("kp_fw", "vin")),
    *(_bam_value(name) for name in ("vin_min", "delay_hold_prob")),
    *(_bam_value(name, _INT) for name in ("min_delay", "max_delay", "delay_update_period")),
    # Flag backlash drives so they never merge with regular ones; without a hinge template, keep the USD flag.
    _ActuatorField(
        "backlash_joint_template",
        BAM_DRIVE_API,
        "newton:hasBacklash",
        _INT,
        lambda cfg, names: dict.fromkeys(names, 1) if cfg.backlash_joint_template is not None else {},
    ),
    # A new prim requires kt, resistance and vin, so the default only fills joints a configured mapping skips.
    _ActuatorField(
        "actuator_effort_limit", BAM_DRIVE_API, "newton:maxEffort", _FLOAT, _bam_effort_limits, _new_prim_effort_limit
    ),
)

_ACTUATOR_LAYOUTS: dict[type, _ActuatorLayout] = {
    IdealPDActuatorCfg: _ActuatorLayout(IdealPDActuator, _PD_API, (*_PD_GAINS, _MAX_EFFORT)),
    DelayedPDActuatorCfg: _ActuatorLayout(DelayedPDActuator, _PD_API, (*_PD_GAINS, _MAX_EFFORT, _DELAY)),
    RemotizedPDActuatorCfg: _ActuatorLayout(RemotizedPDActuator, _PD_API, (*_PD_GAINS, *_POSITION_BASED, _DELAY)),
    DCMotorCfg: _ActuatorLayout(DCMotor, _PD_API, (*_PD_GAINS, *_DC_MOTOR)),
    ActuatorNetMLPCfg: _ActuatorLayout(ActuatorNetMLP, _NEURAL_API, (_NEURAL_MODEL, *_DC_MOTOR)),
    ActuatorNetLSTMCfg: _ActuatorLayout(ActuatorNetLSTM, _NEURAL_API, (_NEURAL_MODEL, *_DC_MOTOR)),
    # Allocate the joint friction constraint; BAM overwrites this seed before each solve.
    BamActuatorCfg: _ActuatorLayout(None, BAM_DRIVE_API, _BAM_FIELDS, joint_attrs=(("newton:friction", _FLOAT, 1.0),)),
}
"""Newton actuator layout of each supported config type; subclasses use their closest base's layout."""


def _actuator_layout(cfg: ActuatorBaseCfg) -> _ActuatorLayout | None:
    """Return the Newton actuator layout of *cfg*'s type, or None if Newton cannot author it."""
    return next((_ACTUATOR_LAYOUTS[cls] for cls in type(cfg).__mro__ if cls in _ACTUATOR_LAYOUTS), None)


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
