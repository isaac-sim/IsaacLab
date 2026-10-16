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
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import torch

from pxr import Sdf, Usd, UsdPhysics, Vt

from ...actuators.actuator_bam_cfg import BAM_STIFF_SOLIMP_FRICTION, BAM_STIFF_SOLREF_FRICTION, BamActuatorCfg
from ...actuators.actuator_base import ActuatorBase
from ...actuators.actuator_base_cfg import ActuatorBaseCfg, _is_implicit_actuator_cfg
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


def resolve_per_dof(
    value: dict[str, float | int] | float | int | None,
    joint_names: list[str],
    cast: type = float,
) -> dict[str, float | int]:
    """Expand scalar or regex-keyed actuator values into a sparse joint-name mapping.

    .. deprecated:: 3.1
        Use :func:`~isaaclab.utils.string.resolve_matching_names_values` for regex mappings.
        Handle None as an empty mapping and broadcast scalars directly.

    Args:
        value: Scalar, regex-keyed mapping, or None for no configured values.
        joint_names: Joint names against which patterns are resolved.
        cast: Conversion applied to each configured value. Defaults to float.

    Returns:
        Configured values keyed by matching joint name; None returns an empty mapping.

    Raises:
        ValueError: If a mapping pattern matches no joint, or a joint matches several patterns.
    """
    warnings.warn(
        "resolve_per_dof is deprecated. Use isaaclab.utils.string.resolve_matching_names_values for regex"
        " mappings, broadcast scalar values directly, and use an empty mapping for None.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _configured_values(value, joint_names, cast)


def define_actuator_properties(
    prim_path: str,
    actuator_cfgs: dict[str, ActuatorBaseCfg],
    stage: Usd.Stage | None = None,
) -> None:
    """Author ``NewtonActuator`` USD prims under an articulation root.

    For every joint covered by an explicit (non-implicit) Lab actuator
    config, any existing ``NewtonActuator`` prim targeting that joint is
    replaced by a new one created from the config values. Joints **not**
    covered by any Lab config keep their USD-authored actuators unchanged.

    The supported config-to-schema mapping is:

    * :class:`~isaaclab.actuators.IdealPDActuatorCfg` →
      ``NewtonPDControlAPI`` + ``NewtonMaxEffortClampingAPI``
    * :class:`~isaaclab.actuators.DCMotorCfg` →
      ``NewtonPDControlAPI`` + ``NewtonDCMotorClampingAPI``
    * :class:`~isaaclab.actuators.DelayedPDActuatorCfg` →
      same as ``IdealPDActuatorCfg`` + ``NewtonActuatorDelayAPI``
    * :class:`~isaaclab.actuators.RemotizedPDActuatorCfg` →
      same as ``DelayedPDActuatorCfg`` + ``NewtonPositionBasedClampingAPI``
    * :class:`~isaaclab.actuators.ActuatorNetMLPCfg` /
      :class:`~isaaclab.actuators.ActuatorNetLSTMCfg` →
      ``NewtonNeuralControlAPI`` (+ ``NewtonDCMotorClampingAPI``)
    * :class:`~isaaclab.actuators.BamActuatorCfg` →
      ``NewtonBAMControlAPI`` + ``NewtonMaxEffortClampingAPI`` (+ ``NewtonActuatorDelayAPI``),
      plus a backlash input processor when the servo reads through a play hinge

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
        ValueError: If BAM is configured without native Newton execution, or if Newton-native execution is
            enabled and an explicit actuator config is unsupported or invalid, a mapping pattern matches no
            joint or a joint matches several, or a component the config adds lacks a required value.
        TypeError: If a config that matches joints leaves a field at ``MISSING``.
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
        ValueError: If the articulation prim is missing, a config is unsupported or invalid, a mapping pattern
            matches no joint or a joint matches several, or a component the config adds lacks a required value.
        TypeError: If a config that matches joints leaves a field at ``MISSING``.
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
        cfg_entries.append((group_name, resolved_cfg, joint_names))
        for jname in joint_names:
            covered_joint_paths.add(joint_inventory[jname])

    _remove_actuator_prims_for_joints(art_prim, covered_joint_paths)

    for group_name, cfg, joint_names in cfg_entries:
        stiffness_map = resolve_per_dof(cfg.stiffness, joint_names)
        damping_map = resolve_per_dof(cfg.damping, joint_names)

        is_neural = isinstance(cfg, (ActuatorNetMLPCfg, ActuatorNetLSTMCfg))
        is_remotized = isinstance(cfg, RemotizedPDActuatorCfg)
        is_dc_motor = isinstance(cfg, DCMotorCfg)
        is_delayed = isinstance(cfg, DelayedPDActuatorCfg)
        is_bam = isinstance(cfg, BamActuatorCfg)
        configured_effort_limit = cfg.actuator_effort_limit
        if is_bam:
            validate(cfg)
            # Float-typed in USD: an int-typed attribute on a weaker layer would truncate the override. The
            # gearbox friction, supply sag and command lag are not authored: Isaac Lab runs them from the config.
            bam_attrs = {
                # Newton's duty-cycle gain is the firmware gain times the fit's error gain.
                "kp": float(cfg.kp_fw) * float(cfg.motor.error_gain),
                "kt": float(cfg.motor.kt),
                "resistance": float(cfg.motor.resistance),
                "vin": float(cfg.vin),
                "max_pwm": float(cfg.motor.max_pwm),
                "max_current": float(cfg.motor.max_current),
                # Isaac Lab powers every BAM group from a battery, which writes the voltage the drive reads.
                "supply_from_control": True,
            }
            # Default to the stall torque at the highest sampled voltage.
            voltage = max(cfg.vin_range) if cfg.vin_range is not None else cfg.vin
            configured_effort_limit = (
                voltage * cfg.motor.kt / cfg.motor.resistance
                if configured_effort_limit is None
                else configured_effort_limit
            )

        effort_map: dict[str, float] = {}
        if not is_remotized:
            if configured_effort_limit is None:
                for joint_name in joint_names:
                    authored_effort_limit = _get_authored_joint_effort_limit(stage, joint_inventory[joint_name])
                    if authored_effort_limit is not None:
                        effort_map[joint_name] = authored_effort_limit
            else:
                effort_map = dict(
                    zip(joint_names, _resolve_matching_values_dense(configured_effort_limit, joint_names))
                )

        vel_limit_map = resolve_per_dof(cfg.actuator_velocity_limit, joint_names) if is_dc_motor else {}
        sat_effort_map = resolve_per_dof(cfg.saturation_effort, joint_names) if is_dc_motor else {}

        # BAM sizes the delay buffer for its longest lag; Isaac Lab draws the lag of every step.
        raw_delay = cfg.max_delay if is_delayed or is_bam else 0
        delay_map = resolve_per_dof(raw_delay, joint_names, cast=int) if raw_delay else {}

        patched_model_path: str | None = None
        if is_neural:
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
            patched_model_path = resave_checkpoint_with_metadata(cfg.network_file, meta)

        for jname in joint_names:
            joint_prim_path = joint_inventory[jname]

            schemas: list[str] = []
            attrs: dict[str, bool | float | int] = {}
            array_attrs: dict[str, list[float]] = {}
            relationships: dict[str, str] = {"targets": joint_prim_path}

            if is_neural:
                schemas.append("NewtonNeuralControlAPI")
            elif is_bam:
                schemas.append("NewtonBAMControlAPI")
                attrs.update(bam_attrs)
                joint_prim = stage.GetPrimAtPath(joint_prim_path)
                # Allocate the friction constraint; Isaac Lab overwrites this seed before each solve.
                joint_prim.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(1.0)
                if cfg.stiff_frictionloss:
                    for name, value in (
                        ("mjc:solreffriction", BAM_STIFF_SOLREF_FRICTION),
                        ("mjc:solimpfriction", BAM_STIFF_SOLIMP_FRICTION),
                    ):
                        joint_prim.CreateAttribute(name, Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(value))
            else:
                schemas.append("NewtonPDControlAPI")
                attrs["kp"] = stiffness_map.get(jname, 0.0)
                attrs["kd"] = damping_map.get(jname, 0.0)

            if is_dc_motor:
                schemas.append("NewtonDCMotorClampingAPI")
                attrs["saturation_effort"] = sat_effort_map.get(jname, 0.0)
                if jname in vel_limit_map:
                    attrs["velocity_limit"] = vel_limit_map[jname]
                if jname in effort_map:
                    attrs["max_motor_effort"] = effort_map[jname]
            elif not is_remotized and jname in effort_map:
                schemas.append("NewtonMaxEffortClampingAPI")
                attrs["max_effort"] = effort_map[jname]

            if is_remotized:
                lookup = cfg.joint_parameter_lookup
                schemas.append("NewtonPositionBasedClampingAPI")
                array_attrs["lookup_positions"] = [row[0] for row in lookup]
                array_attrs["lookup_efforts"] = [row[2] for row in lookup]

            delay_steps = delay_map.get(jname, 0)
            if delay_steps > 0:
                schemas.append("NewtonActuatorDelayAPI")
                attrs["delay_steps"] = delay_steps
                if not is_bam:
                    attrs["max_delay"] = delay_steps

            if is_bam and cfg.backlash_joint_template is not None:
                schemas.append("NewtonActuatorBacklashAPI")
                attrs["velocity_feedback"] = bool(cfg.backlash_velocity_feedback)
                relationships["backlash_joint"] = _bam_play_hinge_path(
                    stage, joint_prim_path, cfg.backlash_joint_template.format(jname)
                )

            act_prim_path = f"{articulation_prim_path}/{group_name}_{jname}_actuator"
            act_prim = stage.DefinePrim(act_prim_path, "NewtonActuator")
            # Replacement may reuse the path of an actuator deactivated above.
            act_prim.SetActive(True)
            # Explicit, so a weaker layer cannot add back the replaced actuator's schemas, such as its backlash.
            act_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))
            _set_actuator_properties(act_prim, attrs, array_attrs, relationships, patched_model_path)


"""
Private helpers.
"""


@dataclass(frozen=True)
class ActuatorField:
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

    default: float | Callable[[ActuatorBaseCfg, Usd.Prim, bool], float | None] | None = None
    """Value for a component the config adds and leaves unset, or a callable ``(cfg, joint_prim, new_prim)``."""

    required: bool = False
    """Whether a component the config adds needs the value; Newton validates the components a USD prim keeps."""

    def configured(self, cfg: ActuatorBaseCfg, joint_names: list[str]) -> dict[str, Any]:
        """Return the values *cfg* sets, keyed by joint name."""
        if self.source is not None:
            return self.source(cfg, joint_names)
        return _configured_values(attrgetter(self.name)(cfg), joint_names, int if self.type_name == _INT else float)


def _set_actuator_properties(
    act_prim: Usd.Prim,
    attrs: dict[str, bool | float | int],
    array_attrs: dict[str, list[float]],
    relationships: dict[str, str],
    model_path: str | None,
) -> None:
    """Author an actuator prim's ``newton:`` attributes and joint relationships.

    A reused prim path keeps the replaced actuator's properties, and Newton reads every ``newton:`` attribute for a
    drive schema without a registered definition, such as BAM's. Properties this call does not author are blocked.
    """
    authored = {f"newton:{to_camel_case(name)}" for name in (*attrs, *array_attrs, *relationships)}
    if model_path is not None:
        act_prim.CreateAttribute("newton:modelPath", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(model_path))
        authored.add("newton:modelPath")

    for attr_name, attr_val in attrs.items():
        usd_name = f"newton:{to_camel_case(attr_name)}"
        if isinstance(attr_val, bool):
            act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Bool).Set(attr_val)
        elif isinstance(attr_val, int):
            act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Int).Set(attr_val)
        else:
            act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Float).Set(float(attr_val))

    for attr_name, attr_val in array_attrs.items():
        usd_name = f"newton:{to_camel_case(attr_name)}"
        act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.FloatArray).Set(attr_val)

    for rel_name, target in relationships.items():
        act_prim.CreateRelationship(f"newton:{to_camel_case(rel_name)}").SetTargets([Sdf.Path(target)])

    for prop in act_prim.GetAuthoredPropertiesInNamespace("newton"):
        if prop.GetName() in authored:
            continue
        if isinstance(prop, Usd.Attribute):
            prop.Block()
        else:
            # An explicit empty target list overrides the weaker layers' targets.
            prop.SetTargets([])


def _bam_play_hinge_path(stage: Usd.Stage, servo_path: str, hinge_name: str) -> str:
    """Return the path of a BAM servo's sibling play hinge.

    Raises:
        ValueError: If the servo or its play hinge is missing or not revolute.
    """

    def is_revolute(path: str) -> bool:
        prim = stage.GetPrimAtPath(path)
        return prim.IsValid() and prim.GetTypeName() == "PhysicsRevoluteJoint"

    if not is_revolute(servo_path):
        raise ValueError(f"BAM backlash requires a revolute servo joint: {servo_path}")
    hinge_path = f"{servo_path.rsplit('/', 1)[0]}/{hinge_name}"
    if not is_revolute(hinge_path):
        raise ValueError(f"BAM backlash requires a revolute sibling play hinge: {hinge_path}")
    return hinge_path


def _get_authored_joint_effort_limit(stage: Usd.Stage, joint_prim_path: str) -> float | None:
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
    """Map joint paths to the active ``NewtonActuator`` prims under *art_prim* that drive them."""
    actuator_prims: dict[str, list[Usd.Prim]] = {}
    # Like ModelBuilder.add_usd: skip inactive and instanced prims, and use only the first target.
    for prim in Usd.PrimRange(art_prim):
        rel = prim.GetRelationship("newton:targets") if prim.GetTypeName() == "NewtonActuator" else None
        targets = rel.GetTargets() if rel else []
        if targets:
            actuator_prims.setdefault(str(targets[0]), []).append(prim)
    return actuator_prims


def _configured_values(value: Any, joint_names: list[str], cast: type = float) -> dict[str, Any]:
    """Return the values a scalar or regex-keyed mapping sets, keyed by the joints it matches."""
    if value is None:
        return {}
    if isinstance(value, dict):
        _, names, values = resolve_matching_names_values(value, joint_names)
        return {name: cast(v) for name, v in zip(names, values, strict=True)}
    return dict.fromkeys(joint_names, cast(value))


def _default_effort_limit(cfg: ActuatorBaseCfg, joint_prim: Usd.Prim, new_prim: bool) -> float | None:
    """Return the limit of an effort clamp the config adds for a joint it does not set.

    On a new prim, a configured mapping resolves unmatched joints to zero, like the Isaac Lab actuator models.
    Otherwise the joint's authored USD drive limit applies.
    """
    if new_prim and cfg.actuator_effort_limit is not None:
        return 0.0
    return _get_authored_joint_effort_limit(joint_prim)


def _exported_model_path(cfg: ActuatorNetMLPCfg | ActuatorNetLSTMCfg, joint_names: list[str]) -> dict[str, Any]:
    """Return a configured neural checkpoint for every joint, or no overrides when the checkpoint is unset."""
    if cfg.network_file is None:
        return {}
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
) -> ActuatorField:
    """Return the BAM drive attribute set by a :class:`BamActuatorCfg` field, named after its DriveBam argument."""
    attr = f"newton:{to_camel_case(name.removeprefix('motor.'))}"
    return ActuatorField(name, BAM_DRIVE_API, attr, type_name, default=default, required=required)


def _bam_model_flag(flag: str) -> ActuatorField:
    """Return a BAM drive flag set by the friction model of the :class:`BamMotorCfg`."""

    def source(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, int]:
        model = cfg.motor.model
        return _configured_values(BAM_MODEL_FLAGS[model][flag] if model is not None else None, joint_names, int)

    return ActuatorField("motor.model", BAM_DRIVE_API, f"newton:{to_camel_case(flag)}", _INT, source, required=True)


def _bam_effort_limits(cfg: BamActuatorCfg, joint_names: list[str]) -> dict[str, float]:
    """Return the configured BAM effort limit, defaulting to the stall torque at the highest sampled voltage."""
    if cfg.actuator_effort_limit is not None:
        return _configured_values(cfg.actuator_effort_limit, joint_names)
    voltage = max(cfg.vin_range) if cfg.vin_range is not None else cfg.vin
    if cfg.motor.kt is None or cfg.motor.resistance is None or voltage is None:
        if any(v is not None for v in (voltage, cfg.motor.kt, cfg.motor.resistance)):
            logger.warning(
                f"The BAM configuration of joints {joint_names} changes only part of the stall torque inputs, so their"
                " limit stays the USD actuator's authored one or, without it, the drive's vin * kt / resistance"
                " default, which ignores vin_range. Set actuator_effort_limit to choose it."
            )
        return {}
    return dict.fromkeys(joint_names, voltage * cfg.motor.kt / cfg.motor.resistance)


_PD_GAINS = (
    ActuatorField("stiffness", _PD_API, "newton:kp", default=0.0),
    ActuatorField("damping", _PD_API, "newton:kd", default=0.0),
)
_MAX_EFFORT = ActuatorField("actuator_effort_limit", _MAX_EFFORT_API, "newton:maxEffort", default=_default_effort_limit)
_DC_MOTOR = (
    ActuatorField("saturation_effort", _DC_MOTOR_API, "newton:saturationEffort", required=True),
    ActuatorField("actuator_velocity_limit", _DC_MOTOR_API, "newton:velocityLimit"),
    ActuatorField("actuator_effort_limit", _DC_MOTOR_API, "newton:maxMotorEffort", default=_default_effort_limit),
)
_DELAY = ActuatorField("max_delay", _DELAY_API, "newton:delaySteps", _INT)
_POSITION_BASED = tuple(
    ActuatorField(
        "joint_parameter_lookup",
        _POSITION_BASED_API,
        attr,
        Sdf.ValueTypeNames.FloatArray,
        lambda cfg, names, column=column: (
            {}
            if cfg.joint_parameter_lookup is None
            else dict.fromkeys(names, [float(r[column]) for r in cfg.joint_parameter_lookup])
        ),
        required=True,
    )
    for attr, column in (("newton:lookupPositions", 0), ("newton:lookupEfforts", 2))
)
_NEURAL_MODEL = ActuatorField(
    "network_file", _NEURAL_API, "newton:modelPath", Sdf.ValueTypeNames.Asset, _exported_model_path, required=True
)

# A new prim requires the fit and supply: the drive would silently fall back to m1 friction and no current limit.
_BAM_FIELDS = (
    *(_bam_value(f"motor.{name}", required=True) for name in ("kt", "resistance", "error_gain")),
    *(_bam_value(f"motor.{name}", required=True) for name in ("max_current", "friction_base")),
    *(_bam_value(f"motor.{name}", default=1.0) for name in ("max_pwm", "dtheta_stribeck", "alpha")),
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
    ActuatorField(
        "backlash_joint_template",
        BAM_DRIVE_API,
        "newton:hasBacklash",
        _INT,
        lambda cfg, names: dict.fromkeys(names, 1) if cfg.backlash_joint_template is not None else {},
    ),
    # A new prim requires kt, resistance and vin, so the default only fills joints a configured mapping skips.
    ActuatorField(
        "actuator_effort_limit", BAM_DRIVE_API, "newton:maxEffort", _FLOAT, _bam_effort_limits, _default_effort_limit
    ),
)

_ACTUATOR_LAYOUTS: dict[type, ActuatorLayout] = {
    IdealPDActuatorCfg: ActuatorLayout(IdealPDActuator, _PD_API, (*_PD_GAINS, _MAX_EFFORT)),
    DelayedPDActuatorCfg: ActuatorLayout(DelayedPDActuator, _PD_API, (*_PD_GAINS, _MAX_EFFORT, _DELAY)),
    RemotizedPDActuatorCfg: ActuatorLayout(RemotizedPDActuator, _PD_API, (*_PD_GAINS, *_POSITION_BASED, _DELAY)),
    DCMotorCfg: ActuatorLayout(DCMotor, _PD_API, (*_PD_GAINS, *_DC_MOTOR)),
    ActuatorNetMLPCfg: ActuatorLayout(ActuatorNetMLP, _NEURAL_API, (_NEURAL_MODEL, *_DC_MOTOR)),
    ActuatorNetLSTMCfg: ActuatorLayout(ActuatorNetLSTM, _NEURAL_API, (_NEURAL_MODEL, *_DC_MOTOR)),
    BamActuatorCfg: ActuatorLayout(None, BAM_DRIVE_API, _BAM_FIELDS),
}
"""Newton actuator layout of each supported config type; subclasses use their closest base's layout."""


def _actuator_layout(cfg: ActuatorBaseCfg) -> ActuatorLayout | None:
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
    # Deferred: importing the converter loads torch._dynamo, which adds seconds to every articulation spawn.
    from torch._export.converter import TS2EPConverter  # noqa: PLC0415

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
