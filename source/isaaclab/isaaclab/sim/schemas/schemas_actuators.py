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

import re
from typing import Any

from pxr import Sdf, Usd, UsdPhysics

from isaaclab.actuators._compat import _resolve_limit_aliases
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.utils.string import _resolve_matching_values_dense, resolve_matching_names, string_to_callable


def _resolve_actuator_class(class_type: type | str) -> type:
    """Resolve and validate an actuator class reference for authoring identity checks."""
    from isaaclab.actuators import ActuatorBase  # noqa: PLC0415

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
    from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP  # noqa: PLC0415
    from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_pd import (  # noqa: PLC0415
        DCMotor,
        DelayedPDActuator,
        IdealPDActuator,
        RemotizedPDActuator,
    )
    from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg, RemotizedPDActuatorCfg  # noqa: PLC0415

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


def _validate_newton_native_actuator_cfgs(actuator_cfgs: dict[str, Any]) -> None:
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
    ``effort_limit``, …) into a single ``{joint_name: value}`` dict that
    the authoring loop can ``.get(jname, default)`` against.

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

    No-ops (returns immediately) when:

    * the active :class:`~isaaclab.sim.SimulationContext` was configured
      with ``use_newton_actuators=False`` (or no context is active), or
    * *prim_path* does not resolve to a valid prim on the stage.

    Must be called **after** the articulation is spawned (joint prims
    exist on stage) and **before** the cloner / ``ModelBuilder.add_usd``
    reads the stage.

    Args:
        prim_path: Root prim path of the articulation (e.g.
            ``"/World/Env_0/Robot"``). May contain a regex pattern; the
            first matching prim is used.
        actuator_cfgs: Mapping of group name to
            :class:`~isaaclab.actuators.ActuatorBaseCfg`.
        stage: USD stage to author on. When ``None``, the current stage
            is used.

    Raises:
        ValueError: If Newton-native execution is enabled and an explicit actuator config is unsupported.
    """
    from isaaclab.sim import SimulationContext  # noqa: PLC0415

    sim_ctx = SimulationContext.instance()
    sim_cfg = sim_ctx.cfg if sim_ctx is not None else None
    if sim_cfg is None or not getattr(sim_cfg, "use_newton_actuators", False):
        return

    from isaaclab.sim.utils.queries import find_first_matching_prim  # noqa: PLC0415
    from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415

    if stage is None:
        stage = get_current_stage()

    first_prim = find_first_matching_prim(prim_path)
    if first_prim is None:
        return
    articulation_prim_path = str(first_prim.GetPath())

    _author_actuator_prims(stage, articulation_prim_path, actuator_cfgs)


def _author_actuator_prims(
    stage: Any,
    articulation_prim_path: str,
    actuator_cfgs: dict[str, Any],
) -> None:
    """Inner authoring routine; exposed separately for test fixtures."""
    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not art_prim.IsValid():
        raise ValueError(f"Articulation prim not found: {articulation_prim_path}")

    _validate_newton_native_actuator_cfgs(actuator_cfgs)

    joint_inventory = _collect_joint_prims(art_prim)
    all_joint_names = list(joint_inventory.keys())

    covered_joint_paths: set[str] = set()

    cfg_entries: list[tuple[str, Any, list[str]]] = []
    for group_name, cfg in actuator_cfgs.items():
        if _is_implicit_actuator_cfg(cfg):
            continue

        _ids, joint_names = resolve_matching_names(cfg.joint_names_expr, all_joint_names)
        if not joint_names:
            continue

        resolved_cfg = cfg.copy()
        # Collection construction emits the deprecation warning later in the
        # normal asset lifecycle. Authoring only needs the normalized value.
        _resolve_limit_aliases(group_name, resolved_cfg, joint_names, warn_deprecated=False)
        cfg_entries.append((group_name, resolved_cfg, joint_names))
        for jname in joint_names:
            covered_joint_paths.add(joint_inventory[jname])

    _remove_actuator_prims_for_joints(art_prim, covered_joint_paths)

    from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

    for group_name, cfg, joint_names in cfg_entries:
        stiffness_map = resolve_per_dof(getattr(cfg, "stiffness", None), joint_names)
        damping_map = resolve_per_dof(getattr(cfg, "damping", None), joint_names)

        is_neural = isinstance(cfg, (ActuatorNetMLPCfg, ActuatorNetLSTMCfg))
        is_remotized = isinstance(cfg, RemotizedPDActuatorCfg)
        is_dc_motor = isinstance(cfg, DCMotorCfg)
        is_delayed = isinstance(cfg, DelayedPDActuatorCfg)

        configured_effort_limit = getattr(cfg, "actuator_effort_limit", None)
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

        vel_limit_map = (
            resolve_per_dof(getattr(cfg, "actuator_velocity_limit", None), joint_names) if is_dc_motor else {}
        )
        sat_effort_map = resolve_per_dof(getattr(cfg, "saturation_effort", None), joint_names) if is_dc_motor else {}

        raw_delay = getattr(cfg, "max_delay", 0) if is_delayed else 0
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
            patched_model_path = _resave_checkpoint_with_metadata(cfg.network_file, meta)

        for jname in joint_names:
            joint_prim_path = joint_inventory[jname]

            schemas: list[str] = []
            attrs: dict[str, float | int] = {}
            array_attrs: dict[str, list[float]] = {}

            if is_neural:
                schemas.append("NewtonNeuralControlAPI")
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

            if is_remotized and isinstance(cfg, RemotizedPDActuatorCfg):
                lookup = cfg.joint_parameter_lookup
                schemas.append("NewtonPositionBasedClampingAPI")
                array_attrs["lookup_positions"] = [row[0] for row in lookup]
                array_attrs["lookup_efforts"] = [row[2] for row in lookup]

            delay_steps = delay_map.get(jname, 0)
            if delay_steps > 0:
                schemas.append("NewtonActuatorDelayAPI")
                attrs["delay_steps"] = delay_steps
                attrs["max_delay"] = delay_steps

            act_prim_path = f"{articulation_prim_path}/{group_name}_{jname}_actuator"
            act_prim = stage.DefinePrim(act_prim_path, "NewtonActuator")

            existing = act_prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
            existing.prependedItems = list(schemas)
            act_prim.SetMetadata("apiSchemas", existing)

            rel = act_prim.CreateRelationship("newton:targets")
            rel.SetTargets([Sdf.Path(joint_prim_path)])

            if patched_model_path is not None:
                act_prim.CreateAttribute("newton:modelPath", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(patched_model_path)
                )

            for attr_name, attr_val in attrs.items():
                usd_name = f"newton:{_snake_to_camel(attr_name)}"
                if isinstance(attr_val, int):
                    act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Int).Set(attr_val)
                else:
                    act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Float).Set(float(attr_val))

            for attr_name, attr_val in array_attrs.items():
                usd_name = f"newton:{_snake_to_camel(attr_name)}"
                act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.FloatArray).Set(attr_val)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_SNAKE_TO_CAMEL_RE = re.compile(r"_([a-z])")


def _snake_to_camel(name: str) -> str:
    """Convert a snake_case name to camelCase."""
    return _SNAKE_TO_CAMEL_RE.sub(lambda m: m.group(1).upper(), name)


def _get_authored_joint_effort_limit(stage: Usd.Stage, joint_prim_path: str) -> float | None:
    """Read a revolute or prismatic joint's authored USD drive effort limit."""
    joint_prim = stage.GetPrimAtPath(joint_prim_path)
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        drive_name = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        drive_name = "linear"
    else:
        return None
    value = UsdPhysics.DriveAPI(joint_prim, drive_name).GetMaxForceAttr().Get()
    return None if value is None else float(value)


def _collect_joint_prims(art_prim: Any) -> dict[str, str]:
    """Collect all joint prims under an articulation subtree.

    Returns:
        Ordered mapping of joint name to full prim path.
    """
    _JOINT_TYPES = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}

    joints: dict[str, str] = {}
    for prim in Usd.PrimRange(art_prim):
        if prim.GetTypeName() in _JOINT_TYPES:
            joints[prim.GetName()] = str(prim.GetPath())
    return joints


def _remove_actuator_prims_for_joints(
    art_prim: Any,
    joint_paths: set[str],
) -> None:
    """Deactivate ``NewtonActuator`` prims whose target is in *joint_paths*.

    Deactivated prims are invisible to ``Usd.PrimRange`` and therefore
    ignored by ``ModelBuilder.add_usd``. Using ``SetActive(False)``
    instead of ``RemovePrim`` works correctly when the prim originates
    from a USD reference or payload.

    Only prims under the *art_prim* subtree are considered.
    """
    to_deactivate: list = []
    for prim in Usd.PrimRange(art_prim):
        if prim.GetTypeName() != "NewtonActuator":
            continue
        rel = prim.GetRelationship("newton:targets")
        if rel and rel.IsValid():
            for target in rel.GetTargets():
                if str(target) in joint_paths:
                    to_deactivate.append(prim)
                    break

    for prim in to_deactivate:
        prim.SetActive(False)


def _resave_checkpoint_with_metadata(
    original_path: str,
    metadata: dict[str, Any],
) -> str:
    """Re-save a neural-network checkpoint with updated metadata.

    Resolves the configured path through the shared asset cache, loads the
    original TorchScript or dict checkpoint, merges *metadata* into any
    existing metadata (Lab config values take precedence), and writes the
    result to a temporary ``.pt`` file that persists for the lifetime of the
    process.

    Returns:
        Path to the temporary checkpoint file.
    """
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from isaaclab.utils.assets import retrieve_file_path  # noqa: PLC0415

    local_path = retrieve_file_path(original_path)

    extra_files: dict[str, str] = {"metadata.json": ""}
    is_torchscript = True
    try:
        net = torch.jit.load(local_path, map_location="cpu", _extra_files=extra_files)
        existing_meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
    except Exception:
        is_torchscript = False
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"Cannot load checkpoint at '{original_path}'; "
                "expected a TorchScript archive or a dict with a 'model' key"
            )
        net = checkpoint["model"]
        existing_meta = checkpoint.get("metadata", {})

    merged = {**existing_meta, **metadata}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    if is_torchscript:
        extra_out = {"metadata.json": json.dumps(merged)}
        torch.jit.save(net, tmp_path, _extra_files=extra_out)
    else:
        torch.save({"model": net, "metadata": merged}, tmp_path)

    return tmp_path
