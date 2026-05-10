# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD authoring for Newton-native actuators.

:func:`author_newton_actuator_prims` translates IsaacLab actuator configs
into ``NewtonActuator`` USD prims. Both the Newton ``ModelBuilder.add_usd``
path and the PhysX adapter's
:meth:`~isaaclab_newton.actuators.adapter.NewtonActuatorAdapter.from_usd`
read the same authored prims, ensuring both backends construct
:class:`~newton.actuators.Actuator` instances with matching parameters.

:func:`author_actuator_prims_for_articulation` is a thin wrapper that
resolves per-articulation pre-conditions (sim cfg gating, prim lookup)
and dispatches into :func:`author_newton_actuator_prims`. It is invoked
from the schema-side articulation initialisation path.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def resolve_per_dof(
    value: dict[str, float | int] | float | int | None,
    joint_names: list[str],
    cast: type = float,
) -> dict[str, float | int]:
    """Expand a scalar or regex-keyed dict cfg value into a per-joint mapping.

    Used by :func:`author_newton_actuator_prims` to flatten the various
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


def author_actuator_prims_for_articulation(cfg: Any, sim_cfg: Any, stage: Any) -> None:
    """Resolve per-articulation authoring args from ``cfg`` + ``sim_cfg`` and dispatch.

    Higher-level wrapper around :func:`author_newton_actuator_prims` that
    handles the pre-conditions previously inlined in
    ``BaseArticulation._author_newton_actuator_prims``:

    * No-op when the simulation is not configured for the Newton actuator
      fast path (``sim_cfg is None`` or ``use_newton_actuators=False``).
    * Falls back from ``cfg.spawn.spawn_path`` to ``cfg.prim_path`` when the
      former is not set, then resolves the first matching USD prim.
    * No-op when no matching prim exists on the stage.

    Keeping this resolution on the schema side lets articulations call
    authoring with a single line, and lets the rules be tested without
    constructing an articulation instance.
    """
    if sim_cfg is None or not getattr(sim_cfg, "use_newton_actuators", False):
        return

    from isaaclab.sim.utils.queries import find_first_matching_prim  # noqa: PLC0415

    spawn_path = getattr(cfg.spawn, "spawn_path", None) if cfg.spawn is not None else None
    search_path = spawn_path if spawn_path is not None else cfg.prim_path
    first_prim = find_first_matching_prim(search_path)
    if first_prim is None:
        return

    author_newton_actuator_prims(
        stage=stage,
        articulation_prim_path=str(first_prim.GetPath()),
        actuator_cfgs=cfg.actuators,
    )


def author_newton_actuator_prims(
    stage: Any,
    articulation_prim_path: str,
    actuator_cfgs: dict[str, Any],
) -> None:
    """Author ``NewtonActuator`` USD prims from IsaacLab actuator configs.

    For every joint covered by an explicit (non-implicit) Lab actuator config,
    any existing ``NewtonActuator`` prim targeting that joint is replaced by a
    new one created from the config values.  Joints **not** covered by any
    Lab config keep their USD-authored actuators unchanged.

    The supported config-to-schema mapping is:

    * :class:`~isaaclab.actuators.IdealPDActuatorCfg` ->
      ``NewtonPDControlAPI`` + ``NewtonMaxEffortClampingAPI``
    * :class:`~isaaclab.actuators.DCMotorCfg` ->
      ``NewtonPDControlAPI`` + ``NewtonDCMotorClampingAPI``
    * :class:`~isaaclab.actuators.DelayedPDActuatorCfg` ->
      same as ``IdealPDActuatorCfg`` + ``NewtonActuatorDelayAPI``
    * :class:`~isaaclab.actuators.RemotizedPDActuatorCfg` ->
      same as ``DelayedPDActuatorCfg`` + ``NewtonPositionBasedClampingAPI``
    * :class:`~isaaclab.actuators.ActuatorNetMLPCfg` /
      :class:`~isaaclab.actuators.ActuatorNetLSTMCfg` ->
      ``NewtonNeuralControlAPI`` (+ ``NewtonDCMotorClampingAPI``)

    Must be called **after** the articulation is spawned (joint prims exist
    on stage) and **before** the cloner / ``ModelBuilder.add_usd`` reads
    the stage.

    Args:
        stage: The USD stage to author prims on.
        articulation_prim_path: Root prim path of the articulation
            (e.g. ``"/World/Env_0/Robot"``).  Must not contain wildcards.
        actuator_cfgs: Mapping of group name to ``ActuatorBaseCfg``.
    """
    from pxr import Sdf  # noqa: PLC0415

    from isaaclab.actuators import ImplicitActuator  # noqa: PLC0415
    from isaaclab.utils.string import resolve_matching_names  # noqa: PLC0415

    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not art_prim.IsValid():
        raise ValueError(f"Articulation prim not found: {articulation_prim_path}")

    joint_inventory = _collect_joint_prims(art_prim)
    all_joint_names = list(joint_inventory.keys())

    covered_joint_paths: set[str] = set()

    cfg_entries: list[tuple[str, Any, list[str]]] = []
    for group_name, cfg in actuator_cfgs.items():
        cls_type = cfg.class_type
        is_implicit = (
            "ImplicitActuator" in cls_type
            if isinstance(cls_type, str)
            else issubclass(cls_type, ImplicitActuator)
        )
        if is_implicit:
            continue

        _ids, joint_names = resolve_matching_names(cfg.joint_names_expr, all_joint_names)
        if not joint_names:
            continue

        cfg_entries.append((group_name, cfg, joint_names))
        for jname in joint_names:
            covered_joint_paths.add(joint_inventory[jname])

    _remove_actuator_prims_for_joints(art_prim, covered_joint_paths)

    from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg  # noqa: PLC0415
    from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg, RemotizedPDActuatorCfg  # noqa: PLC0415

    _SUPPORTED_CFG_TYPES = (
        IdealPDActuatorCfg,
        DCMotorCfg,
        DelayedPDActuatorCfg,
        RemotizedPDActuatorCfg,
        ActuatorNetMLPCfg,
        ActuatorNetLSTMCfg,
    )

    for group_name, cfg, joint_names in cfg_entries:
        if not isinstance(cfg, _SUPPORTED_CFG_TYPES):
            logger.warning(
                "Actuator group '%s' uses config type '%s' which is not supported by Newton-native"
                " actuator authoring. The group will be skipped.",
                group_name,
                type(cfg).__name__,
            )
            continue
        stiffness_map = resolve_per_dof(getattr(cfg, "stiffness", None), joint_names)
        damping_map = resolve_per_dof(getattr(cfg, "damping", None), joint_names)
        effort_map = resolve_per_dof(getattr(cfg, "effort_limit", None), joint_names)

        is_neural = isinstance(cfg, (ActuatorNetMLPCfg, ActuatorNetLSTMCfg))
        is_remotized = isinstance(cfg, RemotizedPDActuatorCfg)
        is_dc_motor = isinstance(cfg, DCMotorCfg)
        is_delayed = isinstance(cfg, DelayedPDActuatorCfg)

        vel_limit_map = resolve_per_dof(getattr(cfg, "velocity_limit", None), joint_names) if is_dc_motor else {}
        sat_effort_map = (
            resolve_per_dof(getattr(cfg, "saturation_effort", None), joint_names) if is_dc_motor else {}
        )

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
            elif jname in effort_map:
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


def _collect_joint_prims(art_prim: Any) -> dict[str, str]:
    """Collect all joint prims under an articulation subtree.

    Returns:
        Ordered mapping of joint name to full prim path.
    """
    from pxr import Usd  # noqa: PLC0415

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
    ignored by ``ModelBuilder.add_usd``.  Using ``SetActive(False)``
    instead of ``RemovePrim`` works correctly when the prim originates
    from a USD reference or payload.

    Only prims under the *art_prim* subtree are considered.
    """
    from pxr import Usd  # noqa: PLC0415

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

    Loads the original TorchScript or dict checkpoint, merges *metadata*
    into any existing metadata (Lab config values take precedence), and
    writes the result to a temporary ``.pt`` file that persists for the
    lifetime of the process.

    Returns:
        Path to the temporary checkpoint file.
    """
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415

    extra_files: dict[str, str] = {"metadata.json": ""}
    is_torchscript = True
    try:
        net = torch.jit.load(original_path, map_location="cpu", _extra_files=extra_files)
        existing_meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
    except Exception:
        is_torchscript = False
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
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
