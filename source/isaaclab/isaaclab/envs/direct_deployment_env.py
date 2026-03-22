# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment environment that runs LEAPP-exported policies in simulation.

This environment bypasses all Isaac Lab managers (observation, action, reward, etc.)
and instead wires raw ``ArticulationData`` properties and ``CommandManager`` outputs
directly to a LEAPP ``InferenceManager``, then writes the model outputs back to the
articulation.  All I/O resolution is driven by the ``kind`` field in the LEAPP YAML.
"""

from __future__ import annotations

import logging
import torch
import yaml
from dataclasses import dataclass
from typing import Any

from leapp import InferenceManager

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.managers import CommandManager, EventManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# I/O spec dataclasses
# ══════════════════════════════════════════════════════════════════


@dataclass
class StateInputSpec:
    """Read a property from ``ArticulationData``, optionally sliced by joint."""

    property_name: str
    joint_ids: list[int] | None = None


@dataclass
class CommandInputSpec:
    """Read a command tensor from ``CommandManager``."""

    command_term_name: str


@dataclass
class OutputSpec:
    """Write a tensor to an ``Articulation`` method, optionally indexed by joint."""

    method_name: str
    joint_ids: list[int] | None = None


# ══════════════════════════════════════════════════════════════════
# Kind → source/target resolution helpers
# ══════════════════════════════════════════════════════════════════

_JOINT_LEVEL_KIND_PREFIXES = ("state/joint/", "target/joint/")
_JOINT_LEVEL_GAIN_KINDS = ("kp", "kd")


def _build_kind_to_property_map() -> dict[str, list[str]]:
    """Scan ``ArticulationData`` for ``_leapp_semantics`` properties.

    Returns a mapping from ``kind`` string to a list of property names that
    carry that kind (there can be more than one, e.g. ``root_lin_vel_b`` and
    ``root_lin_vel_w`` both have ``state/body/linear_velocity``).
    """
    kind_to_props: dict[str, list[str]] = {}
    for prop_name in dir(ArticulationData):
        prop = getattr(ArticulationData, prop_name, None)
        if isinstance(prop, property) and prop.fget and hasattr(prop.fget, "_leapp_semantics"):
            kind = prop.fget._leapp_semantics.kind
            if kind is not None:
                kind_to_props.setdefault(kind, []).append(prop_name)
    return kind_to_props


def _build_kind_to_write_method_map() -> dict[str, str]:
    """Scan ``Articulation`` for ``_leapp_semantics`` methods + hardcoded kp/kd.

    Returns a mapping from output ``kind`` to the method name on ``Articulation``.
    """
    kind_to_method: dict[str, str] = {}
    for method_name in dir(Articulation):
        method = getattr(Articulation, method_name, None)
        if callable(method) and hasattr(method, "_leapp_semantics"):
            kind = method._leapp_semantics.kind
            if kind is not None:
                kind_to_method[kind] = method_name
    kind_to_method["kp"] = "write_joint_stiffness_to_sim"
    kind_to_method["kd"] = "write_joint_damping_to_sim"
    return kind_to_method


def _disambiguate_property(kind: str, leapp_name: str, kind_to_props: dict[str, list[str]]) -> str:
    """Pick the right ``ArticulationData`` property when multiple share a ``kind``.

    The export path uses the property name as the LEAPP input name, so we strip
    the ``_in`` / ``_out`` suffix that LEAPP adds for collision avoidance and match.
    """
    candidates = kind_to_props.get(kind)
    if candidates is None:
        raise ValueError(f"No ArticulationData property found for kind='{kind}'")
    if len(candidates) == 1:
        return candidates[0]
    base_name = leapp_name.removesuffix("_in").removesuffix("_out")
    for prop in candidates:
        if prop == base_name:
            return prop
    return candidates[0]


def _resolve_joint_ids(element_names: list | None, asset: Articulation) -> list[int] | None:
    """Convert ``element_names[0]`` joint names to integer joint indices.

    Returns ``None`` when no slicing is needed (all joints or non-joint tensor).
    """
    if element_names is None:
        return None
    joint_names = element_names[0]
    if not isinstance(joint_names, list) or not joint_names:
        return None
    if joint_names == list(asset.joint_names):
        return None
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    return joint_ids


def _find_command_term_by_hint(kind: str, command_manager: CommandManager) -> str:
    """Find the ``CommandTerm`` name whose ``cfg.cmd_hint`` matches ``kind``."""
    for name, term in command_manager._terms.items():
        if getattr(term.cfg, "cmd_hint", None) == kind:
            return name
    raise ValueError(f"No command term with cmd_hint='{kind}'. Available terms: {list(command_manager._terms.keys())}")


def _find_robot_asset(scene: InteractiveScene) -> Articulation:
    """Return the first ``Articulation`` in the scene (assumed to be the robot)."""
    for entity_name in scene.articulations:
        entity = scene[entity_name]
        if isinstance(entity, Articulation):
            return entity
    raise RuntimeError("No Articulation found in scene")


# ══════════════════════════════════════════════════════════════════
# DirectDeploymentEnv
# ══════════════════════════════════════════════════════════════════


class DirectDeploymentEnv:
    """Runs a LEAPP-exported policy in an Isaac Lab scene.

    The environment sets up the simulation scene and physics from a standard
    Isaac Lab config, then wires raw sensor/command data to a LEAPP
    ``InferenceManager`` and writes the model outputs back to the articulation.

    No observation, action, reward, termination, or curriculum managers are used.
    The LEAPP model already contains all pre/post-processing.
    """

    def __init__(self, cfg: Any, leapp_yaml_path: str):
        """Initialize the deployment environment.

        Args:
            cfg: A ``ManagerBasedRLEnvCfg`` (or compatible) task config.
            leapp_yaml_path: Path to the LEAPP ``.yaml`` pipeline description.
        """

        cfg.scene.num_envs = 1
        cfg.validate()
        self.cfg = cfg
        self._is_closed = False
        self._leapp_yaml_path = leapp_yaml_path
        self._step_count = 0

        # ── Simulation + scene ────────────────────────────────────
        self.sim = SimulationContext(cfg.sim)
        if "cuda" in self.sim.device:
            torch.cuda.set_device(self.sim.device)

        with use_stage(self.sim.get_initial_stage()):
            self.scene = InteractiveScene(cfg.scene)
            attach_stage_to_usd_context()
        self.sim.reset()
        self.scene.update(dt=self.physics_dt)

        # ── Robot asset ───────────────────────────────────────────
        self._asset = _find_robot_asset(self.scene)

        # ── EventManager (optional, for resets) ───────────────────
        self.event_manager: EventManager | None = None
        if hasattr(cfg, "events") and cfg.events is not None:
            self.event_manager = EventManager(cfg.events, self)

        # ── CommandManager (optional, for command/* inputs) ───────
        self.command_manager: CommandManager | None = None
        if hasattr(cfg, "commands") and cfg.commands is not None:
            self.command_manager = CommandManager(cfg.commands, self)

        # ── LEAPP InferenceManager ────────────────────────────────
        self.inference = InferenceManager(leapp_yaml_path)

        # ── Parse YAML and resolve I/O mappings ───────────────────
        with open(leapp_yaml_path) as f:
            self._leapp_desc = yaml.safe_load(f)
        self._input_mapping: dict[str, StateInputSpec | CommandInputSpec] = {}
        self._output_mapping: dict[str, OutputSpec] = {}
        self._resolve_io()

        logger.info(
            "DirectDeploymentEnv ready — %d inputs, %d outputs mapped",
            len(self._input_mapping),
            len(self._output_mapping),
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self) -> str:
        return self.sim.device

    # ── I/O Resolution ────────────────────────────────────────────

    def _resolve_io(self):
        """Build ``_input_mapping`` and ``_output_mapping`` from LEAPP YAML ``kind`` fields."""
        kind_to_props = _build_kind_to_property_map()
        kind_to_write = _build_kind_to_write_method_map()
        pipeline = self._leapp_desc["pipeline"]

        # --- Inputs ---
        for node_name, input_names in pipeline["inputs"].items():
            node = self.inference.nodes[node_name]
            desc_by_name = {d["name"]: d for d in node.input_descriptions}
            for input_name in input_names:
                desc = desc_by_name[input_name]
                kind = desc.get("kind")
                key = f"{node_name}/{input_name}"
                if kind is None:
                    continue
                if kind.startswith("state/"):
                    prop = _disambiguate_property(kind, input_name, kind_to_props)
                    needs_joint_slice = kind.startswith("state/joint/")
                    jids = _resolve_joint_ids(desc.get("element_names"), self._asset) if needs_joint_slice else None
                    self._input_mapping[key] = StateInputSpec(property_name=prop, joint_ids=jids)
                elif kind.startswith("command/"):
                    if self.command_manager is None:
                        raise RuntimeError(
                            f"LEAPP input '{key}' has kind='{kind}' but no CommandManager "
                            "is available (cfg.commands is None)."
                        )
                    term_name = _find_command_term_by_hint(kind, self.command_manager)
                    self._input_mapping[key] = CommandInputSpec(command_term_name=term_name)
                else:
                    logger.warning("Unknown input kind '%s' for '%s' — skipping", kind, key)

        # --- Outputs ---
        for node_name, output_names in pipeline["outputs"].items():
            node = self.inference.nodes[node_name]
            desc_by_name = {d["name"]: d for d in node.output_descriptions}
            for output_name in output_names:
                desc = desc_by_name[output_name]
                kind = desc.get("kind")
                key = f"{node_name}/{output_name}"
                if kind is None:
                    continue
                if kind not in kind_to_write:
                    logger.warning("Unknown output kind '%s' for '%s' — skipping", kind, key)
                    continue
                method_name = kind_to_write[kind]
                needs_joint_ids = kind.startswith("target/joint/") or kind in _JOINT_LEVEL_GAIN_KINDS
                jids = _resolve_joint_ids(desc.get("element_names"), self._asset) if needs_joint_ids else None
                self._output_mapping[key] = OutputSpec(method_name=method_name, joint_ids=jids)

    # ── Read / Write ──────────────────────────────────────────────

    def _read_inputs(self) -> dict[str, torch.Tensor]:
        """Read all mapped inputs from the scene and command manager."""
        inputs: dict[str, torch.Tensor] = {}
        for key, spec in self._input_mapping.items():
            if isinstance(spec, StateInputSpec):
                value = getattr(self._asset.data, spec.property_name)
                if spec.joint_ids is not None:
                    value = value[:, spec.joint_ids]
                inputs[key] = value
            elif isinstance(spec, CommandInputSpec):
                inputs[key] = self.command_manager.get_command(spec.command_term_name)
        return inputs

    def _write_outputs(self, outputs: dict[str, torch.Tensor]):
        """Write model outputs to the articulation."""
        for key, tensor in outputs.items():
            spec = self._output_mapping.get(key)
            if spec is None:
                continue
            method = getattr(self._asset, spec.method_name)
            if spec.joint_ids is not None:
                method(tensor, joint_ids=spec.joint_ids)
            else:
                method(tensor)

    # ── Public API ────────────────────────────────────────────────

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset the scene and inference state.

        Returns:
            The initial input tensors (for logging / debugging).
        """
        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)

        self.scene.reset(env_ids)

        if self.event_manager is not None and "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=self._step_count)
        if self.command_manager is not None:
            self.command_manager.reset(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

        self.inference.reset()

        return self._read_inputs()

    def step(self, external_inputs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Run one environment step: read → infer → write → physics.

        Args:
            external_inputs: Optional overrides keyed by ``"ModelName/input_name"``.
                Takes precedence over auto-resolved state/command values.

        Returns:
            The dict of pipeline outputs from ``InferenceManager.run_policy()``.
        """
        self._step_count += 1

        # 1. Update commands
        if self.command_manager is not None:
            self.command_manager.compute(dt=self.step_dt)

        # 2. Read inputs
        inputs = self._read_inputs()

        # 3. Merge external overrides
        if external_inputs is not None:
            inputs.update(external_inputs)

        # 4. Infer
        with torch.inference_mode():
            outputs = self.inference.run_policy(inputs)

        # 5. Write outputs to asset
        self._write_outputs(outputs)

        # 6. Decimation loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        return outputs

    def close(self):
        """Clean up the environment."""
        if not self._is_closed:
            if self.command_manager is not None:
                del self.command_manager
            if self.event_manager is not None:
                del self.event_manager
            del self.scene
            self._is_closed = True
