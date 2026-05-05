# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment environment that runs LEAPP-exported policies in simulation.

This environment bypasses all Isaac Lab managers (observation, action, reward, etc.)
and instead wires scene entity data properties and ``CommandManager`` outputs directly
to a LEAPP ``InferenceManager``, then writes the model outputs back to the
corresponding scene entities.  All I/O resolution is driven by the
``isaaclab_connection`` field in the LEAPP YAML.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import yaml

try:
    from leapp import InferenceManager
except ImportError as e:
    raise ImportError("LEAPP package is required for policy deployment testing. Install with: pip install leapp") from e

from isaaclab.managers import CommandManager, EventManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import use_stage
from isaaclab.utils.configclass import resolve_cfg_presets

from .ui import ViewportCameraController

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# I/O spec dataclasses
# ══════════════════════════════════════════════════════════════════


@dataclass
class StateInputSpec:
    """Read a property from a scene entity's data object."""

    entity_name: str
    property_name: str
    joint_ids: list[int] | None = None


@dataclass
class CommandInputSpec:
    """Read a command tensor from ``CommandManager``."""

    command_term_name: str


@dataclass
class WriteOutputSpec:
    """Write a tensor to a scene entity method, optionally indexed by joint."""

    entity_name: str
    method_name: str
    value_param: str
    joint_ids: list[int] | None = None


# ══════════════════════════════════════════════════════════════════
# Connection-string helpers
# ══════════════════════════════════════════════════════════════════


def _resolve_joint_ids(element_names: list | None, entity: Any) -> list[int] | None:
    """Convert ``element_names[0]`` joint names to integer joint indices.

    Args:
        element_names: LEAPP element-name metadata for the tensor, or ``None``
            when the tensor does not define named elements.
        entity: Scene entity that may provide ``joint_names`` and
            ``find_joints()`` for name-to-index resolution.

    Returns:
        Joint indices matching ``element_names[0]``, or ``None`` when no
        slicing is needed because all joints are selected, the tensor is not
        joint-indexed, or the entity does not support joint lookup.
    """
    if element_names is None or not hasattr(entity, "find_joints"):
        return None

    # leapp tensor semantics will always store the array in a nested list of lists.
    # NOTE: this is added in explicitly to handle partial joint application. currently
    # this environment does not handle element reordering yet. Thus, this function
    # is specialized to handle joints, hence reading index 0.
    joint_names = element_names[0]
    if not isinstance(joint_names, list) or not joint_names:
        return None
    entity_joint_names = list(entity.joint_names)
    # Only resolve indices when the leading element-name axis actually refers
    # to a subset of this articulation's joints. Other tensors can use axis
    # labels like ["x", "y", "z"] or body names in the first axis.
    matching_joint_names = [name for name in joint_names if name in entity_joint_names]
    if not matching_joint_names:
        return None
    if len(matching_joint_names) != len(joint_names):
        raise ValueError(
            f"LEAPP element names mix joint and non-joint labels for an articulation-backed tensor: {joint_names}"
        )
    if joint_names == entity_joint_names:
        return None
    joint_ids, _ = entity.find_joints(joint_names, preserve_order=True)
    return joint_ids


def _first_param_name(method: Any) -> str:
    """Return the name of the first non-self parameter of *method*.

    Expects a bound method — ``inspect.signature`` on a bound method
    already excludes ``self``, so ``params[0]`` is the first real parameter.

    Args:
        method: Bound method whose first callable parameter should be
            inspected.

    Returns:
        The name of the first non-``self`` parameter.
    """
    params = list(inspect.signature(method).parameters.values())
    if not params:
        raise TypeError(f"{method} has no parameters")
    return params[0].name


# ══════════════════════════════════════════════════════════════════
# LeappDeploymentEnv
# ══════════════════════════════════════════════════════════════════


class LeappDeploymentEnv:
    """Runs a LEAPP-exported policy in an Isaac Lab scene.

    The environment sets up the simulation scene and physics from a standard
    Isaac Lab config, then wires raw sensor/command data to a LEAPP
    ``InferenceManager`` and writes the model outputs back to the corresponding
    scene entities.

    I/O wiring is driven entirely by the ``isaaclab_connection`` metadata field
    in the LEAPP YAML. Each connection string encodes the type of access, the
    scene entity name, and the property or method to call:

    - ``state:{entity}:{property}`` -- read ``scene[entity].data.{property}``
    - ``command:{name}`` -- read ``command_manager.get_command(name)``
    - ``write:{entity}:{method}`` -- call ``scene[entity].{method}(tensor, ...)``

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
        resolve_cfg_presets(cfg)
        self.cfg = cfg
        self._is_closed = False
        self._leapp_yaml_path = leapp_yaml_path
        self._step_count = 0
        self._sim_step_counter = 0

        # ── Simulation + scene ────────────────────────────────────
        self.sim = SimulationContext(cfg.sim)
        if "cuda" in self.sim.device:
            torch.cuda.set_device(self.sim.device)

        with use_stage(self.sim.stage):
            self.scene = InteractiveScene(cfg.scene)
        with use_stage(self.sim.stage):
            self.sim.reset()
        self.scene.update(dt=self.physics_dt)
        self.has_rtx_sensors = bool(self.sim.get_setting("/isaaclab/render/rtx_sensors"))

        # Match the standard env initialization path for viewport camera setup.
        has_visualizers = bool(self.sim.get_setting("/isaaclab/visualizer"))
        if self.sim.has_gui or has_visualizers:
            self.viewport_camera_controller = ViewportCameraController(cast(Any, self), self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # ── EventManager (optional, for resets) ───────────────────
        self.event_manager: EventManager | None = None
        if hasattr(cfg, "events") and cfg.events is not None:
            self.event_manager = EventManager(cfg.events, cast(Any, self))

        # ── CommandManager (optional, for command/* inputs) ───────
        self.command_manager: CommandManager | None = None
        if hasattr(cfg, "commands") and cfg.commands is not None:
            self.command_manager = CommandManager(cfg.commands, cast(Any, self))

        # ── LEAPP InferenceManager ────────────────────────────────
        self.inference = InferenceManager(leapp_yaml_path)

        # ── Parse YAML and resolve I/O mappings ───────────────────
        with open(leapp_yaml_path) as f:
            self._leapp_desc = yaml.safe_load(f)
        self._input_mapping: dict[str, StateInputSpec | CommandInputSpec] = {}
        self._output_mapping: dict[str, WriteOutputSpec] = {}
        self._resolve_io()

        logger.info(
            "LeappDeploymentEnv ready — %d inputs, %d outputs mapped",
            len(self._input_mapping),
            len(self._output_mapping),
        )

        if self.sim.has_gui and getattr(self.cfg, "ui_window_class_type", None) is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            self._window = None

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
        """Build ``_input_mapping`` and ``_output_mapping`` from LEAPP metadata.

        Parses the ``isaaclab_connection`` field in the loaded LEAPP YAML and
        resolves each declared input/output to the corresponding scene entity,
        command term, and optional joint index selection.
        """
        pipeline = self._leapp_desc["pipeline"]

        for node_name, input_names in pipeline["inputs"].items():
            node = self.inference.nodes[node_name]
            desc_by_name = {d["name"]: d for d in node.input_descriptions}
            for input_name in input_names:
                desc = desc_by_name[input_name]
                connection = desc.get("isaaclab_connection")
                if connection is None:
                    continue
                key = f"{node_name}/{input_name}"
                parts = connection.split(":")
                conn_type = parts[0]

                if conn_type == "state":
                    entity_name, prop_name = parts[1], parts[2]
                    entity = self.scene[entity_name]
                    jids = _resolve_joint_ids(desc.get("element_names"), entity)
                    self._input_mapping[key] = StateInputSpec(
                        entity_name=entity_name,
                        property_name=prop_name,
                        joint_ids=jids,
                    )
                elif conn_type == "command":
                    command_name = parts[1]
                    if self.command_manager is None:
                        raise RuntimeError(
                            f"LEAPP input '{key}' requires command '{command_name}' but no "
                            "CommandManager is available (cfg.commands is None)."
                        )
                    self._input_mapping[key] = CommandInputSpec(command_term_name=command_name)
                else:
                    logger.warning("Unknown connection type '%s' for input '%s'", conn_type, key)

        for node_name, output_names in pipeline["outputs"].items():
            node = self.inference.nodes[node_name]
            desc_by_name = {d["name"]: d for d in node.output_descriptions}
            for output_name in output_names:
                desc = desc_by_name[output_name]
                connection = desc.get("isaaclab_connection")
                if connection is None:
                    continue
                key = f"{node_name}/{output_name}"
                parts = connection.split(":")
                conn_type = parts[0]

                if conn_type == "write":
                    entity_name, method_name = parts[1], parts[2]
                    entity = self.scene[entity_name]
                    jids = _resolve_joint_ids(desc.get("element_names"), entity)
                    value_param = _first_param_name(getattr(entity, method_name))
                    self._output_mapping[key] = WriteOutputSpec(
                        entity_name=entity_name,
                        method_name=method_name,
                        value_param=value_param,
                        joint_ids=jids,
                    )
                else:
                    logger.warning("Unknown connection type '%s' for output '%s'", conn_type, key)

    # ── Read / Write ──────────────────────────────────────────────

    def _read_inputs(self) -> dict[str, torch.Tensor]:
        """Read all mapped inputs from scene entities and command manager.

        Returns:
            A mapping from ``"node_name/tensor_name"`` to the tensor value that
            should be passed to the LEAPP inference pipeline.
        """
        inputs: dict[str, torch.Tensor] = {}
        for key, spec in self._input_mapping.items():
            if isinstance(spec, StateInputSpec):
                entity = self.scene[spec.entity_name]
                value = getattr(entity.data, spec.property_name).torch
                if spec.joint_ids is not None:
                    value = value[:, spec.joint_ids]
                inputs[key] = value
            elif isinstance(spec, CommandInputSpec):
                command_manager = self.command_manager
                assert command_manager is not None
                inputs[key] = command_manager.get_command(spec.command_term_name)
        return inputs

    def _write_outputs(self, outputs: dict[str, torch.Tensor]):
        """Write model outputs to scene entities.

        Args:
            outputs: Model outputs keyed by ``"node_name/tensor_name"`` as
                returned by :meth:`step` and ``InferenceManager.run_policy()``.
        """
        for key, tensor in outputs.items():
            spec = self._output_mapping.get(key)
            if spec is None:
                continue
            entity = self.scene[spec.entity_name]
            method = getattr(entity, spec.method_name)
            if spec.joint_ids is not None:
                method(**{spec.value_param: tensor, "joint_ids": spec.joint_ids})
            else:
                method(**{spec.value_param: tensor})

    # ── Public API ────────────────────────────────────────────────

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset the scene and inference state.

        Returns:
            The initial input tensors (for logging / debugging).
        """
        env_ids = [0]

        self.scene.reset(env_ids)

        if self.event_manager is not None and "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=self._step_count)
        if self.command_manager is not None:
            self.command_manager.reset(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

        # If RTX sensors are present, rerender after reset to refresh their outputs.
        if self.has_rtx_sensors and getattr(self.cfg, "num_rerenders_on_reset", 0) > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        if getattr(self.cfg, "wait_for_textures", False) and self.has_rtx_sensors:
            assets_loading = getattr(self.sim.physics_manager, "assets_loading", None)
            if callable(assets_loading):
                while assets_loading():
                    self.sim.render()

        self.inference.reset()

        return self._read_inputs()

    def step(self, external_inputs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Run one environment step: read -> infer -> write -> physics.

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

        # 5. Write outputs to scene entities
        self._write_outputs(outputs)

        # 6. Decimation loop
        is_rendering = self.sim.is_rendering
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        return outputs

    def close(self):
        """Clean up the environment and release simulator-owned resources."""
        if not self._is_closed:
            self.sim.stop()
            if self.command_manager is not None:
                del self.command_manager
            if self.event_manager is not None:
                del self.event_manager
            del self.scene
            if self.viewport_camera_controller is not None:
                del self.viewport_camera_controller
            self.sim.clear_instance()
            if self._window is not None:
                self._window = None
            self._is_closed = True
