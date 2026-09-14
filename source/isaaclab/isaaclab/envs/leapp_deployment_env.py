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
from collections.abc import Callable, Mapping
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
from isaaclab.utils.seed import configure_seed

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
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None


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


@dataclass(frozen=True)
class ControllerOwnedWriteSpec:
    """A simulator articulation write supplied outside the policy graph.

    Attributes:
        capability: Stable name of the capability required from the adapter.
        source_term: Action term that declared ownership during export.
        kind: LEAPP semantic kind of the write target.
        entity_name: Scene articulation receiving the write.
        method_name: Articulation method used by simulation.
        cadence: Point in the environment lifecycle at which the write is applied.
        joint_names: Controlled joints in exported order.
        joint_ids: Scene joint indices in exported order.
    """

    capability: str
    source_term: str
    kind: str
    entity_name: str
    method_name: str
    cadence: str
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]


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


def _resolve_input_transform(data: Any, property_name: str) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Resolve an inherited LEAPP input transform for a data property."""
    for data_cls in type(data).__mro__:
        prop = data_cls.__dict__.get(property_name)
        if not isinstance(prop, property) or prop.fget is None:
            continue
        semantics = getattr(prop.fget, "_leapp_semantics", None)
        if semantics is not None:
            return semantics.input_transform
    return None


def _resolve_output_kind(entity: Any, method_name: str) -> str | None:
    """Resolve the LEAPP output kind inherited by an entity write method."""
    for entity_cls in type(entity).__mro__:
        method = entity_cls.__dict__.get(method_name)
        if not callable(method):
            continue
        semantics = getattr(method, "_leapp_semantics", None)
        if semantics is None:
            continue
        kind = getattr(semantics.kind, "value", semantics.kind)
        return kind if isinstance(kind, str) and kind else None
    return None


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

    def __init__(
        self,
        cfg: Any,
        leapp_yaml_path: str,
        *,
        controller_owned_write_handlers: Mapping[str, Callable[[LeappDeploymentEnv, ControllerOwnedWriteSpec], None]]
        | None = None,
    ):
        """Initialize the deployment environment.

        Args:
            cfg: A ``ManagerBasedRLEnvCfg`` (or compatible) task config.
            leapp_yaml_path: Path to the LEAPP ``.yaml`` pipeline description.
            controller_owned_write_handlers: Mapping from deployment capability
                names in the pipeline metadata to simulator-side handlers. A
                pipeline that requires an unhandled capability is rejected.
        """

        self._is_closed = True
        cfg.scene.num_envs = 1
        cfg.validate()
        self.cfg = cfg
        self._leapp_yaml_path = leapp_yaml_path
        self._sim_step_counter = 0
        self.extras: dict = {}

        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            logger.warning("Seed not set for the environment. The environment creation may not be deterministic.")

        # ── Simulation + scene ────────────────────────────────────
        if SimulationContext.instance() is not None:
            raise RuntimeError("A simulation context already exists. LeappDeploymentEnv must create and own it.")
        self.sim = SimulationContext(cfg.sim)
        try:
            if "cuda" in self.sim.device:
                torch.cuda.set_device(self.sim.device)

            with use_stage(self.sim.stage):
                self.scene = InteractiveScene(cfg.scene)
            self.sim.register_interactive_scene(self.scene)

            # EventManager must exist before simulation starts so prestartup
            # events can author the USD stage before physics handles are created.
            self.event_manager: EventManager | None = None
            if hasattr(cfg, "events") and cfg.events is not None:
                self.event_manager = EventManager(cfg.events, cast(Any, self))
                if "prestartup" in self.event_manager.available_modes:
                    self.event_manager.apply(mode="prestartup")

            with use_stage(self.sim.stage):
                self.sim.reset()
            self.scene.update(dt=self.physics_dt)
            self.sim.physics_manager.set_decimation(self.cfg.decimation)
            self._physics_handles_decimation = self.sim.physics_manager.handles_decimation()
            self.has_rtx_sensors = bool(self.sim.get_setting("/isaaclab/render/rtx_sensors"))

            # ── CommandManager (optional, for command/* inputs) ───
            self.command_manager: CommandManager | None = None
            if hasattr(cfg, "commands") and cfg.commands is not None:
                self.command_manager = CommandManager(cfg.commands, cast(Any, self))

            if self.event_manager is not None and "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

            # ── LEAPP InferenceManager ────────────────────────────
            self.inference = InferenceManager(leapp_yaml_path)

            # ── Parse YAML and resolve I/O mappings ───────────────
            with open(leapp_yaml_path) as f:
                self._leapp_desc = yaml.safe_load(f)
            self._input_mapping: dict[str, StateInputSpec | CommandInputSpec] = {}
            self._output_mapping: dict[str, WriteOutputSpec] = {}
            self._controller_owned_write_handlers = dict(controller_owned_write_handlers or {})
            self._controller_owned_write_specs = self._resolve_controller_owned_writes()
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
        except Exception:
            self.sim.clear_instance()
            raise
        self._is_closed = False

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

    def _resolve_controller_owned_writes(self) -> tuple[ControllerOwnedWriteSpec, ...]:
        """Resolve and validate controller-owned write metadata from the pipeline."""
        pipeline = self._leapp_desc.get("pipeline")
        if not isinstance(pipeline, dict):
            raise ValueError("The LEAPP description must contain a 'pipeline' mapping.")
        configs = pipeline.get("configs", {})
        if not isinstance(configs, dict):
            raise ValueError("LEAPP pipeline 'configs' must be a mapping.")
        isaaclab_cfg = configs.get("isaaclab", {})
        if not isinstance(isaaclab_cfg, dict):
            raise ValueError("LEAPP pipeline 'configs.isaaclab' must be a mapping.")
        controller_cfg = isaaclab_cfg.get("controller_owned_writes")
        if controller_cfg is None:
            requirements = []
        else:
            if (
                not isinstance(controller_cfg, dict)
                or type(controller_cfg.get("schema_version")) is not int
                or controller_cfg["schema_version"] != 1
            ):
                raise ValueError("LEAPP controller-owned writes require schema_version 1.")
            requirements = controller_cfg.get("requirements")
            if not isinstance(requirements, list):
                raise ValueError("LEAPP controller-owned write 'requirements' must be a list.")

        specs: list[ControllerOwnedWriteSpec] = []
        requirement_keys: set[tuple[str, str]] = set()
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, dict):
                raise ValueError(f"Controller-owned write requirement {index} must be a mapping.")
            capability = requirement.get("capability")
            source_term = requirement.get("source_term")
            kind = requirement.get("kind")
            cadence = requirement.get("cadence")
            connection = requirement.get("isaaclab_connection")
            if not all(isinstance(value, str) and value for value in (capability, source_term, kind, connection)):
                raise ValueError(f"Controller-owned write requirement {index} has incomplete string metadata.")
            if cadence != "action_apply":
                raise ValueError(
                    f"Controller-owned write requirement {index} has unsupported cadence '{cadence}'; "
                    "expected 'action_apply'."
                )
            connection_parts = connection.split(":")
            if len(connection_parts) != 3 or connection_parts[0] != "write" or not all(connection_parts[1:]):
                raise ValueError(f"Controller-owned write requirement {index} has invalid connection '{connection}'.")
            entity_name, method_name = connection_parts[1:]
            try:
                entity = self.scene[entity_name]
            except KeyError as exc:
                raise ValueError(
                    f"Controller-owned write requirement {index} references unknown entity '{entity_name}'."
                ) from exc
            element_names = requirement.get("element_names")
            joint_names = self._extract_joint_names(element_names, f"controller requirement {index}")
            if joint_names is None:
                raise ValueError(f"Controller-owned write requirement {index} must name every controlled joint.")
            if len(set(joint_names)) != len(joint_names):
                raise ValueError(f"Controller-owned write requirement {index} contains duplicate joint names.")
            if not hasattr(entity, "joint_names") or not hasattr(entity, "find_joints"):
                raise ValueError(
                    f"Controller-owned write requirement {index} references a non-articulation entity '{entity_name}'."
                )
            unknown_joint_names = sorted(set(joint_names) - set(entity.joint_names))
            if unknown_joint_names:
                raise ValueError(
                    f"Controller-owned write requirement {index} references unknown joints: "
                    + ", ".join(unknown_joint_names)
                )
            joint_ids, resolved_joint_names = entity.find_joints(list(joint_names), preserve_order=True)
            if list(resolved_joint_names) != list(joint_names):
                raise ValueError(f"Controller-owned write requirement {index} did not resolve joints in export order.")
            if not callable(getattr(entity, method_name, None)):
                raise ValueError(
                    f"Controller-owned write requirement {index} references unknown method '{method_name}'."
                )
            if capability == "gravity_compensation" and (
                kind != "target/joint/effort" or method_name != "set_joint_effort_target_index"
            ):
                raise ValueError(
                    "The gravity_compensation capability requires target/joint/effort semantics and "
                    "set_joint_effort_target_index."
                )
            runtime_kind = _resolve_output_kind(entity, method_name)
            if runtime_kind is None:
                raise ValueError(
                    f"Controller-owned write requirement {index} method '{method_name}' is missing or has no "
                    "LEAPP output semantics."
                )
            if kind != runtime_kind:
                raise ValueError(
                    f"Controller-owned write requirement {index} kind '{kind}' does not match runtime method "
                    f"'{method_name}' kind '{runtime_kind}'."
                )
            spec = ControllerOwnedWriteSpec(
                capability=capability,
                source_term=source_term,
                kind=kind,
                entity_name=entity_name,
                method_name=method_name,
                cadence=cadence,
                joint_names=joint_names,
                joint_ids=tuple(int(joint_id) for joint_id in joint_ids),
            )
            requirement_key = (source_term, connection)
            if requirement_key in requirement_keys:
                raise ValueError(
                    f"Controller-owned write requirement {index} duplicates source term '{source_term}' "
                    f"and connection '{connection}'."
                )
            requirement_keys.add(requirement_key)
            for existing_spec in specs:
                if (
                    spec.entity_name == existing_spec.entity_name
                    and spec.kind == existing_spec.kind
                    and set(spec.joint_names).intersection(existing_spec.joint_names)
                ):
                    raise ValueError(
                        f"Controller-owned write requirement {index} overlaps an earlier {spec.kind} requirement "
                        f"on entity '{spec.entity_name}'."
                    )
            specs.append(spec)

        expected_requirements = self._expected_controller_owned_writes()
        actual_requirements = {
            (spec.source_term, f"write:{spec.entity_name}:{spec.method_name}"): (
                spec.capability,
                spec.kind,
                spec.joint_names,
            )
            for spec in specs
        }
        if actual_requirements != expected_requirements:
            missing = sorted(set(expected_requirements) - set(actual_requirements))
            unexpected = sorted(set(actual_requirements) - set(expected_requirements))
            mismatched = sorted(
                key
                for key in set(expected_requirements).intersection(actual_requirements)
                if expected_requirements[key] != actual_requirements[key]
            )
            details = []
            for label, keys in (("missing", missing), ("unexpected", unexpected), ("mismatched", mismatched)):
                if keys:
                    formatted = ", ".join(f"{term}:{connection}" for term, connection in keys)
                    details.append(f"{label}: {formatted}")
            raise RuntimeError(
                "The LEAPP artifact's controller-owned writes do not match the task config ("
                + "; ".join(details)
                + "). Re-export the policy with the current task config."
            )

        missing_capabilities = sorted({spec.capability for spec in specs} - set(self._controller_owned_write_handlers))
        if missing_capabilities:
            raise RuntimeError(
                "The LEAPP pipeline requires controller-owned write handlers for: " + ", ".join(missing_capabilities)
            )
        invalid_handlers = sorted(
            capability
            for capability in {spec.capability for spec in specs}
            if not callable(self._controller_owned_write_handlers[capability])
        )
        if invalid_handlers:
            raise TypeError("Controller-owned write handlers must be callable for: " + ", ".join(invalid_handlers))
        return tuple(specs)

    def _expected_controller_owned_writes(self) -> dict[tuple[str, str], tuple[str, str, tuple[str, ...]]]:
        """Collect controller-owned writes declared by the task action config."""
        actions_cfg = getattr(getattr(self, "cfg", None), "actions", None)
        if actions_cfg is None:
            return {}
        expected: dict[tuple[str, str], tuple[str, str, tuple[str, ...]]] = {}
        cfg_items = actions_cfg.items() if isinstance(actions_cfg, dict) else actions_cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            methods = getattr(term_cfg, "controller_owned_write_methods", {})
            if not isinstance(methods, Mapping):
                raise TypeError(
                    f"Action config '{term_name}' controller_owned_write_methods must be a "
                    "method-to-capability mapping."
                )
            entity_name = getattr(term_cfg, "asset_name", None)
            for method_name, capability in methods.items():
                if not all(isinstance(value, str) and value for value in (entity_name, method_name, capability)):
                    raise ValueError(f"Action config '{term_name}' has incomplete controller-owned write metadata.")
                try:
                    entity = self.scene[entity_name]
                except KeyError as exc:
                    raise ValueError(f"Action config '{term_name}' references unknown entity '{entity_name}'.") from exc
                joint_selectors = getattr(term_cfg, "joint_names", None)
                if not hasattr(entity, "find_joints") or not isinstance(joint_selectors, list):
                    raise ValueError(
                        f"Action config '{term_name}' controller-owned writes require articulation joint selectors."
                    )
                _, joint_names = entity.find_joints(
                    joint_selectors,
                    preserve_order=getattr(term_cfg, "preserve_order", False),
                )
                joint_names = tuple(joint_names)
                if not joint_names or len(set(joint_names)) != len(joint_names):
                    raise ValueError(
                        f"Action config '{term_name}' controller-owned writes must resolve unique controlled joints."
                    )
                runtime_kind = _resolve_output_kind(entity, method_name)
                if runtime_kind is None:
                    raise ValueError(
                        f"Action config '{term_name}' controller-owned method '{method_name}' is missing or has no "
                        "LEAPP output semantics."
                    )
                key = (term_name, f"write:{entity_name}:{method_name}")
                requirement = (capability, runtime_kind, joint_names)
                if key in expected and expected[key] != requirement:
                    raise ValueError(f"Action config '{term_name}' declares conflicting controller capabilities.")
                expected[key] = requirement
        return expected

    @staticmethod
    def _extract_joint_names(element_names: Any, label: str) -> tuple[str, ...] | None:
        """Extract a single joint-name axis from LEAPP element metadata."""
        if element_names is None:
            return None
        if (
            not isinstance(element_names, list)
            or len(element_names) != 1
            or not isinstance(element_names[0], list)
            or not element_names[0]
            or not all(isinstance(name, str) and name for name in element_names[0])
        ):
            raise ValueError(f"{label} must define exactly one non-empty joint-name axis.")
        return tuple(element_names[0])

    def _validate_policy_output_ownership(
        self,
        output_key: str,
        connection: str,
        kind: Any,
        element_names: Any,
    ) -> None:
        """Reject policy outputs that overlap controller-owned write targets."""
        connection_parts = connection.split(":")
        if len(connection_parts) != 3 or connection_parts[0] != "write" or not all(connection_parts[1:]):
            raise ValueError(f"LEAPP output '{output_key}' has invalid connection '{connection}'.")
        entity_name, method_name = connection_parts[1:]
        try:
            entity = self.scene[entity_name]
        except KeyError as exc:
            raise ValueError(f"LEAPP output '{output_key}' references unknown entity '{entity_name}'.") from exc
        if not callable(getattr(entity, method_name, None)):
            raise ValueError(f"LEAPP output '{output_key}' references unknown method '{method_name}'.")
        runtime_kind = _resolve_output_kind(entity, method_name)
        if runtime_kind is None:
            raise ValueError(
                f"LEAPP output '{output_key}' method '{method_name}' is missing or has no LEAPP output semantics."
            )
        artifact_kind = getattr(kind, "value", kind)
        if artifact_kind != runtime_kind:
            raise ValueError(
                f"LEAPP output '{output_key}' kind '{artifact_kind}' does not match runtime method "
                f"'{method_name}' kind '{runtime_kind}'."
            )

        output_joint_names = self._extract_joint_names(element_names, f"LEAPP output '{output_key}'")
        if output_joint_names is not None and hasattr(entity, "joint_names") and hasattr(entity, "find_joints"):
            if len(set(output_joint_names)) != len(output_joint_names):
                raise ValueError(f"LEAPP output '{output_key}' contains duplicate joint names.")
            unknown_joint_names = sorted(set(output_joint_names) - set(entity.joint_names))
            if unknown_joint_names:
                raise ValueError(
                    f"LEAPP output '{output_key}' references unknown joints: " + ", ".join(unknown_joint_names)
                )
            _, resolved_joint_names = entity.find_joints(list(output_joint_names), preserve_order=True)
            if list(resolved_joint_names) != list(output_joint_names):
                raise ValueError(f"LEAPP output '{output_key}' did not resolve joints in export order.")

        for spec in self._controller_owned_write_specs:
            owned_connection = f"write:{spec.entity_name}:{spec.method_name}"
            same_connection = connection == owned_connection
            same_typed_entity = entity_name == spec.entity_name and runtime_kind == spec.kind
            if not same_connection and not same_typed_entity:
                continue
            if output_joint_names is None or set(spec.joint_names).intersection(output_joint_names):
                raise ValueError(
                    f"LEAPP output '{output_key}' overlaps controller-owned {spec.kind} write '{owned_connection}'."
                )

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
                        input_transform=_resolve_input_transform(entity.data, prop_name),
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
                    self._validate_policy_output_ownership(
                        key,
                        connection,
                        desc.get("kind"),
                        desc.get("element_names"),
                    )
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
                if spec.input_transform is not None:
                    transformed = spec.input_transform(value)
                    if transformed.shape != value.shape:
                        raise ValueError(
                            "LEAPP input transforms must preserve tensor shape: "
                            f"got {tuple(value.shape)} -> {tuple(transformed.shape)} for '{key}'."
                        )
                    value = transformed
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

    def _apply_controller_owned_writes(self) -> None:
        """Apply controller-owned writes at the action-manager application cadence."""
        for spec in self._controller_owned_write_specs:
            self._controller_owned_write_handlers[spec.capability](self, spec)

    # ── Public API ────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Reset the scene and inference state.

        Args:
            seed: Seed for reset randomization. If ``None``, preserve the
                current random-generator state.

        Returns:
            The initial input tensors (for logging / debugging).
        """
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        if seed is not None:
            self.seed(seed)

        self.scene.reset(env_ids)

        if self.event_manager is not None and "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
        if self.command_manager is not None:
            self.command_manager.reset(env_ids)
        if self.event_manager is not None:
            self.event_manager.reset(env_ids)

        self.sim.render_context.reset_scene_state_cadence()
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

        # Persistent state created by run_policy() can contain inference tensors.
        with torch.inference_mode():
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

        # 6. Physics stepping
        is_rendering = self.sim.is_rendering
        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self._apply_controller_owned_writes()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self._apply_controller_owned_writes()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)

        if self.event_manager is not None and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        return outputs

    @classmethod
    def simulated_controller_owned_write_handlers(
        cls,
    ) -> dict[str, Callable[[LeappDeploymentEnv, ControllerOwnedWriteSpec], None]]:
        """Return the controller-owned write handlers supported in simulation.

        Returns:
            Capability handlers for built-in simulation equivalents.
        """
        return {"gravity_compensation": cls.apply_simulated_gravity_compensation}

    @staticmethod
    def apply_simulated_gravity_compensation(
        env: LeappDeploymentEnv,
        spec: ControllerOwnedWriteSpec,
    ) -> None:
        """Apply model-based gravity effort for one controller-owned write.

        Args:
            env: Deployment simulation receiving the gravity write.
            spec: Validated articulation-joint write specification.

        Raises:
            ValueError: If the requirement does not describe a joint-effort write.
        """
        if spec.kind != "target/joint/effort" or spec.method_name != "set_joint_effort_target_index":
            raise ValueError(
                "The gravity_compensation capability requires a joint-effort target using "
                "set_joint_effort_target_index."
            )
        entity = env.scene[spec.entity_name]
        joint_ids = list(spec.joint_ids)
        gravity_joint_ids = [joint_id + entity.num_base_dofs for joint_id in joint_ids]
        gravity = entity.data.gravity_compensation_forces.torch[:, gravity_joint_ids]
        gravity = torch.where(torch.isfinite(gravity), gravity, torch.zeros_like(gravity))
        getattr(entity, spec.method_name)(target=gravity, joint_ids=joint_ids)

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the random seed used by deployment simulation.

        Args:
            seed: Random seed. A value of ``-1`` samples a random seed.

        Returns:
            The resolved random seed.
        """
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except (ModuleNotFoundError, AttributeError):
            pass
        return configure_seed(seed)

    def close(self):
        """Clean up the environment and release simulator-owned resources."""
        if not self._is_closed:
            self.sim.stop()
            if self.command_manager is not None:
                del self.command_manager
            if self.event_manager is not None:
                del self.event_manager
            del self.scene
            self.sim.clear_instance()
            if self._window is not None:
                self._window = None
            self._is_closed = True
