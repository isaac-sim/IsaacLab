# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime actuator collection for articulations."""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
import warp as wp
from prettytable import PrettyTable

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from . import actuator_kernels
from .actuator_base import ActuatorBase
from .actuator_base_cfg import (
    ActuatorBaseCfg,
    _is_implicit_actuator_cfg,
    _resolve_effort_limit_aliases,
    _resolve_limit_values,
)
from .actuator_control import ActuatorControl, ActuatorJointProperties
from .actuator_pd import IdealPDActuator, ImplicitActuator

logger = logging.getLogger(__name__)

_DEFAULT_JOINT_EFFORT_LIMIT = 1.0e9


class ActuatorCollection(Mapping[str, ActuatorBase]):
    """Read-only runtime collection of actuator groups for one articulation.

    Mapping entries retain their configured identity. The collection owns
    articulation-wide commands, processed commands, telemetry, and lifecycle.
    Configure membership through :attr:`isaaclab.assets.ArticulationCfg.actuators`
    before construction; assigning or deleting mapping entries raises
    :class:`TypeError`. Each joint can belong to at most one group; overlapping
    joint selections raise :class:`ValueError` during construction.

    Disjoint :class:`~isaaclab.actuators.ImplicitActuator` groups may share one
    internal execution batch; all other groups execute one group at a time.
    """

    # Initialization.

    def __init__(
        self,
        actuator_cfgs: dict[str, ActuatorBaseCfg],
        control: ActuatorControl,
        *,
        debug_value_resolution: bool = False,
    ):
        """Initialize the actuator collection.

        Args:
            actuator_cfgs: Mapping of actuator group names to actuator configs.
            control: Backend control bridge for state reads and sim writes.
            debug_value_resolution: Whether to log actuator value resolution.
        """
        self._control = control
        self._groups: dict[str, ActuatorBase] = {}
        self._groups_by_class: dict[type[ActuatorBase], list[ActuatorBase]] = {}
        self._native_group_names: set[str] = set()
        self._debug_value_resolution = debug_value_resolution
        self._joint_property_resolution_rows: dict[str, dict[str, tuple[tuple[object, ...], ...]]] = {}
        self._has_implicit_actuators = False
        self._launch_cache = _WarpLaunchCache(self.device)

        resolved_cfgs = {name: cfg.copy() for name, cfg in actuator_cfgs.items()}
        resolved_group_joints = self._resolve_group_joints(resolved_cfgs)
        for name, cfg in resolved_cfgs.items():
            self._resolve_deprecated_velocity_limit_alias(name, cfg, resolved_group_joints[name][1])
            _resolve_effort_limit_aliases(name, cfg, resolved_group_joints[name][1])

        self._allocate_buffers()
        self._command = ActuatorCommand(self)
        self._joint_command = ActuatorJointCommand(self)
        self._native_group_names = self._control.prepare_native_actuators(self, resolved_cfgs)
        self._build_groups(resolved_cfgs, resolved_group_joints)
        self._control.finalize_native_actuators(self)
        for actuator_name in self._native_group_names:
            actuator = self._groups[actuator_name]
            if isinstance(actuator, IdealPDActuator):
                actuator._bind_native_actuator_gains(self._control)
        self._validate_coverage()
        self._build_execution_batches()
        if self._debug_value_resolution:
            self._print_value_resolution_table()
        if not self._control.native_actuator_path_active:
            explicit_group_names = [
                name for name, actuator in self._groups.items() if isinstance(actuator, IdealPDActuator)
            ]
            if explicit_group_names:
                warnings.warn(
                    "Isaac Lab execution of explicit actuator models is deprecated. Use Newton actuator execution "
                    f"instead. Affected groups: {', '.join(explicit_group_names)}.",
                    DeprecationWarning,
                    stacklevel=2,
                )

    # Public interface.

    def __getitem__(self, name: str) -> ActuatorBase:
        return self._groups[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __setitem__(self, name: str, actuator: ActuatorBase) -> None:
        raise TypeError("ActuatorCollection membership is fixed after initialization.")

    def __delitem__(self, name: str) -> None:
        raise TypeError("ActuatorCollection membership is fixed after initialization.")

    @property
    def command(self) -> ActuatorCommand:
        """Commands received by the actuator models."""
        return self._command

    @property
    def joint_command(self) -> ActuatorJointCommand:
        """Processed commands produced for the simulated joints.

        This view is not submitted-command telemetry for native controllers, which
        bypass the processed-command arrays.
        """
        return self._joint_command

    @property
    def num_instances(self) -> int:
        """Number of articulation instances."""
        return self._control.num_instances

    @property
    def num_joints(self) -> int:
        """Number of articulation joints."""
        return self._control.num_joints

    @property
    def device(self) -> str:
        """Warp/Torch device string."""
        return self._control.device

    @property
    def has_implicit_actuators(self) -> bool:
        """Whether any configured actuator group is implicit."""
        return self._has_implicit_actuators

    @property
    def computed_effort(self) -> ProxyArray:
        """Joint efforts computed before clipping [N or N·m, depending on joint type]."""
        return self._computed_effort_ta

    @property
    def applied_effort(self) -> ProxyArray:
        """Joint efforts applied after clipping [N or N·m, depending on joint type]."""
        return self._applied_effort_ta

    # Lifecycle.

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset all actuator group states.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        group_env_ids = self._control._normalize_index_sequence(env_ids)
        for actuator in self._groups.values():
            actuator.reset(group_env_ids)
        self._control.reset_native_actuators(slice(None) if group_env_ids is None else group_env_ids)

    def compute(self, dt: float = 0.0) -> None:
        """Compute processed actuator commands and telemetry.

        Args:
            dt: Physics step size [s].
        """
        if self._control.compute_native_actuators(self, dt):
            return

        joint_pos = self._control.joint_pos
        joint_vel = self._control.joint_vel
        for batch in self._execution_batches:
            actuator = batch.actuator
            if type(actuator) is ImplicitActuator:
                self._compute_implicit_batch(batch)
                continue
            joint_indices = actuator.joint_indices
            control_action = ArticulationActions(
                joint_positions=self.command.position.torch[:, joint_indices],
                joint_velocities=self.command.velocity.torch[:, joint_indices],
                joint_efforts=self.command.effort.torch[:, joint_indices],
                joint_indices=joint_indices,
            )
            control_action = actuator.compute(
                control_action,
                joint_pos=joint_pos.torch[:, joint_indices],
                joint_vel=joint_vel.torch[:, joint_indices],
            )
            self._scatter_actuator_output(actuator, control_action, batch.joint_indices_wp)

    def submit_commands(self) -> None:
        """Submit processed actuator command buffers through the backend control object."""
        self._control.submit_commands(self)

    # Construction and property resolution.

    def _allocate_buffers(self) -> None:
        """Allocate articulation-wide command and telemetry buffers."""
        shape = (self.num_instances, self.num_joints)
        self._joint_pos_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_pos_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._computed_effort = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._applied_effort = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._all_joint_ids = wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)

        self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
        self._joint_pos_target_sim_ta = ProxyArray(self._joint_pos_target_sim)
        self._joint_vel_target_sim_ta = ProxyArray(self._joint_vel_target_sim)
        self._joint_effort_target_sim_ta = ProxyArray(self._joint_effort_target_sim)
        self._computed_effort_ta = ProxyArray(self._computed_effort)
        self._applied_effort_ta = ProxyArray(self._applied_effort)

    def _resolve_group_joints(
        self, actuator_cfgs: dict[str, ActuatorBaseCfg]
    ) -> dict[str, tuple[list[int] | ProxyArray, list[str]]]:
        """Resolve group selectors and reject joints assigned to multiple groups."""
        resolved: dict[str, tuple[list[int] | ProxyArray, list[str]]] = {}
        joint_owners: dict[str, str] = {}
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            joint_ids, joint_names = self._control.find_joints(actuator_cfg.joint_names_expr)
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            for joint_name in joint_names:
                owner = joint_owners.get(joint_name)
                if owner is not None and owner != actuator_name:
                    raise ValueError(
                        f"Joint '{joint_name}' is assigned to multiple actuator groups: '{owner}' and"
                        f" '{actuator_name}'."
                    )
                joint_owners[joint_name] = actuator_name
            resolved[actuator_name] = (joint_ids, joint_names)
        return resolved

    def _resolve_deprecated_velocity_limit_alias(
        self, actuator_name: str, cfg: ActuatorBaseCfg, joint_names: list[str]
    ) -> None:
        """Resolve deprecated solver-limit aliases on a copied actuator config."""
        for canonical_name, alias_name in (("joint_velocity_limit", "velocity_limit_sim"),):
            canonical_value = getattr(cfg, canonical_name)
            alias_value = getattr(cfg, alias_name)
            if alias_value is None:
                continue

            warnings.warn(
                f"Actuator group '{actuator_name}' uses deprecated '{alias_name}'. Use "
                f"'{canonical_name}' instead; '{alias_name}' will be removed in 4.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            if canonical_value is None:
                setattr(cfg, canonical_name, alias_value)
            elif _resolve_limit_values(canonical_value, joint_names) != _resolve_limit_values(alias_value, joint_names):
                raise ValueError(
                    f"Actuator group '{actuator_name}' has conflicting '{canonical_name}' and "
                    f"deprecated '{alias_name}' values."
                )

    def _build_groups(
        self,
        actuator_cfgs: dict[str, ActuatorBaseCfg],
        resolved_group_joints: dict[str, tuple[list[int] | ProxyArray, list[str]]],
    ) -> None:
        """Construct actuator groups and apply their resolved joint properties."""
        construction_records: list[tuple[ActuatorJointProperties, torch.Tensor | slice, bool, bool]] = []
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            joint_ids, joint_names = resolved_group_joints[actuator_name]
            if len(joint_names) == self.num_joints:
                actuator_joint_ids: slice | torch.Tensor = slice(None)
            elif isinstance(joint_ids, ProxyArray):
                actuator_joint_ids = joint_ids.torch
            else:
                actuator_joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.int32)

            defaults = self._control.get_default_joint_properties(actuator_joint_ids)
            implicit = _is_implicit_actuator_cfg(actuator_cfg)
            properties, resolution_rows = self._resolve_joint_properties(
                actuator_cfg,
                defaults,
                joint_names,
                actuator_joint_ids,
                implicit=implicit,
            )
            actuator_kwargs = dict(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=actuator_joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=properties.stiffness,
                damping=properties.damping,
                velocity_limit=properties.velocity_limit,
            )
            if implicit:
                effort_limit_name = "joint_effort_limit"
                effort_limit_value = properties.joint_effort_limit
            else:
                effort_limit_name = "actuator_effort_limit"
                effort_limit_value = defaults.joint_effort_limit
            constructor_parameters = inspect.signature(actuator_cfg.class_type.__init__).parameters
            if effort_limit_name not in constructor_parameters and "effort_limit" in constructor_parameters:
                warnings.warn(
                    f"The constructor for actuator class '{actuator_cfg.class_type.__name__}' uses the deprecated "
                    f"'effort_limit' parameter. Rename it to '{effort_limit_name}'; 'effort_limit' support will be "
                    "removed in 4.0.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                effort_limit_name = "effort_limit"
            actuator_kwargs[effort_limit_name] = effort_limit_value
            actuator: ActuatorBase = actuator_cfg.class_type(**actuator_kwargs)
            self._groups[actuator_name] = actuator
            self._groups_by_class.setdefault(type(actuator), []).append(actuator)
            self._has_implicit_actuators = self._has_implicit_actuators or isinstance(actuator, ImplicitActuator)
            if self._debug_value_resolution:
                self._joint_property_resolution_rows[actuator_name] = resolution_rows
            construction_records.append(
                (
                    properties,
                    actuator_joint_ids,
                    implicit,
                    actuator_name in self._native_group_names,
                )
            )

        for properties, joint_ids, implicit, native_managed in construction_records:
            self._control.write_resolved_joint_properties(
                properties,
                joint_ids,
                implicit=implicit,
                native_managed=native_managed,
            )
        for actuator in self._groups.values():
            if isinstance(actuator, ImplicitActuator):
                actuator._bind_joint_drive(self._control)

    def _resolve_joint_properties(
        self,
        cfg: ActuatorBaseCfg,
        defaults: ActuatorJointProperties,
        joint_names: list[str],
        joint_ids: torch.Tensor | slice,
        *,
        implicit: bool,
    ) -> tuple[ActuatorJointProperties, dict[str, tuple[tuple[object, ...], ...]]]:
        """Resolve fresh construction-only joint properties for one actuator group."""
        effort_default = (
            defaults.joint_effort_limit
            if implicit
            else torch.full_like(
                defaults.joint_effort_limit,
                _DEFAULT_JOINT_EFFORT_LIMIT,
            )
        )
        values: dict[str, torch.Tensor] = {}
        resolution_rows: dict[str, tuple[tuple[object, ...], ...]] = {}
        for property_name, cfg_name, default_value in (
            ("stiffness", "stiffness", defaults.stiffness),
            ("damping", "damping", defaults.damping),
            ("armature", "armature", defaults.armature),
            ("friction", "friction", defaults.friction),
            ("dynamic_friction", "dynamic_friction", defaults.dynamic_friction),
            ("viscous_friction", "viscous_friction", defaults.viscous_friction),
            ("effort_limit", "joint_effort_limit", effort_default),
            ("velocity_limit", "joint_velocity_limit", defaults.velocity_limit),
        ):
            cfg_value = getattr(cfg, cfg_name)
            value = self._resolve_joint_property(cfg_value, default_value, joint_names)
            values[property_name] = value
            if self._debug_value_resolution:
                rows = self._joint_property_resolution_rows_for(
                    cfg_value,
                    value,
                    default_value,
                    joint_names,
                    joint_ids,
                )
                if rows:
                    resolution_rows[cfg_name] = rows

        return (
            ActuatorJointProperties(
                stiffness=values["stiffness"],
                damping=values["damping"],
                armature=values["armature"],
                friction=values["friction"],
                dynamic_friction=values["dynamic_friction"],
                viscous_friction=values["viscous_friction"],
                joint_effort_limit=values["effort_limit"],
                velocity_limit=values["velocity_limit"],
            ),
            resolution_rows,
        )

    def _resolve_joint_property(
        self,
        cfg_value: float | dict[str, float] | None,
        default_value: torch.Tensor,
        joint_names: list[str],
    ) -> torch.Tensor:
        """Resolve one fresh group-shaped joint property from config and authored defaults."""
        if cfg_value is None:
            return default_value.clone()
        if isinstance(cfg_value, (float, int)):
            return torch.full_like(default_value, float(cfg_value))
        if isinstance(cfg_value, dict):
            value = torch.zeros_like(default_value)
            indices, _, parsed_values = string_utils.resolve_matching_names_values(cfg_value, joint_names)
            value[:, indices] = torch.tensor(parsed_values, dtype=torch.float32, device=self.device)
            return value
        raise TypeError(
            f"Invalid type for parameter value: {type(cfg_value)} for actuator on joints {joint_names}. "
            "Expected float or dict."
        )

    def _joint_property_resolution_rows_for(
        self,
        cfg_value: float | dict[str, float] | None,
        value: torch.Tensor,
        default_value: torch.Tensor,
        joint_names: list[str],
        joint_ids: torch.Tensor | slice,
    ) -> tuple[tuple[object, ...], ...]:
        """Format construction-time joint-property resolution rows without retaining tensors."""
        if cfg_value is not None and torch.allclose(value, default_value):
            return ()
        if isinstance(joint_ids, slice):
            ids = range(self.num_joints)
        else:
            ids = tuple(int(joint_id) for joint_id in joint_ids.tolist())
        return tuple(
            (
                name,
                ids[index],
                float(default_value[0, index]),
                "Not Specified" if cfg_value is None else float(value[0, index]),
                float(default_value[0, index]) if cfg_value is None else float(value[0, index]),
            )
            for index, name in enumerate(joint_names)
        )

    # Execution planning and runtime.

    def _joint_indices_as_wp(self, actuator: ActuatorBase) -> wp.array(dtype=wp.int32):
        """Return an actuator group's joint indices as a Warp int32 array."""
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return self._all_joint_ids
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            return joint_indices
        return wp.from_torch(joint_indices.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)

    def _joint_indices_as_torch(self, actuator: ActuatorBase) -> torch.Tensor:
        """Return an actuator group's joint indices as a contiguous Torch int32 tensor."""
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return torch.arange(self.num_joints, dtype=torch.int32, device=self.device)
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            joint_indices = wp.to_torch(joint_indices)
        return joint_indices.to(self.device, dtype=torch.int32).contiguous()

    def _make_group_batch(self, name: str, group: ActuatorBase) -> _ExecutionBatch:
        """Create the execution batch for one logical actuator group."""
        joint_indices = self._joint_indices_as_torch(group)
        batch = _ExecutionBatch(
            actuator=group,
            group_names=(name,),
            joint_indices_wp=wp.from_torch(joint_indices, dtype=wp.int32),
        )
        if type(group) is ImplicitActuator:
            self._bind_implicit_kernel_arrays(batch)
        return batch

    def _make_implicit_batch(self, names: tuple[str, ...], groups: tuple[ActuatorBase, ...]) -> _ExecutionBatch:
        """Create one shared execution batch for disjoint implicit actuator groups."""
        joint_indices = torch.cat([self._joint_indices_as_torch(group) for group in groups])
        executor = ImplicitActuator._build_execution_actuator(groups, joint_indices)
        batch = _ExecutionBatch(
            actuator=executor,
            group_names=names,
            joint_indices_wp=wp.from_torch(joint_indices, dtype=wp.int32),
        )
        self._bind_implicit_kernel_arrays(batch)
        self._bind_execution_batch_parameters(batch, groups)
        return batch

    def _bind_implicit_kernel_arrays(self, batch: _ExecutionBatch) -> None:
        """Assemble the implicit kernel argument arrays for one execution batch.

        Existing argument lists are updated in place so holders of the list
        objects observe rebound backend state.
        """
        executor = batch.actuator
        inputs = [
            self._joint_pos_target,
            self._joint_vel_target,
            self._joint_effort_target,
            self._control.joint_pos.warp,
            self._control.joint_vel.warp,
            self._control.joint_stiffness.warp,
            self._control.joint_damping.warp,
            self._control.joint_effort_limits.warp,
            wp.from_torch(executor.velocity_limit, dtype=wp.float32),
            batch.joint_indices_wp,
        ]
        outputs = [
            wp.from_torch(executor.computed_effort, dtype=wp.float32),
            wp.from_torch(executor.applied_effort, dtype=wp.float32),
            self._joint_pos_target_sim,
            self._joint_vel_target_sim,
            self._joint_effort_target_sim,
            self._computed_effort,
            self._applied_effort,
            self._soft_joint_vel_limits,
        ]
        if batch.implicit_inputs is None:
            batch.implicit_inputs = inputs
            batch.implicit_outputs = outputs
        else:
            batch.implicit_inputs[:] = inputs
            batch.implicit_outputs[:] = outputs

    def _build_execution_batches(self) -> None:
        """Build execution batches in actuator configuration order."""
        native_actuator_path_active = self._control.native_actuator_path_active
        batch_by_group: dict[str, _ExecutionBatch] = {}
        for actuator_type in self._groups_by_class:
            names = tuple(name for name, group in self._groups.items() if type(group) is actuator_type)
            groups = tuple(self._groups[name] for name in names)

            if native_actuator_path_active or actuator_type is not ImplicitActuator or len(groups) < 2:
                for name, group in zip(names, groups):
                    batch_by_group[name] = self._make_group_batch(name, group)
                continue

            batch = self._make_implicit_batch(names, groups)
            for name in names:
                batch_by_group[name] = batch

        # Restore configuration order and emit each shared batch once.
        seen: set[int] = set()
        self._execution_batches = []
        for name in self._groups:
            batch = batch_by_group[name]
            if id(batch) not in seen:
                self._execution_batches.append(batch)
                seen.add(id(batch))

    def _bind_execution_batch_parameters(
        self,
        batch: _ExecutionBatch,
        groups: Sequence[ActuatorBase],
    ) -> None:
        """Bind logical implicit group tensors to slices of their shared executor tensors."""
        tensor_names = ("velocity_limit", "computed_effort", "applied_effort")
        bindings: list[tuple[ActuatorBase, str, torch.Tensor]] = []
        start = 0
        for group in groups:
            group_slice = slice(start, start + group.num_joints)
            start += group.num_joints
            for name in tensor_names:
                original = getattr(group, name)
                view = getattr(batch.actuator, name)[:, group_slice]
                if view.shape != original.shape or view.dtype != original.dtype or view.device != original.device:
                    raise ValueError(f"Execution batch view for '{name}' is incompatible with its logical group.")
                bindings.append((group, name, view))

        for group, name, view in bindings:
            setattr(group, name, view)

    def _compute_implicit_batch(self, batch: _ExecutionBatch) -> None:
        """Run one implicit actuator batch through the cached Warp launch."""
        if batch.implicit_inputs is None or batch.implicit_outputs is None:
            raise RuntimeError("Implicit actuator execution batch was not initialized.")
        self._launch_cache.launch(
            ("implicit", id(batch)),
            actuator_kernels.compute_implicit_actuator_batch,
            dim=(self.num_instances, batch.joint_indices_wp.shape[0]),
            inputs=batch.implicit_inputs,
            outputs=batch.implicit_outputs,
        )

    def _rebind_state_inputs(self) -> None:
        """Rebind implicit execution batches after backend state storage is replaced.

        Per-group explicit execution reads backend state through the control
        object on every :meth:`compute` call, so only the cached implicit
        launches hold state references that need rebinding.
        """
        for batch in self._execution_batches:
            if batch.implicit_inputs is None:
                continue
            self._bind_implicit_kernel_arrays(batch)
            self._launch_cache.clear(("implicit", id(batch)))

    def _scatter_actuator_output(
        self,
        actuator: ActuatorBase,
        control_action: ArticulationActions,
        joint_indices: wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Publish one explicit actuator's processed commands and telemetry."""
        if joint_indices is None:
            joint_indices = self._joint_indices_as_wp(actuator)
        inputs = [
            control_action.joint_positions,
            control_action.joint_velocities,
            control_action.joint_efforts,
            actuator.computed_effort,
            actuator.applied_effort,
            actuator.velocity_limit,
            joint_indices,
        ]
        outputs = [
            self._joint_pos_target_sim,
            self._joint_vel_target_sim,
            self._joint_effort_target_sim,
            self._computed_effort,
            self._applied_effort,
            self._soft_joint_vel_limits,
        ]
        wp.launch(
            actuator_kernels.scatter_explicit_actuator_outputs,
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=inputs,
            outputs=outputs,
            device=self.device,
        )

    def _write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write native-controller gains through the backend control bridge."""
        values_snapshot = values.to(self.device, dtype=torch.float32).contiguous().clone()
        self._control.write_native_actuator_gain(attr, values_snapshot, env_ids, joint_ids)

    # Diagnostics.

    def _validate_coverage(self) -> None:
        """Warn when actuator groups do not cover the expected movable joints."""
        if self.num_joints == 0:
            return
        total_act_joints = sum(actuator.num_joints for actuator in self._groups.values())
        expected_joints = self.num_joints - self._control.num_fixed_tendons
        if total_act_joints != expected_joints:
            logger.warning(
                "Actuator groups cover %s joints; expected %s after accounting for fixed tendons.",
                total_act_joints,
                expected_joints,
            )

    def _print_value_resolution_table(self) -> None:
        """Log construction-time differences between authored and configured values."""
        table = PrettyTable(["Group", "Property", "Name", "ID", "USD Value", "ActuatorCfg Value", "Applied"])
        for actuator_group in self._groups:
            group_count = 0
            for property_name, resolution_details in self._joint_property_resolution_rows[actuator_group].items():
                for prop_idx, resolution_detail in enumerate(resolution_details):
                    actuator_group_str = actuator_group if group_count == 0 else ""
                    property_str = property_name if prop_idx == 0 else ""
                    fmt = [f"{value:.2e}" if isinstance(value, float) else str(value) for value in resolution_detail]
                    table.add_row([actuator_group_str, property_str, *fmt])
                    group_count += 1
        logger.warning("\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n%s", table)


@dataclass
class _ExecutionBatch:
    actuator: ActuatorBase
    group_names: tuple[str, ...]
    joint_indices_wp: wp.array(dtype=wp.int32)
    implicit_inputs: list[wp.array(dtype=wp.float32) | wp.array(dtype=wp.int32)] | None = None
    implicit_outputs: list[wp.array(dtype=wp.float32)] | None = None


class ActuatorCommand:
    """Commands received by the actuator models.

    Position and velocity commands use joint-side coordinates. All command
    arrays are indexed by articulation joint, not by motor shaft.

    Index selectors must contain unique environment and joint indices. Repeated
    indices dispatch concurrent writes to the same destination and produce an
    undefined result. Deduplicate selectors or use mask setters.
    """

    def __init__(self, collection: ActuatorCollection) -> None:
        """Initialize the command view.

        Args:
            collection: Owning actuator collection.
        """
        self._collection = collection

    @property
    def position(self) -> ProxyArray:
        """Desired positions [m or rad, depending on joint type]."""
        return self._collection._joint_pos_target_ta

    @property
    def velocity(self) -> ProxyArray:
        """Desired velocities [m/s or rad/s, depending on joint type]."""
        return self._collection._joint_vel_target_ta

    @property
    def effort(self) -> ProxyArray:
        """Effort commands [N or N·m, depending on joint type]."""
        return self._collection._joint_effort_target_ta

    def set_position_index(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set desired positions using indices.

        Args:
            value: Desired positions [m or rad, depending on joint type]. Shape is
                ``(len(env_ids), len(joint_ids))``, or ``(num_instances, num_joints)`` when
                :paramref:`full_data` is true.
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether :paramref:`value` is a full articulation command buffer.
        """
        collection = self._collection
        env_ids_resolved = collection._control.resolve_env_ids(env_ids)
        joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            value,
            env_ids_resolved,
            joint_ids_resolved,
            collection._joint_pos_target,
            full_data=full_data,
            command_name="position",
        )

    def set_velocity_index(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set desired velocities using indices.

        Args:
            value: Desired velocities [m/s or rad/s, depending on joint type]. Shape is
                ``(len(env_ids), len(joint_ids))``, or ``(num_instances, num_joints)`` when
                :paramref:`full_data` is true.
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether :paramref:`value` is a full articulation command buffer.
        """
        collection = self._collection
        env_ids_resolved = collection._control.resolve_env_ids(env_ids)
        joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            value,
            env_ids_resolved,
            joint_ids_resolved,
            collection._joint_vel_target,
            full_data=full_data,
            command_name="velocity",
        )

    def set_effort_index(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set effort commands using indices.

        Args:
            value: Effort commands [N or N·m, depending on joint type]. Shape is
                ``(len(env_ids), len(joint_ids))``, or ``(num_instances, num_joints)`` when
                :paramref:`full_data` is true.
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether :paramref:`value` is a full articulation command buffer.
        """
        collection = self._collection
        env_ids_resolved = collection._control.resolve_env_ids(env_ids)
        joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            value,
            env_ids_resolved,
            joint_ids_resolved,
            collection._joint_effort_target,
            full_data=full_data,
            command_name="effort",
        )

    def set_position_mask(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_mask: wp.array(dtype=wp.bool) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set desired positions using masks.

        Args:
            value: Full articulation position commands [m or rad, depending on joint type]. Shape is
                ``(num_instances, num_joints)``.
            joint_mask: Joint selection mask. Defaults to all joints.
            env_mask: Environment selection mask. Defaults to all environments.
        """
        collection = self._collection
        env_mask_resolved = collection._control.resolve_env_mask(env_mask)
        joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
        self._write_mask_target(
            value,
            env_mask_resolved,
            joint_mask_resolved,
            collection._joint_pos_target,
            command_name="position",
        )

    def set_velocity_mask(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_mask: wp.array(dtype=wp.bool) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set desired velocities using masks.

        Args:
            value: Full articulation velocity commands [m/s or rad/s, depending on joint type]. Shape is
                ``(num_instances, num_joints)``.
            joint_mask: Joint selection mask. Defaults to all joints.
            env_mask: Environment selection mask. Defaults to all environments.
        """
        collection = self._collection
        env_mask_resolved = collection._control.resolve_env_mask(env_mask)
        joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
        self._write_mask_target(
            value,
            env_mask_resolved,
            joint_mask_resolved,
            collection._joint_vel_target,
            command_name="velocity",
        )

    def set_effort_mask(
        self,
        *,
        value: torch.Tensor | wp.array(dtype=wp.float32),
        joint_mask: wp.array(dtype=wp.bool) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set effort commands using masks.

        Args:
            value: Full articulation effort commands [N or N·m, depending on joint type]. Shape is
                ``(num_instances, num_joints)``.
            joint_mask: Joint selection mask. Defaults to all joints.
            env_mask: Environment selection mask. Defaults to all environments.
        """
        collection = self._collection
        env_mask_resolved = collection._control.resolve_env_mask(env_mask)
        joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
        self._write_mask_target(
            value,
            env_mask_resolved,
            joint_mask_resolved,
            collection._joint_effort_target,
            command_name="effort",
        )

    def _write_index_target(
        self,
        target: torch.Tensor | wp.array(dtype=wp.float32),
        env_ids: torch.Tensor | wp.array,
        joint_ids: torch.Tensor | wp.array,
        target_buffer: wp.array(dtype=wp.float32),
        *,
        full_data: bool,
        command_name: str,
    ) -> None:
        collection = self._collection
        expected_shape = (
            (collection.num_instances, collection.num_joints) if full_data else (env_ids.shape[0], joint_ids.shape[0])
        )
        collection._control.assert_shape_and_dtype(target, expected_shape, wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids, joint_ids),
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids, full_data],
            outputs=[target_buffer],
            device=collection.device,
        )
        collection._control.stage_user_command(command_name, collection, env_ids, joint_ids, None, None)

    def _write_mask_target(
        self,
        target: torch.Tensor | wp.array(dtype=wp.float32),
        env_mask: wp.array(dtype=wp.bool),
        joint_mask: wp.array(dtype=wp.bool),
        target_buffer: wp.array(dtype=wp.float32),
        *,
        command_name: str,
    ) -> None:
        collection = self._collection
        collection._control.assert_shape_and_dtype_mask(target, (env_mask, joint_mask), wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_mask,
            dim=(env_mask.shape[0], joint_mask.shape[0]),
            inputs=[target, env_mask, joint_mask],
            outputs=[target_buffer],
            device=collection.device,
        )
        collection._control.stage_user_command(command_name, collection, None, None, env_mask, joint_mask)


class ActuatorJointCommand:
    """Processed commands produced for the simulated joints.

    These arrays contain submitted-command telemetry for Isaac Lab-managed
    actuator models. Native controllers bypass the arrays, so they do not
    provide submitted-command telemetry on a native path.
    """

    def __init__(self, collection: ActuatorCollection) -> None:
        """Initialize the joint command view.

        Args:
            collection: Owning actuator collection.
        """
        self._collection = collection

    @property
    def position(self) -> ProxyArray:
        """Processed position commands [m or rad, depending on joint type]."""
        return self._collection._joint_pos_target_sim_ta

    @property
    def velocity(self) -> ProxyArray:
        """Processed velocity commands [m/s or rad/s, depending on joint type]."""
        return self._collection._joint_vel_target_sim_ta

    @property
    def effort(self) -> ProxyArray:
        """Processed effort commands [N or N·m, depending on joint type]."""
        return self._collection._joint_effort_target_sim_ta
