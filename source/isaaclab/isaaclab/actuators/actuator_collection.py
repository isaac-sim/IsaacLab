# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime actuator collection for articulations."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAliasType

import torch
import warp as wp
from prettytable import PrettyTable

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

from . import actuator_kernels
from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_control import ActuatorControl, ActuatorJointProperties
from .actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator

logger = logging.getLogger(__name__)

_WarpInt32 = TypeAliasType("_WarpInt32", wp.array(dtype=wp.int32))
_WarpInt64 = TypeAliasType("_WarpInt64", wp.array(dtype=wp.int64))
_WarpIndex = TypeAliasType("_WarpIndex", _WarpInt32 | _WarpInt64)


class ActuatorCollection(Mapping[str, ActuatorBase]):
    """Read-only runtime collection of actuator groups for one articulation.

    Mapping entries retain their configured identity. The collection owns
    articulation-wide commands, processed commands, telemetry, and lifecycle.
    Configure membership through :attr:`isaaclab.assets.ArticulationCfg.actuators`
    before construction; assigning or deleting mapping entries raises
    :class:`TypeError`. Each joint can belong to at most one group; overlapping
    joint selections raise :class:`ValueError` during construction.
    """

    @dataclass
    class _ExecutionBatch:
        actuator: ActuatorBase
        group_names: tuple[str, ...]
        group_slices: tuple[slice, ...]
        joint_indices: torch.Tensor
        joint_indices_wp: wp.array(dtype=wp.int32)
        implicit_inputs: list[wp.array(dtype=wp.float32) | wp.array(dtype=wp.int32)] | None = None
        implicit_outputs: list[wp.array(dtype=wp.float32)] | None = None
        control_action: ArticulationActions | None = None
        joint_pos: torch.Tensor | None = None
        joint_vel: torch.Tensor | None = None
        gather_inputs: list[wp.array(dtype=wp.float32) | wp.array(dtype=wp.int32)] | None = None
        gather_outputs: list[wp.array(dtype=wp.float32)] | None = None

    class Command:
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
            joint_ids: Sequence[int] | torch.Tensor | _WarpIndex | None = None,
            env_ids: Sequence[int] | torch.Tensor | _WarpIndex | None = None,
            full_data: bool = False,
        ) -> None:
            """Set desired positions using indices.

            Args:
                value: Desired positions [m or rad, depending on joint type].
                joint_ids: Joint indices. Defaults to all joints.
                env_ids: Environment indices. Defaults to all environments.
                full_data: Whether :paramref:`value` is a full articulation command buffer.
            """
            collection = self._collection
            env_ids_resolved = collection._control.resolve_env_ids(env_ids)
            joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
            collection._write_index_target(
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
            joint_ids: Sequence[int] | torch.Tensor | _WarpIndex | None = None,
            env_ids: Sequence[int] | torch.Tensor | _WarpIndex | None = None,
            full_data: bool = False,
        ) -> None:
            """Set desired velocities using indices.

            Args:
                value: Desired velocities [m/s or rad/s, depending on joint type].
                joint_ids: Joint indices. Defaults to all joints.
                env_ids: Environment indices. Defaults to all environments.
                full_data: Whether :paramref:`value` is a full articulation command buffer.
            """
            collection = self._collection
            env_ids_resolved = collection._control.resolve_env_ids(env_ids)
            joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
            collection._write_index_target(
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
            joint_ids: Sequence[int] | torch.Tensor | _WarpIndex | None = None,
            env_ids: Sequence[int] | torch.Tensor | _WarpIndex | None = None,
            full_data: bool = False,
        ) -> None:
            """Set effort commands using indices.

            Args:
                value: Effort commands [N or N·m, depending on joint type].
                joint_ids: Joint indices. Defaults to all joints.
                env_ids: Environment indices. Defaults to all environments.
                full_data: Whether :paramref:`value` is a full articulation command buffer.
            """
            collection = self._collection
            env_ids_resolved = collection._control.resolve_env_ids(env_ids)
            joint_ids_resolved = collection._control.resolve_joint_ids(joint_ids)
            collection._write_index_target(
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
                value: Full articulation position commands [m or rad, depending on joint type].
                joint_mask: Joint selection mask. Defaults to all joints.
                env_mask: Environment selection mask. Defaults to all environments.
            """
            collection = self._collection
            env_mask_resolved = collection._control.resolve_env_mask(env_mask)
            joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
            collection._write_mask_target(
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
                value: Full articulation velocity commands [m/s or rad/s, depending on joint type].
                joint_mask: Joint selection mask. Defaults to all joints.
                env_mask: Environment selection mask. Defaults to all environments.
            """
            collection = self._collection
            env_mask_resolved = collection._control.resolve_env_mask(env_mask)
            joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
            collection._write_mask_target(
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
                value: Full articulation effort commands [N or N·m, depending on joint type].
                joint_mask: Joint selection mask. Defaults to all joints.
                env_mask: Environment selection mask. Defaults to all environments.
            """
            collection = self._collection
            env_mask_resolved = collection._control.resolve_env_mask(env_mask)
            joint_mask_resolved = collection._control.resolve_joint_mask(joint_mask)
            collection._write_mask_target(
                value,
                env_mask_resolved,
                joint_mask_resolved,
                collection._joint_effort_target,
                command_name="effort",
            )

    class JointCommand:
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
        self._joint_property_resolution_rows: dict[str, dict[str, tuple[tuple[object, ...], ...]]] = {}
        self._has_implicit_actuators = False
        self._launch_cache = _WarpLaunchCache(self.device)

        resolved_cfgs = {name: cfg.copy() for name, cfg in actuator_cfgs.items()}
        for name, cfg in resolved_cfgs.items():
            self._resolve_deprecated_limit_aliases(name, cfg)
            self._resolve_implicit_effort_limit_alias(name, cfg)

        resolved_group_joints = self._resolve_group_joints(resolved_cfgs)
        self._allocate_buffers()
        self._command = self.Command(self)
        self._joint_command = self.JointCommand(self)
        self._native_group_names = self._control.prepare_native_actuators(self, resolved_cfgs)
        self._build_groups(resolved_cfgs, resolved_group_joints)
        self._control.finalize_native_actuators(self)
        self._validate_coverage()
        self._build_execution_batches()
        if debug_value_resolution:
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

    """
    Mapping interface.
    """

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

    """
    Properties.
    """

    @property
    def command(self) -> Command:
        """Commands received by the actuator models."""
        return self._command

    @property
    def joint_command(self) -> JointCommand:
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
    def computed_torque(self) -> ProxyArray:
        """Joint torques computed before clipping [N or N·m, depending on joint type]."""
        return self._computed_torque_ta

    @property
    def applied_torque(self) -> ProxyArray:
        """Joint torques applied after clipping [N or N·m, depending on joint type]."""
        return self._applied_torque_ta

    """
    Operations.
    """

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
            if batch.control_action is not None:
                self._gather_explicit_batch(batch)
                control_action = batch.control_action
                command_pos = control_action.joint_positions
                command_vel = control_action.joint_velocities
                command_effort = control_action.joint_efforts
                control_action = actuator.compute(
                    control_action,
                    joint_pos=batch.joint_pos,
                    joint_vel=batch.joint_vel,
                )
                self._scatter_actuator_output(actuator, control_action, batch.joint_indices_wp)
                control_action.joint_positions = command_pos
                control_action.joint_velocities = command_vel
                control_action.joint_efforts = command_effort
                continue
            joint_indices = actuator.joint_indices if len(batch.group_names) == 1 else batch.joint_indices
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

    """
    Internal helpers.
    """

    def _allocate_buffers(self) -> None:
        shape = (self.num_instances, self.num_joints)
        self._joint_pos_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_pos_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_vel_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._joint_effort_target_sim = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._computed_torque = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._applied_torque = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._gear_ratio = wp.ones(shape, dtype=wp.float32, device=self.device)
        self._all_joint_ids = wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)

        self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
        self._joint_pos_target_sim_ta = ProxyArray(self._joint_pos_target_sim)
        self._joint_vel_target_sim_ta = ProxyArray(self._joint_vel_target_sim)
        self._joint_effort_target_sim_ta = ProxyArray(self._joint_effort_target_sim)
        self._computed_torque_ta = ProxyArray(self._computed_torque)
        self._applied_torque_ta = ProxyArray(self._applied_torque)

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

    def _resolve_deprecated_limit_aliases(self, actuator_name: str, cfg: ActuatorBaseCfg) -> None:
        """Resolve deprecated solver-limit aliases on a copied actuator config."""
        for canonical_name, alias_name in (
            ("joint_effort_limit", "effort_limit_sim"),
            ("joint_velocity_limit", "velocity_limit_sim"),
        ):
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
            elif canonical_value != alias_value:
                raise ValueError(
                    f"Actuator group '{actuator_name}' has conflicting '{canonical_name}' and "
                    f"deprecated '{alias_name}' values."
                )

    def _resolve_implicit_effort_limit_alias(self, actuator_name: str, cfg: ActuatorBaseCfg) -> None:
        """Retain the implicit ``effort_limit`` compatibility alias on a copied config."""
        if not self._is_implicit_cfg(cfg) or cfg.effort_limit is None:
            return
        if cfg.joint_effort_limit is None:
            cfg.joint_effort_limit = cfg.effort_limit
        elif cfg.joint_effort_limit != cfg.effort_limit:
            raise ValueError(
                f"Implicit actuator group '{actuator_name}' has conflicting 'joint_effort_limit' and "
                "'effort_limit' values. Use 'joint_effort_limit' for the solver limit."
            )

    @staticmethod
    def _is_implicit_cfg(cfg: ActuatorBaseCfg) -> bool:
        """Return whether an actuator configuration constructs an implicit model."""
        class_type = cfg.class_type
        return (
            "ImplicitActuator" in class_type
            if isinstance(class_type, str)
            else issubclass(class_type, ImplicitActuator)
        )

    def _build_groups(
        self,
        actuator_cfgs: dict[str, ActuatorBaseCfg],
        resolved_group_joints: dict[str, tuple[list[int] | ProxyArray, list[str]]],
    ) -> None:
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
            implicit = self._is_implicit_cfg(actuator_cfg)
            properties, resolution_rows = self._resolve_joint_properties(
                actuator_cfg,
                defaults,
                joint_names,
                actuator_joint_ids,
                implicit=implicit,
            )
            cfg = actuator_cfg.copy()
            actuator: ActuatorBase = cfg.class_type(
                cfg=cfg,
                joint_names=joint_names,
                joint_ids=actuator_joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=properties.stiffness,
                damping=properties.damping,
                effort_limit=properties.effort_limit if implicit else defaults.effort_limit,
                velocity_limit=properties.velocity_limit,
            )
            self._groups[actuator_name] = actuator
            self._groups_by_class.setdefault(type(actuator), []).append(actuator)
            self._has_implicit_actuators = self._has_implicit_actuators or isinstance(actuator, ImplicitActuator)
            for property_name, rows in actuator.joint_property_resolution_table.items():
                resolution_rows.setdefault(property_name, tuple(tuple(row) for row in rows))
            self._joint_property_resolution_rows[actuator_name] = resolution_rows
            actuator.__dict__.pop("joint_property_resolution_table", None)
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
            actuator._bind_joint_properties(self._control)

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
            defaults.effort_limit
            if implicit
            else torch.full_like(
                defaults.effort_limit,
                ActuatorBase._DEFAULT_MAX_EFFORT_SIM,
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
                effort_limit=values["effort_limit"],
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

    def _joint_indices_as_wp(self, actuator: ActuatorBase) -> wp.array(dtype=wp.int32):
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return self._all_joint_ids
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            return joint_indices
        return wp.from_torch(joint_indices.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)

    def _joint_indices_as_torch(self, actuator: ActuatorBase) -> torch.Tensor:
        if actuator.joint_indices == slice(None) or actuator.joint_indices is None:
            return torch.arange(self.num_joints, dtype=torch.int32, device=self.device)
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, wp.array):
            joint_indices = wp.to_torch(joint_indices)
        return joint_indices.to(self.device, dtype=torch.int32).contiguous()

    def _make_execution_batch(
        self,
        group_names: tuple[str, ...],
        groups: tuple[ActuatorBase, ...],
        joint_indices: torch.Tensor,
        *,
        executor: ActuatorBase | None = None,
    ) -> ActuatorCollection._ExecutionBatch:
        group_slices = []
        start = 0
        for group in groups:
            stop = start + group.num_joints
            group_slices.append(slice(start, stop))
            start = stop
        joint_indices = joint_indices.to(self.device, dtype=torch.int32).contiguous()
        if executor is None:
            executor = groups[0]
        else:
            executor._joint_names = [name for group in groups for name in group.joint_names]
            executor._joint_indices = joint_indices
        batch = self._ExecutionBatch(
            actuator=executor,
            group_names=group_names,
            group_slices=tuple(group_slices),
            joint_indices=joint_indices,
            joint_indices_wp=wp.from_torch(joint_indices, dtype=wp.int32),
        )
        if type(executor) is ImplicitActuator:
            batch.implicit_inputs = [
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
            batch.implicit_outputs = [
                wp.from_torch(executor.computed_effort, dtype=wp.float32),
                wp.from_torch(executor.applied_effort, dtype=wp.float32),
                self._joint_pos_target_sim,
                self._joint_vel_target_sim,
                self._joint_effort_target_sim,
                self._computed_torque,
                self._applied_torque,
                self._soft_joint_vel_limits,
            ]
        elif type(executor) in (IdealPDActuator, DCMotor):
            if len(groups) == 1 and groups[0].joint_indices == slice(None):
                return batch
            shape = (self.num_instances, joint_indices.shape[0])
            command_pos = torch.empty(shape, dtype=torch.float32, device=self.device)
            command_vel = torch.empty_like(command_pos)
            command_effort = torch.empty_like(command_pos)
            joint_pos = torch.empty_like(command_pos)
            joint_vel = torch.empty_like(command_pos)
            batch.control_action = ArticulationActions(
                joint_positions=command_pos,
                joint_velocities=command_vel,
                joint_efforts=command_effort,
                joint_indices=joint_indices,
            )
            batch.joint_pos = joint_pos
            batch.joint_vel = joint_vel
            batch.gather_inputs = [
                self._joint_pos_target,
                self._joint_vel_target,
                self._joint_effort_target,
                self._control.joint_pos.warp,
                self._control.joint_vel.warp,
                batch.joint_indices_wp,
            ]
            batch.gather_outputs = [
                wp.from_torch(command_pos, dtype=wp.float32),
                wp.from_torch(command_vel, dtype=wp.float32),
                wp.from_torch(command_effort, dtype=wp.float32),
                wp.from_torch(joint_pos, dtype=wp.float32),
                wp.from_torch(joint_vel, dtype=wp.float32),
            ]
        return batch

    def _build_execution_batches(self) -> None:
        native_actuator_path_active = self._control.native_actuator_path_active
        batch_by_group: dict[str, ActuatorCollection._ExecutionBatch] = {}
        if not self._groups:
            self._execution_batches = []
            return
        group_joint_indices = {name: self._joint_indices_as_torch(group) for name, group in self._groups.items()}

        for actuator_type in self._groups_by_class:
            names = tuple(name for name, group in self._groups.items() if type(group) is actuator_type)
            groups = tuple(self._groups[name] for name in names)
            joint_indices = [group_joint_indices[name] for name in names]
            parameter_names = actuator_type.__dict__.get("_EXECUTION_PARAMETER_NAMES")

            if native_actuator_path_active or parameter_names is None:
                for name, group, indices in zip(names, groups, joint_indices):
                    batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
                continue

            if len(groups) < 2:
                for name, group, indices in zip(names, groups, joint_indices):
                    batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
                continue

            combined = torch.cat(joint_indices)
            executor = actuator_type._build_execution_actuator(groups)
            batch = self._make_execution_batch(names, groups, combined, executor=executor)
            self._bind_execution_batch_parameters(batch, groups, parameter_names)
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
        batch: ActuatorCollection._ExecutionBatch,
        groups: Sequence[ActuatorBase],
        parameter_names: tuple[str, ...],
    ) -> None:
        tensor_names = (*parameter_names, "computed_effort", "applied_effort")
        bindings: list[tuple[ActuatorBase, str, torch.Tensor]] = []
        for group, group_slice in zip(groups, batch.group_slices):
            for name in tensor_names:
                original = getattr(group, name)
                view = getattr(batch.actuator, name)[:, group_slice]
                if view.shape != original.shape or view.dtype != original.dtype or view.device != original.device:
                    raise ValueError(f"Execution batch view for '{name}' is incompatible with its logical group.")
                bindings.append((group, name, view))

        for group, name, view in bindings:
            setattr(group, name, view)

    def _compute_implicit_batch(self, batch: ActuatorCollection._ExecutionBatch) -> None:
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
        """Rebind execution batches after backend state storage is replaced."""
        for batch in self._execution_batches:
            if batch.implicit_inputs is not None:
                batch.implicit_inputs[3] = self._control.joint_pos.warp
                batch.implicit_inputs[4] = self._control.joint_vel.warp
                batch.implicit_inputs[5] = self._control.joint_stiffness.warp
                batch.implicit_inputs[6] = self._control.joint_damping.warp
                batch.implicit_inputs[7] = self._control.joint_effort_limits.warp
                self._launch_cache.clear(("implicit", id(batch)))
                continue
            if batch.gather_inputs is not None:
                batch.gather_inputs[3] = self._control.joint_pos.warp
                batch.gather_inputs[4] = self._control.joint_vel.warp
                self._launch_cache.clear(("gather", id(batch)))

    def _gather_explicit_batch(self, batch: ActuatorCollection._ExecutionBatch) -> None:
        if batch.gather_inputs is None or batch.gather_outputs is None:
            raise RuntimeError("Explicit actuator execution batch was not initialized.")
        self._launch_cache.launch(
            ("gather", id(batch)),
            actuator_kernels.gather_actuator_batch,
            dim=(self.num_instances, batch.joint_indices_wp.shape[0]),
            inputs=batch.gather_inputs,
            outputs=batch.gather_outputs,
        )

    def _write_index_target(
        self,
        target: torch.Tensor | wp.array(dtype=wp.float32),
        env_ids: torch.Tensor | _WarpIndex,
        joint_ids: torch.Tensor | _WarpIndex,
        target_buffer: wp.array(dtype=wp.float32),
        *,
        full_data: bool,
        command_name: str,
    ) -> None:
        expected_shape = (self.num_instances, self.num_joints) if full_data else (env_ids.shape[0], joint_ids.shape[0])
        self._control.assert_shape_and_dtype(target, expected_shape, wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids, joint_ids),
            dim=(env_ids.shape[0], joint_ids.shape[0]),
            inputs=[target, env_ids, joint_ids, full_data],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.stage_user_command(command_name, self, env_ids, joint_ids, None, None)

    def _write_mask_target(
        self,
        target: torch.Tensor | wp.array(dtype=wp.float32),
        env_mask: wp.array(dtype=wp.bool),
        joint_mask: wp.array(dtype=wp.bool),
        target_buffer: wp.array(dtype=wp.float32),
        *,
        command_name: str,
    ) -> None:
        self._control.assert_shape_and_dtype_mask(target, (env_mask, joint_mask), wp.float32, "target")
        wp.launch(
            actuator_kernels.write_2d_float_with_mask,
            dim=(env_mask.shape[0], joint_mask.shape[0]),
            inputs=[target, env_mask, joint_mask],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.stage_user_command(command_name, self, None, None, env_mask, joint_mask)

    def _scatter_actuator_output(
        self,
        actuator: ActuatorBase,
        control_action: ArticulationActions,
        joint_indices: wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        if joint_indices is None:
            joint_indices = self._joint_indices_as_wp(actuator)
        gear_ratio = getattr(actuator, "gear_ratio", None)
        has_gear_ratio = gear_ratio is not None
        if gear_ratio is None:
            gear_ratio = self._gear_ratio
        inputs = [
            control_action.joint_positions,
            control_action.joint_velocities,
            control_action.joint_efforts,
            actuator.computed_effort,
            actuator.applied_effort,
            gear_ratio,
            actuator.velocity_limit,
            has_gear_ratio,
            joint_indices,
        ]
        outputs = [
            self._joint_pos_target_sim,
            self._joint_vel_target_sim,
            self._joint_effort_target_sim,
            self._computed_torque,
            self._applied_torque,
            self._gear_ratio,
            self._soft_joint_vel_limits,
        ]
        stable_launch = type(actuator) in (IdealPDActuator, DCMotor)
        if stable_launch:
            self._launch_cache.launch(
                ("scatter_outputs", id(actuator)),
                actuator_kernels.scatter_explicit_actuator_outputs,
                dim=(self.num_instances, joint_indices.shape[0]),
                inputs=inputs,
                outputs=outputs,
            )
        else:
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
        env_ids: Sequence[int] | torch.Tensor | _WarpIndex,
        joint_ids: Sequence[int] | torch.Tensor | _WarpIndex,
    ) -> None:
        env_ids = self._as_torch_indices(self._control.resolve_env_ids(env_ids))
        joint_ids = self._as_torch_indices(self._control.resolve_joint_ids(joint_ids))
        values_snapshot = values.to(self.device, dtype=torch.float32).contiguous().clone()
        self._control.write_native_actuator_gain(attr, values_snapshot, env_ids, joint_ids)

    def _as_torch_indices(self, indices: torch.Tensor | _WarpIndex) -> torch.Tensor:
        if isinstance(indices, wp.array):
            indices = wp.to_torch(indices)
        return indices.to(self.device, dtype=torch.long)

    def _validate_coverage(self) -> None:
        if self.num_joints == 0:
            return
        total_act_joints = sum(actuator.num_joints for actuator in self._groups.values())
        expected_joints = self.num_joints - self._control.num_fixed_tendons
        if total_act_joints != expected_joints:
            logger.warning(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                " joints available: %s != %s.",
                total_act_joints,
                expected_joints,
            )

    def _print_value_resolution_table(self) -> None:
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
