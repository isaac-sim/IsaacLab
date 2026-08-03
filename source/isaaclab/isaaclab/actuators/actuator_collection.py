# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime actuator collection for articulations."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
import warp as wp
from prettytable import PrettyTable

from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray

from . import actuator_kernels
from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_control import ActuatorControl
from .actuator_pd import ImplicitActuator

logger = logging.getLogger(__name__)


class ActuatorCollection(Mapping[str, ActuatorBase]):
    """Read-only runtime collection of actuator groups for one articulation.

    The collection owns actuator command buffers, processed joint command buffers,
    actuator telemetry, and actuator-resolved gain/state buffers. Named mapping
    entries are stable logical configuration and access groups, and membership is
    fixed after construction. Compatible groups whose concrete type is the same
    supported stateless actuator class may share a private execution actuator while
    retaining their separate per-joint parameters and group-shaped public values.
    Execution batches are an implementation detail, and users must not depend on
    their count.

    The collection owns lifecycle execution for its managed groups. Calling
    :meth:`~isaaclab.actuators.ActuatorBase.compute` or
    :meth:`~isaaclab.actuators.ActuatorBase.reset` directly on a mapping value is
    unsupported.
    """

    @dataclass
    class _ExecutionBatch:
        actuator: ActuatorBase
        group_names: tuple[str, ...]
        group_slices: tuple[slice, ...]
        joint_indices: torch.Tensor
        joint_indices_wp: wp.array

    class Command:
        """Commands received by the actuator models.

        Position and velocity commands use joint-side coordinates. All command
        arrays are indexed by articulation joint, not by motor shaft.
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
            value: torch.Tensor | wp.array,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
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
            value: torch.Tensor | wp.array,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
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
            value: torch.Tensor | wp.array,
            joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
            env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
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
            value: torch.Tensor | wp.array,
            joint_mask: wp.array | None = None,
            env_mask: wp.array | None = None,
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
            value: torch.Tensor | wp.array,
            joint_mask: wp.array | None = None,
            env_mask: wp.array | None = None,
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
            value: torch.Tensor | wp.array,
            joint_mask: wp.array | None = None,
            env_mask: wp.array | None = None,
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
        """Processed commands produced for the simulated joints."""

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
        self._has_implicit_actuators = False
        self._joint_indices_wp: dict[str, wp.array] = {}

        self._allocate_buffers()
        self._command = self.Command(self)
        self._joint_command = self.JointCommand(self)
        self._native_group_names = self._control.prepare_native_actuators(self, actuator_cfgs)
        self._build_groups(actuator_cfgs)
        self._control.finalize_native_actuators(self)
        self._validate_coverage()
        self._build_execution_batches()
        if debug_value_resolution:
            self._print_value_resolution_table()

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

    """
    Properties.
    """

    @property
    def command(self) -> Command:
        """Commands received by the actuator models."""
        return self._command

    @property
    def joint_command(self) -> JointCommand:
        """Processed commands produced for the simulated joints."""
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

    @property
    def actuator_stiffness(self) -> ProxyArray:
        """Actuator-resolved stiffness values [N/m or N·m/rad, depending on joint type]."""
        return self._actuator_stiffness_ta

    @property
    def actuator_damping(self) -> ProxyArray:
        """Actuator-resolved damping values [N·s/m or N·m·s/rad, depending on joint type]."""
        return self._actuator_damping_ta

    @property
    def soft_joint_vel_limits(self) -> ProxyArray:
        """Actuator-resolved soft joint velocity limits [m/s or rad/s, depending on joint type]."""
        return self._soft_joint_vel_limits_ta

    @property
    def gear_ratio(self) -> ProxyArray:
        """Gear ratio for relating motor torques to applied joint torques [dimensionless]."""
        return self._gear_ratio_ta

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset all actuator group states.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        if env_ids is None:
            env_ids = slice(None)
        for actuator in self._groups.values():
            actuator.reset(env_ids)
        self._control.reset_native_actuators(env_ids)

    def compute(self, dt: float = 0.0) -> None:
        """Compute processed actuator commands and telemetry.

        Args:
            dt: Physics step size [s].
        """
        if self._control.compute_native_actuators(self, dt):
            return

        for batch in self._execution_batches:
            actuator = batch.actuator
            joint_indices = batch.joint_indices
            control_action = ArticulationActions(
                joint_positions=self.command.position.torch[:, joint_indices],
                joint_velocities=self.command.velocity.torch[:, joint_indices],
                joint_efforts=self.command.effort.torch[:, joint_indices],
                joint_indices=joint_indices,
            )
            control_action = actuator.compute(
                control_action,
                joint_pos=self._control.joint_pos.torch[:, joint_indices],
                joint_vel=self._control.joint_vel.torch[:, joint_indices],
            )
            self._bind_execution_batch_outputs(batch)
            self._scatter_actuator_output(actuator, control_action, batch.joint_indices_wp)

    def submit_commands(self) -> None:
        """Submit processed actuator command buffers through the backend control object."""
        self._control.submit_commands(self)

    def write_actuator_stiffness_to_sim(
        self,
        *,
        stiffness: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write actuator stiffness values and propagate them to native controllers."""
        self._write_actuator_gain("kp", stiffness, env_ids, joint_ids, self._actuator_stiffness)

    def write_actuator_damping_to_sim(
        self,
        *,
        damping: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write actuator damping values and propagate them to native controllers."""
        self._write_actuator_gain("kd", damping, env_ids, joint_ids, self._actuator_damping)

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
        self._actuator_stiffness = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._actuator_damping = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._gear_ratio = wp.ones(shape, dtype=wp.float32, device=self.device)
        self._all_env_ids = wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)
        self._all_joint_ids = wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)

        self._joint_pos_target_ta = ProxyArray(self._joint_pos_target)
        self._joint_vel_target_ta = ProxyArray(self._joint_vel_target)
        self._joint_effort_target_ta = ProxyArray(self._joint_effort_target)
        self._joint_pos_target_sim_ta = ProxyArray(self._joint_pos_target_sim)
        self._joint_vel_target_sim_ta = ProxyArray(self._joint_vel_target_sim)
        self._joint_effort_target_sim_ta = ProxyArray(self._joint_effort_target_sim)
        self._computed_torque_ta = ProxyArray(self._computed_torque)
        self._applied_torque_ta = ProxyArray(self._applied_torque)
        self._actuator_stiffness_ta = ProxyArray(self._actuator_stiffness)
        self._actuator_damping_ta = ProxyArray(self._actuator_damping)
        self._soft_joint_vel_limits_ta = ProxyArray(self._soft_joint_vel_limits)
        self._gear_ratio_ta = ProxyArray(self._gear_ratio)

    def _build_groups(self, actuator_cfgs: dict[str, ActuatorBaseCfg]) -> None:
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            joint_ids, joint_names = self._control.find_joints(actuator_cfg.joint_names_expr)
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            if len(joint_names) == self.num_joints:
                actuator_joint_ids: slice | torch.Tensor = slice(None)
            elif isinstance(joint_ids, ProxyArray):
                actuator_joint_ids = joint_ids.torch
            else:
                actuator_joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.int32)

            defaults = self._control.get_default_joint_properties(actuator_joint_ids)
            cfg = actuator_cfg.copy() if hasattr(actuator_cfg, "copy") else actuator_cfg
            actuator: ActuatorBase = cfg.class_type(
                cfg=cfg,
                joint_names=joint_names,
                joint_ids=actuator_joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=defaults.stiffness,
                damping=defaults.damping,
                armature=defaults.armature,
                friction=defaults.friction,
                dynamic_friction=defaults.dynamic_friction,
                viscous_friction=defaults.viscous_friction,
                effort_limit=defaults.effort_limit,
                velocity_limit=defaults.velocity_limit,
            )

            self._groups[actuator_name] = actuator
            self._groups_by_class.setdefault(type(actuator), []).append(actuator)
            self._joint_indices_wp[actuator_name] = self._joint_indices_as_wp(actuator)
            self._has_implicit_actuators = self._has_implicit_actuators or isinstance(actuator, ImplicitActuator)

            self._scatter_resolved_gains(actuator_name, actuator)
            self._control.write_resolved_joint_properties(
                actuator,
                native_managed=actuator_name in self._native_group_names,
            )

    def _joint_indices_as_wp(self, actuator: ActuatorBase) -> wp.array:
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
        if executor is None:
            executor = groups[0]
        group_slices = []
        start = 0
        for group in groups:
            stop = start + group.num_joints
            group_slices.append(slice(start, stop))
            start = stop
        joint_indices = joint_indices.to(self.device, dtype=torch.int32).contiguous()
        executor._joint_names = [name for group in groups for name in group.joint_names]
        executor._joint_indices = joint_indices
        return self._ExecutionBatch(
            actuator=executor,
            group_names=group_names,
            group_slices=tuple(group_slices),
            joint_indices=joint_indices,
            joint_indices_wp=wp.from_torch(joint_indices, dtype=wp.int32),
        )

    def _build_execution_batches(self) -> None:
        native_active = getattr(self._control, "native_active", False)
        batch_by_group: dict[str, ActuatorCollection._ExecutionBatch] = {}
        if not self._groups:
            self._execution_batches = []
            return
        group_joint_indices = {name: self._joint_indices_as_torch(group) for name, group in self._groups.items()}
        joint_use_count = torch.bincount(
            torch.cat(list(group_joint_indices.values())).to(dtype=torch.long),
            minlength=self.num_joints,
        )

        for actuator_type in self._groups_by_class:
            names = tuple(name for name, group in self._groups.items() if type(group) is actuator_type)
            groups = [self._groups[name] for name in names]
            joint_indices = [group_joint_indices[name] for name in names]
            supported = actuator_type.__dict__.get("_supports_execution_aggregation", False)

            if native_active or not supported:
                for name, group, indices in zip(names, groups, joint_indices):
                    batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
                continue

            safe = [
                (name, group, indices)
                for name, group, indices in zip(names, groups, joint_indices)
                if torch.all(joint_use_count[indices.to(dtype=torch.long)] == 1)
            ]
            safe_names_set = {name for name, _, _ in safe}
            unsafe = [
                (name, group, indices)
                for name, group, indices in zip(names, groups, joint_indices)
                if name not in safe_names_set
            ]
            for name, group, indices in unsafe:
                batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
            if len(safe) < 2:
                for name, group, indices in safe:
                    batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
                continue

            safe_names, safe_groups, safe_indices = zip(*safe)
            combined = torch.cat(safe_indices)
            executor = actuator_type._build_execution_actuator(safe_groups)
            executor._joint_indices = combined
            batch = self._make_execution_batch(safe_names, safe_groups, combined, executor=executor)
            self._validate_execution_batch(batch, safe_groups)
            self._bind_execution_batch_parameters(batch, safe_groups)
            for name in safe_names:
                batch_by_group[name] = batch

        seen: set[int] = set()
        self._execution_batches = []
        for name in self._groups:
            batch = batch_by_group[name]
            if id(batch) not in seen:
                self._execution_batches.append(batch)
                seen.add(id(batch))

    def _validate_execution_batch(
        self, batch: ActuatorCollection._ExecutionBatch, groups: Sequence[ActuatorBase]
    ) -> None:
        expected_joint_names = [name for group in groups for name in group.joint_names]
        expected_num_joints = len(expected_joint_names)
        if len(batch.group_names) != len(groups) or len(batch.group_slices) != len(groups):
            raise ValueError("Execution batch group metadata is inconsistent.")
        if any(self._groups[name] is not group for name, group in zip(batch.group_names, groups)):
            raise ValueError("Execution batch group names do not match its logical groups.")
        if batch.actuator.joint_names != expected_joint_names:
            raise ValueError("Execution batch joint names do not match its logical groups.")
        if batch.joint_indices.ndim != 1 or batch.joint_indices.shape[0] != expected_num_joints:
            raise ValueError("Execution batch joint indices do not match its logical groups.")
        if (
            batch.joint_indices.dtype != torch.int32
            or batch.joint_indices.device != torch.device(self.device)
            or not batch.joint_indices.is_contiguous()
        ):
            raise ValueError("Execution batch joint indices use an unexpected dtype or device.")
        if not torch.equal(batch.actuator.joint_indices, batch.joint_indices):
            raise ValueError("Execution actuator joint indices do not match its batch.")
        if (
            batch.joint_indices_wp.shape[0] != expected_num_joints
            or batch.joint_indices_wp.dtype != wp.int32
            or batch.joint_indices_wp.device != wp.get_device(self.device)
        ):
            raise ValueError("Execution batch Warp joint indices do not match its logical groups.")

        expected_start = 0
        for group, group_slice in zip(groups, batch.group_slices):
            expected_stop = expected_start + group.num_joints
            if group_slice != slice(expected_start, expected_stop):
                raise ValueError("Execution batch group slices are not contiguous.")
            expected_start = expected_stop
        if expected_start != expected_num_joints:
            raise ValueError("Execution batch group slices do not cover all executor joints.")

        tensor_names = (*ActuatorBase._EXECUTION_PARAMETER_NAMES, "computed_effort", "applied_effort")
        for name in tensor_names:
            value = getattr(batch.actuator, name)
            if value.shape != (self.num_instances, expected_num_joints):
                raise ValueError(f"Execution batch tensor '{name}' has an unexpected shape.")
            if value.device != torch.device(self.device) or value.dtype != getattr(groups[0], name).dtype:
                raise ValueError(f"Execution batch tensor '{name}' has an unexpected dtype or device.")

    def _bind_execution_batch_parameters(
        self, batch: ActuatorCollection._ExecutionBatch, groups: Sequence[ActuatorBase]
    ) -> None:
        tensor_names = (*ActuatorBase._EXECUTION_PARAMETER_NAMES, "computed_effort", "applied_effort")
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

    def _bind_execution_batch_outputs(self, batch: ActuatorCollection._ExecutionBatch) -> None:
        for group_name, group_slice in zip(batch.group_names, batch.group_slices):
            group = self._groups[group_name]
            group.computed_effort = batch.actuator.computed_effort[:, group_slice]
            group.applied_effort = batch.actuator.applied_effort[:, group_slice]

    def _write_index_target(
        self,
        target: torch.Tensor | wp.array,
        env_ids: torch.Tensor | wp.array,
        joint_ids: torch.Tensor | wp.array,
        target_buffer: wp.array,
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
        target: torch.Tensor | wp.array,
        env_mask: wp.array,
        joint_mask: wp.array,
        target_buffer: wp.array,
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

    def _scatter_resolved_gains(self, actuator_name: str, actuator: ActuatorBase) -> None:
        joint_indices = self._joint_indices_wp[actuator_name]
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(self._all_env_ids, joint_indices),
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=[actuator.stiffness, self._all_env_ids, joint_indices, False],
            outputs=[self._actuator_stiffness],
            device=self.device,
        )
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(self._all_env_ids, joint_indices),
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=[actuator.damping, self._all_env_ids, joint_indices, False],
            outputs=[self._actuator_damping],
            device=self.device,
        )

    def _scatter_actuator_output(
        self,
        actuator: ActuatorBase,
        control_action: ArticulationActions,
        joint_indices: wp.array | None = None,
    ) -> None:
        if joint_indices is None:
            joint_indices = self._joint_indices_as_wp(actuator)
        wp.launch(
            actuator_kernels.scatter_processed_targets,
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=[
                control_action.joint_positions,
                control_action.joint_velocities,
                control_action.joint_efforts,
                joint_indices,
            ],
            outputs=[
                self._joint_pos_target_sim,
                self._joint_vel_target_sim,
                self._joint_effort_target_sim,
            ],
            device=self.device,
        )
        gear_ratio = getattr(actuator, "gear_ratio", None)
        has_gear_ratio = gear_ratio is not None
        if gear_ratio is None:
            gear_ratio = self._gear_ratio
        wp.launch(
            actuator_kernels.scatter_actuator_state_model,
            dim=(self.num_instances, joint_indices.shape[0]),
            inputs=[
                actuator.computed_effort,
                actuator.applied_effort,
                gear_ratio,
                actuator.velocity_limit,
                has_gear_ratio,
                joint_indices,
            ],
            outputs=[
                self._computed_torque,
                self._applied_torque,
                self._gear_ratio,
                self._soft_joint_vel_limits,
            ],
            device=self.device,
        )

    def _write_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
        target_buffer: wp.array,
    ) -> None:
        values_snapshot = values.to(self.device, dtype=torch.float32).contiguous().clone()
        actuator_attr = {"kp": "stiffness", "kd": "damping"}[attr]
        self._write_execution_parameter(actuator_attr, values_snapshot, env_ids, joint_ids)
        env_ids_wp = wp.from_torch(env_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        joint_ids_wp = wp.from_torch(joint_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        values_wp = wp.from_torch(values_snapshot, dtype=wp.float32)
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids_wp, joint_ids_wp),
            dim=(env_ids_wp.shape[0], joint_ids_wp.shape[0]),
            inputs=[values_wp, env_ids_wp, joint_ids_wp, False],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.write_native_actuator_gain(attr, values_snapshot, env_ids, joint_ids)

    def _write_execution_parameter(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        values = values.to(self.device, dtype=torch.float32)
        env_ids = env_ids.to(self.device, dtype=torch.long)
        joint_ids = joint_ids.to(self.device, dtype=torch.long)
        for batch in self._execution_batches:
            batch_joint_ids = batch.joint_indices.to(dtype=torch.long)
            requested_columns, batch_columns = torch.where(joint_ids[:, None] == batch_joint_ids[None, :])
            if requested_columns.numel() == 0:
                continue
            target = getattr(batch.actuator, attr)
            target[env_ids[:, None], batch_columns[None, :]] = values[:, requested_columns]

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
        for actuator_group, actuator in self._groups.items():
            group_count = 0
            for property_name, resolution_details in actuator.joint_property_resolution_table.items():
                for prop_idx, resolution_detail in enumerate(resolution_details):
                    actuator_group_str = actuator_group if group_count == 0 else ""
                    property_str = property_name if prop_idx == 0 else ""
                    fmt = [f"{value:.2e}" if isinstance(value, float) else str(value) for value in resolution_detail]
                    table.add_row([actuator_group_str, property_str, *fmt])
                    group_count += 1
        logger.warning("\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n%s", table)
