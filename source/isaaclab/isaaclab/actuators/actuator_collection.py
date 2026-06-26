# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime actuator collection for articulations."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence

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

    The collection owns actuator command buffers, processed command buffers,
    actuator telemetry, and actuator-resolved gain/state buffers. Named actuator
    groups remain visible through the mapping interface, but membership is fixed
    after construction.
    """

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
        self._native_group_names = self._control.prepare_native_actuators(self, actuator_cfgs)
        self._build_groups(actuator_cfgs)
        self._control.finalize_native_actuators(self)
        self._validate_coverage()
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
    def joint_pos_target(self) -> ProxyArray:
        """Joint position targets [m or rad, depending on joint type]."""
        return self._joint_pos_target_ta

    @property
    def joint_vel_target(self) -> ProxyArray:
        """Joint velocity targets [m/s or rad/s, depending on joint type]."""
        return self._joint_vel_target_ta

    @property
    def joint_effort_target(self) -> ProxyArray:
        """Joint effort targets [N or N·m, depending on joint type]."""
        return self._joint_effort_target_ta

    @property
    def joint_pos_target_sim(self) -> ProxyArray:
        """Processed joint position targets [m or rad, depending on joint type]."""
        return self._joint_pos_target_sim_ta

    @property
    def joint_vel_target_sim(self) -> ProxyArray:
        """Processed joint velocity targets [m/s or rad/s, depending on joint type]."""
        return self._joint_vel_target_sim_ta

    @property
    def joint_effort_target_sim(self) -> ProxyArray:
        """Processed joint effort targets [N or N·m, depending on joint type]."""
        return self._joint_effort_target_sim_ta

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

        for actuator in self._groups.values():
            control_action = ArticulationActions(
                joint_positions=self.joint_pos_target.torch[:, actuator.joint_indices],
                joint_velocities=self.joint_vel_target.torch[:, actuator.joint_indices],
                joint_efforts=self.joint_effort_target.torch[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            control_action = actuator.compute(
                control_action,
                joint_pos=self._control.joint_pos.torch[:, actuator.joint_indices],
                joint_vel=self._control.joint_vel.torch[:, actuator.joint_indices],
            )
            self._scatter_actuator_output(actuator, control_action)

    def submit_commands(self) -> None:
        """Submit processed actuator command buffers through the backend control object."""
        self._control.submit_commands(self)

    def set_joint_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set joint position targets using indices.

        Args:
            target: Joint position targets [m or rad, depending on joint type].
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether ``target`` is a full articulation command buffer.
        """
        env_ids_resolved = self._control.resolve_env_ids(env_ids)
        joint_ids_resolved = self._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            target,
            env_ids_resolved,
            joint_ids_resolved,
            self._joint_pos_target,
            full_data=full_data,
            command_name="position",
        )

    def set_joint_velocity_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set joint velocity targets using indices.

        Args:
            target: Joint velocity targets [m/s or rad/s, depending on joint type].
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether ``target`` is a full articulation command buffer.
        """
        env_ids_resolved = self._control.resolve_env_ids(env_ids)
        joint_ids_resolved = self._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            target,
            env_ids_resolved,
            joint_ids_resolved,
            self._joint_vel_target,
            full_data=full_data,
            command_name="velocity",
        )

    def set_joint_effort_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set joint effort targets using indices.

        Args:
            target: Joint effort targets [N or N·m, depending on joint type].
            joint_ids: Joint indices. Defaults to all joints.
            env_ids: Environment indices. Defaults to all environments.
            full_data: Whether ``target`` is a full articulation command buffer.
        """
        env_ids_resolved = self._control.resolve_env_ids(env_ids)
        joint_ids_resolved = self._control.resolve_joint_ids(joint_ids)
        self._write_index_target(
            target,
            env_ids_resolved,
            joint_ids_resolved,
            self._joint_effort_target,
            full_data=full_data,
            command_name="effort",
        )

    def set_joint_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets using masks."""
        env_mask_resolved = self._control.resolve_env_mask(env_mask)
        joint_mask_resolved = self._control.resolve_joint_mask(joint_mask)
        self._write_mask_target(
            target,
            env_mask_resolved,
            joint_mask_resolved,
            self._joint_pos_target,
            command_name="position",
        )

    def set_joint_velocity_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets using masks."""
        env_mask_resolved = self._control.resolve_env_mask(env_mask)
        joint_mask_resolved = self._control.resolve_joint_mask(joint_mask)
        self._write_mask_target(
            target,
            env_mask_resolved,
            joint_mask_resolved,
            self._joint_vel_target,
            command_name="velocity",
        )

    def set_joint_effort_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint effort targets using masks."""
        env_mask_resolved = self._control.resolve_env_mask(env_mask)
        joint_mask_resolved = self._control.resolve_joint_mask(joint_mask)
        self._write_mask_target(
            target,
            env_mask_resolved,
            joint_mask_resolved,
            self._joint_effort_target,
            command_name="effort",
        )

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

    def _scatter_actuator_output(self, actuator: ActuatorBase, control_action: ArticulationActions) -> None:
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
        env_ids_wp = wp.from_torch(env_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        joint_ids_wp = wp.from_torch(joint_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        values_wp = wp.from_torch(values.to(self.device, dtype=torch.float32).contiguous(), dtype=wp.float32)
        wp.launch(
            actuator_kernels.write_2d_float_with_indices_kernel(env_ids_wp, joint_ids_wp),
            dim=(env_ids_wp.shape[0], joint_ids_wp.shape[0]),
            inputs=[values_wp, env_ids_wp, joint_ids_wp, False],
            outputs=[target_buffer],
            device=self.device,
        )
        self._control.write_native_actuator_gain(attr, values, env_ids, joint_ids)

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
