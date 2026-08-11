# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX actuator control adapter."""

from __future__ import annotations

from typing import TypeAliasType

import torch
import warp as wp

from isaaclab.actuators import ActuatorBase, ActuatorCollection
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.assets.articulation import ordering_kernels

from isaaclab_ov import tensor_types as TT

_WarpInt32 = TypeAliasType("_WarpInt32", wp.array(dtype=wp.int32))
_WarpInt64 = TypeAliasType("_WarpInt64", wp.array(dtype=wp.int64))
_WarpIndex = TypeAliasType("_WarpIndex", _WarpInt32 | _WarpInt64)


class OvPhysxActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the OVPhysX backend."""

    def _write_joint_friction_properties(self, actuator: ActuatorBase) -> None:
        # OVPhysX writes all friction components through one packed binding.
        self._articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=actuator.friction,
            joint_dynamic_friction_coeff=actuator.dynamic_friction,
            joint_viscous_friction_coeff=actuator.viscous_friction,
            joint_ids=actuator.joint_indices,
        )

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection,
        env_ids: torch.Tensor | _WarpIndex | None,
        joint_ids: torch.Tensor | _WarpIndex | None,
        env_mask: wp.array(dtype=wp.bool) | None,
        joint_mask: wp.array(dtype=wp.bool) | None,
    ) -> None:
        """Stage a public-order user command in the corresponding OVPhysX binding."""
        tensor_type, can_write, user_buffer, backend_buffer = self._command_buffers(command_name, collection)
        if not can_write:
            return
        articulation = self._articulation
        target_backend = articulation._get_backend_ordered_joint_buffer(user_buffer, backend_buffer)
        if env_mask is not None:
            articulation._root_view.set_attribute(tensor_type, target_backend, mask=env_mask)
        elif env_ids is not None:
            articulation._root_view.set_attribute(
                tensor_type, target_backend, indices=articulation._get_sim_env_ids(env_ids)
            )
        else:
            articulation._root_view.set_attribute(tensor_type, target_backend)

    def submit_commands(self, collection: ActuatorCollection) -> None:
        articulation = self._articulation
        # Keep applied effort separate from the raw target used by partial writes.
        write_effort = articulation._can_write_effort
        # position and velocity targets only for implicit actuators.
        write_pos = articulation._has_implicit_actuators and articulation._can_write_pos_target
        write_vel = articulation._has_implicit_actuators and articulation._can_write_vel_target
        if articulation.data.has_joint_ordering:
            if write_effort or write_pos or write_vel:
                ordering_kernels.launch_reorder_joint_targets_user_to_backend(
                    user_effort=collection._applied_torque,
                    user_pos_target=collection._joint_pos_target,
                    user_vel_target=collection._joint_vel_target,
                    backend_to_user=articulation.data.joint_ordering.backend_to_user,
                    write_effort=write_effort,
                    write_pos_target=write_pos,
                    write_vel_target=write_vel,
                    write_joint_act=False,
                    backend_effort=articulation._applied_torque_backend,
                    backend_pos_target=articulation._joint_pos_target_backend,
                    backend_vel_target=articulation._joint_vel_target_backend,
                    backend_joint_act=None,
                    device=self.device,
                )
            effort = articulation._applied_torque_backend
            pos_target = articulation._joint_pos_target_backend
            vel_target = articulation._joint_vel_target_backend
        else:
            effort = collection._applied_torque
            pos_target = collection._joint_pos_target
            vel_target = collection._joint_vel_target
        if write_effort:
            articulation._root_view.set_attribute(TT.DOF_ACTUATION_FORCE, effort)
        if write_pos:
            articulation._root_view.set_attribute(TT.DOF_POSITION_TARGET, pos_target)
        if write_vel:
            articulation._root_view.set_attribute(TT.DOF_VELOCITY_TARGET, vel_target)

    def _command_buffers(
        self,
        command_name: str,
        collection: ActuatorCollection,
    ) -> tuple[
        TT.TensorType,
        bool,
        wp.array(dtype=wp.float32),
        wp.array(dtype=wp.float32) | None,
    ]:
        articulation = self._articulation
        if command_name == "position":
            return (
                TT.DOF_POSITION_TARGET,
                articulation._can_write_pos_target,
                collection._joint_pos_target,
                articulation._joint_pos_target_backend,
            )
        if command_name == "velocity":
            return (
                TT.DOF_VELOCITY_TARGET,
                articulation._can_write_vel_target,
                collection._joint_vel_target,
                articulation._joint_vel_target_backend,
            )
        if command_name == "effort":
            return (
                TT.DOF_ACTUATION_FORCE,
                articulation._can_write_effort,
                collection._joint_effort_target,
                articulation._joint_effort_target_backend,
            )
        raise ValueError(f"Unsupported actuator command buffer '{command_name}'.")
