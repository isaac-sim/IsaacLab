# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX actuator control adapter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_control import ActuatorJointProperties, ArticulationActuatorControl
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.utils.queries import find_first_matching_prim

from isaaclab_ov import tensor_types as TT

logger = logging.getLogger(__name__)


class OvPhysxActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the OVPhysX backend."""

    def __init__(self, articulation):
        super().__init__(articulation)
        self._host_actuator_runtime = None

    @property
    def _physx_actuator_wrapper(self):
        """Expose the shared host wrapper for PhysX-native compatibility."""
        return None if self._host_actuator_runtime is None else self._host_actuator_runtime.wrapper

    def prepare_native_actuators(self, collection: ActuatorCollection, actuator_cfgs: dict) -> set[str]:
        """Prepare optional Newton-native explicit actuators for OVPhysX."""
        articulation = self._articulation
        articulation._physx_actuator_wrapper = None
        articulation.newton_actuator_adapter = None
        articulation.newton_default_stiffness = None
        articulation.newton_default_damping = None
        articulation.newton_managed_local_joints = None
        articulation._implicit_dof_mask = None
        articulation._has_newton_actuators = False
        self._host_actuator_runtime = None

        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if not use_newton_actuators:
            return set()
        try:
            from isaaclab_newton.actuators.host_runtime import _HostActuatorRuntime  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_newton", "isaaclab_newton.actuators"}:
                raise
            logger.warning(
                "use_newton_actuators is enabled but 'isaaclab_newton.actuators' is not available. "
                "Newton-native actuators will be disabled and the simulation will fall back to the "
                "Isaac Lab actuator path. Install the isaaclab_newton extension to enable the fast path."
            )
            return set()

        from isaaclab.sim.schemas.schemas_actuators import _validate_newton_native_actuator_cfgs  # noqa: PLC0415
        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415

        _validate_newton_native_actuator_cfgs(actuator_cfgs)
        native_group_names = {
            name for name, actuator_cfg in actuator_cfgs.items() if not self._is_implicit_cfg(actuator_cfg)
        }
        if not native_group_names:
            return set()

        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True
        first_prim = find_first_matching_prim(articulation.cfg.prim_path)
        articulation_prim_path = str(first_prim.GetPath()) if first_prim is not None else None
        self._host_actuator_runtime = _HostActuatorRuntime(articulation, logger=logger)
        self._host_actuator_runtime.prepare(
            collection,
            stage=get_current_stage(),
            articulation_prim_path=articulation_prim_path,
        )
        articulation._physx_actuator_wrapper = self._host_actuator_runtime.wrapper
        articulation.newton_actuator_adapter = self._host_actuator_runtime.adapter
        return native_group_names

    def finalize_native_actuators(self, collection: ActuatorCollection) -> None:
        if self._native_actuator_path_active and self._host_actuator_runtime is not None:
            self._host_actuator_runtime.finalize(collection)

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        if not self._native_actuator_path_active or self._host_actuator_runtime is None:
            return False
        # OVPhysX public-order state uses owned staging buffers, even for identity ordering.
        # Refresh both fields before every controller step so it observes the current simulation state.
        self._articulation._data._refresh_joint_pos()
        self._articulation._data._refresh_joint_vel()
        self._host_actuator_runtime.compute(collection, dt)
        return True

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        if self._native_actuator_path_active and self._host_actuator_runtime is not None:
            self._host_actuator_runtime.reset(env_ids)

    def get_native_actuator_gain(
        self,
        attr: Literal["kp", "kd"],
        joint_ids: torch.Tensor | slice,
    ) -> torch.Tensor | None:
        if self._host_actuator_runtime is None:
            return None
        return self._host_actuator_runtime.get_gain(attr, joint_ids)

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        if self._host_actuator_runtime is not None:
            self._host_actuator_runtime.write_gain(attr, values, env_ids, joint_ids)

    def _write_joint_friction_properties(
        self,
        properties: ActuatorJointProperties,
        joint_ids: torch.Tensor | wp.array | slice,
    ) -> None:
        # OVPhysX writes all friction components through one packed binding.
        self._articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=properties.friction,
            joint_dynamic_friction_coeff=properties.dynamic_friction,
            joint_viscous_friction_coeff=properties.viscous_friction,
            joint_ids=joint_ids,
        )

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection,
        env_ids: torch.Tensor | wp.array | None,
        joint_ids: torch.Tensor | wp.array | None,
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
        # Native telemetry contains the local implicit-drive shadow. Submit raw runtime effort
        # instead, so OVPhysX evaluates each implicit PD drive exactly once.
        write_effort = articulation._can_write_effort
        # position and velocity targets only for implicit actuators.
        write_pos = articulation._has_implicit_actuators and articulation._can_write_pos_target
        write_vel = articulation._has_implicit_actuators and articulation._can_write_vel_target
        user_effort = collection._applied_torque
        if self._host_actuator_runtime is not None:
            user_effort = self._host_actuator_runtime.wrapper.joint_f_2d
        if articulation.data.has_joint_ordering:
            if write_effort or write_pos or write_vel:
                ordering_kernels.launch_reorder_joint_targets_user_to_backend(
                    user_effort=user_effort,
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
            effort = user_effort
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
