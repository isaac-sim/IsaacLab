# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.assets.articulation import ordering_kernels

from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from .articulation import Articulation

_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None

logger = logging.getLogger(__name__)


class NewtonActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the Newton backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: Newton articulation that owns backend simulation handles.
        """
        super().__init__(articulation)
        self._native_active = False

    def prepare_native_actuators(self, collection: ActuatorCollection, actuator_cfgs: dict) -> set[str]:
        articulation = self._articulation
        articulation._has_newton_actuators = False
        articulation._implicit_dof_mask = None
        articulation.newton_actuator_adapter = None
        articulation.newton_default_stiffness = None
        articulation.newton_default_damping = None
        articulation.newton_managed_local_joints = None

        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if use_newton_actuators and not _HAS_NEWTON_ACTUATORS:
            logger.warning(
                "use_newton_actuators is enabled but 'newton.actuators' is not available. "
                "Newton-native actuators will be disabled. Upgrade Newton to >= 1.2.0rc1."
            )
            return set()
        if not (use_newton_actuators and _HAS_NEWTON_ACTUATORS):
            return set()

        self._native_active = True
        articulation._has_newton_actuators = True
        SimulationManager.activate_newton_actuator_path()

        native_group_names: set[str] = set()
        explicit_joint_ids: list[int] = []
        for actuator_name, actuator_cfg in actuator_cfgs.items():
            if self._is_implicit_cfg(actuator_cfg):
                continue
            native_group_names.add(actuator_name)
            joint_ids, _ = articulation.find_joints(actuator_cfg.joint_names_expr)
            explicit_joint_ids.extend(int(joint_id) for joint_id in joint_ids)

        if explicit_joint_ids:
            explicit_ids_t = torch.tensor(
                sorted(set(explicit_joint_ids)),
                dtype=torch.int32,
                device=self.device,
            )
            articulation.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=explicit_ids_t)
            articulation.write_joint_damping_to_sim_index(damping=0.0, joint_ids=explicit_ids_t)

        return native_group_names

    def finalize_native_actuators(self, collection: ActuatorCollection) -> None:
        if not self._native_active:
            return

        from newton import Model as NewtonModel  # noqa: PLC0415

        from isaaclab_newton.actuators import build_implicit_dof_mask  # noqa: PLC0415
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        adapter = SimulationManager._adapter
        if adapter is not None:
            dof_layout = articulation._root_view.frequency_layouts[NewtonModel.AttributeFrequency.JOINT_DOF]
            if dof_layout.slice is not None:
                arti_start = dof_layout.slice.start
            elif dof_layout.indices is not None:
                arti_start = int(dof_layout.indices.numpy()[0])
            else:
                arti_start = 0
            joint_ordering = articulation.data.joint_ordering
            binding = adapter.bind_articulation(
                lab_actuators=dict(collection.items()),
                dof_offset=arti_start,
                num_joints=self.num_joints,
                joint_user_to_backend_indices=(
                    joint_ordering.user_to_backend_indices if joint_ordering is not None else None
                ),
            )
            articulation.newton_actuator_adapter = adapter
            articulation.newton_default_stiffness = binding.stiffness
            articulation.newton_default_damping = binding.damping
            articulation.newton_managed_local_joints = binding.joint_indices
            articulation._implicit_dof_mask = binding.implicit_dof_mask
            articulation._implicit_dof_mask_owner = binding.implicit_dof_mask_owner
            articulation._data._sim_bind_joint_computed_effort = binding.computed_effort_view
        else:
            articulation._implicit_dof_mask, articulation._implicit_dof_mask_owner = build_implicit_dof_mask(
                dict(collection.items()),
                self.num_joints,
                self.device,
            )
            articulation._data._sim_bind_joint_computed_effort = wp.zeros(
                (self.num_instances, self.num_joints),
                dtype=wp.float32,
                device=self.device,
            )

        def _post_actuator() -> None:
            wp.launch(
                actuator_kernels.sync_torque_telemetry,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    articulation._data._sim_bind_joint_pos,
                    articulation._data._sim_bind_joint_vel,
                    collection._joint_pos_target,
                    collection._joint_vel_target,
                    articulation._data.joint_stiffness.warp,
                    articulation._data.joint_damping.warp,
                    articulation._data.joint_effort_limits.warp,
                    articulation._implicit_dof_mask,
                    articulation._data._sim_bind_joint_effort,
                    articulation._data._sim_bind_joint_computed_effort,
                    articulation._joint_user_to_backend_map(),
                    articulation.data.has_joint_ordering,
                ],
                outputs=[
                    collection._computed_torque,
                    collection._applied_torque,
                ],
                device=self.device,
            )

        SimulationManager.register_post_actuator_callback(_post_actuator)

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return self._native_active

    def submit_commands(self, collection: ActuatorCollection) -> None:
        articulation = self._articulation
        if self._native_active:
            # Raw targets go directly to Newton's control object. Newton PD
            # consumes ``joint_act`` for explicit (Newton-managed) joints; the
            # solver's built-in joint drive does the PD for implicit joints
            # (whose stiffness/damping are non-zero in sim) and adds whatever
            # is in ``joint_f`` as feedforward. Identity ordering copies the
            # targets directly, while non-identity ordering gathers all four
            # targets in one launch.
            if not articulation.data.has_joint_ordering:
                articulation.data._sim_bind_joint_position_target.assign(collection._joint_pos_target)
                articulation.data._sim_bind_joint_velocity_target.assign(collection._joint_vel_target)
                articulation.data._sim_bind_joint_act.assign(collection._joint_effort_target)
                articulation.data._sim_bind_joint_effort.assign(collection._joint_effort_target)
            else:
                wp.launch(
                    ordering_kernels.reorder_joint_targets_user_to_backend,
                    dim=(self.num_instances, self.num_joints),
                    inputs=[
                        collection._joint_effort_target,
                        collection._joint_pos_target,
                        collection._joint_vel_target,
                        articulation._joint_backend_to_user_map(),
                        True,
                        True,
                        True,
                        True,
                    ],
                    outputs=[
                        articulation.data._sim_bind_joint_effort,
                        articulation.data._sim_bind_joint_position_target,
                        articulation.data._sim_bind_joint_velocity_target,
                        articulation.data._sim_bind_joint_act,
                    ],
                    device=self.device,
                )
            return

        # Standard Lab actuator path. Identity ordering copies processed
        # targets directly; non-identity ordering gathers them in one launch.
        if not articulation.data.has_joint_ordering:
            articulation.data._sim_bind_joint_effort.assign(collection._joint_effort_target_sim)
            if collection.has_implicit_actuators:
                articulation.data._sim_bind_joint_position_target.assign(collection._joint_pos_target_sim)
                articulation.data._sim_bind_joint_velocity_target.assign(collection._joint_vel_target_sim)
        else:
            wp.launch(
                ordering_kernels.reorder_joint_targets_user_to_backend,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    collection._joint_effort_target_sim,
                    collection._joint_pos_target_sim,
                    collection._joint_vel_target_sim,
                    articulation._joint_backend_to_user_map(),
                    True,
                    collection.has_implicit_actuators,
                    collection.has_implicit_actuators,
                    False,
                ],
                outputs=[
                    articulation.data._sim_bind_joint_effort,
                    articulation.data._sim_bind_joint_position_target,
                    articulation.data._sim_bind_joint_velocity_target,
                    articulation.data._sim_bind_joint_act,
                ],
                device=self.device,
            )

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        if self._native_active and SimulationManager._adapter is not None:
            SimulationManager._adapter.reset(env_ids)

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        # TODO: This routes through per-actuator torch indexing and has no mask
        # variant because the actuator gain buffers are per-actuator torch views
        # over arbitrary joint-index subsets. A single-launch warp path and a mask
        # variant need the actuator-side buffer layout rework, deferred to the
        # actuator rework built on this series.
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        adapter = articulation.newton_actuator_adapter
        if adapter is None:
            return

        env_ids_wp = wp.from_torch(env_ids.to(self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        env_mask = wp.zeros(self.num_instances, dtype=wp.bool, device=self.device)
        wp.launch(
            actuator_kernels.set_mask_kernel,
            dim=env_ids_wp.shape[0],
            inputs=[env_mask, env_ids_wp],
            device=self.device,
        )

        env_ids_long = env_ids.to(self.device, dtype=torch.long).unsqueeze(1)
        joint_ids_backend = joint_ids.to(self.device, dtype=torch.long)
        if articulation.data.has_joint_ordering:
            joint_ids_backend = articulation._joint_user_to_backend_torch[joint_ids_backend]
        joint_ids_backend = joint_ids_backend.unsqueeze(0)

        for actuator in adapter.actuators:
            ctrl = actuator.controller
            if not hasattr(ctrl, attr):
                continue
            cur_wp = articulation._root_view.get_actuator_parameter(actuator, ctrl, attr)
            cur_torch = wp.to_torch(cur_wp)
            cur_torch[env_ids_long, joint_ids_backend] = values.to(cur_torch.device, dtype=cur_torch.dtype)
            articulation._root_view.set_actuator_parameter(
                actuator=actuator,
                component=ctrl,
                name=attr,
                values=cur_wp,
                mask=env_mask,
            )
