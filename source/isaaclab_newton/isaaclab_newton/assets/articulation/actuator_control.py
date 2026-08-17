# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton import build_implicit_dof_mask
from isaaclab.actuators.newton import kernels as actuator_kernels
from isaaclab.actuators.newton.adapter import NewtonActuatorSelection
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.schemas.schemas_actuators import _validate_newton_native_actuator_cfgs

from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from .articulation import Articulation

logger = logging.getLogger(__name__)


class NewtonActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the Newton backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: Newton articulation that owns backend simulation handles.
        """
        super().__init__(articulation)

    def prepare_native_actuators(self, collection: ActuatorCollection, actuator_cfgs: dict) -> set[str]:
        articulation = self._articulation
        articulation._has_newton_actuators = False
        articulation._implicit_dof_mask = None
        articulation.newton_actuator_adapter = None

        if not getattr(articulation._sim_cfg, "use_newton_actuators", False):
            return set()

        _validate_newton_native_actuator_cfgs(actuator_cfgs)
        # Activate the Newton path even without explicit native groups: implicit-only
        # articulations still rely on it for the solver telemetry fast path.
        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True
        SimulationManager.activate_newton_actuator_path()

        return {name for name, actuator_cfg in actuator_cfgs.items() if not _is_implicit_actuator_cfg(actuator_cfg)}

    def finalize_native_actuators(self, collection: ActuatorCollection) -> NewtonActuatorSelection | None:
        if not self._native_actuator_path_active:
            return None

        articulation = self._articulation
        adapter = SimulationManager._adapter
        if adapter is not None:
            arti_start = self._joint_dof_offset()
            binding = adapter.bind_articulation(
                implicit_joint_indices=collection._implicit_group_joint_indices(),
                dof_offset=arti_start,
                num_joints=self.num_joints,
            )
            articulation.newton_actuator_adapter = adapter
            articulation._implicit_dof_mask = binding.implicit_dof_mask
            articulation._implicit_dof_mask_owner = binding.implicit_dof_mask_owner
            articulation._data._sim_bind_joint_computed_effort = binding.computed_effort_view
        else:
            articulation._implicit_dof_mask, articulation._implicit_dof_mask_owner = build_implicit_dof_mask(
                collection._implicit_group_joint_indices(),
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
                    collection._computed_effort,
                    collection._applied_effort,
                ],
                device=self.device,
            )

        SimulationManager.register_post_actuator_callback(_post_actuator)

        if adapter is None:
            return None
        joint_ordering = articulation.data.joint_ordering
        return NewtonActuatorSelection(
            view=articulation._root_view,
            actuators=adapter.actuators,
            joint_user_to_backend_indices=(
                joint_ordering.user_to_backend_indices if joint_ordering is not None else None
            ),
        )

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return self._native_actuator_path_active

    def submit_commands(self, collection: ActuatorCollection) -> None:
        articulation = self._articulation
        if self._native_actuator_path_active:
            # Newton consumes raw explicit-actuator targets through joint_act.
            user_effort = collection._joint_effort_target
            user_pos_target = collection._joint_pos_target
            user_vel_target = collection._joint_vel_target
            write_pos_target = True
            write_vel_target = True
            write_joint_act = True
            if not articulation.data.has_joint_ordering:
                articulation.data._sim_bind_joint_position_target.assign(collection._joint_pos_target)
                articulation.data._sim_bind_joint_velocity_target.assign(collection._joint_vel_target)
                articulation.data._sim_bind_joint_act.assign(collection._joint_effort_target)
                articulation.data._sim_bind_joint_effort.assign(collection._joint_effort_target)
                return
        else:
            # Lab executors publish processed targets; only implicit joints use
            # the backend position and velocity drives.
            user_effort = collection._joint_effort_target_sim
            user_pos_target = collection._joint_pos_target_sim
            user_vel_target = collection._joint_vel_target_sim
            write_pos_target = collection.has_implicit_actuators
            write_vel_target = collection.has_implicit_actuators
            write_joint_act = False
        if not articulation.data.has_joint_ordering:
            articulation.data._sim_bind_joint_effort.assign(collection._joint_effort_target_sim)
            if collection.has_implicit_actuators:
                articulation.data._sim_bind_joint_position_target.assign(collection._joint_pos_target_sim)
                articulation.data._sim_bind_joint_velocity_target.assign(collection._joint_vel_target_sim)
            return

        ordering_kernels.launch_reorder_joint_targets_user_to_backend(
            user_effort=user_effort,
            user_pos_target=user_pos_target,
            user_vel_target=user_vel_target,
            backend_to_user=articulation._joint_backend_to_user_map(),
            write_effort=True,
            write_pos_target=write_pos_target,
            write_vel_target=write_vel_target,
            write_joint_act=write_joint_act,
            backend_effort=articulation.data._sim_bind_joint_effort,
            backend_pos_target=articulation.data._sim_bind_joint_position_target,
            backend_vel_target=articulation.data._sim_bind_joint_velocity_target,
            backend_joint_act=articulation.data._sim_bind_joint_act,
            device=self.device,
        )

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        if self._native_actuator_path_active and SimulationManager._adapter is not None:
            SimulationManager._adapter.reset(env_ids)

    def _joint_dof_offset(self) -> int:
        """Return the first selected joint DOF's model offset within an environment."""
        from newton import Model as NewtonModel  # noqa: PLC0415

        dof_layout = self._articulation._root_view.frequency_layouts[NewtonModel.AttributeFrequency.JOINT_DOF]
        if dof_layout.slice is not None:
            selection_offset = dof_layout.slice.start
        elif dof_layout.indices is not None:
            selection_offset = int(dof_layout.indices.numpy()[0])
        else:
            selection_offset = 0
        return dof_layout.offset + selection_offset
