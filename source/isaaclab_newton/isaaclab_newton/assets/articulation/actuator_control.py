# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton import build_implicit_dof_mask
from isaaclab.actuators.newton import kernels as actuator_kernels
from isaaclab.actuators.newton.adapter import NewtonActuatorSelection
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.schemas.schemas_actuators import validate_newton_native_actuator_cfgs
from isaaclab.utils import index_fill_

from isaaclab_newton.assets.articulation.joint_coordinates import scatter_joint_coordinates
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_newton.physics import StepPhase

if TYPE_CHECKING:
    from .articulation import Articulation

logger = logging.getLogger(__name__)


class _DofBuffer:
    """Present a flat per-DOF buffer as ``joint_f`` so an articulation view can select its own DOFs."""

    def __init__(self, joint_f: wp.array):
        self.joint_f = joint_f


@wp.kernel(enable_backward=False)
def _select_row_dofs(row_mask: wp.array(dtype=wp.bool), dof_mask: wp.array2d(dtype=wp.bool)):
    """Select every DOF of the masked articulation rows in a flat DOF mask viewed per row."""
    row, joint = wp.tid()
    dof_mask[row, joint] = row_mask[row]


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
            # Isaac Lab actuator models compute efforts on the host before every physics step.
            if any(not _is_implicit_actuator_cfg(actuator_cfg) for actuator_cfg in actuator_cfgs.values()):
                SimulationManager.require_host_physics_steps()
            return set()

        validate_newton_native_actuator_cfgs(actuator_cfgs)
        native_group_names = {
            name for name, actuator_cfg in actuator_cfgs.items() if not _is_implicit_actuator_cfg(actuator_cfg)
        }
        if not native_group_names:
            return set()

        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True
        SimulationManager.activate_newton_actuator_path()

        return native_group_names

    def finalize_native_actuators(self, collection: ActuatorCollection) -> NewtonActuatorSelection | None:
        if not self._native_actuator_path_active:
            return None

        articulation = self._articulation
        adapter = SimulationManager.get_actuator_adapter()
        if adapter is not None:
            # View the adapter's flat DOF buffers through this articulation's own layout, which stays correct
            # when worlds hold different robots.
            view = articulation._root_view
            computed_effort = view.get_attribute("joint_f", _DofBuffer(adapter.computed_effort))[:, 0]
            self._reset_dof_mask = wp.zeros(adapter.computed_effort.shape[0], dtype=wp.bool, device=self.device)
            self._reset_dof_view = view.get_attribute("joint_f", _DofBuffer(self._reset_dof_mask))[:, 0]
            self._reset_row_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
            binding = adapter.bind_articulation(
                implicit_joint_indices=collection._implicit_group_joint_indices(),
                dof_offset=self._joint_dof_offset(),
                num_joints=self.num_joints,
                computed_effort_view=computed_effort,
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
            # Telemetry reads _sim_bind_joint_pos inside the decimation loop, ahead of the
            # post-step gather, so the DOF-space view is re-derived here.
            articulation._data._gather_joint_coordinates()
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

        SimulationManager.add_stage(_post_actuator, StepPhase.CONTROL, name="articulation.actuator_telemetry")

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
        """Publish the collection's targets to the backend arrays."""
        articulation = self._articulation
        data = articulation.data
        if self._native_actuator_path_active:
            # Newton consumes raw explicit-actuator targets through joint_act.
            user_effort = collection._joint_effort_target
            user_pos_target = collection._joint_pos_target
            user_vel_target = collection._joint_vel_target
            write_pos_target = True
            write_vel_target = True
            write_joint_act = True
        else:
            # Lab executors publish processed targets; only implicit joints use
            # the backend position and velocity drives.
            user_effort = collection._joint_effort_target_sim
            user_pos_target = collection._joint_pos_target_sim
            user_vel_target = collection._joint_vel_target_sim
            write_pos_target = collection.has_implicit_actuators
            write_vel_target = collection.has_implicit_actuators
            write_joint_act = False

        if data.has_joint_ordering:
            ordering_kernels.launch_reorder_joint_targets_user_to_backend(
                user_effort=user_effort,
                user_pos_target=user_pos_target,
                user_vel_target=user_vel_target,
                backend_to_user=articulation._joint_backend_to_user_map(),
                write_effort=True,
                write_pos_target=write_pos_target,
                write_vel_target=write_vel_target,
                write_joint_act=write_joint_act,
                backend_effort=data._sim_bind_joint_effort,
                backend_pos_target=data._sim_bind_joint_position_target,
                backend_vel_target=data._sim_bind_joint_velocity_target,
                backend_joint_act=data._sim_bind_joint_act,
                device=self.device,
            )
        else:
            data._sim_bind_joint_effort.assign(user_effort)
            if write_pos_target:
                data._sim_bind_joint_position_target.assign(user_pos_target)
            if write_vel_target:
                data._sim_bind_joint_velocity_target.assign(user_vel_target)
            if write_joint_act:
                data._sim_bind_joint_act.assign(user_effort)

        # Newton takes coordinate-layout position targets from 1.6 on; every branch above writes
        # DOF-indexed targets, so the staging buffer has to be scattered across unconditionally.
        # Sequenced after the writes above (not a try/finally) so an exception mid-write leaves the
        # staging buffer unflushed rather than scattering a half-written buffer into joint_target_q.
        if articulation.data._joint_targets_need_conversion:
            scatter_joint_coordinates(
                articulation.data._joint_coord_map,
                articulation.data._sim_bind_joint_position_target,
                articulation.data._sim_bind_joint_target_coords,
                articulation._ALL_ENV_MASK,
            )

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        adapter = SimulationManager.get_actuator_adapter()
        if not self._native_actuator_path_active or adapter is None:
            return
        row_mask = self._reset_row_mask
        row_mask.zero_()
        index_fill_(row_mask, env_ids, True)
        wp.launch(
            _select_row_dofs,
            dim=self._reset_dof_view.shape,
            inputs=[wp.from_torch(row_mask, dtype=wp.bool)],
            outputs=[self._reset_dof_view],
            device=self.device,
        )
        adapter.reset_dofs(self._reset_dof_mask)

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
