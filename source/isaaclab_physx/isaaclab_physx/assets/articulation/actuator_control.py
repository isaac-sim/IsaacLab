# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX actuator control adapter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton.adapter import LightArticulationView, NewtonActuatorSelection
from isaaclab.actuators.newton.physx_runtime import PhysxActuatorRuntime
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.schemas.schemas_actuators import _validate_newton_native_actuator_cfgs
from isaaclab.sim.utils.queries import find_first_matching_prim
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from .articulation import Articulation

logger = logging.getLogger(__name__)


class PhysxActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the PhysX backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: PhysX articulation that owns backend simulation handles.
        """
        super().__init__(articulation)
        self._actuator_runtime = None

    @property
    def _physx_actuator_wrapper(self):
        """Preserve the compatibility wrapper used by native actuator callers."""
        return None if self._actuator_runtime is None else self._actuator_runtime.wrapper

    @property
    def _native_actuator_graphs(self):
        """Expose graph capture state used by native-actuator benchmarks."""
        return None if self._actuator_runtime is None else self._actuator_runtime.native_actuator_graphs

    def prepare_native_actuators(self, collection: ActuatorCollection, actuator_cfgs: dict) -> set[str]:
        articulation = self._articulation
        self._actuator_runtime = None
        articulation._physx_actuator_wrapper = None
        articulation.newton_actuator_adapter = None
        articulation._implicit_dof_mask = None
        articulation._has_newton_actuators = False

        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if not use_newton_actuators:
            return set()

        _validate_newton_native_actuator_cfgs(actuator_cfgs)

        # Activate the runtime even without explicit native groups: implicit-only
        # articulations still rely on it for the solver telemetry fast path.
        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True
        native_group_names = {
            name for name, actuator_cfg in actuator_cfgs.items() if not _is_implicit_actuator_cfg(actuator_cfg)
        }

        first_prim = find_first_matching_prim(articulation.cfg.prim_path)
        art_prim_path = str(first_prim.GetPath()) if first_prim is not None else None
        self._actuator_runtime = PhysxActuatorRuntime(articulation, logger=logger)
        self._actuator_runtime.prepare(
            collection,
            stage=get_current_stage(),
            articulation_prim_path=art_prim_path,
            adapt_usd_actuators=bool(native_group_names),
        )
        articulation._physx_actuator_wrapper = self._actuator_runtime.wrapper
        articulation.newton_actuator_adapter = self._actuator_runtime.adapter

        return native_group_names

    def finalize_native_actuators(self, collection: ActuatorCollection) -> NewtonActuatorSelection | None:
        if not self._native_actuator_path_active or self._actuator_runtime is None:
            return None
        self._actuator_runtime.finalize(collection)
        adapter = self._actuator_runtime.adapter
        if adapter is None:
            return None
        articulation = self._articulation
        return NewtonActuatorSelection(
            view=LightArticulationView(articulation.num_instances, articulation.num_joints, articulation.device),
            actuators=adapter.actuators,
        )

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        if not self._native_actuator_path_active or self._actuator_runtime is None:
            return False

        articulation = self._articulation
        if articulation.data.has_joint_ordering:
            # Non-identity ordering binds the wrapper to public shadow buffers. Refresh before
            # stepping so the native controller observes this PhysX step's state.
            articulation._data._refresh_joint_pos()
            articulation._data._refresh_joint_vel()
        self._actuator_runtime.compute(collection, dt)
        return True

    def submit_commands(self, collection: ActuatorCollection) -> None:
        articulation = self._articulation
        # The articulation flag selects the native wrapper's command buffers.
        if getattr(articulation, "_has_newton_actuators", False):
            # Newton fast path: pos/vel targets pass straight through; ``joint_f_2d`` already
            # merges Newton's explicit-DOF output with user feedforward.
            user_effort = articulation._physx_actuator_wrapper.joint_f_2d
            user_pos_target = collection._joint_pos_target
            user_vel_target = collection._joint_vel_target
        else:
            # Standard Lab actuator path: push the processed staging buffers PhysX-side.
            user_effort = collection._joint_effort_target_sim
            user_pos_target = collection._joint_pos_target_sim
            user_vel_target = collection._joint_vel_target_sim

        if articulation.data.has_joint_ordering:
            ordering_kernels.launch_reorder_joint_targets_user_to_backend(
                user_effort=user_effort,
                user_pos_target=user_pos_target,
                user_vel_target=user_vel_target,
                backend_to_user=articulation.data.joint_ordering.backend_to_user,
                write_effort=True,
                write_pos_target=articulation._has_implicit_actuators,
                write_vel_target=articulation._has_implicit_actuators,
                write_joint_act=False,
                backend_effort=articulation._joint_effort_target_backend,
                backend_pos_target=articulation._joint_pos_target_backend,
                backend_vel_target=articulation._joint_vel_target_backend,
                backend_joint_act=None,
                device=self.device,
            )
            effort_target = articulation._joint_effort_target_backend
            pos_target = articulation._joint_pos_target_backend
            vel_target = articulation._joint_vel_target_backend
        else:
            effort_target = user_effort
            pos_target = user_pos_target
            vel_target = user_vel_target

        articulation.root_view.set_dof_actuation_forces(effort_target, articulation._ALL_INDICES)
        if articulation._has_implicit_actuators:
            articulation.root_view.set_dof_position_targets(pos_target, articulation._ALL_INDICES)
            articulation.root_view.set_dof_velocity_targets(vel_target, articulation._ALL_INDICES)

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        if self._native_actuator_path_active and self._actuator_runtime is not None:
            self._actuator_runtime.reset(env_ids)
