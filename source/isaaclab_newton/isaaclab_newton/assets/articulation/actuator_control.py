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
from newton import JointType

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_bam_cfg import BamActuatorCfg
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton import build_implicit_dof_mask
from isaaclab.actuators.newton import kernels as actuator_kernels
from isaaclab.actuators.newton.adapter import NewtonActuatorSelection
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.schemas.schemas_actuators import validate_newton_native_actuator_cfgs

from isaaclab_newton.actuators.bam import DriveBam, apply_bam_startup_sampling
from isaaclab_newton.assets.articulation.joint_coordinates import scatter_joint_coordinates
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_newton.physics.mjwarp_actuator_bridge import MjWarpActuatorBridge

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
        self._bam_cfgs: dict[str, BamActuatorCfg] = {}

    def prepare_native_actuators(self, collection: ActuatorCollection, actuator_cfgs: dict) -> set[str]:
        articulation = self._articulation
        articulation._has_newton_actuators = False
        articulation._implicit_dof_mask = None
        articulation.newton_actuator_adapter = None

        if not getattr(articulation._sim_cfg, "use_newton_actuators", False):
            return set()

        validate_newton_native_actuator_cfgs(actuator_cfgs)
        native_group_names = {
            name for name, actuator_cfg in actuator_cfgs.items() if not _is_implicit_actuator_cfg(actuator_cfg)
        }
        if not native_group_names:
            return set()

        self._bam_cfgs = {name: cfg for name, cfg in actuator_cfgs.items() if isinstance(cfg, BamActuatorCfg)}
        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True
        # BAM shares supply sag and command delay over an environment's DOFs. The first articulation to
        # activate the path creates every drive's state, so the stride is set for all BAM drives here.
        model_actuators = SimulationManager.backend.model.actuators if SimulationManager.backend is not None else []
        for actuator in model_actuators:
            if isinstance(actuator.drive, DriveBam):
                actuator.drive.env_dof_stride = len(actuator.indices) // self.num_instances
        SimulationManager.activate_newton_actuator_path()

        return native_group_names

    def finalize_native_actuators(self, collection: ActuatorCollection) -> NewtonActuatorSelection | None:
        if not self._native_actuator_path_active:
            return None

        articulation = self._articulation
        # BAM's viscous friction is constant, so it lives in the joint model and survives property resyncs.
        for name, cfg in self._bam_cfgs.items():
            articulation.write_joint_viscous_friction_coefficient_to_sim_index(
                joint_viscous_friction_coeff=cfg.motor.friction_viscous,
                joint_ids=collection._group_joint_indices[name],
            )
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

        SimulationManager.register_post_actuator_callback(_post_actuator)
        if self._bam_cfgs:
            self._sample_bam_startup_parameters()
            # The MJWarp binding needs the solver, which does not exist yet.
            SimulationManager.register_solver_init_callback(self._bind_bam_actuators)

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
        if self._native_actuator_path_active and SimulationManager._adapter is not None:
            SimulationManager._adapter.reset(env_ids)

    def _bam_actuators(self) -> list:
        """Return the Newton actuators with a BAM drive that act on this articulation.

        Newton merges structurally identical actuators across articulations, so an actuator
        belongs here when any of its DOFs falls inside this articulation's DOF block.
        """
        adapter = SimulationManager._adapter
        if adapter is None:
            return []
        first_dof = self._joint_dof_offset()
        last_dof = first_dof + self.num_joints
        owned = []
        for actuator in adapter.actuators:
            if not isinstance(actuator.drive, DriveBam):
                continue
            local_dofs = actuator.indices.numpy() % adapter.num_joints
            if ((local_dofs >= first_dof) & (local_dofs < last_dof)).any():
                owned.append(actuator)
        return owned

    def _sample_bam_startup_parameters(self) -> None:
        """Sample the per-environment BAM supply voltage and sag gain once per Newton actuator.

        Raises:
            ValueError: If BAM groups sharing a Newton actuator disagree on settings that Newton
                does not use to group actuators.
        """
        settings = {(cfg.vin_range, cfg.vin_drop_gain_range, cfg.stiff_frictionloss) for cfg in self._bam_cfgs.values()}
        if len(settings) > 1:
            raise ValueError(
                "BAM groups on one articulation must agree on 'vin_range', 'vin_drop_gain_range' and"
                " 'stiff_frictionloss', because one Newton actuator may cover several groups."
            )
        (setting,) = settings
        cfg = next(iter(self._bam_cfgs.values()))
        for actuator in self._bam_actuators():
            drive = actuator.drive
            if drive.startup_settings is None:
                drive.startup_settings = setting
                apply_bam_startup_sampling(drive, cfg)
            elif drive.startup_settings != setting:
                raise ValueError(
                    "Articulations sharing a Newton BAM actuator must agree on 'vin_range',"
                    f" 'vin_drop_gain_range' and 'stiff_frictionloss' (got {drive.startup_settings} and {setting})."
                )

    def _bind_bam_backlash(self, actuator) -> None:
        """Resolve each servo's sibling play hinge to its model coordinate index.

        Raises:
            ValueError: If a servo or its ``passive_<joint>_backlash`` sibling is missing or not revolute.
        """
        model = SimulationManager.backend.model
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        joint_types = model.joint_type.numpy()
        worlds = model.joint_world.numpy()
        joints = {(int(world), label): index for index, (world, label) in enumerate(zip(worlds, model.joint_label))}
        dof_to_joint = {int(start): index for index, start in enumerate(qd_start[:-1])}
        indices = []
        for dof in actuator.indices.numpy():
            joint = dof_to_joint[int(dof)]
            parent, _, name = model.joint_label[joint].rpartition("/")
            twin_label = f"{parent}/passive_{name}_backlash"
            twin = joints.get((int(worlds[joint]), twin_label))
            if twin is None or joint_types[joint] != JointType.REVOLUTE or joint_types[twin] != JointType.REVOLUTE:
                raise ValueError(f"BAM backlash requires a revolute servo and sibling play hinge: {twin_label}")
            indices.append(int(q_start[twin]))
        actuator.drive.backlash_pos_indices = wp.array(indices, dtype=wp.uint32, device=self.device)

    def _bind_bam_actuators(self) -> None:
        """Bind this articulation's BAM actuators to the MJWarp solver, once per Newton actuator.

        Each step, the pre-actuator hook gathers the previous solve's external load and the
        post-actuator hook publishes the drive's friction budget before the substeps.

        Raises:
            ValueError: If the active solver is not MJWarp.
        """
        solver = SimulationManager._solver
        if not MjWarpActuatorBridge.is_available(solver):
            raise ValueError("BAM actuators require Newton's MJWarp solver (MJWarpSolverCfg).")
        stiff_frictionloss = next(iter(self._bam_cfgs.values())).stiff_frictionloss
        model = SimulationManager.backend.model
        for actuator in self._bam_actuators():
            drive = actuator.drive
            if drive.external_torque is not None:
                continue
            if drive.has_backlash:
                self._bind_bam_backlash(actuator)
            bridge = MjWarpActuatorBridge(solver, model, actuator.indices, self.device)
            drive.external_torque = wp.zeros(actuator.num_actuators, dtype=wp.float32, device=self.device)
            if stiff_frictionloss:
                bridge.stiffen_friction_constraint()
            SimulationManager.register_pre_actuator_callback(
                lambda bridge=bridge, out=drive.external_torque: bridge.gather_external_torque(out)
            )
            SimulationManager.register_post_actuator_callback(
                lambda bridge=bridge, drive=drive: bridge.publish_dof_friction(drive.friction_budget)
            )

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
