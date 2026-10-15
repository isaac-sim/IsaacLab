# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton import Model
from newton.actuators import DriveBAM, InputProcessorBacklash, InputProcessorDelay

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_bam_cfg import BamActuatorCfg
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton import build_implicit_dof_mask
from isaaclab.actuators.newton import kernels as actuator_kernels
from isaaclab.actuators.newton.adapter import NewtonActuatorAdapter, NewtonActuatorSelection
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.schemas.schemas_actuators import validate_newton_native_actuator_cfgs

from isaaclab_newton.actuators.bam import BamCommandLag, BamFriction, BamGroup, InputProcessorPositionBacklash
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
        self.bam_groups: dict[str, BamGroup] = {}
        """Isaac Lab-side runtime of each Newton-executed BAM group, keyed by group name."""

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
            if adapter is None:
                raise RuntimeError(f"BAM groups {sorted(self._bam_cfgs)} have no Newton actuators.")
            self._create_bam_groups(collection, adapter)
            # The MJWarp binding needs the solver, which does not exist yet.
            SimulationManager.register_solver_init_callback(self._bind_bam_friction)

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

    def _create_bam_groups(self, collection: ActuatorCollection, adapter: NewtonActuatorAdapter) -> None:
        """Create the supply, command lag and friction state of each BAM group.

        Each group gets one battery per environment, a lag stream per environment and delay processor when it
        has a delay, and per-DOF friction state. Their resets follow every adapter reset.

        Raises:
            RuntimeError: If a group's DOFs are not driven by Newton's BAM drive with its authored delay and
                backlash processors.
        """
        dofs_per_env = adapter.num_joints
        bam_dofs: set[int] = set()
        delay_slots: dict[int, tuple[InputProcessorDelay, int]] = {}
        backlash_types: dict[int, type] = {}
        for actuator in adapter.actuators:
            if not isinstance(actuator.drive, DriveBAM):
                continue
            indices = actuator.indices.numpy().tolist()
            bam_dofs.update(indices)
            for processor in actuator.input_processors:
                if isinstance(processor, InputProcessorDelay):
                    delay_slots.update((dof, (processor, slot)) for slot, dof in enumerate(indices))
                elif isinstance(processor, InputProcessorBacklash):
                    backlash_types.update(dict.fromkeys(indices, type(processor)))

        # Group DOFs, environment-major and in group joint order. Instance e lives in environment e.
        all_joints = list(range(self.num_joints))
        local_dofs = {}
        for name in self._bam_cfgs:
            ids = collection._group_joint_indices[name]
            ids = all_joints[ids] if isinstance(ids, slice) else ids.tolist()
            backend_ids = np.asarray(self._articulation.map_joint_ids_to_backend(ids))
            local_dofs[name] = self._joint_dof_offset() + backend_ids
        env_offsets = np.arange(self.num_instances)[:, None] * dofs_per_env

        for name, cfg in self._bam_cfgs.items():
            dofs = env_offsets + local_dofs[name]
            if not bam_dofs.issuperset(dofs.ravel().tolist()):
                raise RuntimeError(f"BAM group '{name}' is not driven by Newton's DriveBAM.")
            if cfg.backlash_joint_template is not None:
                # An unregistered processor token is skipped silently when Newton parses the USD.
                expected = InputProcessorBacklash if cfg.backlash_velocity_feedback else InputProcessorPositionBacklash
                if any(backlash_types.get(dof) is not expected for dof in dofs.ravel().tolist()):
                    raise RuntimeError(f"BAM group '{name}' is not driven through a {expected.__name__} processor.")
            nominal_voltage, sag_gain = self._sample_bam_supply(cfg)
            battery_offset = adapter.add_batteries(dofs, nominal_voltage, sag_gain, min_voltage=max(cfg.vin_min, 0.0))
            friction = BamFriction(cfg.motor, wp.array(dofs.ravel(), dtype=wp.uint32, device=self.device), dofs_per_env)
            adapter.register_reset_callback(friction.reset)

            command_lags = []
            if cfg.max_delay > 0:
                # One lag per delay processor, so groups merged into one Newton actuator keep their own settings.
                lag_slots: dict[int, tuple[InputProcessorDelay, list[int], list[int]]] = {}
                for dof in dofs.ravel().tolist():
                    if dof not in delay_slots:
                        raise RuntimeError(f"BAM group '{name}' has max_delay > 0 but no Newton delay processor.")
                    delay, slot = delay_slots[dof]
                    entry = lag_slots.setdefault(id(delay), (delay, [], []))
                    entry[1].append(slot)
                    entry[2].append(dof)
                for delay, slots, lag_dofs in lag_slots.values():
                    lag = BamCommandLag(
                        delay,
                        slots,
                        lag_dofs,
                        dofs_per_env,
                        min_delay=cfg.min_delay,
                        max_delay=cfg.max_delay,
                        hold_prob=cfg.delay_hold_prob,
                        update_period=cfg.delay_update_period,
                    )
                    adapter.register_reset_callback(lag.reset)
                    SimulationManager.register_pre_actuator_callback(lag.step)
                    command_lags.append(lag)
            self.bam_groups[name] = BamGroup(
                num_envs=self.num_instances,
                friction=friction,
                command_lags=command_lags,
                adapter=adapter,
                battery_offset=battery_offset,
            )

    def _sample_bam_supply(self, cfg: BamActuatorCfg) -> tuple[wp.array, wp.array]:
        """Sample the per-environment supply voltage and sag gain, held across resets.

        Both ranges are drawn from the global torch generator, voltage first, with shape ``(num_envs, 1)``.

        Returns:
            The nominal voltage [V] and sag gain [V/(N.m)] of each environment, shape ``(num_envs,)`` each.
        """
        samples = []
        for value_range, default in ((cfg.vin_range, cfg.vin), (cfg.vin_drop_gain_range, 0.0)):
            values = torch.full((self.num_instances, 1), default, device=self.device, dtype=torch.float32)
            if value_range is not None:
                values.uniform_(*value_range)
            samples.append(wp.from_torch(values.view(-1)))
        return samples[0], samples[1]

    def _bind_bam_friction(self) -> None:
        """Bind each BAM group's friction to the MJWarp solver.

        Each step, the pre-actuator hook gathers the previous solve's external load and computes the friction
        budget, and the post-actuator hook publishes the budget and records the applied effort before the
        substeps.

        Raises:
            ValueError: If the active solver is not MJWarp or runs CPU MuJoCo (``use_mujoco_cpu=True``).
        """
        solver = SimulationManager._solver
        if not MjWarpActuatorBridge.is_available(solver):
            raise ValueError(
                "BAM actuators require Newton's MJWarp solver (MJWarpSolverCfg with use_mujoco_cpu=False)."
            )
        model = SimulationManager.backend.model
        for name, group in self.bam_groups.items():
            friction = group.friction
            bridge = MjWarpActuatorBridge(solver, model, friction.dof_indices, self.device)
            if self._bam_cfgs[name].stiff_frictionloss:
                bridge.stiffen_friction_constraint()

            def update_budget(bridge: MjWarpActuatorBridge = bridge, friction: BamFriction = friction) -> None:
                bridge.gather_external_torque(friction.external_torque)
                friction.compute_budget(SimulationManager.backend.state_0.joint_qd)

            def publish_budget(bridge: MjWarpActuatorBridge = bridge, friction: BamFriction = friction) -> None:
                bridge.publish_dof_friction(friction.friction_budget)
                # The adapter zeroes the actuated DOFs of joint_f before the actuators add their clamped effort.
                friction.record_applied_effort(SimulationManager.backend.control.joint_f)

            SimulationManager.register_pre_actuator_callback(update_budget)
            SimulationManager.register_post_actuator_callback(publish_budget)

    def _joint_dof_offset(self) -> int:
        """Return the first selected joint DOF's model offset within an environment."""
        dof_layout = self._articulation._root_view.frequency_layouts[Model.AttributeFrequency.JOINT_DOF]
        if dof_layout.slice is not None:
            selection_offset = dof_layout.slice.start
        elif dof_layout.indices is not None:
            selection_offset = int(dof_layout.indices.numpy()[0])
        else:
            selection_offset = 0
        return dof_layout.offset + selection_offset
