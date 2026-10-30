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
from newton import Model, ModelFlags
from newton.actuators import Actuator, BatteryConstantResistance, InputProcessorDelay
from newton.solvers import SolverMuJoCo

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_bam_cfg import BamActuatorCfg
from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton import build_implicit_dof_mask
from isaaclab.actuators.newton import kernels as actuator_kernels
from isaaclab.actuators.newton.adapter import NewtonActuatorAdapter, NewtonActuatorSelection
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.schemas.schemas_actuators import validate_newton_native_actuator_cfgs

from isaaclab_newton.actuators.bam import BamCommandLag, BamFriction, BamGroup
from isaaclab_newton.assets.articulation.joint_coordinates import scatter_joint_coordinates

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
        self._articulation._backend_manager.activate_newton_actuator_path()

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
        adapter = self._articulation._backend_manager.backend.actuators
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

        self._articulation._backend_manager.register_post_actuator_callback(_post_actuator)
        if self._bam_cfgs:
            if adapter is None:
                raise RuntimeError(f"BAM groups {sorted(self._bam_cfgs)} have no Newton actuators.")
            self._create_bam_groups(collection, adapter)
            # The friction needs the solver's load observables, and the solver does not exist yet.
            self._articulation._backend_manager.register_solver_init_callback(self._bind_bam_friction)

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
        if self._native_actuator_path_active and self._articulation._backend_manager.backend.actuators is not None:
            self._articulation._backend_manager.backend.actuators.reset(env_ids)

    def _create_bam_groups(self, collection: ActuatorCollection, adapter: NewtonActuatorAdapter) -> None:
        """Bind the supply, command lag and friction state of each BAM group.

        Each group's servos draw from one Newton battery per environment, whose nominal voltage and resistance are
        sampled here. A group with a delay gets a lag stream per environment, and every group gets per-DOF friction
        state. Their resets follow every adapter reset.
        """
        # Newton actuator and slot, and supply row, of each DOF whose actuator has supply rows, such as a motor's.
        servos: dict[int, tuple[Actuator, int]] = {}
        supply_rows: dict[int, int] = {}
        for actuator in adapter.actuators:
            if actuator.actuator_indices is not None:
                rows = actuator.actuator_indices.numpy().tolist()
                for slot, dof in enumerate(actuator.indices.numpy().tolist()):
                    servos[dof] = (actuator, slot)
                    supply_rows[dof] = rows[slot]
        battery = next(battery for battery in adapter.batteries if isinstance(battery, BatteryConstantResistance))
        all_joints = list(range(self.num_joints))
        env_offsets = np.arange(self.num_instances)[:, None] * adapter.num_joints

        for name, cfg in self._bam_cfgs.items():
            ids = collection._group_joint_indices[name]
            ids = all_joints[ids] if isinstance(ids, slice) else ids.tolist()
            # Group DOFs, environment-major and in group joint order. Instance e lives in environment e.
            dofs = env_offsets + self._joint_dof_offset() + np.asarray(self._articulation.map_joint_ids_to_backend(ids))
            friction = BamFriction(
                cfg.motor, wp.array(dofs.ravel(), dtype=wp.uint32, device=self.device), adapter.num_joints
            )
            adapter.register_reset_callback(friction.reset)
            if cfg.max_delay > 0:
                self._create_bam_command_lag(cfg, dofs, servos, adapter)
            group = BamGroup(
                num_envs=self.num_instances,
                friction=friction,
                battery=battery,
                battery_ids=_bam_battery_ids(battery, np.vectorize(supply_rows.__getitem__)(dofs)),
            )
            nominal_voltage, resistance = self._sample_bam_supply(cfg)
            wp.to_torch(group.nominal_voltage).copy_(nominal_voltage)
            wp.to_torch(group.battery_resistance).copy_(resistance)
            # The model's control starts at the authored nominal voltage; supply the sampled one from the first step.
            battery.reset(self._articulation._backend_manager.backend.control)
            self.bam_groups[name] = group

    def _create_bam_command_lag(
        self,
        cfg: BamActuatorCfg,
        dofs: np.ndarray,
        servos: dict[int, tuple[Actuator, int]],
        adapter: NewtonActuatorAdapter,
    ) -> None:
        """Draw a BAM group's command lag into the delay processor of the Newton actuator driving it.

        Groups that Newton merges into one actuator write their own slots, so each keeps its lag settings.

        Args:
            cfg: Group configuration.
            dofs: Group DOFs, shape ``(num_envs, num_group_joints)``.
            servos: Newton actuator and slot of each servo DOF.
            adapter: Adapter whose actuators drive the group and whose resets restart the lag.
        """
        actuator = servos[int(dofs[0, 0])][0]
        assert all(servos[dof][0] is actuator for dof in dofs.ravel().tolist()), "BAM group split across actuators"
        lag = BamCommandLag(
            next(processor for processor in actuator.input_processors if isinstance(processor, InputProcessorDelay)),
            [servos[dof][1] for dof in dofs.ravel().tolist()],
            dofs.ravel(),
            adapter.num_joints,
            min_delay=cfg.min_delay,
            max_delay=cfg.max_delay,
            hold_prob=cfg.delay_hold_prob,
            update_period=cfg.delay_update_period,
        )
        adapter.register_reset_callback(lag.reset)
        self._articulation._backend_manager.register_pre_actuator_callback(lag.step)

    def _sample_bam_supply(self, cfg: BamActuatorCfg) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the per-environment nominal supply voltage and battery resistance, held across resets.

        Both ranges are drawn from the global torch generator, voltage first, with shape ``(num_envs, 1)``.

        Returns:
            The nominal voltage [V] and battery resistance [Ohm] of each environment, shape ``(num_envs,)`` each.
        """
        samples = []
        for value_range, default in ((cfg.vin_range, cfg.vin), (cfg.battery_resistance_range, cfg.battery_resistance)):
            values = torch.full((self.num_instances, 1), default, device=self.device, dtype=torch.float32)
            if value_range is not None:
                values.uniform_(*value_range)
            samples.append(values.view(-1))
        return samples[0], samples[1]

    def _bind_bam_friction(self) -> None:
        """Feed each BAM group's friction budget from the solver's load observables.

        Each step, the pre-actuator hook computes the budget from the previous solve's loads and publishes it as the
        servos' joint friction, and the post-actuator hook records the applied effort for the next budget.

        Raises:
            ValueError: If the active solver is not MuJoCo Warp, including CPU MuJoCo (``use_mujoco_cpu=True``).
        """
        solver = self._articulation._backend_manager.backend.solver
        if not isinstance(solver, SolverMuJoCo) or solver.use_mujoco_cpu:
            raise ValueError(
                "BAM actuators require Newton's MJWarp solver (MJWarpSolverCfg with use_mujoco_cpu=False)."
            )
        self._articulation._backend_manager.request_solver_observables(
            {"qfrc_bias", "qfrc_constraint", "qfrc_frictionloss"}
        )
        # The solver applies the published budgets once it re-reads the joint friction.
        self._articulation._backend_manager.request_solver_model_sync(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        model = self._articulation._backend_manager.backend.model
        for group in self.bam_groups.values():

            def update_budget(friction: BamFriction = group.friction) -> None:
                loads = self._articulation._backend_manager.backend.solver_observables
                friction.compute_budget(
                    self._articulation._backend_manager.backend.state_0.joint_qd,
                    loads.qfrc_bias,
                    loads.qfrc_constraint,
                    loads.qfrc_frictionloss,
                    model.joint_friction,
                )

            def record_effort(friction: BamFriction = group.friction) -> None:
                # The adapter zeroes the actuated DOFs of joint_f before the actuators add their clamped effort.
                friction.record_applied_effort(self._articulation._backend_manager.backend.control.joint_f)

            self._articulation._backend_manager.register_pre_actuator_callback(update_budget)
            self._articulation._backend_manager.register_post_actuator_callback(record_effort)

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


def _bam_battery_ids(battery: BatteryConstantResistance, rows: np.ndarray) -> slice:
    """Return the indices of a BAM group's batteries in *battery*, one per environment.

    Isaac Lab authors one battery prim per group that powers exactly its servos, and Newton replicates it per
    environment, numbering batteries environment-major.

    Args:
        battery: Newton's constant-resistance batteries.
        rows: Supply row of each group servo, shape ``(num_envs, num_group_joints)``.

    Returns:
        The evenly spaced indices of each environment's battery in *battery*.
    """
    counts = np.diff(battery.actuator_start.numpy())
    battery_of_row = dict(zip(battery.actuators.numpy().tolist(), np.repeat(np.arange(len(counts)), counts).tolist()))
    owners = np.vectorize(battery_of_row.__getitem__)(rows)
    stride = int(owners[1, 0] - owners[0, 0]) if len(owners) > 1 else 1
    ids = owners[0, 0] + stride * np.arange(len(owners))
    assert stride > 0 and (owners == ids[:, None]).all() and (counts[ids] == rows.shape[1]).all(), (
        f"BAM battery layout mismatch: {owners.tolist()}"
    )
    return slice(int(ids[0]), int(ids[-1]) + 1, stride)
