# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton actuator control adapter."""

from __future__ import annotations

import weakref
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
from isaaclab.sim.schemas.schemas_actuators import validate_newton_native_actuator_cfgs

from isaaclab_newton.assets.articulation.joint_coordinates import scatter_joint_coordinates
from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from .articulation import Articulation

_BAM_ACTUATOR_SETTINGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""BAM settings each Newton actuator was set up with, keyed by the actuator.

Newton merges structurally identical actuators across articulations, so a multi-robot scene can
have one actuator reached by several :class:`NewtonActuatorControl` instances. The entry marks
the actuator as already set up and records the settings, so a second articulation that
configures it differently fails loudly instead of silently losing one of the two. Weak keys, so
the map empties itself with the model.
"""


class NewtonActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the Newton backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: Newton articulation that owns backend simulation handles.
        """
        super().__init__(articulation)
        self._native_actuator_cfgs: dict = {}

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

        self._native_actuator_cfgs = dict(actuator_cfgs)
        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True
        SimulationManager.activate_newton_actuator_path()

        return native_group_names

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
        # Start-up randomization only needs the actuators, so it runs now. Publishing friction
        # to the solver needs a solver, which does not exist while assets are initializing.
        self._sample_bam_startup_parameters()
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
        """Return the Newton actuators driving this articulation with a BAM controller.

        The actuator adapter is simulation-global and Newton merges structurally identical
        actuators across articulations, so the adapter's list is not this articulation's list.
        Actuator indices are global Newton DOFs in environment-major order, so an actuator
        belongs here when any of its per-environment DOF offsets falls inside this
        articulation's block.
        """
        from isaaclab.actuators.newton import ControllerBam  # noqa: PLC0415

        adapter = SimulationManager._adapter
        if adapter is None:
            return []
        first_dof = self._joint_dof_offset()
        last_dof = first_dof + self.num_joints
        owned = []
        for actuator in adapter.actuators:
            if not isinstance(actuator.controller, ControllerBam):
                continue
            local_dofs = actuator.indices.numpy() % adapter.num_joints
            if ((local_dofs >= first_dof) & (local_dofs < last_dof)).any():
                owned.append(actuator)
        return owned

    def _bam_settings(self) -> tuple:
        """Return the BAM settings of this articulation that a Newton actuator cannot carry.

        The start-up randomization ranges and ``stiff_frictionloss`` are not part of Newton's
        actuator-grouping key, so one Newton actuator may span several Lab groups -- and, in a
        multi-robot scene, several articulations. They therefore have to agree wherever they
        meet, which is what this key lets the caller check.

        Raises:
            ValueError: If this articulation's BAM groups disagree.
        """
        from isaaclab.actuators.actuator_bam_cfg import BamActuatorCfg  # noqa: PLC0415

        settings = {
            (cfg.vin_range, cfg.vin_drop_gain_range, cfg.friction_scale_range, cfg.stiff_frictionloss)
            for cfg in self._native_actuator_cfgs.values()
            if isinstance(cfg, BamActuatorCfg)
        }
        if len(settings) > 1:
            raise ValueError(
                "BAM actuator groups on one articulation must agree on 'vin_range',"
                " 'vin_drop_gain_range', 'friction_scale_range' and 'stiff_frictionloss': they are"
                " not part of Newton's actuator-grouping key, so one Newton actuator may cover"
                " several groups and there is no per-group value to apply."
            )
        return next(iter(settings)) if settings else ()

    def _sample_bam_startup_parameters(self) -> None:
        """Draw the start-up per-environment BAM quantities of this articulation's actuators.

        Runs while the model is being built, before any solver exists, because the values feed
        the controller's kernels and do not require a solver. A USD prim is shared by every
        clone, so the ranges cannot be authored and have to be drawn here.

        Each Newton actuator is sampled once. When a second articulation reaches an actuator
        that a first one already sampled, their settings must match: re-drawing would silently
        discard the first articulation's randomization, and applying the first articulation's
        ranges to the second would silently ignore its configuration.

        Raises:
            ValueError: If two articulations sharing a Newton actuator disagree on the settings.
        """
        from isaaclab.actuators.newton import apply_bam_startup_sampling  # noqa: PLC0415

        actuators = self._bam_actuators()
        if not actuators:
            return
        settings = self._bam_settings()
        cfg = self._first_bam_cfg()
        for actuator in actuators:
            previous = _BAM_ACTUATOR_SETTINGS.get(actuator)
            if previous is not None:
                if previous != settings:
                    raise ValueError(
                        "Two articulations share one Newton actuator but configure their BAM"
                        f" groups differently ({previous} vs {settings}). 'vin_range',"
                        " 'vin_drop_gain_range', 'friction_scale_range' and 'stiff_frictionloss'"
                        " are not part of Newton's actuator-grouping key, so structurally"
                        " identical robots are merged into one actuator and cannot carry"
                        " per-articulation values. Use matching start-up ranges and stiff_frictionloss"
                        " for articulations with the same shared controller settings."
                    )
                continue
            _BAM_ACTUATOR_SETTINGS[actuator] = settings
            apply_bam_startup_sampling(actuator.controller, cfg)

    def _first_bam_cfg(self):
        """Return one of this articulation's BAM configs; they agree on everything used here."""
        from isaaclab.actuators.actuator_bam_cfg import BamActuatorCfg  # noqa: PLC0415

        return next(cfg for cfg in self._native_actuator_cfgs.values() if isinstance(cfg, BamActuatorCfg))

    def _bind_bam_actuators(self) -> None:
        """Give this articulation's BAM actuators their per-step MuJoCo Warp channel.

        The BAM servo model needs two things the actuator component interface does not carry:
        it publishes a load-dependent dry-friction budget so the solver performs the stiction
        clipping natively, and it estimates the gearbox load from bias and constraint forces. Both go
        through :class:`~isaaclab_newton.physics.MjWarpActuatorBridge`, on the in-graph pre-
        and post-actuator hooks, so that the load is the previous solve's and the budget
        reaches the substeps of the same iteration.

        Runs on the solver-init hook, which is the first point at which the MuJoCo Warp model
        exists and still precedes CUDA graph capture. Each actuator is bound once: Newton merges
        structurally identical actuators, so a second articulation may reach one that is already
        bound.
        """
        from isaaclab_newton.physics.mjwarp_actuator_bridge import MjWarpActuatorBridge  # noqa: PLC0415

        actuators = self._bam_actuators()
        if not actuators:
            return
        solver = SimulationManager._solver
        if not MjWarpActuatorBridge.is_available(solver):
            raise ValueError(
                "BAM actuators require the Newton MJWarp solver (MJWarpSolverCfg) with"
                " use_newton_actuators=True; controller-side friction is not supported."
            )
        cfg = self._first_bam_cfg()
        num_newton_dofs = SimulationManager.backend.model.joint_dof_count

        for actuator in actuators:
            controller = actuator.controller
            # A bound external-torque array is the marker: it is what the bridge fills.
            if controller.external_torque is not None:
                continue
            bridge = MjWarpActuatorBridge(solver, actuator.indices, num_newton_dofs, self.device)
            external_torque = wp.zeros(actuator.num_actuators, dtype=wp.float32, device=self.device)
            controller.external_torque = external_torque
            if cfg.stiff_frictionloss:
                bridge.stiffen_friction_constraint()

            SimulationManager.register_pre_actuator_callback(
                lambda bridge=bridge, out=external_torque: bridge.gather_external_torque(out)
            )
            SimulationManager.register_post_actuator_callback(
                lambda bridge=bridge, ctrl=controller: bridge.publish_dof_friction(
                    ctrl.friction_budget, ctrl.viscous_damping
                )
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
