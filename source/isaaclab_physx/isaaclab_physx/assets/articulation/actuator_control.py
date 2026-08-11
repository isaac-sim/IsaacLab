# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX actuator control adapter."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.actuators import ActuatorBase, ActuatorCollection
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.assets.articulation import ordering_kernels
from isaaclab.sim.utils.queries import find_first_matching_prim

from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    from .articulation import Articulation

_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None

logger = logging.getLogger(__name__)


class PhysxActuatorControl(ArticulationActuatorControl):
    """Actuator control adapter for the PhysX backend."""

    def __init__(self, articulation: Articulation):
        """Initialize the control adapter.

        Args:
            articulation: PhysX articulation that owns backend simulation handles.
        """
        super().__init__(articulation)
        self._physx_actuator_wrapper = None
        self._all_env_mask: wp.array(dtype=wp.bool) | None = None
        self._all_joint_mask: wp.array(dtype=wp.bool) | None = None
        self._native_actuator_graphs: tuple[wp.Graph, wp.Graph] | None = None
        self._native_actuator_graph_index = 0

    def resolve_env_mask(self, env_mask: wp.array(dtype=wp.bool) | None) -> wp.array(dtype=wp.bool):
        """Resolve an optional environment mask to a full Warp bool mask.

        PhysX's articulation-level mask resolution converts masks to int32
        indices for its index-only tensor API. The collection's mask write
        path consumes full bool masks instead, so normalize here.
        """
        return self._resolve_bool_mask(env_mask, "_all_env_mask", self.num_instances)

    def resolve_joint_mask(self, joint_mask: wp.array(dtype=wp.bool) | None) -> wp.array(dtype=wp.bool):
        """Resolve an optional joint mask to a full Warp bool mask."""
        return self._resolve_bool_mask(joint_mask, "_all_joint_mask", self.num_joints)

    def _resolve_bool_mask(self, mask: wp.array(dtype=wp.bool) | None, cache_attr: str, size: int) -> wp.array(
        dtype=wp.bool
    ):
        if mask is None:
            cached = getattr(self, cache_attr)
            if cached is None:
                cached = wp.ones(size, dtype=wp.bool, device=self.device)
                setattr(self, cache_attr, cached)
            return cached
        if isinstance(mask, wp.array) and mask.dtype == wp.bool:
            return mask
        # Legacy mask resolution accepted any nonzero-selectable mask; keep that.
        mask_torch = wp.to_torch(mask) if isinstance(mask, wp.array) else mask
        return wp.from_torch((mask_torch != 0).contiguous(), dtype=wp.bool)

    def _write_joint_friction_properties(self, actuator: ActuatorBase) -> None:
        articulation = self._articulation
        super()._write_joint_friction_properties(actuator)
        articulation.write_joint_dynamic_friction_coefficient_to_sim_index(
            joint_dynamic_friction_coeff=actuator.dynamic_friction,
            joint_ids=actuator.joint_indices,
        )
        articulation.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff=actuator.viscous_friction,
            joint_ids=actuator.joint_indices,
        )

    def prepare_native_actuators(self, collection: ActuatorCollection, actuator_cfgs: dict) -> set[str]:
        articulation = self._articulation
        articulation._physx_actuator_wrapper = None
        articulation.newton_actuator_adapter = None
        articulation.newton_default_stiffness = None
        articulation.newton_default_damping = None
        articulation.newton_managed_local_joints = None
        articulation._implicit_dof_mask = None
        articulation._has_newton_actuators = False

        use_newton_actuators = getattr(articulation._sim_cfg, "use_newton_actuators", False)
        if use_newton_actuators and not _HAS_NEWTON_ACTUATORS:
            logger.warning(
                "use_newton_actuators is enabled but 'isaaclab_newton.actuators' is not available."
                " Newton-native actuators will be disabled and the simulation will fall back to the"
                " Isaac Lab actuator path. Install the isaaclab_newton extension to enable the fast path."
            )
            return set()
        if not (use_newton_actuators and _HAS_NEWTON_ACTUATORS):
            return set()

        from isaaclab_newton.actuators import NewtonActuatorAdapter, PhysxActuatorWrapper  # noqa: PLC0415

        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415

        self._native_actuator_path_active = True
        articulation._has_newton_actuators = True

        native_group_names = {
            name for name, actuator_cfg in actuator_cfgs.items() if not self._is_implicit_cfg(actuator_cfg)
        }

        self._physx_actuator_wrapper = PhysxActuatorWrapper.create(
            num_envs=self.num_instances,
            num_joints=self.num_joints,
            device=self.device,
        )
        articulation._physx_actuator_wrapper = self._physx_actuator_wrapper

        if native_group_names:
            first_prim = find_first_matching_prim(articulation.cfg.prim_path)
            art_prim_path = str(first_prim.GetPath()) if first_prim is not None else None
            adapter = NewtonActuatorAdapter.from_usd(
                stage=get_current_stage(),
                joint_names=articulation.joint_names,
                num_envs=self.num_instances,
                num_joints=self.num_joints,
                device=self.device,
                articulation_prim_path=art_prim_path,
            )
            wrapper = self._physx_actuator_wrapper
            wrapper.joint_q = articulation._data.joint_pos.warp.reshape(-1)
            wrapper.joint_qd = articulation._data.joint_vel.warp.reshape(-1)
            wrapper.joint_target_q = collection.command.position.warp.reshape(-1)
            wrapper.joint_target_qd = collection.command.velocity.warp.reshape(-1)
            wrapper.joint_target_pos = collection.command.position.warp.reshape(-1)
            wrapper.joint_target_vel = collection.command.velocity.warp.reshape(-1)
            wrapper.joint_act = collection.command.effort.warp.reshape(-1)
            adapter.finalize(wrapper)
            articulation.newton_actuator_adapter = adapter

        return native_group_names

    def finalize_native_actuators(self, collection: ActuatorCollection) -> None:
        if not self._native_actuator_path_active:
            return
        from isaaclab_newton.actuators import build_implicit_dof_mask  # noqa: PLC0415

        articulation = self._articulation
        if articulation.newton_actuator_adapter is not None:
            binding = articulation.newton_actuator_adapter.bind_articulation(
                lab_actuators=dict(collection.items()),
                dof_offset=0,
                num_joints=self.num_joints,
            )
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

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        if not self._native_actuator_path_active:
            return False

        articulation = self._articulation
        if articulation.newton_actuator_adapter is not None:
            adapter = articulation.newton_actuator_adapter
            device = wp.get_device(self.device)
            if device.is_cuda and device.is_capturing and adapter.is_stateful:
                raise RuntimeError(
                    "stateful Newton actuators cannot run inside an outer CUDA graph capture; "
                    "let PhysX capture their alternating state graphs automatically"
                )
            if articulation.data.has_joint_ordering:
                # ``wrapper.joint_q``/``joint_qd`` were bound once (at actuator setup) to
                # ``_data.joint_pos``/``joint_vel``. With identity ordering those bindings alias
                # PhysX-owned memory directly and are always current. With non-identity ordering
                # they alias an owned shadow buffer that is only refreshed when the public getters
                # run -- which otherwise would not happen until the telemetry kernel below reads
                # them, one step too late for the adapter. Force the refresh here so the adapter
                # sees this step's state instead of a stale one-step-old shadow.
                articulation._data._refresh_joint_pos()
                articulation._data._refresh_joint_vel()
            if adapter.is_all_graphable and device.is_cuda:
                if not device.is_capturing:
                    if self._native_actuator_graphs is None:
                        self._capture_native_actuator_graphs(collection)
                    if self._native_actuator_graphs:
                        wp.capture_launch(self._native_actuator_graphs[self._native_actuator_graph_index])
                        adapter._swap_state_buffers()
                        self._native_actuator_graph_index ^= 1
                        return True

        self._run_native_actuator_kernels(collection)
        return True

    def _run_native_actuator_kernels(self, collection: ActuatorCollection) -> None:
        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        wrapper = self._physx_actuator_wrapper
        wrapper.joint_f_2d.assign(collection._joint_effort_target)
        if articulation.newton_actuator_adapter is not None:
            articulation.newton_actuator_adapter.step(wrapper, wrapper, SimulationManager.get_physics_dt())

        wp.launch(
            actuator_kernels.sync_torque_telemetry,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                articulation._data.joint_pos.warp,
                articulation._data.joint_vel.warp,
                collection._joint_pos_target,
                collection._joint_vel_target,
                articulation._data.joint_stiffness.warp,
                articulation._data.joint_damping.warp,
                articulation._data.joint_effort_limits.warp,
                articulation._implicit_dof_mask,
                wrapper.joint_f_2d,
                articulation._data._sim_bind_joint_computed_effort,
                articulation._ALL_JOINT_INDICES,
                False,
            ],
            outputs=[
                collection._computed_torque,
                collection._applied_torque,
            ],
            device=self.device,
        )

    def _capture_native_actuator_graphs(self, collection: ActuatorCollection) -> None:
        adapter = self._articulation.newton_actuator_adapter
        if adapter is None:
            return
        states_a = adapter._states_a
        states_b = adapter._states_b
        graphs = []
        try:
            for _ in range(2):
                with wp.ScopedCapture(device=self.device, force_module_load=True) as capture:
                    self._run_native_actuator_kernels(collection)
                graphs.append(capture.graph)
        except Exception as exc:
            logger.warning("PhysX Newton-actuator CUDA graph capture failed; using eager execution: %s", exc)
            graphs = []
        finally:
            adapter._states_a = states_a
            adapter._states_b = states_b
        self._native_actuator_graphs = tuple(graphs) if graphs else ()
        self._native_actuator_graph_index = 0

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
            # One fused gather replaces the per-target reorder launches. PhysX has no
            # direct-drive joint-act output, so its gated-off output is left unset.
            wp.launch(
                ordering_kernels.reorder_joint_targets_user_to_backend,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    user_effort,
                    user_pos_target,
                    user_vel_target,
                    articulation.data.joint_ordering.backend_to_user,
                    True,
                    articulation._has_implicit_actuators,
                    articulation._has_implicit_actuators,
                    False,
                ],
                outputs=[
                    articulation._joint_effort_target_backend,
                    articulation._joint_pos_target_backend,
                    articulation._joint_vel_target_backend,
                    None,
                ],
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
        if self._native_actuator_path_active and self._articulation.newton_actuator_adapter is not None:
            self._articulation.newton_actuator_adapter.reset(env_ids)

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        adapter = self._articulation.newton_actuator_adapter
        if adapter is None:
            return

        from isaaclab_newton.actuators import kernels as actuator_kernels  # noqa: PLC0415

        env_id_pos = torch.full((self.num_instances,), -1, dtype=torch.int32, device=self.device)
        env_id_pos[env_ids.to(self.device, dtype=torch.long)] = torch.arange(
            env_ids.shape[0],
            dtype=torch.int32,
            device=self.device,
        )
        joint_id_pos = torch.full((self.num_joints,), -1, dtype=torch.int32, device=self.device)
        joint_ids_local = joint_ids.to(self.device, dtype=torch.long)
        joint_id_pos[joint_ids_local] = torch.arange(
            joint_ids.shape[0],
            dtype=torch.int32,
            device=self.device,
        )

        values_wp = wp.from_torch(values.to(self.device, dtype=torch.float32).contiguous(), dtype=wp.float32)
        env_id_pos_wp = wp.from_torch(env_id_pos, dtype=wp.int32)
        joint_id_pos_wp = wp.from_torch(joint_id_pos, dtype=wp.int32)

        for actuator in adapter.actuators:
            ctrl = actuator.controller
            if not hasattr(ctrl, attr):
                continue
            wp.launch(
                actuator_kernels.patch_actuator_param_kernel,
                dim=actuator.indices.shape[0],
                inputs=[
                    actuator.indices,
                    env_id_pos_wp,
                    joint_id_pos_wp,
                    values_wp,
                    0,
                    self.num_joints,
                ],
                outputs=[getattr(ctrl, attr)],
                device=self.device,
            )
