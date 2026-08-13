# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared host runtime for Newton-native actuator execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch
import warp as wp

from .adapter import NewtonActuatorAdapter, build_implicit_dof_mask, read_newton_actuator_gain
from .physx_wrapper import PhysxActuatorWrapper

if TYPE_CHECKING:
    from isaaclab.actuators import ActuatorCollection


class _HostActuatorRuntime:
    """Own Newton-native actuator lifecycle shared by host-PhysX backends."""

    def __init__(self, articulation: Any, *, logger: logging.Logger):
        self._articulation = articulation
        self._logger = logger
        self.wrapper: PhysxActuatorWrapper | None = None
        self.adapter: NewtonActuatorAdapter | None = None
        self.binding: NewtonActuatorAdapter.ArticulationBinding | None = None
        self.native_actuator_graphs: tuple[wp.Graph, wp.Graph] | None = None
        self._native_actuator_graph_index = 0

    def prepare(
        self,
        collection: ActuatorCollection,
        *,
        stage: Any,
        articulation_prim_path: str | None,
    ) -> None:
        """Create and bind the adapter and its pointer-stable host wrapper."""
        articulation = self._articulation
        self.wrapper = PhysxActuatorWrapper.create(
            num_envs=articulation.num_instances,
            num_joints=articulation.num_joints,
            device=articulation.device,
        )
        self.adapter = NewtonActuatorAdapter.from_usd(
            stage=stage,
            joint_names=articulation.joint_names,
            num_envs=articulation.num_instances,
            num_joints=articulation.num_joints,
            device=articulation.device,
            articulation_prim_path=articulation_prim_path,
        )
        self.wrapper.joint_q = articulation._data.joint_pos.warp.reshape(-1)
        self.wrapper.joint_qd = articulation._data.joint_vel.warp.reshape(-1)
        self.wrapper.joint_target_q = collection.command.position.warp.reshape(-1)
        self.wrapper.joint_target_qd = collection.command.velocity.warp.reshape(-1)
        self.wrapper.joint_target_pos = self.wrapper.joint_target_q
        self.wrapper.joint_target_vel = self.wrapper.joint_target_qd
        self.wrapper.joint_act = collection.command.effort.warp.reshape(-1)
        self.adapter.finalize(self.wrapper)

    def finalize(self, collection: ActuatorCollection) -> None:
        """Bind telemetry and native gain defaults after collection construction."""
        articulation = self._articulation
        if self.adapter is None:
            articulation._implicit_dof_mask, articulation._implicit_dof_mask_owner = build_implicit_dof_mask(
                dict(collection.items()), articulation.num_joints, articulation.device
            )
            articulation._data._sim_bind_joint_computed_effort = wp.zeros(
                (articulation.num_instances, articulation.num_joints),
                dtype=wp.float32,
                device=articulation.device,
            )
            return

        self.binding = self.adapter.bind_articulation(
            lab_actuators=dict(collection.items()),
            dof_offset=0,
            num_joints=articulation.num_joints,
        )
        articulation.newton_default_stiffness = self.binding.stiffness
        articulation.newton_default_damping = self.binding.damping
        articulation.newton_managed_local_joints = self.binding.joint_indices
        articulation._implicit_dof_mask = self.binding.implicit_dof_mask
        articulation._implicit_dof_mask_owner = self.binding.implicit_dof_mask_owner
        articulation._data._sim_bind_joint_computed_effort = self.binding.computed_effort_view

    def compute(self, collection: ActuatorCollection, dt: float) -> None:
        """Compute native efforts and synchronize telemetry for one physics step."""
        if self.adapter is not None:
            device = wp.get_device(self._articulation.device)
            if device.is_cuda and device.is_capturing and self.adapter.is_stateful:
                raise RuntimeError(
                    "stateful Newton actuators cannot run inside an outer CUDA graph capture; "
                    "let the host actuator runtime manage their alternating graphs"
                )
            if self.adapter.is_all_graphable and device.is_cuda and not device.is_capturing:
                if self.native_actuator_graphs is None:
                    self._capture_native_actuator_graphs(collection, dt)
                if self.native_actuator_graphs:
                    wp.capture_launch(self.native_actuator_graphs[self._native_actuator_graph_index])
                    self.adapter._swap_state_buffers()
                    self._native_actuator_graph_index ^= 1
                    return
        self._run_native_actuator_kernels(collection, dt)

    def reset(self, env_ids: Sequence[int] | slice) -> None:
        """Reset selected native actuator state."""
        if self.adapter is not None:
            self.adapter.reset(env_ids)

    def get_gain(self, attr: Literal["kp", "kd"], joint_ids: torch.Tensor | slice) -> torch.Tensor | None:
        """Project a native controller gain into public joint order."""
        if self.adapter is None:
            return None
        articulation = self._articulation
        gains, covered = read_newton_actuator_gain(
            self.adapter.actuators,
            attr,
            articulation.num_instances,
            articulation.num_joints,
            0,
            articulation.num_joints,
            articulation.device,
        )
        if not bool(torch.all(covered[joint_ids])):
            return None
        return gains[:, joint_ids]

    def write_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Patch selected native controller gains in-place."""
        if self.adapter is None:
            return
        from . import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        env_id_pos = torch.full((articulation.num_instances,), -1, dtype=torch.int32, device=articulation.device)
        env_id_pos[env_ids.to(articulation.device, dtype=torch.long)] = torch.arange(
            env_ids.shape[0], dtype=torch.int32, device=articulation.device
        )
        joint_id_pos = torch.full((articulation.num_joints,), -1, dtype=torch.int32, device=articulation.device)
        joint_ids_local = joint_ids.to(articulation.device, dtype=torch.long)
        joint_id_pos[joint_ids_local] = torch.arange(joint_ids.shape[0], dtype=torch.int32, device=articulation.device)
        values_wp = wp.from_torch(values.to(articulation.device, dtype=torch.float32).contiguous(), dtype=wp.float32)
        env_id_pos_wp = wp.from_torch(env_id_pos, dtype=wp.int32)
        joint_id_pos_wp = wp.from_torch(joint_id_pos, dtype=wp.int32)
        for actuator in self.adapter.actuators:
            controller = actuator.controller
            if not hasattr(controller, attr):
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
                    articulation.num_joints,
                ],
                outputs=[getattr(controller, attr)],
                device=articulation.device,
            )

    def _run_native_actuator_kernels(self, collection: ActuatorCollection, dt: float) -> None:
        from . import kernels as actuator_kernels  # noqa: PLC0415

        articulation = self._articulation
        if self.wrapper is None:
            raise RuntimeError("Newton-native actuator wrapper was not initialized")
        self.wrapper.joint_f_2d.assign(collection._joint_effort_target)
        if self.adapter is not None:
            self.adapter.step(self.wrapper, self.wrapper, dt)
        wp.launch(
            actuator_kernels.sync_torque_telemetry,
            dim=(articulation.num_instances, articulation.num_joints),
            inputs=[
                articulation._data.joint_pos.warp,
                articulation._data.joint_vel.warp,
                collection._joint_pos_target,
                collection._joint_vel_target,
                articulation._data.joint_stiffness.warp,
                articulation._data.joint_damping.warp,
                articulation._data.joint_effort_limits.warp,
                articulation._implicit_dof_mask,
                self.wrapper.joint_f_2d,
                articulation._data._sim_bind_joint_computed_effort,
                articulation._ALL_JOINT_INDICES,
                False,
            ],
            outputs=[collection._computed_effort, collection._applied_effort],
            device=articulation.device,
        )

    def _capture_native_actuator_graphs(self, collection: ActuatorCollection, dt: float) -> None:
        if self.adapter is None:
            return
        states_a = self.adapter._states_a
        states_b = self.adapter._states_b
        graphs = []
        try:
            for _ in range(2):
                with wp.ScopedCapture(device=self._articulation.device, force_module_load=True) as capture:
                    self._run_native_actuator_kernels(collection, dt)
                graphs.append(capture.graph)
        except Exception as exc:
            self._logger.warning("Host Newton-actuator CUDA graph capture failed; using eager execution: %s", exc)
            graphs = []
        finally:
            self.adapter._states_a = states_a
            self.adapter._states_b = states_b
        self.native_actuator_graphs = tuple(graphs) if graphs else ()
        self._native_actuator_graph_index = 0
