# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import JointType
from newton.selection import ArticulationView

from pxr import UsdGeom

from isaaclab.assets.cable_object.base_cable_object import BaseCableObject
from isaaclab.cloner import queue_replication
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.utils.queries import has_deformable_curve_api, resolve_matching_prims_from_source
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager as SimulationManager

from .cable_object_data import CableObjectData
from .kernels import (
    set_segment_pose_to_sim_index,
    set_segment_pose_to_sim_mask,
    set_segment_velocity_to_sim_index,
    set_segment_velocity_to_sim_mask,
)

if TYPE_CHECKING:
    from isaaclab.assets.cable_object.cable_object_cfg import CableObjectCfg


class CableObject(BaseCableObject):
    """A Newton cable that requires the VBD solver."""

    cfg: CableObjectCfg
    """Configuration instance for the cable object."""

    __backend_name__: str = "newton"
    """The name of the backend for the cable object."""

    def __init__(self, cfg: CableObjectCfg) -> None:
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        queue_replication(cfg)

    @property
    def data(self) -> CableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_view.count

    @property
    def num_segments(self) -> int:
        """Number of rigid segments per cable."""
        return self.root_view.link_count + 1

    @property
    def root_view(self) -> ArticulationView:
        """Articulation view for the cable."""
        return self._root_view

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the cable object's internal buffers.

        Args:
            env_ids: Environment indices. Defaults to all instances.
        """
        del env_ids

    def write_data_to_sim(self) -> None:
        """Write buffered commands to the simulation."""

    def update(self, dt: float) -> None:
        """Update the cable object data.

        Args:
            dt: The time step [s].
        """
        self.data.update(dt)

    def write_segment_pose_to_sim_index(
        self,
        *,
        segment_pose: torch.Tensor | wp.array(dtype=wp.transformf) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Set segment poses for selected environments.

        Args:
            segment_pose: Actor-frame poses in simulation world frame. The Torch shape is
                (len(env_ids), num_segments, 7), with position ``(x, y, z)`` [m] followed by quaternion
                ``(x, y, z, w)``. The Warp shape is (len(env_ids), num_segments), dtype ``wp.transformf``.
            env_ids: Environment indices. If None, all instances are used.
        """
        if isinstance(segment_pose, ProxyArray):
            segment_pose = segment_pose.warp
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(segment_pose, (env_ids.shape[0], self.num_segments), wp.transformf, "segment_pose")
        self._write_segment_state_to_sim(segment_pose, env_ids, set_segment_pose_to_sim_index, "body_q", use_mask=False)

    def write_segment_pose_to_sim_mask(
        self,
        *,
        segment_pose: torch.Tensor | wp.array(dtype=wp.transformf) | ProxyArray,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set segment poses using an environment mask.

        Args:
            segment_pose: Actor-frame poses in simulation world frame. The Torch shape is
                (num_instances, num_segments, 7), with position (x, y, z) [m] followed by quaternion
                (x, y, z, w). The Warp shape is (num_instances, num_segments), dtype wp.transformf.
            env_mask: Environment mask. If None, all instances are used.
        """
        if isinstance(segment_pose, ProxyArray):
            segment_pose = segment_pose.warp
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        self.assert_shape_and_dtype_mask(
            segment_pose, (env_mask,), wp.transformf, "segment_pose", trailing_dims=(self.num_segments,)
        )
        self._write_segment_state_to_sim(segment_pose, env_mask, set_segment_pose_to_sim_mask, "body_q", use_mask=True)

    def write_segment_velocity_to_sim_index(
        self,
        *,
        segment_velocity: torch.Tensor | wp.array(dtype=wp.spatial_vectorf) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Set segment center-of-mass velocities for selected environments.

        Args:
            segment_velocity: Segment center-of-mass velocities in simulation world frame. The Torch shape is
                (len(env_ids), num_segments, 6), with linear ``(x, y, z)`` [m/s] followed by angular
                ``(x, y, z)`` [rad/s] velocity. The Warp shape is (len(env_ids), num_segments), dtype
                ``wp.spatial_vectorf``.
            env_ids: Environment indices. If None, all instances are used.
        """
        if isinstance(segment_velocity, ProxyArray):
            segment_velocity = segment_velocity.warp
        env_ids = self._resolve_env_ids(env_ids)
        self.assert_shape_and_dtype(
            segment_velocity, (env_ids.shape[0], self.num_segments), wp.spatial_vectorf, "segment_velocity"
        )
        self._write_segment_state_to_sim(
            segment_velocity, env_ids, set_segment_velocity_to_sim_index, "body_qd", use_mask=False
        )

    def write_segment_velocity_to_sim_mask(
        self,
        *,
        segment_velocity: torch.Tensor | wp.array(dtype=wp.spatial_vectorf) | ProxyArray,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set segment center-of-mass velocities using an environment mask.

        Args:
            segment_velocity: Segment center-of-mass velocities in simulation world frame. The Torch shape is
                (num_instances, num_segments, 6), with linear (x, y, z) [m/s] followed by angular
                (x, y, z) [rad/s] velocity. The Warp shape is (num_instances, num_segments), dtype
                wp.spatial_vectorf.
            env_mask: Environment mask. If None, all instances are used.
        """
        if isinstance(segment_velocity, ProxyArray):
            segment_velocity = segment_velocity.warp
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        self.assert_shape_and_dtype_mask(
            segment_velocity, (env_mask,), wp.spatial_vectorf, "segment_velocity", trailing_dims=(self.num_segments,)
        )
        self._write_segment_state_to_sim(
            segment_velocity, env_mask, set_segment_velocity_to_sim_mask, "body_qd", use_mask=True
        )

    def _initialize_impl(self) -> None:
        def is_cable_curve(prim) -> bool:
            return prim.IsA(UsdGeom.BasisCurves) and has_deformable_curve_api(prim)

        resolve_kwargs = {"predicate": is_cable_curve, "expected_num_matches": 1}
        curve_prim, curve_path_expr = resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)[0]
        num_segments = int(UsdGeom.BasisCurves(curve_prim).GetCurveVertexCountsAttr().Get()[0]) - 1

        model = SimulationManager.get_model()
        articulation_path_expr = f"{curve_path_expr}_articulation"
        self._root_view = ArticulationView(
            model,
            articulation_path_expr.replace(".*", "*"),
            verbose=False,
        )
        topology_error = "CableObject requires one standalone, unwelded cable articulation per simulation world."
        expected_joint_count = num_segments - 1
        joint_types = self.root_view.get_attribute("joint_type", model).numpy()
        valid_topology = (
            self.root_view.count_per_world == 1
            and self.root_view.joint_count == expected_joint_count
            and self.root_view.link_count == expected_joint_count
            and bool((joint_types == int(JointType.CABLE)).all())
        )
        if not valid_topology:
            raise RuntimeError(topology_error)

        self._ALL_INDICES = wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)
        self._ALL_ENV_MASK = wp.ones((self.num_instances,), dtype=wp.bool, device=self.device)

        self._data = CableObjectData(self.root_view, self.device)
        self._physics_ready_handle = SimulationManager.register_callback(
            self._rebind,
            PhysicsEvent.PHYSICS_READY,
            name=f"cable_object_rebind_{self.cfg.prim_path}",
        )

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None) -> wp.array(
        dtype=wp.int32
    ):
        """Resolve environment indices to a Warp array."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(device=self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        if isinstance(env_ids, Sequence):
            return wp.array(list(env_ids), dtype=wp.int32, device=self.device)
        return env_ids

    def _write_segment_state_to_sim(
        self,
        value: torch.Tensor | wp.array,
        selector: wp.array(dtype=wp.int32) | wp.array(dtype=wp.bool),
        kernel: wp.Kernel,
        state_attribute: str,
        *,
        use_mask: bool,
    ) -> None:
        """Write a validated segment state into the active Newton states."""
        if isinstance(value, torch.Tensor):
            value = value.contiguous()

        for state in self._iter_states():
            wp.launch(
                kernel,
                dim=(selector.shape[0], self.num_segments),
                inputs=[value, selector, self.data._sim_bind_root_body_ids, self.data._sim_bind_link_body_ids],
                outputs=[getattr(state, state_attribute)],
                device=self.device,
            )
        if use_mask:
            SimulationManager.invalidate_body_state(env_mask=selector)
        else:
            SimulationManager.invalidate_body_state(selector)
        self.update(0.0)

    def _iter_states(self):
        """Yield active Newton states."""
        state_0 = SimulationManager.get_state_0()
        state_1 = SimulationManager.get_state_1()
        yield state_0
        if state_1 is not None and state_1 is not state_0:
            yield state_1

    def _rebind(self, _: object) -> None:
        """Rebind simulation arrays after a Newton model rebuild."""
        self._data._create_simulation_bindings()

    def _clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        super()._clear_callbacks()
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
