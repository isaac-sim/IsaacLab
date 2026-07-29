# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy data container for OVPhysX deformable objects."""

from __future__ import annotations

import weakref

import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx.tensor_types import TensorType

from .kernels import compute_mean_vec3f_over_vertices, compute_nodal_state_w, vec6f
from .views import OvPhysxDeformableBodyView


class DeformableObjectData(BaseDeformableObjectData):
    """Data container for an OVPhysX-backed deformable object.

    Simulation state is read lazily into stable Warp allocations. Each public
    :class:`~isaaclab.utils.warp.ProxyArray` is created once and remains valid
    while OVPhysX refreshes its underlying allocation in place.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the deformable object data."""

    def __init__(
        self,
        root_view: OvPhysxDeformableBodyView,
        device: str,
        *,
        position_tensor_type: TensorType,
        velocity_tensor_type: TensorType,
    ) -> None:
        """Initialize the deformable object data.

        Args:
            root_view: OVPhysX deformable body view used for simulation-state reads.
            device: Device used for simulation state and derived computations.
            position_tensor_type: Tensor type used for simulation-node positions.
            velocity_tensor_type: Tensor type used for simulation-node velocities.
        """
        super().__init__(device)
        self._root_view: OvPhysxDeformableBodyView = weakref.proxy(root_view)
        self._position_tensor_type = position_tensor_type
        self._velocity_tensor_type = velocity_tensor_type

        self._num_instances = root_view.count
        self._max_sim_vertices = root_view.max_simulation_nodes_per_body
        self._max_sim_elements = root_view.max_simulation_elements_per_body
        self._max_collision_elements = root_view.max_collision_elements_per_body

        self._nodal_pos_w = TimestampedBufferWarp((self._num_instances, self._max_sim_vertices), device, wp.vec3f)
        self._nodal_vel_w = TimestampedBufferWarp((self._num_instances, self._max_sim_vertices), device, wp.vec3f)
        self._nodal_state_w = TimestampedBufferWarp((self._num_instances, self._max_sim_vertices), device, vec6f)
        self._root_pos_w = TimestampedBufferWarp((self._num_instances,), device, wp.vec3f)
        self._root_vel_w = TimestampedBufferWarp((self._num_instances,), device, wp.vec3f)

        self._nodal_pos_w_proxy: ProxyArray | None = None
        self._nodal_vel_w_proxy: ProxyArray | None = None
        self._nodal_state_w_proxy: ProxyArray | None = None
        self._root_pos_w_proxy: ProxyArray | None = None
        self._root_vel_w_proxy: ProxyArray | None = None

        self.default_nodal_state_w: ProxyArray | None = None
        self.nodal_kinematic_target: ProxyArray | None = None

    @property
    def nodal_pos_w(self) -> ProxyArray:
        """Nodal positions in simulation world frame [m].

        Shape is ``(num_instances, max_sim_vertices_per_body)``, dtype
        ``wp.vec3f``.
        """
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            self._root_view.read_into(
                self._position_tensor_type,
                self._nodal_pos_w.data.view(wp.float32).reshape((self._num_instances, self._max_sim_vertices, 3)),
            )
            self._nodal_pos_w.timestamp = self._sim_timestamp
        if self._nodal_pos_w_proxy is None:
            self._nodal_pos_w_proxy = ProxyArray(self._nodal_pos_w.data)
        return self._nodal_pos_w_proxy

    @property
    def nodal_vel_w(self) -> ProxyArray:
        """Nodal velocities in simulation world frame [m/s].

        Shape is ``(num_instances, max_sim_vertices_per_body)``, dtype
        ``wp.vec3f``.
        """
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            self._root_view.read_into(
                self._velocity_tensor_type,
                self._nodal_vel_w.data.view(wp.float32).reshape((self._num_instances, self._max_sim_vertices, 3)),
            )
            self._nodal_vel_w.timestamp = self._sim_timestamp
        if self._nodal_vel_w_proxy is None:
            self._nodal_vel_w_proxy = ProxyArray(self._nodal_vel_w.data)
        return self._nodal_vel_w_proxy

    @property
    def nodal_state_w(self) -> ProxyArray:
        """Nodal position-velocity states in simulation world frame [m, m/s].

        Shape is ``(num_instances, max_sim_vertices_per_body)``, dtype
        ``vec6f``.
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._max_sim_vertices),
                inputs=[self.nodal_pos_w.warp, self.nodal_vel_w.warp],
                outputs=[self._nodal_state_w.data],
                device=self.device,
            )
            self._nodal_state_w.timestamp = self._sim_timestamp
        if self._nodal_state_w_proxy is None:
            self._nodal_state_w_proxy = ProxyArray(self._nodal_state_w.data)
        return self._nodal_state_w_proxy

    @property
    def root_pos_w(self) -> ProxyArray:
        """Mean simulation-node position in simulation world frame [m].

        Shape is ``(num_instances,)``, dtype ``wp.vec3f``.
        """
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=self._num_instances,
                inputs=[self.nodal_pos_w.warp, self._max_sim_vertices],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        if self._root_pos_w_proxy is None:
            self._root_pos_w_proxy = ProxyArray(self._root_pos_w.data)
        return self._root_pos_w_proxy

    @property
    def root_vel_w(self) -> ProxyArray:
        """Mean simulation-node velocity in simulation world frame [m/s].

        Shape is ``(num_instances,)``, dtype ``wp.vec3f``.
        """
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=self._num_instances,
                inputs=[self.nodal_vel_w.warp, self._max_sim_vertices],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        if self._root_vel_w_proxy is None:
            self._root_vel_w_proxy = ProxyArray(self._root_vel_w.data)
        return self._root_vel_w_proxy
