# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

from isaaclab.utils.warp import ProxyArray


class BaseDeformableObjectData(ABC):
    """Abstract data container for a deformable object.

    This class defines the interface for deformable object data in the simulation.
    The data includes the nodal states of the root deformable body in the object.
    The data is stored in the simulation world frame unless otherwise specified.

    The data is lazily updated, meaning that the data is only updated when it is accessed.
    This is useful when the data is expensive to compute or retrieve. The data is updated
    when the timestamp of the buffer is older than the current simulation timestamp.
    """

    def __init__(self, device: str):
        """Initialize the deformable object data.

        Args:
            device: The device used for processing.
        """
        self.device = device
        # Set initial time stamp
        self._sim_timestamp = 0.0

    def update(self, dt: float):
        """Update the data for the deformable object.

        Args:
            dt: The time step for the update [s]. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Defaults.
    ##

    default_nodal_state_w: ProxyArray | None = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.

    Shape is (num_instances, max_sim_vertices_per_body), dtype ``vec6f``.
    Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
    :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.
    """

    ##
    # Kinematic commands.
    ##

    nodal_kinematic_target: ProxyArray | None = None
    """Simulation mesh kinematic targets for the deformable bodies.

    Shape is (num_instances, max_sim_vertices_per_body), dtype ``wp.vec4f``.
    Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
    :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.

    The kinematic targets are used to drive the simulation mesh vertices to the target positions.
    The targets are stored as (x, y, z, is_not_kinematic) where "is_not_kinematic" is a binary
    flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
    and 1 for non-kinematic vertices.
    """

    ##
    # Properties.
    ##

    @property
    @abstractmethod
    def nodal_pos_w(self) -> ProxyArray:
        """Nodal positions in simulation world frame [m].

        Shape is (num_instances, max_sim_vertices_per_body), dtype ``wp.vec3f``.
        Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
        :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def nodal_vel_w(self) -> ProxyArray:
        """Nodal velocities in simulation world frame [m/s].

        Shape is (num_instances, max_sim_vertices_per_body), dtype ``wp.vec3f``.
        Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
        :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def nodal_state_w(self) -> ProxyArray:
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame [m, m/s].

        Shape is (num_instances, max_sim_vertices_per_body), dtype ``vec6f``.
        Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
        :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.
        """
        raise NotImplementedError()

    ##
    # Derived properties.
    ##

    @property
    @abstractmethod
    def root_pos_w(self) -> ProxyArray:
        """Root position from nodal positions of the simulation mesh for the deformable bodies
        in simulation world frame [m]. Shape is (num_instances,) vec3f.

        This quantity is computed as the mean of the nodal positions.
        Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
        :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_vel_w(self) -> ProxyArray:
        """Root velocity from vertex velocities for the deformable bodies in simulation
        world frame [m/s]. Shape is (num_instances,) vec3f.

        This quantity is computed as the mean of the nodal velocities.
        Use :attr:`ProxyArray.warp` for the underlying :class:`warp.array` or
        :attr:`ProxyArray.torch` for a cached zero-copy :class:`torch.Tensor` view.
        """
        raise NotImplementedError()
