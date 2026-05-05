# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import warp as wp

import omni.physics.tensors.api as physx

from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray

from .kernels import compute_mean_vec3f_over_vertices, compute_nodal_state_w, vec6f


class DeformableObjectData:
    """Data container for a deformable object.

    This class contains the data for a deformable object in the simulation. The data includes the nodal states of
    the root deformable body in the object. The data is stored in the simulation world frame unless otherwise specified.

    A deformable object in PhysX uses two tetrahedral meshes to represent the object:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    The APIs exposed provides the data for both the simulation and collision meshes. These are specified
    by the `sim` and `collision` prefixes in the property names.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.
    """

    def __init__(self, root_view: physx.DeformableBodyView, device: str):
        """Initializes the deformable object data.

        Args:
            root_view: The root deformable body view of the object.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root deformable body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: physx.DeformableBodyView = weakref.proxy(root_view)

        # Store dimensions
        self._num_instances = root_view.count
        self._max_sim_vertices = root_view.max_simulation_nodes_per_body
        self._max_sim_elements = root_view.max_simulation_elements_per_body
        self._max_collision_elements = root_view.max_collision_elements_per_body

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Initialize the lazy buffers.
        # -- node state in simulation world frame
        self._nodal_pos_w = TimestampedBuffer((self._num_instances, self._max_sim_vertices), device, wp.vec3f)
        self._nodal_vel_w = TimestampedBuffer((self._num_instances, self._max_sim_vertices), device, wp.vec3f)
        self._nodal_state_w = TimestampedBuffer((self._num_instances, self._max_sim_vertices), device, vec6f)
        # -- derived: root pos/vel
        self._root_pos_w = TimestampedBuffer((self._num_instances,), device, wp.vec3f)
        self._root_vel_w = TimestampedBuffer((self._num_instances,), device, wp.vec3f)

        # -- Pinned ProxyArray cache (one per read property, lazily created on first access)
        self._nodal_pos_w_ta: ProxyArray | None = None
        self._nodal_vel_w_ta: ProxyArray | None = None
        self._nodal_state_w_ta: ProxyArray | None = None
        self._root_pos_w_ta: ProxyArray | None = None
        self._root_vel_w_ta: ProxyArray | None = None

    def update(self, dt: float):
        """Updates the data for the deformable object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Defaults.
    ##

    default_nodal_state_w: ProxyArray = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, max_sim_vertices_per_body) with dtype vec6f.
    """

    ##
    # Kinematic commands
    ##

    nodal_kinematic_target: ProxyArray = None
    """Simulation mesh kinematic targets for the deformable bodies.
    Shape is (num_instances, max_sim_vertices_per_body) with dtype vec4f.

    The kinematic targets are used to drive the simulation mesh vertices to the target positions.
    The targets are stored as (x, y, z, is_not_kinematic) where "is_not_kinematic" is a binary
    flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
    and 1 for non-kinematic vertices.
    """

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self) -> ProxyArray:
        """Nodal positions in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body) vec3f."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            # get_simulation_nodal_positions() returns (N, V, 3) float32 — view as (N, V) vec3f
            src = (
                self._root_view.get_simulation_nodal_positions()
                .view(wp.vec3f)
                .reshape((self._num_instances, self._max_sim_vertices))
            )
            wp.copy(self._nodal_pos_w.data, src)
            self._nodal_pos_w.timestamp = self._sim_timestamp
            # Rebind ProxyArray since .data was replaced with a new wp.array
            if self._nodal_pos_w_ta is not None:
                self._nodal_pos_w_ta = ProxyArray(self._nodal_pos_w.data)
        if self._nodal_pos_w_ta is None:
            self._nodal_pos_w_ta = ProxyArray(self._nodal_pos_w.data)
        return self._nodal_pos_w_ta

    @property
    def nodal_vel_w(self) -> ProxyArray:
        """Nodal velocities in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body) vec3f."""
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            src = (
                self._root_view.get_simulation_nodal_velocities()
                .view(wp.vec3f)
                .reshape((self._num_instances, self._max_sim_vertices))
            )
            wp.copy(self._nodal_vel_w.data, src)
            self._nodal_vel_w.timestamp = self._sim_timestamp
            # Rebind ProxyArray since .data was replaced with a new wp.array
            if self._nodal_vel_w_ta is not None:
                self._nodal_vel_w_ta = ProxyArray(self._nodal_vel_w.data)
        if self._nodal_vel_w_ta is None:
            self._nodal_vel_w_ta = ProxyArray(self._nodal_vel_w.data)
        return self._nodal_vel_w_ta

    @property
    def nodal_state_w(self) -> ProxyArray:
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
        Shape is (num_instances, max_sim_vertices_per_body) vec6f.
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._max_sim_vertices),
                inputs=[self.nodal_pos_w, self.nodal_vel_w],
                outputs=[self._nodal_state_w.data],
                device=self.device,
            )
            self._nodal_state_w.timestamp = self._sim_timestamp
        if self._nodal_state_w_ta is None:
            self._nodal_state_w_ta = ProxyArray(self._nodal_state_w.data)
        return self._nodal_state_w_ta

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> ProxyArray:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation
        world frame. Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal positions.
        """
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_pos_w, self._max_sim_vertices],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        if self._root_pos_w_ta is None:
            self._root_pos_w_ta = ProxyArray(self._root_pos_w.data)
        return self._root_pos_w_ta

    @property
    def root_vel_w(self) -> ProxyArray:
        """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal velocities.
        """
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_vel_w, self._max_sim_vertices],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        if self._root_vel_w_ta is None:
            self._root_vel_w_ta = ProxyArray(self._root_vel_w.data)
        return self._root_vel_w_ta
