# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


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

    def __init__(self, root_physx_view: physx.SoftBodyView, device: str):
        """Initializes the deformable object data.

        Args:
            root_physx_view: The root deformable body view of the object.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root deformable body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.SoftBodyView = weakref.proxy(root_physx_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Initialize the lazy buffers.
        # -- node state in simulation world frame
        self._nodal_pos_w = TimestampedBuffer()
        self._nodal_vel_w = TimestampedBuffer()
        self._nodal_state_w = TimestampedBuffer()
        # -- mesh element-wise rotations
        self._sim_element_quat_w = TimestampedBuffer()
        self._collision_element_quat_w = TimestampedBuffer()
        # -- mesh element-wise deformation gradients
        self._sim_element_deform_gradient_w = TimestampedBuffer()
        self._collision_element_deform_gradient_w = TimestampedBuffer()
        # -- mesh element-wise stresses
        self._sim_element_stress_w = TimestampedBuffer()
        self._collision_element_stress_w = TimestampedBuffer()

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

    default_nodal_state_w: torch.Tensor = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, max_sim_vertices_per_body, 6).
    """

    ##
    # Kinematic commands
    ##

    nodal_kinematic_target: torch.Tensor = None
    """Simulation mesh kinematic targets for the deformable bodies.
    Shape is (num_instances, max_sim_vertices_per_body, 4).

    The kinematic targets are used to drive the simulation mesh vertices to the target positions.
    The targets are stored as (x, y, z, is_not_kinematic) where "is_not_kinematic" is a binary
    flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
    and 1 for non-kinematic vertices.
    """

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self):
        """Nodal positions in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body, 3)."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            self._nodal_pos_w.data = self._root_physx_view.get_sim_nodal_positions()
            self._nodal_pos_w.timestamp = self._sim_timestamp
        return self._nodal_pos_w.data

    @property
    def nodal_vel_w(self):
        """Nodal velocities in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body, 3)."""
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            self._nodal_vel_w.data = self._root_physx_view.get_sim_nodal_velocities()
            self._nodal_vel_w.timestamp = self._sim_timestamp
        return self._nodal_vel_w.data

    @property
    def nodal_state_w(self):
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
        Shape is (num_instances, max_sim_vertices_per_body, 6).
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            nodal_positions = self.nodal_pos_w
            nodal_velocities = self.nodal_vel_w
            # set the buffer data and timestamp
            self._nodal_state_w.data = torch.cat((nodal_positions, nodal_velocities), dim=-1)
            self._nodal_state_w.timestamp = self._sim_timestamp
        return self._nodal_state_w.data

    @property
    def sim_element_quat_w(self):
        """Simulation mesh element-wise rotations as quaternions for the deformable bodies in simulation world frame.
        Shape is (num_instances, max_sim_elements_per_body, 4).

        The rotations are stored as quaternions in the order (w, x, y, z).
        """
        if self._sim_element_quat_w.timestamp < self._sim_timestamp:
            # convert from xyzw to wxyz
            quats = self._root_physx_view.get_sim_element_rotations().view(self._root_physx_view.count, -1, 4)
            quats = math_utils.convert_quat(quats, to="wxyz")
            # set the buffer data and timestamp
            self._sim_element_quat_w.data = quats
            self._sim_element_quat_w.timestamp = self._sim_timestamp
        return self._sim_element_quat_w.data

    @property
    def collision_element_quat_w(self):
        """Collision mesh element-wise rotations as quaternions for the deformable bodies in simulation world frame.
        Shape is (num_instances, max_collision_elements_per_body, 4).

        The rotations are stored as quaternions in the order (w, x, y, z).
        """
        if self._collision_element_quat_w.timestamp < self._sim_timestamp:
            # convert from xyzw to wxyz
            quats = self._root_physx_view.get_element_rotations().view(self._root_physx_view.count, -1, 4)
            quats = math_utils.convert_quat(quats, to="wxyz")
            # set the buffer data and timestamp
            self._collision_element_quat_w.data = quats
            self._collision_element_quat_w.timestamp = self._sim_timestamp
        return self._collision_element_quat_w.data

    @property
    def sim_element_deform_gradient_w(self):
        """Simulation mesh element-wise second-order deformation gradient tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_sim_elements_per_body, 3, 3).
        """
        if self._sim_element_deform_gradient_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._sim_element_deform_gradient_w.data = (
                self._root_physx_view.get_sim_element_deformation_gradients().view(
                    self._root_physx_view.count, -1, 3, 3
                )
            )
            self._sim_element_deform_gradient_w.timestamp = self._sim_timestamp
        return self._sim_element_deform_gradient_w.data

    @property
    def collision_element_deform_gradient_w(self):
        """Collision mesh element-wise second-order deformation gradient tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_collision_elements_per_body, 3, 3).
        """
        if self._collision_element_deform_gradient_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._collision_element_deform_gradient_w.data = (
                self._root_physx_view.get_element_deformation_gradients().view(self._root_physx_view.count, -1, 3, 3)
            )
            self._collision_element_deform_gradient_w.timestamp = self._sim_timestamp
        return self._collision_element_deform_gradient_w.data

    @property
    def sim_element_stress_w(self):
        """Simulation mesh element-wise second-order Cauchy stress tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_sim_elements_per_body, 3, 3).
        """
        if self._sim_element_stress_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._sim_element_stress_w.data = self._root_physx_view.get_sim_element_stresses().view(
                self._root_physx_view.count, -1, 3, 3
            )
            self._sim_element_stress_w.timestamp = self._sim_timestamp
        return self._sim_element_stress_w.data

    @property
    def collision_element_stress_w(self):
        """Collision mesh element-wise second-order Cauchy stress tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_collision_elements_per_body, 3, 3).
        """
        if self._collision_element_stress_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._collision_element_stress_w.data = self._root_physx_view.get_element_stresses().view(
                self._root_physx_view.count, -1, 3, 3
            )
            self._collision_element_stress_w.timestamp = self._sim_timestamp
        return self._collision_element_stress_w.data

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal positions.
        """
        return self.nodal_pos_w.mean(dim=1)

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal velocities.
        """
        return self.nodal_vel_w.mean(dim=1)
