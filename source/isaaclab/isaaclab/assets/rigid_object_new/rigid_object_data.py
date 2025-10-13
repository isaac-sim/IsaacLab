# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref
import warp as wp

from isaaclab.assets.core.root_properties.root_data import RootData
from isaaclab.utils.helpers import deprecated


class RigidObjectData:
    """Data container for a rigid object.

    This class contains the data for a rigid object in the simulation. The data includes the state of
    the root rigid body and the state of all the bodies in the object. The data is stored in the simulation
    world frame unless otherwise specified.

    For a rigid body, there are two frames of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings of the simulation, the actor frame and the center of mass frame may be the same.
    This needs to be taken into account when interpreting the data.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.
    """

    def __init__(self, root_newton_view, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_newton_view = weakref.proxy(root_newton_view)

        # Initialize the data containers
        self._root_data = RootData(root_newton_view, device)

    def update(self, dt: float):
        self._root_data.update(dt)

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in the local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the articulation root's actor frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._root_data.default_root_pose

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the articulation root's center of mass frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._root_data.default_root_vel

    ##
    # Root state properties.
    ##

    @property
    def root_mass(self) -> wp.array:
        """Root mass ``wp.float32`` in the world frame. Shape is (num_instances,)."""
        return self._root_data.root_mass

    @property
    def root_inertia(self) -> wp.array:
        """Root inertia ``wp.mat33`` in the world frame. Shape is (num_instances, 9)."""
        return self._root_data.root_inertia

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the actor frame of the root rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        return self._root_data.root_link_pose_w


    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        return self._root_data.root_link_vel_w

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the center of mass frame of the root rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        return self._root_data.root_com_pose_w

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        return self._root_data.root_com_vel_w

    @property
    def root_state_w(self) -> wp.array:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """
        return self._root_data.root_state_w

    @property
    def root_link_state_w(self) -> wp.array:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world. The orientation is provided in (w, x, y, z) format.
        """
        return self._root_data.root_link_state_w

    @property
    def root_com_state_w(self) -> wp.array: 
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame is the orientation principle axes of inertia.
        """
        return self._root_data.root_com_state_w

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return self._root_data.projected_gravity_b

    @property
    def heading_w(self) -> wp.array:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        return self._root_data.heading_w

    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_link_lin_vel_b

    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_link_ang_vel_b

    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_com_lin_vel_b

    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_com_ang_vel_b

    @property
    def root_com_pos_b(self) -> wp.array:
        """Root center of mass position in base frame. Shape is (num_instances, 3).

        This quantity is the position of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_com_pos_b

    @property
    def root_com_quat_b(self) -> wp.array:
        """Root center of mass orientation (w, x, y, z) in base frame. Shape is (num_instances, 4).

        This quantity is the orientation of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_com_quat_b   

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> wp.array:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._root_data.root_link_pos_w

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._root_data.root_link_quat_w

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._root_data.root_link_lin_vel_w

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._root_data.root_link_ang_vel_w

    @property
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._root_data.root_com_pos_w

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self._root_data.root_com_quat_w

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._root_data.root_com_lin_vel_w

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._root_data.root_com_ang_vel_w

    @property
    def root_com_quat_b(self) -> wp.array:
        """Root center of mass orientation (w, x, y, z) in base frame. Shape is (num_instances, 4).

        This quantity is the orientation of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_data.root_com_pose_b[:, 3:]

    ##
    # Properties for backwards compatibility.
    ##

    @property
    @deprecated("root_link_pose_w")
    def root_pose_w(self) -> wp.array:
        """Same as :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    @deprecated("root_link_pos_w")
    def root_pos_w(self) -> wp.array:
        """Same as :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    @deprecated("root_link_quat_w")
    def root_quat_w(self) -> wp.array:
        """Same as :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    @deprecated("root_com_vel_w")
    def root_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    @deprecated("root_com_lin_vel_w")
    def root_lin_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    @deprecated("root_com_ang_vel_w")
    def root_ang_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    @deprecated("root_com_lin_vel_b")
    def root_lin_vel_b(self) -> wp.array:
        """Same as :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    @deprecated("root_com_ang_vel_b")
    def root_ang_vel_b(self) -> wp.array:
        """Same as :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    @deprecated("body_com_pos_b")
    def com_pos_b(self) -> wp.array:
        """Same as :attr:`body_com_pos_b`."""
        return self.root_com_pos_b

    @property
    @deprecated("body_com_quat_b")
    def com_quat_b(self) -> wp.array:
        """Same as :attr:`body_com_quat_b`."""
        return self.root_com_quat_b
