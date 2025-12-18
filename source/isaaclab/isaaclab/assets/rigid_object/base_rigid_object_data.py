# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

import warp as wp


class BaseRigidObjectData(ABC):
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

    def __init__(self, root_view, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device

    @abstractmethod
    def update(self, dt: float) -> None:
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        raise NotImplementedError()

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    @property
    @abstractmethod
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the rigid body's center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_root_state(self) -> wp.array:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).

        The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities are
        of the center of mass frame.
        """
        raise NotImplementedError()

    ##
    # Root state properties.
    ##

    @property
    @abstractmethod
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the actor frame of the root rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the center of mass frame of the root rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_state_w(self) -> wp.array:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_state_w(self) -> wp.array:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world. The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_state_w(self) -> wp.array:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame is the orientation principle axes of inertia.
        """
        raise NotImplementedError()

    ##
    # Body state properties.
    ##

    @property
    @abstractmethod
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 1, 7).

        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_vel_w(self) -> wp.array:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_state_w(self) -> wp.array:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position and orientation are of the rigid bodies' actor frame. Meanwhile, the linear and angular
        velocities are of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_state_w(self) -> wp.array:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_state_w(self) -> wp.array:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia. The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.
        Shape is (num_instances, 1, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_mass(self) -> wp.array:
        """Mass of all bodies in the simulation world frame. Shape is (num_instances, 1, 1)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_inertia(self) -> wp.array:
        """Inertia of all bodies in the simulation world frame. Shape is (num_instances, 1, 3, 3)."""
        raise NotImplementedError()

    ##
    # Derived Properties.
    ##

    @property
    @abstractmethod
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def heading_w(self) -> wp.array:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    ##
    # Sliced properties.
    ##

    @property
    @abstractmethod
    def root_link_pos_w(self) -> wp.array:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (x, y, z, w) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (x, y, z, w) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the rigid bodies' actor frame  relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principle axis of inertia of all bodies in simulation world frame.

        Shape is (num_instances, 1, 4). This quantity is the orientation of the rigid bodies' actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, 1, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        raise NotImplementedError()

    ##
    # Properties for backwards compatibility.
    ##

    @property
    @abstractmethod
    def root_pose_w(self) -> wp.array:
        """Same as :attr:`root_link_pose_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_pos_w(self) -> wp.array:
        """Same as :attr:`root_link_pos_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_quat_w(self) -> wp.array:
        """Same as :attr:`root_link_quat_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_lin_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_lin_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_ang_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_ang_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_lin_vel_b(self) -> wp.array:
        """Same as :attr:`root_com_lin_vel_b`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_ang_vel_b(self) -> wp.array:
        """Same as :attr:`root_com_ang_vel_b`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_pose_w(self) -> wp.array:
        """Same as :attr:`body_link_pose_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_pos_w(self) -> wp.array:
        """Same as :attr:`body_link_pos_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_quat_w(self) -> wp.array:
        """Same as :attr:`body_link_quat_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_vel_w(self) -> wp.array:
        """Same as :attr:`body_com_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_lin_vel_w(self) -> wp.array:
        """Same as :attr:`body_com_lin_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_ang_vel_w(self) -> wp.array:
        """Same as :attr:`body_com_ang_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_acc_w(self) -> wp.array:
        """Same as :attr:`body_com_acc_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_lin_acc_w(self) -> wp.array:
        """Same as :attr:`body_com_lin_acc_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_ang_acc_w(self) -> wp.array:
        """Same as :attr:`body_com_ang_acc_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def com_pos_b(self) -> wp.array:
        """Same as :attr:`body_com_pos_b`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def com_quat_b(self) -> wp.array:
        """Same as :attr:`body_com_quat_b`."""

        raise NotImplementedError()
