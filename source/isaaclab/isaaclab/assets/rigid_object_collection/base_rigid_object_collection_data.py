# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

import torch


class BaseRigidObjectCollectionData(ABC):
    """Data container for a rigid object collection.

    This class contains the data for a rigid object collection in the simulation. The data includes the state of
    all the bodies in the collection. The data is stored in the simulation world frame unless otherwise specified.
    The data is in the order ``(num_instances, num_objects, data_size)``, where data_size is the size of the data.

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

    def __init__(self, root_view, num_objects: int, device: str):
        """Initializes the rigid object data.

        Args:
            root_view: The root rigid body collection view.
            num_objects: The number of objects in the collection.
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
    def default_body_pose(self) -> torch.Tensor:
        """Default body pose ``[pos, quat]`` in local environment frame.

        The position and quaternion are of the rigid body's actor frame. Shape is (num_instances, num_bodies, 7).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_body_vel(self) -> torch.Tensor:
        """Default body velocity ``[lin_vel, ang_vel]`` in local environment frame. Shape is


        The linear and angular velocities are of the rigid body's center of mass frame. Shape is
        (num_instances, num_bodies, 6).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_body_state(self) -> torch.Tensor:
        """Default body state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame.

        The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities
        are of the center of mass frame. Shape is (num_instances, num_bodies, 13).
        """
        raise NotImplementedError()

    ##
    # Body state properties.
    ##

    @property
    @abstractmethod
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_state_w(self) -> torch.Tensor:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and orientation are of the rigid bodies' actor frame. Meanwhile, the linear and angular
        velocities are of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia. The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_acc_w(self) -> torch.Tensor:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_mass(self) -> torch.Tensor:
        """Mass of all bodies in the simulation world frame. Shape is (num_instances, num_bodies)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_inertia(self) -> torch.Tensor:
        """Inertia of all bodies in the simulation world frame. Shape is (num_instances, num_bodies, 9)."""
        raise NotImplementedError()

    ##
    # Derived Properties.
    ##

    @property
    @abstractmethod
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, num_bodies, 3)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances, num_bodies).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    ##
    # Sliced properties.
    ##

    @property
    @abstractmethod
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (x, y, z, w) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame  relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (x, y, z, w) of the principle axis of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies, 4). This quantity is the orientation of the rigid bodies' actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        raise NotImplementedError()
