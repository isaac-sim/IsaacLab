# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod

import warp as wp


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
    def default_body_pose(self) -> wp.array:
        """Default body pose ``[pos, quat]`` in local environment frame.

        The position and quaternion are of the rigid body's actor frame.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_body_vel(self) -> wp.array:
        """Default body velocity ``[lin_vel, ang_vel]`` in local environment frame.

        The linear and angular velocities are of the rigid body's center of mass frame.
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def default_body_state(self) -> wp.array:
        """Deprecated, same as :attr:`default_body_pose` and :attr:`default_body_vel`."""
        raise NotImplementedError()

    ##
    # Body state properties.
    ##

    @property
    @abstractmethod
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_vel_w(self) -> wp.array:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_com_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_link_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_com_pose_w` and :attr:`body_com_vel_w`."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_mass(self) -> wp.array:
        """Mass of all bodies in the simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_inertia(self) -> wp.array:
        """Inertia of all bodies in the simulation world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32. In torch this resolves to
        (num_instances, num_bodies, 9).
        """
        raise NotImplementedError()

    ##
    # Derived Properties.
    ##

    @property
    @abstractmethod
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def heading_w(self) -> wp.array:
        """Yaw heading of the base frame (in radians).

        Shape is (num_instances, num_bodies), dtype = wp.float32.
        In torch this resolves to (num_instances, num_bodies).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        raise NotImplementedError()

    ##
    # Sliced properties.
    ##

    @property
    @abstractmethod
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' actor frame relative to the world.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies' center of mass in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' principal axes of inertia.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their
        respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).

        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        raise NotImplementedError()

    """
    Shorthands for commonly used properties.
    """

    @property
    def body_pose_w(self) -> wp.array:
        """Shorthand for :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> wp.array:
        """Shorthand for :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> wp.array:
        """Shorthand for :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> wp.array:
        """Shorthand for :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> wp.array:
        """Shorthand for :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> wp.array:
        """Shorthand for :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    def _create_buffers(self):
        # -- Default mass and inertia (Lazy allocation of default values)
        self._default_mass = None
        self._default_inertia = None

    """
    Deprecated properties for backwards compatibility.
    """

    @property
    def default_object_pose(self) -> wp.array:
        """Deprecated property. Please use :attr:`default_body_pose` instead."""
        warnings.warn(
            "The `default_object_pose` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_body_pose` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_body_pose

    @property
    def default_object_vel(self) -> wp.array:
        """Deprecated property. Please use :attr:`default_body_vel` instead."""
        warnings.warn(
            "The `default_object_vel` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_body_vel` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_body_vel

    @property
    def default_object_state(self) -> wp.array:
        """Deprecated property. Please use :attr:`default_body_state` instead."""
        warnings.warn(
            "The `default_object_state` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_body_state` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_body_state

    @property
    def object_link_pose_w(self):
        """Deprecated property. Please use :attr:`body_link_pose_w` instead."""
        warnings.warn(
            "The `object_link_pose_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_pose_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_pose_w

    @property
    def object_link_vel_w(self):
        """Deprecated property. Please use :attr:`body_link_vel_w` instead."""
        warnings.warn(
            "The `object_link_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_vel_w

    @property
    def object_com_pose_w(self):
        """Deprecated property. Please use :attr:`body_com_pose_w` instead."""
        warnings.warn(
            "The `object_com_pose_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_pose_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_pose_w

    @property
    def object_com_vel_w(self):
        """Deprecated property. Please use :attr:`body_com_vel_w` instead."""
        warnings.warn(
            "The `object_com_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_vel_w

    @property
    def object_state_w(self):
        """Deprecated property. Please use :attr:`body_state_w` instead."""
        warnings.warn(
            "The `object_state_w` property will be deprecated in a IsaacLab 4.0. Please use `body_state_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_state_w

    @property
    def object_link_state_w(self):
        """Deprecated property. Please use :attr:`body_link_state_w` instead."""
        warnings.warn(
            "The `object_link_state_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_state_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_state_w

    @property
    def object_com_state_w(self):
        """Deprecated property. Please use :attr:`body_com_state_w` instead."""
        warnings.warn(
            "The `object_com_state_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_state_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_state_w

    @property
    def object_com_acc_w(self):
        """Deprecated property. Please use :attr:`body_com_acc_w` instead."""
        warnings.warn(
            "The `object_com_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_acc_w

    @property
    def object_com_pose_b(self):
        """Deprecated property. Please use :attr:`body_com_pose_b` instead."""
        warnings.warn(
            "The `object_com_pose_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_pose_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_pose_b

    @property
    def object_link_pos_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_pos_w` instead."""
        warnings.warn(
            "The `object_link_pos_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_pos_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_pos_w

    @property
    def object_link_quat_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_quat_w` instead."""
        warnings.warn(
            "The `object_link_quat_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_quat_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_quat_w

    @property
    def object_link_lin_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_lin_vel_w` instead."""
        warnings.warn(
            "The `object_link_lin_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_lin_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_lin_vel_w

    @property
    def object_link_ang_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_ang_vel_w` instead."""
        warnings.warn(
            "The `object_link_ang_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_ang_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_ang_vel_w

    @property
    def object_com_pos_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_pos_w` instead."""
        warnings.warn(
            "The `object_com_pos_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_pos_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_pos_w

    @property
    def object_com_quat_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_quat_w` instead."""
        warnings.warn(
            "The `object_com_quat_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_quat_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_quat_w

    @property
    def object_com_lin_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_lin_vel_w` instead."""
        warnings.warn(
            "The `object_com_lin_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_vel_w

    @property
    def object_com_ang_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_ang_vel_w` instead."""
        warnings.warn(
            "The `object_com_ang_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_vel_w

    @property
    def object_com_lin_acc_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_lin_acc_w` instead."""
        warnings.warn(
            "The `object_com_lin_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_acc_w

    @property
    def object_com_ang_acc_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_ang_acc_w` instead."""
        warnings.warn(
            "The `object_com_ang_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_acc_w

    @property
    def object_com_pos_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_pos_b` instead."""
        warnings.warn(
            "The `object_com_pos_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_pos_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_pos_b

    @property
    def object_com_quat_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_quat_b` instead."""
        warnings.warn(
            "The `object_com_quat_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_quat_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_quat_b

    @property
    def object_link_lin_vel_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_lin_vel_b` instead."""
        warnings.warn(
            "The `object_link_lin_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_lin_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_lin_vel_b

    @property
    def object_link_ang_vel_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_ang_vel_b` instead."""
        warnings.warn(
            "The `object_link_ang_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_link_ang_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_ang_vel_b

    @property
    def object_com_lin_vel_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_lin_vel_b` instead."""
        warnings.warn(
            "The `object_com_lin_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_vel_b

    @property
    def object_com_ang_vel_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_ang_vel_b` instead."""
        warnings.warn(
            "The `object_com_ang_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_vel_b

    @property
    def object_pose_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_pose_w` instead."""
        warnings.warn(
            "The `object_pose_w` property will be deprecated in a IsaacLab 4.0. Please use `body_link_pose_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_pose_w

    @property
    def object_pos_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_pos_w` instead."""
        warnings.warn(
            "The `object_pos_w` property will be deprecated in a IsaacLab 4.0. Please use `body_link_pos_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_pos_w

    @property
    def object_quat_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_link_quat_w` instead."""
        warnings.warn(
            "The `object_quat_w` property will be deprecated in a IsaacLab 4.0. Please use `body_link_quat_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_quat_w

    @property
    def object_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_vel_w` instead."""
        warnings.warn(
            "The `object_vel_w` property will be deprecated in a IsaacLab 4.0. Please use `body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_vel_w

    @property
    def object_lin_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_lin_vel_w` instead."""
        warnings.warn(
            "The `object_lin_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_vel_w

    @property
    def object_ang_vel_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_ang_vel_w` instead."""
        warnings.warn(
            "The `object_ang_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_vel_w

    @property
    def object_lin_vel_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_lin_vel_b` instead."""
        warnings.warn(
            "The `object_lin_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_vel_b

    @property
    def object_ang_vel_b(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_ang_vel_b` instead."""
        warnings.warn(
            "The `object_ang_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_vel_b

    @property
    def object_acc_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_acc_w` instead."""
        warnings.warn(
            "The `object_acc_w` property will be deprecated in a IsaacLab 4.0. Please use `body_com_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_acc_w

    @property
    def object_lin_acc_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_lin_acc_w` instead."""
        warnings.warn(
            "The `object_lin_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_acc_w

    @property
    def object_ang_acc_w(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_com_ang_acc_w` instead."""
        warnings.warn(
            "The `object_ang_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_acc_w

    """
    Removed - Default values are no longer stored.
    """

    @property
    def default_mass(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_mass` instead and manage the default mass manually."""
        warnings.warn(
            "The `default_mass` property will be deprecated in a IsaacLab 4.0. Please use `body_mass` instead. "
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_mass is None:
            self._default_mass = wp.clone(self.body_mass, self.device)
        return self._default_mass

    @property
    def default_inertia(self) -> wp.array:
        """Deprecated property. Please use :attr:`body_inertia` instead and manage the default inertia manually."""
        warnings.warn(
            "The `default_inertia` property will be deprecated in a IsaacLab 4.0. Please use `body_inertia` instead. "
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_inertia is None:
            self._default_inertia = wp.clone(self.body_inertia, self.device)
        return self._default_inertia
