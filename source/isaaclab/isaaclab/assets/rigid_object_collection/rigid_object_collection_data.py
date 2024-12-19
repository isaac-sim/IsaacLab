# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


class RigidObjectCollectionData:
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

    def __init__(self, root_physx_view: physx.RigidBodyView, num_objects: int, device: str):
        """Initializes the data.

        Args:
            root_physx_view: The root rigid body view.
            num_objects: The number of objects in the collection.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        self.num_objects = num_objects
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.RigidBodyView = weakref.proxy(root_physx_view)
        self.num_instances = self._root_physx_view.count // self.num_objects

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Obtain global physics sim view
        physics_sim_view = physx.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self.num_instances, self.num_objects, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(
            self.num_instances, self.num_objects, 1
        )

        # Initialize the lazy buffers.
        self._object_state_w = TimestampedBuffer()
        self._object_link_state_w = TimestampedBuffer()
        self._object_com_state_w = TimestampedBuffer()
        self._object_acc_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object collection.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Names.
    ##

    object_names: list[str] = None
    """Object names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    default_object_state: torch.Tensor = None
    """Default object state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, num_objects, 13).

    The position and quaternion are of each object's rigid body's actor frame. Meanwhile, the linear and angular velocities are
    of the center of mass frame.
    """

    default_mass: torch.Tensor = None
    """Default object mass read from the simulation. Shape is (num_instances, num_objects, 1)."""

    default_inertia: torch.Tensor = None
    """Default object inertia tensor read from the simulation. Shape is (num_instances, num_objects, 9).

    The inertia is the inertia tensor relative to the center of mass frame. The values are stored in
    the order :math:`[I_{xx}, I_{xy}, I_{xz}, I_{yx}, I_{yy}, I_{yz}, I_{zx}, I_{zy}, I_{zz}]`.
    """

    ##
    # Properties.
    ##

    @property
    def object_state_w(self):
        """Object state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, num_objects, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """

        if self._object_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._reshape_view_to_data(self._root_physx_view.get_transforms().clone())
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            velocity = self._reshape_view_to_data(self._root_physx_view.get_velocities())
            # set the buffer data and timestamp
            self._object_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._object_state_w.timestamp = self._sim_timestamp
        return self._object_state_w.data

    @property
    def object_link_state_w(self):
        """Object center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, num_objects, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world.
        """
        if self._object_link_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._reshape_view_to_data(self._root_physx_view.get_transforms().clone())
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            velocity = self._reshape_view_to_data(self._root_physx_view.get_velocities())

            # adjust linear velocity to link from center of mass
            velocity[..., :3] += torch.linalg.cross(
                velocity[..., 3:], math_utils.quat_rotate(pose[..., 3:7], -self.com_pos_b[..., :]), dim=-1
            )

            # set the buffer data and timestamp
            self._object_link_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._object_link_state_w.timestamp = self._sim_timestamp
        return self._object_link_state_w.data

    @property
    def object_com_state_w(self):
        """Object state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, num_objects, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame is the orientation principle axes of inertia.
        """

        if self._object_com_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._reshape_view_to_data(self._root_physx_view.get_transforms().clone())
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            velocity = self._reshape_view_to_data(self._root_physx_view.get_velocities())

            # adjust pose to center of mass
            pos, quat = math_utils.combine_frame_transforms(
                pose[..., :3], pose[..., 3:7], self.com_pos_b[..., :], self.com_quat_b[..., :]
            )

            # set the buffer data and timestamp
            self._object_com_state_w.data = torch.cat((pos, quat, velocity), dim=-1)
            self._object_com_state_w.timestamp = self._sim_timestamp
        return self._object_com_state_w.data

    @property
    def object_acc_w(self):
        """Acceleration of all objects. Shape is (num_instances, num_objects, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame.
        """
        if self._object_acc_w.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            self._object_acc_w.data = self._reshape_view_to_data(self._root_physx_view.get_accelerations().clone())
            self._object_acc_w.timestamp = self._sim_timestamp
        return self._object_acc_w.data

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, num_objects, 3)."""
        return math_utils.quat_rotate_inverse(self.object_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances, num_objects,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.object_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[..., 1], forward_w[..., 0])

    ##
    # Derived properties.
    ##

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Object position in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the position of the actor frame of the rigid bodies.
        """
        return self.object_state_w[..., :3]

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Object orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, num_objects, 4).

        This quantity is the orientation of the actor frame of the rigid bodies.
        """
        return self.object_state_w[..., 3:7]

    @property
    def object_vel_w(self) -> torch.Tensor:
        """Object velocity in simulation world frame. Shape is (num_instances, num_objects, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return self.object_state_w[..., 7:13]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        """Object linear velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.object_state_w[..., 7:10]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        """Object angular velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.object_state_w[..., 10:13]

    @property
    def object_lin_vel_b(self) -> torch.Tensor:
        """Object linear velocity in base frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.object_quat_w, self.object_lin_vel_w)

    @property
    def object_ang_vel_b(self) -> torch.Tensor:
        """Object angular velocity in base world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.object_quat_w, self.object_ang_vel_w)

    @property
    def object_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self.object_acc_w[..., 0:3]

    @property
    def object_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self.object_acc_w[..., 3:6]

    @property
    def object_link_pos_w(self) -> torch.Tensor:
        """Object link position in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the position of the actor frame of the rigid bodies.
        """
        if self._object_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._reshape_view_to_data(self._root_physx_view.get_transforms().clone())
            return pose[..., :3]
        return self.object_link_state_w[..., :3]

    @property
    def object_link_quat_w(self) -> torch.Tensor:
        """Object link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, num_objects, 4).

        This quantity is the orientation of the actor frame of the rigid bodies.
        """
        if self._object_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._reshape_view_to_data(self._root_physx_view.get_transforms().clone())
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            return pose[..., 3:7]
        return self.object_link_state_w[..., 3:7]

    @property
    def object_link_vel_w(self) -> torch.Tensor:
        """Object link velocity in simulation world frame. Shape is (num_instances, num_objects, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' actor frame.
        """
        return self.object_link_state_w[..., 7:13]

    @property
    def object_link_lin_vel_w(self) -> torch.Tensor:
        """Object link linear velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the rigid bodies' actor frame.
        """
        return self.object_link_state_w[..., 7:10]

    @property
    def object_link_ang_vel_w(self) -> torch.Tensor:
        """Object link angular velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the rigid bodies' actor frame.
        """
        return self.object_link_state_w[..., 10:13]

    @property
    def object_link_lin_vel_b(self) -> torch.Tensor:
        """Object link linear velocity in base frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.object_link_quat_w, self.object_link_lin_vel_w)

    @property
    def object_link_ang_vel_b(self) -> torch.Tensor:
        """Object link angular velocity in base world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.object_link_quat_w, self.object_link_ang_vel_w)

    @property
    def object_com_pos_w(self) -> torch.Tensor:
        """Object center of mass position in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the position of the center of mass frame of the rigid bodies.
        """
        return self.object_com_state_w[..., :3]

    @property
    def object_com_quat_w(self) -> torch.Tensor:
        """Object center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, num_objects, 4).

        This quantity is the orientation of the center of mass frame of the rigid bodies.
        """
        return self.object_com_state_w[..., 3:7]

    @property
    def object_com_vel_w(self) -> torch.Tensor:
        """Object center of mass velocity in simulation world frame. Shape is (num_instances, num_objects, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        if self._object_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocity = self._reshape_view_to_data(self._root_physx_view.get_velocities())
            return velocity
        return self.object_com_state_w[..., 7:13]

    @property
    def object_com_lin_vel_w(self) -> torch.Tensor:
        """Object center of mass linear velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        if self._object_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocity = self._reshape_view_to_data(self._root_physx_view.get_velocities())
            return velocity[..., 0:3]
        return self.object_com_state_w[..., 7:10]

    @property
    def object_com_ang_vel_w(self) -> torch.Tensor:
        """Object center of mass angular velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        if self._object_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocity = self._reshape_view_to_data(self._root_physx_view.get_velocities())
            return velocity[..., 3:6]
        return self.object_com_state_w[..., 10:13]

    @property
    def object_com_lin_vel_b(self) -> torch.Tensor:
        """Object center of mass linear velocity in base frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the center of mass frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.object_link_quat_w, self.object_com_lin_vel_w)

    @property
    def object_com_ang_vel_b(self) -> torch.Tensor:
        """Object center of mass angular velocity in base world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the center of mass frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.object_link_quat_w, self.object_com_ang_vel_w)

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Center of mass of all of the bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the center of mass location relative to its body frame.
        """
        pos = self._root_physx_view.get_coms().to(self.device)[..., :3]
        return self._reshape_view_to_data(pos)

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Orientation (w,x,y,z) of the prinicple axies of inertia of all of the bodies in simulation world frame. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body frame.
        """
        quat = self._root_physx_view.get_coms().to(self.device)[..., 3:7].view(self.num_instances, self.num_objects, 4)
        quat_wxyz = math_utils.convert_quat(quat, to="wxyz")
        return self._reshape_view_to_data(quat_wxyz)

    ##
    # Helpers.
    ##

    def _reshape_view_to_data(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes and arranges the data from the physics view to (num_instances, num_objects, data_size).

        Args:
            data: The data from the physics view. Shape is (num_instances*num_objects, data_size).

        Returns:
            The reshaped data. Shape is (num_objects, num_instances, data_size).
        """
        return torch.einsum("ijk -> jik", data.reshape(self.num_objects, self.num_instances, -1))
