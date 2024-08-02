# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.buffers import TimestampedBuffer


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

    def __init__(self, root_physx_view: physx.RigidBodyView, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.RigidBodyView = weakref.proxy(root_physx_view)

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
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize the lazy buffers.
        self._root_state_w = TimestampedBuffer()
        self._body_acc_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).

    The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities are
    of the center of mass frame.
    """

    default_mass: torch.Tensor = None
    """Default mass read from the simulation. Shape is (num_instances, num_bodies)."""

    ##
    # Properties.
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            velocity = self._root_physx_view.get_velocities()
            # set the buffer data and timestamp
            self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp
        return self._root_state_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape is (num_instances, 1, 13).

        The position and orientation are of the rigid bodies' actor frame. Meanwhile, the linear and angular
        velocities are of the rigid bodies' center of mass frame.
        """
        return self.root_state_w.view(-1, 1, 13)

    @property
    def body_acc_w(self):
        """Acceleration of all bodies. Shape is (num_instances, 1, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame.
        """
        if self._body_acc_w.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            self._body_acc_w.data = self._root_physx_view.get_accelerations().unsqueeze(1)
            self._body_acc_w.timestamp = self._sim_timestamp
        return self._body_acc_w.data

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body.
        """
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self.root_state_w[:, 3:7]

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame.
        """
        return self.root_state_w[:, 7:13]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame.
        """
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame.
        """
        return self.root_state_w[:, 10:13]

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        return self.body_state_w[..., :3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame.
        """
        return self.body_state_w[..., 3:7]

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return self.body_state_w[..., 7:13]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.body_state_w[..., 7:10]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.body_state_w[..., 10:13]

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_acc_w[..., 0:3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_acc_w[..., 3:6]
