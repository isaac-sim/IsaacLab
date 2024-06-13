# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.buffers import TimestampedBuffer


class RigidObjectData:
    """Data container for a rigid object."""

    def __init__(self, root_physx_view: physx.RigidBodyView, device):
        self.device = device
        self._time_stamp = 0.0
        self._root_physx_view: physx.RigidBodyView = root_physx_view

        self.gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self._root_physx_view.count, 1)
        self.forward_vec_b = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)
        self._previous_body_vel_w = torch.zeros((self._root_physx_view.count, 1, 6), device=self.device)

        # Initialize the lazy buffers.
        self._root_state_w: TimestampedBuffer = TimestampedBuffer()
        self._body_acc_w: TimestampedBuffer = TimestampedBuffer()

    def update(self, dt: float):
        self._time_stamp += dt
        # Trigger an update of the body acceleration buffer at a higher frequency since we do finite differencing.
        self.body_acc_w

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13)."""

    default_mass: torch.Tensor = None
    """ Default mass provided by simulation. Shape is (num_instances, num_bodies)."""

    ##
    # Properties.
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13)."""
        if self._root_state_w.update_timestamp < self._time_stamp:
            pose = self._root_physx_view.get_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            velocity = self._root_physx_view.get_velocities()
            self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._root_state_w.update_timestamp = self._time_stamp
        return self._root_state_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape is (num_instances, 1, 13)."""
        return self.root_state_w.view(-1, 1, 13)

    @property
    def body_acc_w(self):
        """Acceleration of all bodies. Shape is (num_instances, 1, 6)."""
        if self._body_acc_w.update_timestamp < self._time_stamp:
            self._body_acc_w.data = (self.body_vel_w - self._previous_body_vel_w) / (
                self._time_stamp - self._body_acc_w.update_timestamp
            )
            self._previous_body_vel_w[:] = self.body_vel_w
            self._body_acc_w.update_timestamp = self._time_stamp
        return self._body_acc_w.data

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.gravity_vec_w)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_quat_w, self.forward_vec_b)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3)."""
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4)."""
        return self.root_state_w[:, 3:7]

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6)."""
        return self.root_state_w[:, 7:13]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3)."""
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3)."""
        return self.root_state_w[:, 10:13]

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3)."""
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_state_w[..., :3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4)."""
        return self.body_state_w[..., 3:7]

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6)."""
        return self.body_state_w[..., 7:13]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_state_w[..., 7:10]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_state_w[..., 10:13]
