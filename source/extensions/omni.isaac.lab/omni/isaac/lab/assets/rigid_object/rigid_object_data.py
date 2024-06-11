# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.utils.math as math_utils


@dataclass
class LazyBuffer:
    data: torch.Tensor = torch.Tensor()
    update_timestamp: float = -1.0


class RigidObjectData:
    """Data container for a rigid object."""

    def __init__(self, root_physx_view: physx.RigidBodyView, device):
        self.device = device
        self.time_stamp = 0.0
        self._root_physx_view: physx.RigidBodyView = root_physx_view

    def update(self, dt: float):
        self.time_stamp += dt

    ##
    # Properties.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Default states.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13)."""

    ##
    # Frame states.
    ##

    _root_state_w: LazyBuffer = LazyBuffer()

    @property
    def root_state_w(self):
        if self._root_state_w.update_timestamp < self.time_stamp:
            pose = self._root_physx_view.get_transforms()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            velocity = self._root_physx_view.get_velocities()
            self._root_state_w.data = torch.cat((pose, velocity), dim=-1)
            self._root_state_w.update_timestamp = self.time_stamp
        return self._root_state_w.data

    """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13)."""

    _root_vel_b: LazyBuffer = LazyBuffer()

    @property
    def root_vel_b(self):
        if self._root_vel_b.update_timestamp < self.time_stamp:
            root_lin_vel_b = math_utils.quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)
            root_ang_vel_b = math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)
            self._root_vel_b.data = torch.cat((root_lin_vel_b, root_ang_vel_b), dim=-1)
            self._root_vel_b.update_timestamp = self.time_stamp
        return self._root_vel_b.data

    _projected_gravity_b: LazyBuffer = LazyBuffer()

    @property
    def projected_gravity_b(self):
        if self._projected_gravity_b.update_timestamp < self.time_stamp:
            gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self._root_physx_view.count, 1)
            self._projected_gravity_b.data = math_utils.quat_rotate_inverse(self.root_quat_w, gravity_vec_w)
            self._projected_gravity_b.update_timestamp = self.time_stamp
        return self._projected_gravity_b.data

    """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""

    _heading_w: LazyBuffer = LazyBuffer()

    @property
    def heading_w(self):
        if self._heading_w.update_timestamp < self.time_stamp:
            forward_vec_b = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)
            forward_w = math_utils.quat_apply(self.root_quat_w, forward_vec_b)
            self._heading_w.data = torch.atan2(forward_w[:, 1], forward_w[:, 0])
            self._heading_w.update_timestamp = self.time_stamp
        return self._heading_w.data

    """Yaw heading of the base frame (in radians). Shape is (num_instances,).

    Note:
        This quantity is computed by assuming that the forward-direction of the base
        frame is along x-direction, i.e. :math:`(1, 0, 0)`.
    """

    _body_state_w: LazyBuffer = LazyBuffer()

    @property
    def body_state_w(self):
        if self._body_state_w.update_timestamp < self.time_stamp:
            self._body_state_w.data = self.root_state_w.view(-1, 1, 13)
            self._body_state_w.update_timestamp = self.time_stamp
        return self._body_state_w.data

    """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
    Shape is (num_instances, 1, 13)."""

    _body_acc_w: LazyBuffer = LazyBuffer()

    @property
    def body_acc_w(self):
        if self._body_acc_w.update_timestamp < self.time_stamp:
            self._body_state_w.data = torch.zeros(self._root_physx_view.count, 1, 6, device=self.device)
            self._body_state_w.update_timestamp = self.time_stamp
        return self._body_state_w.data

    """Acceleration of all bodies. Shape is (num_instances, num_bodies, 6).

    Note:
        This quantity is computed based on the rigid body state from the last step.
    """

    ##
    # Default rigid body properties
    ##

    default_mass: torch.Tensor = None
    """ Default mass provided by simulation. Shape is (num_instances, num_bodies)."""

    """
    Properties
    """

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
        return self.root_vel_b[:, 0:3]

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3)."""
        return self.root_vel_b[:, 3:6]

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

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_acc_w[..., 0:3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3)."""
        return self.body_acc_w[..., 3:6]
