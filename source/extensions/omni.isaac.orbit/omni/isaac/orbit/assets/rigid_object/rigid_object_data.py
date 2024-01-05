# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class RigidObjectData:
    """Data container for a rigid object."""

    ##
    # Properties.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Default states.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (count, 13)."""

    ##
    # Frame states.
    ##

    root_state_w: torch.Tensor = None
    """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (count, 13)."""

    root_vel_b: torch.Tensor = None
    """Root velocity `[lin_vel, ang_vel]` in base frame. Shape is (count, 6)."""

    projected_gravity_b: torch.Tensor = None
    """Projection of the gravity direction on base frame. Shape is (count, 3)."""

    heading_w: torch.Tensor = None
    """Yaw heading of the base frame (in radians). Shape is (count,).

    Note:
        This quantity is computed by assuming that the forward-direction of the base
        frame is along x-direction, i.e. :math:`(1, 0, 0)`.
    """

    body_state_w: torch.Tensor = None
    """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
    Shape is (count, num_bodies, 13)."""

    body_acc_w: torch.Tensor = None
    """Acceleration of all bodies. Shape is (count, num_bodies, 6).

    Note:
        This quantity is computed based on the rigid body state from the last step.
    """

    """
    Properties
    """

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (count, 3)."""
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (count, 4)."""
        return self.root_state_w[:, 3:7]

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (count, 6)."""
        return self.root_state_w[:, 7:13]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (count, 3)."""
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (count, 3)."""
        return self.root_state_w[:, 10:13]

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (count, 3)."""
        return self.root_vel_b[:, 0:3]

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (count, 3)."""
        return self.root_vel_b[:, 3:6]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        return self.body_state_w[..., :3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (count, num_bodies, 4)."""
        return self.body_state_w[..., 3:7]

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (count, num_bodies, 6)."""
        return self.body_state_w[..., 7:13]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        return self.body_state_w[..., 7:10]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        return self.body_state_w[..., 10:13]

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        return self.body_acc_w[..., 0:3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        return self.body_acc_w[..., 3:6]
