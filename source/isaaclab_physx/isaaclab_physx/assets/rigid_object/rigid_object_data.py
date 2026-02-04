# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import warnings
import weakref
from typing import TYPE_CHECKING

import torch

from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.math import (
    combine_frame_transforms,
    normalize,
    quat_apply,
    quat_apply_inverse,
)

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object.rigid_object_view import RigidObjectView

# import logger
logger = logging.getLogger(__name__)


class RigidObjectData(BaseRigidObjectData):
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

    __backend_name__: str = "physx"
    """The name of the backend for the rigid object data."""

    def __init__(self, root_view: RigidObjectView, device: str):
        """Initializes the rigid object data.

        Args:
            root_view: The root rigid body view.
            device: The device used for processing.
        """
        super().__init__(root_view, device)
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: RigidObjectView = weakref.proxy(root_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False

        # Obtain global physics sim view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        gravity = self._physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_view.count, 1)

        self._create_buffers()

    @property
    def is_primed(self) -> bool:
        """Whether the rigid object data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the rigid object data is fully instantiated and ready to use.

        .. note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: The primed state.

        Raises:
            ValueError: If the rigid object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object data is already primed.")
        self._is_primed = value

    def update(self, dt: float) -> None:
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    """
    Defaults.
    """

    @property
    def default_root_pose(self) -> torch.Tensor:
        """Default root pose ``[pos, quat]`` in local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the rigid body's actor frame.
        """
        return self._default_root_pose

    @default_root_pose.setter
    def default_root_pose(self, value: torch.Tensor) -> None:
        """Set the default root pose.

        Args:
            value: The default root pose. Shape is (num_instances, 7).

        Raises:
            ValueError: If the rigid object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object data is already primed.")
        self._default_root_pose = value

    @property
    def default_root_vel(self) -> torch.Tensor:
        """Default root velocity ``[lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the rigid body's center of mass frame.
        """
        return self._default_root_vel

    @default_root_vel.setter
    def default_root_vel(self, value: torch.Tensor) -> None:
        """Set the default root velocity.

        Args:
            value: The default root velocity. Shape is (num_instances, 6).

        Raises:
            ValueError: If the rigid object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object data is already primed.")
        self._default_root_vel = value

    @property
    def default_root_state(self) -> torch.Tensor:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame.

        The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities
        are of the center of mass frame. Shape is (num_instances, 13).
        """
        warnings.warn(
            "Reading the root state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_root_pose and default_root_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.cat([self.default_root_pose, self.default_root_vel], dim=1)

    @default_root_state.setter
    def default_root_state(self, value: torch.Tensor) -> None:
        """Set the default root state.

        Args:
            value: The default root state. Shape is (num_instances, 13).

        Raises:
            ValueError: If the rigid object data is already primed.
        """
        warnings.warn(
            "Setting the root state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_root_pose and default_root_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._is_primed:
            raise ValueError("The rigid object data is already primed.")
        self._default_root_pose = value[:, :7]
        self._default_root_vel = value[:, 7:]

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the actor frame of the root rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_view.get_transforms().clone()
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity
            vel = self.root_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            vel[:, :3] += torch.linalg.cross(
                vel[:, 3:], quat_apply(self.root_link_quat_w, -self.body_com_pos_b[:, 0]), dim=-1
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.data = vel
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the center of mass frame of the root rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = combine_frame_transforms(
                self.root_link_pos_w, self.root_link_quat_w, self.body_com_pos_b[:, 0], self.body_com_quat_b[:, 0]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_view.get_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world. The orientation is provided in (w, x, y, z) format.
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame is the orientation principle axes of inertia.
        """
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            self._root_com_state_w.data = torch.cat((self.root_com_pose_w, self.root_com_vel_w), dim=-1)
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> torch.Tensor:
        """Mass of all bodies in the simulation world frame. Shape is (num_instances, 1, 1)."""
        return self._body_mass.to(self.device)

    @property
    def body_inertia(self) -> torch.Tensor:
        """Inertia of all bodies in the simulation world frame. Shape is (num_instances, 1, 3, 3)."""
        return self._body_inertia.to(self.device)

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 1, 7).

        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        return self.root_link_pose_w.view(-1, 1, 7)

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        return self.root_link_vel_w.view(-1, 1, 6)

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        return self.root_com_pose_w.view(-1, 1, 7)

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        return self.root_com_vel_w.view(-1, 1, 6)

    @property
    def body_state_w(self) -> torch.Tensor:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position and orientation are of the rigid bodies' actor frame. Meanwhile, the linear and angular
        velocities are of the rigid bodies' center of mass frame.
        """
        return self.root_state_w.view(-1, 1, 13)

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        return self.root_link_state_w.view(-1, 1, 13)

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia. The orientation is provided in (w, x, y, z) format.
        """
        return self.root_com_state_w.view(-1, 1, 13)

    @property
    def body_com_acc_w(self) -> torch.Tensor:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.
        Shape is (num_instances, 1, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            self._body_com_acc_w.data = self._root_view.get_accelerations().unsqueeze(1)
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_view.get_coms().to(self.device)
            # set the buffer data and timestamp
            self._body_com_pose_b.data = pose.view(-1, 1, 7)
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    """
    Sliced properties.
    """

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_vel_w[:, 3:6]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_com_vel_w[:, 3:6]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the rigid bodies' actor frame  relative to the world.
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.

        Shape is (num_instances, 1, 4). This quantity is the orientation of the rigid bodies' actor frame.
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, 1, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self.body_com_pose_b[..., 3:7]

    def _create_buffers(self) -> None:
        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()

        # -- Default state
        self._default_root_pose = torch.zeros(self._root_view.count, 7, device=self.device)
        self._default_root_vel = torch.zeros(self._root_view.count, 6, device=self.device)

        # -- Body properties
        self._body_mass = self._root_view.get_masses().to(self.device).clone()
        self._body_inertia = self._root_view.get_inertias().to(self.device).clone()

    """
    Backwards compatibility. (Deprecated properties)
    """

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_link_pose_w` instead."""
        warnings.warn(
            "The `root_pose_w` property will be deprecated in a IsaacLab 4.0. Please use `root_link_pose_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_link_pos_w` instead."""
        warnings.warn(
            "The `root_pos_w` property will be deprecated in a IsaacLab 4.0. Please use `root_link_pos_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_link_quat_w` instead."""
        warnings.warn(
            "The `root_quat_w` property will be deprecated in a IsaacLab 4.0. Please use `root_link_quat_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_com_vel_w` instead."""
        warnings.warn(
            "The `root_vel_w` property will be deprecated in a IsaacLab 4.0. Please use `root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_com_lin_vel_w` instead."""
        warnings.warn(
            "The `root_lin_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `root_com_lin_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_com_ang_vel_w` instead."""
        warnings.warn(
            "The `root_ang_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `root_com_ang_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_com_lin_vel_b` instead."""
        warnings.warn(
            "The `root_lin_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `root_com_lin_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`root_com_ang_vel_b` instead."""
        warnings.warn(
            "The `root_ang_vel_b` property will be deprecated in a IsaacLab 4.0. Please use"
            " `root_com_ang_vel_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_link_pose_w` instead."""
        warnings.warn(
            "The `body_pose_w` property will be deprecated in a IsaacLab 4.0. Please use `body_link_pose_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_link_pos_w` instead."""
        warnings.warn(
            "The `body_pos_w` property will be deprecated in a IsaacLab 4.0. Please use `body_link_pos_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_link_quat_w` instead."""
        warnings.warn(
            "The `body_quat_w` property will be deprecated in a IsaacLab 4.0. Please use `body_link_quat_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_vel_w` instead."""
        warnings.warn(
            "The `body_vel_w` property will be deprecated in a IsaacLab 4.0. Please use `body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_lin_vel_w` instead."""
        warnings.warn(
            "The `body_lin_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_ang_vel_w` instead."""
        warnings.warn(
            "The `body_ang_vel_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_acc_w` instead."""
        warnings.warn(
            "The `body_acc_w` property will be deprecated in a IsaacLab 4.0. Please use `body_com_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_lin_acc_w` instead."""
        warnings.warn(
            "The `body_lin_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_lin_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_ang_acc_w` instead."""
        warnings.warn(
            "The `body_ang_acc_w` property will be deprecated in a IsaacLab 4.0. Please use"
            " `body_com_ang_acc_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_pos_b` instead."""
        warnings.warn(
            "The `com_pos_b` property will be deprecated in a IsaacLab 4.0. Please use `body_com_pos_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_com_quat_b` instead."""
        warnings.warn(
            "The `com_quat_b` property will be deprecated in a IsaacLab 4.0. Please use `body_com_quat_b` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.body_com_quat_b

    """
    Removed - Default values are no longer stored.
    """

    @property
    def default_mass(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_mass` instead and manage the default mass manually."""
        warnings.warn(
            "The `default_mass` property will be deprecated in a IsaacLab 4.0. Please use `body_mass` instead."
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_mass is None:
            self._default_mass = self.body_mass.clone()
        return self._default_mass

    @property
    def default_inertia(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`body_inertia` instead and manage the default inertia manually."""
        warnings.warn(
            "The `default_inertia` property will be deprecated in a IsaacLab 4.0. Please use `body_inertia` instead."
            "The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_inertia is None:
            self._default_inertia = self.body_inertia.clone()
        return self._default_inertia
