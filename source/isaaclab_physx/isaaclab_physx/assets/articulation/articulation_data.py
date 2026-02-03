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

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.math import combine_frame_transforms, normalize, quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets.articulation.articulation_view import ArticulationView

# import logger
logger = logging.getLogger(__name__)


class ArticulationData(BaseArticulationData):
    """Data container for an articulation.

    This class contains the data for an articulation in the simulation. The data includes the state of
    the root rigid body, the state of all the bodies in the articulation, and the joint state. The data is
    stored in the simulation world frame unless otherwise specified.

    An articulation is comprised of multiple rigid bodies or links. For a rigid body, there are two frames
    of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings, the two frames may not coincide with each other. In the robotics sense, the actor frame
    can be interpreted as the link frame.
    """

    __backend_name__: str = "physx"
    """The name of the backend for the articulation data."""

    def __init__(self, root_view: ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_view: The root articulation view.
            device: The device used for processing.
        """
        super().__init__(root_view, device)
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: ArticulationView = weakref.proxy(root_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False

        # obtain global simulation view
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
        """Whether the articulation data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the articulation data is fully instantiated and ready to use.

        .. note:: Once this quantity is set to True, it cannot be changed.

        Args:
            value: The primed state.

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self._is_primed:
            raise ValueError("The articulation data is already primed.")
        self._is_primed = True

    def update(self, dt: float) -> None:
        """Updates the data for the articulation.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""

    """
    Defaults - Initial state.
    """

    @property
    def default_root_pose(self) -> torch.Tensor:
        """Default root pose ``[pos, quat]`` in the local environment frame.

        The position and quaternion are of the articulation root's actor frame. Shape is (num_instances, 7).
        """
        return self._default_root_pose

    @default_root_pose.setter
    def default_root_pose(self, value: torch.Tensor) -> None:
        """Set the default root pose.

        Args:
            value: The default root pose. Shape is (num_instances, 7).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_pose = value

    @property
    def default_root_vel(self) -> torch.Tensor:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame.

        The linear and angular velocities are of the articulation root's center of mass frame.
        Shape is (num_instances, 6).
        """
        return self._default_root_vel

    @default_root_vel.setter
    def default_root_vel(self, value: torch.Tensor) -> None:
        """Set the default root velocity.

        Args:
            value: The default root velocity. Shape is (num_instances, 6).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_vel = value

    @property
    def default_root_state(self) -> torch.Tensor:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame.


        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of its center of mass frame. Shape is (num_instances, 13).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        warnings.warn(
            "Reading the root state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_root_pose and default_root_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.cat([self._default_root_pose, self._default_root_vel], dim=1)

    @default_root_state.setter
    def default_root_state(self, value: torch.Tensor) -> None:
        """Set the default root state.

        Args:
            value: The default root state. Shape is (num_instances, 13).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        warnings.warn(
            "Setting the root state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_root_pose and default_root_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_root_pose = value[:, :7]
        self._default_root_vel = value[:, 7:]

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default joint positions of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_pos

    @default_joint_pos.setter
    def default_joint_pos(self, value: torch.Tensor) -> None:
        """Set the default joint positions.

        Args:
            value: The default joint positions. Shape is (num_instances, num_joints).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_joint_pos = value

    @property
    def default_joint_vel(self) -> torch.Tensor:
        """Default joint velocities of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_joint_vel

    @default_joint_vel.setter
    def default_joint_vel(self, value: torch.Tensor) -> None:
        """Set the default joint velocities.

        Args:
            value: The default joint velocities. Shape is (num_instances, num_joints).

        Raises:
            ValueError: If the articulation data is already primed.
        """
        if self.is_primed:
            raise ValueError("The articulation data is already primed.")
        self._default_joint_vel = value

    """
    Joint commands -- Set into simulation.
    """

    @property
    def joint_pos_target(self) -> torch.Tensor:
        """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_pos_target

    @property
    def joint_vel_target(self) -> torch.Tensor:
        """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_vel_target

    @property
    def joint_effort_target(self) -> torch.Tensor:
        """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._joint_effort_target

    """
    Joint commands -- Explicit actuators.
    """

    @property
    def computed_torque(self) -> torch.Tensor:
        """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

        This quantity is the raw torque output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._computed_torque

    @property
    def applied_torque(self) -> torch.Tensor:
        """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

        These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
        actuator model.
        """
        return self._applied_torque

    """
    Joint properties
    """

    @property
    def joint_stiffness(self) -> torch.Tensor:
        """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_stiffness

    @property
    def joint_damping(self) -> torch.Tensor:
        """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._joint_damping

    @property
    def joint_armature(self) -> torch.Tensor:
        """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_armature

    @property
    def joint_friction_coeff(self) -> torch.Tensor:
        """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_friction_coeff

    @property
    def joint_dynamic_friction_coeff(self) -> torch.Tensor:
        """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_dynamic_friction_coeff

    @property
    def joint_viscous_friction_coeff(self) -> torch.Tensor:
        """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_viscous_friction_coeff

    @property
    def joint_pos_limits(self) -> torch.Tensor:
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        return self._joint_pos_limits

    @property
    def joint_vel_limits(self) -> torch.Tensor:
        """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_vel_limits

    @property
    def joint_effort_limits(self) -> torch.Tensor:
        """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._joint_effort_limits

    """
    Joint properties - Custom.
    """

    @property
    def soft_joint_pos_limits(self) -> torch.Tensor:
        r"""Soft joint positions limits for all joints. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.The soft joint position limits are computed as
        a sub-region of the :attr:`joint_pos_limits` based on the
        :attr:`~isaaclab.assets.ArticulationCfg.soft_joint_pos_limit_factor` parameter.

        Consider the joint position limits :math:`[lower, upper]` and the soft joint position limits
        :math:`[soft_lower, soft_upper]`. The soft joint position limits are computed as:

        .. math::

            soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
            soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

        The soft joint position limits help specify a safety region around the joint limits. It isn't used by the
        simulation, but is useful for learning agents to prevent the joint positions from violating the limits.
        """
        return self._soft_joint_pos_limits

    @property
    def soft_joint_vel_limits(self) -> torch.Tensor:
        """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._soft_joint_vel_limits

    @property
    def gear_ratio(self) -> torch.Tensor:
        """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
        return self._gear_ratio

    """
    Fixed tendon properties.
    """

    @property
    def fixed_tendon_stiffness(self) -> torch.Tensor:
        """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_stiffness

    @property
    def fixed_tendon_damping(self) -> torch.Tensor:
        """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_damping

    @property
    def fixed_tendon_limit_stiffness(self) -> torch.Tensor:
        """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_limit_stiffness

    @property
    def fixed_tendon_rest_length(self) -> torch.Tensor:
        """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_rest_length

    @property
    def fixed_tendon_offset(self) -> torch.Tensor:
        """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        return self._fixed_tendon_offset

    @property
    def fixed_tendon_pos_limits(self) -> torch.Tensor:
        """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""
        return self._fixed_tendon_pos_limits

    """
    Spatial tendon properties.
    """

    @property
    def spatial_tendon_stiffness(self) -> torch.Tensor:
        """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_stiffness

    @property
    def spatial_tendon_damping(self) -> torch.Tensor:
        """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_damping

    @property
    def spatial_tendon_limit_stiffness(self) -> torch.Tensor:
        """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_limit_stiffness

    @property
    def spatial_tendon_offset(self) -> torch.Tensor:
        """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        return self._spatial_tendon_offset

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_view.get_root_transforms().clone()
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
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

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
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

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_view.get_root_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia. Shape is (num_instances, 13).
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
        """Body mass ``wp.float32`` in the world frame. Shape is (num_instances, num_bodies)."""
        return self._body_mass.to(self.device)

    @property
    def body_inertia(self) -> torch.Tensor:
        """Body inertia ``wp.mat33`` in the world frame. Shape is (num_instances, num_bodies, 3, 3)."""
        return self._body_inertia.to(self.device)

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self._root_view.get_link_transforms().clone()
            # set the buffer data and timestamp
            self._body_link_pose_w.data = poses
            self._body_link_pose_w.timestamp = self._sim_timestamp

        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocities = self.body_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            velocities[..., :3] += torch.linalg.cross(
                velocities[..., 3:], quat_apply(self.body_link_quat_w, -self.body_com_pos_b), dim=-1
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.data = velocities
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = combine_frame_transforms(
                self.body_link_pos_w, self.body_link_quat_w, self.body_com_pos_b, self.body_com_quat_b
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._body_com_pose_w.timestamp = self._sim_timestamp

        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_view.get_link_velocities()
            self._body_com_vel_w.timestamp = self._sim_timestamp

        return self._body_com_vel_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        if self._body_state_w.timestamp < self._sim_timestamp:
            self._body_state_w.data = torch.cat((self.body_link_pose_w, self.body_com_vel_w), dim=-1)
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w.data

    @property
    def body_link_state_w(self):
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            self._body_link_state_w.data = torch.cat((self.body_link_pose_w, self.body_link_vel_w), dim=-1)
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w.data

    @property
    def body_com_state_w(self):
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            self._body_com_state_w.data = torch.cat((self.body_com_pose_w, self.body_com_vel_w), dim=-1)
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w.data

    @property
    def body_com_acc_w(self):
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_view.get_link_accelerations()
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
            self._body_com_pose_b.data = pose
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation`_ and the underlying
        `PhysX Tensor API`_.

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """

        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.timestamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    """
    Joint state properties.
    """

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_view.get_dof_positions()
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_view.get_dof_velocities()
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data

    @property
    def joint_acc(self) -> torch.Tensor:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            self._joint_acc.data = (self.joint_vel - self._previous_joint_vel) / time_elapsed
            self._joint_acc.timestamp = self._sim_timestamp
            # update the previous joint velocity
            self._previous_joint_vel[:] = self.joint_vel
        return self._joint_acc.data

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
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

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
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
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self.body_com_pose_b[..., 3:7]

    def _create_buffers(self) -> None:
        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        self._body_link_pose_w = TimestampedBuffer()
        self._body_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_pose_w = TimestampedBuffer()
        self._body_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._body_link_state_w = TimestampedBuffer()
        self._body_com_state_w = TimestampedBuffer()
        # -- joint state
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()
        self._joint_acc = TimestampedBuffer()
        self._body_incoming_joint_wrench_b = TimestampedBuffer()

        num_dofs = self._root_view.shared_metatype.dof_count
        num_fixed_tendons = self._root_view.max_fixed_tendons
        num_spatial_tendons = self._root_view.max_spatial_tendons

        # Default root pose and velocity
        self._default_root_pose = torch.zeros((self._root_view.count, 7), device=self.device)
        self._default_root_vel = torch.zeros((self._root_view.count, 6), device=self.device)
        self._default_joint_pos = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        self._default_joint_vel = torch.zeros((self._root_view.count, num_dofs), device=self.device)

        # Initialize history for finite differencing
        self._previous_joint_vel = self._root_view.get_dof_velocities().clone()

        # Pre-allocated buffers
        # -- Joint commands (set into simulation)
        self._joint_pos_target = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        self._joint_vel_target = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        self._joint_effort_target = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        # -- Joint commands (explicit actuator model)
        self._computed_torque = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        self._applied_torque = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        # -- Joint properties
        self._joint_stiffness = self._root_view.get_dof_stiffnesses().to(self.device).clone()
        self._joint_damping = self._root_view.get_dof_dampings().to(self.device).clone()
        self._joint_armature = self._root_view.get_dof_armatures().to(self.device).clone()
        friction_props = self._root_view.get_dof_friction_properties()
        self._joint_friction_coeff = friction_props[:, :, 0].to(self.device).clone()
        self._joint_dynamic_friction_coeff = friction_props[:, :, 1].to(self.device).clone()
        self._joint_viscous_friction_coeff = friction_props[:, :, 2].to(self.device).clone()
        self._joint_pos_limits = self._root_view.get_dof_limits().to(self.device).clone()
        self._joint_vel_limits = self._root_view.get_dof_max_velocities().to(self.device).clone()
        self._joint_effort_limits = self._root_view.get_dof_max_forces().to(self.device).clone()
        # -- Joint properties (custom)
        self._soft_joint_pos_limits = torch.zeros((self._root_view.count, num_dofs, 2), device=self.device)
        self._soft_joint_vel_limits = torch.zeros((self._root_view.count, num_dofs), device=self.device)
        self._gear_ratio = torch.ones((self._root_view.count, num_dofs), device=self.device)
        # -- Fixed tendon properties
        if num_fixed_tendons > 0:
            self._fixed_tendon_stiffness = self._root_view.get_fixed_tendon_stiffnesses().to(self.device).clone()
            self._fixed_tendon_damping = self._root_view.get_fixed_tendon_dampings().to(self.device).clone()
            self._fixed_tendon_limit_stiffness = (
                self._root_view.get_fixed_tendon_limit_stiffnesses().to(self.device).clone()
            )
            self._fixed_tendon_rest_length = self._root_view.get_fixed_tendon_rest_lengths().to(self.device).clone()
            self._fixed_tendon_offset = self._root_view.get_fixed_tendon_offsets().to(self.device).clone()
            self._fixed_tendon_pos_limits = self._root_view.get_fixed_tendon_limits().to(self.device).clone()
        else:
            self._fixed_tendon_stiffness = None
            self._fixed_tendon_damping = None
            self._fixed_tendon_limit_stiffness = None
            self._fixed_tendon_rest_length = None
            self._fixed_tendon_offset = None
            self._fixed_tendon_pos_limits = None
        # -- Spatial tendon properties
        if num_spatial_tendons > 0:
            self._spatial_tendon_stiffness = self._root_view.get_spatial_tendon_stiffnesses().to(self.device).clone()
            self._spatial_tendon_damping = self._root_view.get_spatial_tendon_dampings().to(self.device).clone()
            self._spatial_tendon_limit_stiffness = (
                self._root_view.get_spatial_tendon_limit_stiffnesses().to(self.device).clone()
            )
            self._spatial_tendon_offset = self._root_view.get_spatial_tendon_offsets().to(self.device).clone()
        else:
            self._spatial_tendon_stiffness = None
            self._spatial_tendon_damping = None
            self._spatial_tendon_limit_stiffness = None
            self._spatial_tendon_offset = None
        # -- Body properties
        self._body_mass = self._root_view.get_masses().to(self.device).clone()
        self._body_inertia = self._root_view.get_inertias().to(self.device).clone()

        # -- Defaults (Lazy allocation of default values)
        self._default_mass = None
        self._default_inertia = None
        self._default_joint_stiffness = None
        self._default_joint_damping = None
        self._default_joint_armature = None
        self._default_joint_friction_coeff = None
        self._default_joint_viscous_friction_coeff = None
        self._default_joint_pos_limits = None
        self._default_fixed_tendon_stiffness = None
        self._default_fixed_tendon_damping = None
        self._default_fixed_tendon_limit_stiffness = None
        self._default_fixed_tendon_rest_length = None
        self._default_fixed_tendon_offset = None
        self._default_fixed_tendon_pos_limits = None
        self._default_spatial_tendon_stiffness = None
        self._default_spatial_tendon_damping = None
        self._default_spatial_tendon_limit_stiffness = None
        self._default_spatial_tendon_offset = None

    """
    Backward compatibility. (Deprecated properties)
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

    @property
    def joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        warnings.warn(
            "The `joint_limits` property will be deprecated in a IsaacLab 4.0. Please use `joint_pos_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_pos_limits` instead."""
        warnings.warn(
            "The `default_joint_limits` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_joint_pos_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        warnings.warn(
            "The `joint_velocity_limits` property will be deprecated in a IsaacLab 4.0. Please use"
            " `joint_vel_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        warnings.warn(
            "The `joint_friction` property will be deprecated in a IsaacLab 4.0. Please use"
            " `joint_friction_coeff` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.joint_friction_coeff

    @property
    def default_joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        warnings.warn(
            "The `default_joint_friction` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_joint_friction_coeff` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        warnings.warn(
            "The `fixed_tendon_limit` property will be deprecated in a IsaacLab 4.0. Please use"
            " `fixed_tendon_pos_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fixed_tendon_pos_limits

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        warnings.warn(
            "The `default_fixed_tendon_limit` property will be deprecated in a IsaacLab 4.0. Please use"
            " `default_fixed_tendon_pos_limits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_fixed_tendon_pos_limits

    """
    Defaults - Default values will no longer be stored.
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

    @property
    def default_joint_stiffness(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_stiffness` instead and manage the default joint stiffness
        manually."""
        warnings.warn(
            "The `default_joint_stiffness` property will be deprecated in a IsaacLab 4.0. Please use `joint_stiffness`"
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_stiffness is None:
            self._default_joint_stiffness = self.joint_stiffness.clone()
        return self._default_joint_stiffness

    @property
    def default_joint_damping(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_damping` instead and manage the default joint damping
        manually."""
        warnings.warn(
            "The `default_joint_damping` property will be deprecated in a IsaacLab 4.0. Please use `joint_damping`"
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_damping is None:
            self._default_joint_damping = self.joint_damping.clone()
        return self._default_joint_damping

    @property
    def default_joint_armature(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_armature` instead and manage the default joint armature
        manually."""
        warnings.warn(
            "The `default_joint_armature` property will be deprecated in a IsaacLab 4.0. Please use `joint_armature`"
            "instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_armature is None:
            self._default_joint_armature = self.joint_armature.clone()
        return self._default_joint_armature

    @property
    def default_joint_friction_coeff(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead and manage the default joint friction
        coefficient manually."""
        warnings.warn(
            "The `default_joint_friction_coeff` property will be deprecated in a IsaacLab 4.0. Please use"
            "`joint_friction_coeff` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_friction_coeff is None:
            self._default_joint_friction_coeff = self.joint_friction_coeff.clone()
        return self._default_joint_friction_coeff

    @property
    def default_joint_viscous_friction_coeff(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_viscous_friction_coeff` instead and manage the default joint
        viscous friction coefficient manually."""
        warnings.warn(
            "The `default_joint_viscous_friction_coeff` property will be deprecated in a IsaacLab 4.0. Please use"
            "`joint_viscous_friction_coeff` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_viscous_friction_coeff is None:
            self._default_joint_viscous_friction_coeff = self.joint_viscous_friction_coeff.clone()
        return self._default_joint_viscous_friction_coeff

    @property
    def default_joint_pos_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead and manage the default joint position
        limits manually."""
        warnings.warn(
            "The `default_joint_pos_limits` property will be deprecated in a IsaacLab 4.0. Please use"
            "`joint_pos_limits` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_joint_pos_limits is None:
            self._default_joint_pos_limits = self.joint_pos_limits.clone()
        return self._default_joint_pos_limits

    @property
    def default_fixed_tendon_stiffness(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_stiffness` instead and manage the default fixed tendon
        stiffness manually."""
        warnings.warn(
            "The `default_fixed_tendon_stiffness` property will be deprecated in a IsaacLab 4.0. Please use"
            "`fixed_tendon_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_stiffness is None:
            self._default_fixed_tendon_stiffness = self.fixed_tendon_stiffness.clone()
        return self._default_fixed_tendon_stiffness

    @property
    def default_fixed_tendon_damping(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_damping` instead and manage the default fixed tendon
        damping manually."""
        warnings.warn(
            "The `default_fixed_tendon_damping` property will be deprecated in a IsaacLab 4.0. Please use"
            "`fixed_tendon_damping` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_damping is None:
            self._default_fixed_tendon_damping = self.fixed_tendon_damping.clone()
        return self._default_fixed_tendon_damping

    @property
    def default_fixed_tendon_limit_stiffness(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_limit_stiffness` instead and manage the default fixed
        tendon limit stiffness manually."""
        warnings.warn(
            "The `default_fixed_tendon_limit_stiffness` property will be deprecated in a IsaacLab 4.0. Please use"
            "`fixed_tendon_limit_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_limit_stiffness is None:
            self._default_fixed_tendon_limit_stiffness = self.fixed_tendon_limit_stiffness.clone()
        return self._default_fixed_tendon_limit_stiffness

    @property
    def default_fixed_tendon_rest_length(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_rest_length` instead and manage the default fixed tendon
        rest length manually."""
        warnings.warn(
            "The `default_fixed_tendon_rest_length` property will be deprecated in a IsaacLab 4.0. Please use"
            "`fixed_tendon_rest_length` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_rest_length is None:
            self._default_fixed_tendon_rest_length = self.fixed_tendon_rest_length.clone()
        return self._default_fixed_tendon_rest_length

    @property
    def default_fixed_tendon_offset(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_offset` instead and manage the default fixed tendon
        offset manually."""
        warnings.warn(
            "The `default_fixed_tendon_offset` property will be deprecated in a IsaacLab 4.0. Please use"
            "`fixed_tendon_offset` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_offset is None:
            self._default_fixed_tendon_offset = self.fixed_tendon_offset.clone()
        return self._default_fixed_tendon_offset

    @property
    def default_fixed_tendon_pos_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead and manage the default fixed tendon
        position limits manually."""
        warnings.warn(
            "The `default_fixed_tendon_pos_limits` property will be deprecated in a IsaacLab 4.0. Please use"
            "`fixed_tendon_pos_limits` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_fixed_tendon_pos_limits is None:
            self._default_fixed_tendon_pos_limits = self.fixed_tendon_pos_limits.clone()
        return self._default_fixed_tendon_pos_limits

    @property
    def default_spatial_tendon_stiffness(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`spatial_tendon_stiffness` instead and manage the default spatial
        tendon stiffness manually."""
        warnings.warn(
            "The `default_spatial_tendon_stiffness` property will be deprecated in a IsaacLab 4.0. Please use"
            "`spatial_tendon_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_stiffness is None:
            self._default_spatial_tendon_stiffness = self.spatial_tendon_stiffness.clone()
        return self._default_spatial_tendon_stiffness

    @property
    def default_spatial_tendon_damping(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`spatial_tendon_damping` instead and manage the default spatial tendon
        damping manually."""
        warnings.warn(
            "The `default_spatial_tendon_damping` property will be deprecated in a IsaacLab 4.0. Please use"
            "`spatial_tendon_damping` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_damping is None:
            self._default_spatial_tendon_damping = self.spatial_tendon_damping.clone()
        return self._default_spatial_tendon_damping

    @property
    def default_spatial_tendon_limit_stiffness(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`spatial_tendon_limit_stiffness` instead and manage the default
        spatial tendon limit stiffness manually."""
        warnings.warn(
            "The `default_spatial_tendon_limit_stiffness` property will be deprecated in a IsaacLab 4.0. Please use"
            "`spatial_tendon_limit_stiffness` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_limit_stiffness is None:
            self._default_spatial_tendon_limit_stiffness = self.spatial_tendon_limit_stiffness.clone()
        return self._default_spatial_tendon_limit_stiffness

    @property
    def default_spatial_tendon_offset(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`spatial_tendon_offset` instead and manage the default spatial tendon
        offset manually."""
        warnings.warn(
            "The `default_spatial_tendon_offset` property will be deprecated in a IsaacLab 4.0. Please use"
            "`spatial_tendon_offset` instead. The default value will need to be managed manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_spatial_tendon_offset is None:
            self._default_spatial_tendon_offset = self.spatial_tendon_offset.clone()
        return self._default_spatial_tendon_offset
