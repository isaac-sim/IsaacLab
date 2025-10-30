# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import weakref

import warp as wp
import logging
import warnings
from typing import TYPE_CHECKING

from isaaclab.sim._impl.newton_manager import NewtonManager
from newton.selection import ArticulationView as NewtonArticulationView
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.newton.assets.core.kernels import (
    vec13f,
    combine_pose_and_velocity_to_state,
    combine_pose_and_velocity_to_state_batched,
    split_transform_array_to_position_array,
    split_transform_array_to_quaternion_array,
    split_spatial_vectory_array_to_linear_velocity_array,
    split_spatial_vectory_array_to_angular_velocity_array,
    split_transform_batched_array_to_position_batched_array,
    split_transform_batched_array_to_quaternion_batched_array,
    split_spatial_vectory_batched_array_to_linear_velocity_batched_array,
    split_spatial_vectory_batched_array_to_angular_velocity_batched_array,
    combine_frame_transforms_partial_batch,
    derive_body_acceleration_from_velocity_batched,
    generate_pose_from_position_with_unit_quaternion_batched,
    project_com_velocity_to_link_frame_batch,
    combine_frame_transforms_partial_root,
    compute_heading,
    project_com_velocity_to_link_frame_root,
    project_vec_from_pose_single,
    project_velocities_to_frame,
    derive_joint_acceleration_from_velocity,
    make_joint_pos_limits_from_lower_and_upper_limits,
)
from isaaclab.utils.helpers import deprecated, warn_overhead_cost
from isaaclab.utils.buffers import TimestampedWarpBuffer

from isaaclab.newton import Frontend, QuaternionFormat

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
warnings.simplefilter("once", UserWarning)
logging.captureWarnings(True)

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

    ..note:: This class is implemented so that all the properties can be accessed as either a Torch tensor or a Warp
    array. However, all the operations are performed on Warp arrays. To enable this, there is a set of internal
    only helper functions that perform all the warp operations. Internal classes depending on this class should not use
    properties directly! This is because they can either be Torch tensors or Warp arrays, and all the internal
    operations should be performed on the Warp arrays. Hence, internal operations should instead use the "private" 
    helper functions / attributes.
    """

    def __init__(self, root_view, device: str, frontend: Frontend = Frontend.TORCH, quaternion_format: QuaternionFormat = QuaternionFormat.XYZW):
        """Initializes the articulation data.

        Args:
            root_view: The root articulation view.
            device: The device used for processing.
            frontend: The frontend to use for the data.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: NewtonArticulationView = weakref.proxy(root_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0
        # obtain global simulation view
        gravity = wp.to_torch(NewtonManager.get_model().gravity)[0]
        gravity_dir = [float(i) / sum(gravity) for i in gravity]
        # Initialize constants
        self.GRAVITY_VEC_W = wp.vec3f(gravity_dir[0], gravity_dir[1], gravity_dir[2])
        self.FORWARD_VEC_B = wp.vec3f((1.0, 0.0, 0.0))
        # Set the frontend
        self._frontend = frontend
        self._quaternion_format = quaternion_format
        if self._frontend == Frontend.WARP:
            assert self._quaternion_format == QuaternionFormat.XYZW, "WARP frontend only supports XYZW quaternion format"
        else:
            if self._quaternion_format == QuaternionFormat.XYZW:
                warnings.warn("Using Torch frontend with XYZW quaternion format. Are you sure this is the desired behavior?")

        # Create the simulation bindings and buffers
        self._create_simulation_bindings()
        self._create_buffers()

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    @property
    def default_root_pose(self) -> wp.array | torch.Tensor:
        """Default root pose ``[pos, quat]`` in the local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the articulation root's actor frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._convert_pose_to_quaternion_format(self._cast_to_frontend(self._default_root_pose))

    @property
    def default_root_vel(self) -> wp.array | torch.Tensor:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the articulation root's center of mass frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._cast_to_frontend(self._default_root_vel)

    @property
    def default_joint_pos(self) -> wp.array | torch.Tensor:
        """Default joint positions of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._cast_to_frontend(self._default_joint_pos)

    @property
    def default_joint_vel(self) -> wp.array | torch.Tensor:
        """Default joint velocities of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._cast_to_frontend(self._default_joint_vel)

    ###
    # Joint commands. -- Set into the simulation 
    ###

    @property
    def joint_target(self) -> wp.array | torch.Tensor:
        """Joint target. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_target)

    @property
    def joint_pos_target(self) -> wp.array | torch.Tensor:
        return self.actuator_target

    @property
    def joint_vel_target(self) -> wp.array | torch.Tensor:
        return self.actuator_target

    @property
    def joint_effort(self) -> wp.array | torch.Tensor:
        """Joint effort. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_effort)

    ##
    # Joint commands -- Explicit actuators.
    ##

    @property
    def computed_effort(self) -> wp.array | torch.Tensor:
        """Joint efforts computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

        This quantity is the raw effort output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return self._cast_to_frontend(self._computed_effort)

    @property
    def applied_effort(self) -> wp.array | torch.Tensor:
        """Joint efforts applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

        These efforts are set into the simulation, after clipping the :attr:`computed_effort` based on the
        actuator model.
        """
        return self._cast_to_frontend(self._applied_effort)

    @property
    def actuator_stiffness(self) -> wp.array | torch.Tensor:
        """Actuator stiffness. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._actuator_stiffness)

    @property
    def actuator_damping(self) -> wp.array | torch.Tensor:
        """Actuator damping. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._actuator_damping)

    @property
    def actuator_control_mode(self) -> wp.array | torch.Tensor:
        """Actuator control mode. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._actuator_control_mode)

    @property
    def actuator_target(self) -> wp.array | torch.Tensor:
        """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return  self._cast_to_frontend(self._actuator_target)

    @property
    def actuator_effort_target(self) -> wp.array | torch.Tensor:
        """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return self._cast_to_frontend(self._actuator_effort_target)

    ##
    # Joint properties. -- Set into the simulation 
    ##

    @property
    def joint_control_mode(self) -> wp.array | torch.Tensor:
        """Joint control mode. Shape is (num_instances, num_joints).

        When using implicit actuator models Newton needs to know how the joints are controlled.
        The control mode can be one of the following:

        * None: 0
        * Position control: 1
        * Velocity control: 2

        This quantity is set by the :meth:`Articulation.write_joint_control_mode_to_sim` method.
        """
        return self._cast_to_frontend(self._sim_bind_joint_control_mode_sim)

    @property
    def joint_stiffness(self) -> wp.array | torch.Tensor:
        """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._cast_to_frontend(self._sim_bind_joint_stiffness_sim)

    @property
    def joint_damping(self) -> wp.array | torch.Tensor:
        """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return self._cast_to_frontend(self._sim_bind_joint_damping_sim)

    @property
    def joint_armature(self) -> wp.array | torch.Tensor:
        """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_armature)

    @property
    def joint_friction_coeff(self) -> wp.array | torch.Tensor:
        """Joint friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_friction_coeff)


    @property
    def joint_pos_limits_lower(self) -> wp.array | torch.Tensor:
        """Joint position limits lower provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_pos_limits_lower)

    @property
    def joint_pos_limits_upper(self) -> wp.array | torch.Tensor:
        """Joint position limits upper provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_pos_limits_upper)

    @property
    def joint_pos_limits(self) -> wp.array | torch.Tensor:
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        return self._cast_to_frontend(self._aggregate_joint_pos_limits())

    @property
    def joint_vel_limits(self) -> wp.array | torch.Tensor:
        """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_vel_limits_sim)

    @property
    def joint_effort_limits(self) -> wp.array | torch.Tensor:
        """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_effort_limits_sim)

    ##
    # Joint properties - Custom.
    ##

    @property
    def joint_dynamic_friction_coeff(self) -> wp.array | torch.Tensor:
        """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._joint_dynamic_friction)

    @property
    def joint_viscous_friction_coeff(self) -> wp.array | torch.Tensor:
        """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._joint_viscous_friction)

    @property
    def soft_joint_pos_limits(self) -> wp.array | torch.Tensor:
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
        return self._cast_to_frontend(self._soft_joint_pos_limits)

    @property
    def soft_joint_vel_limits(self) -> wp.array | torch.Tensor:
        """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return self._cast_to_frontend(self._soft_joint_vel_limits)

    @property
    def gear_ratio(self) -> wp.array | torch.Tensor: #TODO: Mayank got some comments
        """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._gear_ratio)

    ##
    # Fixed tendon properties.
    ##

    @property
    def fixed_tendon_stiffness(self) -> torch.Tensor | wp.array:
        """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon stiffness is not supported in Newton.")

    @property
    def fixed_tendon_damping(self) -> torch.Tensor | wp.array:
        """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon damping is not supported in Newton.")

    @property
    def fixed_tendon_limit_stiffness(self) -> torch.Tensor | wp.array:
        """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon limit stiffness is not supported in Newton.")

    @property
    def fixed_tendon_rest_length(self) -> torch.Tensor | wp.array:
        """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon rest length is not supported in Newton.")

    @property
    def fixed_tendon_offset(self) -> torch.Tensor | wp.array:
        """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
        raise NotImplementedError("Fixed tendon offset is not supported in Newton.")

    @property
    def fixed_tendon_pos_limits(self) -> torch.Tensor | wp.array:
        """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""
        raise NotImplementedError("Fixed tendon position limits is not supported in Newton.")

    ##
    # Spatial tendon properties.
    ##

    @property
    def spatial_tendon_stiffness(self) -> torch.Tensor | wp.array:
        """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon stiffness is not supported in Newton.")

    @property
    def spatial_tendon_damping(self) -> torch.Tensor | wp.array:
        """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon damping is not supported in Newton.")

    @property
    def spatial_tendon_limit_stiffness(self) -> torch.Tensor | wp.array:
        """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon limit stiffness is not supported in Newton.")

    @property
    def spatial_tendon_offset(self) -> torch.Tensor | wp.array:
        """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
        raise NotImplementedError("Spatial tendon offset is not supported in Newton.")

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> wp.array | torch.Tensor:
        """Root link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._convert_pose_to_quaternion_format(self._cast_to_frontend(self._sim_bind_root_link_pose_w))

    @property
    def root_link_vel_w(self) -> wp.array | torch.Tensor:
        """Root link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        return self._cast_to_frontend(self._compute_root_link_vel_w())

    @property
    def root_com_pose_w(self) -> wp.array | torch.Tensor:
        """Root center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._convert_pose_to_quaternion_format(self._cast_to_frontend(self._compute_root_com_pose_w()))

    @property
    def root_com_vel_w(self) -> wp.array | torch.Tensor:
        """Root center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        return self._cast_to_frontend(self._sim_bind_root_com_vel_w)

    @property
    def root_state_w(self) -> wp.array | torch.Tensor:
        """Root state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """
        return self._convert_state_to_quaternion_format(self._cast_to_frontend(self._merge_pose_and_velocity_to_state(self._sim_bind_root_link_pose_w, self._sim_bind_root_com_vel_w)))

    @property
    def root_link_state_w(self) -> wp.array | torch.Tensor:
        """Root link state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's actor frame.
        """
        return self._convert_state_to_quaternion_format(self._cast_to_frontend(self._merge_pose_and_velocity_to_state(self._sim_bind_root_link_pose_w, self._compute_root_link_vel_w())))

    @property
    def root_com_state_w(self) -> wp.array | torch.Tensor:
        """Root center of mass state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's center of mass frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """
        return self._convert_state_to_quaternion_format(self._cast_to_frontend(self._merge_pose_and_velocity_to_state(self._compute_root_com_pose_w(), self._sim_bind_root_com_vel_w)))
    
    ##
    # Body state properties.
    ##

    @property
    def body_mass(self) -> wp.array | torch.Tensor:
        """Body mass ``wp.float32`` in the world frame. Shape is (num_instances, num_bodies)."""
        return self._cast_to_frontend(self._sim_bind_body_mass)

    @property
    def body_inertia(self) -> wp.array | torch.Tensor:
        """Body inertia ``wp.mat33`` in the world frame. Shape is (num_instances, num_bodies, 3, 3)."""
        return self._cast_to_frontend(self._sim_bind_body_inertia)

    @property
    def body_link_pose_w(self) -> wp.array | torch.Tensor:
        """Body link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._convert_pose_to_quaternion_format(self._cast_to_frontend(self._sim_bind_body_link_pose_w))

    @property
    def body_link_vel_w(self) -> wp.array | torch.Tensor:
        """Body link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        return self._cast_to_frontend(self._compute_body_link_vel_w())

    @property
    def body_com_pose_w(self) -> wp.array | torch.Tensor:
        """Body center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._convert_pose_to_quaternion_format(self._cast_to_frontend(self._compute_body_com_pose_w()))
    
    @property
    def body_com_vel_w(self) -> wp.array | torch.Tensor:
        """Body center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        return self._cast_to_frontend(self._sim_bind_body_com_vel_w)

    @property
    def body_state_w(self) -> wp.array | torch.Tensor:
        """State of all bodies ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation links' actor frame relative to the world.
        The velocity is of the articulation links' center of mass frame.
        """

        return self._convert_state_to_quaternion_format(self._cast_to_frontend(self._merge_pose_and_velocity_to_state(self._sim_bind_body_link_pose_w, self._sim_bind_body_com_vel_w)))

    @property
    def body_link_state_w(self) -> wp.array | torch.Tensor:
        """State of all bodies' link frame ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """

        return self._convert_state_to_quaternion_format(self._cast_to_frontend(self._merge_pose_and_velocity_to_state(self._sim_bind_body_link_pose_w, self._compute_body_link_vel_w())))

    @property
    def body_com_state_w(self) -> wp.array | torch.Tensor:
        """State of all bodies center of mass ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """

        return self._convert_state_to_quaternion_format(self._cast_to_frontend(self._merge_pose_and_velocity_to_state(self._compute_body_com_pose_w(), self._sim_bind_body_com_vel_w)))

    @property
    def body_com_acc_w(self) -> wp.array | torch.Tensor:
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The acceleration is in the form of [wx, wy, wz, vx, vy, vz].
        All values are relative to the world.
        """
        return self._cast_to_frontend(self._compute_body_com_acc_w())
    
    @property
    def body_com_pose_b(self) -> wp.array | torch.Tensor:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._convert_pose_to_quaternion_format(self._cast_to_frontend(self._compute_body_com_pose_b()))

    #TODO: Make sure this is implemented when the feature is available in Newton.
    @property
    def body_incoming_joint_wrench_b(self) -> wp.array | torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.
        """
        raise NotImplementedError("Body incoming joint wrench in body frame is not implemented for Newton.")

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self) -> wp.array | torch.Tensor:
        """Joint positions. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_pos)

    @property
    def joint_vel(self) -> wp.array | torch.Tensor:
        """Joint velocities. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._sim_bind_joint_vel)

    @property
    def joint_acc(self) -> wp.array | torch.Tensor:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        return self._cast_to_frontend(self._compute_joint_acc())

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self) -> wp.array | torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return self._cast_to_frontend(self._compute_projected_gravity_b())

    @property
    def heading_w(self) -> wp.array | torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        return self._cast_to_frontend(self._compute_heading_w())

    @property
    def root_link_vel_b(self) -> wp.array | torch.Tensor:
        """Root link velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [wx, wy, wz, vx, vy, vz].
        """
        return self._cast_to_frontend(self._compute_root_link_vel_b())

    @property
    def root_com_vel_b(self) -> wp.array | torch.Tensor:
        """Root center of mass velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [wx, wy, wz, vx, vy, vz].
        """
        return self._cast_to_frontend(self._compute_root_com_vel_b())

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor | wp.array:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._compute_root_link_vel_b()))

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor | wp.array:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._compute_root_link_vel_b()))

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor | wp.array:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._compute_root_com_vel_b()))

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor | wp.array:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._compute_root_com_vel_b()))

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> wp.array | torch.Tensor:
        """Root link position ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._cast_to_frontend(self._split_transform_to_position(self._sim_bind_root_link_pose_w))

    @property
    def root_link_quat_w(self) -> wp.array | torch.Tensor:
        """Root link orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._convert_quaternion_format(self._cast_to_frontend(self._split_transform_to_quaternion(self._sim_bind_root_link_pose_w)))

    @property
    def root_link_lin_vel_w(self) -> wp.array | torch.Tensor:
        """Root linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._compute_root_link_vel_w()))

    @property
    def root_link_ang_vel_w(self) -> wp.array | torch.Tensor:
        """Root link angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._compute_root_link_vel_w()))

    @property
    def root_com_pos_w(self) -> wp.array | torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._cast_to_frontend(self._split_transform_to_position(self._compute_root_com_pose_w()))

    @property
    def root_com_quat_w(self) -> wp.array | torch.Tensor:
        """Root center of mass orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the root rigid body's center of mass frame.
        """
        return self._convert_quaternion_format(self._cast_to_frontend(self._split_transform_to_quaternion(self._compute_root_com_pose_w())))

    @property
    def root_com_lin_vel_w(self) -> wp.array | torch.Tensor:
        """Root center of mass linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances,).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._sim_bind_root_com_vel_w))

    @property
    def root_com_ang_vel_w(self) -> wp.array | torch.Tensor:
        """Root center of mass angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._sim_bind_root_com_vel_w))

    @property
    def body_link_pos_w(self) -> wp.array | torch.Tensor:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self._cast_to_frontend(self._split_transform_to_position(self._sim_bind_body_link_pose_w))

    @property
    def body_link_quat_w(self) -> wp.array | torch.Tensor:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return self._convert_quaternion_format(self._cast_to_frontend(self._split_transform_to_quaternion(self._sim_bind_body_link_pose_w)))

    @property
    def body_link_lin_vel_w(self) -> wp.array | torch.Tensor:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._compute_body_link_vel_w()))

    @property
    def body_link_ang_vel_w(self) -> wp.array | torch.Tensor:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._compute_body_link_vel_w()))

    @property
    def body_com_pos_w(self) -> wp.array | torch.Tensor:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return self._cast_to_frontend(self._split_transform_to_position(self._compute_body_com_pose_w()))

    @property
    def body_com_quat_w(self) -> wp.array | torch.Tensor:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the articulation bodies' actor frame.
        """
        return self._convert_quaternion_format(self._cast_to_frontend(self._split_transform_to_quaternion(self._compute_body_com_pose_w())))

    @property
    def body_com_lin_vel_w(self) -> wp.array | torch.Tensor:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._sim_bind_body_com_vel_w))

    @property
    def body_com_ang_vel_w(self) -> wp.array | torch.Tensor:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._sim_bind_body_com_vel_w))

    @property
    def body_com_lin_acc_w(self) -> wp.array | torch.Tensor:
        """Linear acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_linear_velocity(self._compute_body_com_acc_w()))

    @property
    def body_com_ang_acc_w(self) -> wp.array | torch.Tensor:
        """Angular acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self._cast_to_frontend(self._split_velocity_to_angular_velocity(self._compute_body_com_acc_w()))

    @property
    def body_com_pos_b(self) -> wp.array | torch.Tensor:
        """Center of mass position ``wp.vec3f`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The position is in the form of [x, y, z].
        This quantity is the position of the center of mass frame of the rigid body relative to the body's link frame.
        """
        return self._cast_to_frontend(self._sim_bind_body_com_pos_b)

    @property
    def body_com_quat_b(self) -> wp.array | torch.Tensor:
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self._convert_quaternion_format(self._cast_to_frontend(self._split_transform_to_quaternion(self._compute_body_com_pose_b())))

    ##
    # Backward compatibility. -- Deprecated properties.
    ##

    @property
    @deprecated("default_root_pose", "default_root_vel", since="3.0.0", remove_in="4.0.0")
    def default_root_state(self) -> wp.array | torch.Tensor:
        """Same as :attr:`default_root_pose`."""
        state = wp.zeros((self._root_view.count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_view.count),
            device=self.device,
            inputs=[
                self._default_root_pose,
                self._default_root_vel,
                state,
            ],
        )
        return self._convert_state_to_quaternion_format(self._cast_to_frontend(state))

    @property
    @deprecated("root_link_pose_w", since="3.0.0", remove_in="4.0.0")
    def root_pose_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @property
    @deprecated("root_link_pos_w", since="3.0.0", remove_in="4.0.0")
    def root_pos_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    @deprecated("root_link_quat_w", since="3.0.0", remove_in="4.0.0")
    def root_quat_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    @deprecated("root_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_vel_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @property
    @deprecated("root_com_lin_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_lin_vel_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @property
    @deprecated("root_com_ang_vel_w", since="3.0.0", remove_in="4.0.0")
    def root_ang_vel_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @property
    @deprecated("root_com_lin_vel_b", since="3.0.0", remove_in="4.0.0")
    def root_lin_vel_b(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @property
    @deprecated("root_com_ang_vel_b", since="3.0.0", remove_in="4.0.0")
    def root_ang_vel_b(self) -> wp.array | torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @property
    @deprecated("body_link_pose_w", since="3.0.0", remove_in="4.0.0")
    def body_pose_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @property
    @deprecated("body_link_pos_w", since="3.0.0", remove_in="4.0.0")
    def body_pos_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @property
    @deprecated("body_link_quat_w", since="3.0.0", remove_in="4.0.0")
    def body_quat_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @property
    @deprecated("body_com_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_vel_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @property
    @deprecated("body_com_lin_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_lin_vel_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @property
    @deprecated("body_com_ang_vel_w", since="3.0.0", remove_in="4.0.0")
    def body_ang_vel_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @property
    @deprecated("body_com_acc_w", since="3.0.0", remove_in="4.0.0")
    def body_acc_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @property
    @deprecated("body_com_lin_acc_w", since="3.0.0", remove_in="4.0.0")
    def body_lin_acc_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @property
    @deprecated("body_com_ang_acc_w", since="3.0.0", remove_in="4.0.0")
    def body_ang_acc_w(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @property
    @deprecated("body_com_pos_b", since="3.0.0", remove_in="4.0.0")
    def com_pos_b(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @property
    @deprecated("body_com_quat_b", since="3.0.0", remove_in="4.0.0")
    def com_quat_b(self) -> wp.array | torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @property
    @deprecated("joint_pos_limits", since="3.0.0", remove_in="4.0.0")
    def joint_limits(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        return self.joint_pos_limits

    @property
    @deprecated("joint_friction_coeff", since="3.0.0", remove_in="4.0.0")
    def joint_friction(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        return self.joint_friction_coeff

    @property
    @deprecated("fixed_tendon_limit", since="3.0.0", remove_in="4.0.0")
    def fixed_tendon_limit(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        return self.fixed_tendon_pos_limits

    @property
    @deprecated("applied_effort", since="3.0.0", remove_in="4.0.0")
    def applied_torque(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`applied_effort` instead."""
        return self.applied_effort

    @property
    @deprecated("computed_effort", since="3.0.0", remove_in="4.0.0")
    def computed_torque(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`computed_effort` instead."""
        return self.computed_effort

    @property
    @deprecated("joint_dynamic_friction_coeff", since="3.0.0", remove_in="4.0.0")
    def joint_dynamic_friction(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`joint_dynamic_friction_coeff` instead."""
        return self.joint_dynamic_friction_coeff

    @property
    @deprecated("actuator_effort_target", since="3.0.0", remove_in="4.0.0")
    def joint_effort_target(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`actuator_effort_target` instead."""
        return self.actuator_effort_target

    @property
    @deprecated("joint_viscous_friction_coeff", since="3.0.0", remove_in="4.0.0")
    def joint_viscous_friction(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`joint_viscous_friction_coeff` instead."""
        return self.joint_viscous_friction_coeff

    @property
    @deprecated("joint_vel_limits", since="3.0.0", remove_in="4.0.0")
    def joint_velocity_limits(self) -> wp.array | torch.Tensor:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        return self.joint_vel_limits

    ###
    # Helper functions.
    ###
    
    def _cast_to_frontend(self, data: wp.array) -> wp.array | torch.Tensor:
        return wp.to_torch(data) if self._frontend == Frontend.TORCH else data

    def _convert_quaternion_format(self, quaternion: torch.Tensor) -> torch.Tensor:
        if self._frontend == Frontend.TORCH:
            return quaternion.clone().roll(1, dims=-1) if self._quaternion_format == QuaternionFormat.WXYZ else quaternion
        else:
            return quaternion

    def _convert_state_to_quaternion_format(self, state: torch.Tensor) -> torch.Tensor:
        if self._frontend == Frontend.TORCH:
            state_ = state.clone()
            state_[..., 3:7] = self._convert_quaternion_format(state[..., 3:7])
            return state_
        else:
            return state

    def _convert_pose_to_quaternion_format(self, pose: torch.Tensor) -> torch.Tensor:
        if self._frontend == Frontend.TORCH:
            pose_ = pose.clone()
            pose_[..., 3:7] = self._convert_quaternion_format(pose[..., 3:7])
            return pose_
        else:
            return pose

    def _create_simulation_bindings(self) -> None:
        """Create simulation bindings for the root data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """
        # -- root properties
        self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(NewtonManager.get_state_0())
        self._sim_bind_root_com_vel_w = self._root_view.get_root_velocities(NewtonManager.get_state_0())
        # -- body properties
        self._sim_bind_body_com_pos_b = self._root_view.get_attribute("body_com", NewtonManager.get_model())
        self._sim_bind_body_link_pose_w = self._root_view.get_link_transforms(NewtonManager.get_state_0())
        self._sim_bind_body_com_vel_w = self._root_view.get_link_velocities(NewtonManager.get_state_0())
        self._sim_bind_body_mass = self._root_view.get_attribute("body_mass", NewtonManager.get_model())
        self._sim_bind_body_inertia = self._root_view.get_attribute("body_inertia", NewtonManager.get_model())
        self._sim_bind_body_external_wrench = self._root_view.get_attribute("body_f", NewtonManager.get_state_0())
        # -- joint properties
        self._sim_bind_joint_pos_limits_lower = self._root_view.get_attribute(
            "joint_limit_lower", NewtonManager.get_model()
        )
        self._sim_bind_joint_pos_limits_upper = self._root_view.get_attribute(
            "joint_limit_upper", NewtonManager.get_model()
        )
        self._sim_bind_joint_stiffness_sim = self._root_view.get_attribute(
            "joint_target_ke", NewtonManager.get_model()
        )
        self._sim_bind_joint_damping_sim = self._root_view.get_attribute(
            "joint_target_kd", NewtonManager.get_model()
        )
        self._sim_bind_joint_armature = self._root_view.get_attribute(
            "joint_armature", NewtonManager.get_model()
        )
        self._sim_bind_joint_friction_coeff = self._root_view.get_attribute(
            "joint_friction", NewtonManager.get_model()
        )
        self._sim_bind_joint_vel_limits_sim = self._root_view.get_attribute(
            "joint_velocity_limit", NewtonManager.get_model()
        )
        self._sim_bind_joint_effort_limits_sim = self._root_view.get_attribute(
            "joint_effort_limit", NewtonManager.get_model()
        )
        self._sim_bind_joint_control_mode_sim = self._root_view.get_attribute(
            "joint_dof_mode", NewtonManager.get_model()
        )
        # -- joint states
        self._sim_bind_joint_pos = self._root_view.get_dof_positions(NewtonManager.get_state_0())
        self._sim_bind_joint_vel = self._root_view.get_dof_velocities(NewtonManager.get_state_0())
        # -- joint commands (sent to the simulation)
        self._sim_bind_joint_effort = self._root_view.get_attribute("joint_f", NewtonManager.get_control())
        self._sim_bind_joint_target = self._root_view.get_attribute(
            "joint_target", NewtonManager.get_control()
        )

    def _create_buffers(self) -> None:
        """Create buffers for the root data."""

        # Short-hand for the number of instances, number of links, and number of joints.
        n = self._root_view.count
        nl = self._root_view.link_count
        nd = self._root_view.joint_dof_count

        # MASKS
        self.ALL_ENV_MASK = wp.ones((n,), dtype=wp.bool, device=self.device)
        self.ALL_BODY_MASK = wp.ones((nl,), dtype=wp.bool, device=self.device)
        self.ALL_JOINT_MASK = wp.ones((nd,), dtype=wp.bool, device=self.device)
        self.ENV_MASK = wp.zeros((n,), dtype=wp.bool, device=self.device)
        self.BODY_MASK = wp.zeros((nl,), dtype=wp.bool, device=self.device)
        self.JOINT_MASK = wp.zeros((nd,), dtype=wp.bool, device=self.device)

        # Initialize history for finite differencing. If the articulation is fixed, the root com velocity is not
        # available, so we use zeros.
        try:
            self._previous_root_com_vel = wp.clone(self._root_view.get_root_velocities(NewtonManager.get_state_0()))
        except:
            self._previous_root_com_vel = wp.zeros((n, nl), dtype=wp.spatial_vectorf, device=self.device)
        # -- default root pose and velocity
        self._default_root_pose = wp.zeros((n), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((n), dtype=wp.spatial_vectorf, device=self.device)
        # -- default joint positions and velocities
        self._default_joint_pos = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        self._default_joint_vel = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        # -- joint commands (sent to the actuator from the user)
        self._actuator_target = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        self._actuator_effort_target = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        # -- computed joint efforts from the actuator models
        self._computed_effort = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        self._applied_effort = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        # -- joint properties for the actuator models
        self._actuator_stiffness = wp.clone(self._sim_bind_joint_stiffness_sim)
        self._actuator_damping = wp.clone(self._sim_bind_joint_damping_sim)
        self._actuator_control_mode = wp.clone(self._sim_bind_joint_control_mode_sim)
        # -- other data that are filled based on explicit actuator models
        self._joint_dynamic_friction = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        self._joint_viscous_friction = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        self._soft_joint_vel_limits = wp.zeros((n, nd), dtype=wp.float32, device=self.device)
        self._gear_ratio = wp.ones((n, nd), dtype=wp.float32, device=self.device)
        # -- update the soft joint position limits
        self._soft_joint_pos_limits = wp.zeros((n, nd), dtype=wp.vec2f, device=self.device)

        # Initialize history for finite differencing
        self._previous_joint_vel = wp.clone(self._root_view.get_dof_velocities(NewtonManager.get_state_0()))
        self._previous_body_com_vel = wp.clone(self._root_view.get_link_velocities(NewtonManager.get_state_0()))

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_vel_w = TimestampedWarpBuffer(shape=(n,), dtype=wp.spatial_vectorf)
        self._root_link_vel_b = TimestampedWarpBuffer(shape=(n,), dtype=wp.spatial_vectorf)
        self._projected_gravity_b = TimestampedWarpBuffer(shape=(n,), dtype=wp.vec3f)
        self._heading_w = TimestampedWarpBuffer(shape=(n,), dtype=wp.float32)
        self._body_link_vel_w = TimestampedWarpBuffer(shape=(n, nl), dtype=wp.spatial_vectorf)
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedWarpBuffer(shape=(n,), dtype=wp.transformf)
        self._root_com_vel_b = TimestampedWarpBuffer(shape=(n,), dtype=wp.spatial_vectorf)
        self._root_com_acc_w = TimestampedWarpBuffer(shape=(n,), dtype=wp.spatial_vectorf)
        self._body_com_pose_w = TimestampedWarpBuffer(shape=(n, nl), dtype=wp.transformf)
        self._body_com_acc_w = TimestampedWarpBuffer(shape=(n, nl), dtype=wp.spatial_vectorf)
        # -- joint state
        self._joint_acc = TimestampedWarpBuffer(shape=(n, nd), dtype=wp.float32)
        # self._body_incoming_joint_wrench_b = TimestampedWarpBuffer(shape=(n, nd), dtype=wp.spatial_vectorf)

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint and body acceleration buffers at a higher frequency since we do finite
        # differencing.
        self._compute_joint_acc()
        self._compute_body_com_acc_w()

    ###
    # Internal joint property calls.
    ###

    def _compute_joint_acc(self) -> wp.array:
        """Compute the joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            wp.launch(
                derive_joint_acceleration_from_velocity,
                dim=(self._root_view.count, self._root_view.joint_dof_count),
                inputs=[
                    self._sim_bind_joint_vel,
                    self._previous_joint_vel,
                    NewtonManager.get_dt(),
                    self._joint_acc.data,
                ],
            )
            self._joint_acc.timestamp = self._sim_timestamp
        return self._joint_acc.data

    def _aggregate_joint_pos_limits(self) -> wp.array:
        """Aggregate the joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        out = wp.zeros((self._root_view.count, self._root_view.joint_dof_count), dtype=wp.vec2f, device=self.device)
        wp.launch(
            make_joint_pos_limits_from_lower_and_upper_limits,
            dim=(self._root_view.count, self._root_view.joint_dof_count),
            inputs=[
                self._sim_bind_joint_pos_limits_lower,
                self._sim_bind_joint_pos_limits_upper,
                out,
            ],
        )
        return out

    ###
    # Internal root property calls.
    ###

    def _compute_root_link_vel_w(self) -> wp.array:
        """Compute the root link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                project_com_velocity_to_link_frame_root,
                dim=(self._root_view.count),
                device=self.device,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._root_link_vel_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    def _compute_root_com_pose_w(self) -> wp.array:
        """Compute the root center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's center of mass frame relative to the world.

        .. note:: The quaternion is in the form of [x, y, z, w].

        Returns:
            The root center of mass pose in simulation world frame. Shape is (num_instances,).
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial_root,
                dim=(self._root_view.count),
                device=self.device,
                inputs=[
                    self._sim_bind_root_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._root_com_pose_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    def _compute_projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3).
        
        Returns:
            The projection of the gravity direction on base frame. Shape is (num_instances, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_vec_from_pose_single,
                dim=self._root_view.count,
                inputs=[
                    self.GRAVITY_VEC_W,
                    self._sim_bind_root_link_pose_w,
                    self._projected_gravity_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    def _compute_heading_w(self) -> wp.array:
        r"""Yaw heading of the base frame (in radians). Shape is (num_instances,).

        .. note:: This quantity is computed by assuming that the forward-direction of the base frame is along
            x-direction, i.e. :math:`(1, 0, 0)`.
        
        Returns:
            The yaw heading of the base frame in radians. Shape is (num_instances,).
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_heading,
                dim=self._root_view.count,
                inputs=[
                    self.FORWARD_VEC_B,
                    self._sim_bind_root_link_pose_w,
                    self._heading_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data

    def _compute_root_link_vel_b(self) -> wp.array:
        """Root link velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].

        Returns:
            The root link velocity in base frame. Shape is (num_instances,).
        """
        if self._root_link_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_view.count,
                inputs=[
                    self._compute_root_link_vel_w(),
                    self._sim_bind_root_link_pose_w,
                    self._root_link_vel_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_link_vel_b.timestamp = self._sim_timestamp
        return self._root_link_vel_b.data

    def _compute_root_com_vel_b(self) -> wp.array:
        """Root center of mass velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].

        Returns:
            The root center of mass velocity in base frame. Shape is (num_instances,).
        """
        if self._root_com_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_view.count,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._root_com_vel_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_com_vel_b.timestamp = self._sim_timestamp
        return self._root_com_vel_b.data


    ###
    # Internal body property calls.
    ###

    def _compute_body_link_vel_w(self) -> wp.array:
        """Body link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # Project the velocity from the center of mass frame to the link frame
            wp.launch(
                project_com_velocity_to_link_frame_batch,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._sim_bind_body_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._body_link_vel_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.timestamp = self._sim_timestamp
        return self._body_link_vel_w.data

    def _compute_body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # Apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial_batch,
                dim=(self._root_view.count, self._root_view.link_count),
                device=self.device,
                inputs=[
                    self._sim_bind_body_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._body_com_pose_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.timestamp = self._sim_timestamp
        return self._body_com_pose_w.data

    def _compute_body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). All values are relative to the world.

        .. note:: The acceleration is in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            wp.launch(
                derive_body_acceleration_from_velocity_batched,
                dim=(self._root_view.count, self._root_view.link_count),
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    self._previous_body_com_vel,
                    NewtonManager.get_dt(),
                    self._body_com_acc_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_com_acc_w.timestamp = self._sim_timestamp
        return self._body_com_acc_w.data

    @warn_overhead_cost(
        "body_com_pos_b",
        "This function outputs the pose of the CoM, containing both position and orientation. However, in Newton, the"
        " CoM is always aligned with the link frame. This means that the quaternion is always [0, 0, 0, 1]. Consider"
        " using the position only instead.",
    )
    def _compute_body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        out = wp.zeros(
            (self._root_view.count, self._root_view.link_count), dtype=wp.transformf, device=self.device
        )
        wp.launch(
            generate_pose_from_position_with_unit_quaternion_batched,
            dim=(self._root_view.count, self._root_view.link_count),
            inputs=[
                self._sim_bind_body_com_pos_b,
                out,
            ],
        )
        return out

    ###
    # Internal helper kernel launches.
    ###

    @warn_overhead_cost("N/A", "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays directly instead.")
    def _merge_pose_and_velocity_to_state(self, transform: wp.array(dtype=wp.transformf), velocity: wp.array(dtype=wp.spatial_vectorf)) -> wp.array(dtype=vec13f):
        """Merges a pose array and a velocity array into a state array.

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, 7) is actually an array of shape (num_instances) of type wp.transformf. With
            wp.transformf being an array of 7 float32.

        ..note:: the state is in the form of [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz].

        Args:
            pose: The pose array. Shape is (num_instances, 7).
            velocity: The velocity array. Shape is (num_instances, 6).

        Returns:
            The state array. Shape is (num_instances, 13).
        """
        state = wp.zeros((self._root_view.count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_view.count,),
            device=self.device,
            inputs=[
                transform,
                velocity,
                state,
            ],
        )
        return state

    @warn_overhead_cost("N/A", "Launches a kernel to split the transform array to a position array. Consider using the transform array directly instead.")
    def _split_transform_to_position(self, transform: wp.array(dtype=wp.transformf)) -> wp.array(dtype=wp.vec3f):
        """Split the transform array to a position array.

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, 7) is actually an array of shape (num_instances) of type wp.transformf. With
            wp.transformf being an array of 7 float32.

        Args:
            transform: The transform array. Shape is (num_instances, 7).

        Returns:
            The position array. Shape is (num_instances, 3).
        """

        out = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)
        wp.launch(
            split_transform_array_to_position_array,
            dim=self._root_view.count,
            inputs=[
                transform,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to split the transform array to a quaternion array. Consider using the transform array directly instead.")
    def _split_transform_to_quaternion(self, transform: wp.array(dtype=wp.transformf)) -> wp.array(dtype=wp.quatf):
        """Split the transform array to a quaternion array.

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, 7) is actually an array of shape (num_instances) of type wp.transformf. With
            wp.transformf being an array of 7 float32.

        ..note:: the quaternion is in the form of [x, y, z, w].

        Args:
            transform: The transform array. Shape is (num_instances, 7).

        Returns:
            The quaternion array. Shape is (num_instances, 4).
        """
        out = wp.zeros((self._root_view.count,), dtype=wp.quatf, device=self.device)
        wp.launch(
            split_transform_array_to_quaternion_array,
            dim=self._root_view.count,
            inputs=[
                transform,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to split the velocity array to a linear velocity array. Consider using the velocity array directly instead.")
    def _split_velocity_to_linear_velocity(self, velocity: wp.array(dtype=wp.spatial_vectorf)) -> wp.array(dtype=wp.vec3f):
        """Split the velocity array to a linear velocity array..

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, 6) is actually an array of shape (num_instances) of type wp.spatial_vectorf. With
            wp.spatial_vectorf being an array of 6 float32.

        ..note:: the velocity is in the form of [vx, vy, vz, wx, wy, wz].

        Args:
            velocity: The velocity array. Shape is (num_instances, 6).

        Returns:
            The linear velocity array. Shape is (num_instances, 3).
        """
        out = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)
        wp.launch(
            split_spatial_vectory_array_to_linear_velocity_array,
            dim=self._root_view.count,
            inputs=[
                velocity,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to split the velocity array to an angular velocity array. Consider using the velocity array directly instead.")
    def _split_velocity_to_angular_velocity(self, velocity: wp.array(dtype=wp.spatial_vectorf)) -> wp.array(dtype=wp.vec3f):
        """Split the velocity array to an angular velocity array.

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, 6) is actually an array of shape (num_instances) of type wp.spatial_vectorf. With
            wp.spatial_vectorf being an array of 6 float32.

        ..note:: the velocity is in the form of [vx, vy, vz, wx, wy, wz].

        Args:
            velocity: The velocity array. Shape is (num_instances, 6).

        Returns:
            The angular velocity array. Shape is (num_instances, 3).
        """
        out = wp.zeros((self._root_view.count,), dtype=wp.vec3f, device=self.device)
        wp.launch(
            split_spatial_vectory_array_to_angular_velocity_array,
            dim=self._root_view.count,
            inputs=[
                velocity,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to merge a pose and a velocity into a state. Consider using the pose and velocity arrays directly instead.")
    def _merge_pose_and_velocity_to_state_batched(self, transform: wp.array(dtype=wp.transformf), velocity: wp.array(dtype=wp.spatial_vectorf)) -> wp.array(dtype=vec13f):
        """Merges a 2D pose array and a 2D velocity array into a 2D state array.

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, num_bodies, 7) is actually an array of shape (num_instances, num_bodies) of type
            wp.transformf. With wp.transformf being an array of 7 float32.

        ..note:: the state is in the form of [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz].

        Args:
            pose: The pose array. Shape is (num_instances, 7).
            velocity: The velocity array. Shape is (num_instances, 6).

        Returns:
            The state array. Shape is (num_instances, 13).
        """
        state = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_view.count, self._root_view.link_count),
            device=self.device,
            inputs=[
                transform,
                velocity,
                state,
            ],
        )
        return state

    @warn_overhead_cost("N/A", "Launches a kernel to split the transform array to a position array. Consider using the transform array directly instead.")
    def _split_transform_to_position_batched(self, transform: wp.array(dtype=wp.transformf)) -> wp.array(dtype=wp.vec3f):
        """Splits a 2D transform array to a 2D position array.

        ..note:: the shapes are given has float32 shapes, but the actual shapes are not including the last dimension.
            i.e. (num_instances, num_bodies, 7) is actually an array of shape (num_instances, num_bodies) of type
            wp.transformf. With wp.transformf being an array of 7 float32.

        Args:
            transform: The transform array. Shape is (num_instances, num_bodies, 7).

        Returns:
            The position array. Shape is (num_instances, num_bodies, 3).
        """

        out = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            split_transform_batched_array_to_position_batched_array,
            dim=(self._root_view.count, self._root_view.link_count),
            inputs=[
                transform,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to split the transform array to a quaternion array. Consider using the transform array directly instead.")
    def _split_transform_to_quaternion_batched(self, transform: wp.array(dtype=wp.transformf)) -> wp.array(dtype=wp.quatf):
        """Splits a 2D transform array to a 2D quaternion array.

        ..note:: the shapes are given has float32 shape, but the actual shapes are not including the last dimension.
            i.e. (num_instances, num_bodies, 7) is actually an array of shape (num_instances, num_bodies) of type
            wp.transformf. With wp.transformf being an array of 7 float32.

        ..note:: the quaternion is in the form of [x, y, z, w].

        Args:
            transform: The transform array. Shape is (num_instances, num_bodies, 7).

        Returns:
            The quaternion array. Shape is (num_instances, num_bodies, 4).
        """
        out = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=wp.quatf, device=self.device)
        wp.launch(
            split_transform_batched_array_to_quaternion_batched_array,
            dim=(self._root_view.count, self._root_view.link_count),    
            inputs=[
                transform,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to split the velocity array to a linear velocity array. Consider using the velocity array directly instead.")
    def _split_velocity_to_linear_velocity_batched(self, velocity: wp.array(dtype=wp.spatial_vectorf)) -> wp.array(dtype=wp.vec3f):
        """Splits a 2D velocity array to a 2D linear velocity array.

        ..note:: the shapes are given has float32 shape, but the actual shapes are not including the last dimension.
            i.e. (num_instances, num_bodies, 6) is actually an array of shape (num_instances, num_bodies) of type
            wp.spatial_vectorf. With wp.spatial_vectorf being an array of 6 float32.
        
        ..note:: the velocity is in the form of [vx, vy, vz, wx, wy, wz].

        Args:
            velocity: The velocity array. Shape is (num_instances, num_bodies, 6).

        Returns:
            The linear velocity array. Shape is (num_instances, num_bodies, 3).
        """
        out = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            split_spatial_vectory_batched_array_to_linear_velocity_batched_array,
            dim=(self._root_view.count, self._root_view.link_count),
            inputs=[
                velocity,
                out,
            ],
        )
        return out

    @warn_overhead_cost("N/A", "Launches a kernel to split the velocity array to an angular velocity array. Consider using the velocity array directly instead.")
    def _split_velocity_to_angular_velocity_batched(self, velocity: wp.array(dtype=wp.spatial_vectorf)) -> wp.array(dtype=wp.vec3f):
        """Splits a 2D velocity array to a 2D angular velocity array.

        ..note:: the shapes are given has float32 shape, but the actual shapes are not including the last dimension.
            i.e. (num_instances, num_bodies, 6) is actually an array of shape (num_instances, num_bodies) of type
            wp.spatial_vectorf. With wp.spatial_vectorf being an array of 6 float32.
        
        ..note:: the velocity is in the form of [vx, vy, vz, wx, wy, wz].

        Args:
            velocity: The velocity array. Shape is (num_instances, num_bodies, 6).

        Returns:
            The angular velocity array. Shape is (num_instances, num_bodies, 3).
        """
        out = wp.zeros((self._root_view.count, self._root_view.link_count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            split_spatial_vectory_batched_array_to_angular_velocity_batched_array,
            dim=(self._root_view.count, self._root_view.link_count),
            inputs=[
                velocity,
                out,
            ],
        )
        return out