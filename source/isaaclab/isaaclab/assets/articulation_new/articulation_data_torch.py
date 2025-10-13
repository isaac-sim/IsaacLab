# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import warp as wp
import torch

from isaaclab.assets.core.root_properties.root_data import RootData
from isaaclab.assets.core.body_properties.body_data import BodyData
from isaaclab.assets.core.joint_properties.joint_data import JointData
from isaaclab.assets.core.kernels import vec13f, combine_pose_and_velocity_to_state_batched
from isaaclab.utils.helpers import deprecated


class ArticulationDataWarp:
    def __init__(self, root_newton_view, device: str):
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_newton_view = weakref.proxy(root_newton_view)

        # Initialize the data containers
        self._root_data = RootData(root_newton_view, device)
        self._body_data = BodyData(root_newton_view, device)
        self._joint_data = JointData(root_newton_view, device)

    @property
    def joint_data(self) -> JointData:
        return self._joint_data
    
    @property
    def body_data(self) -> BodyData:
        return self._body_data
    
    @property
    def root_data(self) -> RootData:
        return self._root_data

    def update(self, dt: float):
        self._root_data.update(dt)
        self._body_data.update(dt)
        self._joint_data.update(dt)

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
    def default_root_pose(self) -> torch.Tensor:
        """Default root pose ``[pos, quat]`` in the local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the articulation root's actor frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return wp.to_torch(self._root_data.default_root_pose)

    @property
    def default_root_vel(self) -> torch.Tensor:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the articulation root's center of mass frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return wp.to_torch(self._root_data.default_root_vel)

    @property
    def default_joint_pos(self) -> torch.Tensor:
        """Default joint positions of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return wp.to_torch(self._joint_data.default_joint_pos)

    @property
    def default_joint_vel(self) -> torch.Tensor:
        """Default joint velocities of all joints. Shape is (num_instances, num_joints).

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return wp.to_torch(self._joint_data.default_joint_vel)

    ##
    # Joint commands -- From the user to the actuator model.
    ##

    @property
    def joint_target(self) -> torch.Tensor:
        """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return wp.to_torch(self._joint_data.joint_target)

    @property
    def joint_effort_target(self) -> torch.Tensor:
        """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

        For an implicit actuator model, the targets are directly set into the simulation.
        For an explicit actuator model, the targets are used to compute the joint efforts (see :attr:`applied_torque`),
        which are then set into the simulation.
        """
        return wp.to_torch(self._joint_data.joint_effort_target)

    ##
    # Joint commands -- Explicit actuators.
    ##

    @property
    def computed_effort(self) -> torch.Tensor:
        """Joint efforts computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

        This quantity is the raw effort output from the actuator mode, before any clipping is applied.
        It is exposed for users who want to inspect the computations inside the actuator model.
        For instance, to penalize the learning agent for a difference between the computed and applied torques.
        """
        return wp.to_torch(self._joint_data.computed_effort)

    @property
    def applied_effort(self) -> torch.Tensor:
        """Joint efforts applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

        These efforts are set into the simulation, after clipping the :attr:`computed_effort` based on the
        actuator model.
        """
        return wp.to_torch(self._joint_data.applied_effort)

    @property
    def joint_stiffness(self) -> torch.Tensor:
        """Joint stiffness. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_stiffness)

    @property
    def joint_damping(self) -> torch.Tensor:
        """Joint damping. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_damping)

    @property
    def joint_control_mode(self) -> torch.Tensor:
        """Joint control mode. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_control_mode)

    ###
    # Joint commands. (Directly binded to the simulation)
    ###

    @property
    def joint_target_sim(self) -> torch.Tensor:
        """Joint target. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_target_sim)

    @property
    def joint_effort_sim(self) -> torch.Tensor:
        """Joint effort. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_effort_sim)

    ##
    # Joint properties. (Directly binded to the simulation)
    ##

    @property
    def joint_control_mode_sim(self) -> torch.Tensor:
        """Joint control mode. Shape is (num_instances, num_joints).

        When using implicit actuator models Newton needs to know how the joints are controlled.
        The control mode can be one of the following:

        * None: 0
        * Position control: 1
        * Velocity control: 2

        This quantity is set by the :meth:`Articulation.write_joint_control_mode_to_sim` method.
        """
        return wp.to_torch(self._joint_data.joint_control_mode_sim)

    @property
    def joint_stiffness_sim(self) -> torch.Tensor:
        """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return wp.to_torch(self._joint_data.joint_stiffness_sim)

    @property
    def joint_damping_sim(self) -> torch.Tensor:
        """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

        In the case of explicit actuators, the value for the corresponding joints is zero.
        """
        return wp.to_torch(self._joint_data.joint_damping_sim)

    @property
    def joint_armature(self) -> torch.Tensor:
        """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_armature)

    @property
    def joint_friction_coeff(self) -> torch.Tensor:
        """Joint friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_friction_coeff)

    @property
    def joint_pos_limits_lower(self) -> torch.Tensor:
        """Joint position limits lower provided to the simulation. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_pos_limits_lower)

    @property
    def joint_pos_limits_upper(self) -> torch.Tensor:
        """Joint position limits upper provided to the simulation. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_pos_limits_upper)


    @property
    def joint_vel_limits(self) -> torch.Tensor:
        """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_vel_limits)

    @property
    def joint_effort_limits(self) -> torch.Tensor:
        """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_effort_limits)

    ##
    # Joint states
    ##

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_pos)

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_vel)

    ##
    # Joint properties - Custom.
    ##

    @property
    def joint_dynamic_friction(self) -> torch.Tensor:
        """Joint dynamic friction. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_dynamic_friction)

    @property
    def joint_viscous_friction(self) -> torch.Tensor:
        """Joint viscous friction. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_viscous_friction)

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
        return wp.to_torch(self._joint_data.soft_joint_pos_limits)

    @property
    def soft_joint_vel_limits(self) -> torch.Tensor:
        """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

        These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
        has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
        """
        return wp.to_torch(self._joint_data.soft_joint_vel_limits)

    @property
    def gear_ratio(self) -> torch.Tensor:
        """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.gear_ratio)

    ##
    # Root state properties.
    ##

    @property
    def root_mass(self) -> torch.Tensor:
        """Root mass ``wp.float32`` in the world frame. Shape is (num_instances,)."""
        return wp.to_torch(self._root_data.root_mass)

    @property
    def root_inertia(self) -> torch.Tensor:
        """Root inertia ``wp.mat33`` in the world frame. Shape is (num_instances, 9)."""
        return wp.to_torch(self._root_data.root_inertia)

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return wp.to_torch(self._root_data.root_link_pose_w)

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """

        return wp.to_torch(self._root_data.root_link_vel_w)

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """

        return wp.to_torch(self._root_data.root_com_pose_w)

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        return wp.to_torch(self._root_data.root_com_vel_w)

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """
        return wp.to_torch(self._root_data.root_state_w)

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root link state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's actor frame.
        """
        return wp.to_torch(self._root_data.root_link_state_w)

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root center of mass state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's center of mass frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """
        return wp.to_torch(self._root_data.root_com_state_w)
    
    @property
    def root_com_pose_b(self) -> torch.Tensor:
        """Root center of mass pose ``wp.transformf`` in base frame. Shape is (num_instances,).

        This quantity is the pose of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return wp.to_torch(self._root_data.root_com_pose_b)

    @property
    def root_com_pos_b(self) -> torch.Tensor:
        """Root center of mass position ``wp.vec3f`` in base frame. Shape is (num_instances, 3).

        This quantity is the position of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return wp.to_torch(self._root_data.root_com_pos_b)
    
    @property
    def root_com_quat_b(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in base frame. Shape is (num_instances, 4).

        This quantity is the orientation of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return wp.to_torch(self._root_data.root_com_quat_b)

    ##
    # Body state properties.
    ##

    @property
    def body_mass(self) -> torch.Tensor:
        """Body mass ``wp.float32`` in the world frame. Shape is (num_instances, num_bodies)."""
        return wp.to_torch(self._body_data.body_mass)

    @property
    def body_inertia(self) -> torch.Tensor:
        """Body inertia ``wp.mat33`` in the world frame. Shape is (num_instances, num_bodies, 9)."""
        return wp.to_torch(self._body_data.body_inertia)

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return wp.to_torch(self._body_data.body_link_pose_w)

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        return wp.to_torch(self._body_data.body_link_vel_w)

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return wp.to_torch(self._body_data.body_com_pose_w)
    
    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        return wp.to_torch(self._body_data.body_com_vel_w)

    @property
    def body_state_w(self) -> torch.Tensor:
        """State of all bodies ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation links' actor frame relative to the world.
        The velocity is of the articulation links' center of mass frame.
        """

        return wp.to_torch(self._body_data.body_state_w)

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies' link frame ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """

        return wp.to_torch(self._body_data.body_link_state_w)

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies center of mass ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """

        return wp.to_torch(self._body_data.body_com_state_w)

    @property
    def body_com_acc_w(self) -> torch.Tensor:
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The acceleration is in the form of [wx, wy, wz, vx, vy, vz].
        All values are relative to the world.
        """
        return wp.to_torch(self._body_data.body_com_acc_w)
    
    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position ``wp.vec3f`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The position is in the form of [x, y, z].
        This quantity is the position of the center of mass frame of the rigid body relative to the body's link frame.
        """
        return wp.to_torch(self._body_data.body_com_pos_b)

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        return wp.to_torch(self._body_data.body_com_pose_b)

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force>`__
        and the underlying `PhysX Tensor API <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force>`__ .
        """
        return wp.to_torch(self._joint_data.body_incoming_joint_wrench_b)

    ##
    # Joint state properties.
    ##

    @property
    def joint_acc(self) -> torch.Tensor:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        return wp.to_torch(self._joint_data.joint_acc)
    ##
    # Derived Properties.
    ##

    # FIXME: USE SIM_BIND_LINK_POSE_W RATHER THAN JUST THE QUATERNION
    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return wp.to_torch(self._root_data.projected_gravity_b)

    # FIXME: USE SIM_BIND_LINK_POSE_W RATHER THAN JUST THE QUATERNION
    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        return wp.to_torch(self._root_data.heading_w)

    @property
    def root_link_vel_b(self) -> torch.Tensor:
        """Root link velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [wx, wy, wz, vx, vy, vz].
        """
        return wp.to_torch(self._root_data.root_link_vel_b)

    @property
    def root_com_vel_b(self) -> torch.Tensor:
        """Root center of mass velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [wx, wy, wz, vx, vy, vz].
        """
        return wp.to_torch(self._root_data.root_com_vel_b)

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return wp.to_torch(self._root_data.root_link_pos_w)

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return wp.to_torch(self._root_data.root_link_quat_w)

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return wp.to_torch(self._root_data.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return wp.to_torch(self._root_data.root_link_ang_vel_w)

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return wp.to_torch(self._root_data.root_com_pos_w)

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the root rigid body's center of mass frame.
        """
        return wp.to_torch(self._root_data.root_com_quat_w)

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances,).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return wp.to_torch(self._root_data.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return wp.to_torch(self._root_data.root_com_ang_vel_w)

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return wp.to_torch(self._body_data.body_link_pos_w)

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        return wp.to_torch(self._body_data.body_link_quat_w)

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return wp.to_torch(self._body_data.body_link_lin_vel_w)

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return wp.to_torch(self._body_data.body_link_ang_vel_w)

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return wp.to_torch(self._body_data.body_com_pos_w)

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the articulation bodies' actor frame.
        """
        return wp.to_torch(self._body_data.body_com_quat_w)

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return wp.to_torch(self._body_data.body_com_lin_vel_w)

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return wp.to_torch(self._body_data.body_com_ang_vel_w)

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return wp.to_torch(self._body_data.body_com_lin_acc_w)

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return wp.to_torch(self._body_data.body_com_ang_acc_w)

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return wp.to_torch(self._body_data.body_com_quat_b)

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity ``wp.vec3f`` in base frame. Shape is (num_instances).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return wp.to_torch(self._root_data.root_link_lin_vel_b)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity ``wp.vec3f`` in base world frame. Shape is (num_instances).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return wp.to_torch(self._root_data.root_link_ang_vel_b)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity ``wp.vec3f`` in base frame. Shape is (num_instances).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return wp.to_torch(self._root_data.root_com_lin_vel_b)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return wp.to_torch(self._root_data.root_com_ang_vel_b)

    @property
    def joint_pos_limits(self) -> torch.Tensor:
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        return wp.to_torch(self._joint_data.joint_pos_limits)

    ##
    # Backward compatibility. Need to nuke these properties in a future release.
    ##

    @property
    @deprecated("default_root_pose")
    def default_root_state(self) -> torch.Tensor:
        """Same as :attr:`default_root_pose`."""
        state = wp.zeros((self._root_newton_view.count), dtype=vec13f, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_newton_view.count),
            device=self.device,
            inputs=[
                self.default_root_pose,
                self.default_root_vel,
                state,
            ],
        )
        return wp.to_torch(state)

    @property
    @deprecated("root_link_pose_w")
    def root_pose_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        return wp.to_torch(self.root_link_pose_w)

    @property
    @deprecated("root_link_pos_w")
    def root_pos_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        return wp.to_torch(self.root_link_pos_w)

    @property
    @deprecated("root_link_quat_w")
    def root_quat_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        return wp.to_torch(self.root_link_quat_w)

    @property
    @deprecated("root_com_vel_w")
    def root_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        return wp.to_torch(self.root_com_vel_w)

    @property
    @deprecated("root_com_lin_vel_w")
    def root_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        return wp.to_torch(self.root_com_lin_vel_w)

    @property
    @deprecated("root_com_ang_vel_w")
    def root_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        return wp.to_torch(self.root_com_ang_vel_w)

    @property
    @deprecated("root_com_lin_vel_b")
    def root_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        return wp.to_torch(self.root_com_lin_vel_b)

    @property
    @deprecated("root_com_ang_vel_b")
    def root_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        return wp.to_torch(self.root_com_ang_vel_b)

    @property
    @deprecated("body_link_pose_w")
    def body_pose_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        return wp.to_torch(self.body_link_pose_w)

    @property
    @deprecated("body_link_pos_w")
    def body_pos_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        return wp.to_torch(self.body_link_pos_w)

    @property
    @deprecated("body_link_quat_w")
    def body_quat_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        return wp.to_torch(self.body_link_quat_w)

    @property
    @deprecated("body_com_vel_w")
    def body_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        return wp.to_torch(self.body_com_vel_w)

    @property
    @deprecated("body_com_lin_vel_w")
    def body_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        return wp.to_torch(self.body_com_lin_vel_w)

    @property
    @deprecated("body_com_ang_vel_w")
    def body_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        return wp.to_torch(self.body_com_ang_vel_w)

    @property
    @deprecated("body_com_acc_w")
    def body_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        return wp.to_torch(self.body_com_acc_w)

    @property
    @deprecated("body_com_lin_acc_w")
    def body_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        return wp.to_torch(self.body_com_lin_acc_w)

    @property
    @deprecated("body_com_ang_acc_w")
    def body_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        return wp.to_torch(self.body_com_ang_acc_w)

    @property
    @deprecated("body_com_pos_b")
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        return wp.to_torch(self.body_com_pos_b)

    @property
    @deprecated("body_com_quat_b")
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        return wp.to_torch(self.body_com_quat_b)

    @property
    @deprecated("joint_pos_limits")
    def joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        return wp.to_torch(self.joint_pos_limits)

    @property
    @deprecated("joint_friction_coeff")
    def joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        return wp.to_torch(self.joint_friction_coeff)
