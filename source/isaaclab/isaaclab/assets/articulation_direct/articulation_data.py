# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import weakref

import omni.log
import warp as wp

from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.buffers import TimestampedWarpBuffer

from .kernels import *

def deprecate(*args, **kwargs):
    def wrapper(func):
        if kwargs.get("replacement") is not None:
            omni.log.warn(f"DeprecationWarning: {func.__name__} is deprecated and will be removed in a future version. Use {kwargs.get('replacement')} instead.")
        else:
            omni.log.warn(f"DeprecationWarning: {func.__name__} is deprecated and will be removed in a future version.")
        return func(*args, **kwargs)
    return wrapper


def warn_overhead_cost(*args, **kwargs):
    def wrapper(func):
        omni.log.warn(f"OverheadWarning: {func.__name__} is expensive and should be avoided. Instead of getting sliced data, use the whole data in the target kernel.")
        return func(*args, **kwargs)
    return wrapper


# FIXME: Need to create one for each articulation view.
class NewtonAutoMapper:
    NewtonArticulationView = None

    @staticmethod
    def cast_to_torch(data):
        return wp.to_torch(data)

    @staticmethod
    def cast_to_wp(data, dtype: wp.dtype | None = None):
        if dtype is not None:
            return wp.from_torch(data, dtype=dtype)
        else:
            return wp.from_torch(data)

    def __init__(self, default, dtype):
        self._default = default
        self._dtype = dtype

    @classmethod
    def set_articulation_view(cls, articulation_view):
        cls.NewtonArticulationView = articulation_view

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, type):
        return getattr(self.NewtonArticulationView, self._name, self._default)

    def __set__(self, value):
        setattr(self.NewtonArticulationView, self._name, value)
        

class ArticulationData:
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

    def __init__(self, root_newton_view, device: str):
        """Initializes the articulation data.

        Args:
            root_newton_view: The root articulation view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_newton_view = weakref.proxy(root_newton_view)
        NewtonAutoMapper.set_articulation_view(self._root_newton_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # obtain global simulation view
        gravity = NewtonManager.get_model().gravity
        gravity_dir = [float(i) / sum(gravity) for i in gravity]
        # Initialize constants
        self.GRAVITY_VEC_W = wp.vec3f(gravity_dir[0], gravity_dir[1], gravity_dir[2])
        self.FORWARD_VEC_B = wp.vec3f((1.0, 0.0, 0.0))

        # Initialize history for finite differencing
        self._previous_body_com_vel = self._root_newton_view.get_link_velocities(NewtonManager.get_state_0()).clone()
        self._previous_joint_vel = self._root_newton_view.get_dof_velocities(NewtonManager.get_state_0()).clone()

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        #self._root_link_pose_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.transformf)
        self._root_link_vel_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        self._root_link_vel_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        #self._body_link_pose_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_bodies), dtype=wp.transformf)
        self._body_link_vel_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_bodies), dtype=wp.spatial_vectorf)
        self._projected_gravity_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count, 3), dtype=wp.vec3f)
        self._heading_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.float32)
        # -- com frame w.r.t. link frame
        #self._body_com_position_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_bodies), dtype=wp.vec3f)
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.transformf)
        #self._root_com_vel_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        self._root_com_vel_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        self._body_com_pose_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_bodies), dtype=wp.transformf)
        #self._body_com_vel_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_bodies), dtype=wp.spatial_vectorf)
        self._body_com_acc_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_bodies), dtype=wp.spatial_vectorf)
        # -- joint state
        self._joint_pos = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_joints), dtype=wp.float32)
        self._joint_vel = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_joints), dtype=wp.float32)
        self._joint_acc = TimestampedWarpBuffer(shape=(self._root_newton_view.count, self._root_newton_view.num_joints), dtype=wp.float32)
        #self._body_incoming_joint_wrench_b = TimestampedWarpBuffer()

    def get_from_newton(self, name: str, container):
        return self._root_newton_view.get_attribute(name, container)

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc
        self.body_com_acc_w

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
    # Defaults - Initial state.
    ##

    default_root_pose: wp.array = None
    """Default root pose ``[pos, quat]`` in the local environment frame. Shape is (num_instances, 7).

    The position and quaternion are of the articulation root's actor frame.

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_root_vel: wp.array = None
    """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 6).
    
    The linear and angular velocities are of the articulation root's center of mass frame.
    
    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_pos: wp.array = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """

    default_joint_vel: wp.array = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    ##
    # Root state properties. (Directly binded to the simulation)
    ##

    root_link_pose_w: wp.array = None
    """Root link pose ``wp.transformf`` in the world frame. Shape is (num_instances,).
    
    The pose is in the form of [pos, quat]. The orientation is provided in (x, y, z, w) format.
    """

    root_com_vel_w: wp.array = None
    """Root center of mass velocity ``wp.spatial_vectorf`` in the world frame. Shape is (num_instances,).
    
    The velocity is in the form of [ang_vel, lin_vel].
    """

    ##
    # Body state properties. (Directly binded to the simulation)
    ##

    body_link_pose_w: wp.array = None
    """Body link pose ``wp.transformf`` in the world frame. Shape is (num_instances, num_bodies).
    
    The pose is in the form of [pos, quat]. The orientation is provided in (x, y, z, w) format.
    """

    body_com_vel_w: wp.array = None
    """Body center of mass velocity ``wp.spatial_vectorf`` in the world frame. Shape is (num_instances, num_bodies).
    
    The velocity is in the form of [ang_vel, lin_vel].
    """
    
    body_com_pos_b: wp.array = None
    """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

    Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
    This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
    The orientation is provided in (x, y, z, w) format.
    """

    ##
    # Joint commands -- Set into simulation.
    ##

    joint_target: wp.array = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    joint_effort_target: wp.array = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """

    ##
    # Joint commands -- Explicit actuators.
    ##

    computed_torque: wp.array = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.
    """

    applied_torque: wp.array = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """

    ##
    # Joint properties. (Directly binded to the simulation)
    ##

    joint_control_mode: wp.array = None
    """Joint control mode. Shape is (num_instances, num_joints).

    When using implicit actuator models Newton needs to know how the joints are controlled.
    The control mode can be one of the following:

    * None: 0
    * Position control: 1
    * Velocity control: 2

    This quantity is set by the :meth:`Articulation.write_joint_control_mode_to_sim` method.
    """

    joint_stiffness: wp.array = None
    """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_damping: wp.array = None
    """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """

    joint_armature: wp.array = None
    """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_friction_coeff: wp.array = None
    """Joint friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_pos_limits_lower: wp.array = None
    """Joint position limits lower provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_pos_limits_upper: wp.array = None
    """Joint position limits upper provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_vel_limits: wp.array = None
    """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""

    joint_effort_limits: wp.array = None
    """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""

    ##
    # Joint properties - Custom.
    ##

    soft_joint_pos_limits: wp.array = None
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

    soft_joint_vel_limits: wp.array = None
    """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

    These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
    has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
    """

    gear_ratio: wp.array = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: wp.array = None
    """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_damping: wp.array = None
    """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_limit_stiffness: wp.array = None
    """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_rest_length: wp.array = None
    """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_offset: wp.array = None
    """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""

    fixed_tendon_pos_limits: wp.array = None
    """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""

    ##
    # Spatial tendon properties.
    ##

    spatial_tendon_stiffness: wp.array = None
    """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_damping: wp.array = None
    """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_limit_stiffness: wp.array = None
    """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    spatial_tendon_offset: wp.array = None
    """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        
        return self._root_newton_view.get_root_transforms(NewtonManager.get_state_0())

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                project_com_velocity_to_link_frame_batch,
                dim=(self._root_newton_view.count),
                device=self.device,
                inputs=[
                    self.root_com_vel_w,
                    self.body_link_pose_w,
                    self.body_com_pos_b,
                    self._root_link_vel_w.data,
                ]
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial,
                dim=(self._root_newton_view.count),
                device=self.device,
                inputs=[
                    self.root_link_pose_w,
                    self.body_com_pos_b,
                    self._root_com_pose_w.data,
                ]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """

        return self._root_newton_view.get_root_velocities(NewtonManager.get_state_0())

    @property
    def root_state_w(self) -> tuple[wp.array, wp.array]:
        """Root state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """

        return self.root_link_pose_w, self.root_com_vel_w

    @property
    def root_link_state_w(self) -> tuple[wp.array, wp.array]:
        """Root link state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's actor frame.
        """

        return self.root_link_pose_w, self.root_link_vel_w

    @property
    def root_com_state_w(self) -> tuple[wp.array, wp.array]:
        """Root center of mass state ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances,), (num_instances,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation root's center of mass frame relative to the world.
        The velocity is of the articulation root's center of mass frame.
        """

        return self.root_com_pose_w, self.root_com_vel_w

    ##
    # Body state properties.
    ##

    @property
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # Project the velocity from the center of mass frame to the link frame
            wp.launch(
                project_com_velocity_to_link_frame_batch,
                dim=(self._root_newton_view.count, self._root_newton_view.num_bodies),
                device=self.device,
                inputs=[
                    self.body_com_vel_w,
                    self.body_link_pose_w,
                    self.body_com_pos_b,
                    self._body_link_vel_w.data,
                ]
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.timestamp = self._sim_timestamp
        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # Apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial_batch,
                dim=(self._root_newton_view.count, self._root_newton_view.num_bodies),
                device=self.device,
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pos_b,
                    self._body_com_pose_w.data,
                ]
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.timestamp = self._sim_timestamp
        return self._body_com_pose_w.data

    @property
    def body_state_w(self) -> tuple[wp.array, wp.array]:
        """State of all bodies ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The pose is of the articulation links' actor frame relative to the world.
        The velocity is of the articulation links' center of mass frame.
        """

        return self.body_link_pose_w, self.body_com_vel_w

    @property
    def body_link_state_w(self) -> tuple[wp.array, wp.array]:
        """State of all bodies' link frame ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """

        return self.body_link_pose_w, self.body_link_vel_w

    @property
    def body_com_state_w(self) -> tuple[wp.array, wp.array]:
        """State of all bodies center of mass ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        The velocity is in the form of [wx, wy, wz, vx, vy, vz].

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """

        return self.body_com_pose_w, self.body_com_vel_w

    @property
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). The acceleration is in the form of [wx, wy, wz, vx, vy, vz].
        All values are relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            dt = self._sim_timestamp - self._body_com_vel_w.timestamp
            wp.launch(
                derive_body_acceleration_from_velocity,
                dim=(self._root_newton_view.count, self._root_newton_view.num_bodies),
                inputs=[
                    self.body_com_vel_w,
                    self._previous_body_com_vel,
                    dt,
                    self._body_com_acc_w.data,
                ]
            )
            # set the buffer data and timestamp
            self._body_com_acc_w.timestamp = self._sim_timestamp
        return self._body_com_acc_w.data

    @warn_overhead_cost
    @property
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.transformf, device=self.device)
        wp.launch(
            generate_body_com_pose_b,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_pos_b,
                out,
            ]
        )
        return out

    @property
    def body_incoming_joint_wrench_b(self) -> wp.array:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force>`__
        and the underlying `PhysX Tensor API <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force>`__ .
        """
        raise NotImplementedError("Body incoming joint wrench in body frame is not implemented for Newton.")
        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_physx_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.time_stamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self) -> wp.array:
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_newton_view.get_dof_positions(NewtonManager.get_state_0())
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data

    @property
    def joint_vel(self) -> wp.array:
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_newton_view.get_dof_velocities(NewtonManager.get_state_0())
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data

    @property
    def joint_acc(self) -> wp.array:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            dt = self._sim_timestamp - self._joint_acc.timestamp
            wp.launch(
                derive_joint_acceleration_from_velocity,
                dim=(self._root_newton_view.count, self._root_newton_view.num_joints),
                inputs=[
                    self._joint_vel.data,
                    self._previous_joint_vel,
                    dt,
                    self._joint_acc.data,
                ]
            )
            self._joint_acc.timestamp = self._sim_timestamp
        return self._joint_acc.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_vec_from_quat_single,
                dim=self._root_newton_view.count,
                inputs=[
                    self.GRAVITY_VEC_W,
                    self.root_link_quat_w,
                    self._projected_gravity_b.data,
                ]
            )
            # set the buffer data and timestamp
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_heading,
                dim=self._root_newton_view.count,
                inputs=[
                    self.FORWARD_VEC_B,
                    self.root_link_quat_w,
                    self._heading_w.data,
                ]
            )
            # set the buffer data and timestamp
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data


    @property
    def root_link_vel_b(self) -> wp.array:
        """Root link velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [wx, wy, wz, vx, vy, vz].
        """
        if self._root_link_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_newton_view.count,
                inputs=[
                    self.root_link_vel_w,
                    self.root_link_pose_w,
                    self._root_link_vel_b.data,
                ]
            )
            self._root_link_vel_b.timestamp = self._sim_timestamp
        return self._root_link_vel_b.data

    @property
    def root_com_vel_b(self) -> wp.array:
        """Root center of mass velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        Velocity is provided in the form of [wx, wy, wz, vx, vy, vz].
        """
        if self._root_com_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_newton_view.count,
                inputs=[
                    self.root_com_vel_w,
                    self.root_link_pose_w,
                    self._root_com_vel_b.data,
                ]
            )
            self._root_com_vel_b.timestamp = self._sim_timestamp
        return self._root_com_vel_b.data


    ##
    # Sliced properties.
    ##

    @warn_overhead_cost
    @property
    def root_link_pos_w(self) -> wp.array:
        """Root link position ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_position,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_link_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the actor frame of the root rigid body.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.quatf, device=self.device)
        wp.launch(
            get_quat,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_link_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_link_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_link_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_position,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_com_pose_w,
                out,
            ]
        )
        return out
    
    @warn_overhead_cost
    @property
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the root rigid body's center of mass frame.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.quatf, device=self.device)
        wp.launch(
            get_quat,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_com_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances,).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_com_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_com_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_position,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_link_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_link_quat_w(self) -> wp.array:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.quatf, device=self.device)
        wp.launch(
            get_quat,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_link_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_link_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_link_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_position,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_quat_w(self) -> wp.array:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        Format is ``(w, x, y, z)``.
        This quantity is the orientation of the articulation bodies' actor frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.quatf, device=self.device)
        wp.launch(
            get_quat,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_pose_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_vel_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_acc_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_acc_w,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.quatf, device=self.device)
        wp.launch(
            get_quat,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.body_com_pose_b,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity ``wp.vec3f`` in base frame. Shape is (num_instances).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_link_vel_b,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity ``wp.vec3f`` in base world frame. Shape is (num_instances).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_link_vel_b,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity ``wp.vec3f`` in base frame. Shape is (num_instances).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_linear_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_com_vel_b,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        out = wp.zeros((self._root_newton_view.count), dtype=wp.vec3f, device=self.device)
        wp.launch(
            get_angular_velocity,
            dim=self._root_newton_view.count,
            inputs=[
                self.root_com_vel_b,
                out,
            ]
        )
        return out

    @warn_overhead_cost
    @property
    def joint_pos_limits(self) -> wp.array:
        """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

        The limits are in the order :math:`[lower, upper]`.
        """
        out = wp.zeros((self._root_newton_view.count, self._root_newton_view.count), dtype=wp.vec2f, device=self.device)
        wp.launch(
            make_joint_pos_limits_from_lower_and_upper_limits,
            dim=(self._root_newton_view.count, self._root_newton_view.count),
            inputs=[
                self.joint_pos_limit_lower,
                self.joint_pos_limits_upper,
                out,
            ]
        )
        return out

    ##
    # Backward compatibility. Need to nuke these properties in a future release.
    ##

    @deprecate(replacement="root_link_pose_w")
    @property
    def root_pose_w(self) -> wp.array:
        """Same as :attr:`root_link_pose_w`."""
        return self.root_link_pose_w

    @deprecate(replacement="root_link_pos_w")
    @property
    def root_pos_w(self) -> wp.array:
        """Same as :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @deprecate(replacement="root_link_quat_w")
    @property
    def root_quat_w(self) -> wp.array:
        """Same as :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @deprecate(replacement="root_com_vel_w")
    @property
    def root_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_vel_w`."""
        return self.root_com_vel_w

    @deprecate(replacement="root_com_lin_vel_w")
    @property
    def root_lin_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_lin_vel_w`."""
        return self.root_com_lin_vel_w

    @deprecate(replacement="root_com_ang_vel_w")
    @property
    def root_ang_vel_w(self) -> wp.array:
        """Same as :attr:`root_com_ang_vel_w`."""
        return self.root_com_ang_vel_w

    @deprecate(replacement="root_com_lin_vel_b")
    @property
    def root_lin_vel_b(self) -> wp.array:
        """Same as :attr:`root_com_lin_vel_b`."""
        return self.root_com_lin_vel_b

    @deprecate(replacement="root_com_ang_vel_b")
    @property
    def root_ang_vel_b(self) -> wp.array:
        """Same as :attr:`root_com_ang_vel_b`."""
        return self.root_com_ang_vel_b

    @deprecate(replacement="body_link_pose_w")
    @property
    def body_pose_w(self) -> wp.array:
        """Same as :attr:`body_link_pose_w`."""
        return self.body_link_pose_w

    @deprecate(replacement="body_link_pos_w")
    @property
    def body_pos_w(self) -> wp.array:
        """Same as :attr:`body_link_pos_w`."""
        return self.body_link_pos_w

    @deprecate(replacement="body_link_quat_w")
    @property
    def body_quat_w(self) -> wp.array:
        """Same as :attr:`body_link_quat_w`."""
        return self.body_link_quat_w

    @deprecate(replacement="body_com_vel_w")
    @property
    def body_vel_w(self) -> wp.array:
        """Same as :attr:`body_com_vel_w`."""
        return self.body_com_vel_w

    @deprecate(replacement="body_com_lin_vel_w")
    @property
    def body_lin_vel_w(self) -> wp.array:
        """Same as :attr:`body_com_lin_vel_w`."""
        return self.body_com_lin_vel_w

    @deprecate(replacement="body_com_ang_vel_w")
    @property
    def body_ang_vel_w(self) -> wp.array:
        """Same as :attr:`body_com_ang_vel_w`."""
        return self.body_com_ang_vel_w

    @deprecate(replacement="body_com_acc_w")
    @property
    def body_acc_w(self) -> wp.array:
        """Same as :attr:`body_com_acc_w`."""
        return self.body_com_acc_w

    @deprecate(replacement="body_com_lin_acc_w")
    @property
    def body_lin_acc_w(self) -> wp.array:
        """Same as :attr:`body_com_lin_acc_w`."""
        return self.body_com_lin_acc_w

    @deprecate(replacement="body_com_ang_acc_w")
    @property
    def body_ang_acc_w(self) -> wp.array:
        """Same as :attr:`body_com_ang_acc_w`."""
        return self.body_com_ang_acc_w

    @deprecate(replacement="body_com_pos_b")
    @property
    def com_pos_b(self) -> wp.array:
        """Same as :attr:`body_com_pos_b`."""
        return self.body_com_pos_b

    @deprecate(replacement="body_com_quat_b")
    @property
    def com_quat_b(self) -> wp.array:
        """Same as :attr:`body_com_quat_b`."""
        return self.body_com_quat_b

    @deprecate(replacement="joint_pos_limits")
    @property
    def joint_limits(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        omni.log.warn(
            "The `joint_limits` property will be deprecated in a future release. Please use `joint_pos_limits` instead."
        )
        return self.joint_pos_limits

    def joint_friction(self) -> wp.array:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        omni.log.warn(
            "The `joint_friction` property will be deprecated in a future release. Please use"
            " `joint_friction_coeff` instead."
        )
        return self.joint_friction_coeff

    @deprecate(replacement="fixed_tendon_pos_limits")
    @property
    def fixed_tendon_limit(self) -> wp.array:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        omni.log.warn(
            "The `fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `fixed_tendon_pos_limits` instead."
        )
        return self.fixed_tendon_pos_limits