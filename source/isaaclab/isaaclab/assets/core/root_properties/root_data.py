# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import warp as wp

from newton.selection import ArticulationView as NewtonArticulationView
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.buffers import TimestampedWarpBuffer
from isaaclab.utils.helpers import deprecated, warn_overhead_cost

from isaaclab.assets.core.kernels import (
    combine_frame_transforms_partial,
    combine_pose_and_velocity_to_state,
    compute_heading,
    project_com_velocity_to_link_frame,
    project_vec_from_quat_single,
    project_velocities_to_frame,
    derive_body_acceleration_from_velocity,
    generate_pose_from_position_with_unit_quaternion,
    vec13f,
)


class RootData:
    def __init__(self, root_newton_view: NewtonArticulationView, device: str):
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
        self._root_newton_view: NewtonArticulationView = weakref.proxy(root_newton_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # obtain global simulation view
        gravity = wp.to_torch(NewtonManager.get_model().gravity)[0]
        gravity_dir = [float(i) / sum(gravity) for i in gravity]
        # Initialize constants
        self.GRAVITY_VEC_W = wp.vec3f(gravity_dir[0], gravity_dir[1], gravity_dir[2])
        self.FORWARD_VEC_B = wp.vec3f((1.0, 0.0, 0.0))

        self._create_simulation_bindings()
        self._create_buffers()

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the root acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.root_com_acc_w

    def _create_simulation_bindings(self):
        """Create simulation bindings for the root data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """
        self._sim_bind_root_link_pose_w = self._root_newton_view.get_root_transforms(NewtonManager.get_state_0())
        self._sim_bind_root_com_vel_w = self._root_newton_view.get_root_velocities(NewtonManager.get_state_0())
        self._sim_bind_body_com_pos_b = self._root_newton_view.get_attribute("body_com", NewtonManager.get_model())[:, 0]
        self._sim_bind_root_mass = self._root_newton_view.get_attribute("mass", NewtonManager.get_model())[:, 0]
        self._sim_bind_root_inertia = self._root_newton_view.get_attribute("inertia", NewtonManager.get_model())[:, 0]
        self._sim_bind_root_external_wrench = self._root_newton_view.get_attribute("body_f", NewtonManager.get_state_0())[:, 0]

    def _create_buffers(self):
        """Create buffers for the root data."""
        # Initialize history for finite differencing
        self._previous_root_com_vel = wp.clone(self._root_newton_view.get_root_velocities(NewtonManager.get_state_0()))

        # -- default root pose and velocity
        self._default_root_pose = wp.zeros((self._root_newton_view.count), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((self._root_newton_view.count), dtype=wp.spatial_vectorf, device=self.device)

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_vel_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        self._root_link_vel_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        self._projected_gravity_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.vec3f)
        self._heading_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.float32)
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.transformf)
        self._root_com_vel_b = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)
        self._root_com_acc_w = TimestampedWarpBuffer(shape=(self._root_newton_view.count), dtype=wp.spatial_vectorf)

    ##
    # Direct simulation bindings accessors.
    ##

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``wp.transformf`` in the world frame. Shape is (num_instances,).

        The pose is in the form of [pos, quat]. The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_root_link_pose_w

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``wp.spatial_vectorf`` in the world frame. Shape is (num_instances,).

        The velocity is in the form of [ang_vel, lin_vel].
        """
        return self._sim_bind_root_com_vel_w

    ##
    # Default accessors.
    ##

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in the local environment frame. Shape is (num_instances, 7).

        The position and quaternion are of the articulation root's actor frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_root_pose

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in the local environment frame. Shape is (num_instances, 6).

        The linear and angular velocities are of the articulation root's center of mass frame.

        This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
        """
        return self._default_root_vel

    @default_root_pose.setter
    def default_root_pose(self, value: wp.array) -> None:
        self._default_root_pose = value

    @default_root_vel.setter
    def default_root_vel(self, value: wp.array) -> None:
        self._default_root_vel = value    

    ##
    # Root state properties.
    ##

    @property
    def root_mass(self) -> wp.array:
        """Root mass ``wp.float32`` in the world frame. Shape is (num_instances,)."""
        return self._sim_bind_root_mass

    @property
    def root_inertia(self) -> wp.array:
        """Root inertia ``wp.mat33`` in the world frame. Shape is (num_instances, 9)."""
        return self._sim_bind_root_inertia

    @property
    def root_external_wrench(self) -> wp.array:
        """Root external wrench ``wp.spatial_vectorf`` in the world frame. Shape is (num_instances, 6)."""
        return self._sim_bind_root_external_wrench
        
    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances,). Velocities are in the form of [wx, wy, wz, vx, vy, vz].
        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                project_com_velocity_to_link_frame,
                dim=(self._root_newton_view.count),
                device=self.device,
                inputs=[
                    self.root_com_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._sim_bind_body_com_pos_b,
                    self._root_link_vel_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``wp.transformf`` in simulation world frame.

        Shapes are (num_instances,). The pose is in the form of [pos, quat].
        This quantity is the pose of the articulation root's center of mass frame relative to the world.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            wp.launch(
                combine_frame_transforms_partial,
                dim=(self._root_newton_view.count),
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

    # TODO: Pre-allocate the state array. 
    @property
    @warn_overhead_cost(
        "root_link_pose_w and root_com_vel_w",
        "This function combines the root link pose and root center of mass velocity into a single state. However, Newton"
        " outputs pose and velocities separately. Consider using only one of them instead. If both are required, use both"
        " root_link_pose_w and root_com_vel_w instead.",
    )
    def root_state_w(self) -> wp.array:
        """Root state ``vec13f`` in simulation world frame.

        Shapes are (num_instances, 13). The pose is in the form of [pos, quat].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's center of mass frame.

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        state = wp.zeros((self._root_newton_view.count, 13), dtype=wp.float32, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_newton_view.count,),
            device=self.device,
            inputs=[
                self._sim_bind_root_link_pose_w,
                self._sim_bind_root_com_vel_w,
                state,
            ],
        )
        return state

    @property
    @warn_overhead_cost(
        "root_link_pose_w and root_link_vel_w",
        "This function combines the root link pose and root link velocity into a single state. However, Newton"
        " outputs pose and velocities separately. Consider using only one of them instead. If both are required, use both"
        " root_link_pose_w and root_link_vel_w instead.",
    )
    def root_link_state_w(self) -> wp.array:
        """Root link state ``vec13f`` in simulation world frame.

        Shapes are (num_instances, 13). The pose is in the form of [pos, quat].
        The pose is of the articulation root's actor frame relative to the world.
        The velocity is of the articulation root's actor frame.

        .. note:: The velocity is in the form of [wx, wy, wz, vx, vy, vz].
        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        state = wp.zeros((self._root_newton_view.count, 13), dtype=wp.float32, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_newton_view.count,),
            device=self.device,
            inputs=[
                self._sim_bind_root_link_pose_w,
                self.root_link_vel_w,
                state,
            ],
        )
        return state

    @property
    @warn_overhead_cost(
        "root_com_pose_w and root_com_vel_w",
        "This function combines the root center of mass pose and root center of mass velocity into a single state. However, Newton"
        " outputs pose and velocities separately. Consider using only one of them instead. If both are required, use both"
        " root_com_pose_w and root_com_vel_w instead.",
    )
    def root_com_state_w(self) -> wp.array:
        """Root center of mass state ``vec13f`` in simulation world frame.

        Shapes are (num_instances, 13). The pose is in the form of [pos, quat].
        The pose is of the articulation root's center of mass frame relative to the world.
        The velocity is of the articulation root's center of mass frame.

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        state = wp.zeros((self._root_newton_view.count, 13), dtype=wp.float32, device=self.device)
        wp.launch(
            combine_pose_and_velocity_to_state,
            dim=(self._root_newton_view.count,),
            device=self.device,
            inputs=[
                self.root_com_pose_w,
                self._sim_bind_root_com_vel_w,
                state,
            ],
        )
        return state

    @property
    def root_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,).
        All values are relative to the world.

        .. note:: The acceleration is in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._root_com_acc_w.timestamp < self._sim_timestamp:
            wp.launch(
                derive_body_acceleration_from_velocity,
                dim=(self._root_newton_view.count),
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._previous_root_com_vel,
                    NewtonManager.get_dt(),
                    self._root_com_acc_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._root_com_acc_w.timestamp = self._sim_timestamp
        return self._root_com_acc_w.data

    @property
    def root_com_pos_b(self) -> wp.array:
        """Root center of mass position ``wp.vec3f`` in base frame. Shape is (num_instances, 3).

        This quantity is the position of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._sim_bind_body_com_pos_b

    @property
    @warn_overhead_cost(
        "body_com_pose_b",
        "This function outputs the pose of the CoM, containing both position and orientation. However, in Newton, the"
        " CoM is always aligned with the link frame. This means that the quaternion is always [0, 0, 0, 1]. Consider"
        " using the position only instead.",
    )
    def root_com_pose_b(self) -> wp.array:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        out = wp.zeros(
            (self._root_newton_view.count,), dtype=wp.transformf, device=self.device
        )
        wp.launch(
            generate_pose_from_position_with_unit_quaternion,
            dim=self._root_newton_view.count,
            inputs=[
                self._sim_bind_body_com_pos_b,
                out,
            ],
        )
        return out
    
    @property
    def root_com_quat_b(self) -> wp.array:
        """Root center of mass orientation (w, x, y, z) in base frame. Shape is (num_instances, 4).

        This quantity is the orientation of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self.root_com_pose_b[:, 3:]

    ##
    # Derived Properties.
    ##

    # FIXME: USE SIM_BIND_LINK_POSE_W RATHER THAN JUST THE QUATERNION
    @property
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_vec_from_quat_single,
                dim=self._root_newton_view.count,
                inputs=[
                    self.GRAVITY_VEC_W,
                    self.root_link_quat_w,
                    self._projected_gravity_b.data,
                ],
            )
            # set the buffer data and timestamp
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    # FIXME: USE SIM_BIND_LINK_POSE_W RATHER THAN JUST THE QUATERNION
    @property
    def heading_w(self) -> wp.array:
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
                ],
            )
            # set the buffer data and timestamp
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data

    @property
    def root_link_vel_b(self) -> wp.array:
        """Root link velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._root_link_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_newton_view.count,
                inputs=[
                    self.root_link_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._root_link_vel_b.data,
                ],
            )
            self._root_link_vel_b.timestamp = self._sim_timestamp
        return self._root_link_vel_b.data

    @property
    def root_com_vel_b(self) -> wp.array:
        """Root center of mass velocity ``wp.spatial_vectorf`` in base frame. Shape is (num_instances).

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._root_com_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                project_velocities_to_frame,
                dim=self._root_newton_view.count,
                inputs=[
                    self._sim_bind_root_com_vel_w,
                    self._sim_bind_root_link_pose_w,
                    self._root_com_vel_b.data,
                ],
            )
            self._root_com_vel_b.timestamp = self._sim_timestamp
        return self._root_com_vel_b.data

    ##
    # Strided properties.
    ##

    @property
    @warn_overhead_cost("root_link_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def root_link_pos_w(self) -> wp.array:
        """Root link position ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._sim_bind_root_link_pose_w[:, :3]

    @property
    @warn_overhead_cost("root_link_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        This quantity is the orientation of the actor frame of the root rigid body.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        return self._sim_bind_root_link_pose_w[:, 3:]

    @property
    @warn_overhead_cost("root_link_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self.root_link_vel_w[:, :3]

    @property
    @warn_overhead_cost("root_link_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self.root_link_vel_w[:, 3:]

    @property
    @warn_overhead_cost("root_com_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_com_pose_w[:, :3]

    @property
    @warn_overhead_cost("root_com_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation ``wp.quatf`` in simulation world frame. Shape is (num_instances,).

        This quantity is the orientation of the root rigid body's center of mass frame.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        return self.root_com_pose_w[:, 3:]

    @property
    @warn_overhead_cost("root_com_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances,).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._sim_bind_root_com_vel_w[:, :3]

    @property
    @warn_overhead_cost("root_com_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity ``wp.vec3f`` in simulation world frame. Shape is (num_instances).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._sim_bind_root_com_vel_w[:, 3:]

    @property
    @warn_overhead_cost("root_link_vel_b", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity ``wp.vec3f`` in base frame. Shape is (num_instances).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return self.root_link_vel_b[:, :3]

    @property
    @warn_overhead_cost("root_link_vel_b", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity ``wp.vec3f`` in base world frame. Shape is (num_instances).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        return self.root_link_vel_b[:, 3:]

    @property
    @warn_overhead_cost("root_com_vel_b", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity ``wp.vec3f`` in base frame. Shape is (num_instances).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return self.root_com_vel_b[:, :3]

    @property
    @warn_overhead_cost("root_com_vel_b", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        return self.root_com_vel_b[:, 3:]