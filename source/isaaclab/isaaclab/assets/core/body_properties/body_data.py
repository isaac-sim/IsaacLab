# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import warp as wp

from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.buffers import TimestampedWarpBuffer
from isaaclab.utils.helpers import deprecated, warn_overhead_cost

from isaaclab.assets.core.kernels import (
    combine_frame_transforms_partial_batch,
    combine_pose_and_velocity_to_state_batched,
    derive_body_acceleration_from_velocity_batched,
    generate_pose_from_position_with_unit_quaternion_batched,
    project_com_velocity_to_link_frame_batch,
    vec13f,
)


class BodyData:
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

        # Set initial time stamp
        self._sim_timestamp = 0.0

        self._create_simulation_bindings()
        self._create_buffers()

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the body acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.body_com_acc_w

    def _create_simulation_bindings(self):
        """Create simulation bindings for the body data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """
        self._sim_bind_body_link_pose_w = self._root_newton_view.get_link_transforms(NewtonManager.get_state_0())
        self._sim_bind_body_com_vel_w = self._root_newton_view.get_link_velocities(NewtonManager.get_state_0())
        self._sim_bind_body_com_pos_b = self._root_newton_view.get_attribute("body_com", NewtonManager.get_model())
        self._sim_bind_body_mass = self._root_newton_view.get_attribute("mass", NewtonManager.get_model())
        self._sim_bind_body_inertia = self._root_newton_view.get_attribute("inertia", NewtonManager.get_model())
        self._sim_bind_body_external_wrench = self._root_newton_view.get_attribute("body_f", NewtonManager.get_state_0())

    def _create_buffers(self):
        """Create buffers for the body data."""
        # Initialize history for finite differencing
        self._previous_body_com_vel = wp.clone(self._root_newton_view.get_link_velocities(NewtonManager.get_state_0()))

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._body_link_vel_w = TimestampedWarpBuffer(
            shape=(self._root_newton_view.count, self._root_newton_view.link_count), dtype=wp.spatial_vectorf
        )
        # -- com frame w.r.t. world frame
        self._body_com_pose_w = TimestampedWarpBuffer(
            shape=(self._root_newton_view.count, self._root_newton_view.link_count), dtype=wp.transformf
        )
        self._body_com_acc_w = TimestampedWarpBuffer(
            shape=(self._root_newton_view.count, self._root_newton_view.link_count), dtype=wp.spatial_vectorf
        )

    ##
    # Direct simulation bindings accessors.
    ##

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``wp.transformf`` in the world frame. Shape is (num_instances, num_bodies).

        The pose is in the form of [pos, quat]. The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_body_link_pose_w

    @property
    def body_com_vel_w(self) -> wp.array:
        """Body center of mass velocity ``wp.spatial_vectorf`` in the world frame. Shape is (num_instances, num_bodies).

        The velocity is in the form of [ang_vel, lin_vel].
        """
        return self._sim_bind_body_com_vel_w

    @property
    def body_com_pos_b(self) -> wp.array:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_body_com_pos_b

    ##
    # Body state properties.
    ##

    @property
    def body_mass(self) -> wp.array:
        """Body mass ``wp.float32`` in the world frame. Shape is (num_instances, num_bodies)."""
        return self._sim_bind_body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Body inertia ``wp.mat33`` in the world frame. Shape is (num_instances, num_bodies, 9)."""
        return self._sim_bind_body_inertia

    @property
    def external_wrench(self) -> wp.array:
        """External wrench ``wp.spatial_vectorf`` in the world frame. Shape is (num_instances, num_bodies, 6)."""
        return self._sim_bind_body_external_wrench

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
                dim=(self._root_newton_view.count, self._root_newton_view.link_count),
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
                dim=(self._root_newton_view.count, self._root_newton_view.link_count),
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

    @property
    @warn_overhead_cost(
        "body_link_pose_w and/or body_com_vel_w",
        "This function outputs the state of the link frame, containing both pose and velocity. However, Newton outputs"
        " pose and velocities separately. Consider using only one of them instead. If both are required, use both"
        " body_link_pose_w and body_com_vel_w instead.",
    )
    def body_state_w(self) -> wp.array:
        """State of all bodies ``vec13f`` in simulation world frame.

        Shapes are (num_instances, num_bodies, 13). The pose is in the form of [pos, quat].

        The pose is of the articulation links' actor frame relative to the world.
        The velocity is of the articulation links' center of mass frame.

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        .. note:: The quaternion is in the form of [x, y, z, w].
        """

        state = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.link_count), dtype=vec13f, device=self.device
        )
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_newton_view.count, self._root_newton_view.link_count),
            device=self.device,
            inputs=[
                self._sim_bind_body_link_pose_w,
                self._sim_bind_body_com_vel_w,
                state,
            ],
        )
        return state

    @property
    @warn_overhead_cost(
        "body_link_pose_w and/or body_link_vel_w",
        "This function outputs the state of the link frame, containing both pose and velocity. However, Newton outputs"
        " pose and velocities separately. Consider using only one of them instead. If both are required, use both"
        " body_link_pose_w and body_link_vel_w instead.",
    )
    def body_link_state_w(self) -> wp.array:
        """State of all bodies' link frame ``vec13f`` in simulation world frame.

        Shapes are (num_instances, num_bodies, 13). The pose is in the form of [pos, quat].
        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        .. note:: The quaternion is in the form of [x, y, z, w].
        """

        state = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.link_count), dtype=vec13f, device=self.device
        )
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_newton_view.count, self._root_newton_view.link_count),
            device=self.device,
            inputs=[
                self._sim_bind_body_link_pose_w,
                self.body_link_vel_w,
                state,
            ],
        )
        return state

    @property
    @warn_overhead_cost(
        "body_com_pose_w and/or body_com_vel_w",
        "This function outputs the state of the CoM, containing both pose and velocity. However, Newton outputs pose"
        " and velocities separately. Consider using only one of them instead. If both are required, use both"
        " body_com_pose_w and body_com_vel_w instead.",
    )
    def body_com_state_w(self) -> wp.array:
        """State of all bodies center of mass ``[wp.transformf, wp.spatial_vectorf]`` in simulation world frame.

        Shapes are (num_instances, num_bodies,), (num_instances, num_bodies,). The pose is in the form of [pos, quat].

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.

        .. note:: The velocity is in the form of [vx, vy, vz, wx, wy, wz].
        .. note:: The quaternion is in the form of [x, y, z, w].
        """

        state = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.link_count), dtype=vec13f, device=self.device
        )
        wp.launch(
            combine_pose_and_velocity_to_state_batched,
            dim=(self._root_newton_view.count, self._root_newton_view.link_count),
            device=self.device,
            inputs=[
                self.body_com_pose_w,
                self._sim_bind_body_com_vel_w,
                state,
            ],
        )
        return state

    @property
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies center of mass ``wp.spatial_vectorf`` in simulation world frame.

        Shapes are (num_instances, num_bodies,). All values are relative to the world.

        .. note:: The acceleration is in the form of [vx, vy, vz, wx, wy, wz].
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            wp.launch(
                derive_body_acceleration_from_velocity_batched,
                dim=(self._root_newton_view.count, self._root_newton_view.link_count),
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

    @property
    @warn_overhead_cost(
        "body_com_pose_b",
        "This function outputs the pose of the CoM, containing both position and orientation. However, in Newton, the"
        " CoM is always aligned with the link frame. This means that the quaternion is always [0, 0, 0, 1]. Consider"
        " using the position only instead.",
    )
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``wp.transformf`` of all bodies in their respective body's link frames.

        Shapes are (num_instances, num_bodies,). The pose is in the form of [pos, quat].
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        out = wp.zeros(
            (self._root_newton_view.count, self._root_newton_view.link_count), dtype=wp.transformf, device=self.device
        )
        wp.launch(
            generate_pose_from_position_with_unit_quaternion_batched,
            dim=(self._root_newton_view.count, self._root_newton_view.link_count),
            inputs=[
                self.body_com_pos_b,
                out,
            ],
        )
        return out

    ##
    # Strided properties.
    ##

    @property
    @warn_overhead_cost("root_link_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        return self._sim_bind_body_link_pose_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_link_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def body_link_quat_w(self) -> wp.array:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        return self._sim_bind_body_link_pose_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_link_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_link_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        return self.body_link_vel_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_com_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame ``wp.vec3f``. Shape is (num_instances, num_bodies).

        This quantity is the position of the articulation bodies' actor frame.
        """
        return self.body_com_pose_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_com_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def body_com_quat_w(self) -> wp.array:
        """Orientation ``wp.quatf`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the orientation of the articulation bodies' actor frame.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        return self.body_com_pose_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_com_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        return self._sim_bind_body_com_vel_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_com_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        return self._sim_bind_body_com_vel_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_com_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[:, :, :3]

    @property
    @warn_overhead_cost("root_com_vel_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the velocity instead.")
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration ``wp.vec3f`` of all bodies in simulation world frame. Shape is (num_instances, num_bodies).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        return self.body_com_acc_w[:, :, 3:]

    @property
    @warn_overhead_cost("root_com_pose_w", "Returns a strided array, this is fine if this results is called only once at the beginning of the simulation."
    "However, if this is called multiple times, consider using the whole of the pose instead.")
    def body_com_quat_b(self) -> wp.array:
        """Orientation ``wp.quatf`` of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.

        .. note:: The quaternion is in the form of [x, y, z, w].
        """
        return self.body_com_pose_b[:, :, 3:]