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
import warp as wp

from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.math import normalize

from isaaclab_newton.assets import kernels as shared_kernels
from isaaclab_newton.physics import NewtonManager as SimulationManager

if TYPE_CHECKING:
    from newton.selection import ArticulationView


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

    __backend_name__: str = "newton"
    """The name of the backend for the rigid object data."""

    def __init__(self, root_view: ArticulationView, device: str):
        """Initializes the rigid object data.

        Args:
            root_view: The root rigid body view.
            device: The device used for processing.
        """
        super().__init__(root_view, device)
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: ArticulationView = weakref.proxy(root_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False

        # Convert to direction vector
        gravity = wp.to_torch(SimulationManager.get_model().gravity)[0]
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self._root_view.count, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_view.count, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = wp.from_torch(gravity_dir, dtype=wp.vec3f)
        self.FORWARD_VEC_B = wp.from_torch(forward_vec, dtype=wp.vec3f)

        self._create_simulation_bindings()
        self._create_buffers()

    @property
    def is_primed(self) -> bool:
        """Whether the rigid object data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the rigid object data is fully instantiated and ready to use.

        .. note::
            Once this quantity is set to True, it cannot be changed.

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
        # Trigger an update of the body com acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.body_com_acc_w

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    """
    Defaults.
    """

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose ``[pos, quat]`` in local environment frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The position and quaternion are of the rigid body's actor frame.
        """
        return self._default_root_pose

    @default_root_pose.setter
    def default_root_pose(self, value: wp.array) -> None:
        """Set the default root pose.

        Args:
            value: The default root pose. Shape is (num_instances, 7).

        Raises:
            ValueError: If the rigid object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object data is already primed.")
        self._default_root_pose.assign(value)

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity ``[lin_vel, ang_vel]`` in local environment frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        The linear and angular velocities are of the rigid body's center of mass frame.
        """
        return self._default_root_vel

    @default_root_vel.setter
    def default_root_vel(self, value: wp.array) -> None:
        """Set the default root velocity.

        Args:
            value: The default root velocity. Shape is (num_instances, 6).

        Raises:
            ValueError: If the rigid object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object data is already primed.")
        self._default_root_vel.assign(value)

    """
    Root state properties.
    """

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        This quantity is the pose of the actor frame of the root rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_root_link_pose_w

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity and compute link velocity
            wp.launch(
                shared_kernels.get_root_link_vel_from_root_com_vel,
                dim=self._num_instances,
                inputs=[
                    self.root_com_vel_w,
                    self.root_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._root_link_vel_w.data,
                ],
                device=self.device,
            )
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        This quantity is the pose of the center of mass frame of the root rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            wp.launch(
                shared_kernels.get_root_com_pose_from_root_link_pose,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._root_com_pose_w.data,
                ],
                device=self.device,
            )
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        return self._sim_bind_root_com_vel_w

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> wp.array:
        """Mass of all bodies in the simulation world frame.

        Shape is (num_instances, 1, 1), dtype = wp.float32.
        In torch this resolves to (num_instances, 1, 1).
        """
        return self._body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Inertia of all bodies in the simulation world frame.

        Shape is (num_instances, 1, 9), dtype = wp.float32.
        In torch this resolves to (num_instances, 1, 9).
        """
        return self._body_inertia

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.transformf. In torch this resolves to (num_instances, 1, 7).
        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self._sim_bind_body_link_pose_w

    @property
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 1, 6).
        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        return self.root_link_vel_w.reshape((self._num_instances, 1))

    @property
    def body_com_pose_w(self) -> wp.array:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.transformf. In torch this resolves to (num_instances, 1, 7).
        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        return self.root_com_pose_w.reshape((self._num_instances, 1))

    @property
    def body_com_vel_w(self) -> wp.array:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 1, 6).
        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        return self._sim_bind_body_com_vel_w

    @property
    def body_com_acc_w(self) -> wp.array:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.

        Shape is (num_instances, 1), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 1, 6).
        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.derive_body_acceleration_from_body_com_velocities,
                dim=(self._num_instances, 1),
                device=self.device,
                inputs=[
                    self._sim_bind_body_com_vel_w,
                    SimulationManager.get_dt(),
                    self._previous_body_com_vel,
                ],
                outputs=[
                    self._body_com_acc_w.data,
                ],
            )
            # set the buffer data and timestamp
            self._body_com_acc_w.timestamp = self._sim_timestamp
            # update the previous velocity
        return self._body_com_acc_w.data

    @property
    def body_com_pos_b(self) -> wp.array:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the center of mass location relative to its body's link frame.
        """
        return self._sim_bind_body_com_pos_b

    @property
    def body_com_pose_b(self) -> wp.array:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, 1), dtype = wp.transformf. In torch this resolves to (num_instances, 1, 7).
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        warnings.warn(
            "In Newton, body com pose always has unit quaternion. Consider using body_com_pos_b instead."
            "Querying this property requires appending a unit quaternion to the position which is expensive.",
            category=UserWarning,
            stacklevel=2,
        )
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            wp.launch(
                shared_kernels.make_dummy_body_com_pose_b,
                dim=(self._num_instances, 1),
                inputs=[
                    self.body_com_pos_b,
                ],
                outputs=[
                    self._body_com_pose_b.data,
                ],
                device=self.device,
            )
            self._body_com_pose_b.timestamp = self._sim_timestamp
        return self._body_com_pose_b.data

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> wp.array:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.GRAVITY_VEC_W, self.root_link_quat_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    @property
    def heading_w(self) -> wp.array:
        """Yaw heading of the base frame (in radians).

        Shape is (num_instances,), dtype = wp.float32. In torch this resolves to (num_instances,).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.root_heading_w,
                dim=self._num_instances,
                inputs=[self.FORWARD_VEC_B, self.root_link_quat_w],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        return self._heading_w.data

    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in base frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        if self._root_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_lin_vel_w, self.root_link_quat_w],
                outputs=[self._root_link_lin_vel_b.data],
                device=self.device,
            )
            self._root_link_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_link_lin_vel_b.data

    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in base frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        if self._root_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_link_ang_vel_w, self.root_link_quat_w],
                outputs=[self._root_link_ang_vel_b.data],
                device=self.device,
            )
            self._root_link_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_link_ang_vel_b.data

    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        if self._root_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_lin_vel_w, self.root_link_quat_w],
                outputs=[self._root_com_lin_vel_b.data],
                device=self.device,
            )
            self._root_com_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_com_lin_vel_b.data

    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root center of mass angular velocity in base frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        if self._root_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_1D_kernel,
                dim=self._num_instances,
                inputs=[self.root_com_ang_vel_w, self.root_link_quat_w],
                outputs=[self._root_com_ang_vel_b.data],
                device=self.device,
            )
            self._root_com_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_com_ang_vel_b.data

    """
    Sliced properties.
    """

    @property
    def root_link_pos_w(self) -> wp.array:
        """Root link position in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self._root_link_pos_w, self.root_link_pose_w)

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._get_quat_from_transform(self._root_link_quat_w, self.root_link_pose_w)

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root linear velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._get_top_from_spatial_vector(self._root_link_lin_vel_w, self.root_link_vel_w)

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._get_bottom_from_spatial_vector(self._root_link_ang_vel_w, self.root_link_vel_w)

    @property
    def root_com_pos_w(self) -> wp.array:
        """Root center of mass position in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        return self._get_pos_from_transform(self._root_com_pos_w, self.root_com_pose_w)

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        return self._get_quat_from_transform(self._root_com_quat_w, self.root_com_pose_w)

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root center of mass linear velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_top_from_spatial_vector(self._root_com_lin_vel_w, self.root_com_vel_w)

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root center of mass angular velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._get_bottom_from_spatial_vector(self._root_com_ang_vel_w, self.root_com_vel_w)

    @property
    def body_link_pos_w(self) -> wp.array:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        return self._get_pos_from_transform(self._body_link_pos_w, self.body_link_pose_w)

    @property
    def body_link_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.quatf. In torch this resolves to (num_instances, 1, 4).
        This quantity is the orientation of the rigid bodies' actor frame relative to the world.
        """
        return self._get_quat_from_transform(self._body_link_quat_w, self.body_link_pose_w)

    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the linear velocity of the rigid bodies' actor frame relative to the world.
        """
        return self._get_top_from_spatial_vector(self._body_link_lin_vel_w, self.body_link_vel_w)

    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the angular velocity of the rigid bodies' actor frame relative to the world.
        """
        return self._get_bottom_from_spatial_vector(self._body_link_ang_vel_w, self.body_link_vel_w)

    @property
    def body_com_pos_w(self) -> wp.array:
        """Positions of all bodies' center of mass in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the position of the rigid bodies' center of mass frame.
        """
        return self._get_pos_from_transform(self._body_com_pos_w, self.body_com_pose_w)

    @property
    def body_com_quat_w(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.quatf. In torch this resolves to (num_instances, 1, 4).
        This quantity is the orientation of the principal axes of inertia of the rigid bodies.
        """
        return self._get_quat_from_transform(self._body_com_quat_w, self.body_com_pose_w)

    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self._get_top_from_spatial_vector(self._body_com_lin_vel_w, self.body_com_vel_w)

    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self._get_bottom_from_spatial_vector(self._body_com_ang_vel_w, self.body_com_vel_w)

    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self._get_top_from_spatial_vector(self._body_com_lin_acc_w, self.body_com_acc_w)

    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self._get_bottom_from_spatial_vector(self._body_com_ang_acc_w, self.body_com_acc_w)

    @property
    def body_com_quat_b(self) -> wp.array:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their
        respective link frames.

        Shape is (num_instances, 1), dtype = wp.quatf. In torch this resolves to (num_instances, 1, 4).
        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        return self._get_quat_from_transform(self._body_com_quat_b, self.body_com_pose_b)

    def _create_simulation_bindings(self) -> None:
        """Create simulation bindings for the root data.

        Direct simulation bindings are pointers to the simulation data, their data is not copied, and should
        only be updated using warp kernels. Any modifications made to the bindings will be reflected in the simulation.
        Hence we encourage users to carefully think about the data they modify and in which order it should be updated.

        .. caution:: This is possible if and only if the properties that we access are strided from newton and not
        indexed. Newton willing this is the case all the time, but we should pay attention to this if things look off.
        """
        # -- root properties
        if self._root_view.is_fixed_base:
            self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(SimulationManager.get_state_0())[
                :, 0, 0
            ]
        else:
            self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms(SimulationManager.get_state_0())[:, 0]
        self._sim_bind_root_com_vel_w = self._root_view.get_root_velocities(SimulationManager.get_state_0())
        if self._sim_bind_root_com_vel_w is not None:
            if self._root_view.is_fixed_base:
                self._sim_bind_root_com_vel_w = self._sim_bind_root_com_vel_w[:, 0, 0]
            else:
                self._sim_bind_root_com_vel_w = self._sim_bind_root_com_vel_w[:, 0]
        # -- body properties
        self._sim_bind_body_com_pos_b = self._root_view.get_attribute("body_com", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_link_pose_w = self._root_view.get_link_transforms(SimulationManager.get_state_0())[:, 0]
        self._sim_bind_body_com_vel_w = self._root_view.get_link_velocities(SimulationManager.get_state_0())[:, 0]
        self._sim_bind_body_mass = self._root_view.get_attribute("body_mass", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_inertia = self._root_view.get_attribute("body_inertia", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_external_wrench = self._root_view.get_attribute("body_f", SimulationManager.get_state_0())[
            :, 0
        ]

    def _create_buffers(self) -> None:
        """Create buffers for the root data."""
        super()._create_buffers()
        self._num_instances = self._root_view.count
        # Initialize history for finite differencing. If the rigid object is fixed, the root com velocity is not
        # available, so we use zeros.
        if self._root_view.get_root_velocities(SimulationManager.get_state_0()) is None:
            logger.warning(
                "Failed to get root com velocity. If the rigid object is fixed, this is expected. "
                "Setting root com velocity to zeros."
            )
            self._sim_bind_root_com_vel_w = wp.zeros(
                (self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
            )
            self._sim_bind_body_com_vel_w = wp.zeros(
                (self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
            )
        # -- default root pose and velocity
        self._default_root_pose = wp.zeros((self._num_instances,), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((self._num_instances,), dtype=wp.spatial_vectorf, device=self.device)

        # -- Body properties
        self._body_mass = wp.clone(self._root_view.get_masses(), device=self.device)
        self._body_inertia = wp.clone(self._root_view.get_inertias(), device=self.device)

        # Initialize history for finite differencing
        self._previous_body_com_vel = wp.clone(self._root_view.get_link_velocities(SimulationManager.get_state_0()))[
            :, 0
        ]

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_vel_w = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._root_link_vel_b = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._projected_gravity_b = TimestampedBuffer(shape=(self._num_instances,), dtype=wp.vec3f, device=self.device)
        self._heading_w = TimestampedBuffer(shape=(self._num_instances,), dtype=wp.float32, device=self.device)
        self._body_link_vel_w = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer(shape=(self._num_instances,), dtype=wp.transformf, device=self.device)
        self._root_com_vel_b = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._root_com_acc_w = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._body_com_acc_w = TimestampedBuffer(
            shape=(self._num_instances, 1), dtype=wp.spatial_vectorf, device=self.device
        )
        # Empty memory pre-allocations
        self._root_state_w = None
        self._root_link_state_w = None
        self._root_com_state_w = None
        self._body_com_quat_b = None
        self._root_link_lin_vel_b = None
        self._root_link_ang_vel_b = None
        self._root_com_lin_vel_b = None
        self._root_com_ang_vel_b = None
        self._root_link_pos_w = None
        self._root_link_quat_w = None
        self._root_link_lin_vel_w = None
        self._root_link_ang_vel_w = None
        self._root_com_pos_w = None
        self._root_com_quat_w = None
        self._root_com_lin_vel_w = None
        self._root_com_ang_vel_w = None
        self._body_link_pos_w = None
        self._body_link_quat_w = None
        self._body_link_lin_vel_w = None
        self._body_link_ang_vel_w = None
        self._body_com_pos_w = None
        self._body_com_quat_w = None
        self._body_com_lin_vel_w = None
        self._body_com_ang_vel_w = None
        self._body_com_lin_acc_w = None
        self._body_com_ang_acc_w = None
        self._body_com_pose_b = None

    """
    Internal helpers.
    """

    def _get_pos_from_transform(self, source: wp.array | None, transform: wp.array) -> wp.array:
        """Generates a position array from a transform array.

        Args:
            transform: The transform array. Shape is (N) dtype=wp.transformf.

        Returns:
            The position array. Shape is (N) dtype=wp.vec3f.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if transform.is_contiguous():
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=transform.ptr,
                    shape=transform.shape,
                    dtype=wp.vec3f,
                    strides=transform.strides,
                    device=self.device,
                )
            else:
                # If the array is no contiguous, we need to create a new array to write to.
                source = wp.zeros((transform.shape[0], 3), dtype=wp.vec3f, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the position part of the transform.
        if not transform.is_contiguous():
            # Launch the right kernel based on the shape of the source array.
            if len(source.shape) > 1:
                wp.launch(
                    shared_kernels.split_transform_to_pos_2d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_transform_to_pos_1d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
        return source

    def _get_quat_from_transform(self, source: wp.array | None, transform: wp.array) -> wp.array:
        """Generates a quaternion array from a transform array.

        Args:
            transform: The transform array. Shape is (N) dtype=wp.transformf.

        Returns:
            The quaternion array. Shape is (N) dtype=wp.quatf.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if transform.is_contiguous():
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=transform.ptr + 3 * 4,
                    shape=transform.shape,
                    dtype=wp.quatf,
                    strides=transform.strides,
                    device=self.device,
                )
            else:
                # If the array is no contiguous, we need to create a new array to write to.
                source = wp.zeros((transform.shape[0], 4), dtype=wp.quatf, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the quaternion part of the transform.
        if not transform.is_contiguous():
            # Launch the right kernel based on the shape of the source array.
            if len(source.shape) > 1:
                wp.launch(
                    shared_kernels.split_transform_to_quat_2d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_transform_to_quat_1d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
        # Return the source array. (no-op if the array is contiguous.)
        return source

    def _get_top_from_spatial_vector(self, source: wp.array | None, spatial_vector: wp.array) -> wp.array:
        """Gets the top part of a spatial vector array.

        For instance the linear velocity is the top part of a velocity vector.

        Args:
            spatial_vector: The spatial vector array. Shape is (N) dtype=wp.spatial_vectorf.

        Returns:
            The top part of the spatial vector array. Shape is (N) dtype=wp.vec3f.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if spatial_vector.is_contiguous():
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=spatial_vector.ptr,
                    shape=spatial_vector.shape,
                    dtype=wp.vec3f,
                    strides=spatial_vector.strides,
                    device=self.device,
                )
            else:
                # If the array is no contiguous, we need to create a new array to write to.
                source = wp.zeros((spatial_vector.shape[0], 3), dtype=wp.vec3f, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the top part of the spatial vector.
        if not spatial_vector.is_contiguous():
            # Launch the right kernel based on the shape of the source array.
            if len(source.shape) > 1:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_top_2d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_top_1d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
        # Return the source array. (no-op if the array is contiguous.)
        return source

    def _get_bottom_from_spatial_vector(self, source: wp.array | None, spatial_vector: wp.array) -> wp.array:
        """Gets the bottom part of a spatial vector array.

        For instance the angular velocity is the bottom part of a velocity vector.

        Args:
            spatial_vector: The spatial vector array. Shape is (N) dtype=wp.spatial_vectorf.

        Returns:
            The bottom part of the spatial vector array. Shape is (N) dtype=wp.vec3f.
        """
        # Check if we already created the lazy buffer.
        if source is None:
            if spatial_vector.is_contiguous():
                # Check if the array is contiguous. If so, we can just return a strided array.
                # Then this update becomes a no-op.
                return wp.array(
                    ptr=spatial_vector.ptr + 3 * 4,
                    shape=spatial_vector.shape,
                    dtype=wp.vec3f,
                    strides=spatial_vector.strides,
                    device=self.device,
                )
            else:
                # If the array is no contiguous, we need to create a new array to write to.
                source = wp.zeros((spatial_vector.shape[0], 3), dtype=wp.vec3f, device=self.device)

        # If the array is not contiguous, we need to launch the kernel to get the bottom part of the spatial vector.
        if not spatial_vector.is_contiguous():
            # Launch the right kernel based on the shape of the source array.
            if len(source.shape) > 1:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_bottom_2d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_bottom_1d,
                    dim=source.shape,
                    inputs=[
                        source,
                    ],
                    outputs=[
                        source,
                    ],
                    device=self.device,
                )
        # Return the source array. (no-op if the array is contiguous.)
        return source

    """
    Deprecated properties.
    """

    @property
    def root_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_com_vel_w`."""
        warnings.warn(
            "The `root_state_w` property will be deprecated in IsaacLab 4.0. Please use `root_link_pose_w` and "
            "`root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_state_w.data,
                ],
                device=self.device,
            )
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`root_link_pose_w` and :attr:`root_link_vel_w`."""
        warnings.warn(
            "The `root_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `root_link_pose_w` and "
            "`root_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.root_link_vel_w,
                ],
                outputs=[
                    self._root_link_state_w.data,
                ],
                device=self.device,
            )
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`root_com_pose_w` and :attr:`root_com_vel_w`."""
        warnings.warn(
            "The `root_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `root_com_pose_w` and "
            "`root_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_com_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_com_state_w.data,
                ],
                device=self.device,
            )
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    @property
    def default_root_state(self) -> wp.array:
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
        if self._default_root_state is None:
            self._default_root_state = wp.zeros((self._num_instances), dtype=shared_kernels.vec13f, device=self.device)
        wp.launch(
            shared_kernels.concat_root_pose_and_vel_to_state,
            dim=self._num_instances,
            inputs=[
                self._default_root_pose,
                self._default_root_vel,
            ],
            outputs=[
                self._default_root_state,
            ],
            device=self.device,
        )
        return self._default_root_state

    @property
    def body_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_com_vel_w`."""
        warnings.warn(
            "The `body_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Access internal buffer directly to avoid cascading deprecation warnings from root_state_w
        if self._root_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_state_w.data,
                ],
                device=self.device,
            )
            self._root_state_w.timestamp = self._sim_timestamp
        return self._root_state_w.data.reshape((self._num_instances, 1))

    @property
    def body_link_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_link_vel_w`."""
        warnings.warn(
            "The `body_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Access internal buffer directly to avoid cascading deprecation warnings from root_link_state_w
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_link_pose_w,
                    self.root_link_vel_w,
                ],
                outputs=[
                    self._root_link_state_w.data,
                ],
                device=self.device,
            )
            self._root_link_state_w.timestamp = self._sim_timestamp
        return self._root_link_state_w.data.reshape((self._num_instances, 1))

    @property
    def body_com_state_w(self) -> wp.array:
        """Deprecated, same as :attr:`body_com_pose_w` and :attr:`body_com_vel_w`."""
        warnings.warn(
            "The `body_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_com_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_root_pose_and_vel_to_state,
                dim=self._num_instances,
                inputs=[
                    self.root_com_pose_w,
                    self.root_com_vel_w,
                ],
                outputs=[
                    self._root_com_state_w.data,
                ],
                device=self.device,
            )
            self._root_com_state_w.timestamp = self._sim_timestamp
        return self._root_com_state_w.data.reshape((self._num_instances, 1))
