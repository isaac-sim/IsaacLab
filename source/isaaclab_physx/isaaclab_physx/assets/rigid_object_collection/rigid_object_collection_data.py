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

from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import BaseRigidObjectCollectionData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.math import normalize
from isaaclab.utils.warp import ProxyArray

from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    import omni.physics.tensors.api as physx

# import logger
logger = logging.getLogger(__name__)


class RigidObjectCollectionData(BaseRigidObjectCollectionData):
    """Data container for a rigid object collection.

    This class contains the data for a rigid object collection in the simulation. The data includes the state of
    all the bodies in the collection. The data is stored in the simulation world frame unless otherwise specified.
    The data is in the order ``(num_instances, num_objects, data_size)``, where data_size is the size of the data.

    For a rigid body, there are two frames of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings of the simulation, the actor frame and the center of mass frame may be the same.
    This needs to be taken into account when interpreting the data.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.

    .. note::
        **Pull-to-refresh model.** Properties pull fresh data from the PhysX tensor API on first
        access per timestamp and cache the result. This differs from Newton, where buffers are
        refreshed automatically by the simulation.

    .. note::
        **ProxyArray pointer stability.** Each :class:`ProxyArray` wrapper is created once and
        reused because the PhysX tensor API returns views into stable, pre-allocated GPU buffers
        whose device pointer does not change across simulation steps.
    """

    __backend_name__: str = "physx"
    """The name of the backend for the rigid object collection data."""

    def __init__(self, root_view: physx.RigidBodyView, num_bodies: int, device: str):
        """Initializes the rigid object data.

        Args:
            root_view: The root rigid body collection view.
            num_bodies: The number of bodies in the collection.
            device: The device used for processing.
        """
        super().__init__(root_view, num_bodies, device)
        self.num_bodies = num_bodies
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_view: physx.RigidBodyView = weakref.proxy(root_view)
        self.num_instances = self._root_view.count // self.num_bodies

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
        self.GRAVITY_VEC_W = ProxyArray(
            wp.from_torch(gravity_dir.repeat(self.num_instances, self.num_bodies, 1), dtype=wp.vec3f)
        )
        self.FORWARD_VEC_B = ProxyArray(
            wp.from_torch(
                torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_instances, self.num_bodies, 1),
                dtype=wp.vec3f,
            )
        )

        self._create_buffers()

    @property
    def is_primed(self) -> bool:
        """Whether the rigid object collection data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the rigid object collection data is fully instantiated and ready to use.

        .. note::
            Once this quantity is set to True, it cannot be changed.

        Args:
            value: The primed state.

        Raises:
            ValueError: If the rigid object collection data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object collection data is already primed.")
        self._is_primed = value

    def update(self, dt: float) -> None:
        """Updates the data for the rigid object collection.

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
    def default_body_pose(self) -> ProxyArray:
        """Default body pose ``[pos, quat]`` in local environment frame.

        The position and quaternion are of the rigid body's actor frame.
        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).
        """
        if self._default_body_pose_ta is None:
            self._default_body_pose_ta = ProxyArray(self._default_body_pose)
        return self._default_body_pose_ta

    @default_body_pose.setter
    def default_body_pose(self, value: wp.array) -> None:
        """Set the default body pose.

        Args:
            value: The default body pose. Shape is (num_instances, num_bodies, 7).

        Raises:
            ValueError: If the rigid object collection data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object collection data is already primed.")
        self._default_body_pose.assign(value)

    @property
    def default_body_vel(self) -> ProxyArray:
        """Default body velocity ``[lin_vel, ang_vel]`` in local environment frame.

        The linear and angular velocities are of the rigid body's center of mass frame.
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).
        """
        if self._default_body_vel_ta is None:
            self._default_body_vel_ta = ProxyArray(self._default_body_vel)
        return self._default_body_vel_ta

    @default_body_vel.setter
    def default_body_vel(self, value: wp.array) -> None:
        """Set the default body velocity.

        Args:
            value: The default body velocity. Shape is (num_instances, num_bodies, 6).

        Raises:
            ValueError: If the rigid object collection data is already primed.
        """
        if self._is_primed:
            raise ValueError("The rigid object collection data is already primed.")
        self._default_body_vel.assign(value)

    """
    Body state properties.
    """

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).
        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation and reshape
            pose = self._reshape_view_to_data_2d(self._root_view.get_transforms().view(wp.transformf))
            # set the buffer data and timestamp
            self._body_link_pose_w.data = pose
            self._body_link_pose_w.timestamp = self._sim_timestamp
            # Rebind ProxyArray since reshape creates a new wp.array each time
            if self._body_link_pose_w_ta is not None:
                self._body_link_pose_w_ta = ProxyArray(pose)
                self._body_link_pos_w_ta = None
                self._body_link_quat_w_ta = None

        if self._body_link_pose_w_ta is None:
            self._body_link_pose_w_ta = ProxyArray(self._body_link_pose_w.data)
        return self._body_link_pose_w_ta

    @property
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).
        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_link_vel_from_body_com_vel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_com_vel_w,
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._body_link_vel_w.data,
                ],
                device=self.device,
            )
            self._body_link_vel_w.timestamp = self._sim_timestamp

        if self._body_link_vel_w_ta is None:
            self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
        return self._body_link_vel_w_ta

    @property
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).
        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[
                    self._body_com_pose_w.data,
                ],
                device=self.device,
            )
            self._body_com_pose_w.timestamp = self._sim_timestamp

        if self._body_com_pose_w_ta is None:
            self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).
        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            vel = self._reshape_view_to_data_2d(self._root_view.get_velocities().view(wp.spatial_vectorf))
            self._body_com_vel_w.data = vel
            self._body_com_vel_w.timestamp = self._sim_timestamp
            # Rebind ProxyArray since reshape creates a new wp.array each time
            if self._body_com_vel_w_ta is not None:
                self._body_com_vel_w_ta = ProxyArray(vel)
                self._body_com_lin_vel_w_ta = None
                self._body_com_ang_vel_w_ta = None

        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(self._body_com_vel_w.data)
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf. In torch this resolves to
        (num_instances, num_bodies, 6).
        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            acc = self._reshape_view_to_data_2d(self._root_view.get_accelerations().view(wp.spatial_vectorf))
            self._body_com_acc_w.data = acc
            self._body_com_acc_w.timestamp = self._sim_timestamp
            # Rebind ProxyArray since reshape creates a new wp.array each time
            if self._body_com_acc_w_ta is not None:
                self._body_com_acc_w_ta = ProxyArray(acc)
                self._body_com_lin_acc_w_ta = None
                self._body_com_ang_acc_w_ta = None
        if self._body_com_acc_w_ta is None:
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
        return self._body_com_acc_w_ta

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, num_bodies), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_bodies, 7).
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # obtain the coms
            poses = self._reshape_view_to_data_2d(self._root_view.get_coms().view(wp.transformf))
            # read data from simulation
            self._body_com_pose_b.data.assign(poses)
            self._body_com_pose_b.timestamp = self._sim_timestamp

        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    @property
    def body_mass(self) -> ProxyArray:
        """Mass of all bodies in the simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.float32.
        In torch this resolves to (num_instances, num_bodies).
        """
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Inertia of all bodies in the simulation world frame.

        Shape is (num_instances, num_bodies, 9), dtype = wp.float32.
        In torch this resolves to (num_instances, num_bodies, 9).
        """
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia)
        return self._body_inertia_ta

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction on base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        """
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.GRAVITY_VEC_W, self.body_link_quat_w],
                outputs=[self._projected_gravity_b.data],
                device=self.device,
            )
            self._projected_gravity_b.timestamp = self._sim_timestamp
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b.data)
        return self._projected_gravity_b_ta

    @property
    def heading_w(self) -> ProxyArray:
        """Yaw heading of the base frame (in radians).

        Shape is (num_instances, num_bodies), dtype = wp.float32. In torch this resolves to (num_instances, num_bodies).

        .. note::
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        if self._heading_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.body_heading_w,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.FORWARD_VEC_B, self.body_link_quat_w],
                outputs=[self._heading_w.data],
                device=self.device,
            )
            self._heading_w.timestamp = self._sim_timestamp
        if self._heading_w_ta is None:
            self._heading_w_ta = ProxyArray(self._heading_w.data)
        return self._heading_w_ta

    @property
    def body_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        if self._body_link_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_link_lin_vel_w, self.body_link_quat_w],
                outputs=[self._body_link_lin_vel_b.data],
                device=self.device,
            )
            self._body_link_lin_vel_b.timestamp = self._sim_timestamp
        if self._body_link_lin_vel_b_ta is None:
            self._body_link_lin_vel_b_ta = ProxyArray(self._body_link_lin_vel_b.data)
        return self._body_link_lin_vel_b_ta

    @property
    def body_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        if self._body_link_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_link_ang_vel_w, self.body_link_quat_w],
                outputs=[self._body_link_ang_vel_b.data],
                device=self.device,
            )
            self._body_link_ang_vel_b.timestamp = self._sim_timestamp
        if self._body_link_ang_vel_b_ta is None:
            self._body_link_ang_vel_b_ta = ProxyArray(self._body_link_ang_vel_b.data)
        return self._body_link_ang_vel_b_ta

    @property
    def body_com_lin_vel_b(self) -> ProxyArray:
        """Root center of mass linear velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        if self._body_com_lin_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_com_lin_vel_w, self.body_link_quat_w],
                outputs=[self._body_com_lin_vel_b.data],
                device=self.device,
            )
            self._body_com_lin_vel_b.timestamp = self._sim_timestamp
        if self._body_com_lin_vel_b_ta is None:
            self._body_com_lin_vel_b_ta = ProxyArray(self._body_com_lin_vel_b.data)
        return self._body_com_lin_vel_b_ta

    @property
    def body_com_ang_vel_b(self) -> ProxyArray:
        """Root center of mass angular velocity in base frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        if self._body_com_ang_vel_b.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.quat_apply_inverse_2D_kernel,
                dim=(self.num_instances, self.num_bodies),
                inputs=[self.body_com_ang_vel_w, self.body_link_quat_w],
                outputs=[self._body_com_ang_vel_b.data],
                device=self.device,
            )
            self._body_com_ang_vel_b.timestamp = self._sim_timestamp
        if self._body_com_ang_vel_b_ta is None:
            self._body_com_ang_vel_b_ta = ProxyArray(self._body_com_ang_vel_b.data)
        return self._body_com_ang_vel_b_ta

    """
    Sliced properties.
    """

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).
        This quantity is the orientation of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the linear velocity of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the angular velocity of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies' center of mass in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the position of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).
        This quantity is the orientation of the principal axes of inertia of the rigid bodies.
        """
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_bodies, 3).
        This quantity is the center of mass location relative to its body's link frame.
        """
        parent = self.body_com_pose_b
        if self._body_com_pos_b_ta is None:
            self._body_com_pos_b_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_b_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies in their
        respective link frames.

        Shape is (num_instances, num_bodies), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_bodies, 4).
        This quantity is the orientation of the principal axes of inertia relative to its body's link frame.
        """
        parent = self.body_com_pose_b
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_b_ta

    def _create_buffers(self) -> None:
        super()._create_buffers()
        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._body_link_pose_w = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.transformf)
        self._body_link_vel_w = TimestampedBuffer(
            (self.num_instances, self.num_bodies), self.device, wp.spatial_vectorf
        )
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.transformf)
        # -- com frame w.r.t. world frame
        self._body_com_pose_w = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.transformf)
        self._body_com_vel_w = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.spatial_vectorf)
        self._body_com_acc_w = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.spatial_vectorf)
        # -- combined state (these are cached as they concatenate)
        self._body_state_w = TimestampedBuffer(
            (self.num_instances, self.num_bodies), self.device, shared_kernels.vec13f
        )
        self._body_link_state_w = TimestampedBuffer(
            (self.num_instances, self.num_bodies), self.device, shared_kernels.vec13f
        )
        self._body_com_state_w = TimestampedBuffer(
            (self.num_instances, self.num_bodies), self.device, shared_kernels.vec13f
        )

        # -- Default state
        self._default_body_pose = wp.zeros(
            (self.num_instances, self.num_bodies), dtype=wp.transformf, device=self.device
        )
        self._default_body_vel = wp.zeros(
            (self.num_instances, self.num_bodies), dtype=wp.spatial_vectorf, device=self.device
        )
        self._default_body_state = None

        # -- Derived properties
        self._projected_gravity_b = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.vec3f)
        self._heading_w = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.float32)
        self._body_link_lin_vel_b = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.vec3f)
        self._body_link_ang_vel_b = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.vec3f)
        self._body_com_lin_vel_b = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.vec3f)
        self._body_com_ang_vel_b = TimestampedBuffer((self.num_instances, self.num_bodies), self.device, wp.vec3f)

        # -- Body properties (stored in instance order: num_instances, num_bodies[, data_dim])
        # Masses: view returns (B*I, 1) in view order. _reshape_view_to_data gives (I, B) in instance order.
        self._body_mass = self._reshape_view_to_data_2d(self._root_view.get_masses()).reshape(
            (self.num_instances, self.num_bodies)
        )
        # Inertias: view returns (B*I, 9) in view order. Need (I, B, 9) in instance order.
        # _reshape_view_to_data only handles single-element dtypes, so we use _reshape_view_to_data_3d.
        self._body_inertia = self._reshape_view_to_data_3d(self._root_view.get_inertias(), 9)

        # Initialize ProxyArray wrappers
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create pinned ProxyArray wrappers for all data buffers.

        This is called once from :meth:`_create_buffers` during initialization.
        PhysX tensor API buffers have stable GPU pointers across simulation steps,
        so no rebinding is needed (unlike Newton).
        """
        # -- Pinned ProxyArray cache (one per read property, lazily created on first access)
        # Defaults
        self._default_body_pose_ta: ProxyArray | None = None
        self._default_body_vel_ta: ProxyArray | None = None
        # Body state (timestamped)
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._body_com_acc_w_ta: ProxyArray | None = None
        self._body_com_pose_b_ta: ProxyArray | None = None
        # Body properties
        self._body_mass_ta: ProxyArray | None = None
        self._body_inertia_ta: ProxyArray | None = None
        # Derived properties (timestamped)
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._heading_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_b_ta: ProxyArray | None = None
        self._body_link_ang_vel_b_ta: ProxyArray | None = None
        self._body_com_lin_vel_b_ta: ProxyArray | None = None
        self._body_com_ang_vel_b_ta: ProxyArray | None = None
        # Sliced properties (body link)
        self._body_link_pos_w_ta: ProxyArray | None = None
        self._body_link_quat_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_w_ta: ProxyArray | None = None
        self._body_link_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (body com)
        self._body_com_pos_w_ta: ProxyArray | None = None
        self._body_com_quat_w_ta: ProxyArray | None = None
        self._body_com_lin_vel_w_ta: ProxyArray | None = None
        self._body_com_ang_vel_w_ta: ProxyArray | None = None
        self._body_com_lin_acc_w_ta: ProxyArray | None = None
        self._body_com_ang_acc_w_ta: ProxyArray | None = None
        # Sliced properties (body com in body frame)
        self._body_com_pos_b_ta: ProxyArray | None = None
        self._body_com_quat_b_ta: ProxyArray | None = None
        # Deprecated state-concat properties
        self._default_body_state_ta: ProxyArray | None = None
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

    """
    Helpers.
    """

    def _reshape_view_to_data_2d(self, data: wp.array) -> wp.array:
        """Reshapes and arranges the data from the physics view to (num_instances, num_bodies, data_size).

        Args:
            data: The data from the physics view. Shape is (num_instances * num_bodies, data_size).

        Returns:
            The reshaped data. Shape is (num_instances, num_bodies, data_size).
        """
        # The view returns data ordered as (body0_env0, body0_env1, ..., body1_env0, body1_env1, ...)
        # i.e. shape (num_bodies, num_instances) when reshaped.
        # We need (num_instances, num_bodies) so we create a strided view with transposed strides.
        # Use data.device for the strided view (PhysX returns masses/coms/inertias on CPU),
        # then clone to self.device (handles both contiguity and device transfer).
        element_size = wp.types.type_size_in_bytes(data.dtype)
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_instances, self.num_bodies),
            dtype=data.dtype,
            strides=(element_size, self.num_instances * element_size),
            device=data.device,
        )
        return wp.clone(strided_view, self.device)

    def _reshape_view_to_data_3d(self, data: wp.array, data_dim: int) -> wp.array:
        """Reshapes and arranges 2D view data to (num_instances, num_bodies, data_dim).

        Args:
            data: The data from the physics view. Shape is (num_instances * num_bodies, data_dim).
            data_dim: The trailing dimension size.

        Returns:
            The reshaped data. Shape is (num_instances, num_bodies, data_dim).
        """
        # The view returns data ordered as (body0_env0, body0_env1, ..., body1_env0, body1_env1, ...)
        # We need (num_instances, num_bodies, data_dim), so stride through the flat float32 data.
        # Use data.device for the strided view (PhysX returns masses/coms/inertias on CPU),
        # then clone to self.device (handles both contiguity and device transfer).
        element_size = wp.types.type_size_in_bytes(wp.float32)
        row_size = element_size * data_dim
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_instances, self.num_bodies, data_dim),
            dtype=wp.float32,
            strides=(row_size, self.num_instances * row_size, element_size),
            device=data.device,
        )
        return wp.clone(strided_view, self.device)

    def _get_pos_from_transform(self, transform: wp.array) -> ProxyArray:
        """Generates a position array from a transform array."""
        return wp.array(
            ptr=transform.ptr,
            shape=transform.shape,
            dtype=wp.vec3f,
            strides=transform.strides,
            device=self.device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> ProxyArray:
        """Generates a quaternion array from a transform array."""
        return wp.array(
            ptr=transform.ptr + 3 * 4,
            shape=transform.shape,
            dtype=wp.quatf,
            strides=transform.strides,
            device=self.device,
        )

    def _get_lin_vel_from_spatial_vector(self, spatial_vector: wp.array) -> ProxyArray:
        """Generates a linear velocity array from a spatial vector array."""
        return wp.array(
            ptr=spatial_vector.ptr,
            shape=spatial_vector.shape,
            dtype=wp.vec3f,
            strides=spatial_vector.strides,
            device=self.device,
        )

    def _get_ang_vel_from_spatial_vector(self, spatial_vector: wp.array) -> ProxyArray:
        """Generates an angular velocity array from a spatial vector array."""
        return wp.array(
            ptr=spatial_vector.ptr + 3 * 4,
            shape=spatial_vector.shape,
            dtype=wp.vec3f,
            strides=spatial_vector.strides,
            device=self.device,
        )

    """
    Deprecated properties.
    """

    @property
    def default_body_state(self) -> ProxyArray:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame.

        The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities
        are of the center of mass frame. Shape is (num_instances, num_bodies, 13).
        """
        warnings.warn(
            "Reading the body state directly is deprecated since IsaacLab 3.0 and will be removed in a future version. "
            "Please use the default_body_pose and default_body_vel properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._default_body_state is None:
            self._default_body_state = wp.zeros(
                (self.num_instances, self.num_bodies), dtype=shared_kernels.vec13f, device=self.device
            )
        wp.launch(
            shared_kernels.concat_body_pose_and_vel_to_state,
            dim=(self.num_instances, self.num_bodies),
            inputs=[
                self._default_body_pose,
                self._default_body_vel,
            ],
            outputs=[
                self._default_body_state,
            ],
            device=self.device,
        )
        if self._default_body_state_ta is None:
            self._default_body_state_ta = ProxyArray(self._default_body_state)
        return self._default_body_state_ta

    @property
    def body_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_com_vel_w`."""
        warnings.warn(
            "The `body_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_vel_w,
                ],
                outputs=[
                    self._body_state_w.data,
                ],
                device=self.device,
            )
            self._body_state_w.timestamp = self._sim_timestamp

        if self._body_state_w_ta is None:
            self._body_state_w_ta = ProxyArray(self._body_state_w.data)
        return self._body_state_w_ta

    @property
    def body_link_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_link_pose_w` and :attr:`body_link_vel_w`."""
        warnings.warn(
            "The `body_link_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_link_pose_w` and "
            "`body_link_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_link_vel_w,
                ],
                outputs=[
                    self._body_link_state_w.data,
                ],
                device=self.device,
            )
            self._body_link_state_w.timestamp = self._sim_timestamp

        if self._body_link_state_w_ta is None:
            self._body_link_state_w_ta = ProxyArray(self._body_link_state_w.data)
        return self._body_link_state_w_ta

    @property
    def body_com_state_w(self) -> ProxyArray:
        """Deprecated, same as :attr:`body_com_pose_w` and :attr:`body_com_vel_w`."""
        warnings.warn(
            "The `body_com_state_w` property will be deprecated in IsaacLab 4.0. Please use `body_com_pose_w` and "
            "`body_com_vel_w` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.concat_body_pose_and_vel_to_state,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_com_pose_w,
                    self.body_com_vel_w,
                ],
                outputs=[
                    self._body_com_state_w.data,
                ],
                device=self.device,
            )
            self._body_com_state_w.timestamp = self._sim_timestamp

        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w.data)
        return self._body_com_state_w_ta
