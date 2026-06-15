# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed RigidObjectData implementation."""

from __future__ import annotations

import math
import warnings

import torch
import warp as wp

from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.buffers import reset_timestamps
from isaaclab.utils.math import normalize
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.physics import OvPhysxManager as SimulationManager


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

    .. note::
        **Pull-to-refresh model.** Properties pull fresh data from the PhysX tensor API on first
        access per timestamp and cache the result. This differs from Newton, where buffers are
        refreshed automatically by the simulation.

    .. note::
        **ProxyArray pointer stability.** Each :class:`ProxyArray` wrapper is created once and
        reused because the PhysX tensor API returns views into stable, pre-allocated GPU buffers
        whose device pointer does not change across simulation steps.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object data."""

    def __init__(
        self,
        bindings: dict,
        device: str,
        check_shapes: bool = True,
    ):
        """Initializes the rigid object data.

        Args:
            bindings: The OVPhysX tensor bindings dict keyed by tensor-type constant.
                ``num_instances`` is read from ``bindings[RIGID_BODY_POSE].count`` and
                ``num_bodies`` is fixed at 1; ``body_names`` is set by
                :meth:`~isaaclab_ovphysx.assets.RigidObject._initialize_impl`.
            device: The device used for processing.
            check_shapes: Whether to enforce internal shape/dtype invariants on
                lazy reads. Defaults to ``True``; production callers thread this
                from :attr:`~isaaclab.assets.AssetBaseCfg.disable_shape_checks`.
        """
        super().__init__(bindings, device)
        # Set the tensor bindings (OVPhysX exposes per-tensor-type bindings rather than a single view).
        self._bindings = bindings
        self._check_shapes = check_shapes
        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False
        root_pose = self._bindings[TT.RIGID_BODY_POSE]
        self._num_instances = root_pose.count
        self._num_bodies = 1

        if SimulationManager._sim is not None and hasattr(SimulationManager._sim, "cfg"):
            gravity = SimulationManager._sim.cfg.gravity
        else:
            gravity = (0.0, 0.0, -9.81)

        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        # When gravity is disabled (cfg.gravity == (0, 0, 0)), normalize() would NaN.
        if torch.linalg.norm(gravity_dir) > 0.0:
            gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self._num_instances, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._num_instances, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = ProxyArray(wp.from_torch(gravity_dir, dtype=wp.vec3f))
        self.FORWARD_VEC_B = ProxyArray(wp.from_torch(forward_vec, dtype=wp.vec3f))

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
            dt: The time step for the update [s]. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the body com acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.body_com_acc_w

    def _reset_pose(self, from_link: bool = True) -> None:
        """Reset pose-dependent cached rigid object properties.

        Writing a root pose moves the body, so the composite root state buffers go stale.

        Args:
            from_link: Set ``True`` when the root link pose was written so the derived root
                center-of-mass pose (:attr:`root_com_pose_w`) is also invalidated; set ``False`` when
                the center-of-mass pose was written directly so it is not clobbered. Defaults to True.
        """
        # The root com pose is derived from the root link pose, so only invalidate it when the link
        # pose was the quantity written (otherwise we would clobber the freshly-written com pose).
        # The composite root state buffers always go stale on a pose write.
        reset_timestamps(
            [
                self._root_com_pose_w if from_link else None,
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
            ]
        )

    def _reset_velocity(self, from_com: bool = True) -> None:
        """Reset velocity-dependent cached rigid object properties.

        Writing a root velocity changes the body velocities, so the composite root state buffers go stale.

        Args:
            from_com: Set ``True`` when the root center-of-mass velocity was written so the derived root
                link velocity (:attr:`root_link_vel_w`) is also invalidated; set ``False`` when the link
                velocity was written directly so it is not clobbered. Defaults to True.
        """
        # The root link velocity is derived from the root com velocity, so only invalidate it when the
        # com velocity was the quantity written (otherwise we would clobber the freshly-written value).
        # The composite root state buffers always go stale on a velocity write.
        reset_timestamps(
            [
                self._root_link_vel_w if from_com else None,
                self._root_state_w,
                self._root_link_state_w,
                self._root_com_state_w,
            ]
        )

    """
    Names.
    """

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    """
    Defaults.
    """

    @property
    def default_root_pose(self) -> ProxyArray:
        """Default root pose ``[pos, quat]`` in simulation world frame [m, -].
        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).

        Populated from :attr:`RigidObjectCfg.init_state` during initialisation.
        """
        if self._default_root_pose_ta is None:
            self._default_root_pose_ta = ProxyArray(self._default_root_pose)
        return self._default_root_pose_ta

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
    def default_root_vel(self) -> ProxyArray:
        """Default root velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].
        Shape is (num_instances,), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, 6).

        Populated from :attr:`RigidObjectCfg.init_state` during initialisation.
        """
        if self._default_root_vel_ta is None:
            self._default_root_vel_ta = ProxyArray(self._default_root_vel)
        return self._default_root_vel_ta

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
    def root_link_pose_w(self) -> ProxyArray:
        """Root link pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        This quantity is the pose of the actor frame of the root rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            self._read_binding_into(TT.RIGID_BODY_POSE, self._root_link_pose_w.data)
            self._root_link_pose_w.timestamp = self._sim_timestamp
        if self._root_link_pose_w_ta is None:
            self._root_link_pose_w_ta = ProxyArray(self._root_link_pose_w.data)
        return self._root_link_pose_w_ta

    @property
    def root_link_vel_w(self) -> ProxyArray:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_root_link_vel_from_root_com_vel,
                dim=self._num_instances,
                inputs=[
                    self.root_com_vel_w,
                    self.root_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[self._root_link_vel_w.data],
                device=self.device,
            )
            self._root_link_vel_w.timestamp = self._sim_timestamp
        if self._root_link_vel_w_ta is None:
            self._root_link_vel_w_ta = ProxyArray(self._root_link_vel_w.data)
        return self._root_link_vel_w_ta

    @property
    def root_com_pose_w(self) -> ProxyArray:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame [m, -].

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

        if self._root_com_pose_w_ta is None:
            self._root_com_pose_w_ta = ProxyArray(self._root_com_pose_w.data)
        return self._root_com_pose_w_ta

    @property
    def root_com_vel_w(self) -> ProxyArray:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances,), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 6).
        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._read_binding_into(TT.RIGID_BODY_VELOCITY, self._root_com_vel_w.data)
            self._root_com_vel_w.timestamp = self._sim_timestamp
        if self._root_com_vel_w_ta is None:
            self._root_com_vel_w_ta = ProxyArray(self._root_com_vel_w.data)
        return self._root_com_vel_w_ta

    """
    Body state properties.
    """

    @property
    def body_mass(self) -> ProxyArray:
        """Mass of all bodies [kg].

        Shape is (num_instances, 1), dtype = wp.float32.
        In torch this resolves to (num_instances, 1).
        """
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Inertia tensor of all bodies, expressed at the center of mass [kg·m²].

        Shape is (num_instances, 1, 9), dtype = wp.float32. The 9 components are the row-major
        flatten of the 3×3 inertia matrix ``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)``.
        In torch this resolves to (num_instances, 1, 9).
        """
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia)
        return self._body_inertia_ta

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, 1), dtype = wp.transformf. In torch this resolves to (num_instances, 1, 7).
        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        parent = self.root_link_pose_w
        if self._body_link_pose_w_ta is None:
            self._body_link_pose_w_ta = ProxyArray(parent.warp.reshape((self._num_instances, 1)))
        return self._body_link_pose_w_ta

    @property
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, 1), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 1, 6).
        This quantity contains the linear and angular velocities of the body's link (actor) frame
        relative to the world.
        """
        parent = self.root_link_vel_w
        if self._body_link_vel_w_ta is None:
            self._body_link_vel_w_ta = ProxyArray(parent.warp.reshape((self._num_instances, 1)))
        return self._body_link_vel_w_ta

    @property
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.transformf. In torch this resolves to (num_instances, 1, 7).
        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (x, y, z, w) format.
        """
        parent = self.root_com_pose_w
        if self._body_com_pose_w_ta is None:
            self._body_com_pose_w_ta = ProxyArray(parent.warp.reshape((self._num_instances, 1)))
        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, 1), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 1, 6).
        This quantity contains the linear and angular velocities of the body's center of mass frame
        relative to the world.
        """
        parent = self.root_com_vel_w
        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(parent.warp.reshape((self._num_instances, 1)))
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame [m/s², rad/s²].

        Shape is (num_instances, 1), dtype = wp.spatial_vectorf. In torch this resolves to (num_instances, 1, 6).
        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            if self._previous_body_com_vel is None:
                self._previous_body_com_vel = wp.clone(self.body_com_vel_w.warp)
            wp.launch(
                shared_kernels.derive_body_acceleration_from_body_com_velocities,
                dim=(self._num_instances, 1),
                device=self.device,
                inputs=[
                    self.body_com_vel_w.warp,
                    SimulationManager.get_physics_dt(),
                    self._previous_body_com_vel,
                ],
                outputs=[
                    self._body_com_acc_w.data,
                ],
            )
            self._body_com_acc_w.timestamp = self._sim_timestamp
        if self._body_com_acc_w_ta is None:
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
        return self._body_com_acc_w_ta

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.

        Shape is (num_instances, 1), dtype = wp.transformf. In torch this resolves to (num_instances, 1, 7).
        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            self._read_binding_into(TT.RIGID_BODY_COM_POSE, self._body_com_pose_b.data)
            self._body_com_pose_b.timestamp = self._sim_timestamp

        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> ProxyArray:
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
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b.data)
        return self._projected_gravity_b_ta

    @property
    def heading_w(self) -> ProxyArray:
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
        if self._heading_w_ta is None:
            self._heading_w_ta = ProxyArray(self._heading_w.data)
        return self._heading_w_ta

    @property
    def root_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in base frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root link frame relative to the world,
        expressed in the root link's actor frame.
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
        if self._root_link_lin_vel_b_ta is None:
            self._root_link_lin_vel_b_ta = ProxyArray(self._root_link_lin_vel_b.data)
        return self._root_link_lin_vel_b_ta

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in base frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the root link frame relative to the world,
        expressed in the root link's actor frame.
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
        if self._root_link_ang_vel_b_ta is None:
            self._root_link_ang_vel_b_ta = ProxyArray(self._root_link_ang_vel_b.data)
        return self._root_link_ang_vel_b_ta

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        """Root center of mass linear velocity in base frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root center of mass frame relative to the world,
        expressed in the root link's actor frame.
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
        if self._root_com_lin_vel_b_ta is None:
            self._root_com_lin_vel_b_ta = ProxyArray(self._root_com_lin_vel_b.data)
        return self._root_com_lin_vel_b_ta

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        """Root center of mass angular velocity in base frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the root center of mass frame relative to the world,
        expressed in the root link's actor frame.
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
        if self._root_com_ang_vel_b_ta is None:
            self._root_com_ang_vel_b_ta = ProxyArray(self._root_com_ang_vel_b.data)
        return self._root_com_ang_vel_b_ta

    """
    Sliced properties.
    """

    @property
    def root_link_pos_w(self) -> ProxyArray:
        """Root link position in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        parent = self.root_link_pose_w
        if self._root_link_pos_w_ta is None:
            self._root_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_link_pos_w_ta

    @property
    def root_link_quat_w(self) -> ProxyArray:
        """Root link orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        This quantity is the orientation of the actor frame of the root rigid body.
        """
        parent = self.root_link_pose_w
        if self._root_link_quat_w_ta is None:
            self._root_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_link_quat_w_ta

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root linear velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        parent = self.root_link_vel_w
        if self._root_link_lin_vel_w_ta is None:
            self._root_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_link_lin_vel_w_ta

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root link angular velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        parent = self.root_link_vel_w
        if self._root_link_ang_vel_w_ta is None:
            self._root_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_link_ang_vel_w_ta

    @property
    def root_com_pos_w(self) -> ProxyArray:
        """Root center of mass position in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the position of the center of mass frame of the root rigid body relative to the world.
        """
        parent = self.root_com_pose_w
        if self._root_com_pos_w_ta is None:
            self._root_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_com_pos_w_ta

    @property
    def root_com_quat_w(self) -> ProxyArray:
        """Root center of mass orientation (x, y, z, w) in simulation world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        This quantity is the orientation of the principal axes of inertia of the root rigid body relative to the world.
        """
        parent = self.root_com_pose_w
        if self._root_com_quat_w_ta is None:
            self._root_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_com_quat_w_ta

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        """Root center of mass linear velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        parent = self.root_com_vel_w
        if self._root_com_lin_vel_w_ta is None:
            self._root_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_com_lin_vel_w_ta

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        """Root center of mass angular velocity in simulation world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        parent = self.root_com_vel_w
        if self._root_com_ang_vel_w_ta is None:
            self._root_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_com_ang_vel_w_ta

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.quatf. In torch this resolves to (num_instances, 1, 4).
        This quantity is the orientation of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the linear velocity of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the angular velocity of the rigid bodies' actor frame relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies' center of mass in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the position of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.quatf. In torch this resolves to (num_instances, 1, 4).
        This quantity is the orientation of the principal axes of inertia of the rigid bodies.
        """
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies in simulation world frame.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames.

        Shape is (num_instances, 1), dtype = wp.vec3f. In torch this resolves to (num_instances, 1, 3).
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

        Shape is (num_instances, 1), dtype = wp.quatf. In torch this resolves to (num_instances, 1, 4).
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
        self._root_link_pose_w = TimestampedBuffer((self._num_instances), self.device, wp.transformf)
        self._root_link_vel_w = TimestampedBuffer((self._num_instances), self.device, wp.spatial_vectorf)
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer((self._num_instances, 1), self.device, wp.transformf)
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer((self._num_instances), self.device, wp.transformf)
        self._root_com_vel_w = TimestampedBuffer((self._num_instances), self.device, wp.spatial_vectorf)
        self._body_com_acc_w = TimestampedBuffer((self._num_instances, 1), self.device, wp.spatial_vectorf)
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer((self._num_instances), self.device, shared_kernels.vec13f)
        self._root_link_state_w = TimestampedBuffer((self._num_instances), self.device, shared_kernels.vec13f)
        self._root_com_state_w = TimestampedBuffer((self._num_instances), self.device, shared_kernels.vec13f)
        # -- derived properties (these are cached to avoid repeated memory allocations)
        self._projected_gravity_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._heading_w = TimestampedBuffer((self._num_instances), self.device, wp.float32)
        self._root_link_lin_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_link_ang_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_com_lin_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)
        self._root_com_ang_vel_b = TimestampedBuffer((self._num_instances), self.device, wp.vec3f)

        # -- Default state
        self._default_root_pose = wp.zeros((self._num_instances), dtype=wp.transformf, device=self.device)
        self._default_root_vel = wp.zeros((self._num_instances), dtype=wp.spatial_vectorf, device=self.device)
        self._default_root_state = None

        # -- Previous body com velocity
        self._previous_body_com_vel = None

        # -- Pinned-host staging buffers for CPU-only bindings on a non-CPU sim
        # (lazily allocated, keyed by tensor type).
        self._cpu_staging_buffers: dict[int, wp.array] = {}

        # -- Body properties (semi-static; read once from CPU-only bindings).
        # The wheel exposes ``RIGID_BODY_MASS`` as ``(N,)`` and ``RIGID_BODY_INERTIA`` as ``(N, 9)``;
        # the ``BaseRigidObjectData`` contract is ``(N, 1)`` and ``(N, 1, 9)`` respectively, so we
        # read into a flat buffer and reshape (zero-copy) after the read.
        mass_binding = self._bindings[TT.RIGID_BODY_MASS]
        inertia_binding = self._bindings[TT.RIGID_BODY_INERTIA]
        self._body_mass = wp.zeros(mass_binding.shape, dtype=wp.float32, device=self.device)
        self._body_inertia = wp.zeros(inertia_binding.shape, dtype=wp.float32, device=self.device)
        self._read_binding_into(TT.RIGID_BODY_MASS, self._body_mass)
        self._read_binding_into(TT.RIGID_BODY_INERTIA, self._body_inertia)
        self._body_mass = self._body_mass.reshape((self._num_instances, 1))
        self._body_inertia = self._body_inertia.reshape((self._num_instances, 1, 9))

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
        self._default_root_pose_ta: ProxyArray | None = None
        self._default_root_vel_ta: ProxyArray | None = None
        # Root state (timestamped)
        self._root_link_pose_w_ta: ProxyArray | None = None
        self._root_link_vel_w_ta: ProxyArray | None = None
        self._root_com_pose_w_ta: ProxyArray | None = None
        self._root_com_vel_w_ta: ProxyArray | None = None
        # Body properties
        self._body_mass_ta: ProxyArray | None = None
        self._body_inertia_ta: ProxyArray | None = None
        # Body state (reshaped from root)
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._body_com_acc_w_ta: ProxyArray | None = None
        self._body_com_pose_b_ta: ProxyArray | None = None
        # Derived properties (timestamped)
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._heading_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_b_ta: ProxyArray | None = None
        self._root_link_ang_vel_b_ta: ProxyArray | None = None
        self._root_com_lin_vel_b_ta: ProxyArray | None = None
        self._root_com_ang_vel_b_ta: ProxyArray | None = None
        # Sliced properties (root link)
        self._root_link_pos_w_ta: ProxyArray | None = None
        self._root_link_quat_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_w_ta: ProxyArray | None = None
        self._root_link_ang_vel_w_ta: ProxyArray | None = None
        # Sliced properties (root com)
        self._root_com_pos_w_ta: ProxyArray | None = None
        self._root_com_quat_w_ta: ProxyArray | None = None
        self._root_com_lin_vel_w_ta: ProxyArray | None = None
        self._root_com_ang_vel_w_ta: ProxyArray | None = None
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
        self._default_root_state_ta: ProxyArray | None = None
        self._root_state_w_ta: ProxyArray | None = None
        self._root_link_state_w_ta: ProxyArray | None = None
        self._root_com_state_w_ta: ProxyArray | None = None
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

    """
    Internal helpers.
    """

    def _get_binding(self, tensor_type: int):
        """Return the binding for the given tensor type, or None."""
        return self._bindings.get(tensor_type)

    def _read_binding_into(self, tensor_type: int, dst: wp.array) -> None:
        """Read the OVPhysX TensorBinding for *tensor_type* into *dst*.

        Adapter that replaces PhysX's view-getter pattern: the wheel exposes
        ``binding.read(target)`` rather than a getter returning a wp.array, so
        we read into a flat float32 view of *dst*. CPU-only bindings on a
        non-CPU sim go through a lazily-allocated pinned-host wp.array to
        satisfy the wheel's device match.
        """
        binding = self._bindings[tensor_type]
        if self._check_shapes:
            dst_bytes = dst.size * wp.types.type_size_in_bytes(dst.dtype)
            binding_bytes = 4 * math.prod(binding.shape)
            assert dst_bytes >= binding_bytes, (
                f"_read_binding_into: dst buffer too small for binding {tensor_type!r} "
                f"({dst_bytes} B < {binding_bytes} B). Caller allocated dst with "
                f"shape={tuple(dst.shape)}, dtype={dst.dtype}; binding shape={tuple(binding.shape)}."
            )
        # Build a flat float32 view of dst matching the binding's shape.
        if dst.dtype == wp.float32:
            view = dst
        else:
            view = wp.array(
                ptr=dst.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(dst.device),
                copy=False,
            )
        if tensor_type in TT._CPU_ONLY_TYPES and str(view.device) != "cpu":
            staging = self._cpu_staging_buffers.get(tensor_type)
            if staging is None:
                staging = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
                self._cpu_staging_buffers[tensor_type] = staging
            binding.read(staging)
            wp.copy(view, staging)
        else:
            binding.read(view)

    def _get_pos_from_transform(self, transform: wp.array) -> wp.array:
        """Generates a position array from a transform array."""
        return wp.array(
            ptr=transform.ptr,
            shape=transform.shape,
            dtype=wp.vec3f,
            strides=transform.strides,
            device=self.device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> wp.array:
        """Generates a quaternion array from a transform array."""
        return wp.array(
            ptr=transform.ptr + 3 * 4,
            shape=transform.shape,
            dtype=wp.quatf,
            strides=transform.strides,
            device=self.device,
        )

    def _get_lin_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        """Generates a linear velocity array from a spatial vector array."""
        return wp.array(
            ptr=sv.ptr,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self.device,
        )

    def _get_ang_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        """Generates an angular velocity array from a spatial vector array."""
        return wp.array(
            ptr=sv.ptr + 3 * 4,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self.device,
        )

    """
    Deprecated properties.
    """

    @property
    def default_root_state(self) -> ProxyArray:
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
        if self._default_root_state_ta is None:
            self._default_root_state_ta = ProxyArray(self._default_root_state)
        return self._default_root_state_ta

    @property
    def root_state_w(self) -> ProxyArray:
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

        if self._root_state_w_ta is None:
            self._root_state_w_ta = ProxyArray(self._root_state_w.data)
        return self._root_state_w_ta

    @property
    def root_link_state_w(self) -> ProxyArray:
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

        if self._root_link_state_w_ta is None:
            self._root_link_state_w_ta = ProxyArray(self._root_link_state_w.data)
        return self._root_link_state_w_ta

    @property
    def root_com_state_w(self) -> ProxyArray:
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

        if self._root_com_state_w_ta is None:
            self._root_com_state_w_ta = ProxyArray(self._root_com_state_w.data)
        return self._root_com_state_w_ta

    @property
    def body_state_w(self) -> ProxyArray:
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
        if self._body_state_w_ta is None:
            self._body_state_w_ta = ProxyArray(self._root_state_w.data.reshape((self._num_instances, 1)))
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
        if self._body_link_state_w_ta is None:
            self._body_link_state_w_ta = ProxyArray(self._root_link_state_w.data.reshape((self._num_instances, 1)))
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
        # Access internal buffer directly to avoid cascading deprecation warnings from root_com_state_w
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
        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._root_com_state_w.data.reshape((self._num_instances, 1)))
        return self._body_com_state_w_ta
