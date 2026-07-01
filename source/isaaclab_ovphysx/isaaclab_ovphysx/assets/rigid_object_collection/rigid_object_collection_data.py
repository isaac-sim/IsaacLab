# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import BaseRigidObjectCollectionData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.buffers import reset_timestamps
from isaaclab.utils.math import normalize
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.physics import OvPhysxManager as SimulationManager


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
        **Pull-to-refresh model.** Properties pull fresh data from the OVPhysX tensor API on first access
        per timestamp and cache the result. This differs from Newton, where buffers are refreshed
        automatically by the simulation.

    .. note::
        **ProxyArray pointer stability.** Each :class:`ProxyArray` wrapper is created once and reused
        because the OVPhysX tensor API returns views into stable, pre-allocated GPU buffers whose device
        pointer does not change across simulation steps.
    """

    __backend_name__: str = "ovphysx"
    """The name of the backend for the rigid object collection data."""

    def __init__(
        self,
        root_view: dict[int, Any],
        num_bodies: int,
        device: str,
    ):
        """Initializes the rigid object data.

        Args:
            root_view: Fused TensorBinding dict, keyed by TensorType constant. Each value is a single
                :class:`TensorBinding` spanning all bodies in the collection.
            num_bodies: The number of bodies in the collection.
            device: The device used for processing.
        """
        super().__init__(root_view, num_bodies, device)
        # Store the bindings dict (the equivalent of the root view in PhysX).
        self._bindings = root_view
        self._binding_getter = None  # may be set externally after construction
        self.num_bodies = num_bodies
        self._num_bodies = num_bodies
        # Set initial time stamp
        self._sim_timestamp = 0.0
        self._is_primed = False
        # Body-major read scratch buffers (keyed by tensor_type). Allocated on the binding's own
        # device — pinned host for CPU-only bindings, GPU for the rest — so ``binding.read(scratch)``
        # never crosses devices.
        self._cpu_staging_buffers: dict[int, wp.array] = {}

        # Read num_instances from the LINK_POSE binding. The native fused multi-prim binding lays
        # elements out body-major-flat with ``shape == (N * B, 7)`` and ``count == N * B``. The
        # articulation-mode mock used by iface tests exposes an instance-major view directly with
        # ``shape == (N, B, 7)`` and ``count == N``. Dispatch via the binding's exposed shape.
        pose_binding = self._bindings[TT.LINK_POSE]
        if len(pose_binding.shape) >= 2 and pose_binding.shape[1] == num_bodies:
            self.num_instances = pose_binding.count
        else:
            self.num_instances = pose_binding.count // num_bodies
        self._num_instances = self.num_instances

        if SimulationManager._sim is not None and hasattr(SimulationManager._sim, "cfg"):
            gravity = SimulationManager._sim.cfg.gravity
        else:
            gravity = (0.0, 0.0, -9.81)

        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        if torch.linalg.norm(gravity_dir) > 0.0:
            gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self.num_instances, self.num_bodies, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_instances, self.num_bodies, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = ProxyArray(wp.from_torch(gravity_dir, dtype=wp.vec3f))
        self.FORWARD_VEC_B = ProxyArray(wp.from_torch(forward_vec, dtype=wp.vec3f))

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
            dt: The time step for the update [s]. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Prime the FD-dependent COM acceleration so the first read returns a sensible (zero) value.
        _ = self.body_com_acc_w

    def _reset_pose(self, from_link: bool = True) -> None:
        """Reset pose-dependent cached rigid object collection properties.

        Writing body poses moves the bodies, so the composite body state buffers go stale.

        Args:
            from_link: Set ``True`` when the body link poses were written so the derived body
                center-of-mass poses (:attr:`body_com_pose_w`) are also invalidated; set ``False`` when
                the center-of-mass poses were written directly so they are not clobbered. Defaults to True.
        """
        # The body com poses are derived from the body link poses, so only invalidate them when the link
        # poses were the quantity written (otherwise we would clobber the freshly-written com poses).
        # The composite body state buffers always go stale on a pose write.
        reset_timestamps(
            [
                self._body_com_pose_w if from_link else None,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
            ]
        )

    def _reset_velocity(self, from_com: bool = True) -> None:
        """Reset velocity-dependent cached rigid object collection properties.

        Writing body velocities changes the body velocities, so the composite body state buffers go stale.

        Args:
            from_com: Set ``True`` when the body center-of-mass velocities were written so the derived body
                link velocities (:attr:`body_link_vel_w`) are also invalidated; set ``False`` when the link
                velocities were written directly so they are not clobbered. Defaults to True.
        """
        # The body link velocities are derived from the body com velocities, so only invalidate them when
        # the com velocities were the quantity written (otherwise we would clobber the freshly-written value).
        # The composite body state buffers always go stale on a velocity write.
        reset_timestamps(
            [
                self._body_link_vel_w if from_com else None,
                self._body_state_w,
                self._body_link_state_w,
                self._body_com_state_w,
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
    def default_body_pose(self) -> ProxyArray:
        """Default body pose ``[pos, quat]`` in local environment frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        Set by :meth:`~RigidObjectCollection._process_cfg` during initialization.
        """
        if self._default_body_pose_ta is None:
            self._default_body_pose_ta = ProxyArray(self._default_body_pose)
        return self._default_body_pose_ta

    @default_body_pose.setter
    def default_body_pose(self, value: wp.array) -> None:
        self._default_body_pose.assign(value)

    @property
    def default_body_vel(self) -> ProxyArray:
        """Default body velocity ``[lin_vel, ang_vel]`` in local environment frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        Set by :meth:`~RigidObjectCollection._process_cfg` during initialization.
        """
        if self._default_body_vel_ta is None:
            self._default_body_vel_ta = ProxyArray(self._default_body_vel)
        return self._default_body_vel_ta

    @default_body_vel.setter
    def default_body_vel(self, value: wp.array) -> None:
        self._default_body_vel.assign(value)

    @property
    def default_body_state(self) -> ProxyArray:
        """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame.

        Deprecated. Use :attr:`default_body_pose` and :attr:`default_body_vel` instead.

        Shape is (num_instances, num_bodies), dtype = ``vec13f``.
        In torch this resolves to (num_instances, num_bodies, 13).
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
            self._default_body_state_ta = ProxyArray(self._default_body_state)
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
        return self._default_body_state_ta

    """
    Body state properties.
    """

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Body link pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        This quantity is the pose of the actor frame of the rigid body relative to
        the world. The orientation is provided in (x, y, z, w) format.
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            self._read_transform_binding(TT.LINK_POSE, self._body_link_pose_w)
            # Invalidate sliced sub-component proxies so they are rebuilt from the
            # updated buffer on next access.
            self._body_link_pos_w_ta = None
            self._body_link_quat_w_ta = None
        if self._body_link_pose_w_ta is None:
            self._body_link_pose_w_ta = ProxyArray(self._body_link_pose_w.data)
        return self._body_link_pose_w_ta

    @property
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        This quantity contains the linear and angular velocities of the actor frame
        of the rigid body relative to the world.
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
                outputs=[self._body_link_vel_w.data],
                device=self.device,
            )
            self._body_link_vel_w.timestamp = self._sim_timestamp
            self._body_link_lin_vel_w_ta = None
            self._body_link_ang_vel_w_ta = None
        if self._body_link_vel_w_ta is None:
            self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
        return self._body_link_vel_w_ta

    @property
    def body_com_pose_w(self) -> ProxyArray:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame [m, -].

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        This quantity is the pose of the center of mass frame of the rigid body
        relative to the world. The orientation is provided in (x, y, z, w) format.
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_com_pose_from_body_link_pose,
                dim=(self.num_instances, self.num_bodies),
                inputs=[
                    self.body_link_pose_w,
                    self.body_com_pose_b,
                ],
                outputs=[self._body_com_pose_w.data],
                device=self.device,
            )
            self._body_com_pose_w.timestamp = self._sim_timestamp
            self._body_com_pos_w_ta = None
            self._body_com_quat_w_ta = None
        if self._body_com_pose_w_ta is None:
            self._body_com_pose_w_ta = ProxyArray(self._body_com_pose_w.data)
        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        This quantity contains the linear and angular velocities of the rigid body's
        center of mass frame relative to the world.
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._read_spatial_vector_binding(TT.LINK_VELOCITY, self._body_com_vel_w)
            self._body_com_lin_vel_w_ta = None
            self._body_com_ang_vel_w_ta = None
        if self._body_com_vel_w_ta is None:
            self._body_com_vel_w_ta = ProxyArray(self._body_com_vel_w.data)
        return self._body_com_vel_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame [m/s², rad/s²].

        Shape is (num_instances, num_bodies), dtype = ``wp.spatial_vectorf``.
        In torch this resolves to (num_instances, num_bodies, 6).
        This quantity is the acceleration of the rigid bodies' center of mass frame relative
        to the world, derived by finite differencing consecutive COM velocities.
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            if self._previous_body_com_vel is None:
                self._previous_body_com_vel = wp.clone(self.body_com_vel_w.warp)
            wp.launch(
                shared_kernels.derive_body_acceleration_from_body_com_velocities,
                dim=(self.num_instances, self.num_bodies),
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
            self._body_com_lin_acc_w_ta = None
            self._body_com_ang_acc_w_ta = None
        if self._body_com_acc_w_ta is None:
            self._body_com_acc_w_ta = ProxyArray(self._body_com_acc_w.data)
        return self._body_com_acc_w_ta

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body link frames [m, -].

        Shape is (num_instances, num_bodies), dtype = ``wp.transformf``.
        In torch this resolves to (num_instances, num_bodies, 7).
        This quantity is the pose of the center of mass frame of the rigid body
        relative to the body's link frame. The orientation is provided in
        (x, y, z, w) format.
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            self._read_transform_binding(TT.BODY_COM_POSE, self._body_com_pose_b)
            self._body_com_pos_b_ta = None
            self._body_com_quat_b_ta = None
        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b.data)
        return self._body_com_pose_b_ta

    @property
    def body_mass(self) -> ProxyArray:
        """Mass of all bodies [kg].

        Shape is (num_instances, num_bodies), dtype = ``wp.float32``.
        In torch this resolves to (num_instances, num_bodies).
        """
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass.data)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Inertia tensor of all bodies, expressed at the center of mass [kg·m²].

        Shape is (num_instances, num_bodies, 9), dtype = ``wp.float32``.
        The 9 components are the row-major flatten of the 3×3 inertia matrix
        ``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)``.
        In torch this resolves to (num_instances, num_bodies, 9).
        """
        if self._body_inertia_ta is None:
            self._body_inertia_ta = ProxyArray(self._body_inertia.data)
        return self._body_inertia_ta

    """
    Deprecated state-concat properties.
    """

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
                inputs=[self.body_link_pose_w, self.body_com_vel_w],
                outputs=[self._body_state_w.data],
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
                inputs=[self.body_link_pose_w, self.body_link_vel_w],
                outputs=[self._body_link_state_w.data],
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
                inputs=[self.body_com_pose_w, self.body_com_vel_w],
                outputs=[self._body_com_state_w.data],
                device=self.device,
            )
            self._body_com_state_w.timestamp = self._sim_timestamp
        if self._body_com_state_w_ta is None:
            self._body_com_state_w_ta = ProxyArray(self._body_com_state_w.data)
        return self._body_com_state_w_ta

    """
    Sliced properties.
    """

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Positions of all bodies in simulation world frame [m].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        This quantity is the position of the rigid bodies' actor frame relative to
        the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.quatf``.
        In torch this resolves to (num_instances, num_bodies, 4).
        This quantity is the orientation of the rigid bodies' actor frame relative
        to the world.
        """
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        This quantity is the linear velocity of the rigid bodies' actor frame
        relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        This quantity is the angular velocity of the rigid bodies' actor frame
        relative to the world.
        """
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_link_lin_vel_b(self) -> ProxyArray:
        """Linear velocity of all bodies in their respective body (actor) frames [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
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
        """Angular velocity of all bodies in their respective body (actor) frames [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
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
    def body_com_pos_w(self) -> ProxyArray:
        """Positions of all bodies' center of mass in simulation world frame [m].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies), dtype = ``wp.quatf``.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies' center of mass in simulation world frame [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies' center of mass in simulation world frame [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_vel_b(self) -> ProxyArray:
        """Linear velocity of all bodies' center of mass in their respective body (actor) frames [m/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
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
        """Angular velocity of all bodies' center of mass in their respective body (actor) frames [rad/s].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
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

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Linear acceleration of all bodies' center of mass in simulation world frame [m/s²].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Angular acceleration of all bodies' center of mass in simulation world frame [rad/s²].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center of mass position of all of the bodies in their respective link frames [m].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_b
        if self._body_com_pos_b_ta is None:
            self._body_com_pos_b_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_b_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Orientation (x, y, z, w) of the principal axes of inertia of all of the bodies
        in their respective link frames.

        Shape is (num_instances, num_bodies), dtype = ``wp.quatf``.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_b
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_b_ta

    """
    Derived Properties.
    """

    @property
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction onto each body frame [-].

        Shape is (num_instances, num_bodies), dtype = ``wp.vec3f``.
        In torch this resolves to (num_instances, num_bodies, 3).
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
        """Yaw heading of each body frame [rad].

        Shape is (num_instances, num_bodies), dtype = ``wp.float32``.
        In torch this resolves to (num_instances, num_bodies).

        .. note::
            This quantity is computed by assuming that the forward-direction of each
            body frame is along the x-direction, i.e. :math:`(1, 0, 0)`.
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

    def _create_buffers(self) -> None:
        """Eagerly allocate every per-body TimestampedBuffer and the slots for
        cached :class:`ProxyArray` wrappers.

        Buffers use direct ``(num_instances, num_bodies, D)`` shapes, matching
        the fused binding output.  No flat+strided tricks are needed because the
        fused binding returns a contiguous ``(N, B, D)`` array directly.
        """
        super()._create_buffers()

        N = self.num_instances
        B = self.num_bodies

        # -- link frame w.r.t. world frame
        self._body_link_pose_w = TimestampedBuffer((N, B), self.device, wp.transformf)
        self._body_link_vel_w = TimestampedBuffer((N, B), self.device, wp.spatial_vectorf)
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer((N, B), self.device, wp.transformf)
        # -- com frame w.r.t. world frame
        self._body_com_pose_w = TimestampedBuffer((N, B), self.device, wp.transformf)
        self._body_com_vel_w = TimestampedBuffer((N, B), self.device, wp.spatial_vectorf)
        # -- combined state (cached, used by deprecated concat properties)
        self._body_state_w = TimestampedBuffer((N, B), self.device, shared_kernels.vec13f)
        self._body_link_state_w = TimestampedBuffer((N, B), self.device, shared_kernels.vec13f)
        self._body_com_state_w = TimestampedBuffer((N, B), self.device, shared_kernels.vec13f)
        # -- derived properties (in-body-frame velocities)
        self._body_link_lin_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._body_link_ang_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._body_com_lin_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._body_com_ang_vel_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        # -- derived properties (acceleration via finite differencing)
        self._body_com_acc_w = TimestampedBuffer((N, B), self.device, wp.spatial_vectorf)
        # Holds the previous-step COM velocity for FD; initialised lazily on first access.
        self._previous_body_com_vel: wp.array | None = None
        # -- derived properties (projected gravity and heading)
        self._projected_gravity_b = TimestampedBuffer((N, B), self.device, wp.vec3f)
        self._heading_w = TimestampedBuffer((N, B), self.device, wp.float32)

        # -- Body properties: mass (N, B) and inertia (N, B, 9).
        # Initialised eagerly from the CPU-only bindings.
        self._body_mass = TimestampedBuffer((N, B), self.device, wp.float32)
        self._body_inertia = TimestampedBuffer((N, B, 9), self.device, wp.float32)

        # Pinned CPU staging buffers used by mass/com/inertia setters.
        pinned = self.device != "cpu"
        self._cpu_body_mass = wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_body_coms = wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=pinned)
        self._cpu_body_inertia = wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=pinned)

        # Eagerly read mass and inertia (CPU-only bindings) at construction time.
        # The native fused binding returns body-major flat data ``(N*B[, D])``;
        # the articulation-mode mock returns instance-major ``(N, B[, D])``.
        # In either case we reshape on the CPU into instance-major numpy arrays.
        def _read_cpu(tensor_type, trailing_dim=None):
            binding = self._get_binding(tensor_type)
            if binding is None:
                return None
            np_buf = np.zeros(binding.shape, dtype=np.float32)
            binding.read(np_buf)
            if binding.count == N:
                # Mock fast-path: already (N, B[, D]).
                return np_buf
            # Native fused path: body-major flat -> instance-major via reshape+transpose.
            if trailing_dim is None:
                return np_buf.reshape(B, N).T.copy()
            return np_buf.reshape(B, N, trailing_dim).transpose(1, 0, 2).copy()

        np_mass = _read_cpu(TT.BODY_MASS)
        if np_mass is not None:
            wp.copy(self._body_mass.data, wp.from_numpy(np_mass, dtype=wp.float32, device=self.device))
            self._body_mass.timestamp = self._sim_timestamp

        np_inertia = _read_cpu(TT.BODY_INERTIA, trailing_dim=9)
        if np_inertia is not None:
            wp.copy(
                self._body_inertia.data,
                wp.from_numpy(np_inertia, dtype=wp.float32, device=self.device),
            )
            self._body_inertia.timestamp = self._sim_timestamp

        # -- Defaults (allocated here, filled by _process_cfg after __init__).
        # Zero-initialized buffers; populated by RigidObjectCollection._process_cfg.
        self._default_body_pose = wp.zeros((N, B), dtype=wp.transformf, device=self.device)
        self._default_body_vel = wp.zeros((N, B), dtype=wp.spatial_vectorf, device=self.device)

        # Initialize ProxyArray wrappers.
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create pinned :class:`ProxyArray` wrappers for all data buffers.

        This is called once from :meth:`_create_buffers` during initialization.
        OVPhysX tensor API buffers have stable GPU pointers across simulation steps,
        so no rebinding is needed (unlike Newton).
        """
        # Defaults
        self._default_body_pose_ta: ProxyArray | None = None
        self._default_body_vel_ta: ProxyArray | None = None
        # Body state (timestamped)
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_b_ta: ProxyArray | None = None
        # Body properties
        self._body_mass_ta: ProxyArray | None = None
        self._body_inertia_ta: ProxyArray | None = None
        # Derived properties (in-body-frame velocities)
        self._body_link_lin_vel_b_ta: ProxyArray | None = None
        self._body_link_ang_vel_b_ta: ProxyArray | None = None
        self._body_com_lin_vel_b_ta: ProxyArray | None = None
        self._body_com_ang_vel_b_ta: ProxyArray | None = None
        # Derived properties (FD acceleration)
        self._body_com_acc_w_ta: ProxyArray | None = None
        self._body_com_lin_acc_w_ta: ProxyArray | None = None
        self._body_com_ang_acc_w_ta: ProxyArray | None = None
        # Derived properties (projected gravity and heading)
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._heading_w_ta: ProxyArray | None = None
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
        # Sliced properties (body com in body frame)
        self._body_com_pos_b_ta: ProxyArray | None = None
        self._body_com_quat_b_ta: ProxyArray | None = None
        # Deprecated state-concat properties
        self._default_body_state: wp.array | None = None
        self._default_body_state_ta: ProxyArray | None = None
        self._body_state_w_ta: ProxyArray | None = None
        self._body_link_state_w_ta: ProxyArray | None = None
        self._body_com_state_w_ta: ProxyArray | None = None

    """
    Helpers.
    """

    def _get_binding(self, tensor_type: int):
        """Return the binding for the given tensor type, or None.

        Args:
            tensor_type: The TensorType constant identifying which simulation buffer.

        Returns:
            The cached :class:`TensorBinding`, or ``None`` if not available.
        """
        b = self._bindings.get(tensor_type)
        if b is not None:
            return b
        if self._binding_getter is not None:
            b = self._binding_getter(tensor_type)
            if b is not None:
                self._bindings[tensor_type] = b
            return b
        return None

    def _read_view_scratch(self, tensor_type: int, binding) -> wp.array:
        """Return a cached body-major scratch float32 buffer matching ``binding.shape``.

        Allocated on the binding's own device (GPU bindings → GPU, CPU-only
        bindings → pinned host) so that ``binding.read(scratch)`` never crosses
        devices.  The scratch buffer is body-major flat
        ``(num_bodies * num_instances[, D])`` and is reshaped into instance-major
        ``(N, B[, D])`` by :meth:`_reshape_view_to_data_2d` /
        :meth:`_reshape_view_to_data_3d` before being copied into the user-facing
        timestamped buffer.

        Args:
            tensor_type: TensorType key (used as cache key).
            binding: The fused :class:`TensorBinding` whose shape the scratch
                must match.

        Returns:
            Cached body-major scratch buffer for reads.
        """
        scratch = self._cpu_staging_buffers.get(tensor_type)
        if scratch is not None:
            return scratch
        binding_device = "cpu" if tensor_type in TT._CPU_ONLY_TYPES else self.device
        pinned = binding_device == "cpu" and self.device != "cpu"
        if pinned:
            scratch = wp.zeros(binding.shape, dtype=wp.float32, device="cpu", pinned=True)
        else:
            scratch = wp.zeros(binding.shape, dtype=wp.float32, device=binding_device)
        self._cpu_staging_buffers[tensor_type] = scratch
        return scratch

    def _read_binding_into_instance_major(self, tensor_type: int, buf: TimestampedBuffer, floats_per_elem: int) -> None:
        """Read a fused binding into the instance-major ``buf.data``.

        The native fused multi-prim binding returns data in body-major flat order
        ``(body0_env0, body0_env1, ..., body1_env0, ...)`` with
        ``binding.count == num_instances * num_bodies``.  The articulation-mode
        mock used by iface tests instead exposes a directly instance-major view
        with ``binding.count == num_instances`` and shape ``(N, B[, D])``.

        This method dispatches to the right path:

        * **Mock fast-path** (``binding.count == num_instances``): a float32 view
          of the destination buffer is filled directly via ``binding.read()``.
        * **Native fused path** (``binding.count == num_instances * num_bodies``):
          ``binding.read()`` fills a body-major scratch, which is then reshaped
          into instance-major order via :meth:`_reshape_view_to_data_2d` (for
          single-element-per-body fields like mass) or
          :meth:`_reshape_view_to_data_3d` (for fields with a trailing
          ``floats_per_elem`` dimension), and the result is copied into the
          destination buffer.

        Args:
            tensor_type: TensorType key identifying the binding.
            buf: Timestamped buffer to refresh.  ``buf.data`` is
                ``(num_instances, num_bodies)`` for single-element-per-body
                fields (e.g. mass), or ``(num_instances, num_bodies, ...)`` for
                multi-component fields.
            floats_per_elem: Number of trailing ``float32`` elements per body
                (e.g. 7 for transformf, 6 for spatial_vectorf, 9 for inertia).
                Pass 1 for plain scalar fields like mass.
        """
        if buf.timestamp >= self._sim_timestamp:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return

        B = self.num_bodies

        # Disambiguate via the binding's exposed shape: the articulation-mode
        # mock returns a directly instance-major view ``(N, B[, D])`` while the
        # native fused multi-prim binding lays elements body-major-flat with
        # ``shape == (N * B[, D])``.
        is_mock_layout = len(binding.shape) >= 2 and binding.shape[1] == B

        if is_mock_layout:
            if buf.data.dtype == wp.float32:
                view = buf.data
            else:
                view = wp.array(
                    ptr=buf.data.ptr,
                    shape=binding.shape,
                    dtype=wp.float32,
                    device=str(buf.data.device),
                    copy=False,
                )
            binding.read(view)
            buf.timestamp = self._sim_timestamp
            return

        # Native fused path: read body-major scratch then strided-view reshape.
        scratch = self._read_view_scratch(tensor_type, binding)
        binding.read(scratch)
        if floats_per_elem <= 1:
            reshaped = self._reshape_view_to_data_2d(scratch)
        else:
            reshaped = self._reshape_view_to_data_3d(scratch, floats_per_elem)
        # Copy into buf.data, reinterpreting structured-dtype buffers as float32.
        if buf.data.dtype == wp.float32:
            dst_view = buf.data
        else:
            dst_view = wp.array(
                ptr=buf.data.ptr,
                shape=reshaped.shape,
                dtype=wp.float32,
                device=str(buf.data.device),
                copy=False,
            )
        wp.copy(dst_view, reshaped)
        buf.timestamp = self._sim_timestamp

    def _read_transform_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a pose binding (``wp.transformf`` buffer), skipping if fresh.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped :class:`wp.transformf` buffer to refresh.
        """
        self._read_binding_into_instance_major(tensor_type, buf, floats_per_elem=7)

    def _read_spatial_vector_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a velocity binding (``wp.spatial_vectorf`` buffer), skipping if fresh.

        Args:
            tensor_type: TensorType key.
            buf: Timestamped :class:`wp.spatial_vectorf` buffer to refresh.
        """
        self._read_binding_into_instance_major(tensor_type, buf, floats_per_elem=6)

    def _reshape_view_to_data_2d(self, data: wp.array) -> wp.array:
        """Reshape body-major flat data into instance-major ``(num_instances, num_bodies)``.

        The native fused binding lays elements out body-major:
        ``(body0_env0, body0_env1, ..., body1_env0, body1_env1, ...)``.  This helper
        constructs a strided view that traverses the data in instance-major order
        ``(env0_body0, env0_body1, ..., env1_body0, ...)``, then clones it onto
        :attr:`device` for contiguity and (when needed) cross-device transfer.

        Args:
            data: Body-major flat buffer.  Shape is ``(num_bodies * num_instances,)``
                with any single-element dtype.

        Returns:
            Contiguous instance-major buffer with shape ``(num_instances, num_bodies)``
            on :attr:`device`.
        """
        element_size = wp.types.type_size_in_bytes(data.dtype)
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_instances, self.num_bodies),
            dtype=data.dtype,
            strides=(element_size, self.num_instances * element_size),
            device=str(data.device),
        )
        return wp.clone(strided_view, self.device)

    def _reshape_view_to_data_3d(self, data: wp.array, data_dim: int) -> wp.array:
        """Reshape body-major flat data into instance-major ``(num_instances, num_bodies, data_dim)``.

        Companion of :meth:`_reshape_view_to_data_2d` for fields with a trailing
        per-body dimension (e.g. pose has 7, spatial velocity has 6, inertia has 9).

        Args:
            data: Body-major flat buffer with shape ``(num_bodies * num_instances, data_dim)``
                or ``(num_bodies * num_instances,)`` reinterpreted as ``data_dim``-wide rows.
            data_dim: Trailing per-body dimension size.

        Returns:
            Contiguous instance-major buffer with shape
            ``(num_instances, num_bodies, data_dim)`` on :attr:`device`.
        """
        element_size = wp.types.type_size_in_bytes(wp.float32)
        row_size = element_size * data_dim
        strided_view = wp.array(
            ptr=data.ptr,
            shape=(self.num_instances, self.num_bodies, data_dim),
            dtype=wp.float32,
            strides=(row_size, self.num_instances * row_size, element_size),
            device=str(data.device),
        )
        return wp.clone(strided_view, self.device)

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
