# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed RigidObjectData implementation."""

from __future__ import annotations

import numpy as np
import warp as wp

from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets.kernels import _compose_root_com_pose


class RigidObjectData(BaseRigidObjectData):
    """OVPhysX implementation of :class:`~isaaclab.assets.BaseRigidObjectData`.

    Reads simulation state on demand through ``ovphysx`` ``TensorBinding``
    objects keyed by :data:`isaaclab_ovphysx.tensor_types.RIGID_BODY_ROOT_POSE`
    and friends. Buffers are timestamped so each binding is read at most once
    per sim step.

    This skeleton task only provides the constructor, count properties,
    ``update`` / ``_invalidate_caches`` lifecycle hooks, and ``_process_cfg``
    to populate default-state buffers from the asset's config. Property reads
    are layered on by subsequent tasks.
    """

    def __init__(self, bindings: dict, device: str):
        self._bindings = bindings
        self._device = device
        self._num_instances: int = 0
        self._num_bodies: int = 1
        self._is_primed: bool = False
        self._sim_time: float = 0.0
        self._timestamps: dict[str, float] = {}
        self._default_root_pose: wp.array | None = None
        self._default_root_velocity: wp.array | None = None
        # Cached TimestampedBuffers for root state (allocated lazily on first access).
        self._root_link_pose_w_buf: TimestampedBuffer | None = None
        self._root_link_vel_w_buf: TimestampedBuffer | None = None
        self._root_com_pose_w_buf: TimestampedBuffer | None = None
        self._root_com_vel_w_buf: TimestampedBuffer | None = None
        # Body-COM-in-body-frame buffer (shape (N,1), dtype=wp.transformf).
        self._body_com_pose_b_buf: TimestampedBuffer | None = None
        # Float32 view cache so binding.read() always sees the same object.
        self._read_view_cache: dict = {}
        # ProxyArray wrappers (created once from the underlying buffer.data).
        self._root_link_pose_w_ta: ProxyArray | None = None
        self._root_link_vel_w_ta: ProxyArray | None = None
        self._root_com_pose_w_ta: ProxyArray | None = None
        self._root_com_vel_w_ta: ProxyArray | None = None
        # Sliced view ProxyArrays.
        self._root_link_pos_w_ta: ProxyArray | None = None
        self._root_link_quat_w_ta: ProxyArray | None = None
        self._root_lin_vel_w_ta: ProxyArray | None = None
        self._root_ang_vel_w_ta: ProxyArray | None = None
        self._root_com_pos_w_ta: ProxyArray | None = None
        self._root_com_quat_w_ta: ProxyArray | None = None
        self._root_com_lin_vel_w_ta: ProxyArray | None = None
        self._root_com_ang_vel_w_ta: ProxyArray | None = None

    # --- counts -------------------------------------------------------
    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def device(self) -> str:
        return self._device

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
        self._is_primed = True

    # --- update / cache invalidation ----------------------------------
    def update(self, dt: float) -> None:
        """Advance the cached sim time. Per-property freshness checks happen
        lazily on access; nothing to do up front here."""
        self._sim_time += dt

    def _invalidate_caches(self, env_ids=None) -> None:
        """Coarse cache invalidation: reset every per-buffer timestamp so the
        next property access unconditionally re-reads from the binding. Called
        by :meth:`RigidObject.reset` and by every body-property setter on
        :class:`RigidObject`. The ``env_ids`` argument is accepted for parity
        with the articulation API but the caches stored on this single-body
        asset are full-tensor, so a fine-grained invalidation is not necessary
        here.
        """
        self._timestamps.clear()
        for buf in (
            self._root_link_pose_w_buf,
            self._root_link_vel_w_buf,
            self._root_com_pose_w_buf,
            self._root_com_vel_w_buf,
            self._body_com_pose_b_buf,
        ):
            if buf is not None:
                buf.timestamp = -1.0

    # --- defaults -----------------------------------------------------
    def _process_cfg(self, cfg: RigidObjectCfg) -> None:
        """Populate :attr:`_default_root_pose` and
        :attr:`_default_root_velocity` from ``cfg.init_state``. Called by
        :meth:`isaaclab_ovphysx.assets.RigidObject._initialize_impl` after
        ``_create_buffers``."""
        N = self._num_instances
        device = self._device
        # Pose: (px, py, pz, qx, qy, qz, qw)
        np_pose = np.tile(
            np.array(tuple(cfg.init_state.pos) + tuple(cfg.init_state.rot), dtype=np.float32),
            (N, 1),
        )
        self._default_root_pose = wp.zeros(N, dtype=wp.transformf, device=device)
        wp.copy(
            self._default_root_pose,
            wp.from_numpy(np_pose, dtype=wp.transformf, device=device),
        )
        # Velocity: (vx, vy, vz, wx, wy, wz)
        np_vel = np.tile(
            np.array(tuple(cfg.init_state.lin_vel) + tuple(cfg.init_state.ang_vel), dtype=np.float32),
            (N, 1),
        )
        self._default_root_velocity = wp.zeros(N, dtype=wp.spatial_vectorf, device=device)
        wp.copy(
            self._default_root_velocity,
            wp.from_numpy(np_vel, dtype=wp.spatial_vectorf, device=device),
        )

    # --- internal helpers: buffer allocation ----------------------------
    def _ensure_root_buffers(self) -> None:
        """Allocate root-state TimestampedBuffers on first use.

        Called lazily from root-state properties so that the buffers are
        only created after ``_num_instances`` and ``_device`` are set by
        the owning :class:`RigidObject`.
        """
        if self._root_link_pose_w_buf is not None:
            return
        N = self._num_instances
        dev = self._device
        self._root_link_pose_w_buf = TimestampedBuffer(N, dev, wp.transformf)
        self._root_link_vel_w_buf = TimestampedBuffer(N, dev, wp.spatial_vectorf)
        self._root_com_pose_w_buf = TimestampedBuffer(N, dev, wp.transformf)
        self._root_com_vel_w_buf = TimestampedBuffer(N, dev, wp.spatial_vectorf)
        # (N, 1) 2-D buffer for body_com_pose_b, required by _compose_root_com_pose.
        self._body_com_pose_b_buf = TimestampedBuffer((N, 1), dev, wp.transformf)

    # --- internal helpers: read from bindings --------------------------
    def _get_binding(self, tensor_type: int):
        """Return the binding for the given tensor type, or None."""
        return self._bindings.get(tensor_type)

    def _get_read_view(self, tensor_type: int, wp_array: wp.array, floats_per_elem: int = 0) -> wp.array | None:
        """Return a stable float32 view of *wp_array* sized to the binding shape.

        Cached so that binding.read() always sees the same object.
        """
        cache_key = (tensor_type, wp_array.ptr)
        cached = self._read_view_cache.get(cache_key)
        if cached is not None:
            return cached
        binding = self._get_binding(tensor_type)
        if binding is None:
            self._read_view_cache[cache_key] = None
            return None
        if floats_per_elem > 0:
            view = wp.array(
                ptr=wp_array.ptr,
                shape=binding.shape,
                dtype=wp.float32,
                device=str(wp_array.device),
                copy=False,
            )
        else:
            view = wp_array
        self._read_view_cache[cache_key] = view
        return view

    def _read_transform_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a pose binding (float32 view of transformf buffer), skipping if fresh."""
        if buf.timestamp >= self._sim_time:
            return
        view = self._get_read_view(tensor_type, buf.data, 7)
        if view is None:
            return
        self._get_binding(tensor_type).read(view)
        buf.timestamp = self._sim_time

    def _read_spatial_vector_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a velocity binding (float32 view of spatial_vectorf buffer), skipping if fresh."""
        if buf.timestamp >= self._sim_time:
            return
        view = self._get_read_view(tensor_type, buf.data, 6)
        if view is None:
            return
        self._get_binding(tensor_type).read(view)
        buf.timestamp = self._sim_time

    # --- internal helpers: slice extraction ----------------------------
    def _get_pos_from_transform(self, transform: wp.array) -> wp.array:
        return wp.array(
            ptr=transform.ptr,
            shape=transform.shape,
            dtype=wp.vec3f,
            strides=transform.strides,
            device=self._device,
        )

    def _get_quat_from_transform(self, transform: wp.array) -> wp.array:
        return wp.array(
            ptr=transform.ptr + 3 * 4,
            shape=transform.shape,
            dtype=wp.quatf,
            strides=transform.strides,
            device=self._device,
        )

    def _get_lin_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        return wp.array(
            ptr=sv.ptr,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self._device,
        )

    def _get_ang_vel_from_spatial_vector(self, sv: wp.array) -> wp.array:
        return wp.array(
            ptr=sv.ptr + 3 * 4,
            shape=sv.shape,
            dtype=wp.vec3f,
            strides=sv.strides,
            device=self._device,
        )

    # --- abstract property stubs (implemented by subsequent tasks) ----
    @property
    def default_root_pose(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def default_root_vel(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def default_root_state(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_pose_w(self) -> ProxyArray:
        """Root link pose ``[pos, quat]`` in simulation world frame [m, -].
        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).

        This quantity is the pose of the rigid body's actor frame relative to
        the world. The orientation is provided in (x, y, z, w) format.
        """
        self._ensure_root_buffers()
        self._read_transform_binding(TT.RIGID_BODY_ROOT_POSE, self._root_link_pose_w_buf)
        if self._root_link_pose_w_ta is None:
            self._root_link_pose_w_ta = ProxyArray(self._root_link_pose_w_buf.data)
        return self._root_link_pose_w_ta

    @property
    def root_link_vel_w(self) -> ProxyArray:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].
        Shape is (num_instances,), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, 6).

        This quantity contains the linear and angular velocities of the rigid
        body's actor frame relative to the world.
        """
        self._ensure_root_buffers()
        self._read_spatial_vector_binding(TT.RIGID_BODY_ROOT_VELOCITY, self._root_link_vel_w_buf)
        if self._root_link_vel_w_ta is None:
            self._root_link_vel_w_ta = ProxyArray(self._root_link_vel_w_buf.data)
        return self._root_link_vel_w_ta

    @property
    def root_com_pose_w(self) -> ProxyArray:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame [m, -].
        Shape is (num_instances,), dtype = wp.transformf.
        In torch this resolves to (num_instances, 7).

        This quantity is the pose of the rigid body's center of mass frame
        relative to the world. The orientation is provided in (x, y, z, w) format.
        """
        self._ensure_root_buffers()
        # Refresh body-frame COM offset from the RIGID_BODY_COM_POSE binding.
        if self._body_com_pose_b_buf.timestamp < self._sim_time:
            self._read_transform_binding(TT.RIGID_BODY_COM_POSE, self._body_com_pose_b_buf)
        if self._root_com_pose_w_buf.timestamp < self._sim_time:
            wp.launch(
                _compose_root_com_pose,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self._body_com_pose_b_buf.data],
                outputs=[self._root_com_pose_w_buf.data],
                device=self._device,
            )
            self._root_com_pose_w_buf.timestamp = self._sim_time
        if self._root_com_pose_w_ta is None:
            self._root_com_pose_w_ta = ProxyArray(self._root_com_pose_w_buf.data)
        return self._root_com_pose_w_ta

    @property
    def root_com_vel_w(self) -> ProxyArray:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame
        [m/s, rad/s].
        Shape is (num_instances,), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, 6).

        For a single rigid body the COM velocity equals the root link velocity
        read from the RIGID_BODY_ROOT_VELOCITY binding.
        """
        self._ensure_root_buffers()
        self._read_spatial_vector_binding(TT.RIGID_BODY_ROOT_VELOCITY, self._root_com_vel_w_buf)
        if self._root_com_vel_w_ta is None:
            self._root_com_vel_w_ta = ProxyArray(self._root_com_vel_w_buf.data)
        return self._root_com_vel_w_ta

    @property
    def root_state_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_state_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_state_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_link_pose_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_link_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_pose_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_state_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_link_state_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_state_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_acc_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_pose_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_mass(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_inertia(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def projected_gravity_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def heading_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_lin_vel_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_pos_w(self) -> ProxyArray:
        """Root link position in simulation world frame [m].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        parent = self.root_link_pose_w
        if self._root_link_pos_w_ta is None:
            self._root_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_link_pos_w_ta

    @property
    def root_link_quat_w(self) -> ProxyArray:
        """Root link orientation (x, y, z, w) in simulation world frame [-].
        Shape is (num_instances,), dtype = wp.quatf.
        In torch this resolves to (num_instances, 4).
        """
        parent = self.root_link_pose_w
        if self._root_link_quat_w_ta is None:
            self._root_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_link_quat_w_ta

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root link linear velocity in simulation world frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        parent = self.root_link_vel_w
        if self._root_lin_vel_w_ta is None:
            self._root_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_lin_vel_w_ta

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root link angular velocity in simulation world frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        parent = self.root_link_vel_w
        if self._root_ang_vel_w_ta is None:
            self._root_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_ang_vel_w_ta

    @property
    def root_com_pos_w(self) -> ProxyArray:
        """Root center of mass position in simulation world frame [m].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        parent = self.root_com_pose_w
        if self._root_com_pos_w_ta is None:
            self._root_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._root_com_pos_w_ta

    @property
    def root_com_quat_w(self) -> ProxyArray:
        """Root center of mass orientation (x, y, z, w) in simulation world frame [-].
        Shape is (num_instances,), dtype = wp.quatf.
        In torch this resolves to (num_instances, 4).
        """
        parent = self.root_com_pose_w
        if self._root_com_quat_w_ta is None:
            self._root_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._root_com_quat_w_ta

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        """Root center of mass linear velocity in simulation world frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        parent = self.root_com_vel_w
        if self._root_com_lin_vel_w_ta is None:
            self._root_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._root_com_lin_vel_w_ta

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        """Root center of mass angular velocity in simulation world frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        parent = self.root_com_vel_w
        if self._root_com_ang_vel_w_ta is None:
            self._root_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._root_com_ang_vel_w_ta

    @property
    def body_link_pos_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_link_quat_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_pos_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_quat_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_pos_b(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def body_com_quat_b(self) -> ProxyArray:
        raise NotImplementedError
