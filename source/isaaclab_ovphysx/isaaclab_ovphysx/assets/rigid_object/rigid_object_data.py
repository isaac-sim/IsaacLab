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
from isaaclab_ovphysx.assets.kernels import (
    _compose_root_com_pose,
    _compute_heading,
    _projected_gravity,
    _world_vel_to_body_ang,
    _world_vel_to_body_lin,
)


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
        # Body acceleration buffer (allocated lazily; None until _ensure_derived_buffers).
        self._body_acc_w_buf: TimestampedBuffer | None = None
        # Derived-property buffers (allocated lazily on first access).
        self._projected_gravity_b_buf: TimestampedBuffer | None = None
        self._heading_w_buf: TimestampedBuffer | None = None
        self._root_link_lin_vel_b_buf: TimestampedBuffer | None = None
        self._root_link_ang_vel_b_buf: TimestampedBuffer | None = None
        self._root_com_lin_vel_b_buf: TimestampedBuffer | None = None
        self._root_com_ang_vel_b_buf: TimestampedBuffer | None = None
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
        # Body-state singleton-dim ProxyArrays ((N,1,k) views of root buffers).
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._body_com_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._body_link_pos_w_ta: ProxyArray | None = None
        self._body_link_quat_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_w_ta: ProxyArray | None = None
        self._body_link_ang_vel_w_ta: ProxyArray | None = None
        self._body_com_pos_w_ta: ProxyArray | None = None
        self._body_com_quat_w_ta: ProxyArray | None = None
        self._body_com_lin_vel_w_ta: ProxyArray | None = None
        self._body_com_ang_vel_w_ta: ProxyArray | None = None
        # Body acceleration ProxyArrays.
        self._body_acc_w_ta: ProxyArray | None = None
        self._body_link_lin_acc_w_ta: ProxyArray | None = None
        self._body_link_ang_acc_w_ta: ProxyArray | None = None
        self._body_com_lin_acc_w_ta: ProxyArray | None = None
        self._body_com_ang_acc_w_ta: ProxyArray | None = None
        # Body property buffers (semi-static; lazy-allocated in _ensure_body_prop_buffers).
        self._body_mass_buf: TimestampedBuffer | None = None
        self._body_inertia_buf: TimestampedBuffer | None = None
        # Body property ProxyArrays.
        self._body_mass_ta: ProxyArray | None = None
        self._body_inertia_ta: ProxyArray | None = None
        self._body_com_pose_b_ta: ProxyArray | None = None
        self._body_com_pos_b_ta: ProxyArray | None = None
        self._body_com_quat_b_ta: ProxyArray | None = None
        # Derived-property ProxyArrays.
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._heading_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_b_ta: ProxyArray | None = None
        self._root_link_ang_vel_b_ta: ProxyArray | None = None
        self._root_com_lin_vel_b_ta: ProxyArray | None = None
        self._root_com_ang_vel_b_ta: ProxyArray | None = None
        # Gravity and forward constants (allocated lazily in _ensure_derived_buffers).
        self.GRAVITY_VEC_W: ProxyArray | None = None
        self.FORWARD_VEC_B: ProxyArray | None = None

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
            self._body_acc_w_buf,
            self._projected_gravity_b_buf,
            self._heading_w_buf,
            self._root_link_lin_vel_b_buf,
            self._root_link_ang_vel_b_buf,
            self._root_com_lin_vel_b_buf,
            self._root_com_ang_vel_b_buf,
            self._body_mass_buf,
            self._body_inertia_buf,
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

    def _ensure_derived_buffers(self) -> None:
        """Allocate derived-property and body-acceleration TimestampedBuffers on first use.

        Also initialises :attr:`GRAVITY_VEC_W` and :attr:`FORWARD_VEC_B` on first call,
        mirroring the per-instance tiled constants used by
        :class:`~isaaclab_ovphysx.assets.articulation.ArticulationData`.
        """
        if self._projected_gravity_b_buf is not None:
            return
        N = self._num_instances
        dev = self._device
        # Body acceleration (spatial vector per instance, same shape as ROOT_VELOCITY).
        self._body_acc_w_buf = TimestampedBuffer(N, dev, wp.spatial_vectorf)
        # Derived scalar / vector outputs.
        self._projected_gravity_b_buf = TimestampedBuffer(N, dev, wp.vec3f)
        self._heading_w_buf = TimestampedBuffer(N, dev, wp.float32)
        self._root_link_lin_vel_b_buf = TimestampedBuffer(N, dev, wp.vec3f)
        self._root_link_ang_vel_b_buf = TimestampedBuffer(N, dev, wp.vec3f)
        self._root_com_lin_vel_b_buf = TimestampedBuffer(N, dev, wp.vec3f)
        self._root_com_ang_vel_b_buf = TimestampedBuffer(N, dev, wp.vec3f)
        # Gravity and forward constants (tiled per-instance, matching articulation pattern).
        # Guard against no sim context in mock/test environments.
        from isaaclab.physics import PhysicsManager

        gravity = (0.0, 0.0, -9.81)
        if PhysicsManager._sim is not None and hasattr(PhysicsManager._sim, "cfg"):
            gravity = PhysicsManager._sim.cfg.gravity
        gravity_np = np.array(gravity, dtype=np.float32)
        gravity_mag = np.linalg.norm(gravity_np)
        if gravity_mag == 0.0:
            gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            gravity_dir = gravity_np / gravity_mag
        gravity_dir_tiled = np.tile(gravity_dir, (N, 1))
        forward_tiled = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (N, 1))
        self.GRAVITY_VEC_W = ProxyArray(wp.from_numpy(gravity_dir_tiled, dtype=wp.vec3f, device=dev))
        self.FORWARD_VEC_B = ProxyArray(wp.from_numpy(forward_tiled, dtype=wp.vec3f, device=dev))

    def _ensure_body_prop_buffers(self) -> None:
        """Allocate body-property TimestampedBuffers on first use.

        ``body_mass`` needs a ``(N, 1)`` float32 buffer that matches the
        ``RIGID_BODY_MASS`` binding shape exactly. ``body_inertia`` needs a
        ``(N, 9)`` flat float32 buffer so that ``binding.read()`` can fill it
        directly; the ``(N, 1, 9)`` view is constructed zero-copy in the
        property accessor.
        """
        if self._body_mass_buf is not None:
            return
        N = self._num_instances
        dev = self._device
        self._body_mass_buf = TimestampedBuffer((N, 1), dev, wp.float32)
        # Store flat (N, 9) so binding.read() sees the correct shape.
        self._body_inertia_buf = TimestampedBuffer((N, 9), dev, wp.float32)

    def _read_flat_binding(self, tensor_type: int, buf: TimestampedBuffer) -> None:
        """Read a flat (float32) CPU binding into a TimestampedBuffer, skipping if fresh."""
        if buf.timestamp >= self._sim_time:
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        # CPU-only bindings: read via numpy then copy to the target device.
        np_buf = np.zeros(binding.shape, dtype=np.float32)
        binding.read(np_buf)
        wp.copy(buf.data, wp.from_numpy(np_buf, dtype=wp.float32, device=self._device))
        buf.timestamp = self._sim_time

    # --- internal helpers: singleton-dim reshape -----------------------
    def _reshape_to_body_view(self, arr: wp.array, dtype) -> wp.array:
        """Return a zero-copy (N, 1) view of a 1-D (N,) warp array.

        For ``num_bodies=1`` this turns every root buffer into a body-tensor
        with the singleton body dimension that the base API expects, so that
        downstream torch callers see shape ``(N, 1, k)`` without any copy.

        Args:
            arr: 1-D warp array of shape ``(N,)``.
            dtype: The warp scalar/struct dtype (e.g. ``wp.transformf``).

        Returns:
            A zero-copy warp array of shape ``(N, 1)`` with ``dtype``.
        """
        N = arr.shape[0]
        elem_bytes = wp.types.type_size_in_bytes(dtype)
        stride_n = elem_bytes  # tightly packed 1-D source
        return wp.array(
            ptr=arr.ptr,
            shape=(N, 1),
            dtype=dtype,
            strides=(stride_n, stride_n),
            device=self._device,
            copy=False,
        )

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
        """Body link pose ``[pos, quat]`` in simulation world frame [m, -].
        Shape is (num_instances, num_bodies), dtype = wp.transformf.
        In torch this resolves to (num_instances, num_bodies, 7).

        For a single rigid body ``num_bodies=1``, this is a zero-copy
        ``(N, 1)`` view of :attr:`root_link_pose_w`.
        """
        _ = self.root_link_pose_w  # ensure root buffer is fresh
        if self._body_link_pose_w_ta is None:
            view = self._reshape_to_body_view(self._root_link_pose_w_buf.data, wp.transformf)
            self._body_link_pose_w_ta = ProxyArray(view)
        return self._body_link_pose_w_ta

    @property
    def body_link_vel_w(self) -> ProxyArray:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame [m/s, rad/s].
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        For a single rigid body ``num_bodies=1``, this is a zero-copy
        ``(N, 1)`` view of :attr:`root_link_vel_w`.
        """
        _ = self.root_link_vel_w  # ensure root buffer is fresh
        if self._body_link_vel_w_ta is None:
            view = self._reshape_to_body_view(self._root_link_vel_w_buf.data, wp.spatial_vectorf)
            self._body_link_vel_w_ta = ProxyArray(view)
        return self._body_link_vel_w_ta

    @property
    def body_com_pose_w(self) -> ProxyArray:
        """Body center-of-mass pose ``[pos, quat]`` in simulation world frame [m, -].
        Shape is (num_instances, num_bodies), dtype = wp.transformf.
        In torch this resolves to (num_instances, num_bodies, 7).

        For a single rigid body ``num_bodies=1``, this is a zero-copy
        ``(N, 1)`` view of :attr:`root_com_pose_w`.
        """
        _ = self.root_com_pose_w  # ensure root COM buffer is fresh
        if self._body_com_pose_w_ta is None:
            view = self._reshape_to_body_view(self._root_com_pose_w_buf.data, wp.transformf)
            self._body_com_pose_w_ta = ProxyArray(view)
        return self._body_com_pose_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Body center-of-mass velocity ``[lin_vel, ang_vel]`` in simulation world frame
        [m/s, rad/s].
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        For a single rigid body ``num_bodies=1``, this is a zero-copy
        ``(N, 1)`` view of :attr:`root_com_vel_w`.
        """
        _ = self.root_com_vel_w  # ensure root COM vel buffer is fresh
        if self._body_com_vel_w_ta is None:
            view = self._reshape_to_body_view(self._root_com_vel_w_buf.data, wp.spatial_vectorf)
            self._body_com_vel_w_ta = ProxyArray(view)
        return self._body_com_vel_w_ta

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
    def body_link_acc_w(self) -> ProxyArray:
        """Body link acceleration ``[lin_acc, ang_acc]`` in simulation world frame
        [m/s^2, rad/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        Requires the ovphysx wheel to expose ``RIGID_BODY_ACCELERATION``.

        Raises:
            NotImplementedError: If the ``RIGID_BODY_ACCELERATION`` binding is absent,
                e.g. when running with a wheel that does not yet provide this tensor type.
        """
        self._ensure_derived_buffers()
        if TT.RIGID_BODY_ACCELERATION not in self._bindings or self._bindings[TT.RIGID_BODY_ACCELERATION] is None:
            raise NotImplementedError(
                "body acceleration requires ovphysx wheel with RIGID_BODY_ACCELERATION; "
                "see docs/superpowers/specs/2026-04-27-ovphysx-rigid-body-tensortypes-gap.md"
            )
        self._read_spatial_vector_binding(TT.RIGID_BODY_ACCELERATION, self._body_acc_w_buf)
        if self._body_acc_w_ta is None:
            view = self._reshape_to_body_view(self._body_acc_w_buf.data, wp.spatial_vectorf)
            self._body_acc_w_ta = ProxyArray(view)
        return self._body_acc_w_ta

    @property
    def body_com_acc_w(self) -> ProxyArray:
        """Body center-of-mass acceleration ``[lin_acc, ang_acc]`` in simulation world frame
        [m/s^2, rad/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.spatial_vectorf.
        In torch this resolves to (num_instances, num_bodies, 6).

        For a single rigid body ``num_bodies=1``, identical to :attr:`body_link_acc_w`.

        Raises:
            NotImplementedError: If the ``RIGID_BODY_ACCELERATION`` binding is absent.
        """
        return self.body_link_acc_w

    @property
    def body_com_pose_b(self) -> ProxyArray:
        """Center-of-mass pose ``[pos, quat]`` of all bodies in their respective body frames [m, -].
        Shape is (num_instances, num_bodies), dtype = wp.transformf.
        In torch this resolves to (num_instances, num_bodies, 7).

        For a single rigid body ``num_bodies=1``, the body frame equals the root
        link frame.  The orientation is provided in (x, y, z, w) format.
        """
        self._ensure_root_buffers()
        self._read_transform_binding(TT.RIGID_BODY_COM_POSE, self._body_com_pose_b_buf)
        if self._body_com_pose_b_ta is None:
            self._body_com_pose_b_ta = ProxyArray(self._body_com_pose_b_buf.data)
        return self._body_com_pose_b_ta

    @property
    def body_mass(self) -> ProxyArray:
        """Mass of all bodies [kg].
        Shape is (num_instances, num_bodies), dtype = wp.float32.
        In torch this resolves to (num_instances, num_bodies).
        """
        self._ensure_body_prop_buffers()
        self._read_flat_binding(TT.RIGID_BODY_MASS, self._body_mass_buf)
        if self._body_mass_ta is None:
            self._body_mass_ta = ProxyArray(self._body_mass_buf.data)
        return self._body_mass_ta

    @property
    def body_inertia(self) -> ProxyArray:
        """Flattened inertia tensor of all bodies [kg*m^2].
        Shape is (num_instances, num_bodies, 9), dtype = wp.float32.
        In torch this resolves to (num_instances, num_bodies, 9).

        Stored as a flattened 3x3 inertia matrix per body.
        """
        self._ensure_body_prop_buffers()
        self._read_flat_binding(TT.RIGID_BODY_INERTIA, self._body_inertia_buf)
        if self._body_inertia_ta is None:
            # Zero-copy reshape from (N, 9) to (N, 1, 9).
            raw = self._body_inertia_buf.data
            N = raw.shape[0]
            view = wp.array(
                ptr=raw.ptr,
                shape=(N, 1, 9),
                dtype=wp.float32,
                device=self._device,
                copy=False,
            )
            self._body_inertia_ta = ProxyArray(view)
        return self._body_inertia_ta

    @property
    def projected_gravity_b(self) -> ProxyArray:
        """Projection of the gravity direction on the root body frame [-].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        self._ensure_derived_buffers()
        if self._projected_gravity_b_buf.timestamp < self._sim_time:
            wp.launch(
                _projected_gravity,
                dim=self._num_instances,
                inputs=[self.GRAVITY_VEC_W, self.root_link_pose_w],
                outputs=[self._projected_gravity_b_buf.data],
                device=self._device,
            )
            self._projected_gravity_b_buf.timestamp = self._sim_time
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b_buf.data)
        return self._projected_gravity_b_ta

    @property
    def heading_w(self) -> ProxyArray:
        """Yaw heading of the root body frame [rad].
        Shape is (num_instances,), dtype = wp.float32.

        .. note::
            Computed assuming the forward direction in the body frame is along x,
            i.e. :math:`(1, 0, 0)`.
        """
        self._ensure_derived_buffers()
        if self._heading_w_buf.timestamp < self._sim_time:
            wp.launch(
                _compute_heading,
                dim=self._num_instances,
                inputs=[self.FORWARD_VEC_B, self.root_link_pose_w],
                outputs=[self._heading_w_buf.data],
                device=self._device,
            )
            self._heading_w_buf.timestamp = self._sim_time
        if self._heading_w_ta is None:
            self._heading_w_ta = ProxyArray(self._heading_w_buf.data)
        return self._heading_w_ta

    @property
    def root_link_lin_vel_b(self) -> ProxyArray:
        """Root link linear velocity in the root body frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        self._ensure_derived_buffers()
        if self._root_link_lin_vel_b_buf.timestamp < self._sim_time:
            wp.launch(
                _world_vel_to_body_lin,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_lin_vel_b_buf.data],
                device=self._device,
            )
            self._root_link_lin_vel_b_buf.timestamp = self._sim_time
        if self._root_link_lin_vel_b_ta is None:
            self._root_link_lin_vel_b_ta = ProxyArray(self._root_link_lin_vel_b_buf.data)
        return self._root_link_lin_vel_b_ta

    @property
    def root_link_ang_vel_b(self) -> ProxyArray:
        """Root link angular velocity in the root body frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        self._ensure_derived_buffers()
        if self._root_link_ang_vel_b_buf.timestamp < self._sim_time:
            wp.launch(
                _world_vel_to_body_ang,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_link_vel_w],
                outputs=[self._root_link_ang_vel_b_buf.data],
                device=self._device,
            )
            self._root_link_ang_vel_b_buf.timestamp = self._sim_time
        if self._root_link_ang_vel_b_ta is None:
            self._root_link_ang_vel_b_ta = ProxyArray(self._root_link_ang_vel_b_buf.data)
        return self._root_link_ang_vel_b_ta

    @property
    def root_com_lin_vel_b(self) -> ProxyArray:
        """Root center-of-mass linear velocity in the root body frame [m/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        self._ensure_derived_buffers()
        if self._root_com_lin_vel_b_buf.timestamp < self._sim_time:
            wp.launch(
                _world_vel_to_body_lin,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
                outputs=[self._root_com_lin_vel_b_buf.data],
                device=self._device,
            )
            self._root_com_lin_vel_b_buf.timestamp = self._sim_time
        if self._root_com_lin_vel_b_ta is None:
            self._root_com_lin_vel_b_ta = ProxyArray(self._root_com_lin_vel_b_buf.data)
        return self._root_com_lin_vel_b_ta

    @property
    def root_com_ang_vel_b(self) -> ProxyArray:
        """Root center-of-mass angular velocity in the root body frame [rad/s].
        Shape is (num_instances,), dtype = wp.vec3f.
        In torch this resolves to (num_instances, 3).
        """
        self._ensure_derived_buffers()
        if self._root_com_ang_vel_b_buf.timestamp < self._sim_time:
            wp.launch(
                _world_vel_to_body_ang,
                dim=self._num_instances,
                inputs=[self.root_link_pose_w, self.root_com_vel_w],
                outputs=[self._root_com_ang_vel_b_buf.data],
                device=self._device,
            )
            self._root_com_ang_vel_b_buf.timestamp = self._sim_time
        if self._root_com_ang_vel_b_ta is None:
            self._root_com_ang_vel_b_ta = ProxyArray(self._root_com_ang_vel_b_buf.data)
        return self._root_com_ang_vel_b_ta

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
        """Position of all bodies in simulation world frame [m].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_link_pose_w
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) of all bodies in simulation world frame [-].
        Shape is (num_instances, num_bodies), dtype = wp.quatf.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_link_pose_w
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_link_quat_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Linear velocity of all bodies in simulation world frame [m/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_link_vel_w
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Angular velocity of all bodies in simulation world frame [rad/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_link_vel_w
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_pos_w(self) -> ProxyArray:
        """Center-of-mass position of all bodies in simulation world frame [m].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_w
        if self._body_com_pos_w_ta is None:
            self._body_com_pos_w_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_w_ta

    @property
    def body_com_quat_w(self) -> ProxyArray:
        """Center-of-mass orientation (x, y, z, w) of all bodies in simulation world frame [-].
        Shape is (num_instances, num_bodies), dtype = wp.quatf.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_w
        if self._body_com_quat_w_ta is None:
            self._body_com_quat_w_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Center-of-mass linear velocity of all bodies in simulation world frame [m/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Center-of-mass angular velocity of all bodies in simulation world frame [rad/s].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_vel_w
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_vel_w_ta

    @property
    def body_com_lin_acc_w(self) -> ProxyArray:
        """Center-of-mass linear acceleration of all bodies in simulation world frame [m/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).

        Raises:
            NotImplementedError: If the ``RIGID_BODY_ACCELERATION`` binding is absent.
        """
        parent = self.body_com_acc_w
        if self._body_com_lin_acc_w_ta is None:
            self._body_com_lin_acc_w_ta = ProxyArray(self._get_lin_vel_from_spatial_vector(parent.warp))
        return self._body_com_lin_acc_w_ta

    @property
    def body_com_ang_acc_w(self) -> ProxyArray:
        """Center-of-mass angular acceleration of all bodies in simulation world frame [rad/s^2].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).

        Raises:
            NotImplementedError: If the ``RIGID_BODY_ACCELERATION`` binding is absent.
        """
        parent = self.body_com_acc_w
        if self._body_com_ang_acc_w_ta is None:
            self._body_com_ang_acc_w_ta = ProxyArray(self._get_ang_vel_from_spatial_vector(parent.warp))
        return self._body_com_ang_acc_w_ta

    @property
    def body_com_pos_b(self) -> ProxyArray:
        """Center-of-mass position of all bodies in their respective body frames [m].
        Shape is (num_instances, num_bodies), dtype = wp.vec3f.
        In torch this resolves to (num_instances, num_bodies, 3).
        """
        parent = self.body_com_pose_b
        if self._body_com_pos_b_ta is None:
            self._body_com_pos_b_ta = ProxyArray(self._get_pos_from_transform(parent.warp))
        return self._body_com_pos_b_ta

    @property
    def body_com_quat_b(self) -> ProxyArray:
        """Center-of-mass orientation (x, y, z, w) of all bodies in their respective body frames [-].
        Shape is (num_instances, num_bodies), dtype = wp.quatf.
        In torch this resolves to (num_instances, num_bodies, 4).
        """
        parent = self.body_com_pose_b
        if self._body_com_quat_b_ta is None:
            self._body_com_quat_b_ta = ProxyArray(self._get_quat_from_transform(parent.warp))
        return self._body_com_quat_b_ta
