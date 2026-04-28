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
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx import tensor_types as TT  # noqa: F401  (used by subsequent tasks)


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
        self.is_primed: bool = False
        self._sim_time: float = 0.0
        self._timestamps: dict[str, float] = {}
        self._default_root_pose: wp.array | None = None
        self._default_root_velocity: wp.array | None = None

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

    # --- update / cache invalidation ----------------------------------
    def update(self, dt: float) -> None:
        """Advance the cached sim time. Per-property freshness checks happen
        lazily on access; nothing to do up front here."""
        self._sim_time += dt

    def _invalidate_caches(self, env_ids=None) -> None:
        """Coarse cache invalidation: clear every property timestamp so the
        next access re-reads from the binding. Called by
        :meth:`isaaclab_ovphysx.assets.RigidObject.reset` and by every
        body-property setter on :class:`RigidObject`. The ``env_ids``
        argument is accepted for parity with the articulation API but the
        caches stored on this single-body asset are full-tensor, so a
        fine-grained invalidation is not necessary here."""
        self._timestamps.clear()

    # --- defaults -----------------------------------------------------
    def _process_cfg(self, cfg: RigidObjectCfg) -> None:
        """Populate :attr:`_default_root_pose` and
        :attr:`_default_root_velocity` from ``cfg.init_state``. Called by
        :meth:`isaaclab_ovphysx.assets.RigidObject._initialize_impl` after
        ``_create_buffers``."""
        N = self._num_instances
        device = self._device
        # Pose: (px, py, pz, qx, qy, qz, qw)
        pose_np = np.broadcast_to(
            np.asarray([*cfg.init_state.pos, *cfg.init_state.rot], dtype=np.float32),
            (N, 7),
        ).copy()
        pose_arr = wp.from_numpy(pose_np, dtype=wp.float32, device=device)
        # Reinterpret-cast (N, 7) float32 → (N,) transformf — same trick as articulation.
        self._default_root_pose = wp.array(
            ptr=pose_arr.ptr, shape=(N,), dtype=wp.transformf, device=device, copy=False,
        )
        # Velocity: (vx, vy, vz, wx, wy, wz)
        vel_np = np.broadcast_to(
            np.asarray([*cfg.init_state.lin_vel, *cfg.init_state.ang_vel], dtype=np.float32),
            (N, 6),
        ).copy()
        vel_arr = wp.from_numpy(vel_np, dtype=wp.float32, device=device)
        self._default_root_velocity = wp.array(
            ptr=vel_arr.ptr, shape=(N,), dtype=wp.spatial_vectorf, device=device, copy=False,
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
        raise NotImplementedError

    @property
    def root_link_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_pose_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_vel_w(self) -> ProxyArray:
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def root_link_quat_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_pos_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_quat_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_lin_vel_w(self) -> ProxyArray:
        raise NotImplementedError

    @property
    def root_com_ang_vel_w(self) -> ProxyArray:
        raise NotImplementedError

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
