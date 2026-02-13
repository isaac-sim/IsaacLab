# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock rigid object asset for testing without Isaac Sim."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

try:
    from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
except (ImportError, ModuleNotFoundError):
    # Direct import bypassing isaaclab.assets.__init__.py (which needs omni.timeline)
    import importlib.util
    from pathlib import Path

    _file = Path(__file__).resolve().parents[3] / "assets" / "rigid_object" / "base_rigid_object_data.py"
    _spec = importlib.util.spec_from_file_location("_base_rigid_object_data", str(_file))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseRigidObjectData = _mod.BaseRigidObjectData


class MockRigidObjectData(BaseRigidObjectData):
    """Mock data container for rigid object asset.

    This class inherits from BaseRigidObjectData to get shorthand and deprecated properties
    for free (e.g., root_pose_w, body_pos_w, com_pos_b, default_mass, etc.).
    All tensor properties return zero warp arrays with correct shapes if not explicitly set.
    """

    def __init__(
        self,
        num_instances: int,
        body_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock rigid object data.

        Args:
            num_instances: Number of rigid object instances.
            body_names: Names of bodies (single body for rigid object).
            device: Device for tensor allocation.
        """
        super().__init__(root_view=None, device=device)
        self._create_buffers()

        self._num_instances = num_instances
        self._num_bodies = 1  # Rigid object always has 1 body

        self.body_names = body_names or ["body"]

        # Default states
        self._default_root_pose: wp.array | None = None
        self._default_root_vel: wp.array | None = None

        # Root state (link frame)
        self._root_link_pose_w: wp.array | None = None
        self._root_link_vel_w: wp.array | None = None

        # Root state (CoM frame)
        self._root_com_pose_w: wp.array | None = None
        self._root_com_vel_w: wp.array | None = None

        # Body state (link frame)
        self._body_link_pose_w: wp.array | None = None
        self._body_link_vel_w: wp.array | None = None

        # Body state (CoM frame)
        self._body_com_pose_w: wp.array | None = None
        self._body_com_vel_w: wp.array | None = None
        self._body_com_acc_w: wp.array | None = None
        self._body_com_pose_b: wp.array | None = None

        # Body properties
        self._body_mass: wp.array | None = None
        self._body_inertia: wp.array | None = None

    def _identity_quat(self, *shape: int) -> wp.array:
        """Create identity quaternion warp array (w, x, y, z) = (1, 0, 0, 0)."""
        quat_np = np.zeros((*shape, 4), dtype=np.float32)
        quat_np[..., 0] = 1.0
        return wp.array(quat_np, dtype=wp.float32, device=self.device)

    # -- Default state properties --

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose. dtype=wp.transformf, shape: (N,)."""
        if self._default_root_pose is None:
            pose_np = np.zeros((self._num_instances, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            return wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._default_root_pose

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity. Shape: (N, 6)."""
        if self._default_root_vel is None:
            return wp.zeros((self._num_instances, 6), dtype=wp.float32, device=self.device)
        return self._default_root_vel

    @property
    def default_root_state(self) -> wp.array:
        """Default root state. Shape: (N, 13)."""
        pose = wp.to_torch(self.default_root_pose)
        vel = wp.to_torch(self.default_root_vel)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    # -- Root state properties (link frame) --

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose in world frame. dtype=wp.transformf, shape: (N,)."""
        if self._root_link_pose_w is None:
            pose_np = np.zeros((self._num_instances, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            return wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._root_link_pose_w

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity in world frame. Shape: (N, 6)."""
        if self._root_link_vel_w is None:
            return wp.zeros((self._num_instances, 6), dtype=wp.float32, device=self.device)
        return self._root_link_vel_w

    @property
    def root_link_state_w(self) -> wp.array:
        """Root link state in world frame. Shape: (N, 13)."""
        pose = wp.to_torch(self.root_link_pose_w)
        vel = wp.to_torch(self.root_link_vel_w)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def root_link_pos_w(self) -> wp.array:
        """Root link position. Shape: (N,), dtype=wp.vec3f."""
        t = self.root_link_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def root_link_quat_w(self) -> wp.array:
        """Root link orientation. Shape: (N,), dtype=wp.quatf."""
        t = self.root_link_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def root_link_lin_vel_w(self) -> wp.array:
        """Root link linear velocity. Shape: (N, 3)."""
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> wp.array:
        """Root link angular velocity. Shape: (N, 3)."""
        return self.root_link_vel_w[:, 3:6]

    # -- Root state properties (CoM frame) --

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root CoM pose in world frame. dtype=wp.transformf, shape: (N,)."""
        if self._root_com_pose_w is None:
            return wp.clone(self.root_link_pose_w, self.device)
        return self._root_com_pose_w

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root CoM velocity in world frame. Shape: (N, 6)."""
        if self._root_com_vel_w is None:
            return wp.clone(self.root_link_vel_w, self.device)
        return self._root_com_vel_w

    @property
    def root_com_state_w(self) -> wp.array:
        """Root CoM state in world frame. Shape: (N, 13)."""
        pose = wp.to_torch(self.root_com_pose_w)
        vel = wp.to_torch(self.root_com_vel_w)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    @property
    def root_state_w(self) -> wp.array:
        """Root state (link pose + CoM velocity). Shape: (N, 13)."""
        pose = wp.to_torch(self.root_link_pose_w)
        vel = wp.to_torch(self.root_com_vel_w)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def root_com_pos_w(self) -> wp.array:
        """Root CoM position. Shape: (N,), dtype=wp.vec3f."""
        t = self.root_com_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def root_com_quat_w(self) -> wp.array:
        """Root CoM orientation. Shape: (N,), dtype=wp.quatf."""
        t = self.root_com_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def root_com_lin_vel_w(self) -> wp.array:
        """Root CoM linear velocity. Shape: (N, 3)."""
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> wp.array:
        """Root CoM angular velocity. Shape: (N, 3)."""
        return self.root_com_vel_w[:, 3:6]

    # -- Body state properties (link frame) --

    @property
    def body_link_pose_w(self) -> wp.array:
        """Body link pose in world frame. dtype=wp.transformf, shape: (N, 1)."""
        if self._body_link_pose_w is None:
            pose_np = np.zeros((self._num_instances, 1, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            return wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._body_link_pose_w

    @property
    def body_link_vel_w(self) -> wp.array:
        """Body link velocity in world frame. Shape: (N, 1, 6)."""
        if self._body_link_vel_w is None:
            return wp.zeros((self._num_instances, 1, 6), dtype=wp.float32, device=self.device)
        return self._body_link_vel_w

    @property
    def body_link_state_w(self) -> wp.array:
        """Body link state in world frame. Shape: (N, 1, 13)."""
        pose = wp.to_torch(self.body_link_pose_w)
        vel = wp.to_torch(self.body_link_vel_w)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def body_link_pos_w(self) -> wp.array:
        """Body link position. Shape: (N, 1), dtype=wp.vec3f."""
        t = self.body_link_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def body_link_quat_w(self) -> wp.array:
        """Body link orientation. Shape: (N, 1), dtype=wp.quatf."""
        t = self.body_link_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def body_link_lin_vel_w(self) -> wp.array:
        """Body link linear velocity. Shape: (N, 1, 3)."""
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> wp.array:
        """Body link angular velocity. Shape: (N, 1, 3)."""
        return self.body_link_vel_w[..., 3:6]

    # -- Body state properties (CoM frame) --

    @property
    def body_com_pose_w(self) -> wp.array:
        """Body CoM pose in world frame. dtype=wp.transformf, shape: (N, 1)."""
        if self._body_com_pose_w is None:
            return wp.clone(self.body_link_pose_w, self.device)
        return self._body_com_pose_w

    @property
    def body_com_vel_w(self) -> wp.array:
        """Body CoM velocity in world frame. Shape: (N, 1, 6)."""
        if self._body_com_vel_w is None:
            return wp.clone(self.body_link_vel_w, self.device)
        return self._body_com_vel_w

    @property
    def body_com_state_w(self) -> wp.array:
        """Body CoM state in world frame. Shape: (N, 1, 13)."""
        pose = wp.to_torch(self.body_com_pose_w)
        vel = wp.to_torch(self.body_com_vel_w)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    @property
    def body_state_w(self) -> wp.array:
        """Body state (link pose + CoM vel). Shape: (N, 1, 13)."""
        pose = wp.to_torch(self.body_link_pose_w)
        vel = wp.to_torch(self.body_com_vel_w)
        return wp.from_torch(torch.cat([pose, vel], dim=-1))

    @property
    def body_com_acc_w(self) -> wp.array:
        """Body CoM acceleration in world frame. Shape: (N, 1, 6)."""
        if self._body_com_acc_w is None:
            return wp.zeros((self._num_instances, 1, 6), dtype=wp.float32, device=self.device)
        return self._body_com_acc_w

    @property
    def body_com_pose_b(self) -> wp.array:
        """Body CoM pose in body frame. dtype=wp.transformf, shape: (N, 1)."""
        if self._body_com_pose_b is None:
            pose_np = np.zeros((self._num_instances, 1, 7), dtype=np.float32)
            pose_np[..., 6] = 1.0  # identity quat qw=1, transformf layout: (px,py,pz,qx,qy,qz,qw)
            return wp.array(pose_np, dtype=wp.float32, device=self.device).view(wp.transformf)
        return self._body_com_pose_b

    # Sliced properties (zero-copy pointer arithmetic on transformf)
    @property
    def body_com_pos_w(self) -> wp.array:
        """Body CoM position. Shape: (N, 1), dtype=wp.vec3f."""
        t = self.body_com_pose_w
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def body_com_quat_w(self) -> wp.array:
        """Body CoM orientation. Shape: (N, 1), dtype=wp.quatf."""
        t = self.body_com_pose_w
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    @property
    def body_com_lin_vel_w(self) -> wp.array:
        """Body CoM linear velocity. Shape: (N, 1, 3)."""
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> wp.array:
        """Body CoM angular velocity. Shape: (N, 1, 3)."""
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> wp.array:
        """Body CoM linear acceleration. Shape: (N, 1, 3)."""
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> wp.array:
        """Body CoM angular acceleration. Shape: (N, 1, 3)."""
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> wp.array:
        """Body CoM position in body frame. Shape: (N, 1), dtype=wp.vec3f."""
        t = self.body_com_pose_b
        return wp.array(ptr=t.ptr, shape=t.shape, dtype=wp.vec3f, strides=t.strides, device=self.device)

    @property
    def body_com_quat_b(self) -> wp.array:
        """Body CoM orientation in body frame. Shape: (N, 1), dtype=wp.quatf."""
        t = self.body_com_pose_b
        return wp.array(ptr=t.ptr + 3 * 4, shape=t.shape, dtype=wp.quatf, strides=t.strides, device=self.device)

    # -- Body properties --

    @property
    def body_mass(self) -> wp.array:
        """Body mass. Shape: (N, 1, 1)."""
        if self._body_mass is None:
            return wp.ones((self._num_instances, 1, 1), dtype=wp.float32, device=self.device)
        return self._body_mass

    @property
    def body_inertia(self) -> wp.array:
        """Body inertia (flattened 3x3). Shape: (N, 1, 9)."""
        if self._body_inertia is None:
            inertia_np = np.zeros((self._num_instances, 1, 9), dtype=np.float32)
            inertia_np[..., 0] = 1.0  # Ixx
            inertia_np[..., 4] = 1.0  # Iyy
            inertia_np[..., 8] = 1.0  # Izz
            return wp.array(inertia_np, dtype=wp.float32, device=self.device)
        return self._body_inertia

    # -- Derived properties --

    @property
    def projected_gravity_b(self) -> wp.array:
        """Gravity projection on body. Shape: (N, 3)."""
        gravity_np = np.zeros((self._num_instances, 3), dtype=np.float32)
        gravity_np[:, 2] = -1.0
        return wp.array(gravity_np, dtype=wp.float32, device=self.device)

    @property
    def heading_w(self) -> wp.array:
        """Yaw heading in world frame. Shape: (N,)."""
        return wp.zeros((self._num_instances,), dtype=wp.float32, device=self.device)

    @property
    def root_link_lin_vel_b(self) -> wp.array:
        """Root link linear velocity in body frame. Shape: (N, 3)."""
        return wp.clone(self.root_link_lin_vel_w, self.device)

    @property
    def root_link_ang_vel_b(self) -> wp.array:
        """Root link angular velocity in body frame. Shape: (N, 3)."""
        return wp.clone(self.root_link_ang_vel_w, self.device)

    @property
    def root_com_lin_vel_b(self) -> wp.array:
        """Root CoM linear velocity in body frame. Shape: (N, 3)."""
        return wp.clone(self.root_com_lin_vel_w, self.device)

    @property
    def root_com_ang_vel_b(self) -> wp.array:
        """Root CoM angular velocity in body frame. Shape: (N, 3)."""
        return wp.clone(self.root_com_ang_vel_w, self.device)

    # -- Shorthand properties (root_pose_w, body_pos_w, com_pos_b, etc.) --
    # Inherited from BaseRigidObjectData

    # -- Deprecated properties (default_mass, default_inertia) --
    # Inherited from BaseRigidObjectData

    # -- Update method --

    def update(self, dt: float) -> None:
        """Update data (no-op for mock)."""
        pass

    # -- Setters --

    def set_default_root_pose(self, value: torch.Tensor) -> None:
        self._default_root_pose = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_default_root_vel(self, value: torch.Tensor) -> None:
        self._default_root_vel = wp.from_torch(value.to(self.device).contiguous())

    def set_root_link_pose_w(self, value: torch.Tensor) -> None:
        self._root_link_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_root_link_vel_w(self, value: torch.Tensor) -> None:
        self._root_link_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_root_com_pose_w(self, value: torch.Tensor) -> None:
        self._root_com_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_root_com_vel_w(self, value: torch.Tensor) -> None:
        self._root_com_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_link_pose_w(self, value: torch.Tensor) -> None:
        self._body_link_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_body_link_vel_w(self, value: torch.Tensor) -> None:
        self._body_link_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_com_pose_w(self, value: torch.Tensor) -> None:
        self._body_com_pose_w = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_body_com_vel_w(self, value: torch.Tensor) -> None:
        self._body_com_vel_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_com_acc_w(self, value: torch.Tensor) -> None:
        self._body_com_acc_w = wp.from_torch(value.to(self.device).contiguous())

    def set_body_com_pose_b(self, value: torch.Tensor) -> None:
        self._body_com_pose_b = wp.from_torch(value.to(self.device).contiguous()).view(wp.transformf)

    def set_body_mass(self, value: torch.Tensor) -> None:
        self._body_mass = wp.from_torch(value.to(self.device).contiguous())

    def set_body_inertia(self, value: torch.Tensor) -> None:
        self._body_inertia = wp.from_torch(value.to(self.device).contiguous())

    def set_mock_data(self, **kwargs) -> None:
        """Bulk setter for mock data."""
        for key, value in kwargs.items():
            setter = getattr(self, f"set_{key}", None)
            if setter is not None:
                setter(value)
            else:
                raise ValueError(f"Unknown property: {key}")


class MockRigidObject:
    """Mock rigid object asset for testing without Isaac Sim.

    This class mimics the interface of BaseRigidObject for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        body_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock rigid object.

        Args:
            num_instances: Number of rigid object instances.
            body_names: Names of bodies (single body for rigid object).
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_bodies = 1
        self._device = device
        self._body_names = body_names or ["body"]

        self._data = MockRigidObjectData(
            num_instances=num_instances,
            body_names=self._body_names,
            device=device,
        )

    # -- Properties --

    @property
    def data(self) -> MockRigidObjectData:
        """Data container for the rigid object."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of rigid object instances."""
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies (always 1 for rigid object)."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Body names."""
        return self._body_names

    @property
    def device(self) -> str:
        """Device for tensor allocation."""
        return self._device

    @property
    def root_view(self) -> None:
        """Returns None (no physics view in mock)."""
        return None

    @property
    def instantaneous_wrench_composer(self) -> None:
        """Returns None (no wrench composer in mock)."""
        return None

    @property
    def permanent_wrench_composer(self) -> None:
        """Returns None (no wrench composer in mock)."""
        return None

    # -- Core methods --

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset rigid object state for specified environments."""
        pass

    def write_data_to_sim(self) -> None:
        """Write data to simulation (no-op for mock)."""
        pass

    def update(self, dt: float) -> None:
        """Update rigid object data."""
        self._data.update(dt)

    # -- Finder methods --

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies by name regex patterns."""
        if isinstance(name_keys, str):
            name_keys = [name_keys]

        matched_indices = []
        matched_names = []

        if preserve_order:
            for key in name_keys:
                pattern = re.compile(key)
                for i, name in enumerate(self._body_names):
                    if pattern.fullmatch(name) and i not in matched_indices:
                        matched_indices.append(i)
                        matched_names.append(name)
        else:
            for i, name in enumerate(self._body_names):
                for key in name_keys:
                    pattern = re.compile(key)
                    if pattern.fullmatch(name):
                        matched_indices.append(i)
                        matched_names.append(name)
                        break

        return matched_indices, matched_names

    # -- State writer methods (no-op for mock) --

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root state to simulation."""
        pass

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root CoM state to simulation."""
        pass

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root link state to simulation."""
        pass

    def write_root_pose_to_sim(
        self,
        root_pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root pose to simulation."""
        pass

    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root link pose to simulation."""
        pass

    def write_root_com_pose_to_sim(
        self,
        root_pose: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root CoM pose to simulation."""
        pass

    def write_root_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root velocity to simulation."""
        pass

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root CoM velocity to simulation."""
        pass

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write root link velocity to simulation."""
        pass

    # -- Setter methods --

    def set_masses(
        self,
        masses: torch.Tensor,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set body masses."""
        pass

    def set_coms(
        self,
        coms: torch.Tensor,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set body centers of mass."""
        pass

    def set_inertias(
        self,
        inertias: torch.Tensor,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set body inertias."""
        pass

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = True,
    ) -> None:
        """Set external forces and torques."""
        pass

    # -- Index/Mask methods (no-op for mock) --

    def write_root_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_root_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    def write_root_link_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_root_link_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    def write_root_com_pose_to_sim_index(
        self,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_root_com_pose_to_sim_mask(
        self,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    def write_root_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_root_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    def write_root_com_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_root_com_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    def write_root_link_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def write_root_link_velocity_to_sim_mask(
        self,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        pass

    def set_masses_index(
        self,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def set_masses_mask(
        self,
        masses: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        pass

    def set_coms_index(
        self,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def set_coms_mask(
        self,
        coms: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        pass

    def set_inertias_index(
        self,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        pass

    def set_inertias_mask(
        self,
        inertias: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        body_mask: wp.array | None = None,
    ) -> None:
        pass
