# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock rigid object asset for testing without Isaac Sim."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


class MockRigidObjectData:
    """Mock data container for rigid object asset.

    This class mimics the interface of BaseRigidObjectData for testing purposes.
    All tensor properties return zero tensors with correct shapes if not explicitly set.
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
        self._num_instances = num_instances
        self._num_bodies = 1  # Rigid object always has 1 body
        self._device = device

        self.body_names = body_names or ["body"]

        # Default states
        self._default_root_pose: torch.Tensor | None = None
        self._default_root_vel: torch.Tensor | None = None

        # Root state (link frame)
        self._root_link_pose_w: torch.Tensor | None = None
        self._root_link_vel_w: torch.Tensor | None = None

        # Root state (CoM frame)
        self._root_com_pose_w: torch.Tensor | None = None
        self._root_com_vel_w: torch.Tensor | None = None

        # Body state (link frame)
        self._body_link_pose_w: torch.Tensor | None = None
        self._body_link_vel_w: torch.Tensor | None = None

        # Body state (CoM frame)
        self._body_com_pose_w: torch.Tensor | None = None
        self._body_com_vel_w: torch.Tensor | None = None
        self._body_com_acc_w: torch.Tensor | None = None
        self._body_com_pose_b: torch.Tensor | None = None

        # Body properties
        self._body_mass: torch.Tensor | None = None
        self._body_inertia: torch.Tensor | None = None

    @property
    def device(self) -> str:
        """Device for tensor allocation."""
        return self._device

    def _identity_quat(self, *shape: int) -> torch.Tensor:
        """Create identity quaternion tensor (w, x, y, z) = (1, 0, 0, 0)."""
        quat = torch.zeros(*shape, 4, device=self._device)
        quat[..., 0] = 1.0
        return quat

    # -- Default state properties --

    @property
    def default_root_pose(self) -> torch.Tensor:
        """Default root pose. Shape: (N, 7)."""
        if self._default_root_pose is None:
            pose = torch.zeros(self._num_instances, 7, device=self._device)
            pose[:, 3:7] = self._identity_quat(self._num_instances)
            return pose
        return self._default_root_pose

    @property
    def default_root_vel(self) -> torch.Tensor:
        """Default root velocity. Shape: (N, 6)."""
        if self._default_root_vel is None:
            return torch.zeros(self._num_instances, 6, device=self._device)
        return self._default_root_vel

    @property
    def default_root_state(self) -> torch.Tensor:
        """Default root state. Shape: (N, 13)."""
        return torch.cat([self.default_root_pose, self.default_root_vel], dim=-1)

    # -- Root state properties (link frame) --

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose in world frame. Shape: (N, 7)."""
        if self._root_link_pose_w is None:
            pose = torch.zeros(self._num_instances, 7, device=self._device)
            pose[:, 3:7] = self._identity_quat(self._num_instances)
            return pose
        return self._root_link_pose_w

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity in world frame. Shape: (N, 6)."""
        if self._root_link_vel_w is None:
            return torch.zeros(self._num_instances, 6, device=self._device)
        return self._root_link_vel_w

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root link state in world frame. Shape: (N, 13)."""
        return torch.cat([self.root_link_pose_w, self.root_link_vel_w], dim=-1)

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position. Shape: (N, 3)."""
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation. Shape: (N, 4)."""
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root link linear velocity. Shape: (N, 3)."""
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity. Shape: (N, 3)."""
        return self.root_link_vel_w[:, 3:6]

    # -- Root state properties (CoM frame) --

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root CoM pose in world frame. Shape: (N, 7)."""
        if self._root_com_pose_w is None:
            return self.root_link_pose_w.clone()
        return self._root_com_pose_w

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root CoM velocity in world frame. Shape: (N, 6)."""
        if self._root_com_vel_w is None:
            return self.root_link_vel_w.clone()
        return self._root_com_vel_w

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root CoM state in world frame. Shape: (N, 13)."""
        return torch.cat([self.root_com_pose_w, self.root_com_vel_w], dim=-1)

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state (link pose + CoM velocity). Shape: (N, 13)."""
        return torch.cat([self.root_link_pose_w, self.root_com_vel_w], dim=-1)

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root CoM position. Shape: (N, 3)."""
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root CoM orientation. Shape: (N, 4)."""
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root CoM linear velocity. Shape: (N, 3)."""
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root CoM angular velocity. Shape: (N, 3)."""
        return self.root_com_vel_w[:, 3:6]

    # -- Body state properties (link frame) --

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose in world frame. Shape: (N, 1, 7)."""
        if self._body_link_pose_w is None:
            pose = torch.zeros(self._num_instances, 1, 7, device=self._device)
            pose[..., 3:7] = self._identity_quat(self._num_instances, 1)
            return pose
        return self._body_link_pose_w

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity in world frame. Shape: (N, 1, 6)."""
        if self._body_link_vel_w is None:
            return torch.zeros(self._num_instances, 1, 6, device=self._device)
        return self._body_link_vel_w

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """Body link state in world frame. Shape: (N, 1, 13)."""
        return torch.cat([self.body_link_pose_w, self.body_link_vel_w], dim=-1)

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Body link position. Shape: (N, 1, 3)."""
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Body link orientation. Shape: (N, 1, 4)."""
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Body link linear velocity. Shape: (N, 1, 3)."""
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Body link angular velocity. Shape: (N, 1, 3)."""
        return self.body_link_vel_w[..., 3:6]

    # -- Body state properties (CoM frame) --

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body CoM pose in world frame. Shape: (N, 1, 7)."""
        if self._body_com_pose_w is None:
            return self.body_link_pose_w.clone()
        return self._body_com_pose_w

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body CoM velocity in world frame. Shape: (N, 1, 6)."""
        if self._body_com_vel_w is None:
            return self.body_link_vel_w.clone()
        return self._body_com_vel_w

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """Body CoM state in world frame. Shape: (N, 1, 13)."""
        return torch.cat([self.body_com_pose_w, self.body_com_vel_w], dim=-1)

    @property
    def body_com_acc_w(self) -> torch.Tensor:
        """Body CoM acceleration in world frame. Shape: (N, 1, 6)."""
        if self._body_com_acc_w is None:
            return torch.zeros(self._num_instances, 1, 6, device=self._device)
        return self._body_com_acc_w

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Body CoM pose in body frame. Shape: (N, 1, 7)."""
        if self._body_com_pose_b is None:
            pose = torch.zeros(self._num_instances, 1, 7, device=self._device)
            pose[..., 3:7] = self._identity_quat(self._num_instances, 1)
            return pose
        return self._body_com_pose_b

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Body CoM position. Shape: (N, 1, 3)."""
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Body CoM orientation. Shape: (N, 1, 4)."""
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Body CoM linear velocity. Shape: (N, 1, 3)."""
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Body CoM angular velocity. Shape: (N, 1, 3)."""
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Body CoM linear acceleration. Shape: (N, 1, 3)."""
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Body CoM angular acceleration. Shape: (N, 1, 3)."""
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Body CoM position in body frame. Shape: (N, 1, 3)."""
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Body CoM orientation in body frame. Shape: (N, 1, 4)."""
        return self.body_com_pose_b[..., 3:7]

    # -- Body properties --

    @property
    def body_mass(self) -> torch.Tensor:
        """Body mass. Shape: (N, 1, 1)."""
        if self._body_mass is None:
            return torch.ones(self._num_instances, 1, 1, device=self._device)
        return self._body_mass

    @property
    def body_inertia(self) -> torch.Tensor:
        """Body inertia (flattened 3x3). Shape: (N, 1, 9)."""
        if self._body_inertia is None:
            inertia = torch.zeros(self._num_instances, 1, 9, device=self._device)
            inertia[..., 0] = 1.0  # Ixx
            inertia[..., 4] = 1.0  # Iyy
            inertia[..., 8] = 1.0  # Izz
            return inertia
        return self._body_inertia

    # -- Derived properties --

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity projection on body. Shape: (N, 3)."""
        gravity = torch.zeros(self._num_instances, 3, device=self._device)
        gravity[:, 2] = -1.0
        return gravity

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading in world frame. Shape: (N,)."""
        return torch.zeros(self._num_instances, device=self._device)

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in body frame. Shape: (N, 3)."""
        return self.root_link_lin_vel_w.clone()

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in body frame. Shape: (N, 3)."""
        return self.root_link_ang_vel_w.clone()

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root CoM linear velocity in body frame. Shape: (N, 3)."""
        return self.root_com_lin_vel_w.clone()

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root CoM angular velocity in body frame. Shape: (N, 3)."""
        return self.root_com_ang_vel_w.clone()

    # -- Convenience aliases for root state (without _link_ or _com_ prefix) --

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position (alias for root_link_pos_w). Shape: (N, 3)."""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (alias for root_link_quat_w). Shape: (N, 4)."""
        return self.root_link_quat_w

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Root pose (alias for root_link_pose_w). Shape: (N, 7)."""
        return self.root_link_pose_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity (alias for root_com_vel_w). Shape: (N, 6)."""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity (alias for root_com_lin_vel_w). Shape: (N, 3)."""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity (alias for root_com_ang_vel_w). Shape: (N, 3)."""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in body frame (alias for root_com_lin_vel_b). Shape: (N, 3)."""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in body frame (alias for root_com_ang_vel_b). Shape: (N, 3)."""
        return self.root_com_ang_vel_b

    # -- Convenience aliases for body state (without _link_ or _com_ prefix) --

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Body positions (alias for body_link_pos_w). Shape: (N, 1, 3)."""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Body orientations (alias for body_link_quat_w). Shape: (N, 1, 4)."""
        return self.body_link_quat_w

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Body poses (alias for body_link_pose_w). Shape: (N, 1, 7)."""
        return self.body_link_pose_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Body velocities (alias for body_com_vel_w). Shape: (N, 1, 6)."""
        return self.body_com_vel_w

    @property
    def body_state_w(self) -> torch.Tensor:
        """Body states (alias for body_com_state_w). Shape: (N, 1, 13)."""
        return self.body_com_state_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Body linear velocities (alias for body_com_lin_vel_w). Shape: (N, 1, 3)."""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Body angular velocities (alias for body_com_ang_vel_w). Shape: (N, 1, 3)."""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Body accelerations (alias for body_com_acc_w). Shape: (N, 1, 6)."""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Body linear accelerations (alias for body_com_lin_acc_w). Shape: (N, 1, 3)."""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Body angular accelerations (alias for body_com_ang_acc_w). Shape: (N, 1, 3)."""
        return self.body_com_ang_acc_w

    # -- CoM in body frame --

    @property
    def com_pos_b(self) -> torch.Tensor:
        """CoM position in body frame. Shape: (N, 3)."""
        return self.body_com_pose_b[:, 0, :3]

    @property
    def com_quat_b(self) -> torch.Tensor:
        """CoM orientation in body frame. Shape: (N, 4)."""
        return self.body_com_pose_b[:, 0, 3:7]

    # -- Update method --

    def update(self, dt: float) -> None:
        """Update data (no-op for mock)."""
        pass

    # -- Setters --

    def set_default_root_pose(self, value: torch.Tensor) -> None:
        self._default_root_pose = value.to(self._device)

    def set_default_root_vel(self, value: torch.Tensor) -> None:
        self._default_root_vel = value.to(self._device)

    def set_root_link_pose_w(self, value: torch.Tensor) -> None:
        self._root_link_pose_w = value.to(self._device)

    def set_root_link_vel_w(self, value: torch.Tensor) -> None:
        self._root_link_vel_w = value.to(self._device)

    def set_root_com_pose_w(self, value: torch.Tensor) -> None:
        self._root_com_pose_w = value.to(self._device)

    def set_root_com_vel_w(self, value: torch.Tensor) -> None:
        self._root_com_vel_w = value.to(self._device)

    def set_body_link_pose_w(self, value: torch.Tensor) -> None:
        self._body_link_pose_w = value.to(self._device)

    def set_body_link_vel_w(self, value: torch.Tensor) -> None:
        self._body_link_vel_w = value.to(self._device)

    def set_body_com_pose_w(self, value: torch.Tensor) -> None:
        self._body_com_pose_w = value.to(self._device)

    def set_body_com_vel_w(self, value: torch.Tensor) -> None:
        self._body_com_vel_w = value.to(self._device)

    def set_body_com_acc_w(self, value: torch.Tensor) -> None:
        self._body_com_acc_w = value.to(self._device)

    def set_body_com_pose_b(self, value: torch.Tensor) -> None:
        self._body_com_pose_b = value.to(self._device)

    def set_body_mass(self, value: torch.Tensor) -> None:
        self._body_mass = value.to(self._device)

    def set_body_inertia(self, value: torch.Tensor) -> None:
        self._body_inertia = value.to(self._device)

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
