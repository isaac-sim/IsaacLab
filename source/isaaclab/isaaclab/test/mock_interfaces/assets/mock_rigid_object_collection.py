# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock rigid object collection asset for testing without Isaac Sim."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


class MockRigidObjectCollectionData:
    """Mock data container for rigid object collection asset.

    This class mimics the interface of BaseRigidObjectCollectionData for testing purposes.
    All tensor properties return zero tensors with correct shapes if not explicitly set.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        body_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock rigid object collection data.

        Args:
            num_instances: Number of environment instances.
            num_bodies: Number of bodies in the collection.
            body_names: Names of bodies.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_bodies = num_bodies
        self._device = device

        self.body_names = body_names or [f"body_{i}" for i in range(num_bodies)]

        # Default states
        self._default_body_pose: torch.Tensor | None = None
        self._default_body_vel: torch.Tensor | None = None

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
    def default_body_pose(self) -> torch.Tensor:
        """Default body poses. Shape: (N, num_bodies, 7)."""
        if self._default_body_pose is None:
            pose = torch.zeros(self._num_instances, self._num_bodies, 7, device=self._device)
            pose[..., 3:7] = self._identity_quat(self._num_instances, self._num_bodies)
            return pose
        return self._default_body_pose

    @property
    def default_body_vel(self) -> torch.Tensor:
        """Default body velocities. Shape: (N, num_bodies, 6)."""
        if self._default_body_vel is None:
            return torch.zeros(self._num_instances, self._num_bodies, 6, device=self._device)
        return self._default_body_vel

    @property
    def default_body_state(self) -> torch.Tensor:
        """Default body states. Shape: (N, num_bodies, 13)."""
        return torch.cat([self.default_body_pose, self.default_body_vel], dim=-1)

    # -- Body state properties (link frame) --

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link poses in world frame. Shape: (N, num_bodies, 7)."""
        if self._body_link_pose_w is None:
            pose = torch.zeros(self._num_instances, self._num_bodies, 7, device=self._device)
            pose[..., 3:7] = self._identity_quat(self._num_instances, self._num_bodies)
            return pose
        return self._body_link_pose_w

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocities in world frame. Shape: (N, num_bodies, 6)."""
        if self._body_link_vel_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, 6, device=self._device)
        return self._body_link_vel_w

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """Body link states in world frame. Shape: (N, num_bodies, 13)."""
        return torch.cat([self.body_link_pose_w, self.body_link_vel_w], dim=-1)

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Body link positions. Shape: (N, num_bodies, 3)."""
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Body link orientations. Shape: (N, num_bodies, 4)."""
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Body link linear velocities. Shape: (N, num_bodies, 3)."""
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Body link angular velocities. Shape: (N, num_bodies, 3)."""
        return self.body_link_vel_w[..., 3:6]

    # -- Body state properties (CoM frame) --

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body CoM poses in world frame. Shape: (N, num_bodies, 7)."""
        if self._body_com_pose_w is None:
            return self.body_link_pose_w.clone()
        return self._body_com_pose_w

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body CoM velocities in world frame. Shape: (N, num_bodies, 6)."""
        if self._body_com_vel_w is None:
            return self.body_link_vel_w.clone()
        return self._body_com_vel_w

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """Body CoM states in world frame. Shape: (N, num_bodies, 13)."""
        return torch.cat([self.body_com_pose_w, self.body_com_vel_w], dim=-1)

    @property
    def body_com_acc_w(self) -> torch.Tensor:
        """Body CoM accelerations in world frame. Shape: (N, num_bodies, 6)."""
        if self._body_com_acc_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, 6, device=self._device)
        return self._body_com_acc_w

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Body CoM poses in body frame. Shape: (N, num_bodies, 7)."""
        if self._body_com_pose_b is None:
            pose = torch.zeros(self._num_instances, self._num_bodies, 7, device=self._device)
            pose[..., 3:7] = self._identity_quat(self._num_instances, self._num_bodies)
            return pose
        return self._body_com_pose_b

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Body CoM positions. Shape: (N, num_bodies, 3)."""
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Body CoM orientations. Shape: (N, num_bodies, 4)."""
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Body CoM linear velocities. Shape: (N, num_bodies, 3)."""
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Body CoM angular velocities. Shape: (N, num_bodies, 3)."""
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Body CoM linear accelerations. Shape: (N, num_bodies, 3)."""
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Body CoM angular accelerations. Shape: (N, num_bodies, 3)."""
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Body CoM positions in body frame. Shape: (N, num_bodies, 3)."""
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Body CoM orientations in body frame. Shape: (N, num_bodies, 4)."""
        return self.body_com_pose_b[..., 3:7]

    # -- Body properties --

    @property
    def body_mass(self) -> torch.Tensor:
        """Body masses. Shape: (N, num_bodies)."""
        if self._body_mass is None:
            return torch.ones(self._num_instances, self._num_bodies, device=self._device)
        return self._body_mass

    @property
    def body_inertia(self) -> torch.Tensor:
        """Body inertias (flattened 3x3). Shape: (N, num_bodies, 9)."""
        if self._body_inertia is None:
            inertia = torch.zeros(self._num_instances, self._num_bodies, 9, device=self._device)
            inertia[..., 0] = 1.0  # Ixx
            inertia[..., 4] = 1.0  # Iyy
            inertia[..., 8] = 1.0  # Izz
            return inertia
        return self._body_inertia

    # -- Derived properties --

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity projection on bodies. Shape: (N, num_bodies, 3)."""
        gravity = torch.zeros(self._num_instances, self._num_bodies, 3, device=self._device)
        gravity[..., 2] = -1.0
        return gravity

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading per body. Shape: (N, num_bodies)."""
        return torch.zeros(self._num_instances, self._num_bodies, device=self._device)

    @property
    def body_link_lin_vel_b(self) -> torch.Tensor:
        """Body link linear velocities in body frame. Shape: (N, num_bodies, 3)."""
        return self.body_link_lin_vel_w.clone()

    @property
    def body_link_ang_vel_b(self) -> torch.Tensor:
        """Body link angular velocities in body frame. Shape: (N, num_bodies, 3)."""
        return self.body_link_ang_vel_w.clone()

    @property
    def body_com_lin_vel_b(self) -> torch.Tensor:
        """Body CoM linear velocities in body frame. Shape: (N, num_bodies, 3)."""
        return self.body_com_lin_vel_w.clone()

    @property
    def body_com_ang_vel_b(self) -> torch.Tensor:
        """Body CoM angular velocities in body frame. Shape: (N, num_bodies, 3)."""
        return self.body_com_ang_vel_w.clone()

    # -- Convenience alias --

    @property
    def body_state_w(self) -> torch.Tensor:
        """Body states (alias for body_com_state_w). Shape: (N, num_bodies, 13)."""
        return self.body_com_state_w

    # -- Update method --

    def update(self, dt: float) -> None:
        """Update data (no-op for mock)."""
        pass

    # -- Setters --

    def set_default_body_pose(self, value: torch.Tensor) -> None:
        self._default_body_pose = value.to(self._device)

    def set_default_body_vel(self, value: torch.Tensor) -> None:
        self._default_body_vel = value.to(self._device)

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


class MockRigidObjectCollection:
    """Mock rigid object collection asset for testing without Isaac Sim.

    This class mimics the interface of BaseRigidObjectCollection for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        body_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock rigid object collection.

        Args:
            num_instances: Number of environment instances.
            num_bodies: Number of bodies in the collection.
            body_names: Names of bodies.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_bodies = num_bodies
        self._device = device
        self._body_names = body_names or [f"body_{i}" for i in range(num_bodies)]

        self._data = MockRigidObjectCollectionData(
            num_instances=num_instances,
            num_bodies=num_bodies,
            body_names=self._body_names,
            device=device,
        )

    # -- Properties --

    @property
    def data(self) -> MockRigidObjectCollectionData:
        """Data container for the rigid object collection."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of environment instances."""
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the collection."""
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

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ) -> None:
        """Reset rigid object collection state for specified environments."""
        pass

    def write_data_to_sim(self) -> None:
        """Write data to simulation (no-op for mock)."""
        pass

    def update(self, dt: float) -> None:
        """Update rigid object collection data."""
        self._data.update(dt)

    # -- Finder methods --

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Find bodies by name regex patterns.

        Returns:
            Tuple of (body_mask, body_names, body_indices).
        """
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

        # Create body mask
        body_mask = torch.zeros(self._num_bodies, dtype=torch.bool, device=self._device)
        body_mask[matched_indices] = True

        return body_mask, matched_names, matched_indices

    # -- State writer methods (no-op for mock) --

    def write_body_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body states to simulation."""
        pass

    def write_body_com_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body CoM states to simulation."""
        pass

    def write_body_link_state_to_sim(
        self,
        body_states: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body link states to simulation."""
        pass

    def write_body_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body poses to simulation."""
        pass

    def write_body_link_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body link poses to simulation."""
        pass

    def write_body_com_pose_to_sim(
        self,
        body_poses: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body CoM poses to simulation."""
        pass

    def write_body_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body velocities to simulation."""
        pass

    def write_body_com_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body CoM velocities to simulation."""
        pass

    def write_body_link_velocity_to_sim(
        self,
        body_velocities: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Write body link velocities to simulation."""
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
