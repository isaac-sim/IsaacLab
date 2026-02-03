# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock contact sensor for testing without Isaac Sim."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


class MockContactSensorData:
    """Mock data container for contact sensor.

    This class mimics the interface of BaseContactSensorData for testing purposes.
    All tensor properties return zero tensors with correct shapes if not explicitly set.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        device: str = "cpu",
        history_length: int = 0,
        num_filter_bodies: int = 0,
    ):
        """Initialize mock contact sensor data.

        Args:
            num_instances: Number of environment instances.
            num_bodies: Number of bodies with contact sensors.
            device: Device for tensor allocation.
            history_length: Length of history buffer for forces.
            num_filter_bodies: Number of filter bodies for force matrix.
        """
        self._num_instances = num_instances
        self._num_bodies = num_bodies
        self._device = device
        self._history_length = history_length
        self._num_filter_bodies = num_filter_bodies

        # Internal storage for mock data
        self._pos_w: torch.Tensor | None = None
        self._quat_w: torch.Tensor | None = None
        self._net_forces_w: torch.Tensor | None = None
        self._net_forces_w_history: torch.Tensor | None = None
        self._force_matrix_w: torch.Tensor | None = None
        self._force_matrix_w_history: torch.Tensor | None = None
        self._contact_pos_w: torch.Tensor | None = None
        self._friction_forces_w: torch.Tensor | None = None
        self._last_air_time: torch.Tensor | None = None
        self._current_air_time: torch.Tensor | None = None
        self._last_contact_time: torch.Tensor | None = None
        self._current_contact_time: torch.Tensor | None = None

    # -- Properties --

    @property
    def pos_w(self) -> torch.Tensor | None:
        """Position of sensor origins in world frame. Shape: (N, B, 3)."""
        if self._pos_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, 3, device=self._device)
        return self._pos_w

    @property
    def quat_w(self) -> torch.Tensor | None:
        """Orientation (w, x, y, z) in world frame. Shape: (N, B, 4)."""
        if self._quat_w is None:
            # Default to identity quaternion
            quat = torch.zeros(self._num_instances, self._num_bodies, 4, device=self._device)
            quat[..., 0] = 1.0
            return quat
        return self._quat_w

    @property
    def pose_w(self) -> torch.Tensor | None:
        """Pose in world frame (pos + quat). Shape: (N, B, 7)."""
        return torch.cat([self.pos_w, self.quat_w], dim=-1)

    @property
    def net_forces_w(self) -> torch.Tensor:
        """Net normal contact forces in world frame. Shape: (N, B, 3)."""
        if self._net_forces_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, 3, device=self._device)
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> torch.Tensor | None:
        """History of net forces. Shape: (N, T, B, 3)."""
        if self._history_length == 0:
            return None
        if self._net_forces_w_history is None:
            return torch.zeros(self._num_instances, self._history_length, self._num_bodies, 3, device=self._device)
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> torch.Tensor | None:
        """Filtered contact forces. Shape: (N, B, M, 3)."""
        if self._num_filter_bodies == 0:
            return None
        if self._force_matrix_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, self._num_filter_bodies, 3, device=self._device)
        return self._force_matrix_w

    @property
    def force_matrix_w_history(self) -> torch.Tensor | None:
        """History of filtered forces. Shape: (N, T, B, M, 3)."""
        if self._history_length == 0 or self._num_filter_bodies == 0:
            return None
        if self._force_matrix_w_history is None:
            return torch.zeros(
                self._num_instances,
                self._history_length,
                self._num_bodies,
                self._num_filter_bodies,
                3,
                device=self._device,
            )
        return self._force_matrix_w_history

    @property
    def contact_pos_w(self) -> torch.Tensor | None:
        """Contact point positions in world frame. Shape: (N, B, M, 3)."""
        if self._num_filter_bodies == 0:
            return None
        if self._contact_pos_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, self._num_filter_bodies, 3, device=self._device)
        return self._contact_pos_w

    @property
    def friction_forces_w(self) -> torch.Tensor | None:
        """Friction forces in world frame. Shape: (N, B, M, 3)."""
        if self._num_filter_bodies == 0:
            return None
        if self._friction_forces_w is None:
            return torch.zeros(self._num_instances, self._num_bodies, self._num_filter_bodies, 3, device=self._device)
        return self._friction_forces_w

    @property
    def last_air_time(self) -> torch.Tensor:
        """Time in air before last contact. Shape: (N, B)."""
        if self._last_air_time is None:
            return torch.zeros(self._num_instances, self._num_bodies, device=self._device)
        return self._last_air_time

    @property
    def current_air_time(self) -> torch.Tensor:
        """Current time in air. Shape: (N, B)."""
        if self._current_air_time is None:
            return torch.zeros(self._num_instances, self._num_bodies, device=self._device)
        return self._current_air_time

    @property
    def last_contact_time(self) -> torch.Tensor:
        """Time in contact before last detach. Shape: (N, B)."""
        if self._last_contact_time is None:
            return torch.zeros(self._num_instances, self._num_bodies, device=self._device)
        return self._last_contact_time

    @property
    def current_contact_time(self) -> torch.Tensor:
        """Current time in contact. Shape: (N, B)."""
        if self._current_contact_time is None:
            return torch.zeros(self._num_instances, self._num_bodies, device=self._device)
        return self._current_contact_time

    # -- Setters --

    def set_pos_w(self, value: torch.Tensor) -> None:
        """Set position in world frame."""
        self._pos_w = value.to(self._device)

    def set_quat_w(self, value: torch.Tensor) -> None:
        """Set orientation in world frame."""
        self._quat_w = value.to(self._device)

    def set_net_forces_w(self, value: torch.Tensor) -> None:
        """Set net contact forces."""
        self._net_forces_w = value.to(self._device)

    def set_net_forces_w_history(self, value: torch.Tensor) -> None:
        """Set net forces history."""
        self._net_forces_w_history = value.to(self._device)

    def set_force_matrix_w(self, value: torch.Tensor) -> None:
        """Set filtered contact forces."""
        self._force_matrix_w = value.to(self._device)

    def set_force_matrix_w_history(self, value: torch.Tensor) -> None:
        """Set filtered forces history."""
        self._force_matrix_w_history = value.to(self._device)

    def set_contact_pos_w(self, value: torch.Tensor) -> None:
        """Set contact point positions."""
        self._contact_pos_w = value.to(self._device)

    def set_friction_forces_w(self, value: torch.Tensor) -> None:
        """Set friction forces."""
        self._friction_forces_w = value.to(self._device)

    def set_last_air_time(self, value: torch.Tensor) -> None:
        """Set last air time."""
        self._last_air_time = value.to(self._device)

    def set_current_air_time(self, value: torch.Tensor) -> None:
        """Set current air time."""
        self._current_air_time = value.to(self._device)

    def set_last_contact_time(self, value: torch.Tensor) -> None:
        """Set last contact time."""
        self._last_contact_time = value.to(self._device)

    def set_current_contact_time(self, value: torch.Tensor) -> None:
        """Set current contact time."""
        self._current_contact_time = value.to(self._device)

    def set_mock_data(
        self,
        pos_w: torch.Tensor | None = None,
        quat_w: torch.Tensor | None = None,
        net_forces_w: torch.Tensor | None = None,
        net_forces_w_history: torch.Tensor | None = None,
        force_matrix_w: torch.Tensor | None = None,
        force_matrix_w_history: torch.Tensor | None = None,
        contact_pos_w: torch.Tensor | None = None,
        friction_forces_w: torch.Tensor | None = None,
        last_air_time: torch.Tensor | None = None,
        current_air_time: torch.Tensor | None = None,
        last_contact_time: torch.Tensor | None = None,
        current_contact_time: torch.Tensor | None = None,
    ) -> None:
        """Bulk setter for mock data.

        Args:
            pos_w: Position in world frame. Shape: (N, B, 3).
            quat_w: Orientation in world frame. Shape: (N, B, 4).
            net_forces_w: Net contact forces. Shape: (N, B, 3).
            net_forces_w_history: History of net forces. Shape: (N, T, B, 3).
            force_matrix_w: Filtered contact forces. Shape: (N, B, M, 3).
            force_matrix_w_history: History of filtered forces. Shape: (N, T, B, M, 3).
            contact_pos_w: Contact point positions. Shape: (N, B, M, 3).
            friction_forces_w: Friction forces. Shape: (N, B, M, 3).
            last_air_time: Time in air before last contact. Shape: (N, B).
            current_air_time: Current time in air. Shape: (N, B).
            last_contact_time: Time in contact before last detach. Shape: (N, B).
            current_contact_time: Current time in contact. Shape: (N, B).
        """
        if pos_w is not None:
            self.set_pos_w(pos_w)
        if quat_w is not None:
            self.set_quat_w(quat_w)
        if net_forces_w is not None:
            self.set_net_forces_w(net_forces_w)
        if net_forces_w_history is not None:
            self.set_net_forces_w_history(net_forces_w_history)
        if force_matrix_w is not None:
            self.set_force_matrix_w(force_matrix_w)
        if force_matrix_w_history is not None:
            self.set_force_matrix_w_history(force_matrix_w_history)
        if contact_pos_w is not None:
            self.set_contact_pos_w(contact_pos_w)
        if friction_forces_w is not None:
            self.set_friction_forces_w(friction_forces_w)
        if last_air_time is not None:
            self.set_last_air_time(last_air_time)
        if current_air_time is not None:
            self.set_current_air_time(current_air_time)
        if last_contact_time is not None:
            self.set_last_contact_time(last_contact_time)
        if current_contact_time is not None:
            self.set_current_contact_time(current_contact_time)


class MockContactSensor:
    """Mock contact sensor for testing without Isaac Sim.

    This class mimics the interface of BaseContactSensor for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        body_names: list[str] | None = None,
        device: str = "cpu",
        history_length: int = 0,
        num_filter_bodies: int = 0,
    ):
        """Initialize mock contact sensor.

        Args:
            num_instances: Number of environment instances.
            num_bodies: Number of bodies with contact sensors.
            body_names: Names of bodies with contact sensors.
            device: Device for tensor allocation.
            history_length: Length of history buffer for forces.
            num_filter_bodies: Number of filter bodies for force matrix.
        """
        self._num_instances = num_instances
        self._num_bodies = num_bodies
        self._body_names = body_names or [f"body_{i}" for i in range(num_bodies)]
        self._device = device
        self._data = MockContactSensorData(num_instances, num_bodies, device, history_length, num_filter_bodies)

    # -- Properties --

    @property
    def data(self) -> MockContactSensorData:
        """Data container for the sensor."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of sensor instances."""
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies with contact sensors."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors."""
        return self._body_names

    @property
    def contact_view(self) -> None:
        """Returns None (no PhysX view in mock)."""
        return None

    @property
    def device(self) -> str:
        """Device for tensor allocation."""
        return self._device

    # -- Methods --

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies by name regex patterns.

        Args:
            name_keys: Regex pattern(s) to match body names.
            preserve_order: If True, preserve order of name_keys in output.

        Returns:
            Tuple of (body_indices, body_names) matching the patterns.
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

        return matched_indices, matched_names

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Check which bodies established contact within dt seconds.

        Args:
            dt: Time window to check for first contact.
            abs_tol: Absolute tolerance for contact time comparison.

        Returns:
            Boolean tensor of shape (N, B) indicating first contact.
        """
        return self._data.current_contact_time < (dt + abs_tol)

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Check which bodies broke contact within dt seconds.

        Args:
            dt: Time window to check for first air.
            abs_tol: Absolute tolerance for air time comparison.

        Returns:
            Boolean tensor of shape (N, B) indicating first air.
        """
        return self._data.current_air_time < (dt + abs_tol)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset sensor state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        # No-op for mock - data persists until explicitly changed
        pass

    def update(self, dt: float, force_recompute: bool = False) -> None:
        """Update sensor.

        Args:
            dt: Time step since last update.
            force_recompute: Force recomputation of buffers.
        """
        # No-op for mock - data is set explicitly
        pass
