# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared mock interfaces for testing and benchmarking Newton-based asset classes.

This module provides mock implementations of Newton simulation components that can
be used to test ArticulationData, RigidObjectData, and related classes without
requiring an actual simulation environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import warp as wp


class MockNewtonModel:
    """Mock Newton model that provides gravity."""

    def __init__(self, gravity: tuple[float, float, float] = (0.0, 0.0, -9.81), device: str = "cpu"):
        self._gravity = wp.array([gravity], dtype=wp.vec3f, device=device)

    @property
    def gravity(self):
        return self._gravity


class MockWrenchComposer:
    """Mock WrenchComposer for testing without full simulation infrastructure.

    This class mimics the interface of :class:`isaaclab.utils.wrench_composer.WrenchComposer`
    and can be used to test Articulation and RigidObject classes.
    """

    def __init__(self, asset):
        """Initialize the mock wrench composer.

        Args:
            asset: The asset (Articulation or RigidObject) to compose wrenches for.
        """
        self.num_envs = asset.num_instances
        self.num_bodies = asset.num_bodies
        self.device = asset.device
        self._active = False

        # Create buffers matching the real WrenchComposer
        self._composed_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._composed_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self._ALL_ENV_MASK_WP = wp.ones((self.num_envs,), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK_WP = wp.ones((self.num_bodies,), dtype=wp.bool, device=self.device)

    @property
    def active(self) -> bool:
        """Whether the wrench composer is active."""
        return self._active

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's com frame."""
        return self._composed_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's com frame."""
        return self._composed_torque_b

    def set_forces_and_torques(
        self,
        forces=None,
        torques=None,
        positions=None,
        body_ids=None,
        env_ids=None,
        body_mask=None,
        env_mask=None,
        is_global: bool = False,
    ):
        """Mock set_forces_and_torques - just marks as active."""
        self._active = True

    def add_forces_and_torques(
        self,
        forces=None,
        torques=None,
        positions=None,
        body_ids=None,
        env_ids=None,
        body_mask=None,
        env_mask=None,
        is_global: bool = False,
    ):
        """Mock add_forces_and_torques - just marks as active."""
        self._active = True

    def reset(self, env_ids=None, env_mask=None):
        """Reset the composed force and torque."""
        self._composed_force_b.zero_()
        self._composed_torque_b.zero_()
        self._active = False


def create_mock_newton_manager(
    patch_path: str,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
):
    """Create a mock NewtonManager for testing.

    Args:
        patch_path: The module path to patch
            (e.g., "isaaclab_newton.assets.articulation.articulation_data.NewtonManager").
        gravity: Gravity vector to use for the mock model.

    Returns:
        A context manager that patches the NewtonManager.
    """
    mock_model = MockNewtonModel(gravity)
    mock_state = MagicMock()
    mock_control = MagicMock()

    return patch(
        patch_path,
        **{
            "get_model.return_value": mock_model,
            "get_state_0.return_value": mock_state,
            "get_control.return_value": mock_control,
            "get_dt.return_value": 0.01,
        },
    )


class MockNewtonContactSensor:
    """Mock Newton contact sensor for testing without full simulation infrastructure.

    This class mimics the interface of Newton's SensorContact and can be used to test
    ContactSensor classes without requiring an actual simulation environment.
    """

    def __init__(
        self,
        num_sensing_objs: int,
        num_counterparts: int = 1,
        device: str = "cuda:0",
    ):
        """Initialize the mock contact sensor.

        Args:
            num_sensing_objs: Number of sensing objects (e.g., bodies or shapes).
            num_counterparts: Number of counterparts per sensing object.
            device: Device to use.
        """
        self.device = device
        self.shape: tuple[int, int] = (num_sensing_objs, num_counterparts)
        self.sensing_objs: list[tuple[int, int]] = [(i, 1) for i in range(num_sensing_objs)]
        self.counterparts: list[tuple[int, int]] = [(i, 1) for i in range(num_counterparts)]
        self.reading_indices: list[list[int]] = [list(range(num_counterparts)) for _ in range(num_sensing_objs)]

        # Net force array (n_sensing_objs, n_counterparts) of vec3
        self._net_force = wp.zeros(num_sensing_objs * num_counterparts, dtype=wp.vec3, device=device)
        self.net_force = self._net_force.reshape(self.shape)

    def eval(self, contacts):
        """Mock eval - does nothing since forces are set directly via set_mock_data."""
        pass

    def get_total_force(self) -> wp.array:
        """Get the total net force measured by the contact sensor."""
        return self.net_force

    def set_mock_data(self, net_force: wp.array | None = None):
        """Set mock contact force data.

        Args:
            net_force: Force data shaped (num_sensing_objs, num_counterparts) of vec3.
                       If None, zeros the force data.
        """
        if net_force is None:
            self._net_force.zero_()
        else:
            self._net_force.assign(net_force)
