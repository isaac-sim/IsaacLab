# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton fixture components used by isolated tests and benchmarks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import warp as wp


class MockNewtonModel:
    """Mock Newton model that provides gravity and articulation topology."""

    def __init__(
        self,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        device: str = "cpu",
        num_instances: int = 1,
        num_bodies: int = 1,
        num_joints: int = 0,
        is_fixed_base: bool = False,
    ):
        self.world_count = num_instances
        self._gravity = wp.array([gravity] * (num_instances + 1), dtype=wp.vec3f, device=device)
        num_dofs = num_joints + (0 if is_fixed_base else 6)
        self.articulation_count = num_instances
        self.max_joints_per_articulation = num_bodies
        self.max_dofs_per_articulation = num_dofs
        self.joint_dof_count = num_instances * num_dofs
        self.body_count = num_instances * num_bodies

    @property
    def gravity(self):
        return self._gravity


def create_mock_newton_manager(
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    device: str = "cpu",
    num_instances: int = 1,
    num_bodies: int = 1,
    num_joints: int = 0,
    is_fixed_base: bool = False,
):
    """Create a mock NewtonManager for testing.

    Args:
        gravity: Gravity vector to use for the mock model.
        device: Device for mock buffers.
        num_instances: Number of articulation instances in the mock model.
        num_bodies: Number of bodies in each mock articulation.
        num_joints: Number of joint degrees of freedom in each mock articulation.
        is_fixed_base: Whether the mock articulation has a fixed base.

    Returns:
        An independent mock manager with explicit model and state ownership.
    """
    mock_model = MockNewtonModel(
        gravity,
        device=device,
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        is_fixed_base=is_fixed_base,
    )
    mock_state = MagicMock()
    mock_control = MagicMock()

    manager = MagicMock()
    manager.get_model.return_value = mock_model
    manager.get_state_0.return_value = mock_state
    manager.get_control.return_value = mock_control
    manager.backend = SimpleNamespace(
        is_stepping=False, transforms_may_change_on_graph_replay=False, device=wp.get_device(device)
    )
    return manager
